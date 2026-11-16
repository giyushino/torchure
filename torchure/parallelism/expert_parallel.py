"""
expert parallel: split a moe layer's experts across the mesh's ep axis and
move the tokens to them, instead of moving the expert weights to the tokens.

each ep rank holds `num_experts / ep` experts of every bank (`Shard(0)` on
the expert dim) and its OWN batch: ep ranks are data-parallel for everything
that is not an expert. inside a moe layer the (token, expert) pairs a rank
routed are sent to the ranks holding those experts and the outputs come
back:

    counts per expert          --all_gather-->   who sends how many to whom
    rows, grouped by expert    --all_to_all-->   to the rank holding the expert
    local experts (one grouped matmul over everyone's rows)
    outputs                    --all_to_all-->   back, in the order they left

two all_to_alls per layer in forward and two in backward (the grads retrace
the same route). the split sizes have to be python ints, so each layer costs
one host sync.

this file knows nothing about any model: the model hands over its expert
banks (`model.ep_plan()`, models/moe.py's ExpertBank) and the bank calls
back through `bank.remote`.

grads. the objective is the mean of every data rank's loss. a dense param
is replicated on all of them, so its grad is the average over all of them
(ddp / fsdp do that, over an axis that includes ep). an expert lives on ONE
ep rank, which has already summed the grads of every ep peer's loss that
flowed back through the dispatch. so its grad is that sum / ep (`sync()`),
averaged only over the ranks holding a replica of the same expert: the dp
axes without ep. the trainer tells ddp / fsdp which params those are.

two ways to build, as for tensor parallel: from full weights (every rank
inits the full model on one seed, EP keeps its slice), or from a meta model
(EP only rewrites shapes; after materialize + init, `finish_init`).

tests: uv run tests/expert_parallel.py  (gloo/cpu, no gpus needed)
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from torchure.core.collective import MeshLike, all_gather, all_to_all, broadcast
from torchure.core.placement import Shard
from torchure.models.moe import ExpertBank


class _Exchange(torch.autograd.Function):
    """uneven all_to_all of rows in forward; the reverse trip for the grad."""

    @staticmethod
    def forward(ctx, x, mesh, axis, send, recv):
        ctx.mesh, ctx.axis, ctx.send, ctx.recv = mesh, axis, send, recv
        return all_to_all(x, mesh, axis, input_splits=send, output_splits=recv)

    @staticmethod
    def backward(ctx, grad):
        back = all_to_all(
            grad, ctx.mesh, ctx.axis, input_splits=ctx.recv, output_splits=ctx.send
        )
        return back, None, None, None, None


class EP:
    def __init__(self, mesh: MeshLike, banks: Iterable[ExpertBank], dim: str = "ep"):
        self.mesh = mesh
        self.dim = dim
        self.group_size = mesh.size(dim)
        # (module, attribute, unsharded shape) of every sharded weight. looked
        # up through the module: materializing a meta model replaces the
        # Parameter objects
        self._entries: list[tuple[nn.Module, str, tuple[int, ...]]] = []
        self._by_id: dict[int, tuple[int, ...]] | None = None
        if self.group_size == 1:
            return

        for bank in banks:
            assert bank.num_experts % self.group_size == 0, (
                f"{bank.num_experts} experts don't divide over {dim}={self.group_size}"
            )
            for linear in (bank.gate, bank.up, bank.down):
                self._shard(linear)
            bank.remote = self._remote

    # ---- construction ----------------------------------------------------

    def _shard(self, module: nn.Module) -> None:
        """make `module.weight` this rank's slice of the experts (dim 0)."""
        old = module._parameters["weight"]
        full_shape = tuple(old.shape)
        if old.is_meta:
            local_shape = Shard(0).local_shape(full_shape, self.group_size)
            module._parameters["weight"] = nn.Parameter(
                torch.empty(local_shape, dtype=old.dtype, device="meta"),
                requires_grad=old.requires_grad,
            )
        else:
            local = Shard(0).local_slice(
                old.data, self.group_size, self.mesh.coordinate(self.dim)
            )
            # an owning copy, so the full weight is freed
            old.data = local.contiguous().clone()
        self._entries.append((module, "weight", full_shape))
        self._by_id = None

    def _lookup(self) -> dict[int, tuple[int, ...]]:
        """id(current param) -> unsharded shape, rebuilt if params were replaced."""
        if self._by_id is not None and self._entries:
            module, name, _ = self._entries[0]
            if id(module._parameters[name]) not in self._by_id:
                self._by_id = None
        if self._by_id is None:
            self._by_id = {
                id(module._parameters[name]): full for module, name, full in self._entries
            }
        return self._by_id

    def finish_init(self, model: nn.Module) -> None:
        """
        call once after initializing a model built from meta. each rank
        initialized on its own seed so its experts are its own; this makes
        everything else, buffers included, agree with ep coordinate 0.
        """
        if self.group_size == 1:
            return
        sharded = self._lookup()
        for param in model.parameters():
            if id(param) not in sharded:
                broadcast(param.data, self.mesh, self.dim, src=0)
        for buf in model.buffers():
            broadcast(buf, self.mesh, self.dim, src=0)

    # ---- dispatch --------------------------------------------------------

    @torch.compiler.disable
    def _remote(self, bank: ExpertBank, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        `bank(x, counts)` across the ep group. x (M, in) is this rank's rows
        sorted by (global) expert id, counts (num_experts,) how many each
        expert gets. returns their outputs (M, out) in the same order.
        """
        size = self.group_size
        per_rank = bank.num_experts // size
        me = self.mesh.coordinate(self.dim)

        # table[sender, expert]: every rank's counts, so each knows what arrives
        table = all_gather(counts, self.mesh, self.dim).view(size, bank.num_experts)
        # rows sorted by expert are already grouped by destination rank
        send = counts.view(size, per_rank).sum(dim=1).tolist()
        mine = table[:, me * per_rank: (me + 1) * per_rank]   # (sender, local expert)
        recv = mine.sum(dim=1).tolist()

        arrived = _Exchange.apply(x, self.mesh, self.dim, send, recv)
        # arrivals are ordered by sender, then expert; the bank wants them
        # ordered by expert alone
        expert_of = torch.arange(per_rank, device=x.device).repeat(size)
        expert_of = expert_of.repeat_interleave(mine.reshape(-1))
        order = expert_of.argsort(stable=True)
        out = bank.local(arrived[order], mine.sum(dim=0))
        # undo the regrouping, then send every row back where it came from
        restored = torch.empty_like(out).index_copy(0, order, out)
        return _Exchange.apply(restored, self.mesh, self.dim, recv, send)

    # ---- trainer surface -------------------------------------------------

    @property
    def shard_dims(self) -> tuple[str, ...]:
        """mesh axes param/grad state is sharded over."""
        return (self.dim,) if self.group_size > 1 else ()

    def owns(self, param: nn.Parameter) -> bool:
        """is `param` an expert weight sharded over ep?"""
        return id(param) in self._lookup()

    def shard_dim_of(self, param: nn.Parameter) -> int | None:
        """the tensor dim `param` is split along over ep, or None if replicated."""
        return 0 if self.owns(param) else None

    def full_shape_of(self, param: nn.Parameter) -> tuple[int, ...]:
        """the shape `param` had before ep split it (its own shape if it didn't)."""
        return self._lookup().get(id(param), tuple(param.shape))

    def shard_dims_of(self, param: nn.Parameter) -> tuple[str, ...]:
        """the mesh axes this one param is sharded over: (ep,) or ()."""
        return (self.dim,) if self.owns(param) else ()

    def sync(self) -> None:
        """
        finish the expert grads: divide by ep (see the module docstring).
        call once per optimizer step, after the data-parallel sync() and
        before clipping.
        """
        if self.group_size == 1:
            return
        grads = [
            module._parameters[name].grad
            for module, name, _ in self._entries
            if module._parameters[name].grad is not None
        ]
        if grads:
            torch._foreach_mul_(grads, 1.0 / self.group_size)
