"""
tensor parallel: split the weights of a layer across the mesh's tp axis, so
each rank computes a slice of the layer's features.

two ways to split a linear (`weight[out, in]`):

    column   split the OUTPUT features: weight is Shard(0). takes a replicated
             input, each rank computes its slice of the output. no comm in
             forward; the grad wrt the input is a partial sum.
    row      split the INPUT features: weight is Shard(1). takes an input
             already sharded on the feature dim, each rank computes a partial
             sum of the full output.

they come in pairs, as a `region`: a module whose input and output are
replicated and whose insides are sharded -- column linears first, then ops
that never mix features across the split (elementwise, per-head attention),
then one row linear last. a region costs one collective in forward (its
output) and one in backward (the grad of its input).

this file knows nothing about any model. the model says which of its modules
are which (`Qwen3.tp_plan`); the split itself is forced by the dataflow.

    column      linears sharded on dim 0 (bias too, if any)
    row         linears sharded on dim 1 (no bias: every rank would add it)
    regions     modules that get the two collectives. the replicated input is
                the first positional arg
    shared      modules INSIDE a region whose params are replicated but see
                only this rank's features (a norm applied per head). each
                rank's grad is a partial sum, so sync() all_reduces them
    embeddings  nn.Embedding modules feeding the residual stream
    heads       linears mapping the residual stream to vocab logits
    vocab       the subset of embeddings + heads whose weight is sharded on
                dim 0, the vocab (a tied weight is listed under both)
    seq_local   modules with replicated params that sit BETWEEN regions (the
                block norms). only matters under sequence parallel

vocab parallel. a sharded embedding looks up only the ids in its own rows and
contributes zeros for the rest; the ranks' outputs sum to the full lookup. a
sharded head produces this rank's slice of the logits. called as a module it
gathers them into full logits; the memory-efficient path is the fused loss
(loss/fused_linear_ce.py), which takes the local weight and never builds them.

sequence parallel (`sequence_parallel=True`). between regions the activations
are replicated, which under plain tp means every rank holds all of them. sp
shards them along the sequence dim instead: a region gathers the sequence on
entry and reduce-scatters it on exit, the same bytes on the wire as the two
all_reduces it replaces. ops between regions (norms, residual adds) then run
on 1/tp of the tokens, so `seq_local` params get partial grads and join
`shared`.

    op                     forward              backward
    region entry           identity             all_reduce
      under sp             all_gather(seq)      reduce_scatter(seq)
    region exit            all_reduce           identity
      under sp             reduce_scatter(seq)  all_gather(seq)
    embedding out, sp      (as region exit when vocab-sharded, else:)
                           local slice(seq)     all_gather(seq)
    head input, sp         all_gather(seq)      local slice(seq)
    head input, vocab      identity             all_reduce
    head output, vocab     all_gather(vocab)    local slice(vocab)

everything else is replicated and needs nothing: tp peers get the same batch
and the same activations, so they compute the same grads. nothing re-syncs
those replicas, which relies on the kernels being deterministic across peers,
same as megatron.

two ways to build:

    from full weights (tests, small models): init the full model on the same
    seed on every rank, then TP(...) keeps this rank's slices.

    from meta: build the model under torch.device("meta"), TP(...) only
    rewrites shapes; then fsdp materializes it (or core.meta.materialize
    without fsdp), the model's init fills this rank's shards -- so it must be
    elementwise -- and finish_init() makes the replicated params agree.

usage:
    tp = TP(mesh, **model.tp_plan(mesh.size("tp")))
    ...
    loss.backward(); parallel.sync(); tp.sync(); optimizer.step()

tp.sync() must come after the data-parallel sync: that one is still reducing
the same grad tensors in the background until it returns. tp composes with
ddp or fsdp on the dp axes: fsdp shards whatever tp left on this rank.

compile: the collectives are opaque to torch.compile (`compiler.disable`), so
a compiled block splits into graphs around them and they run eagerly between.

tests: uv run tests/tensor_parallel.py  (gloo/cpu, no gpus needed)
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from torchure.core.collective import (
    MeshLike,
    all_gather,
    all_reduce,
    broadcast,
    reduce_scatter,
)
from torchure.core.placement import Shard


def _local(x: torch.Tensor, mesh: MeshLike, axis: str, dim: int) -> torch.Tensor:
    """this rank's chunk of `x` along tensor dim `dim`."""
    return Shard(dim % x.dim()).local_slice(x, mesh.size(axis), mesh.coordinate(axis))


class _ReduceGrad(torch.autograd.Function):
    """identity in forward; all_reduce(sum) of the grad in backward."""

    @staticmethod
    def forward(ctx, x, mesh, axis):
        ctx.mesh, ctx.axis = mesh, axis
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        # a copy: autograd may hand the same grad tensor to another consumer
        return all_reduce(grad.contiguous().clone(), ctx.mesh, ctx.axis, "sum"), None, None


class _ReduceOutput(torch.autograd.Function):
    """all_reduce(sum) in forward; identity in backward."""

    @staticmethod
    def forward(ctx, x, mesh, axis):
        # a copy: whatever produced x may need its own output in backward
        return all_reduce(x.contiguous().clone(), mesh, axis, "sum")

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


class _Gather(torch.autograd.Function):
    """
    all_gather along `dim` in forward. backward depends on what consumed the
    gathered tensor: `partial` consumers (column linears) each produce a
    partial grad, so reduce_scatter sums them and keeps this rank's chunk;
    otherwise every rank already holds the complete grad and keeps its chunk.
    """

    @staticmethod
    def forward(ctx, x, mesh, axis, dim, partial):
        ctx.mesh, ctx.axis, ctx.dim, ctx.partial = mesh, axis, dim, partial
        return all_gather(x, mesh, axis, gather_dim=dim)

    @staticmethod
    def backward(ctx, grad):
        if ctx.partial:
            out = reduce_scatter(grad, ctx.mesh, ctx.axis, "sum", scatter_dim=ctx.dim)
        else:
            out = _local(grad, ctx.mesh, ctx.axis, ctx.dim)
        return out, None, None, None, None


class _Scatter(torch.autograd.Function):
    """reduce_scatter(sum) along `dim` in forward; all_gather in backward."""

    @staticmethod
    def forward(ctx, x, mesh, axis, dim):
        ctx.mesh, ctx.axis, ctx.dim = mesh, axis, dim
        return reduce_scatter(x, mesh, axis, "sum", scatter_dim=dim)

    @staticmethod
    def backward(ctx, grad):
        return all_gather(grad, ctx.mesh, ctx.axis, gather_dim=ctx.dim), None, None, None


class _Slice(torch.autograd.Function):
    """this rank's chunk along `dim` in forward; all_gather in backward."""

    @staticmethod
    def forward(ctx, x, mesh, axis, dim):
        ctx.mesh, ctx.axis, ctx.dim = mesh, axis, dim
        return _local(x, mesh, axis, dim).contiguous()

    @staticmethod
    def backward(ctx, grad):
        return all_gather(grad, ctx.mesh, ctx.axis, gather_dim=ctx.dim), None, None, None


# every op below is opaque to torch.compile: it launches collectives


@torch.compiler.disable
def _enter_region(x, mesh, axis, seq_dim):
    if seq_dim is not None:
        return _Gather.apply(x, mesh, axis, seq_dim, True)
    if not (torch.is_grad_enabled() and x.requires_grad):
        return x
    return _ReduceGrad.apply(x, mesh, axis)


@torch.compiler.disable
def _exit_region(x, mesh, axis, seq_dim):
    if seq_dim is not None:
        return _Scatter.apply(x, mesh, axis, seq_dim)
    return _ReduceOutput.apply(x, mesh, axis)


@torch.compiler.disable
def _gather(x, mesh, axis, dim):
    return _Gather.apply(x, mesh, axis, dim, False)


@torch.compiler.disable
def _slice(x, mesh, axis, dim):
    return _Slice.apply(x, mesh, axis, dim)


@torch.compiler.disable
def _reduce_grad(x, mesh, axis):
    if not (torch.is_grad_enabled() and x.requires_grad):
        return x
    return _ReduceGrad.apply(x, mesh, axis)


class TP:
    def __init__(
        self,
        mesh: MeshLike,
        column: Iterable[nn.Linear] = (),
        row: Iterable[nn.Linear] = (),
        regions: Iterable[nn.Module] = (),
        shared: Iterable[nn.Module] = (),
        embeddings: Iterable[nn.Embedding] = (),
        heads: Iterable[nn.Linear] = (),
        vocab: Iterable[nn.Module] = (),
        seq_local: Iterable[nn.Module] = (),
        dim: str = "tp",
        sequence_parallel: bool = False,
        seq_dim: int = 1,
    ):
        self.mesh = mesh
        self.dim = dim
        self.group_size = mesh.size(dim)
        self.sequence_parallel = sequence_parallel and self.group_size > 1
        # the tensor dim activations are sharded along between regions, or None
        self._seq_dim = seq_dim if self.sequence_parallel else None

        # (module, attribute, sharded tensor dim, unsharded shape). looked up
        # through the module, not by param id: materializing a meta model
        # replaces the Parameter objects
        self._entries: list[tuple[nn.Module, str, int, tuple[int, ...]]] = []
        self._replaced: dict[int, tuple[nn.Parameter, nn.Parameter]] = {}
        self._by_id: dict[int, tuple[int, tuple[int, ...]]] | None = None
        self._shared_modules: list[nn.Module] = []
        self._vocab_rows: dict[int, int] = {}       # id(module) -> rows per rank
        self._outside: dict[int, torch.Tensor] = {}  # id(embedding) -> ids not in its rows
        self.vocab_parallel = False
        if self.group_size == 1:
            return

        for linear in column:
            self._shard(linear, "weight", 0)
            if linear.bias is not None:
                self._shard(linear, "bias", 0)
            linear.out_features = linear.weight.shape[0]
        for linear in row:
            assert linear.bias is None, (
                "a row-parallel linear can't have a bias: every rank would add "
                "it and the reduction would sum it tp times"
            )
            self._shard(linear, "weight", 1)
            linear.in_features = linear.weight.shape[1]

        embeddings, heads, vocab = list(embeddings), list(heads), list(vocab)
        vocab_ids = {id(m) for m in vocab}
        assert vocab_ids <= {id(m) for m in embeddings + heads}, (
            "every `vocab` module must also be listed under embeddings or heads"
        )
        for module in vocab:
            self._shard(module, "weight", 0)
            self._vocab_rows[id(module)] = module.weight.shape[0]
        self.vocab_parallel = bool(vocab)

        self._shared_modules = list(shared)
        if self.sequence_parallel:
            self._shared_modules += list(seq_local)

        for region in regions:
            region.register_forward_pre_hook(self._region_pre)
            region.register_forward_hook(self._region_post)
        for emb in embeddings:
            if id(emb) in vocab_ids:
                emb.register_forward_pre_hook(self._embedding_pre)
                emb.register_forward_hook(self._embedding_post)
            elif self.sequence_parallel:
                emb.register_forward_hook(self._seq_slice_post)
        for head in heads:
            if id(head) in vocab_ids:
                head.register_forward_pre_hook(self._vocab_head_pre)
                head.register_forward_hook(self._vocab_head_post)
            elif self.sequence_parallel:
                head.register_forward_pre_hook(self._head_pre)

    # ---- construction ----------------------------------------------------

    def _shard(self, module: nn.Module, name: str, tensor_dim: int) -> None:
        """
        make `module.<name>` this rank's slice along `tensor_dim`: sliced out
        of real weights, or just reshaped when the model is on meta. a param
        reached twice (a tied weight) is sharded once and stays tied.
        """
        old = module._parameters[name]
        if id(old) in self._replaced:
            # the tie's other end: point it at the same sharded Parameter
            module._parameters[name] = self._replaced[id(old)][1]
            dim, full = self._lookup()[id(self._replaced[id(old)][1])]
            assert dim == tensor_dim, f"tied param sharded on dims {dim} and {tensor_dim}"
            self._entries.append((module, name, dim, full))
            return
        full_shape = tuple(old.shape)
        assert full_shape[tensor_dim] % self.group_size == 0, (
            f"can't split dim {tensor_dim} of shape {full_shape} over "
            f"{self.dim}={self.group_size}"
        )
        if old.is_meta:
            local_shape = Shard(tensor_dim).local_shape(full_shape, self.group_size)
            new = nn.Parameter(
                torch.empty(local_shape, dtype=old.dtype, device="meta"),
                requires_grad=old.requires_grad,
            )
            module._parameters[name] = new
        else:
            local = Shard(tensor_dim).local_slice(
                old.data, self.group_size, self.mesh.coordinate(self.dim)
            )
            # an owning copy, so the full weight is freed
            old.data = local.contiguous().clone()
            new = old
        # holding `old` keeps its id from being reused while we key on it
        self._replaced[id(old)] = (old, new)
        self._entries.append((module, name, tensor_dim, full_shape))
        self._by_id = None

    def _lookup(self) -> dict[int, tuple[int, tuple[int, ...]]]:
        """id(current param) -> (sharded dim, unsharded shape), rebuilt if params were replaced."""
        if self._by_id is not None and self._entries:
            module, name, _, _ = self._entries[0]
            if id(module._parameters[name]) not in self._by_id:
                self._by_id = None
        if self._by_id is None:
            self._by_id = {
                id(module._parameters[name]): (dim, full)
                for module, name, dim, full in self._entries
            }
        return self._by_id

    def finish_init(self, model: nn.Module) -> None:
        """
        call once after initializing a model built from meta (and after
        fsdp's finish_init, if any). each rank initialized on its own seed so
        its tp shards are its own; this makes everything NOT sharded over tp,
        buffers included, agree with tp coordinate 0.
        """
        if self.group_size == 1:
            return
        sharded = self._lookup()
        for param in model.parameters():
            if id(param) not in sharded:
                broadcast(param.data, self.mesh, self.dim, src=0)
        for buf in model.buffers():
            broadcast(buf, self.mesh, self.dim, src=0)

    # ---- hooks -----------------------------------------------------------

    def _region_pre(self, module: nn.Module, args: tuple) -> tuple:
        return (_enter_region(args[0], self.mesh, self.dim, self._seq_dim), *args[1:])

    def _region_post(self, module: nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
        return _exit_region(output, self.mesh, self.dim, self._seq_dim)

    def _embedding_pre(self, module: nn.Module, args: tuple) -> tuple:
        # ids -> row indices into this rank's shard; ids it doesn't hold look
        # up row 0 and are zeroed in the post hook
        rows = self._vocab_rows[id(module)]
        local = args[0] - self.mesh.coordinate(self.dim) * rows
        outside = (local < 0) | (local >= rows)
        self._outside[id(module)] = outside
        return (local.masked_fill(outside, 0), *args[1:])

    def _embedding_post(self, module: nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
        outside = self._outside.pop(id(module))
        output = output.masked_fill(outside.unsqueeze(-1), 0)
        return _exit_region(output, self.mesh, self.dim, self._seq_dim)

    def _seq_slice_post(self, module: nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
        return _slice(output, self.mesh, self.dim, self._seq_dim)

    def _head_pre(self, module: nn.Module, args: tuple) -> tuple:
        return (self.head_input(args[0]), *args[1:])

    def _vocab_head_pre(self, module: nn.Module, args: tuple) -> tuple:
        # a column linear: the grad of its input is a partial sum
        return (_reduce_grad(self.head_input(args[0]), self.mesh, self.dim), *args[1:])

    def _vocab_head_post(self, module: nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
        return _gather(output, self.mesh, self.dim, -1)

    def head_input(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        the final hidden states as a head (or a loss computed straight from
        the head's weight) must see them: the whole sequence. a gather under
        sequence parallel, otherwise a no-op. the head hooks call this; a
        loss that bypasses the head module must call it itself.
        """
        if self._seq_dim is None:
            return hidden
        return _gather(hidden, self.mesh, self.dim, self._seq_dim)

    @property
    def vocab_axis(self) -> tuple[MeshLike, str] | None:
        """(mesh, axis) the vocab is sharded over, or None if it isn't."""
        return (self.mesh, self.dim) if self.vocab_parallel else None

    # ---- trainer surface -------------------------------------------------

    @property
    def shard_dims(self) -> tuple[str, ...]:
        """mesh axes param/grad state is sharded over."""
        return (self.dim,) if self.group_size > 1 else ()

    def shard_dim_of(self, param: nn.Parameter) -> int | None:
        """the tensor dim `param` is split along over tp, or None if replicated."""
        entry = self._lookup().get(id(param))
        return None if entry is None else entry[0]

    def full_shape_of(self, param: nn.Parameter) -> tuple[int, ...]:
        """the shape `param` had before tp split it (its own shape if it didn't)."""
        entry = self._lookup().get(id(param))
        return tuple(param.shape) if entry is None else entry[1]

    def shard_dims_of(self, param: nn.Parameter) -> tuple[str, ...]:
        """the mesh axes this one param is sharded over: (tp,) or ()."""
        return (self.dim,) if id(param) in self._lookup() else ()

    def sync(self) -> None:
        """
        finish the grads of the shared params: one all_reduce(sum) over tp for
        all of them. call once per optimizer step, after the last backward and
        after the data-parallel sync() has returned.
        """
        seen: set[int] = set()
        grads = []
        for module in self._shared_modules:
            for param in module.parameters():
                if param.grad is not None and id(param) not in seen:
                    seen.add(id(param))
                    grads.append(param.grad)
        if not grads:
            return
        flat = torch.cat([g.reshape(-1) for g in grads])
        all_reduce(flat, self.mesh, self.dim, "sum")
        offset = 0
        for grad in grads:
            grad.copy_(flat[offset: offset + grad.numel()].view_as(grad))
            offset += grad.numel()
