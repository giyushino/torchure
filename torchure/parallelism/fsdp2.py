"""
fsdp2: per-parameter Shard(0) sharding along `dp_shard`; hsdp when
`replicate_dim` is set too.

each param keeps ONE nn.Parameter whose `.data` is swapped between its fp32
shard (optimizer) and a gathered full buffer (forward/backward). that keeps the
tied embedding tied, lets grad hooks register once, and means the optimizer
never sees a DTensor.

units: the caller passes the submodules to treat as units (for a transformer,
its blocks); fsdp never looks at the model's attribute names. unit 0 is the
root unit, everything NOT inside a passed unit (for a transformer: embed, final
norm, head). it is held unsharded from forward until its backward, because
code outside `model.forward` may still read those weights (a fused loss reading
the head weight, for one). each passed unit goes through:

    forward-pre    all_gather; prefetch next unit
    forward-post   reshard (storage resize to 0, not a dropped ref)
    backward-pre   re-gather; prefetch previous unit
    post-backward  reduce_scatter grads -> sharded fp32, reshard

grads are reduce-scattered every microbatch, so `requires_sync` is ignored.
not done yet: dedicated comm stream, flat per-unit buffers, uneven shards.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from torchure.core.collective import (
    MeshLike,
    all_gather,
    all_reduce,
    broadcast,
    reduce_scatter,
)
from torchure.core.placement import Placements, Replicate, Shard

# unit lifecycle states
_FREE = "free"            # params point at shards; buffer storage is 0 bytes
_GATHERING = "gathering"  # all_gathers launched, not yet waited
_UNSHARDED = "unsharded"  # params point at the full buffers


def _free_storage(t: torch.Tensor) -> None:
    """release a buffer's memory while keeping the tensor object and shape.

    a dropped python reference is NOT enough: activations saved for backward
    hold the weight tensor alive, so the allocator never gets the block back.
    resizing the storage is what actually frees it, and it is why the same
    tensor object (and therefore the same compile guard) survives every step.
    """
    t.untyped_storage().resize_(0)


def _alloc_storage(t: torch.Tensor) -> None:
    nbytes = t.numel() * t.element_size()
    if t.untyped_storage().size() != nbytes:
        t.untyped_storage().resize_(nbytes)


@dataclass
class _ParamState:
    name: str
    param: nn.Parameter
    sharded: torch.Tensor           # fp32 Shard(0) master; param.data when free
    unsharded: torch.Tensor         # compute-dtype full buffer; storage 0 when free
    placements: Placements          # layout over (dp_replicate, dp_shard)
    sharded_grad: torch.Tensor | None = None


@dataclass
class _Unit:
    index: int
    module: nn.Module
    params: list[_ParamState]
    reshard_after_forward: bool
    status: str = _FREE
    pending: int = 0                # params still owing a grad this backward
    works: list = field(default_factory=list)


@dataclass
class _PendingReduce:
    state: _ParamState
    out: torch.Tensor
    work: object
    grad_ref: torch.Tensor          # keeps the input alive until the work completes


class FSDP:
    """
    shard `model` along `shard_dim`; a no-op when that axis has size 1.
    `units` are the submodules gathered/resharded independently, in any order
    (execution order is recorded on the first forward); params outside them
    form the root unit. units must not nest or share a param.
    `replicate_dim` adds hsdp: reduce-scatter on shard_dim, then all-reduce on
    replicate_dim. `param_dtype` is the gathered/compute dtype (None = no
    cast); `reduce_dtype` is the grad reduction dtype (None = grad's dtype).
    """

    def __init__(
        self,
        model: nn.Module,
        mesh: MeshLike,
        units: Iterable[nn.Module],
        shard_dim: str = "dp_shard",
        replicate_dim: str | None = "dp_replicate",
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = torch.float32,
        prefetch: int = 1,
    ):
        self.model = model
        self.mesh = mesh
        self.shard_dim = shard_dim
        self.replicate_dim = replicate_dim
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.prefetch = prefetch
        self.group_size = mesh.size(shard_dim)
        self.replicate_size = (
            mesh.size(replicate_dim) if replicate_dim is not None else 1
        )

        # mirrors DDP.requires_sync so trainer loops stay one idiom; see the
        # property below for why fsdp ignores it
        self._requires_sync = True
        self._pending: list[_PendingReduce] = []        # reduce-scatters in flight
        self._all_reducing: list[_PendingReduce] = []   # hsdp all-reduces in flight
        self._fired: set[int] = set()
        self._backwards = 0
        self._units: list[_Unit] = []
        self._states: list[_ParamState] = []
        self._by_param: dict[int, tuple[_Unit, _ParamState]] = {}
        self._exec_order: list[int] = []
        self._order_checked = False

        if self.group_size == 1:
            # no-op: not one hook registered, not one buffer allocated
            return

        self._replicate(model)
        self._build_units(model, list(units))
        self._shard()
        self._register_hooks()

    # ---- construction ----------------------------------------------------

    def _replicate(self, model: nn.Module) -> None:
        """
        make every rank's full weights identical BEFORE slicing.

        this is load-bearing, not defensive. v0 shards by "init the full
        model on every rank, keep your own slice" -- which is only correct if
        the ranks agreed on the full model, and nothing in this repo seeds
        torch. without this broadcast each rank would keep a slice of a
        DIFFERENT model and train a silently corrupt average. broadcasting
        (rather than seeding) also stays correct if init order ever changes.
        """
        for tensor in [*model.parameters(), *model.buffers()]:
            broadcast(tensor.detach(), self.mesh, self.shard_dim, src=0)
            if self.replicate_size > 1:
                broadcast(tensor.detach(), self.mesh, self.replicate_dim, src=0)

    def _build_units(self, model: nn.Module, modules: list[nn.Module]) -> None:
        # qualified names, so states (and later the sharded checkpoint) are
        # keyed the way model.named_parameters() keys them
        prefix_of = {id(m): name for name, m in model.named_modules()}
        for module in modules:
            assert id(module) in prefix_of and module is not model, (
                f"fsdp unit {type(module).__name__} is not a submodule of the model"
            )
        unit_param_ids = {id(p) for m in modules for p in m.parameters()}

        # unit 0, the root: everything NOT in a passed unit. a weight tied
        # between two root modules (embed + head) lands here once, because
        # named_parameters() dedupes.
        root_named = [
            (n, p)
            for n, p in model.named_parameters()
            if p.requires_grad and id(p) not in unit_param_ids
        ]
        units = [(model, root_named, False)]
        for module in modules:
            prefix = prefix_of[id(module)]
            named = [
                (f"{prefix}.{n}", p)
                for n, p in module.named_parameters()
                if p.requires_grad
            ]
            units.append((module, named, True))

        seen: dict[int, int] = {}
        for index, (module, named, reshard) in enumerate(units):
            states = []
            for name, param in named:
                # a param reachable from two units would be gathered twice
                # into two storages and get half its grad (a weight tied
                # across units, or one unit nested inside another)
                assert id(param) not in seen, (
                    f"param {name!r} appears in units {seen[id(param)]} and {index}; "
                    "a shared weight must live in exactly one unit"
                )
                seen[id(param)] = index
                states.append(self._make_state(name, param))
            unit = _Unit(index, module, states, reshard_after_forward=reshard)
            unit.pending = len(states)
            self._units.append(unit)
            self._states.extend(states)
            for st in states:
                self._by_param[id(st.param)] = (unit, st)

    def _make_state(self, name: str, param: nn.Parameter) -> _ParamState:
        placement = Shard(0)
        full = param.data
        assert full.dim() > 0, (
            f"param {name!r} is 0-dim; Shard(0) needs a dim to split. make it "
            "shape (1,)-or-larger, or freeze it"
        )
        assert full.shape[0] % self.group_size == 0, (
            f"param {name!r} has shape {tuple(full.shape)}; dim 0 is not divisible "
            f"by dp_shard={self.group_size}. v0 asserts rather than padding -- "
            "padding would go in placement.Shard (pad the shard, trim after the "
            "gather) and the unpadded size would have to reach the checkpoint"
        )
        coord = self.mesh.coordinate(self.shard_dim)
        # .clone() so the master owns its storage: a bare narrow() view would
        # keep the whole unsharded tensor alive and save exactly nothing
        sharded = (
            placement.local_slice(full, self.group_size, coord).contiguous().clone()
        )

        buf_dtype = self.param_dtype or full.dtype
        unsharded = torch.empty(full.shape, dtype=buf_dtype, device=full.device)
        _free_storage(unsharded)

        placements: Placements = (
            (Replicate(), placement) if self.replicate_size > 1 else (placement,)
        )
        return _ParamState(name, param, sharded, unsharded, placements)

    def _shard(self) -> None:
        """point every param at its master shard and drop the full weights."""
        for st in self._states:
            st.param.data = st.sharded

    def _register_hooks(self) -> None:
        for unit in self._units:
            unit.module.register_forward_pre_hook(self._make_forward_pre(unit))
            if unit.reshard_after_forward:
                unit.module.register_forward_hook(self._make_forward_post(unit))
                unit.module.register_full_backward_pre_hook(
                    self._make_backward_pre(unit)
                )
            for st in unit.params:
                st.param.register_post_accumulate_grad_hook(self._make_grad_hook(unit))

    # ---- gather / free ---------------------------------------------------

    def _start_gather(self, unit: _Unit) -> None:
        """launch the unit's all_gathers; does not wait. idempotent."""
        if unit.status != _FREE:
            return
        for st in unit.params:
            _alloc_storage(st.unsharded)
            src = st.sharded
            if src.dtype != st.unsharded.dtype:
                # the bf16 staging copy of the local shard. it must stay
                # referenced until the collective completes, which is what the
                # tuple below is for -- dropping it here is a use-after-free
                # that only shows up under load.
                src = src.to(st.unsharded.dtype)
            _, work = all_gather(
                src,
                self.mesh,
                self.shard_dim,
                gather_dim=0,
                async_op=True,
                output_tensor=st.unsharded,
            )
            unit.works.append((work, src))
        unit.status = _GATHERING

    def _wait_gather(self, unit: _Unit) -> None:
        """block until the unit's params are usable, then point them at it."""
        if unit.status == _UNSHARDED:
            return
        self._start_gather(unit)
        for work, _src in unit.works:
            work.wait()
        unit.works.clear()
        for st in unit.params:
            st.param.data = st.unsharded
            self._detach_grad(st)
        unit.status = _UNSHARDED

    def _detach_grad(self, st: _ParamState) -> None:
        """
        clear `.grad` before a backward writes an unsharded-shaped one into it.

        after sync() publishes, `param.grad` holds the sharded fp32 grad; the
        next backward would try to accumulate an unsharded grad into it and
        hit a shape error. the normal path is that the trainer's zero_grad
        already set it to None. if it didn't -- sync() called mid-accumulation
        -- take the value back into sharded_grad instead of dropping it, so no
        microbatch's contribution is silently lost.
        """
        grad = st.param.grad
        if grad is None:
            return
        assert grad.shape == st.sharded.shape, (
            f"param {st.name!r} entered forward holding an unsharded grad "
            f"{tuple(grad.shape)}; expected the sharded {tuple(st.sharded.shape)}"
        )
        st.sharded_grad = grad if st.sharded_grad is None else st.sharded_grad + grad
        st.param.grad = None

    def _reshard(self, unit: _Unit) -> None:
        if unit.status == _FREE:
            return
        assert unit.status == _UNSHARDED, f"reshard of unit {unit.index} mid-gather"
        for st in unit.params:
            st.param.data = st.sharded
        for st in unit.params:
            _free_storage(st.unsharded)
        unit.status = _FREE

    # ---- hooks ----------------------------------------------------------
    #
    # every hook body is @torch.compiler.disable'd. `Module.compile()` compiles
    # `_call_impl`, NOT just `forward`, so dynamo traces registered hooks into
    # the block's graph -- and these hooks resize storages. inductor then sees
    # `inductor.resize_storage_bytes_` with a symbolic size and dies with
    # "LoweringException: TODO: dynamic shapes" (the size is symbolic because
    # dynamo watched param.data change shape between the sharded and unsharded
    # forms and gave up on specializing it).
    #
    # disabling makes dynamo treat the hook as an opaque call and graph-break
    # there. that costs nothing: the break lands at the block boundary, which
    # is where the compiled region already starts and ends, so the block body
    # is still one graph. this is the concrete reason per-block granularity was
    # the right call -- at whole-model granularity every one of these breaks
    # would be splitting the middle of the graph.

    def _make_forward_pre(self, unit: _Unit):
        @torch.compiler.disable
        def hook(module, args):
            if not self._order_checked:
                self._exec_order.append(unit.index)
            self._wait_gather(unit)
            # prefetch: launch the NEXT units' gathers so their bytes move
            # while this unit computes. without this the step is strictly
            # serial comm -> compute and the tps number is ugly.
            for nxt in self._neighbours(unit.index, forward=True):
                self._start_gather(nxt)

        return hook

    def _make_forward_post(self, unit: _Unit):
        @torch.compiler.disable
        def hook(module, args, output):
            self._reshard(unit)

        return hook

    def _make_backward_pre(self, unit: _Unit):
        @torch.compiler.disable
        def hook(module, grad_output):
            self._wait_gather(unit)
            for prv in self._neighbours(unit.index, forward=False):
                self._start_gather(prv)

        return hook

    def _make_grad_hook(self, unit: _Unit):
        @torch.compiler.disable
        def hook(param):
            unit.pending -= 1
            if unit.pending == 0:
                unit.pending = len(unit.params)
                self._post_backward(unit)

        return hook

    def _neighbours(self, index: int, forward: bool) -> list[_Unit]:
        """the next (or previous) `prefetch` units in execution order."""
        order = (
            self._exec_order if self._order_checked else list(range(len(self._units)))
        )
        try:
            pos = order.index(index)
        except ValueError:
            return []
        step = 1 if forward else -1
        out = []
        for k in range(1, self.prefetch + 1):
            j = pos + step * k
            if 0 <= j < len(order):
                out.append(self._units[order[j]])
        return out

    # ---- grad reduction --------------------------------------------------

    def _post_backward(self, unit: _Unit) -> None:
        # drain the PREVIOUS unit's reduce-scatters first: they have had this
        # unit's whole backward to complete, so the wait is usually free, and
        # it bounds in-flight unsharded grads to one unit's worth
        self._drain()

        for st in unit.params:
            grad = st.param.grad
            assert grad is not None, (
                f"param {st.name!r} fired its post-accumulate hook with no grad"
            )
            st.param.grad = None
            if self.reduce_dtype is not None and grad.dtype != self.reduce_dtype:
                grad = grad.to(self.reduce_dtype)
            out, work = reduce_scatter(
                grad, self.mesh, self.shard_dim, "avg", scatter_dim=0, async_op=True
            )
            self._pending.append(_PendingReduce(st, out, work, grad))

        self._reshard(unit)

        self._fired.add(unit.index)
        if len(self._fired) == len(self._units):
            self._fired.clear()
            self._backwards += 1
            if not self._order_checked and self._exec_order:
                self._check_order()

    def _drain(self, final: bool = False) -> None:
        """
        hsdp pipeline: wait the previous unit's reduce-scatters and launch
        their replica all-reduces async, finishing the ones from the unit
        before. `final` (from sync) flushes everything.
        """
        self._finish_all_reduces()
        while self._pending:
            p = self._pending.pop(0)
            p.work.wait()
            if self.replicate_size > 1:
                _, work = all_reduce(
                    p.out, self.mesh, self.replicate_dim, "avg", async_op=True
                )
                self._all_reducing.append(_PendingReduce(p.state, p.out, work, p.out))
            else:
                self._accumulate(p.state, p.out)
        if final:
            self._finish_all_reduces()

    def _finish_all_reduces(self) -> None:
        while self._all_reducing:
            p = self._all_reducing.pop(0)
            p.work.wait()
            self._accumulate(p.state, p.out)

    @staticmethod
    def _accumulate(st: _ParamState, out: torch.Tensor) -> None:
        if out.dtype != st.sharded.dtype:
            out = out.to(st.sharded.dtype)
        st.sharded_grad = out if st.sharded_grad is None else st.sharded_grad + out

    def _check_order(self) -> None:
        """
        pin the recorded forward order once. a mismatch against the static
        order degrades prefetch silently into no prefetching at all, which
        costs throughput and nothing else -- so it is an assert, not a guess.
        """
        self._order_checked = True
        assert sorted(self._exec_order) == list(range(len(self._units))), (
            f"forward visited units {self._exec_order}, expected each of "
            f"{list(range(len(self._units)))} exactly once"
        )

    # ---- trainer surface -------------------------------------------------

    @property
    def shard_dims(self) -> tuple[str, ...]:
        """
        mesh axes grad state is sharded over, for clip_grad_norm. under hsdp
        grads are already averaged across dp_replicate, so only dp_shard.
        """
        return (self.shard_dim,) if self.group_size > 1 else ()

    def shard_dim_of(self, param: nn.Parameter) -> int | None:
        """
        the tensor dim `param` is sharded along, or None if it isn't sharded.
        what the checkpointer needs to save and reshard this run's state.
        """
        entry = self._by_param.get(id(param))
        if entry is None:
            return None
        unit, st = entry
        assert unit.status == _FREE, (
            f"param {st.name!r} is gathered (unit {unit.index} is {unit.status}); "
            "checkpoint between steps, when every param points at its shard"
        )
        return 0

    @property
    def requires_sync(self) -> bool:
        return self._requires_sync

    @requires_sync.setter
    def requires_sync(self, value: bool) -> None:
        """
        accepted and ignored, so the trainer's accumulation loop is identical
        for ddp and fsdp.

        ddp uses it to suppress grad sync until the last microbatch, because
        its collective is over the FULL grad and doing it K times would be K
        times the bytes. fsdp reduce-scatters every microbatch on purpose: the
        alternative is holding a full unsharded grad per unit across the whole
        accumulation, which is the memory this phase exists to reclaim.
        """
        self._requires_sync = value

    def sync(self) -> None:
        """
        wait for every outstanding reduce-scatter and publish `.grad`.

        call between the last backward of a step and clip/optimizer.step, the
        same place DDP.sync() goes. after this, every param's `.grad` is the
        sharded fp32 gradient and has the same shape as `param.data`, so
        clip_grad_norm (with shard_dims=("dp_shard",)) and the optimizer both
        see consistent shapes.
        """
        if self.group_size == 1:
            return
        self._drain(final=True)
        assert not self._fired, (
            f"units {sorted(self._fired)} of {len(self._units)} finished their "
            "backward -- some unit never completed. a frozen or unused param "
            "leaves its unit's counter permanently short"
        )
        assert self._backwards > 0, (
            "sync() with no completed backward since the last sync()"
        )
        for st in self._states:
            if st.sharded_grad is not None:
                # hand ownership to .grad; the optimizer's zero_grad frees it,
                # and starting the next step from None is what keeps
                # accumulation from carrying across steps
                st.param.grad = st.sharded_grad
                st.sharded_grad = None
        self._backwards = 0

    # ---- introspection (used by tests/fsdp2.py) --------------------------

    def sharded_numel(self) -> int:
        """total params this rank actually owns. the memory exit criterion."""
        return sum(st.sharded.numel() for st in self._states)

    def unsharded_bytes_live(self) -> int:
        """bytes currently held by gathered buffers, across all units."""
        return sum(st.unsharded.untyped_storage().size() for st in self._states)

    def placements_of(self, name: str) -> Placements:
        for st in self._states:
            if st.name == name:
                return st.placements
        raise KeyError(name)
