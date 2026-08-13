"""
fsdp2: per-parameter sharding along the mesh's `dp_shard` axis.

each parameter's master copy is a Shard(0) slice in fp32. before a unit runs,
its shards are all-gathered into a full-size buffer (optionally cast to a
compute dtype on the way); after it runs, that buffer's storage is released.
gradients are reduce-scattered back down to sharded fp32, so params, grads and
optimizer state all scale as 1/dp_shard -- the last of those being the actual
memory win, and it happens for free only because the optimizer is built from
`model.parameters()` AFTER sharding.

## the one structural idea

each logical parameter keeps exactly ONE nn.Parameter object for the lifetime
of the model, and `param.data` is swapped between two storages:

    resharded:  param.data -> the fp32 Shard(0) master     (what the optimizer sees)
    unsharded:  param.data -> the compute-dtype full buffer (what forward sees)

that single decision buys three things that are otherwise fiddly:

- **the tied embedding works for free.** `lm_head.weight is token_emb.weight`
  is one Parameter object reachable from two modules, so swapping `.data`
  updates both views at once. there is no way to accidentally gather it twice
  or give the two modules different storages. the ctor asserts the tie rather
  than assuming it.
- **grad hooks are registered once.** `register_post_accumulate_grad_hook`
  lives on the Parameter, so it survives every swap -- the same mechanism ddp
  uses, with one counter per unit instead of per bucket.
- **the optimizer never sees a DTensor.** params are plain tensors at runtime
  with their placements recorded in `_ParamState`; fused AdamW, the
  `torch._foreach_*` calls in grad_helper, and every `p.grad` access work
  untouched. see core/dtensor.py for why that decision was made.

## units

a unit is the granularity of one gather. unit 0 is everything outside
`model.blocks` -- token_emb, the final norm, lm_head -- and each transformer
block is its own unit. the tied weight FORCES embed and head into one unit;
splitting them would double-gather one storage and corrupt the grad flow.

unit 0 is gathered at the model's forward-pre hook and is **not** resharded
until its backward completes. that is deliberate: `ARObjective.compute_loss`
calls `model(..., return_final_hidden=True)` and then hands
`model.lm_head.weight` to the fused linear+CE, so the head weight is consumed
*after* the model's forward has returned -- any reshard on a forward-post hook
would hand the objective a freed weight. holding one unit unsharded for the
step costs one transient full-size copy of the embedding and does not touch
the memory exit criterion, which is about master params + grads + adam state.

## lifecycle per block unit

    forward-pre    all_gather shards -> full buffer; prefetch the NEXT unit
    forward-post   point params back at the shards, resize the buffer to 0
    backward-pre   re-gather; prefetch the PREVIOUS unit
    post-backward  reduce_scatter grads -> sharded fp32, free, reshard

resharding is a storage resize, not a dropped python reference. saved
activations from the block's backward hold a reference to the weight tensor,
so refcounting alone would never release it -- the run would be numerically
perfect and save no memory at all, which is exactly why tests/fsdp2.py asserts
storage size directly rather than trusting a peak-memory number.

## grad accumulation

grads are reduce-scattered EVERY microbatch and accumulated in the sharded
fp32 `sharded_grad`, rather than accumulated unsharded and reduced once. that
is one extra collective per microbatch in exchange for never holding a full
unsharded grad per unit, which is the whole reason to be here. "avg" is linear,
so avg-then-sum over microbatches equals ddp's sum-then-avg. `requires_sync`
exists only so the trainer loop can stay one idiom across ddp and fsdp; fsdp
ignores it (see the attribute's docstring).

## what v0 does NOT do

- **no dedicated comm stream.** collectives are launched with `async_op=True`
  on the default stream, which is how parallelism/data_parallel.py already
  gets its overlap on nccl. a stream fsdp owns, plus explicit events and
  buffer lifetime across the stream boundary, is the next perf step and the
  single biggest source of nondeterministic corruption -- it should land with
  a profiler trace proving it, not on faith.
- **one all_gather per PARAM, not per unit.** a flat per-unit buffer would cut
  ~9 collectives per block to 1, but the gathered layout is rank-major, so
  reassembling per-param views out of it needs a copy-out pass
  (`split_with_sizes_copy`) and a second live buffer. correctness first: the
  per-param path needs neither, and the structure here (buffers owned by
  `_ParamState`, gathers launched by the unit) is what a flat-buffer swap
  would replace.
- **no activation checkpointing interaction.** recompute re-runs a block's
  forward during backward, so the unit must be unsharded then too. the
  backward-pre hook happens to do exactly that, but it is untested.
- **uneven shards.** divisibility is asserted loudly; nothing pads.
"""

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
    placements: Placements          # recorded for phase 3's sharded checkpoint
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
    shard `model` along `mesh`'s `shard_dim` axis.

    at `mesh.size(shard_dim) == 1` this is a total no-op -- no hooks, no
    buffers, no collectives -- the same way DDP is at dp=1, so a config can
    drop to one rank without changing code paths.

    `replicate_dim` turns this into hsdp: grads are reduce-scattered along
    `shard_dim`, then all-reduced along `replicate_dim`. at size 1 that second
    collective is skipped entirely.

    dtypes: `param_dtype` is what the gathered buffers (and therefore the
    forward) use; None means "no cast", which keeps an fp32 model bit-exact
    and is what the cpu parity tests rely on. masters are always the dtype the
    model was built in. `reduce_dtype` is what grads are reduced in: fp32
    costs an upcast of the unsharded grad but avoids accumulation error across
    dp_shard ranks; None means "reduce in whatever dtype the grad already is",
    which halves the bytes when param_dtype is bf16.
    """

    def __init__(
        self,
        model: nn.Module,
        mesh: MeshLike,
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
        self._pending: list[_PendingReduce] = []
        self._fired: set[int] = set()
        self._backwards = 0
        self._units: list[_Unit] = []
        self._states: list[_ParamState] = []
        self._exec_order: list[int] = []
        self._order_checked = False

        if self.group_size == 1:
            # no-op: not one hook registered, not one buffer allocated
            return

        self._replicate(model)
        self._build_units(model)
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

    def _build_units(self, model: nn.Module) -> None:
        assert hasattr(model, "blocks"), (
            "fsdp needs model.blocks (roadmap principle 4: per-block granularity "
            "is the contract for compile, wrapping, checkpointing and pp splits)"
        )
        block_param_ids = {id(p) for b in model.blocks for p in b.parameters()}

        # unit 0: everything NOT in a block -- token_emb, final norm, lm_head.
        # the tie puts embed and head here together by construction.
        root_named = [
            (n, p)
            for n, p in model.named_parameters()
            if p.requires_grad and id(p) not in block_param_ids
        ]
        units = [(model, root_named, False)]
        for block in model.blocks:
            named = [(n, p) for n, p in block.named_parameters() if p.requires_grad]
            units.append((block, named, True))

        seen: dict[int, int] = {}
        for index, (module, named, reshard) in enumerate(units):
            states = []
            for name, param in named:
                # a param reachable from two units would be gathered twice
                # into two storages and get half its grad -- the exact failure
                # the tied embedding would cause if it were split
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

        if hasattr(model, "lm_head") and hasattr(model, "token_emb"):
            assert model.lm_head.weight is model.token_emb.weight, (
                "expected tied lm_head/token_emb weights. an untied head is fine "
                "in principle but changes unit 0's memory profile, so fail loudly "
                "rather than silently train something the notes don't describe"
            )

    def _make_state(self, name: str, param: nn.Parameter) -> _ParamState:
        placement = Shard(0)
        full = param.data
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

    def _drain(self) -> None:
        while self._pending:
            p = self._pending.pop(0)
            p.work.wait()
            out = p.out
            if self.replicate_size > 1:
                # hsdp: the shards are reduced within dp_shard, now average the
                # replicas. two collectives, one ordering, no new state.
                all_reduce(out, self.mesh, self.replicate_dim, "avg")
            if out.dtype != p.state.sharded.dtype:
                out = out.to(p.state.sharded.dtype)
            st = p.state
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
        self._drain()
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
