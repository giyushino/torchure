"""
fsdp2: shard params, grads and optimizer state along `dp_shard`; gather a
unit's weights just before it runs and free them right after.

scope/checklist: notes/FSDP2_SCOPE.md. measurements: notes/FSDP2_H100.md.
tests: tests/fsdp2.py (gloo/cpu, mp.spawn).

===========================================================================
design
===========================================================================

**unit = one transformer block**, plus a root unit (token_emb + lm_head +
final norm). the tie `lm_head.weight is token_emb.weight` forces those into
one unit. the root is `reshard_after_forward=False`: the fused CE reads
`model.lm_head.weight` after the model's forward has returned.

**params stay plain nn.Parameter; `.data` is swapped.** the same Parameter
objects live for the whole run:

    sharded:    p.data -> fp32 view into unit.shard_flat        (optimizer)
    unsharded:  p.data -> bf16 view into unit.unsharded_flat    (compute)

the optimizer is built after _parallelize, so adamw allocates its state
against the sharded shapes. that's the memory win.

**per-param Shard(0), one flat shard buffer per unit, copy-out after gather.**
rank r's shard_flat is each param's r-th dim-0 chunk, concatenated:

    shard_flat (rank r):   [ p0[r] | p1[r] | p2[r] ]          S elements

all_gather stacks the rank-major buffers, so the gathered buffer is

    gather_buf (R, S):     [ p0[0] p1[0] p2[0] ]
                           [ p0[1] p1[1] p2[1] ]
                           ...

and p0 is NOT contiguous in it: its chunks sit S elements apart. no view
fixes this (F.linear won't take a strided 3-d weight), so unshard does one
on-device copy-out into a per-param-contiguous buffer:

    unsharded_flat:        [ p0 (all R chunks) | p1 | p2 ]      R*S elements

    torch.split_with_sizes_copy(gather_buf.view(R, S), [n0, n1, n2], dim=1,
                                out=[u.view(R, n) for u in unsharded views])

reduce-scatter is the mirror: grads arrive per-param contiguous, get
chunk-cat'd into (R, S) rank-major, reduce-scattered to (S,), and
upcast-accumulated into the fp32 shard grad. both copies run at HBM
bandwidth against a comm that runs at link bandwidth. the alternative
(FSDP1-style flat sharding of the concatenated unit) is zero-copy but a
rank's shard straddles param boundaries, so per-param shapes, clip norm and
checkpoint metadata all get harder. not worth it.

**mixed precision.** master shards fp32, gather in bf16, reduce-scatter in
bf16, accumulate into fp32 grad shards. every param is gathered in
param_dtype, norms included: the "bf16 norm gains freeze under adamw" bug in
Trainer._init_weights was about bf16 *master* params. here the optimizer
only ever sees the fp32 shard, so a bf16 compute copy is harmless. one dtype
per unit keeps it at one gather per unit.

**no dedicated comm stream (v0).** nccl already runs collectives on its own
internal stream. `async_op=True` plus a deferred `work.wait()`, which makes
the current stream wait on the collective, gives the overlap. prefetch is
just "launch unit i+1's gather before unit i computes".

**grad accumulation reduce-scatters every microbatch** into the fp32 shard
grad. `requires_sync` is accepted and ignored, so the trainer loop stays
shared with ddp.

**hooks must run eagerly.** `nn.Module.compile()` compiles `_call_impl`,
which includes the module's hooks, so dynamo would try to trace collectives
and `.data` swaps. the trainer compiles `block.forward` instead.

===========================================================================
lifecycle of one block unit, one microbatch
===========================================================================

    block forward-pre   unshard + wait_unshard, record order, prefetch next
    block forward-post  reshard; register a hook on the block's OUTPUT tensor
                        (fires when grad wrt output is ready = right before
                        this block's backward). skipped under no_grad.
    output grad hook    unshard + wait_unshard, prefetch previous
    post-accum-grad     per-param countdown; at 0 -> post_backward:
                        fold previous microbatch's reduce-scatter, chunk-cat
                        grads, p.grad = None, reshard, launch reduce-scatter
    FSDP.sync()         every unit: finalize_grads (fold + install p.grad)

root unit: unshard in the MODEL's forward-pre; it never reshards in forward.
under no_grad (eval), the model's forward-post reshards it instead, since no
backward will.

===========================================================================
what will break
===========================================================================

- **free by resizing storage, not by dropping references.** the unit keeps
  its unsharded views alive on purpose (stable objects across steps), so
  only `untyped_storage().resize_(0)` actually frees. getting this wrong
  still trains correctly and saves nothing. tests assert storage size.
- **p.grad must be None when backward starts.** autograd accumulates the
  unsharded grad into an existing sharded-shaped p.grad and dies (or worse).
  this relies on `optimizer.zero_grad()` with set_to_none=True, the default.
- **backward reads the param, not a snapshot.** autograd saves the leaf
  Parameter itself, so backward sees whatever `.data` is at that moment. the
  output grad hook must have re-gathered first. if not, you get a shape
  error in the backward. loud, but check it under compile too.
- **meta-device init** is v1. v0 inits the full model on every rank and
  slices. free each param's full tensor as its shard is built (unit by
  unit), or the transient peak is full model + all shards (notes: 5 B/param).
"""

from dataclasses import dataclass
from enum import Enum, auto

import torch
import torch.distributed as dist
import torch.nn as nn

from torchure.core.collective import MeshLike


@dataclass
class MixedPrecisionPolicy:
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype = torch.float32

    @staticmethod
    def fp32() -> "MixedPrecisionPolicy":
        return MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)


@dataclass
class _ParamInfo:
    """
    one param's place in its unit. `offset`/`numel` index the SHARDED flat
    buffer. the param's unsharded region is [R*offset, R*(offset+numel)),
    because unsharded_flat is per-param contiguous in the same order.
    """

    param: nn.Parameter
    fqn: str                       # for error messages
    full_shape: torch.Size
    shard_shape: torch.Size        # (full_shape[0] // R, *full_shape[1:])
    offset: int
    numel: int                     # sharded numel
    shard_view: torch.Tensor | None = None      # into shard_flat, master dtype
    unsharded_view: torch.Tensor | None = None  # into unsharded_flat, full_shape


class _State(Enum):
    SHARDED = auto()
    GATHERING = auto()   # all_gather launched, not yet waited / copied out
    UNSHARDED = auto()


class FSDPUnit:
    """
    one all-gather's worth of params.

    storage it owns:

      shard_flat        (S,)    master dtype, persistent. p.data views when sharded.
      grad_shard_flat   (S,)    master dtype, persistent. accumulates reduced
                                grads across microbatches; p.grad views after sync.
      unsharded_flat    (R*S,)  param dtype, allocated once, storage resized
                                to 0 whenever sharded. p.data views when unsharded.
      gather_buf        (R*S,)  param dtype, transient: gather -> copy-out -> drop.
      reduce_in/out             transient, held with the Work handle until waited.
    """

    def __init__(
        self,
        name: str,
        params: list[tuple[str, nn.Parameter]],
        mesh: MeshLike,
        shard_dim: str,
        replicate_dim: str | None,
        policy: MixedPrecisionPolicy,
        reshard_after_forward: bool = True,
    ):
        """
        `params` is (fqn, param) pairs, already deduplicated by identity and
        in the same order on every rank. FSDP._build_units does the dedup,
        because "every param in exactly one unit" is a cross-unit property.
        """
        self.name = name
        self.mesh = mesh
        self.shard_dim = shard_dim
        self.replicate_dim = replicate_dim
        self.policy = policy
        self.reshard_after_forward = reshard_after_forward
        self.shard_size = mesh.size(shard_dim)

        self.params: list[_ParamInfo] = []
        self.shard_numel = 0  # S

        self.shard_flat: torch.Tensor | None = None
        self.grad_shard_flat: torch.Tensor | None = None
        self.unsharded_flat: torch.Tensor | None = None

        self.state = _State.SHARDED
        self._gather_buf: torch.Tensor | None = None
        self._gather_work: dist.Work | None = None

        # post-accumulate-grad countdown. re-armed in post_backward
        self.pending = len(params)
        # (work, reduce_in, reduce_out) of the in-flight reduce-scatter.
        # reduce_in is held only to keep it alive until the wait
        self._reduce: tuple[dist.Work, torch.Tensor, torch.Tensor] | None = None
        # reduce-scatters folded into grad_shard_flat since the last sync.
        # 0 means the next fold overwrites instead of accumulating
        self.folds = 0

        self._build_shards(params)
        self._alloc_buffers()

    # --- construction -----------------------------------------------------

    def _build_shards(self, params: list[tuple[str, nn.Parameter]]) -> None:
        """
        fill self.params and allocate shard_flat (S,) in master dtype.

        every rank ran init_weights() on the full model with the same seed,
        so each rank slices out its own dim-0 chunk locally, with no comm
        needed:

            chunk = p.detach()[c*d0//R : (c+1)*d0//R]    c = mesh.coordinate(shard_dim)

        copy it into shard_flat[offset : offset+numel], then set
        `p.data = shard_view` right away so the full tensor is freed as you
        go.

        - assert `full_shape[0] % R == 0` and name the fqn. no padding in v0.
        - cast to master_dtype here. the model may arrive in any dtype.
        - `p.data = ...` keeps the tie for free (one object, two attributes).
          rebinding `module._parameters[name]` would silently break it.
        """
        offset = 0
        device = params[0][1].device

        for fqn, param in params:
            assert param.shape[0] % self.shard_size == 0, \
            f"{fqn}: dim 0 ({param.shape[0]}) not divisible by shard size {self.shard_size}"

            shard_shape = torch.Size((param.shape[0] // self.shard_size, *param.shape[1:]))
            numel = shard_shape.numel()
            
            self.params.append(_ParamInfo(
                param=param, fqn=fqn, full_shape=param.shape,
                shard_shape=shard_shape, offset=offset, numel=numel,
            ))
            offset += numel

        self.shard_numel = offset

        self.shard_flat = torch.empty(self.shard_numel, dtype=self.policy.master_dtype, device=device)
        pos = self.mesh.coordinate(self.shard_dim)

        for param_info in self.params:
            rows = param_info.shard_shape[0]
            chunk = param_info.param.detach()[pos * rows: (pos + 1) * rows]
            view = self.shard_flat[param_info.offset : param_info.offset + param_info.numel].view(param_info.shard_shape)
            view.copy_(chunk)
            param_info.shard_view = view
            param_info.param.data = view
            

    def _alloc_buffers(self) -> None:
        """
        allocate unsharded_flat (R*S,) in param dtype, build each info's
        unsharded_view = unsharded_flat[R*off : R*(off+n)].view(full_shape),
        then `unsharded_flat.untyped_storage().resize_(0)`. the views stay
        valid Python objects over 0-byte storage until the next unshard.

        allocate grad_shard_flat (S,) in master dtype. it's small (1/R).
        """
        raise NotImplementedError

    # --- forward ----------------------------------------------------------

    def unshard(self) -> None:
        """
        launch the gather and don't wait. no-op unless SHARDED (prefetch may
        already have launched it).

          staged = shard_flat.to(param_dtype)       # a real alloc when dtypes differ
          gather_buf = empty(R*S, param_dtype)
          _, work = all_gather(staged, mesh, shard_dim, async_op=True,
                               output_tensor=gather_buf)

        stash gather_buf and work, and set state = GATHERING. also resize
        unsharded_flat's storage back to R*S here, so the allocation happens
        on the compute stream before the copy-out writes to it.
        """
        raise NotImplementedError

    def wait_unshard(self) -> None:
        """
        GATHERING -> UNSHARDED. idempotent once UNSHARDED; asserts it isn't
        SHARDED (someone forgot to unshard).

          work.wait()
          torch.split_with_sizes_copy(gather_buf.view(R, S), [info.numel ...],
                                      dim=1, out=[v.view(R, n) ...])
          drop gather_buf
          p.data = info.unsharded_view   for every info

        `v.view(R, n)` is legal because v is contiguous and d0 % R == 0: row
        r of that view is exactly p[r]'s chunk.
        """
        raise NotImplementedError

    def reshard(self) -> None:
        """
        p.data = shard_view for every param, then
        unsharded_flat.untyped_storage().resize_(0). state = SHARDED.

        callers decide whether to call it (reshard_after_forward is a policy
        for the hooks, not a guard in here). post_backward and eval need to
        reshard the root unconditionally.
        """
        raise NotImplementedError

    # --- backward ---------------------------------------------------------

    def on_grad_ready(self, param: nn.Parameter) -> None:
        """
        post-accumulate-grad hook. decrement pending, and at 0 call
        post_backward.

        the tied embedding is one leaf, so the engine sums both contributions
        before its AccumulateGrad runs and this fires once. count unique
        params, not uses.
        """
        raise NotImplementedError

    def post_backward(self) -> None:
        """
        all of this unit's grads are complete, in unsharded p.grad.

          1. _fold_reduced()   the previous microbatch's reduce-scatter, if any.
             it must land before its output buffer is overwritten.
          2. reduce_in = cat([p.grad.view(R, -1) for params], dim=1)
                         .to(reduce_dtype)                        # (R, S) rank-major
             (or torch._chunk_cat(grads, dim=0, num_chunks=R))
          3. p.grad = None for every param   # next microbatch's backward needs None
          4. reshard()                       # root included
          5. reduce_out = empty(S); work = reduce_scatter(reduce_in.view(-1), ...,
                                   "avg", async_op=True, output_tensor=reduce_out)
             self._reduce = (work, reduce_in, reduce_out)
          6. pending = len(self.params)

        assert each grad has full_shape before the cat. a sharded-shaped
        grad here means p.grad wasn't None when backward started (zero_grad
        with set_to_none=False).

        is it safe to reshard at step 4? by the time every param's
        AccumulateGrad has run, every node that read these weights has been
        enqueued (MmBackward produces grad_input and grad_weight together),
        and the storage free is ordered on the same stream.
        """
        raise NotImplementedError

    def _fold_reduced(self) -> None:
        """
        wait the in-flight reduce-scatter (no-op if none). if replicate_dim
        is set, all_reduce(reduce_out, mesh, replicate_dim, "avg"); that's
        hsdp, and it's synchronous in v0. then:

          folds == 0:  grad_shard_flat.copy_(reduce_out)
          else:        grad_shard_flat.add_(reduce_out)
          folds += 1;  self._reduce = None
        """
        raise NotImplementedError

    def finalize_grads(self) -> None:
        """
        called from FSDP.sync(). run _fold_reduced(), then set
        `p.grad = grad_shard_view` (grad_shard_flat[off:off+n].view(shard_shape)),
        then folds = 0 so the next step's first fold overwrites.

        after this, p.data and p.grad are both sharded master dtype, and
        clip_grad_norm with shard_dims=("dp_shard",) is correct. the views
        get dropped by optimizer.zero_grad(), which is fine: grad_shard_flat
        persists and gets overwritten.
        """
        raise NotImplementedError


class FSDP:
    """
    model-level wrapper. same surface as DDP (`requires_sync`, `sync()`,
    `shard_dims`), so Trainer.train_step doesn't branch.

    `dp_shard == 1` is a TOTAL no-op, the way DDP is at dp == 1.
    """

    def __init__(
        self,
        model: nn.Module,
        mesh: MeshLike,
        shard_dim: str = "dp_shard",
        replicate_dim: str | None = None,
        policy: MixedPrecisionPolicy | None = None,
        prefetch_depth: int = 1,
    ):
        """
        `replicate_dim` is hsdp: reduce-scatter along shard_dim, then
        all_reduce the sharded grads along replicate_dim.

        `prefetch_depth` is how many units ahead to launch gathers. each one
        costs a unit's worth of gather_buf + unsharded_flat while live.
        notes/FSDP2_H100.md: 1 captures most of the win, and 2 adds ~1%.
        """
        self.mesh = mesh
        self.shard_dim = shard_dim
        self.replicate_dim = replicate_dim
        self.policy = policy or MixedPrecisionPolicy()
        self.prefetch_depth = prefetch_depth
        self.shard_size = mesh.size(shard_dim)
        self.requires_sync = True  # accepted and ignored; see module docstring

        self.units: list[FSDPUnit] = []
        self._forward_order: list[FSDPUnit] = []
        self._order_recorded = False

        if self.shard_size == 1:
            # without this, dp_shard=1 x dp_replicate=N would sync no grads
            # at all, and replicas would drift silently
            assert replicate_dim is None or mesh.size(replicate_dim) == 1, (
                f"fsdp with {shard_dim}=1 and {replicate_dim}={mesh.size(replicate_dim)} "
                "is pure replication; use parallelism='ddp'"
            )
            return

        self.units = self._build_units(model)
        self._register_hooks(model)

    @property
    def shard_dims(self) -> tuple[str, ...]:
        """
        mesh axes param/grad state is sharded over. feeds clip_grad_norm, and
        doubles as "is model state sharded" for the trainer's checkpoint
        guard. that's why dp_shard == 1 answers ().
        """
        return () if self.shard_size == 1 else (self.shard_dim,)

    # --- setup ------------------------------------------------------------

    def _build_units(self, model: nn.Module) -> list[FSDPUnit]:
        """
        """
        assert model.blocks and len(model.blocks) > 0, "blocks must already be constructed"

        seen: dict[int, str] = {}
        block_params: list[list[tuple[str, nn.Parameter]]] = []

        for i, block in enumerate(model.blocks):
            name = f"blocks.{i}"
            params = []

            for fqn, param in block.named_parameters(prefix=name):
                assert id(param) not in seen, (
                    f"{fqn} already claimed by {seen[id(param)]}; a param in two "
                    "units is gathered and reduce-scattered twice"
                )
                seen[id(param)] = name
                params.append((fqn, param))

            block_params.append(params)

        root_params = []
        for fqn, param in model.named_parameters():
            if id(param) not in seen:
                seen[id(param)] = "root"
                root_params.append((fqn, param))

        missing = {id(p) for p in model.parameters()} - set(seen)
        assert not missing, (
            f"{len(missing)} param in no unit, require sync"
        )

        def make(name, params, reshard_after_forward=True):
            return FSDPUnit(
                name, params, self.mesh, self.shard_dim, self.replicate_dim,
                self.policy, reshard_after_forward=reshard_after_forward,
            )

        units = [make("root", root_params, reshard_after_forward=False)]
        units += [make(f"blocks.{i}", p) for i, p in enumerate(block_params)]
        return units


    def _register_hooks(self, model: nn.Module) -> None:
        """
        registered once, never removed.

          model.register_forward_pre_hook   -> _pre_forward(root)
          model.register_forward_hook       -> reshard root iff not grad enabled
          block.register_forward_pre_hook   -> _pre_forward(unit)
          block.register_forward_hook       -> _post_forward(unit, output)
          p.register_post_accumulate_grad_hook -> unit.on_grad_ready  (all units)

        no register_full_backward_pre_hook: it wraps inputs/outputs in
        BackwardHookFunction nodes, doesn't fire when the inputs don't need
        grad, and is unsupported inside compiled regions. a plain tensor hook
        on the block's output does the same job.
        """
        raise NotImplementedError

    # --- hook bodies ------------------------------------------------------

    def _pre_forward(self, unit: FSDPUnit) -> None:
        """unit.unshard(); unit.wait_unshard(); _record_order(unit); _prefetch(unit, backward=False)."""
        raise NotImplementedError

    def _post_forward(self, unit: FSDPUnit, output: torch.Tensor) -> None:
        """
        if unit.reshard_after_forward: unit.reshard().
        if output.requires_grad: output.register_hook(lambda g: self._pre_backward(unit))
        (return None from the tensor hook, so the grad passes through unchanged).
        """
        raise NotImplementedError

    def _pre_backward(self, unit: FSDPUnit) -> None:
        """unit.unshard(); unit.wait_unshard(); _prefetch(unit, backward=True)."""
        raise NotImplementedError

    # --- prefetch ---------------------------------------------------------

    def _record_order(self, unit: FSDPUnit) -> None:
        """
        append on the first forward only. use execution order, not
        construction order. set _order_recorded in sync() after the first
        step, so a second forward can't double-append.
        """
        raise NotImplementedError

    def _prefetch(self, unit: FSDPUnit, backward: bool) -> None:
        """
        call unshard() (launch only, no wait) on the next prefetch_depth
        units after `unit`: forward order going forward, reversed going
        backward. no-op until the order is recorded.
        """
        raise NotImplementedError

    # --- step boundary ----------------------------------------------------

    def sync(self) -> None:
        """
        between the last microbatch's backward and clip_grad_norm.

        - assert every unit's pending == len(params), i.e. its post_backward
          fired and re-armed. ddp-style message: frozen param, or a skipped
          block.
        - finalize_grads() on every unit.
        - assert every unit is SHARDED, root included. a unit left gathered
          trains fine and saves no memory.
        """
        if self.shard_size == 1:
            return
        raise NotImplementedError

    def state_dict(self) -> dict:
        """
        phase 3. must raise until sharded checkpointing exists: rank 0's
        model.state_dict() is one shard, and would be written as if it were
        the model.
        """
        raise NotImplementedError(
            "sharded checkpointing is phase 3; an fsdp run cannot checkpoint yet"
        )


if __name__ == "__main__":
    print("fsdp2: see tests/fsdp2.py")
