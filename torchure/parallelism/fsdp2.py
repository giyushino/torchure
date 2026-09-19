"""
fsdp2: per-parameter sharding along the mesh's `dp_shard` axis.

each param's master copy is a fp32 Shard(0) slice. before a unit runs, its
shards are all-gathered into a full-size compute-dtype buffer; after it runs
that buffer's storage is released. grads are reduce-scattered back down to
sharded fp32, so params, grads AND optimizer state all scale as 1/dp_shard --
the last of those being the actual memory win, and it happens for free only
because the optimizer is built from `model.parameters()` AFTER sharding.

## the one structural idea

each logical param keeps exactly ONE nn.Parameter object for the life of the
model, and `param.data` is swapped between two storages:

    resharded:  param.data -> the fp32 shard view      (what the optimizer sees)
    unsharded:  param.data -> the compute-dtype view    (what forward sees)

that single decision buys three things that are otherwise fiddly:

- the tied embedding works for free. `lm_head.weight is token_emb.weight` is
  one Parameter reachable from two modules, so swapping `.data` updates both
  views at once. rebinding `module._parameters[name]` instead silently breaks
  the tie and you get two independently gathered copies of the biggest weight
  in the model with grads flowing to only one.
- grad hooks register once. `register_post_accumulate_grad_hook` lives on the
  Parameter, so it survives every swap -- the same mechanism ddp uses, with one
  counter per unit instead of per bucket.
- the optimizer never sees a DTensor. params are plain tensors at runtime, so
  fused AdamW, the `torch._foreach_*` calls in grad_helper and every `p.grad`
  access work untouched.

## units

a unit is the granularity of one gather. unit 0 is the root -- token_emb, the
final norm, lm_head -- and each transformer block is its own unit. the tied
weight FORCES embed and head into one unit; splitting them would double-gather
one storage and corrupt the grad flow.

the root is `reshard_after_forward=False` on purpose: `ARObjective.compute_loss`
calls `model(..., return_final_hidden=True)` and then hands `model.lm_head.weight`
to the fused linear+CE, so the head weight is consumed AFTER the model's forward
has returned. any reshard on a forward-post hook would hand the objective a
freed weight.

## flat buffers, not one collective per param

each unit owns two flat buffers and every param is a VIEW into them. this is
the thing to get right up front, because retrofitting it means changing who
owns the storage:

    shard_flat      fp32, persistent, 1/dp_shard of the unit's params. the
                    master weights; never freed. optimizer state is allocated
                    against views of this.
    unsharded_flat  compute-dtype, allocated ONCE and reused every step so the
                    param views stay the same tensor objects across steps
                    (stable for dynamo guards -- this is what
                    collective.all_gather's `output_tensor=` exists for).
                    storage resized to 0 between uses.

one gather per unit rather than ~9 per block is worth real time on this box:
measured 4-rank bus bandwidth is 66.5 GB/s at 16 MiB messages against 78.1 GB/s
at 256 MiB, and collectives are ~53% of a 12B step (see notes/FSDP2_H100.md).

the gathered layout is rank-major, so reassembling per-param views out of one
buffer needs a copy-out pass (`split_with_sizes_copy`) or a per-param offset
scheme. that is the fiddly part and it is why `_ParamInfo` carries offsets.

## lifecycle per unit

    forward-pre    unshard (all_gather, async) -> wait -> prefetch the NEXT
    forward-post   reshard: point params at shards, resize buffer to 0
    backward-pre   unshard -> wait -> prepare_grads -> prefetch the PREVIOUS
    post-backward  reduce_scatter grads -> sharded, free grad_flat, reshard
    sync()         wait every reduce-scatter, publish fp32 `.grad`

resharding is a storage RESIZE, not a dropped python reference. saved
activations from the block's backward hold a reference to the weight view, so
refcounting alone would never release it -- the run would be numerically
perfect and save no memory at all. that is why tests/fsdp2.py asserts storage
size directly rather than trusting a peak-memory number.

## grad accumulation

grads are reduce-scattered EVERY microbatch and accumulated in the sharded
fp32 grad, rather than accumulated unsharded and reduced once. that is one
extra collective per microbatch in exchange for never holding a full unsharded
grad per unit, which is the whole reason to be here. "avg" is linear, so
avg-then-sum over microbatches equals ddp's sum-then-avg.

`requires_sync` exists only so the trainer loop stays one idiom across ddp and
fsdp; fsdp accepts it and ignores it.

NOTE the cost: every microbatch pays a full all-gather of every weight in the
forward and again in the backward. ddp pays one all-reduce per STEP regardless
of microbatch count. so under fsdp, prefer micro_batch_size over accumulation
steps far more aggressively than under ddp -- 4B goes 30.1k -> 44.5k tok/s
going from micro_bs 1 to 4.

## traps that fail silently

- **hooks must be `@torch.compiler.disable`d.** `Module.compile()` compiles
  `_call_impl`, NOT just `forward`, so dynamo traces registered hooks into the
  block's graph -- and these hooks resize storages. inductor then dies with
  "LoweringException: TODO: dynamic shapes" on `resize_storage_bytes_` with a
  symbolic size (symbolic because dynamo watched param.data change shape
  between the sharded and unsharded forms). disabling makes dynamo graph-break
  at the hook, which costs nothing: the break lands at the block boundary,
  where the compiled region already starts and ends.
- **broadcast before slicing, or seed identically.** nothing in this repo seeds
  torch by default. if ranks disagree on the full model, each keeps a slice of
  a DIFFERENT model and trains a silently corrupt average.
- **clear `.grad` before a backward writes an unsharded-shaped one into it.**
  after sync() publishes, `param.grad` holds the SHARDED grad; the next
  backward tries to accumulate an unsharded grad into it and hits a shape
  error -- or worse, doesn't.
- **clip_grad_norm must reduce over dp_shard.** leaving `shard_dims=()` gives a
  silently too-small norm (each rank measures 1/dp_shard of the squares), which
  under-clips rather than crashing.

## targets (4xH100 NVL, from notes/FSDP2_H100.md)

the reference implementation this replaces measured, so there is a number to
beat rather than a vibe:

    0.6B, micro_bs=4   186,622 tok/s   28.75 GiB peak
    8B,   micro_bs=2    23,003 tok/s   69.71 GiB peak

`torchure/parallelism/fsdp2_reference.py` is that implementation, frozen. it is
kept importable ONLY so the parity tests have something to diff against; delete
it once this file passes them.

## deliberately out of scope for v0

- uneven / padded shards (assert divisibility; note where padding would go).
- meta-device init -- v0 inits the full model and slices, which costs a
  transient 5 bytes/param and is a hard wall at ~18.6B params on this box.
- sharded checkpointing (phase 3) -- `state_dict` raises.
- activation checkpointing. it is the highest-leverage thing NOT here: it caps
  micro_bs, which caps MFU (29% at 12B) and the size ceiling (14B). it needs
  `reshard_after_forward=False` on checkpointed units, which is why that is a
  real per-unit flag rather than a constant.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from torchure.core.collective import MeshLike


def free_storage(t: torch.Tensor) -> None:
    """
    release a buffer's memory while keeping the tensor object and its shape.

    a dropped python reference is NOT enough: activations saved for backward
    hold the weight views alive, so the allocator never gets the block back.
    resizing the storage is what actually frees it, and it is why the same
    tensor object (and therefore the same dynamo guard) survives every step.
    """
    t.untyped_storage().resize_(0)


def alloc_storage(t: torch.Tensor) -> None:
    """inverse of free_storage. idempotent."""
    nbytes = t.numel() * t.element_size()
    if t.untyped_storage().size() != nbytes:
        t.untyped_storage().resize_(nbytes)


@dataclass
class MixedPrecisionPolicy:
    """
    what dtype each leg of the step runs in.

    a dataclass so the parity tests can run the whole thing in fp32
    (`MixedPrecisionPolicy.fp32()`) and get near-exact agreement with a
    single-process reference -- gloo's bf16 support is patchy, and a parity test
    that loosens its tolerance to accommodate the comm dtype isn't testing what
    it claims to.

    `reduce_dtype` defaults to bf16 rather than fp32 on measured evidence: at
    11.8B, bf16 reduction is 12% FASTER and 5 GiB smaller than fp32, because
    fp32 reduction upcasts each unit's full unsharded grad before the
    reduce-scatter. at dp_shard=4 the accumulation error is 4 terms.
    """

    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype = torch.float32
    # module types whose params are never downcast. RMSNorm gains are fp32
    # parameters on purpose -- see Trainer._init_weights for what happens when
    # they aren't (bf16 spacing at 1.0 is ~0.0039 against a ~3e-4 update, so
    # every step rounds back to the same value and the gains freeze at init).
    keep_master: tuple[type[nn.Module], ...] = (nn.RMSNorm, nn.LayerNorm)

    @staticmethod
    def fp32() -> "MixedPrecisionPolicy":
        """no-op policy: everything fp32. what the cpu parity tests run."""
        return MixedPrecisionPolicy(
            param_dtype=torch.float32, reduce_dtype=torch.float32
        )


@dataclass
class _ParamInfo:
    """where one param lives inside its unit's flat buffers."""

    param: nn.Parameter
    module: nn.Module
    name: str
    full_shape: torch.Size
    offset: int          # element offset into the SHARDED flat buffer
    numel: int           # sharded numel


class FSDPUnit:
    """
    one all-gather's worth of parameters: a transformer block, or the root
    (token_emb + lm_head + final norm).

    owns the flat buffers described in the module docstring and swaps
    `.data` / `.grad` between views of them.
    """

    def __init__(
        self,
        name: str,
        modules: list[nn.Module],
        mesh: MeshLike,
        shard_dim: str,
        policy: MixedPrecisionPolicy,
        reshard_after_forward: bool = True,
    ):
        """
        collects every param owned by `modules` (dedup by identity -- the tied
        embedding appears under two modules and must be sharded ONCE), builds
        shard_flat by slicing dim 0 of each full param, points each param's
        `.data` at its shard view, and drops the full tensors.

        `reshard_after_forward=False` is the root unit: it stays gathered from
        its forward-pre until its post-backward, because the objective consumes
        lm_head.weight after the model's forward returns. it is also what
        activation checkpointing will need on every checkpointed unit, so keep
        it a real per-unit policy rather than a constant.
        """
        self.name = name
        self.modules = modules
        self.mesh = mesh
        self.shard_dim = shard_dim
        self.policy = policy
        self.reshard_after_forward = reshard_after_forward

        self.shard_size = mesh.size(shard_dim)
        self.params: list[_ParamInfo] = []
        self.is_unsharded = False
        self.pending = 0

        self.shard_flat: torch.Tensor | None = None
        self.unsharded_flat: torch.Tensor | None = None
        self.grad_flat: torch.Tensor | None = None
        self._work = None

        raise NotImplementedError("FSDPUnit.__init__")

    # --- construction -----------------------------------------------------

    def _collect_params(self, modules: list[nn.Module]) -> list[nn.Parameter]:
        """
        unique params under `modules`, in a deterministic order (module
        traversal order, identical on every rank -- rank-dependent ordering
        would put the collectives in different orders and deadlock nccl).

        dedup by `id`, NOT by name: `lm_head.weight is token_emb.weight` is one
        storage under two names and must appear exactly once. note
        `named_parameters()` already dedups, so counting it will not catch a
        broken tie -- the `is` check is what does.
        """
        raise NotImplementedError

    def _build_shards(self) -> None:
        """
        allocate shard_flat and fill it.

        v0 shard construction: every rank has already run `init_weights()` on
        the full model and been broadcast into agreement, so slicing dim 0
        locally is bit-identical across ranks with zero comm, and init_weights'
        residual scaling stays a whole-tensor op.

        the slice must be `.clone()`d into the flat buffer, not kept as a
        narrow() view -- a view keeps the whole unsharded tensor alive and saves
        exactly nothing.

        assert `full_shape[0] % shard_size == 0` per param and NAME the param in
        the message. no padding in v0.
        """
        raise NotImplementedError

    def _alloc_buffers(self) -> None:
        """
        allocate unsharded_flat (param_dtype, full unit numel) once, then
        immediately resize its storage to 0. build the param views into it now,
        so they are stable objects for the life of the run.
        """
        raise NotImplementedError

    # --- forward ----------------------------------------------------------

    def unshard(self, async_op: bool = True) -> None:
        """
        resize unsharded_flat's storage back up, cast shard_flat into the
        param_dtype staging shard, and all_gather into unsharded_flat.

        does NOT wait -- that's `wait_unshard`. the split is the whole point:
        prefetch calls this for unit i+1 while unit i computes.

        the cast needs its own staging buffer (a param_dtype copy of the local
        shard). it is per-unit and reused, and it must stay REFERENCED until
        the collective completes -- dropping it here is a use-after-free that
        only shows up under load.

        no-op if already unsharded (prefetch may have got here first).
        """
        raise NotImplementedError

    def wait_unshard(self) -> None:
        """
        wait the gather, then point every param's `.data` at its unsharded
        view. idempotent -- both the forward-pre hook and a prefetch can reach
        it.
        """
        raise NotImplementedError

    def reshard(self) -> None:
        """
        point every param's `.data` back at its fp32 shard view, then
        `free_storage(unsharded_flat)`.

        order matters: repoint first, free second. no-op when
        `reshard_after_forward` is False.
        """
        raise NotImplementedError

    # --- backward ---------------------------------------------------------

    def prepare_grads(self) -> None:
        """
        allocate/zero grad_flat and set each `param.grad` to its unsharded
        view, so autograd accumulates directly into the flat buffer -- one
        reduce-scatter input, no per-param grad allocations, and no foreach copy
        the way ddp's buckets need one.

        called from the backward-pre hook, after wait_unshard.

        this is also where a leftover SHARDED `.grad` from the previous sync()
        has to be dealt with: the trainer's zero_grad normally cleared it, but
        if it didn't, take the value into the pending sharded grad rather than
        dropping it, or a microbatch's contribution vanishes silently.

        the tied embedding accumulates two contributions into the same view.
        that is correct and is what the tied-embedding test pins.
        """
        raise NotImplementedError

    def on_grad_ready(self, param: nn.Parameter) -> None:
        """
        post-accumulate-grad hook, one countdown per unit (ddp's `_Bucket`
        machinery, coarser). when it hits zero the unit's grads are complete ->
        `post_backward()`.

        re-arm `pending` on launch, like ddp does, so grad accumulation works
        without re-registering anything.
        """
        raise NotImplementedError

    def post_backward(self) -> None:
        """
        reduce_scatter(grad_flat, mesh, shard_dim, "avg", async_op=True) into a
        per-unit sharded output buffer, then reshard the params and free
        grad_flat.

        the "avg" is what makes N ranks x bs=1 equal 1 process x bs=N, same as
        ddp. collective.reduce_scatter handles async "avg" on gloo by
        pre-dividing out-of-place on purpose -- under fsdp the input IS the
        unsharded grad buffer, and scaling it in place would corrupt every
        later read -- so the cpu tests exercise the same overlapped path the gpu
        does.

        stash the work handle; `finalize_grads` does the upcast-accumulate.
        """
        raise NotImplementedError

    def finalize_grads(self) -> None:
        """
        wait the reduce-scatter, then accumulate the sharded result into the
        fp32 `param.grad` (`+=`, not `=` -- accumulation reduce-scatters every
        microbatch into the same fp32 shard).

        under hsdp this is also where the second collective goes: all_reduce
        the sharded grads along replicate_dim after the reduce-scatter along
        shard_dim.

        after this, `param.data` and `param.grad` are both sharded fp32 and the
        optimizer can step.
        """
        raise NotImplementedError


class FSDP:
    """
    the model-level wrapper. mirrors DDP's shape so the trainer's
    `_parallelize` doesn't grow a second idiom -- the trainer talks to it
    through `requires_sync`, `sync()` and `shard_dims` only:

        self.parallel = FSDP(model, mesh, shard_dim="dp_shard")
        ...
        loss.backward()
        self.parallel.sync()
        clip_grad_norm(model.parameters(), mesh, max_norm,
                       shard_dims=self.parallel.shard_dims)
        optimizer.step()

    `dp_shard == 1` must be a TOTAL no-op: no units, no hooks, no buffers, no
    groups touched, exactly the way DDP is at `dp == 1`. tests/fsdp2.py asserts
    it, and the trainer's checkpoint guard depends on it.
    """

    def __init__(
        self,
        model: nn.Module,
        mesh: MeshLike,
        shard_dim: str = "dp_shard",
        replicate_dim: str | None = None,
        policy: MixedPrecisionPolicy | None = None,
        prefetch_depth: int = 2,
    ):
        """
        `replicate_dim` is hsdp: after the reduce-scatter along shard_dim,
        all_reduce the sharded grads along replicate_dim. two collectives, one
        ordering, no new state. None disables it, and a replicate_dim of size 1
        must degenerate to the same thing.

        `prefetch_depth` is how many units ahead to launch gathers for. measured
        at 7.6B: depth 0 costs 14% (20,191 tok/s), 1 gets 23,032, 2 gets 23,343,
        3 is past the knee. it does NOT change peak memory at these sizes
        (69.71 GiB at every depth) because the peak is set by activations, not
        gather buffers -- so there is no reason to lower it.
        """
        self.model = model
        self.mesh = mesh
        self.shard_dim = shard_dim
        self.replicate_dim = replicate_dim
        self.policy = policy or MixedPrecisionPolicy()
        self.prefetch_depth = prefetch_depth
        self.shard_size = mesh.size(shard_dim)
        self.replicate_size = (
            mesh.size(replicate_dim) if replicate_dim is not None else 1
        )
        # accepted and ignored -- see the module docstring. it exists so the
        # trainer's accumulation loop is identical for ddp and fsdp.
        self.requires_sync = True

        self.units: list[FSDPUnit] = []
        self._forward_order: list[FSDPUnit] = []
        self._order_recorded = False

        if self.shard_size == 1:
            # total no-op: not one hook registered, not one buffer allocated
            return

        raise NotImplementedError("FSDP.__init__")

    @property
    def shard_dims(self) -> tuple[str, ...]:
        """
        mesh axes across which param/grad state is sharded. the trainer hands
        this straight to clip_grad_norm, so the strategy owns the answer instead
        of the training loop branching on which one is live.

        it doubles as "is model state sharded" for the trainer's checkpoint
        guard, which is why `dp_shard == 1` must answer `()` rather than the
        axis name: at size 1 nothing is split and rank 0's tensors really are
        the model.
        """
        return () if self.shard_size == 1 else (self.shard_dim,)

    # --- setup ------------------------------------------------------------

    def _replicate(self, model: nn.Module) -> None:
        """
        make every rank's full weights identical BEFORE slicing.

        load-bearing, not defensive: v0 shards by "init the full model on every
        rank, keep your own slice", which is only correct if the ranks agreed on
        the full model. broadcasting rather than seeding also stays correct if
        init order ever changes.
        """
        raise NotImplementedError

    def _build_units(self, model: nn.Module) -> list[FSDPUnit]:
        """
        unit 0 = root (token_emb + lm_head + final norm,
        reshard_after_forward=False), then one unit per entry in `model.blocks`.

        `model.blocks` is the contract (it is the unit of compile, fsdp
        wrapping, activation checkpointing and pp splitting), so assert it
        exists rather than walking the module tree heuristically.

        assert the tie -- `model.lm_head.weight is model.token_emb.weight` --
        instead of assuming it.

        every param must land in exactly one unit. check that the union covers
        `model.parameters()` and that no param is claimed twice; a param that
        falls through gets no gather and no reduce-scatter, and drifts between
        ranks silently.
        """
        raise NotImplementedError

    def _register_hooks(self) -> None:
        """
        per unit:
          module.register_forward_pre_hook        -> unshard + wait + prefetch
          module.register_forward_hook            -> reshard (unless root)
          module.register_full_backward_pre_hook  -> unshard + wait +
                                                     prepare_grads + prefetch
        per param:
          param.register_post_accumulate_grad_hook -> unit.on_grad_ready

        EVERY hook body must be `@torch.compiler.disable`d -- see the module
        docstring's traps section. this is the one that costs an afternoon.

        registered once at init and never removed, like ddp. one FSDP instance
        per model.

        for the root (token_emb, lm_head, norm) the forward hooks go on the
        modules that BOUND the unit's use -- unshard on token_emb's pre-forward,
        and no reshard hook at all, since it is released in post_backward.
        """
        raise NotImplementedError

    # --- prefetch ---------------------------------------------------------

    def _record_order(self, unit: FSDPUnit) -> None:
        """
        append to `_forward_order` on the first forward only. EXECUTION order,
        not construction order -- they agree for qwen3 today and won't for the
        first model that reorders or reuses blocks.
        """
        raise NotImplementedError

    def _prefetch(self, unit: FSDPUnit, backward: bool) -> None:
        """
        launch `prefetch_depth` units' gathers ahead of `unit`: forward order
        going forward, its exact reverse going backward.

        no-op until the order is recorded (the first forward is strictly serial
        comm->compute, and that's fine). a mismatch degrades silently into no
        prefetching at all rather than into a wrong answer, which is why
        tests/fsdp2.py checks the recorded order directly instead of inferring
        it from a throughput number.
        """
        raise NotImplementedError

    # --- step boundary ----------------------------------------------------

    def sync(self) -> None:
        """
        called between `loss.backward()` and `clip_grad_norm`. waits every
        outstanding reduce-scatter and runs each unit's `finalize_grads`.

        assert every unit fired, with ddp's error-message discipline -- a unit
        that never completed means a param got no grad (frozen), or a hook
        didn't fire because a block was skipped, and the resulting grads are
        wrong rather than absent.

        also assert every unit is resharded on the way out, root included:
        leaving one gathered is the failure mode where fsdp trains perfectly
        and saves no memory.
        """
        if self.shard_size == 1:
            return
        raise NotImplementedError

    def state_dict(self) -> dict:
        """
        phase 3. until sharded checkpointing exists this MUST raise: rank 0
        writing `model.state_dict()` under fsdp writes one rank's shard as if it
        were the model, and it is a corrupt file that only announces itself at
        resume. the trainer refuses to start an fsdp run with checkpointing
        enabled rather than relying on this firing at save time.
        """
        raise NotImplementedError(
            "sharded checkpointing is phase 3; an fsdp run cannot checkpoint yet"
        )

    # --- introspection (used by tests/fsdp2.py) ---------------------------

    def sharded_numel(self) -> int:
        """total params this rank actually owns. the memory exit criterion."""
        return sum(info.numel for unit in self.units for info in unit.params)

    def unsharded_bytes_live(self) -> int:
        """
        bytes currently held by gathered buffers, across all units. the
        storage-actually-freed invariant reads this BETWEEN units during a real
        forward -- after the whole step it is uninformative.
        """
        return sum(
            unit.unsharded_flat.untyped_storage().size()
            for unit in self.units
            if unit.unsharded_flat is not None
        )


if __name__ == "__main__":
    print("fsdp2: see tests/fsdp2.py")
