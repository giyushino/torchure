"""
fsdp2: per-parameter sharding of params, grads and optimizer state along the
mesh's `dp_shard` axis, with the unsharded weights materialized only for the
window in which they're used.

ddp holds a full fp32 replica of params + grads + adam state on every rank and
all-reduces the grads. fsdp holds 1/dp_shard of each, all-gathers a unit's
weights just before it runs, throws them away right after, and replaces the
all-reduce with a reduce-scatter (half the bytes). on this hardware that is a
*throughput* lever as much as a memory one -- DDP.md measured 442 ms/step of
all-reduce against a ~350 ms backward window at 8 gpus, so 156 ms/step is
exposed no matter how it's bucketed.

full checklist, decisions and exit criteria: notes/FSDP2_SCOPE.md.
tests: tests/fsdp2.py (gloo/cpu, mp.spawn) -- correctness first, then the
invisible invariants (storage actually freed, exactly one gather per unit).

===========================================================================
the decisions, made
===========================================================================

**unit = one transformer block.** plus one "root" unit holding
(token_emb + lm_head + final norm). the tie (`lm_head.weight is
token_emb.weight`) forces those two into the same unit -- sharding them
separately double-gathers one storage and corrupts grad flow. assert the tie,
don't assume it.

**per-param logical Shard(0), one flat storage per unit.** these aren't
exclusive and the scope doc's question answers itself: shard each param
logically on dim 0 (clean checkpoint metadata, clean clip_grad_norm), but back
every shard in a unit with ONE flat fp32 buffer and hand each param a view.
that makes unshard exactly one all_gather per unit instead of one per param --
the same lesson ddp's buckets already taught on this box.

**params stay plain nn.Parameter; DTensor is edge metadata.** one Parameter
object per param for the whole run, whose `.data` is swapped:

    resharded:  p.data -> fp32 shard view into unit.shard_flat     (optimizer)
    unsharded:  p.data -> bf16 view into unit.unsharded_flat       (forward)

so `model.parameters()` keeps returning the same objects, fused adamw allocates
its state against the *sharded* fp32 shapes (that's the memory win, and it is
free only because the trainer builds the optimizer after _parallelize), and
nothing in the hot path unwraps a subclass.

**mixed precision.** sharded master params fp32. gathered params bf16 -- the
gather is the biggest collective in the step and doing it in bf16 halves it.
grads arrive bf16. the reduce-scatter runs in **bf16** ("avg"), and its output
is upcast into the fp32 sharded `.grad`: fp32 comm would give back the 2x we
just bought, and bf16 accumulation error across dp_shard ranks is bounded by
the tree depth, not by dp_shard. keep norm weights fp32 through all of it --
the "bf16 params freeze under adamw" lesson in Trainer._init_weights applies to
anything this path might quietly downcast. autocast stays on: its param casting
becomes redundant once params arrive bf16, its activation policy does not.

**grad accumulation reduce-scatters every microbatch** into the sharded fp32
`.grad`, rather than holding a full unsharded grad per unit across microbatches.
one extra collective per microbatch, sharded memory -- which is the entire
reason fsdp is being added. consequence worth stating: `requires_sync` (ddp's
no_sync switch) is a **no-op here**, kept only so Trainer.train_step doesn't
grow a second idiom.

**re-unshard in backward via module full-backward-pre-hooks**, not an autograd
function on the unit's output. hooks are simpler and sit outside the compiled
region by construction (`_compile` compiles `block.forward`). the thing to
watch is stage 3: an autograd function is what survives dynamo reordering the
graph, so if the compile composition test goes bad, that's the escape hatch --
not a bucket-size tweak.

**checkpointing is guarded, not fixed.** sharded save/load is phase 3. until
then `Checkpointer.save_model` would write one rank's shard as if it were the
model. FSDP.state_dict raises; the trainer must refuse to run fsdp with
checkpointing enabled rather than write a corrupt file.

===========================================================================
the lifecycle of one unit, one step
===========================================================================

    forward pre-hook   unshard: all_gather(shard_flat -> unsharded_flat) in
                       bf16 on the comm stream; default stream waits on the
                       event; p.data -> views. then kick off unit i+1's gather
                       (prefetch) so it flies while this unit computes.
    forward post-hook  reshard: p.data -> fp32 shard views, and
                       unsharded_flat.untyped_storage().resize_(0).
                       ROOT UNIT EXCEPTED -- see below.
    backward pre-hook  unshard again (storage().resize_(numel) + all_gather),
                       and prefetch unit i-1. also point each p.grad at a view
                       of a zeroed unsharded bf16 grad_flat, so autograd
                       accumulates straight into the flat buffer instead of
                       allocating a grad per param.
    post-accum-grad    one counter per unit (ddp's machinery, one counter per
                       unit rather than per bucket). when it hits zero:
                       reduce_scatter(grad_flat, "avg", async) -> sharded bf16,
                       free grad_flat and unsharded_flat, and on the wait
                       upcast-accumulate into the fp32 p.grad.
    sync()             wait every outstanding reduce-scatter, then clip, step.

**the root unit never reshards mid-step.** token_emb runs at the very start of
forward and lm_head at the very end -- and `ARObjective.compute_loss` reads
`model.lm_head.weight` *after* the model's forward has already returned, to
feed the fused linear+ce. any reshard-on-forward-post design hands the
objective a freed weight. so the root unit is gathered once at the start of
forward and stays unsharded until its post-backward. this is what real fsdp2
does for the root module too (`reshard_after_forward=False`). the cost is
honest and should be measured: one unsharded copy of the tied embedding held
for the step (~311 MB bf16 at qwen3-0.6B). the alternative -- moving the fused
ce inside the model's forward -- is cheaper in memory and worse for the
objective abstraction; if the memory bites, that's the trade to revisit.

===========================================================================
the thing that will actually break
===========================================================================

**storage lifetime across streams.** ddp got away with no comm stream because
`async_op=True` on the default stream still overlaps with nccl's own. fsdp
needs a stream it owns plus explicit events, and the gathered buffer's storage
must stay alive from the gather on the comm stream until the last consumer on
the compute stream is done. get it wrong and it corrupts nondeterministically,
under memory pressure, hours in. `record_stream` (or an explicit event the
allocator can see) on every buffer that crosses the boundary, and treat "which
stream does this storage belong to" as something asserted, not assumed.

**freeing means resizing storage to 0, not dropping a python reference.** the
block's saved activations hold references to the gathered weight tensors, so
`del` frees nothing. resize the untyped storage. and note the failure mode:
getting this wrong still *trains correctly*, it just doesn't save any memory --
which is why tests/fsdp2.py asserts storage size directly, between units,
during a real forward.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from torchure.core.collective import MeshLike


@dataclass
class MixedPrecisionPolicy:
    """
    what dtype each leg of the step runs in. defaults are the recipe argued for
    in the module docstring; they're a dataclass so the parity tests can run the
    whole thing in fp32 (`MixedPrecisionPolicy.fp32()`) and get near-exact
    agreement with a single-process reference.
    """

    # dtype the params are gathered and computed in
    param_dtype: torch.dtype = torch.bfloat16
    # dtype the reduce-scatter runs in
    reduce_dtype: torch.dtype = torch.bfloat16
    # dtype of the sharded master params and of `.grad` the optimizer reads
    master_dtype: torch.dtype = torch.float32
    # module types whose params are never downcast. RMSNorm gains are fp32
    # parameters on purpose -- see Trainer._init_weights for what happens when
    # they aren't.
    keep_master: tuple[type[nn.Module], ...] = (nn.RMSNorm, nn.LayerNorm)

    @staticmethod
    def fp32() -> "MixedPrecisionPolicy":
        """no-op policy: everything fp32. what the cpu parity tests run."""
        return MixedPrecisionPolicy(
            param_dtype=torch.float32, reduce_dtype=torch.float32
        )


@dataclass
class _ParamInfo:
    """where one param lives inside its unit's two flat buffers."""

    param: nn.Parameter
    module: nn.Module
    name: str            # attribute name on `module`, for re-binding .data
    full_shape: torch.Size
    offset: int          # element offset into the SHARDED flat buffer
    numel: int           # sharded numel


class FSDPUnit:
    """
    one all-gather's worth of parameters: a transformer block, or the root
    (token_emb + lm_head + norm).

    owns three storages and swaps `.data` / `.grad` between them:

      shard_flat      fp32, persistent, 1/dp_shard of the unit's params.
                      the master weights. never freed. the optimizer's state is
                      allocated against views of this.
      unsharded_flat  bf16, allocated once and REUSED every step so the param
                      views are the same tensor objects across steps (stable
                      for dynamo guards -- this is what collective.all_gather's
                      `output_tensor=` was added for). storage resized to 0
                      between uses.
      grad_flat       bf16, unsharded, zeroed before each backward; autograd
                      accumulates into views of it. freed after the
                      reduce-scatter.
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
        its pre-forward until its post-backward, because the objective consumes
        lm_head.weight after the model's forward returns.
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
        self.pending = 0  # post-accumulate-grad countdown, like ddp's buckets
        raise NotImplementedError

    # --- construction -----------------------------------------------------

    def _collect_params(self, modules: list[nn.Module]) -> list[nn.Parameter]:
        """
        unique params under `modules`, in a deterministic order (module
        traversal order, which is identical on every rank). dedup by `id`, NOT
        by name: `lm_head.weight is token_emb.weight` is one storage under two
        names and must appear exactly once.
        """
        raise NotImplementedError

    def _build_shards(self) -> None:
        """
        allocate shard_flat and fill it.

        v0 shard construction (roadmap): every rank has already run
        `init_weights()` on the full model with the same seed, so slicing dim 0
        locally is bit-identical across ranks with zero comm, and init_weights'
        residual scaling stays a whole-tensor op. meta-device init and
        shard-local rng are a later nicety.

        assert `full_shape[0] % shard_size == 0` per param and name the param in
        the message. no padding in v0 (see dtensor.py for where it would go).
        """
        raise NotImplementedError

    def _alloc_buffers(self) -> None:
        """
        allocate unsharded_flat (bf16, full unit numel) once, then immediately
        resize its storage to 0. build the param views into it now, so they are
        stable objects for the life of the run.
        """
        raise NotImplementedError

    # --- forward ----------------------------------------------------------

    def unshard(self, async_op: bool = True) -> None:
        """
        resize unsharded_flat's storage back up, cast shard_flat into the bf16
        staging shard, and all_gather into unsharded_flat on the comm stream.

        does NOT wait -- that's `wait_unshard`. the split is the whole point:
        prefetch calls this for unit i+1 while unit i computes.

        the cast needs its own staging buffer (bf16 copy of the local shard);
        it's per-unit and reused, and it is a real allocation, not free.
        no-op if already unsharded (prefetch may have got here first).
        """
        raise NotImplementedError

    def wait_unshard(self) -> None:
        """
        have the compute stream wait on the gather's event, then point every
        param's `.data` at its bf16 view. idempotent.
        """
        raise NotImplementedError

    def reshard(self) -> None:
        """
        point every param's `.data` back at its fp32 shard view and
        `unsharded_flat.untyped_storage().resize_(0)`.

        resize, not `del` -- saved activations still reference the view
        tensors, so dropping python references frees nothing and the memory win
        silently doesn't happen. no-op when reshard_after_forward is False.
        """
        raise NotImplementedError

    # --- backward ---------------------------------------------------------

    def prepare_grads(self) -> None:
        """
        allocate/zero grad_flat and set each `param.grad` to its unsharded bf16
        view, so autograd accumulates directly into the flat buffer -- one
        reduce-scatter input, no per-param grad allocations, and no foreach copy
        the way ddp's buckets need one.

        called from the backward pre-hook, after wait_unshard. note the tied
        embedding accumulates two contributions into the same view; that is
        correct and is exactly what tests/fsdp2.py's tied-embedding test pins.
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
        per-unit sharded bf16 output buffer, then reshard the params and free
        grad_flat.

        the "avg" is what makes N ranks x bs=1 equal 1 process x bs=N, same as
        ddp. collective.reduce_scatter already handles async "avg" on gloo by
        pre-dividing (it divides out-of-place on purpose -- under fsdp the input
        IS the unsharded grad buffer and scaling it in place would corrupt every
        later read), so the cpu tests exercise the same overlapped path the gpu
        does. no fixup owed after the wait beyond the upcast.

        stash the work handle; `finalize_grads` does the upcast-accumulate.
        """
        raise NotImplementedError

    def finalize_grads(self) -> None:
        """
        wait the reduce-scatter, then accumulate the sharded bf16 result into
        the fp32 `param.grad` (`+=`, not `=` -- grad accumulation reduce-scatters
        every microbatch into the same fp32 shard).

        after this, `param.data` and `param.grad` are both sharded fp32 and the
        optimizer can step. this is also the point at which clip_grad_norm's
        `shard_dims=("dp_shard",)` is telling the truth; an unsharded-shaped
        grad sneaking through gives a silently *too large* norm, not a crash.
        """
        raise NotImplementedError


class FSDP:
    """
    the model-level wrapper. mirrors DDP's shape so the trainer's
    `_parallelize` doesn't grow a second idiom:

        self.fsdp = FSDP(model, mesh, shard_dim="dp_shard")
        ...
        loss.backward()
        self.fsdp.sync()
        clip_grad_norm(model.parameters(), mesh, max_norm,
                       shard_dims=("dp_shard",))
        optimizer.step()

    `dp_shard == 1` must be a TOTAL no-op: no hooks, no buffers, no groups
    touched, exactly the way DDP is at `dp == 1`. tests/fsdp2.py asserts it.
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
        `replicate_dim` is hsdp (stage 4): after the reduce-scatter along
        shard_dim, all_reduce the sharded grads along replicate_dim. two
        collectives, one ordering, no new state. None disables it, and
        `replicate_dim` of size 1 must degenerate to the same thing.

        `prefetch_depth` is how many units ahead to launch gathers for. 1 by
        default; it costs one extra unsharded buffer live at a time, so it is a
        memory knob as much as a throughput one.
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
            return
        raise NotImplementedError

    @property
    def shard_dims(self) -> tuple[str, ...]:
        """
        mesh axes across which param/grad state is sharded. this is what the
        trainer passes to clip_grad_norm -- an unsharded-shaped grad sneaking
        through with `()` gives a silently too-large norm, not a crash.

        it doubles as "is model state sharded" for the checkpoint guard, which
        is why `dp_shard == 1` must answer `()` rather than the axis name: at
        size 1 nothing is split and rank 0's tensors really are the model.
        """
        return () if self.shard_size == 1 else (self.shard_dim,)

    # --- setup ------------------------------------------------------------

    def _build_units(self, model: nn.Module) -> list[FSDPUnit]:
        """
        unit 0 = root (token_emb + lm_head + final norm, reshard_after_forward
        False), then one unit per entry in `model.blocks`.

        `model.blocks` is the contract (roadmap principle 4: it's the unit of
        compile, fsdp wrapping, activation checkpointing and pp splitting), so
        assert it exists rather than walking the module tree heuristically.

        assert the tie -- `model.lm_head.weight is model.token_emb.weight` --
        instead of assuming it. an untied model isn't wrong, it just needs the
        two in one unit for a different reason (both are vocab-sized), and the
        assert is where that gets noticed.

        every param must land in exactly one unit. check that the union covers
        `model.parameters()` and that no param is claimed twice; a param that
        falls through gets no gather, no reduce-scatter, and drifts between
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

        registered once at init and never removed, like ddp. one FSDP instance
        per model.

        for a multi-module unit (the root: token_emb, lm_head, norm) the
        forward hooks go on the modules that BOUND the unit's use -- unshard on
        token_emb's pre-forward, and for the root there is no reshard hook at
        all, since it's released in post_backward.
        """
        raise NotImplementedError

    # --- prefetch ---------------------------------------------------------

    def _record_order(self, unit: FSDPUnit) -> None:
        """
        append to `_forward_order` on the first forward only. execution order,
        not construction order -- they agree for qwen3 today and won't for the
        first model that reorders or reuses blocks.
        """
        raise NotImplementedError

    def _prefetch(self, unit: FSDPUnit, backward: bool) -> None:
        """
        launch `prefetch_depth` units' gathers ahead of `unit`: forward order
        going forward, its exact reverse going backward.

        no-op until the order is recorded (the first forward is strictly
        serial comm->compute, and that's fine). a mismatch here degrades
        silently into no prefetching at all rather than into a wrong answer,
        which is why tests/fsdp2.py checks the recorded order directly.
        """
        raise NotImplementedError

    # --- step boundary ----------------------------------------------------

    def sync(self) -> None:
        """
        called between `loss.backward()` and `clip_grad_norm`. waits every
        outstanding reduce-scatter and runs each unit's `finalize_grads`.

        assert every unit fired, with ddp's error message discipline -- a unit
        that never completed means a param got no grad (frozen), or a hook
        didn't fire because a block was skipped, and the resulting grads are
        wrong rather than absent.

        also assert every unit is resharded on the way out, root included:
        leaving one gathered is the failure mode where fsdp trains perfectly and
        saves no memory.
        """
        if self.shard_size == 1:
            return
        raise NotImplementedError

    def state_dict(self) -> dict:
        """
        phase 3. until sharded checkpointing exists this MUST raise: rank 0
        writing `model.state_dict()` under fsdp writes one rank's shard as if it
        were the model, and it's a corrupt file that only announces itself at
        resume. the trainer should refuse to start an fsdp run with
        checkpointing enabled rather than rely on this firing at save time.
        """
        raise NotImplementedError(
            "sharded checkpointing is phase 3; an fsdp run cannot checkpoint yet"
        )


if __name__ == "__main__":
    print("fsdp2: see tests/fsdp2.py")
