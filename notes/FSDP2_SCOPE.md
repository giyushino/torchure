# fsdp2 scope

checklist for roadmap phase 2. no code here — this is the list of things that
have to exist, the decisions that have to be made before writing them, and the
tests that decide whether it's done.

ordered by dependency. each stage has an exit criterion; nothing counts as
done until its test passes on gloo/cpu (`mp.spawn`, no gpus), then nccl.

---

## stage 0 — gaps in what already exists

these are prerequisites, not fsdp itself. all of them are things the current
code does that fsdp will break on.

- [ ] **`all_gather` can't gather into a caller-owned buffer.** it allocates a
      fresh output every call. fsdp wants one preallocated per-unit buffer that
      it reuses across steps (and frees between them), otherwise the allocator
      churns a full unsharded copy per unit per forward. add an `out=` path (or
      a sibling entry point) that writes into a given tensor.
- [ ] **no dtype cast on the comm path.** gathering fp32 master shards in bf16
      halves the biggest collective in the step. that needs a staging buffer
      holding the bf16 copy of the local shard. decide where that buffer lives
      (per unit, reused) — it's a real allocation, not free.
- [ ] **`reduce_scatter` refuses async `"avg"` on gloo** (no `ReduceOp.AVG`;
      it sums, waits, then divides locally, and asserts on `async_op=True`).
      this is exactly the call the backward path wants overlapped, so the cpu
      parity tests can't exercise the overlapped path as written. pick one:
      reduce with `"sum"` and scale locally after the wait (works everywhere,
      one extra foreach), or keep `"avg"` and accept that gloo tests run the
      sync path. write down which, because it changes what the cpu tests prove.
- [ ] **no dedicated comm stream anywhere.** ddp got away with it because
      `async_op=True` on the default stream still overlaps with compute on
      nccl's own stream. fsdp needs a stream it owns plus explicit events, and
      needs the buffers kept alive across the stream boundary. this is the
      single biggest source of nondeterministic corruption in the whole phase —
      treat "which stream does this tensor's storage belong to" as a thing you
      assert, not assume.
- [ ] **the objective reaches into `model.lm_head.weight` from outside the
      model.** `ARObjective.compute_loss` calls `model(..., return_final_hidden=True)`
      and then hands `model.lm_head.weight` to the fused linear+ce, so the head
      weight is consumed *after* the model's forward has returned. any design
      that unshards on a module forward-pre-hook and reshards on the
      forward-post-hook will hand the objective a resharded (or freed) weight.
      decide the fix now: keep the head's unit unsharded until loss is
      computed, or move the fused-ce call inside the model's forward, or have
      the objective ask the wrapper for the unsharded weight. this is the first
      thing that will break and the least obvious.
- [ ] **`clip_grad_norm` needs sharded grads.** it already takes `shard_dims`,
      so the math is there — but it reads `p.grad` off the param list. settle
      what `p.grad` *is* after fsdp (sharded fp32? a dtensor?) and make the
      call site pass `shard_dims=("dp_shard",)`. an unsharded-shaped grad
      sneaking in gives a silently wrong (too large) norm, not a crash.
- [ ] **checkpointing will silently save one rank's shard as if it were the
      model.** phase 3 fixes it properly; for this phase add a loud guard so an
      fsdp run can't write a corrupt checkpoint in the meantime.
- [ ] **mesh axes.** `dp` becomes `dp_replicate` × `dp_shard`, with
      `mesh.flatten(("dp_replicate", "dp_shard"), "dp")` for the dataloader
      (that's what `flatten` was built for). config validation: reject a spec
      where the product doesn't match, and keep `dp: N` working as a shorthand
      for `dp_shard: N` (or `dp_replicate: N` — pick one and document it).

## stage 1 — dtensor / placement bookkeeping

roadmap 2.1. bookkeeping wrapper with explicit `redistribute`, **not** a
`__torch_dispatch__` subclass — nothing downstream needs op propagation.

- [ ] `Shard(dim)`, `Replicate()`, `Partial(op)` as value types (hashable,
      comparable, printable — they end up in checkpoint metadata).
- [ ] a wrapper holding `(local_tensor, mesh, placements)` with
      `from_local / to_local / full_tensor / redistribute`.
- [ ] the redistribute table: `Shard(d)→Replicate` (all_gather),
      `Replicate→Shard(d)` (local slice, no comm), `Partial→Replicate`
      (all_reduce), `Partial→Shard(d)` (reduce_scatter),
      `Shard(i)→Shard(j)` (all_to_all).
- [ ] **decision: divisibility vs padding.** the collectives assert
      divisibility and don't pad. every qwen3 param's dim 0 happens to be
      divisible by 8, so v0 can assert loudly and move on — but write down that
      the first model with an awkward vocab or head count will need padding,
      and where it would go. don't discover this at 3am.
- [ ] **decision: does the optimizer ever see a dtensor?** if params stay
      plain tensors and dtensor is only used for save/load metadata, fused adamw
      keeps working untouched. if params become dtensors, every `p.grad`,
      `foreach` call, and the clip path has to unwrap. the cheap path is: plain
      tensors at runtime, placements recorded alongside. recommend that; note
      it as the reason if you later want real dtensor.

**exit:** every redistribute cell tested against `full_tensor()` ground truth
on gloo, both directions where they exist.

## stage 2 — sharding + the parameter lifecycle

roadmap 2.2. the core of the phase.

- [ ] **unit definition.** a unit = one transformer block, plus one unit for
      `(token_emb + lm_head + norm)`. the tied weight (`lm_head.weight is
      token_emb.weight`) *forces* those two into one unit — sharding them
      separately double-gathers one storage and corrupts grad flow. assert the
      tie rather than assuming it.
- [ ] **decision: per-param shards vs one flat shard per unit.** per-param
      `Shard(0)` is what the roadmap says and what maps cleanly to checkpoint
      metadata; one flat buffer per unit is what makes it fast — one all-gather
      per unit instead of one per param, which is the same lesson the ddp
      buckets already taught on this box. these aren't exclusive: shard
      per-param logically, back them with one flat storage per unit and views
      into it. decide before writing anything, because it determines the shape
      of everything else.
- [ ] **shard construction at init.** v0 per the roadmap: init full weights on
      every rank with a fixed seed, slice locally. bit-identical, trivially
      correct, and `init_weights`'s residual scaling stays a whole-tensor
      operation. meta-device init and shard-local rng are a later nicety —
      note them, don't build them. (`_build_model → _parallelize → _init_weights`
      in the trainer is already in the right order for this.)
- [ ] **unshard (pre-forward).** all-gather the unit's shards into its buffer,
      cast to bf16 on the way, point the modules' params at views of the
      gathered buffer.
- [ ] **reshard (post-forward).** the gathered storage must actually be
      released, and the *only* way that happens is if nothing else holds a
      reference: saved activations from the block's backward hold the weight
      tensor. this is why real fsdp resizes the storage to 0 rather than
      dropping a python reference. get this right or the memory win doesn't
      exist — and it will still train correctly while not saving memory, which
      is why stage 5 tests the storage size directly.
- [ ] **re-unshard (pre-backward).** decide the mechanism: module full-backward
      pre-hooks, or an autograd function inserted on the unit's output. hooks
      are simpler; an autograd function is what survives the graph being
      reordered by compile. pick with the compile composition test in mind.
- [ ] **post-backward.** reduce-scatter the unit's unsharded grads into sharded
      fp32 grads, free the unsharded grad storage, publish `.grad` on the
      sharded params. the trigger is the same
      `register_post_accumulate_grad_hook` machinery ddp uses — one counter per
      unit rather than per bucket.
- [ ] **prefetch.** record unit execution order on the first forward; use it to
      launch unit *i+1*'s gather while unit *i* computes, and the reverse order
      in backward. without this fsdp is strictly serial comm→compute and the
      tps number will be ugly. depth of prefetch (1 unit? 2?) is a knob with a
      memory cost — make it config, default 1.
- [ ] **grad accumulation / no_sync.** two options with different memory
      profiles: keep unsharded grads accumulated across microbatches (fast, but
      holds a full unsharded grad per unit), or reduce-scatter every microbatch
      and accumulate in the sharded fp32 grad (one extra collective per
      microbatch, sharded memory). the second is almost certainly right here
      given why fsdp is being added at all. mirror ddp's `requires_sync`
      attribute so the trainer loop doesn't grow a second idiom.
- [ ] **mixed precision policy, written down explicitly.** sharded master
      params fp32; gathered params bf16; grads produced bf16; **decide** whether
      the reduce-scatter runs in bf16 (half the comm, accumulation error across
      dp_shard ranks) or fp32 (upcast before the collective, 2× comm). also
      decide what happens to autocast, which currently handles activations —
      once params arrive already bf16, autocast's param casting is redundant
      but its activation policy isn't. and keep the norm weights fp32: the
      whole "bf16 params freeze under adamw" lesson in `_init_weights` applies
      to anything the sharding path might quietly downcast.
- [ ] **optimizer on sharded params.** adam state allocates against the sharded
      shapes — this is the actual memory win, and it happens for free *only if*
      the optimizer is constructed after sharding. the trainer builds the
      optimizer from `model.parameters()`; confirm those are the sharded ones by
      then, and that `fused=True` still applies (it wants same-device,
      same-dtype, contiguous — flat-per-unit storage helps here).

**exit:** grads after one backward match the single-process reference; adam
state numel per rank scales as 1/dp_shard.

## stage 3 — composition

roadmap 2.4.

- [ ] **per-block `torch.compile`.** hooks sit outside the compiled region by
      construction (`_compile` compiles `block.forward`), but swapping the
      storage a param view points at, every step, is exactly the kind of thing
      dynamo guards on. expect recompiles or guard failures; the test is loss
      equality vs eager, and the diagnostic is `TORCH_LOGS=recompiles`.
- [ ] **activation checkpointing per block.** lands here because it's the knob
      that buys microbatch headroom (CHANGES.md round 2: bs=4 was +11% tps when
      memory allowed). the interaction to watch: recompute re-runs the block's
      forward during backward, so the unit must be unsharded *then* too — this
      is the second re-gather that naive implementations miss.
- [ ] **the fused linear+ce path** — see stage 0. it consumes the head weight
      outside the model's forward.
- [ ] **the cuda prefetcher and the comm stream** both queue work on side
      streams now. confirm they don't fight over the same events.

## stage 4 — hsdp

roadmap 2.5. cheap once both mesh axes exist.

- [ ] reduce-scatter along `dp_shard`, then all_reduce the sharded grads along
      `dp_replicate`. two collectives, one ordering, no new state.
- [ ] config validation and a clear error when the product doesn't match world
      size (the mesh already asserts this; make the message name the axes).
- [ ] `dp_replicate=1` and `dp_shard=1` must both degenerate to exactly the
      existing behaviour, no-op ctor included, the way `DDP` does at `dp=1`.

---

## tests

new files, matching the existing `tests/` style: `mp.spawn`, gloo, cpu-only,
tiny real qwen3 (not an mlp — the tied embedding and rmsnorm structure are
what break), a deterministic batch each rank recomputes locally so no gathers
are needed for the comparison, and a `TESTS` list swept over parameters.

**`tests/dtensor.py`**
- [ ] every redistribute cell vs `full_tensor()` ground truth, at 2 and 4 ranks.
- [ ] `all_gather(reduce_scatter(x)) == all_reduce(x)` — the identity fsdp is
      built on. (`collective.py` already claims this is tested; make it explicit
      at the dtensor level too.)
- [ ] round trip: shard then gather returns the original bit-exact.
- [ ] non-divisible shape asserts loudly rather than silently truncating.

**`tests/fsdp2.py`** — correctness
- [ ] **grad parity**: N ranks × bs=1 under fsdp == 1 process × bs=N. same
      invariant as `tests/data_parallel.py`, same tolerance discipline.
- [ ] **ddp cross-check**: same model, same batch, ddp vs fsdp grads agree.
      this is the one that catches reduce-scatter ordering/scaling mistakes,
      because both paths are supposed to compute the identical thing.
- [ ] **multi-step**: 5 optimizer steps, fp32 tiny model, loss curve identical
      to single-process. catches optimizer-state sharding bugs that a
      single-backward test can't see.
- [ ] **tied embedding**: assert emb and head land in one unit; assert the
      shared weight's grad equals the sum of both contributions; assert it's
      gathered exactly once per forward.
- [ ] **grad accumulation**: K microbatches × N ranks == 1 process × bs=N*K,
      with exactly the intended number of collectives (see counting below).
- [ ] **clip_grad_norm**: sharded norm with `shard_dims=("dp_shard",)` equals
      the single-process norm, and post-clip grads match.
- [ ] **hsdp**: `dp_replicate=2 × dp_shard=2` grads == `dp=4` ddp grads.

**`tests/fsdp2.py`** — the invariants that are invisible from the outside
these matter more than the parity tests: fsdp that never frees memory, or
gathers twice, still trains perfectly and just doesn't do its job.
- [ ] **collective counting**: wrap the collective entry points and assert
      exactly one all-gather per unit per forward, one per unit per backward
      (or zero if the forward's gather was kept), one reduce-scatter per unit
      per step. `tests/data_parallel.py` already does launch-counting — reuse
      the approach.
- [ ] **storage actually freed**: after a unit's forward completes, its
      unsharded buffer's storage size is 0 (or the buffer is provably
      unreferenced). assert this *between* units during a real forward, not
      after the whole step.
- [ ] **memory accounting**: sum of param + grad + adam-state numel per rank is
      within a small constant of `total/dp_shard`. cheap to compute on cpu, and
      it's the actual reason this phase exists. the gpu version of this is a
      number in CHANGES.md, not a test.
- [ ] **prefetch order**: the recorded unit order matches module execution
      order forward, and its exact reverse in backward. a mismatch degrades
      silently into no prefetching at all.
- [ ] **no-op at size 1**: `dp_shard=1` registers no hooks, allocates no
      buffers, launches no collectives.

**gpu-only, not in the cpu suite**
- [ ] compile on/off loss equality, and recompile count stays bounded across
      steps.
- [ ] activation checkpointing on/off loss equality, including the recompute
      re-gather.
- [ ] a profiler trace showing gather overlapping compute — the phase's perf
      exit criterion, recorded in CHANGES.md, not asserted in a test.

---

## exit criteria (from the roadmap, restated concretely)

1. parity vs ddp on 2 ranks — fp32 tiny model near-exact; bf16 0.6B matching
   trajectory.
2. params + grads + adam states per gpu shrink ~linearly in `dp_shard`
   (0.6B adam fp32 ≈ 4.5GB → ~2.3GB at shard=2).
3. tps within ~10% of ddp at 0.6B / 2 gpus — fsdp overhead is real at this
   scale; record it honestly rather than tuning until it looks good.
4. compile and activation checkpointing both compose, ablated in CHANGES.md.

## deliberately out of scope

- `__torch_dispatch__` sharding propagation (stage 1's whole point).
- uneven / padded shards (assert divisibility, note where padding would go).
- meta-device init and shard-local rng (v0 inits full and slices).
- sharded/resharding checkpoints — phase 3, gated behind a loud guard here.
- fp8 comm, `all_gather` fusion across units, cpu offload.
