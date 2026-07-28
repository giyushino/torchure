# DDP — work log

> **stale (as of 7/27):** this note describes an `overlap` config knob and an
> `overlap=False` mode of `sync()`. that mode existed only as the correctness
> baseline / ablation and was **removed before merge** — `DDP.__init__` takes
> `(model, mesh, dim, bucket_mb)` and there is no non-overlap path. the `+11.4%`
> ablation number below was real when measured; the knob it used is gone. left
> as-is otherwise, it's a work log.

goal: get data parallel training up and running (roadmap phase 1), tested with

```
uv run torchrun --nproc-per-node 8 -m torchure.train.train --config configs/qwen3_dense_climbmix_ddp.json
```

priority order per the brief: throughput first, clean code second. so this
lands the roadmap 1.2 *overlap* version directly (bucketed grad sync launched
from backward hooks), with the 1.1 non-overlap version kept as a config knob —
it doubles as the correctness baseline and the ablation that measures what
overlap buys.

## hardware reality check (drives the whole design)

`nvidia-smi topo -m`: 8×A40, **no NVLink**. GPUs are paired behind PCIe
switches (PIX within a pair), everything else crosses the CPU/NUMA
interconnect (SYS), split 4+4 across two NUMA nodes. the fp32 grad payload is
2.26 GiB/step (0.6B params), and a ring all-reduce moves ~2× that through the
slowest link. on this box grad sync is *the* cost of DDP — hiding it behind
backward compute is where all the throughput is, which is why overlap wasn't
deferred to a later pass.

## what landed

### `torchure/parallelism/data_parallel.py` (new — was an empty stub)

`DDP(model, mesh, dim="dp", bucket_mb=25, overlap=True)` + `ddp.sync()`
between `loss.backward()` and `optimizer.step()`. mechanics:

- **replicate at init**: broadcast every param + buffer from dp-coord 0, so
  ranks agree even though each rank runs `init_weights()` with its own rng.
  per-tensor broadcasts, one-time cost, no flattening needed.
- **buckets in reverse `model.parameters()` order** (~ backward completion
  order), default cap 25MB. `register_post_accumulate_grad_hook` counts down
  each bucket; when a bucket fills, its all-reduce ("avg", `async_op=True`)
  launches immediately, so early buckets fly while backward still computes.
  fill order is identical across ranks (same graph → same engine schedule),
  so nccl sees the same collective order everywhere — the same assumption
  pytorch DDP makes.
- **multi-param buckets** share a preallocated flat buffer: grads are copied
  in with one `torch._foreach_copy_` launch, `.grad` is repointed at views of
  the buffer, the buffer is reduced. the reduced values are then already what
  the optimizer reads — **no copy back out** (pytorch DDP's
  `gradient_as_bucket_view`, minus the copy-in only when `zero_grad`
  uses `set_to_none=False`; both zero_grad modes are correct).
- **single-param buckets for anything ≥ the cap**: reduced **in place via
  `.grad`**, no buffer, no copy. this exists for the tied-embedding grad
  (151936×1024 fp32 = 622 MiB — 26% of the whole payload): staging it through
  a buffer would burn a 1.2 GiB read+write and 622 MiB of transient memory
  for nothing. it also always finishes last (embedding grad completes at the
  very end of backward, tied or not), so it's the unavoidable exposed tail.
- **`sync()`** waits on all work handles; with `overlap=False` it instead
  launches every bucket back-to-back there (still batched on the comm
  stream), which is the ablation/baseline mode.
- **failure over hang**: if a bucket never filled (frozen param, grad
  accumulation), `sync()` asserts with a count instead of deadlocking inside
  nccl. grad accumulation needs real `no_sync` semantics — roadmap 1.3,
  deliberately not built yet.
- **dp=1 is a complete no-op**: no hooks, no groups touched, zero per-step
  overhead. single gpu stays the same code path (principle: world_size=1 is
  not a special case).

### `torchure/core/collective.py` — one contract change (the "note why" bit)

the only abstraction change this task needed. `all_reduce(op="avg")` on
backends without `ReduceOp.AVG` (gloo) was sum-then-divide and asserted on
`async_op=True` ("revisit when a consumer needs it"). the ddp hooks are that
consumer: they need `(tensor, work)` back from an "avg" so the wait can happen
later, on gloo too (the parity tests run on gloo/cpu). fix: **pre-divide,
then sum** — same mean for a fixed group size, no post-wait fixup, so async
works and the branch got *smaller*. nccl still uses `ReduceOp.AVG` directly
(no extra kernel on the hot path). `reduce_scatter`'s gloo-avg branch is
untouched — no consumer yet (fsdp2 will decide).

everything else (broadcast, async all_reduce, barrier, mesh groups/coords)
was already sufficient — the collectives layer held up as designed.

### trainer wiring (`torchure/train/trainer.py`)

- `_parallelize` builds `self.ddp` (before `_compile` on purpose: hooks sit
  on params at the AccumulateGrad boundary, outside the compiled regions —
  per-block compile and the hooks compose without graph breaks).
- `train_step`: `loss.backward()` → `self.ddp.sync()` → `optimizer.step()`.
  logged loss is dp-averaged (scalar all-reduce, noise next to the step).
- tps in the step print is now **global** (seq_len × batch_size × dp),
  step/peak-mem prints are rank-0 only.
- config knobs under `"ddp"`: `bucket_mb` (default 25), `overlap` (default
  true). absent section → defaults, so old configs keep working.

### checkpointing under dp (`checkpointer.py`, `trainer.checkpoint/_resume`)

the ddp test config never hits a save (save_steps 10000 > total 100), but
leaving 8 ranks racing to write the same files would be a landmine:

- rank 0 writes model/optimizer/scheduler/trainer (they're replicas);
  **every dp rank writes its own dataloader file** (`dataloader_dp{r}.pt` —
  each rank's stream position is different state). single-gpu runs now write
  `dataloader_dp0.pt`; old checkpoints with `dataloader.pt` won't resume
  (none live that matter).
- `barrier()` fences the save so no rank races into the next step (or exits)
  mid-write.
- `_resume` asserts the checkpoint's mesh == current mesh: resuming 8-way
  data over a different dp size would silently misread shard positions.
- known wart carried forward: rank 0's rng snapshot lands on every rank at
  resume. harmless while nothing samples per step; needs per-rank seed
  derivation (roadmap 0.3) before dropout/masked-diffusion.

### misc

- `utils.debug_time` prints on rank 0 only (RANK env; unset == single
  process == unchanged). same for the dataloader's "num shards" print.
- `tests/checkpointer.py` updated for the dataloader filename.

## correctness evidence

`tests/data_parallel.py` (new; gloo/cpu, `uv run tests/data_parallel.py
--world-size 4`) — the roadmap exit-criterion parity, on a tiny *real* Qwen3
(2 layers, vocab 64) so the tied embedding is exercised (one param, two grad
contributions, AccumulateGrad must fire exactly once):

- `test_replicate` — post-init params bitwise equal to dp-coord-0's.
- grad parity, N ranks × bs=1 vs 1 process × bs=N (every rank recomputes the
  reference locally; deterministic batch): swept over bucket shapes —
  one flat bucket / many flat buckets / all single-param in-place
  (`bucket_mb=0`) / `overlap=False`. two backward-sync-zero cycles each, to
  catch rearm and handle-clearing bugs.

all 5 pass; the 12 collective tests still pass after the avg change; the
checkpointer roundtrip still passes.

## the grad-sync-is-6x-slower-than-backward investigation

first 8-gpu run: loss curve healthy (12.13 fresh CE → 7.7 by step 20, same
shape as single gpu) but ~17.6k **global** tps ≈ 2.2k/gpu vs ~13k single-gpu
— 17% scaling. step time 3.7s vs 0.62s single-gpu, i.e. ~3.1s of exposed
comm. worked outward from the ddp code:

1. **raw nccl all_reduce microbench** (`scratchpad allreduce_bench.py`,
   25/128/622 MiB): 1.30 GB/s bus bandwidth at every size. so the ddp
   integration was innocent — 2.26 GiB at 1.3 GB/s *is* 3.0s; overlap can't
   hide comm that's 6x the backward.
2. **fabric checks**: `pcie.link.gen.current` reads gen1 idle (power
   management — red herring; retrains to gen4 under load). pinned h2d/d2h:
   13–25 GB/s per gpu. direct gpu↔gpu copies: 26 GB/s intra-numa (even
   across switch pairs), 13 GB/s cross-numa, p2p access granted everywhere,
   no iommu on the cmdline. fabric fine.
3. **`NCCL_DEBUG=INFO`**: the smoking gun. ring 0-1-2-3-4-5-6-7 uses
   P2P/CUMEM inside pcie-switch pairs but **SHM/direct/direct for every
   pair-to-pair hop** — nccl's default `NCCL_P2P_LEVEL` stops p2p at the
   host bridge, and its host-staged SHM path runs ~1.3 GB/s here. (also why
   the earlier `NCCL_P2P_DISABLE=1` probe barely moved: the slow hops were
   already SHM.)
4. **fix: `NCCL_P2P_LEVEL=SYS`** (allow p2p at any topology distance —
   measured fine on this box): 1.30 → **10.9 GB/s**, 8.4x. swept
   alternatives: `NCCL_ALGO=Tree` is catastrophic (0.25 GB/s — tree
   endpoints fall onto the socket path), extra channels
   (`NCCL_MIN_NCHANNELS=4`) neutral. ring + p2p wins.

set as `os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")` in train.py next to
the rendezvous defaults: overridable by env for a box where distant p2p
genuinely is slower than SHM (the classic across-QPI pathology — real, just
not here). at 10.9 GB/s the 2.26 GiB payload costs ~390ms of wire time,
which is the same order as backward, so overlap goes from pointless to
load-bearing.

## measurements

box idle for all runs (co-tenant load halves TPS on this machine — always
check `nvidia-smi` before trusting numbers). seq_len 4096, bs 2/gpu,
bucket_mb 25, overlap on, steady-state steps (10+), `NCCL_P2P_LEVEL=SYS`.
global tps = tokens through the whole job per second; the 100-step 8-gpu run
finished loss 7.56 (the unigram-entropy plateau — expected at this token
budget).

| gpus | global tps | tps/gpu | scaling eff | peak mem/gpu |
|------|-----------|---------|-------------|--------------|
| 1    | 13.3k     | 13.3k   | 100%        | 23.88 GiB    |
| 2    | 23.1k     | 11.6k   | 87%         | 25.44 GiB    |
| 4    | 43.2k     | 10.8k   | 81%         | 25.44 GiB    |
| 8    | 73.5k     | 9.2k    | 69%         | 25.44 GiB    |

- **before the nccl fix, 8 gpus ran 17.6k global (17% eff)** — the p2p-level
  find is worth 4.2x end to end.
- **overlap ablation (8 gpus)**: 66.0k with `"overlap": false` (buckets still
  launched back-to-back, one wait) vs 73.5k overlapped → **+11.4%** from
  launching inside backward. this quantifies the overlap instead of the
  roadmap's profiler-trace criterion; an actual trace is still owed.
- memory: +1.56 GiB over single gpu = the flat grad buffers (non-embedding
  params, fp32), matching the design (the 622 MiB embedding grad has no
  buffer by construction).
- where the remaining 8-gpu gap lives (886ms/step vs 620ms single):
  ~105ms is the embedding-tail reduce that structurally can't overlap;
  the rest is partially-hidden bucket comm + p2p reads stealing memory
  bandwidth from compute. levers, in order of expected value: bf16 grad
  compression (halves wire time, needs a loss-curve ablation), bucket-size
  sweep, profiler trace to see what's actually exposed.

## roadmap phase-1 exit criteria status

- (a) parity: gloo tests green (grad parity N×bs1 vs 1×bsN, tied embedding
  exercised, all bucket shapes, both sync modes). ✓
- (b) scaling table for 1/2/4(/8) gpus: above. ✓
- (c) profiler trace showing overlap: quantified via the overlap ablation
  instead; trace still owed. (partial)

## deliberately not built (and why)

- **grad accumulation / `no_sync`** — no consumer: the trainer has no
  microbatch loop yet. the hooks make it a small, contained change
  (suppress launch, reduce on the last microbatch) when 1.3 lands.
- **global grad-norm clipping** — same: no clipping anywhere in the trainer
  yet; for pure DP post-sync local norm is already the global norm.
- **bf16 grad compression** — would halve the wire payload on a box where
  comm is the bottleneck, but it changes numerics; measure overlap first,
  ablate later with a loss-curve comparison, not as part of "get it running".
- **sharded/distributed checkpointing** — roadmap phase 3, after fsdp2.
