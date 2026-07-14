# roadmap

where torchure is going: a pure-pytorch training stack that composes every
major parallelism (DP/DDP, FSDP2, TP/SP, CP, EP, PP) over a named device mesh,
and trains multiple model families (dense AR, MoE, masked-diffusion LMs)
through the same trainer/objective/dataloader abstractions.

this document is ordered by dependency, not by calendar. each phase lists what
gets built, the design decisions that have to be made (with the current lean),
and a concrete exit criterion — nothing counts as done until the parity test
passes and the numbers are in CHANGES.md.

## guiding principles

these already shaped the code; write them down so future phases don't drift:

1. **own the primitives.** all comm goes through `torchure/core/collective.py`
   — nothing else touches `torch.distributed` comm ops directly. mesh, dtensor,
   fsdp2, tp, cp, ep are implemented here, not imported from
   `torch.distributed.*` wrappers. pytorch's versions are the reference to test
   against, not the dependency.
2. **correctness before throughput.** every parallelism must reproduce the
   single-gpu loss curve (tiny model, fp32, tight tolerance) before any
   overlap/perf work on it starts. gloo/cpu first (`mp.spawn`, no gpus needed),
   nccl second.
3. **measure everything.** the CHANGES.md discipline (baseline, ablation,
   adopted-or-rejected, profile snapshot) extends to distributed: every phase
   ends with a scaling table (tps/gpu, memory/gpu, comm overlap % from the
   profiler trace).
4. **per-block granularity is the contract.** `model.blocks` is the unit of
   compile, FSDP wrapping, activation checkpointing, and PP stage splitting.
   new architectures must expose it.
5. **config-driven.** every parallelism/feature is a json config knob with a
   safe default; eager/single-gpu must always keep working for debugging.
6. **minimal deps.** no framework imports (megatron, deepspeed, accelerate,
   liger). optional integrations (wandb, torchao fp8) stay optional.

## current state (2026-07)

| area | state |
|------|-------|
| `core/collective.py` | signatures + contracts done, bodies `NotImplementedError`; tests written (`tests/collective.py`) |
| `core/mesh.py`, `core/dtensor.py`, `core/placement.py` | empty stubs |
| `parallelism/{data_parallel,fsdp2,context_parallel,expert_parallel}.py` | empty stubs |
| models | qwen3 dense (0.6B validated, per-block compile, ~13k tps on A40); llama3 stubbed |
| trainer | single-gpu; lifecycle already split build → parallelize → init → optimizer → data so distributed slots in without reordering |
| checkpointing | naive `torch.save`, has path bugs (`os.mkdir` on the `.pt` file path), no resume path wired |
| objectives | AR CE only; fused-linear-CE **described as adopted in CHANGES.md round 2 but the implementation (`torchure/loss/fused_linear_ce.py` + `ARObjective` wiring) is not in the tree** — only `tests/fused_linear_ce.py` exists. re-land or re-do it (item P-1) |
| dataloader | streaming HF + packing + stateful + CUDA prefetcher; takes global rank/world_size (needs mesh coords later); packing allows cross-document attention (no doc mask) |

---

# part 1 — parallelism

## phase 0: distributed foundations

everything else stacks on this. smallest phase, highest leverage.

**0.1 implement `core/collective.py`.** the contracts and tests exist; make
them green. order of implementation: `all_reduce` → `broadcast` → `all_gather`
→ `reduce_scatter` (verify the fsdp identity test) → `all_to_all` →
`ring_send_recv` → `barrier`. gloo/cpu via `uv run tests/collective.py
--world-size 4`, then nccl via torchrun on a multi-gpu box. known traps are
already documented in the docstrings (coordinate vs global rank, gloo has no
`ReduceOp.AVG`, p2p deadlock without `batch_isend_irecv`).

**0.2 `core/mesh.py`.** `init_mesh({"pp": 1, "dp_replicate": 1, "dp_shard": 2,
"cp": 1, "tp": 2, "ep": 1})`-style constructor satisfying `MeshLike`.
decisions:
- **rank layout**: row-major with the *last* dim fastest-varying, and the
  convention that tp (then cp) are innermost so their groups land on
  NVLink/intra-node links. document the layout once, in mesh.py, with a
  worked 8-rank example.
- **group construction**: `dist.new_group` is itself a collective — all ranks
  must create every group in the same order, including groups they're not in.
  build all groups eagerly at init, cache by dim name.
- **flattening/submeshes**: data loading needs a single "which batch shard am
  i" coordinate = flattened (dp_replicate × dp_shard) index, with tp/cp/pp
  peers sharing it. provide `mesh.flatten(("dp_replicate", "dp_shard"))` or
  equivalent coords helper from day one — retrofitting this is painful.
- size-1 dims must be free (no groups, collectives become no-ops or asserts).

**0.3 trainer bootstrap.** torchrun entrypoint: read
`RANK/LOCAL_RANK/WORLD_SIZE` from env, `init_process_group` with an explicit
timeout, `destroy_process_group` on exit, single-gpu path unchanged when
`WORLD_SIZE` is absent. rank-0-only logging helper, per-dim seed derivation
(same seed across tp/cp for identical init, different seed per dp shard for
data), and route the dataloader's rank/world_size through mesh coords instead
of globals (the trainer TODO already notes this).

**exit criteria:** all `tests/collective.py` tests pass on gloo (4 ranks) and
nccl (2 gpus), including `test_subgroup_isolation` against the real mesh.
trainer still hits ~13k tps single-gpu.

## phase 1: data parallel (DDP)

simplest parallelism, and it exercises the whole stack (mesh, seeding, data
sharding, logging) with only one collective.

**1.1 `parallelism/data_parallel.py` v0 — correctness.** replicate the model
(broadcast params from dp-rank 0 after init so ranks agree even if RNG
drifts); after `loss.backward()`, `all_reduce(grad, mesh, "dp", "avg")` per
param (or one flattened buffer). loss logging averaged over dp for readable
curves.

**1.2 v1 — overlap.** bucketed gradient sync using
`register_post_accumulate_grad_hook` + `async_op=True`: fill ~25MB buckets in
reverse-parameter order, launch the all-reduce when a bucket fills, wait on
all works before `optimizer.step()`. this is where the async `(tensor, work)`
contract and the gloo-avg-after-wait caveat earn their keep.

**1.3 gradient accumulation + clipping.** accumulate locally for K
microbatches, sync grads once per optimizer step (no_sync semantics — the
hooks must respect this). global grad-norm clipping: for pure DP the grads
are identical post-sync so the local norm is correct, but write it against
the mesh anyway (`all_reduce` of squared norms) — FSDP/TP will need the
distributed version, and it's 5 lines now vs a refactor later.

**exit criteria:** (a) parity: 2 ranks × bs=1 matches 1 rank × bs=2 loss
trajectory on identical data order (this forces the dataloader sharding to be
right); (b) scaling: tps table for 1/2/4 gpus in CHANGES.md; (c) profiler
trace showing grad all-reduce overlapped with backward.

## phase 2: DTensor + FSDP2

the centerpiece. FSDP2-style = per-parameter sharding over DTensors, not
flat-parameter FSDP1.

**2.1 `core/placement.py` + `core/dtensor.py`.** `Shard(dim)`, `Replicate()`,
`Partial(op)`; `DTensor` holding (local_tensor, mesh, placements) with
`from_local / to_local / full_tensor / redistribute`. **key design decision,
make it explicitly:** this is a *bookkeeping wrapper with explicit
redistribute*, NOT a `__torch_dispatch__` tensor subclass with sharding
propagation for every aten op. pytorch's DTensor does full op dispatch; that's
a compiler-sized project and none of fsdp2/tp/cp/checkpointing here needs it —
they all know exactly which collective they want. redistribute table to
implement and test:

| from → to | collective |
|-----------|-----------|
| Shard(d) → Replicate | all_gather(gather_dim=d) |
| Replicate → Shard(d) | local slice (no comm) |
| Partial → Replicate | all_reduce |
| Partial → Shard(d) | reduce_scatter(scatter_dim=d) |
| Shard(i) → Shard(j) | all_to_all |

test each against `full_tensor()` ground truth on gloo; uneven-shard padding
is out of scope for v0 (assert divisibility loudly, same policy as the
collectives).

**2.2 `parallelism/fsdp2.py`.** per-unit sharding where a unit = one
transformer block, plus one unit for (token_emb + lm_head + final norm) —
**the tied embedding forces embed and head into the same unit** (one weight,
two modules; sharding them in different units double-gathers and corrupts the
grad flow). mechanics per unit:
- params stored as `Shard(0)` DTensors (fp32 sharded "master" copy);
- pre-forward: async all-gather the unit's params (bf16 cast on the way — this
  replaces autocast for params; keep autocast for activations initially),
  prefetch the *next* unit's gather while the current one computes;
- post-forward: free unsharded params;
- pre-backward: re-gather; post-backward: reduce-scatter grads ("avg") into
  fp32 sharded grads, free unsharded;
- optimizer states built on the sharded fp32 params — adam states shard for
  free, which is the actual memory win.
- comm on a dedicated CUDA stream; wait-events at use sites.

**2.3 meta init.** trainer's `build → parallelize → init` order was designed
for this: build on `meta` device, shard, `to_empty(device)`, then
`init_weights`. v0 pragmatism: at ≤1B params, init full weights on each rank
with a fixed seed and slice locally — bit-identical across ranks, trivially
correct. proper shard-local RNG init is a later nicety; note it and move on.

**2.4 composition checks.** per-block `torch.compile` must keep working
(hooks sit outside the compiled region — this is exactly why per-block
granularity was chosen in change #2). activation checkpointing per block
(`checkpoint(block, ...)`) lands here too: it's the knob that buys microbatch
headroom, and CHANGES.md round 2 already identified bs=4 as +11% tps when
memory allows.

**2.5 HSDP.** once dp splits into (dp_replicate, dp_shard): reduce-scatter
along dp_shard, then all_reduce the sharded grads along dp_replicate. cheap to
add once both dims exist in the mesh; mostly a config/validation exercise.

**exit criteria:** (a) parity vs DDP on 2 ranks (fp32 tiny model: near-exact;
bf16 0.6B: matching trajectory); (b) memory: params+grads+adam states per gpu
shrink ~linearly in dp_shard (measure: 0.6B adam fp32 states ≈ 4.5GB → ~2.3GB
at shard=2); (c) tps within ~10% of DDP at 0.6B/2 gpus (FSDP overhead is real
at small scale — record it honestly); (d) compile + activation checkpointing
both composing, ablated in CHANGES.md.

## phase 3: distributed checkpointing

do this immediately after fsdp2, before TP — resumable multi-gpu runs are the
prerequisite for every long experiment that follows, and the current
checkpointer doesn't survive contact with sharded state.

- **fix the existing bugs first** (`os.mkdir` is called on the file path, so
  `torch.save` targets a directory; no load path exists).
- **sharded save:** each rank writes its local shards + a metadata file
  (param name → global shape, placements, mesh dims). one file per rank per
  step directory; `barrier` to fence completion, rank 0 writes a `done`
  marker last (crash-consistent).
- **resharding load:** load a checkpoint saved at dp_shard=N onto dp_shard=M.
  v0: rank 0 reassembles full tensors and each rank slices what it needs
  (fine at ≤1B); v1: index-based reads of only the needed byte ranges.
- **full state:** model, optimizer (states reshard exactly like their
  params), scheduler (`last_epoch` — the known TODO in `_build_scheduler`),
  dataloader state per dp rank (StatefulDataLoader already provides
  `state_dict`), RNG states (cpu + cuda, per rank), step counter, config
  snapshot for provenance.
- **async save** (dump to pinned cpu, write in a background thread) once the
  sync path is trusted.
- decide and document the relationship to `torch.distributed.checkpoint`:
  same on-disk philosophy, own implementation (consistent with principle 1),
  but keep an `export_hf`/`export_full` path that produces a plain
  consolidated `state_dict` for interop.

**exit criteria:** the golden resumption test — train 20 steps, kill, resume
at step 10 on a *different* dp_shard, and the loss from steps 11–20 matches
the uninterrupted run exactly (fp32) / to bf16 noise. this test then runs for
every parallelism added later.

## phase 4: tensor parallel (+ sequence parallel)

first parallelism that puts collectives *inside* the model's forward/backward.

**4.1 autograd-aware collectives.** the collective.py docstring already
reserves this: same signatures, internals swapped to
`torch.distributed._functional_collectives` (compile-friendly) or minimal
`autograd.Function` pairs. the four needed:

| fwd | bwd |
|-----|-----|
| identity | all_reduce (colwise input) |
| all_reduce | identity (rowwise output) |
| all_gather(seq) | reduce_scatter(seq) (SP enter) |
| reduce_scatter(seq) | all_gather(seq) (SP exit) |

these must trace under per-block `torch.compile` — this is the phase that
proves the compile strategy survives distribution.

**4.2 the qwen3 plan.** in `parallelism/tensor_parallel.py` + a per-model
plan (see part 2):
- attention: q/k/v colwise sharded *by head* (q_norm/k_norm are per-head
  RMSNorms so they shard along for free), o_proj rowwise. constraint to
  assert: `num_kv_heads % tp == 0` — qwen3-0.6B has 8 kv heads, so tp ≤ 8
  before kv replication is needed (don't build kv replication until a config
  hits it).
- mlp: gate/up colwise, down rowwise. `ffn_dim = 2730` is not divisible by
  hardware-friendly tp degrees — round it up to a multiple of 64/tp at model
  construction when tp > 1 (config knob, matches what real qwen checkpoints
  do with intermediate sizes).
- embedding + lm_head: vocab-parallel (`Shard(0)` on the vocab dim,
  151936 = 1187·2⁷ so tp up to 128 divides). tied weight means one shard
  serves both.
- **loss parallel:** with a vocab-sharded head, cross-entropy needs the global
  logsumexp: local max → all_reduce(max) → local sum of exp → all_reduce(sum),
  and the target logit gathered from whichever rank owns it. this must be
  designed *together with* the chunked fused-linear-CE (item P-1) — the
  chunked structure already isolates the softmax math, so add the two
  all_reduces inside the per-chunk fp32 region. getting these to compose is
  the difference between TP saving lm_head memory or wasting it.

**4.3 sequence parallel.** norms and residual adds operate on replicated
activations under plain TP; SP shards them along seq between the attention/mlp
regions (all_gather on entry to attention/mlp, reduce_scatter on exit —
exactly the fwd/bwd pairs above). worth doing immediately after TP works: it's
a pure activation-memory win at zero extra comm volume.

**exit criteria:** (a) tp=2 forward logits match single-gpu to fp32 tolerance
on a tiny model, loss trajectory matches at 0.6B bf16; (b) fsdp2(dp_shard) ×
tp 2d run trains correctly (first real 2d mesh test); (c) memory: lm_head +
activation savings measured with/without SP; (d) per-block compile still
capturing one graph.

## phase 5: context parallel (ring attention)

the numerically hairiest phase; budget time for the softmax accounting, not
the comms.

- **sharding:** zigzag/load-balanced layout for causal attention — with cp=g,
  rank i owns seq chunks (i, 2g−1−i) so every rank does equal work despite
  causality. `seq_len % (2·cp) == 0` asserted. rope must use *global*
  position_ids for the local chunks (the cacheable-arange fast path needs a
  per-cp-rank variant).
- **ring attention:** K/V shards circulate via `ring_send_recv` (this is why
  it exists); each step computes partial attention and merges via online
  softmax (running max + rescaled accumulator + lse). problem: `F.sdpa`
  doesn't expose lse — options, in preference order: (1) call
  `torch.ops.aten._scaled_dot_product_flash_attention` directly (returns
  lse), (2) flex_attention with lse output, (3) hand-written blockwise
  attention for the tiny-model correctness path. backward does the reverse
  ring pass. overlap the next send/recv with the current chunk's compute
  (async_op — second place the handle contract pays off).
- **plumbing:** all cp ranks of a dp group consume the *same* batch, each
  keeping its seq shards (mesh-coord dataloader from 0.2 does this); labels
  shard identically; loss is summed over local tokens then
  all_reduced("sum") over cp with the token count.

**exit criteria:** (a) cp=2 matches single-gpu loss on identical data (fp32
tiny model near-exact — this is the test that catches lse bugs); (b)
demonstrated long-context run: seq_len ≥ 16k at 0.6B on 2×A40 that OOMs
without cp; (c) composes with fsdp2 (dp_shard × cp).

## phase 6: MoE + expert parallel

gated on the qwen3-moe model existing (part 2, M4). ep is a *model-specific*
parallelism — build it against the moe arch, not in the abstract.

- **single-gpu moe first:** router (softmax top-k, k=2-ish), N expert SwiGLUs,
  dropless (no capacity factor) v0 with a grouped-GEMM path (torch ≥2.10 has
  `torch._grouped_mm`; fall back to a bmm/loop for correctness), load-balance
  aux loss + router z-loss, logging of per-expert token counts (dead-expert
  detection). validate loss goes down before any distribution.
- **ep:** experts sharded across the "ep" mesh dim; token dispatch =
  `all_to_all` with **uneven splits** — this is the anticipated extension to
  `collective.all_to_all(input_splits/output_splits)`, added now that its
  consumer exists (the docstring's rule). combine on the way back reverses
  the permutation. non-expert params (attention, norms, embeddings) stay on
  the dense path (fsdp over dp), expert params get fsdp over the (dp/ep)
  complement — this param-group split is the fiddly part of composing ep with
  fsdp2, design it on paper first.
- overlap dispatch/combine with shared-expert or attention compute later;
  correctness first.

**exit criteria:** (a) ep=2 matches single-gpu moe loss on identical data;
(b) aux losses logged and balanced (no expert <5% of uniform share after
warmup); (c) fsdp2 × ep composes; (d) scaling table dense-equivalent vs moe
tps.

## phase 7: pipeline parallel

deliberately last: highest scheduling complexity, least value on single-node
gpu counts where fsdp2×tp already fits everything this repo trains. build it
for completeness of the composition story (and because the repo's point is to
build these things).

- **stage split:** `model.blocks` is already the unit; stage 0 takes embed +
  blocks[:k], last stage takes blocks[k:] + norm + head. tied embeddings
  across stages: either untie for pp (llama3 path makes this natural) or
  all_reduce the shared grad between first/last stage — decide per model
  plan.
- **p2p activations:** send/recv between stage neighbors (generalize
  `ring_send_recv` to directed `send_recv(dst/src)`); fixed (shape, dtype)
  protocol per microbatch, asserted at init.
- **schedules:** GPipe fill-drain v0 (simple, correct, memory-hungry) →
  1F1B v1 (the real one) → interleaved/virtual stages only if multi-node ever
  happens. microbatch count = config, must divide global batch.
- loss/metrics live on the last stage; broadcast scalars to rank 0 for
  logging.

**exit criteria:** pp=2 (GPipe then 1F1B) matches single-gpu loss;
bubble-fraction measured and matching theory ((p−1)/(m+p−1)); pp × dp
composes.

## phase 8: full composition + config schema

mostly integration and validation once phases 0–7 exist:

- config gains a `"parallelism"` block:
  `{"dp_replicate": 1, "dp_shard": 4, "tp": 2, "cp": 1, "ep": 1, "pp": 1}`;
  trainer's `_parallelize` becomes a single call:
  `parallelize(model, mesh, plan, cfg)` applying, in order: pp split → tp/sp
  shard → cp wrap → moe/ep → fsdp2 → compile → activation checkpointing.
  (order matters and is a doc-worthy invariant: fsdp wraps last so it shards
  whatever the others left local.)
- a `validate(cfg, model_cfg, world_size)` pass that turns every divisibility
  and sizing rule accumulated above (heads % tp, kv_heads % tp, seq % 2cp,
  vocab % tp, product of dims == world_size, global_batch = micro × accum ×
  dp) into loud errors at startup instead of NCCL hangs at step 0.
- golden 3d test: dp_shard=2 × tp=2 × cp=2 on 8 gpus (or gloo/cpu tiny model)
  reproducing the 1-gpu curve. this test is the repo's crown jewel; wire it
  first, then debug until green.

---

# part 2 — model architectures

runs as a parallel track; each item unblocks or exercises a parallelism phase.

**M1 — model contract (before llama3).** small ABC or protocol, enforced by
`models/builder.py`:
- `init_weights()` (residual-scaled, as qwen3 does now);
- `blocks: nn.ModuleList` — the compile/FSDP/AC/PP unit (principle 4);
- `forward(input_ids, attention_mask=None, return_final_hidden=False)` —
  the final-hidden path is what the fused-CE objective consumes, and it keeps
  `lm_head` intact for eval/generation;
- a config dataclass per model (json still the surface format);
- `parallel_plan()` or a registry entry in `torchure/parallelism/plans/`
  mapping module names → tp styles (colwise/rowwise/vocab), fsdp units, pp
  cut points, moe modules. plans live *next to the parallelism code*, keyed
  by model name — models stay parallelism-agnostic, plans stay model-aware.

**M2 — llama3 (fill the stub).** structurally qwen3 minus qk-norm, plus
untied embeddings and different ffn sizing/rope base. its real job is proving
M1 isn't qwen-shaped: no code in trainer/parallelism may mention a concrete
model class. cheap once M1 exists.

**M3 — HF checkpoint import.** state-dict key mapping for qwen3/llama3 →
load real pretrained weights. two payoffs: (a) the strongest correctness test
available — logits vs `transformers` on the same inputs to bf16 tolerance,
catching rope/norm/gqa bugs that random-init training hides; (b) unlocks
finetuning/SFT as a supported workflow, which is what makes the repo useful
beyond pretraining toys. inverse direction (`export_hf`) comes with phase 3's
consolidated export.

**M4 — qwen3-moe.** dense qwen3 + router/experts per phase 6. keep the dense
block class shared; only the mlp swaps.

**M5 — generation + eval.** needed by M3 verification, DLLM sampling, and
any real finetune: KV cache in the attention module (the padded/mask path
already exists), greedy/top-p sampler, a small eval loop (held-out val loss
per N steps, rank-0 only at first, dp-sharded later). deliberately minimal —
no serving ambitions.

**M6 — masked-diffusion LM (DLLM).** the objective abstraction in
`torchure/objectives/` was built for this moment:
- model: needs bidirectional attention — `is_causal=False` path through sdpa
  (the mask plumbing exists; add a `causal: bool` to the model config);
- objective: sample masking ratio t per sequence, mask tokens, CE on masked
  positions only, loss reweighted by 1/t (LLaDA-style). lands as
  `objectives/masked_diffusion.py` behind the existing builder — trainer
  unchanged, which is the test of the abstraction;
- dataloader: same packing machinery, no label shifting; masking happens in
  the objective (gpu-side) so the loader stays objective-agnostic;
- sampling: iterative unmasking loop (uses M5's machinery);
- parallelism note: everything except cp works unchanged; cp's causal zigzag
  balancing becomes plain contiguous sharding for bidirectional attention.

**M7 — continuous-diffusion LM (stretch/research).** embedding-space
diffusion. keep as an explicit maybe: it stresses the objective abstraction
(loss in embedding space, model predicts noise/x0) but has no parallelism
implications beyond M6. don't design for it until M6 works.

---

# part 3 — trainer hardening (parallel track)

**T1 — real `train()` loop.** replace `train_n_step_test`: configured
total_steps, grad accumulation (interacts with 1.3), grad clipping, wsd
scheduler already there, rank-0 structured logging (jsonl file + stdout;
wandb strictly optional), tps + **MFU** (flops model for dense + moe — the
DEV_LOG "mfu so low :(" deserves a real number), grad-norm/param-norm/lr
logged every step, loss-spike flagging.

**T2 — resumption end-to-end.** wire phase 3's checkpointer into the loop
(save every N steps, `--resume` path), including scheduler `last_epoch`,
dataloader state, RNG. the golden test from phase 3 runs in CI-mode (gloo,
tiny model) forever after.

**T3 — launch + ops hygiene.** `scripts/` gets torchrun launchers
(single-node first, rendezvous-based multi-node later), NCCL env defaults,
process-group timeout + `TORCH_NCCL_TRACE_BUFFER` notes for debugging hangs,
graceful SIGTERM checkpoint-and-exit (preemptible-friendly).

**T4 — config validation.** phase 8's validate pass, plus schema defaults so
configs stay short. keep json.

---

# part 4 — performance backlog (ongoing, CHANGES.md-driven)

- **P-1: re-land chunked fused-linear-CE.** the round-2 CHANGES.md entry
  documents the design and results (+3.5% tps, −20% peak mem) and the test
  exists, but the implementation isn't in the tree. re-land behind
  `fused_linear_ce: true`, then keep it in mind through phase 4 (loss
  parallel must compose with the chunked path).
- **P-2: document-masked attention.** packing currently lets tokens attend
  across document boundaries (mask removed in the Jun-18 devlog for speed).
  fix with flex_attention block-causal-with-doc-ids or varlen flash — also
  the natural vehicle for cp's lse needs (phase 5) and DLLM masking (M6).
  measure quality effect at 0.6B before making it default.
- **P-3: comm/compute overlap audits.** after each parallelism phase, a
  profiler trace goes in CHANGES.md with exposed-comm %. targets: DDP <5%
  exposed, fsdp2 prefetch hiding all-gathers, ep dispatch overlapped.
- **P-4: fp8 (torchao) — stretch.** optional dep, per-block compile
  composes; only after bf16 distributed is boring.
- **P-5: benchmark suite.** `scripts/bench.sh` producing the README table:
  {1,2,4,8}×{A40,H100} × {ddp, fsdp2, +tp, +cp} at fixed global batch.
  fills the empty H100 row while it's at it.

---

# sequencing

```
phase 0 (foundations)
  ├─► phase 1 (DDP) ─► phase 2 (FSDP2) ─► phase 3 (dist ckpt) ─► phase 4 (TP/SP) ─► phase 5 (CP)
  │                                                                    │
  │        M1 (contract) ─► M2 (llama3) ─► M3 (HF import) ─► M5 (gen/eval)
  │                                            M4 (qwen3-moe) ─► phase 6 (EP)
  │                                                                    │
  │        T1 (train loop) ─► T2 (resume)                    M6 (DLLM)─┘
  │
  └─────────────────────────────► phase 7 (PP) ─► phase 8 (composition)
```

rough waves, each ending in a CHANGES.md entry with parity + scaling numbers:

1. **wave 1:** phase 0 + T1 + M1 — collectives green, mesh, real train loop.
2. **wave 2:** phase 1 + phase 2 — DDP then FSDP2; first multi-gpu numbers.
3. **wave 3:** phase 3 + T2 + M2 + M3 — durable runs, second architecture,
   HF-verified correctness.
4. **wave 4:** phase 4 + P-1 + P-2 — TP/SP with loss parallel over fused CE.
5. **wave 5:** phase 5 + M5 — CP + long context; generation/eval.
6. **wave 6:** M4 + phase 6 — MoE + EP.
7. **wave 7:** M6 — DLLM training through the same trainer.
8. **wave 8:** phase 7 + phase 8 — PP and the full n-d composition test.

hardware reality: waves 1–4 are testable on gloo/cpu + 2 gpus; waves 5–6 want
4–8 gpus for meaningful numbers (correctness still fine on 2/gloo); wave 8's
8-gpu golden test can run tiny-model-on-cpu until hardware shows up.

# non-goals

- inference serving / deployment (generation exists only for eval + sampling);
- RL post-training (SFT via M3 is in scope; PPO/GRPO machinery is not, until
  the pretraining stack is done);
- framework generality: no plugin systems, no yaml DSLs, no HF Trainer
  compat — json config + registries only;
- uneven-shard padding, elastic world sizes, fault-tolerant collectives —
  assert loudly and keep moving;
- non-NVIDIA backends (gloo is for tests, not training).
