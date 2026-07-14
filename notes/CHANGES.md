# Optimization Changes

Tracking throughput optimizations to the training stack. Metric: **TPS**
(tokens/sec) reported by `uv run torchure/train/trainer.py` (steady-state, i.e.
ignoring step 0 warmup). Config: `configs/qwen3_dense_climbmix.json`
(qwen3 0.6B dense, seq_len=4096, batch_size=2 => 8192 tok/step). GPU: A40.

Focus: single-GPU throughput now, but **prefer changes that also carry over to
distributed** (per-block compile, kernel-level wins, RoPE/norm fixes) over
single-GPU-only tricks.

## Results

| config | steady TPS | vs baseline |
|--------|-----------|-------------|
| baseline (eager) | ~8650 | — |
| **+ per-block compile (default)** | **~12720** | **+47%** |
| + `compile_mode=max-autotune-no-cudagraphs` | ~12940 | +50% |

Loss trajectory is unchanged (12.16 -> 9.22 over 10 steps, matches the eager
baseline's 12.20 -> 9.25), so the speedup is free of numerical regressions.

Baseline was ~0.95 s/step, ~20% MFU. Post-compile ~0.64 s/step.

Notable pre-existing issues spotted:
- RMSNorm dtype mismatch warning (bf16 input vs fp32 weight -> no fused kernel).
- `torch.compile` commented out in trainer.
- RoPE cos/sin recomputed in fp32 every forward though identical each step.
- No TF32 / matmul precision configured.

---

## Changes

### 1. TF32 + matmul precision (neutral, kept)
`torch.backends.cuda.matmul.allow_tf32`, cudnn tf32, `set_float32_matmul_precision("high")`.
No measurable change (~8650) because bf16 autocast already handles the big
matmuls; kept as good practice / helps the fp32 leftovers.

### 2. Per-block `torch.compile` (~8650 -> ~12750, +47%)
Re-enabled compile as `_compile()`, compiling each `Qwen3TransformerBlock`
individually instead of the whole model. Same granularity FSDP2 uses: one graph
captured and reused across all identical blocks (fast warmup), composes with
per-block FSDP/activation-checkpointing later. Gated by `config["compile"]`
(default on). Step-0 pays a one-time compile cost.

### 3. RoPE cos/sin caching (measured +0.26%, ~noise)
`Qwen3RotaryEmbedding.forward` now caches cos/sin per `(seq_len, device)` for
the packed/no-mask training path (position_ids == arange(seq_len), identical
every step). Padded/eval path (mask present) still recomputes. Loss unchanged.

Ablated properly (30 steady steps, 2 replicates each, cache on vs. forced
recompute):

| mode | mean TPS (2 runs) | mean step |
|------|-------------------|-----------|
| cache on  | ~12711 | 644.5 ms |
| cache off | ~12678 | 646.2 ms |

within-run std ~25-40 TPS. So it's a **real but negligible ~0.26% (~1.7 ms/step)**
-- cache-on won both replicate pairs (directionally consistent, not zero), but
the effect is a rounding error on throughput. Kept for cleanliness (removes
redundant per-step recompute + a fp32 `(B,1,S,head_dim)` alloc from the eager
region), NOT as a throughput optimization. Fine to drop with ~no cost.

### 4. `compile_mode` config knob (opt-in, ~+2%)
`_compile` reads `config["compile_mode"]` (default `"default"`).
`"max-autotune-no-cudagraphs"` autotunes the block GEMMs for ~12940 vs ~12720
TPS, but warmup is much longer and that cost multiplies once every rank
compiles in the distributed setting, so it's off by default.

---

## Investigated but NOT adopted

- **Whole-model compile** (single graph incl. embed + lm_head): ~12570 TPS,
  *slower* than per-block (~12720) and worse for distributed (a per-block graph
  composes with FSDP2 wrapping / activation checkpointing; one giant graph does
  not). Rejected.
- **Fusing lm_head + cross-entropy into the compiled region**: the head GEMM
  (1024x151936) and the softmax reduction don't fuse into one kernel anyway, and
  the whole-model-compile test already included the head with no gain. A real
  win here needs a *chunked* fused-linear-CE (Liger-style) that avoids
  materializing the full logits — noted as future work, deliberately skipped for
  now as a mostly single-GPU-memory optimization.
- **TF32 flags**: no throughput change (bf16 autocast already owns the big
  matmuls) but kept as correct-by-default hygiene.

## Profile snapshot (post-compile, per step ~0.64s)

Where the time goes now (from `torch.profiler`):
- compiled transformer blocks: fwd ~146 ms + bwd ~316 ms (~72%) — real compute,
  flash-attn kernels, already compiled.
- lm_head (tied, uncompiled) fwd ~47 ms + bwd ~37 ms (~13%).
- fused AdamW over ~0.57B params: ~44 ms (~7%) — already fused/foreach.

The remaining headroom is mostly in the transformer matmuls themselves (compute-
bound, good kernels) and the large-vocab head — i.e. it now needs either bigger
per-GPU batch (higher GEMM efficiency; a config/scale decision) or a chunked
fused-linear-CE, both better evaluated alongside the distributed work.

---

# Round 2 (2026-07-02)

Both "future work" items from round 1 done: chunked fused-linear-CE and the
batch-size scan it unlocks. Same setup (A40, qwen3 0.6B, seq_len=4096,
`configs/qwen3_dense_climbmix.json`), steady-state TPS from
`uv run torchure/train/trainer.py`.

## Results

| config (batch_size=2 unless noted) | steady TPS | peak mem | vs round-1 best |
|--------|-----------|----------|-----------------|
| round-1 best (per-block compile, unfused CE) | ~12730 | 29.7 GiB | — |
| + fused linear CE, chunk=1024 (fp32 chunk grad_W) | ~12100 | 20.5 GiB | −5.0% |
| + fused linear CE, chunk=1024 (bf16 chunk grad_W) | ~12460 | 20.2 GiB | −2.1% |
| + fused linear CE, chunk=2048 | ~12960 | 21.4 GiB | +1.8% |
| **+ fused linear CE, chunk=4096 (adopted)** | **~13180** | **23.7 GiB** | **+3.5%** |
| + fused linear CE, chunk=8192 (single chunk) | ~13230 | 28.1 GiB | +3.9% |
| + compiled final norm (adopted, on top of chunk=4096) | ~13180 | 23.7 GiB | +3.5% |
| chunk=4096 @ batch_size=4 (measured, not adopted) | ~14150 | 34.5 GiB | +11% |

Loss trajectory unchanged (~12.1 -> ~9.3 over 10 steps in every configuration;
differences are bf16 noise). Net adopted: **~12730 -> ~13180 TPS (+3.5%) and
29.7 -> 23.7 GiB peak (−20%)** at identical training semantics.

## Changes

### 5. Chunked fused linear + cross-entropy (+3.5% TPS, −20% peak mem)
`torchure/loss/fused_linear_ce.py`, wired through `ARObjective`
(`fused_linear_ce: true`, `ce_chunk_size: 4096` in the objective config;
fallback to the old logits path with `fused_linear_ce: false`).

The old path materialized full logits (B,S,151936) in bf16 (2.5 GB), upcast
them to fp32 inside `F.cross_entropy` (5 GB), and saved logits-sized state for
backward. The new path flattens to N=B*S rows and, per chunk: computes the
chunk's logits GEMM, its loss contribution (fp32 logsumexp), and — because
d(loss)/d(logits) = softmax − onehot is closed-form — the grads w.r.t. hidden
and lm_head.weight *in the forward pass* (Liger-style). Backward just rescales
the stashed grads. Same GEMM count as the unfused path (3), so it's not slower,
and nothing logits-sized outlives one chunk. Per-chunk softmax math is
`torch.compile`d (one graph, all chunks same shape).

Correctness: `tests/fused_linear_ce.py` checks loss + both grads against
lm_head + `F.cross_entropy` (fp32 reference), incl. ignore_index, uneven
chunks, and the real (2,4096,1024,151936) shape. Loss matches to 6 decimals;
grads within bf16 tolerance. Tied-embedding grad accumulation (token_emb +
lm_head share the weight) is exercised by the real training run: loss curve
matches the unfused path.

Tuning notes (why the table looks like that):
- chunk grad_W in bf16, accumulated into an fp32 buffer, instead of casting
  each chunk's grad_W to fp32 first: the explicit cast burned a full (V,D)
  fp32 read+write per chunk (~+360 TPS from this alone).
- chunk=4096 over 1024/2048: fewer passes over the 622 MB fp32 grad_W
  accumulator dominate the tradeoff; 8192 (no chunking) buys only +0.4% more
  for +4.4 GiB, so 4096 is the knee. chunk_size keeps loss-side memory
  *constant in batch size*, which is the property that matters under FSDP.

Distributed relevance: this is primarily a per-GPU activation-memory win
(-6 GiB), i.e. microbatch headroom under FSDP2, and it removes the largest
uncompiled eager region. It composes with per-block compile and does not touch
parallelism-facing structure (model still exposes plain `lm_head` for eval /
generation; the fused path is objective-side only, via
`model(..., return_final_hidden=True)`).

### 6. Compile the final RMSNorm (neutral TPS, fixes eager fallback)
`model.norm.compile()` alongside the per-block compile in `_compile`. Kills the
round-1 "RMSNorm dtype mismatch (bf16 input, fp32 weight -> no fused kernel)"
warning; TPS-neutral within noise. Kept: free, block-granular, composes with
FSDP2.

### 7. Measurement fixes (no perf effect)
- `train_n_step_test` computed TPS with a hardcoded batch size of 2; now reads
  `data.batch_size` from config.
- prints `torch.cuda.max_memory_allocated()` at the end of the run, since the
  fused-CE work is as much a memory optimization as a speed one.

## Measured but deliberately NOT adopted

- **batch_size=4**: ~14150 TPS (+11%), 34.5 GiB, fits comfortably now that the
  logits are gone. Not adopted because global batch is a training
  hyperparameter, not a free throughput knob — but this is the per-GPU
  microbatch headroom available when the distributed work picks a batch plan.
  batch_size=6+ would exceed the A40 at seq_len=4096 without activation
  checkpointing.
- **ce_chunk_size=8192** (i.e. unchunked): +0.4% TPS for +4.4 GiB. Wrong side
  of the memory/speed tradeoff, and the advantage shrinks as batch grows.

## Where the time goes now (step ~620 ms @ bs=2)

- compiled transformer blocks fwd+bwd: ~460 ms (~74%) — compute-bound flash +
  GEMM kernels.
- fused linear-CE region: ~125 ms (~20%) — 3 head GEMMs (2.55 TFLOP each) +
  softmax passes; already within ~2x of A40 bf16 peak on the GEMMs.
- fused AdamW: ~44 ms (~7%).

Everything is accounted for; further single-GPU gains are kernel-level
(max-autotune already gives ~+2% as an opt-in) or batch-size scaling. The
sensible next lever is the distributed work itself.

