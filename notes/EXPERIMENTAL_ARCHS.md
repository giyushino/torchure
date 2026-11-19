# experimental architectures: masked diffusion, looped, vlm

three model families on top of qwen3, on branch `claude/experimental-arch`.
the question was how much code each needs beyond its model file. the answer:
diffusion is a config flag plus an objective, looped is one file, and the vlm
is the one that reaches into the dataloader.

| arch | config | new code | touches outside its own files |
|---|---|---|---|
| masked diffusion lm | `qwen3_diffusion_climbmix.json` | `objectives/masked_diffusion.py`, `inference/diffusion_generate.py` | qwen3 `causal` flag; per-token `weights` in the fused loss |
| looped transformer | `looped_qwen3_climbmix.json` | `models/looped/looped.py` | qwen3 `transformer_blocks()` / `run_block()` hooks, `depth` for the init scale |
| qwen3 + siglip vlm | `qwen3_vl_coco.json` | `models/vlm/{siglip,vlm}.py`, `dataloader/vlm.py` | qwen3 `inputs_embeds`; `data.image_field`; objectives pass extra batch keys to the model; trainer hands the model's image spec to the loader; `pillow` extra |

## design

**diffusion** (MDLM / LLaDA). `causal: false` switches sdpa to bidirectional
attention, on both the flash path and the padding-mask path. the objective
draws t ~ U(eps, 1) per row, masks each token with probability t (never an
`ignore_index` position), and returns `sum over masked (1/t) * nll / num_tokens`,
with no next-token shift. `[MASK]` is id 151669, the first unused slot of
qwen's padded 151936 vocab, so no embedding resize is needed. the fused loss
gained an optional per-token weight on both the single-device and the
vocab-parallel path; it normalizes by masked tokens, so the objective folds
`num_masked / num_tokens` into the weights. the sampler uses llada's
low-confidence remasking (reveal the most confident `gen_len / steps`
predictions per step).

**looped** (prelude / core x num_loops / coda). the core is a single module in
`model.blocks`, so it is one fsdp unit (gathered once, used `num_loops` times)
and one compile graph (the loop is unrolled). the model registers it only
through `blocks`; a second `self.core` attribute would put every core weight
in the state_dict twice. activation checkpointing runs per block application
inside the core; checkpointing the whole core would rematerialize every
iteration at once. the residual init scale uses the unrolled depth.
`model.core.num_loops` can be changed after training.

**vlm** (llava-style). siglip-base tower (module names match hf
`SiglipVisionModel`, so pretrained weights are a key rename), a 2-layer mlp
projector with 2x2 pixel shuffle (196 patches → 49 tokens), and
`masked_scatter` of the features into the `<|image_pad|>` placeholders.
qwen's own `<|vision_start|>` / `<|image_pad|>` / `<|vision_end|>` tokens are
used. data: one image per row, no packing, right-padded; only caption + eos
are scored. COCO captions (`jxie/coco_captions`, 182 parquet train shards)
stream through the existing hf path.

## verification

cpu (`tests/experimental_archs.py`, in ci; 11 tests):
- bidirectional: a future token changes earlier logits iff `causal=False`; the
  mask path equals the flash path on a full mask
- diffusion noise/loss: masks only on valid positions, masked fraction tracks t,
  loss equals the hand-written weighted nll; the sampler fills every mask at
  `gen_len / steps` per step
- looped: equals a flat qwen3 whose block list repeats the core (same
  weights); no aliased state_dict entries; checkpointing gives identical
  loss/grads; meta build → materialize → init works
- vlm: changing one image moves only its own row, and only from the image span
  on; grads reach every vision/projector param; collator layout on real
  tokenizer + PIL images
- 2-rank gloo fsdp: looped and vlm grads match single-process over 2 cycles;
  weighted fused loss on the vocab-parallel path matches the reference

also: `tests/fused_linear_ce.py` (cuda) with weights, up to the real
4096 x 151936 shape, rel grad err ≤ 3.3e-3. the full existing tp suite (14
scenarios: ddp/fsdp x tp x vocab/sp/overlap, act ckpt, training, checkpoint,
meta init) passes with the looped model swapped in (scratch harness, not
committed). all existing cpu suites still pass.

gpu smoke, 1x H100 NVL, `--steps N`, compile on, bf16, ClimbMix seq 4096 mb 2
(vlm: COCO, seq 128 mb 64). tps is the steady state (steps 20–40):

| run | params | loss step 10 → last | tok/s | peak GiB |
|---|---|---|---|---|
| qwen3 AR baseline | 574M | 10.55 → 7.98 (40) | 48,900 | 22.4 |
| diffusion (`causal: false`) | 574M | 10.35 → 8.06 (40) | 38,900 | 22.4 |
| looped 2 + 4x6 + 2 (depth 28) | 275M | 11.20 → 7.96 (40) | 50,200 | 18.8 |
| looped + activation ckpt | 275M | 11.20 → 8.03 (30) | 41,200 | 10.6 |
| vlm (siglip-b + qwen3 0.6B) | 664M | 8.96 → 4.93 (60) | 55,800* | 28.0 |

\* counts padded positions and image tokens; about 440 images/s.

- diffusion is ~20% slower than AR at the same shape: bidirectional attention
  can't skip the upper triangle, which is significant at 4096.
- looped matches AR throughput at the same effective depth with 48% of the
  params; the memory saving is optimizer state and weights. its loss here is
  no evidence of quality (40 steps).
- loss going down is a wiring check, not a quality claim. none of these were
  trained long enough to compare against AR.

## 4x H100 NVL, cu130 (11/19)

ddp, dp 4, `configs/*_ddp_h100.json`, the same shape as the AR baseline
(micro 4, global 32, seq 4096; vlm: micro 64, global 256, seq 128). protocol
from TORCHTITAN_COMPARE.md: `--steps 60`, mean tps over steps 21–60, torch
2.10.0+cu130; run-to-run spread is about ±1k.

| run | params | tok/s | vs AR | peak GiB |
|---|---|---|---|---|
| qwen3 AR baseline | 574M | 238,184 | — | 36.9 |
| diffusion | 574M | 199,416 | −16.3% | 36.9 |
| looped (depth 28) | 275M | 248,884 | +4.5% | 31.4 |
| vlm | 664M | 172,258* | — | 29.9 |

\* includes padding and image tokens; ~5.3 steps/s x 256 = ~1,350 images/s.

- AR reproduces the 239.2k baseline, so the shared qwen3 changes cost nothing.
- diffusion's gap shrank from ~20% (1 gpu, cu128) to 16%: the extra
  attention work is a smaller share once ddp comm is in the step.
- looped is faster than AR at the same depth because it all-reduces 48% of
  the gradients.
- vlm scales ~77% from 1 gpu, but that compares a cu128 1-gpu run with a
  cu130 4-gpu run. likely cause: a short step (32k tokens) with a full 664M
  gradient all-reduce each step. not profiled. COCO's 182 shards don't divide
  over 4 ranks, so each rank reads the whole stream; a rank's loader still
  delivers ~1,000 images/s (measured), well above the ~340/s it needs.

### vlm micro batch 64 -> 128

| vlm micro (global) | tok/s | images/s | peak GiB |
|---|---|---|---|
| 64 (256) | 172,258 | ~1,350 | 29.9 |
| **128 (512)** | **216,110** | **~1,690** | 45.1 |

+25%, kept in `qwen3_vl_coco_ddp_h100.json`. the vlm's 32k-token step was
dominated by per-step costs (full 664M-param grad all-reduce, adamw);
doubling the step amortizes them. this is a global batch change (256 -> 512
captions), not only a throughput knob. the text-only archs don't have this
headroom: ddp micro 4 -> 8 on the 0.6B AR model is noise
(TORCHTITAN_COMPARE.md), with gemm + attention ~76% of the step.

## open problems / known gaps

- **vlm + tensor parallel: not supported** (`tp_plan` raises). tp's embedding
  hooks shard the sequence before the image features are spliced in.
- **vlm under fsdp**: the vision tower and projector live in the root unit, so
  they are gathered for the whole step and not compiled. fine at siglip-base
  (86M); a so400m tower wants its blocks as units.
- **vlm data**: one image per row, padded, no packing. ~45% of each 128-token
  row is padding at COCO caption lengths. no pretrained weight loading yet
  (siglip or qwen); trained from scratch the tower learns little in a smoke
  run, and nothing here shows the model uses the image beyond the cpu test.
  a real check is caption loss with shuffled images vs true images.
- **diffusion eval**: `eval_ppl.py` is AR-only and will report a meaningless
  number on a diffusion checkpoint. needs a monte-carlo ELBO over t.
- **diffusion + packing**: packed documents now see each other in both
  directions. llada trains this way, but document masking would be cleaner.
  main's `doc_mask` can't be reused as is: its mask is causal, and it finds
  documents by eos in the input ids, which the objective may have masked.
  the trainer rejects `doc_mask` with `causal: false`. a fix would take the
  document ids from the clean batch and drop the `q >= kv` term.
- **diffusion + cp**: cp is not implemented yet; when it is, the causal
  zigzag balancing is unnecessary for bidirectional attention.
- **looped**: no random-depth training or state injection (Geiping et al.),
  so extrapolating to more loops at inference is not expected to help.
- multi-gpu: only ddp was run on gpus. fsdp and tp for the new models were
  only tested on gloo/cpu.
