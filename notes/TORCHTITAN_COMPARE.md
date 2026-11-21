# torchtitan comparison + perf hill climb (11/19, 4x h100 nvl)

all numbers: aggregate tok/s across 4 gpus, seq 4096, mean of steps 21-60
of a 60-step `--steps` run unless noted (torchtitan: its per-device `tps`
x 4, cross-checked against log timestamps). same box, same night.

## headline

**the cuda build, not the framework or torch version, was the gap.** pypi's
default torch 2.10 wheel is cuda 12.8; the same torch 2.10 built for cuda
13.0 is +22% on 0.6B (195k -> 239k), from newer cublas/cudnn kernels.
at equal attention work, torchure on cuda 13 beats torchtitan main (torch
2.16 nightly, cuda 13.2) on every shape tried: plain causal everywhere, and
with document masking on c4 (270k vs 250k, 0.6B ddp micro 4).

| shape | torchure, cu128 (old pin) | torchure, cu130 (new pin) | best torchure config | torchtitan |
|---|---|---|---|---|
| 0.6B ddp, micro 4 | 195.9k | 239.2k | 242.8k (bf16 reduce) | 213k |
| 0.6B fsdp, micro 4 | 186.4k | 227.1k | 232.1k (bf16 reduce) | 199k |
| 0.6B fsdp, micro 8 | — | 242.7k | **245.5k** (bf16 reduce) | 217.5k |
| 0.6B ddp, micro 8 | — | — | — | 219.3k |
| 4B fsdp, micro 4 (global 16) | 45.5k | 49.5k | **50.2k** (bf16 reduce) | 46.6k |

global batch 32 x 4096 for 0.6B. torchtitan ran its own c4_test data; data
loading is not a bottleneck for either.

## pinning down the cuda effect

0.6B ddp micro 4, torchure unchanged, only the torch wheel swapped:

| torch | cuda | tok/s |
|---|---|---|
| 2.10.0 | 12.8 (pypi default, the old lock) | 195.9k |
| 2.14.1 | 12.6 | 194.8k |
| 2.14.1 | 13.2 | 237.8k |
| 2.10.0 | 13.0 | 238.9k |
| 2.15.0 | 13.2 | 238.4k |
| 2.16 nightly | 13.2 | ~237k (3 log points) |

the torch version is worth nothing here; the cuda major is worth 22%. so the
pin stays `torch==2.10` and pyproject now pulls it from pytorch's cu130 index
(`[tool.uv.sources]`). **cuda 13 needs nvidia driver >= 580** (this box: 610).
on an older driver, delete the `[tool.uv.sources]` block and `uv lock`. ci is
unaffected (`uv pip install --torch-backend=cpu` ignores uv sources).
`uv sync` drops the optional wandb extra: use `uv sync --extra wandb`.

the 4B gain is smaller (+9%): its time is in big gemms that cuda 12.8's
cublas already ran well. kimi 4B is unchanged (~19k at step 20, matching
KIMI_MOE_H100.md): its time is in the kda triton kernels.

## torchtitan setup

main @ 1ae2914, torch 2.16.0.dev20261118+cu132, python 3.12, a scratch
recipe module mirroring its qwen3_14b recipe (`build_model_config("0.6B" |
"4B")`, AdamW, ChunkedLossWrapper CE, no checkpointing, no activation remat,
its default regional compile: loss / activation / rope).

- **attention mismatch.** torchtitan main dropped sdpa for language models;
  every backend is block-causal over per-document positions, and varlen
  needs flash-attn 3 (not installed). on c4's short docs, doc masking skips
  most of attention: 0.6B ddp goes 213k -> **250k** with masking. that's
  less work, not a faster stack, so the table above patches its collator to
  emit positions 0..4095 per row (one "document" per row = plain causal over
  4096 with rope across docs, exactly torchure's semantics).
- torchure has doc masking since 11/19 (`doc_mask`, below): on c4,
  torchure ddp micro 4 with doc_mask does 270k vs torchtitan's 250k.

## hill climb (on cu130): what moved and what didn't

profile of 0.6B ddp micro 4 (`torchure/profiling/overlap.py`), per step: compute
~524ms (gemm 47%, attention 29% — cudnn sdpa bwd alone 112ms — triton
fused elementwise 18%, adamw 2%), nccl 159ms of which 16ms (2.9%) exposed.
single gpu does 64.5k (258k x4), so ddp costs ~8%: ~3% exposed comm, ~3%
compute slowed by overlapping nccl kernels, ~1% idle.

| change | 0.6B tok/s | verdict |
|---|---|---|
| cuda 12.8 -> 13.0 | 195.9k -> 239.2k | **kept** (pin) |
| ddp `reduce_dtype: bfloat16` (new) | 239.2k -> 242.8k | kept as opt-in; exposed comm 16 -> 10.7ms |
| fsdp `reduce_dtype: bfloat16` | 227.1k -> 232.1k (micro 4) | opt-in (FSDP2_H100.md already recommends it) |
| fsdp micro 4 -> 8, same global batch | 232.1k -> 245.5k | **kept** in `qwen3_dense_climbmix_fsdp_h100.json` |
| ddp micro 4 -> 8 | 239.8k | noise |
| ddp bucket 10 / 100 MB | 236.8k / 239.6k | noise |
| NCCL_PROTO=Simple (nccl picked RING_LL) | 238.9k / 241.8k | noise |
| NCCL_MAX_NCHANNELS 2 / 4 / 8 | 238.6k / 240.2k / 241.5k | worse or noise |
| fsdp prefetch 2 | 241.7k | worse |
| compile `max-autotune-no-cudagraphs` | 238.5k | noise |
| whole-model compile (ddp) vs per-block | 243.3k vs 242.8k | noise |
| sdpa flash instead of cudnn | 237.6k | noise |
| ce_chunk_size 8192 / 16384 | 243.7k / 244.5k | noise |

run-to-run spread is ~±1k. everything left is gemm and attention kernel
time at ~40% mfu on a d=1024 model — kernel territory, stopped there.

ddp `reduce_dtype` is tested in `tests/data_parallel.py` (bf16 parity,
tolerance relative to the grad scale: ranks that nearly cancel leave a few
bf16 ulps of error in the average).

## cleanup done alongside

removed empty stubs (`parallelism/context_parallel.py`, `models/llama3/`,
`optimizer/hyperball.py`, the `llama3_2: None` registry entry), unused
`utils.load_json`, ddp's never-read `launched` flag, an unused test mesh
class and two dead test lines. full cpu suite + gpu fused-CE test pass on
the cu130 build.

## document masking (11/19, later)

`"doc_mask": true` (qwen3_dense, olmo3): `layers.doc_mask` turns each row's
eos tokens into doc ids and per-doc positions, builds a flex attention
BlockMask once per forward (one per sliding window size for olmo3), and the
attention calls a compiled `flex_attention` with it. `tests/doc_mask.py`
(ci): a packed row under doc_mask gives the same logits as each doc run
alone (cpu: max diff 1e-6; gpu bf16: ~1.5% of the logit scale), for qwen3
and for olmo3 with windows shorter than the row. the mixture loader's
sources must end docs with `data.eos_token` (checked at startup).

0.6B, 4x h100 nvl, 60 steps, mean of 21-60, same session:

| layout | mb | climbmix causal | climbmix doc_mask | c4 doc_mask |
|---|---|---|---|---|
| ddp | 4 | 238.6k | 264.8k (+11%) | 269.5k |
| fsdp, bf16 reduce | 8 | 245.9k | 276.0k (+12%) | 280.8k |

torchtitan doc-masked (flex, c4, ddp micro 4): 250k. peak memory is
unchanged (37.0 / 47.5 GiB). kimi has no doc_mask support (the trainer
refuses it).

other shapes, climbmix, causal -> doc_mask, same session:

| run | causal | doc_mask | mem (GiB) |
|---|---|---|---|
| 4B fsdp micro 4, global 16, bf16 reduce | 50.7k | 54.1k (+7%) | 82.1 / 82.3 |
| 0.6B fsdp 2 x tp 2, sp_dim batch, micro 16, bf16 reduce | 208.4k | 229.3k (+10%) | 49.7 / 50.5 |
| 0.6B fsdp micro 8 + activation checkpointing | 198.4k | 221.1k (+11%) | 8.0 / 8.0 |
| olmo3 7B fsdp micro 2, global 8, act ckpt (window 4096 = full at seq 4096) | 24.5k | 24.8k (+1%) | 31.5 / 31.6 |
| olmo3 7B, same, sliding_window 1024 | 22.0k | 25.2k (+15%) | 31.6 / 31.6 |

(olmo3 runs are 30 steps, tps at step 30.) the 7B gains little: at
d=4096 attention is a small share of the step. its windowed causal path
builds a dense S x S bool mask for sdpa, so flex's block-sparse window is
the bigger win there. activation checkpointing with doc_mask gives the
same loss to 4 decimals as without it, every logged step through 60.

tp is covered by `tests/tensor_parallel.py` (3 doc-mask scenarios: tp,
fsdp x tp + vocab + sp, batch-sp; forward, grads, act ckpt, training,
checkpoint, meta init all match the single-process model). those run on
cpu, where `masked_attention` falls back to sdpa with the same mask made
dense: compiled flex backward fails on cpu in torch 2.10 (inductor
IndexError, even with no mask). the flex kernel itself is checked on gpu by
`tests/doc_mask.py`.

## nanochat (11/19)

karpathy/nanochat @ 92d63d4, its own env (torch 2.9.1, FA3 via `kernels`),
`base_train` on 4 gpus, 60 iterations, mean of 21-60. its model is not
qwen3 (relu^2 mlp, value embeddings, 32k vocab, muon), so compare mfu, each
from the model's own `6N + 12·L·H·head_dim·seq` flops/token:

| run | flops/token | tok/s | mfu |
|---|---|---|---|
| nanochat d28 x 1024, seq 4096, full attn, batch 32x4096, cu128 | 3.72e9 | 376.1k | 41.9% |
| same, torch 2.9.1 cu130 | 3.72e9 | 378.0k | 42.2% |
| nanochat d24 defaults (seq 2048, SSSL windows, 1M batch) | 4.78e9 | 295.4k | 42.2% |
| torchure qwen3 0.6B ddp micro 4 | 6.39e9 | 239.2k | 45.8% |
| torchure qwen3 0.6B fsdp micro 8, bf16 reduce | 6.39e9 | 245.5k | 47.0% |

nanochat gets nothing from cuda 13 (+0.5%): it runs attention in FA3, not
cudnn sdpa, which supports cudnn's attention kernels being much of
torchure's cu130 gain. its mfu also pays for muon's newton-schulz, which the
flop count doesn't credit.
