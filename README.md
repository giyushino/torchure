# torchure

A training stack for transformer language models, written in plain PyTorch.

Most LLM training code builds on large frameworks: Megatron, DeepSpeed,
Accelerate, or PyTorch's own `torch.distributed` sharding APIs. torchure
implements those pieces itself: the device mesh, the collectives, and data,
fully sharded, tensor and expert parallelism. Each one is small enough to
read, and each is tested against PyTorch's own version. The only runtime
dependencies are `torch`, `numpy`, `tokenizers` and `pyarrow`.

It's meant for:

- **learning how distributed training works.** You can follow a gradient
  from the backward pass through the bucketed all-reduce or reduce-scatter,
  without digging through framework layers.
- **training small-to-mid-size models** (0.6B–7B) on anything from one GPU
  to several nodes, with the same code and only a config change.

## What's included

| | |
|---|---|
| **Models** | Qwen3 (dense), OLMo 3 (sliding-window attention), and a Kimi-style mixture-of-experts model with linear attention |
| **Experimental** | A masked-diffusion LM (`qwen3_diffusion_climbmix.json`, sampled with `torchure.inference.diffusion_generate`), a looped transformer (`looped_qwen3_climbmix.json`), and a Qwen3 + SigLIP VLM on COCO captions (`qwen3_vl_coco.json`, needs `uv sync --extra vision`). See `notes/EXPERIMENTAL_ARCHS.md` |
| **Parallelism** | DDP, FSDP (a from-scratch FSDP2), HSDP, tensor + sequence parallel, expert parallel, freely combined on one device mesh |
| **Data** | Streams datasets straight from the Hugging Face Hub, HTTP or local disk (jsonl, parquet, pre-tokenized shards) and tokenizes on the fly, so there's no preprocessing step. Weighted mixes of several datasets whose weights can change over training |
| **Training** | bf16, fused linear + cross-entropy loss, `torch.compile`, activation checkpointing, document masking, WSD/cosine schedules, optional wandb logging |
| **Checkpoints** | Sharded and async. They resume on a *different* number of GPUs or a different layout, and a job that gets SIGTERM saves before it exits |
| **Inference** | Export any checkpoint to a single `state_dict`, measure perplexity, or run a simple text-continuation REPL |

## Quick start

You need Python 3.14, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU.

```bash
git clone <this repo> && cd torchure
uv sync                      # or: uv sync --extra wandb
```

> **Driver note:** torch is installed from PyTorch's CUDA 13.2 index, which
> needs NVIDIA driver ≥ 580. On an older driver, delete the
> `[tool.uv.sources]` block in `pyproject.toml` and run `uv lock` to get
> PyPI's default build. A CUDA 12 build works the same, just about 20%
> slower on H100.

Train a 0.6B Qwen3 on one GPU. Data streams from the Hub, so there's
nothing to download first:

```bash
uv run -m torchure.train.train --config configs/qwen3_dense_climbmix.json
```

Same model, 8 GPUs, data parallel:

```bash
uv run torchrun --nproc-per-node 8 -m torchure.train.train \
    --config configs/qwen3_dense_climbmix_ddp.json
```

Add `--steps 20` to either command for a quick benchmark: it runs 20 steps,
skips checkpointing, and prints throughput and peak memory.

After training, use a checkpoint:

```bash
# perplexity on fresh batches
uv run -m torchure.inference.eval_ppl --config <config> --step <step>
# interactive text continuation
uv run -m torchure.inference.chat --config <config> --step <step>
# one plain state_dict file, however the checkpoint was sharded
uv run -m torchure.inference.export --help
```

## Configs

Each run is described by one JSON file. A few things to know:

- **Configs inherit.** `"base": "other.json"` starts from another config and
  overrides parts of it, so most files in `configs/` only list what differs
  from `qwen3_dense_climbmix.json`.
- **Configs are validated.** Every config is checked against the schema in
  `torchure/config.py` before anything is built. Unknown keys are errors, and
  optional keys get their defaults there.
- **`mesh` picks the parallel layout.** Its axes are `dp_replicate`,
  `dp_shard` (or just `dp`), `tp` (tensor parallel) and `ep` (expert parallel,
  for MoE models). `"parallelism": "ddp"` or `"fsdp"` says how the data
  parallel axes hold the model.
- **`data` picks the training data.** Use either `source` (one dataset) or
  `mix` (a Python file of weighted sources, like `mixes/example_mix.py`).

Starting points:

| config | what it runs |
|---|---|
| `qwen3_dense_climbmix.json` | Qwen3 0.6B, single GPU |
| `qwen3_dense_climbmix_ddp.json` | the same, data parallel |
| `qwen3_dense_climbmix_fsdp_h100.json` / `_hsdp_h100.json` / `_tp_h100.json` | FSDP, hybrid sharding, tensor parallel |
| `qwen3_dense_mix_smoke.json` | a two-dataset mix whose weights change over training |
| `olmo3_7b_dolma.json` | OLMo 3 7B |
| `kimi_k31_4b_climbmix_fsdp_ep.json` | 4B mixture-of-experts with FSDP + expert parallel |

## Performance

All numbers are bf16, sequence length 4096, global tokens/sec.

**vs torchtitan** (PyTorch's reference trainer), 4× H100 NVL, Qwen3 with
plain causal attention:

| model, layout | torchure | torchtitan |
|---|---|---|
| 0.6B, DDP | 243k | 213k |
| 0.6B, FSDP | 246k | 218k |
| 4B, FSDP | 50.2k | 46.6k |

**Scaling** to 2 nodes × 8 H100 SXM (InfiniBand), FSDP:

| model | 8 GPUs | 16 GPUs | MFU at 16 |
|---|---|---|---|
| Qwen3 0.6B | 528k | 1.06M | 41.9% |
| Qwen3 4B | 119k | 217k | 43.0% |

**8× B300** (one node, Modal), Qwen3 4B with document masking, 40-step
runs with tok/s averaged over steps 21–40:

| layout | micro batch | tok/s | peak mem / GPU | MFU |
|---|---|---|---|---|
| FSDP | 8 | 245k | 141 GiB | 42.8% |
| FSDP | 12 | 250k | 206 GiB | 43.6% |
| DDP | 8 | 238k | 188 GiB | 41.5% |

That is about 31k tok/s per GPU, 2.3× an H100 NVL on the same model
(54.1k on 4 GPUs with document masking). MFU is against 2,250 TFLOP/s dense
bf16 and counts full causal attention, which document masking partly skips,
so it reads a little high. Run with `scripts/modal_launch.py` and
`scripts/sweeps/b300.json`.

**Document masking** (`"doc_mask": true`) stops tokens from attending across
packed documents. On short-document data it's also faster: 0.6B FSDP goes
from 246k to 276k tok/s. It changes what the model learns, though, so loss
curves aren't comparable with plain causal runs.

The old README (`readme_old.md`) has the full tables, and `notes/` has
the write-ups: the torchtitan comparison, DDP/FSDP/TP sweeps on several GPU
types, the MoE work, and the data loader design.

## Repository layout

```
torchure/
  core/          device mesh, collectives, a minimal DTensor, grad clipping
  parallelism/   ddp, fsdp2, tensor parallel (+ comm overlap), expert parallel
  models/        qwen3, olmo3, kimi_k31, shared layers and moe
  dataloader/    streaming sources, mixing, packing, prefetch
  checkpoint/    sharded, async, reshardable checkpoints
  loss/          cross entropy and fused linear + cross entropy
  train/         entrypoint, trainer
  profiling/     comm / compute overlap profiler
  inference/     export, perplexity, chat
configs/         run configs
mixes/           data mix scripts
tests/           standalone test scripts
notes/           design notes, benchmarks, roadmap
```

## Tests

Each test is a standalone script. Most run on CPU with the gloo backend, so
they don't need GPUs:

```bash
uv run tests/fsdp2.py --world-size 4
uv run tests/tensor_parallel.py
uv run tests/sharded_checkpoint.py
```

`.github/workflows/ci.yml` has the full list CI runs.

## License

MIT. See [LICENSE](LICENSE).
