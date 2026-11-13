# torchure

a pure pytorch training stack for autoregressive transformer LLMs. no
megatron, deepspeed, accelerate, or liger: the device mesh, collectives, and
parallelism are implemented here and tested against pytorch's own.

## usage

one json config drives a run, from a single gpu up to multi-node data parallel
(ddp, fsdp, hsdp), with no code changes to scale up or down. see `configs/`
for examples and `notes/` for optimization history and the roadmap.

```bash
uv pip install -e .            # add ".[wandb]" for wandb logging

# single gpu
uv run -m torchure.train.train --config configs/qwen3_dense_climbmix.json

# multi gpu
uv run torchrun --nproc-per-node 8 -m torchure.train.train \
    --config configs/qwen3_dense_climbmix_ddp.json

# benchmark mode: N steps, no checkpointing, prints tps + peak mem
uv run torchrun --nproc-per-node 8 -m torchure.train.train \
    --config configs/qwen3_dense_climbmix_ddp.json --steps 20
```

## performance

all numbers are bf16, seq_len 4096, global tokens/sec.

### ddp vs fsdp vs hsdp, 2x8 H100 SXM (infiniband)

cells are tps / mfu (989 TFLOPS peak, `6N + 12·L·H·head_dim·seq` flops/token).
no grad accumulation; mb = micro batch per gpu. hsdp = fsdp within a node,
replicated across nodes. fsdp is torchure's own fsdp2 (`parallelism/fsdp2.py`).

**qwen3 0.6B**

| layout | mb 4 | mb 8 | mb 12 |
|--------|------|------|-------|
| ddp, 8 gpus | 483k / 38.2% | 509k / 40.3% | - |
| fsdp, 8 gpus | 483k / 38.2% | 513k / 40.6% | 528k / 41.8% |
| ddp, 16 gpus | 928k / 36.7% | 989k / 39.1% | - |
| fsdp, 16 gpus | 782k / 30.9% | 990k / 39.1% | **1.06M / 41.9%** |
| hsdp, 2x8 gpus | 753k / 29.8% | 902k / 35.7% | 974k / 38.5% |

**qwen3 4B**

| layout | mb 1 | mb 2 | mb 3 | mb 4 |
|--------|------|------|------|------|
| ddp, 8 gpus | 84k / 33.4% | - | - | - |
| fsdp, 8 gpus | 97k / 38.6% | 112k / 44.6% | 119k / **47.3%** | oom |
| ddp, 16 gpus | 120k / 23.8% | - | - | - |
| fsdp, 16 gpus | 105k / 20.8% | 160k / 31.6% | 202k / 40.0% | **217k** / 43.0% |
| hsdp, 2x8 gpus | 100k / 19.8% | 148k / 29.4% | 179k / 35.5% | oom |

### single node, qwen3 0.6B, mb 2

| hardware | tps | ddp scaling eff |
|----------|-----|-----------------|
| 1x A40 | ~13.3k | |
| 1x A6000 | ~14.4k | |
| 1x H100 NVL | ~54.5k | |
| 8x A40 | ~73.5k | 69% |
| 8x A6000 | ~86.4k | 60% |
| 4x H100 NVL | ~198k | 92% |

### tensor parallel, 4x H100 NVL (nvlink pairs 0-1 and 2-3)

cells are tps / mfu (835 TFLOPS peak, same flops formula) / peak mem. 50-step
runs, sequence parallel on unless noted. dp x tp layouts put each tp pair on
an nvlink pair. `sp_dim` is what sp shards between regions (`tp.sp_dim`):
"seq" (default) or "batch", which keeps each rank's chunk contiguous so the
collectives need no layout copies (needs micro batch % tp == 0).

**qwen3 0.6B**, global batch 32, no grad accumulation

| layout | mb | result |
|--------|----|--------|
| ddp, 4 gpus | 8 | 198k / 37.1% / 56.2 GiB |
| fsdp, 4 gpus | 8 | **198k / 37.1%** / 47.1 GiB |
| tp 4 | 32 | 83k / 15.5% / 59.6 GiB |
| fsdp 2 x tp 2, no sp | 16 | 172k / 32.2% / 53.3 GiB |

| dp 2 x tp 2, mb 16 | fsdp | ddp |
|--------------------|------|-----|
| sp_dim seq | 162k / 30.4% / 49.8 GiB | 152k / 28.6% / 56.3 GiB |
| sp_dim seq, overlap backward | 157k / 29.5% / 54.7 GiB | 147k / 27.6% / 61.1 GiB |
| sp_dim seq, overlap collective_matmul | 155k / 29.0% / 54.7 GiB | 144k / 26.9% / 61.1 GiB |
| sp_dim batch | **174k / 32.6%** / 49.7 GiB | **162k / 30.5%** / 56.3 GiB |
| sp_dim batch, overlap backward | 171k / 32.2% / 54.5 GiB | 160k / 30.0% / 61.1 GiB |
| sp_dim batch, overlap collective_matmul | 168k / 31.5% / 54.5 GiB | 156k / 29.3% / 61.1 GiB |

**qwen3 4B**, global batch 16

| layout | mb | result |
|--------|----|--------|
| ddp, 4 gpus | 4 | oom |
| fsdp, 4 gpus | 4 | **44.4k / 41.7%** / 82.1 GiB |
| ddp 2 x tp 2 | 8 | oom |
| ddp 2 x tp 2 (2x accum) | 4 | 36.2k / 34.0% / 77.0 GiB |
| fsdp 2 x tp 2, sp_dim seq | 8 | 39.3k / 36.9% / 86.4 GiB |
| fsdp 2 x tp 2, sp_dim batch | 8 | 41.5k / 39.0% / 86.3 GiB |
| fsdp 2 x tp 2, any overlap | 8 | oom (seq + collective_matmul hangs instead) |
| tp 4 | 16 | oom |
| tp 4 (2x accum) | 8 | 23.0k / 21.6% / 63.2 GiB |

takeaways: on one node, fsdp over all gpus beats every tp layout at these
sizes (tp shards model state no further than fsdp already does; it only
helps once one sequence's activations at mb 1 don't fit). `sp_dim: "batch"`
is a free +6-7% for tp. the overlap modes (`tp.overlap`) lose 1-5% and ~5 GiB
here: nvlink leaves little comm to hide. tp 4 spans the pcie hop between the
nvlink pairs.
