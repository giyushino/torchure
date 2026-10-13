# torchure

pure pytorch training stack for autoregressive transformer LLMs. no
megatron/deepspeed/accelerate/liger — the mesh, collectives, and parallelism
are implemented here and tested against pytorch's own.

## intended use

one json config drives a run from a single gpu up to multi-node data parallel
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

### single node, qwen3 0.6B, batch 2/gpu

| hardware | tps | ddp scaling eff |
|----------|-----|-----------------|
| 1x A40 | ~13.3k | |
| 1x A6000 | ~14.4k | |
| 1x H100 NVL | ~54.5k | |
| 8x A40 | ~73.5k | 69% |
| 8x A6000 | ~86.4k | 60% |
| 4x H100 NVL | ~198k | 92% |

### ddp vs fsdp vs hsdp, 2x8 H100 SXM (infiniband)

cells are tps / mfu (989 TFLOPS peak, 6N + 12·L·H·head_dim·seq flops/token).
no grad accumulation; mb = micro batch per gpu. hsdp = fsdp within a node,
replicated across nodes. fsdp is torchure's own fsdp2 (branch `claude_fsdp`).

qwen3 0.6B:

| layout | mb 4 | mb 8 | mb 12 |
|--------|------|------|-------|
| ddp, 8 gpus | 483k / 38.2% | 509k / 40.3% | - |
| fsdp, 8 gpus | 483k / 38.2% | 513k / 40.6% | 528k / 41.8% |
| ddp, 16 gpus | 928k / 36.7% | 989k / 39.1% | - |
| fsdp, 16 gpus | 782k / 30.9% | 990k / 39.1% | **1.06M / 41.9%** |
| hsdp, 2x8 gpus | 753k / 29.8% | 902k / 35.7% | 974k / 38.5% |

qwen3 4B:

| layout | mb 1 | mb 2 | mb 3 | mb 4 |
|--------|------|------|------|------|
| ddp, 8 gpus | 84k / 33.4% | - | - | - |
| fsdp, 8 gpus | 97k / 38.6% | 112k / 44.6% | 119k / **47.3%** | oom |
| ddp, 16 gpus | 120k / 23.8% | - | - | - |
| fsdp, 16 gpus | 105k / 20.8% | 160k / 31.6% | 202k / 40.0% | **217k** / 43.0% |
| hsdp, 2x8 gpus | 100k / 19.8% | 148k / 29.4% | 179k / 35.5% | oom |
