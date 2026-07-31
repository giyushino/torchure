# torchure

pure pytorch training stack with minimal deps.

no framework imports (megatron, deepspeed, accelerate, liger) — the mesh,
collectives, and parallelism are implemented here, with pytorch's own
implementations as the thing to test against, not the thing to depend on.

## what it can do

- **train autoregressive transformer LLMs**, single GPU up to N-way data
  parallel, from one json config — no code changes to scale a run up or down
- **data parallel training** with bucketed, overlapped gradient all-reduce
  (grad sync runs concurrently with the rest of backward, not after it),
  gradient accumulation via no_sync semantics, and gradient clipping —
  fsdp2/tp/cp/ep are the natural next axes on the same mesh, not built yet
- **bf16 mixed precision**, fp32 master weights and optimizer state — the
  standard recipe, not a naive full-bf16 cast (which silently stalls
  training once updates fall below what bf16's mantissa can resolve)
- **per-block torch.compile**, at the granularity fsdp2 will eventually want:
  one compiled region per transformer block, so warmup happens once and is
  reused across every identical block instead of once per layer
- **streaming, resumable data loading** straight from HF datasets — sequence
  packing (no padding waste), a stateful dataloader that checkpoints its own
  position, and a CUDA prefetcher that overlaps the host→device copy with
  compute
- **memory-efficient loss** — a chunked fused linear+cross-entropy that never
  materializes the (batch, seq, vocab) logits tensor, otherwise the single
  biggest activation at large vocab sizes
- **fused adamw + a working warmup-stable-decay schedule** (cosine is stubbed,
  not wired up yet)
- **full checkpoint/resume** — model, optimizer, scheduler, dataloader
  position, and rng state, all restorable via `resume: null | "auto" | <step>`
- **a from-scratch mesh + collectives layer** underneath all of the above —
  a named device mesh over the flat rank space (per-axis process groups,
  coordinates, virtual/flattened axes) and mesh-aware wrappers over every
  collective op (`all_reduce`, `broadcast`, `all_gather`, `reduce_scatter`,
  `all_to_all`, `ring_send_recv`). this is the only place that talks to
  `torch.distributed` directly, so every parallelism strategy above it is
  backend-agnostic

everything is driven by a json config; see `configs/` for single-gpu and
data-parallel examples. optimization history, the ddp investigation, and
the roadmap live under `notes/`.

## running

single gpu and multi gpu are the same code path — world_size is just 1:

```bash
# single gpu
uv run -m torchure.train.train --config configs/qwen3_dense_climbmix.json

# multi gpu
uv run torchrun --nproc-per-node 8 -m torchure.train.train \
    --config configs/qwen3_dense_climbmix_ddp.json

# benchmark mode: N steps, no checkpointing, prints tps + peak mem
uv run -m torchure.train.train --config configs/qwen3_dense_climbmix.json --steps 20

uv run torchrun --nproc-per-node 8 -m torchure.train.train \
    --config configs/qwen3_dense_climbmix_ddp.json --steps 20
```

## tests

no gpus needed for any of these except the last (gloo/cpu, `mp.spawn`):

```bash
uv run tests/collective.py --world-size 4     # 12 collective op tests
uv run tests/dataloader_packing.py --check    # packer equality + invariants
uv run tests/checkpointer.py                  # save/load roundtrip
uv run tests/data_parallel.py --world-size 4  # ddp grad parity vs single process
uv run tests/fused_linear_ce.py               # needs cuda
```

## throughput

tokens/sec for qwen3 0.6B dense, seq_len 4096, batch size 2/gpu, bf16:

| hardware | tps |
|----------|-----|
| 1x A40 | ~13.3k |
| 1x A6000 | ~14.4k |
| 1x H100 NVL | ~54.5k |

### ddp

global tps (tokens through the whole job per second), same per-gpu config:

| hardware | tps | scaling eff |
|----------|-----|-------------|
| 8x A40 | ~73.5k | 69% |
| 8x A6000 | ~86.4k | 60% |
| 4x H100 NVL | ~198k | 92% |

on a pcie-only box grad sync is the whole ballgame; `notes/DDP.md` has the
`NCCL_P2P_LEVEL` investigation that took 8x A40 from 17% to 69% scaling.

## installation

if you have uv on your machine, no need to create the conda env

```bash
conda create -n fresh python==3.14
conda activate fresh
pip install uv
uv pip install -e .
```
### Optional: Weights & Biases logging

Experiment tracking is off by default. To enable it:

```bash
uv pip install -e ".[wandb]"
```

## future plans

support training AR LLMs, DLLMs, continuous diffusion LLMs, etc — focusing
on AR for now.
