# torchure
pure pytorch training stack with minimal deps

no framework imports (megatron, deepspeed, accelerate, liger) — the mesh,
collectives, and parallelism are implemented here, with pytorch's versions as
the thing to test against rather than the thing to depend on.

## what's here
- `torchure/models/` — self-contained qwen3 dense (GQA + RoPE + SwiGLU, tied embeddings), llama3 stubbed
- `torchure/dataloader/` — streaming HF datasets, sequence packing (stream + best-fit-decreasing), stateful dataloader, CUDA prefetcher (overlaps h2d copy with compute)
- `torchure/objectives/` + `torchure/loss/` — AR next-token cross entropy, plus a chunked fused linear+CE that never materializes the (B, S, vocab) logits
- `torchure/optimizer/` — fused adamw, wsd scheduler (cosine still a stub)
- `torchure/core/mesh.py` — named device mesh over the flat rank space: per-axis process groups, coordinates, and `flatten()` for virtual axes (e.g. one "which batch shard am i" dp coord spanning dp_replicate × dp_shard)
- `torchure/core/collective.py` — mesh-aware all_reduce / broadcast / all_gather / reduce_scatter / all_to_all / ring_send_recv. the only module that touches `torch.distributed` comm ops directly
- `torchure/parallelism/data_parallel.py` — DDP: bucketed grad all-reduce launched from backward hooks, overlapped with the rest of backward. fsdp2/tp/cp/ep are still empty stubs
- `torchure/train/trainer.py` — trainer: bf16 autocast, per-block torch.compile (fsdp2-friendly granularity), dp-aware loss logging
- `torchure/checkpoint/` — per-step model/optimizer/scheduler/dataloader/rng saving, `resume: null | "auto" | <step>`

everything is driven by a json config, see `configs/qwen3_dense_climbmix.json`
(single gpu) and `configs/qwen3_dense_climbmix_ddp.json` (8-way dp).

optimization history lives in `notes/CHANGES.md`, the ddp work log in
`notes/DDP.md`, where it's going in `notes/ROADMAP.md`.

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
| 1x H100 (PCIe) | ~54.5k |

### ddp
global tps (tokens through the whole job per second), same per-gpu config:

| hardware | tps | scaling eff |
|----------|-----|-------------|
| 8x A40 | ~73.5k | 69% |
| 8x A6000 | ~86.4 k | 60% |
| 4x H100 (PCIe) | ~198k | 92% |

on a pcie-only box grad sync is the whole ballgame; `notes/DDP.md` has the
`NCCL_P2P_LEVEL` investigation that took 8x A40 from 17% to 69% scaling.

## installation
if you have uv on your machine, no need
to create the conda env

```bash
conda create -n fresh python==3.14
conda activate fresh
pip install uv
uv pip install -e .
```

## future plans
support training AR LLMs, DLLMs,
continuous diffusion LLMs, etc

focusing on AR for now
