# torchure
pure pytorch training stack with minimal deps

## what's here
- `torchure/models/` — self-contained qwen3 dense (GQA + RoPE + SwiGLU, tied embeddings), llama3 stubbed
- `torchure/dataloader/` — streaming HF datasets, sequence packing, stateful dataloader, CUDA prefetcher (overlaps h2d copy with compute)
- `torchure/objectives/` + `torchure/loss/` — AR next-token cross entropy
- `torchure/optimizer/` — fused adamw, wsd scheduler
- `torchure/train/trainer.py` — single gpu trainer: bf16 autocast, per-block torch.compile (fsdp2-friendly granularity)
- `torchure/checkpoint/` — basic model/optimizer/dataloader saving
- `torchure/core/` + `torchure/parallelism/` — empty for now, mesh/dtensor + fsdp2/tp/cp/ep land here next

everything is driven by a json config, see `configs/qwen3_dense_climbmix.json`

optimization history lives in `CHANGES.md`

## throughput
tokens/sec for qwen3 0.6B dense, seq_len 4096, batch size 2, bf16:

| hardware | tps |
|----------|-----|
| 1x A40 | ~13.3k |
| 1x H100 (PCIe) | ~54.5k |

### ddp
| hardware | tps |
|----------|-----|
| 4x H100 (PCIe) | ~198k |

## running
```bash
uv run torchure/train/trainer.py
```

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
