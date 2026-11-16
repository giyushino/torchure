"""
write a checkpoint's model weights as ONE plain state_dict file.

a run under fsdp, tensor parallel or expert parallel saves its weights as
one file per shard. this reassembles them (the same resharding load every
resume uses) into a single `torch.save`d state_dict that loads anywhere with
`model.load_state_dict(torch.load(path))`.

    uv run -m torchure.inference.export \
        --config configs/qwen3_dense_climbmix_fsdp_h100.json --step 42000 \
        --out qwen3_0.6b_step42000.pt

single process, cpu; needs enough host memory for the full weights.
"""
import argparse

import torch

from torchure.checkpoint.checkpointer import Checkpointer
from torchure.config import load_config
from torchure.models.builder import build_model
from torchure.utils import checkpoint_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--step", type=int, default=None, help="default: the latest complete step")
    parser.add_argument("--out", required=True, help="path of the state_dict file to write")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpointer = Checkpointer(checkpoint_dir(config))
    step = checkpointer.latest() if args.step is None else checkpointer.valid_step(args.step)
    assert step is not None, (
        "no complete checkpoint"
        + (f" at step {args.step}" if args.step is not None else "")
        + f" in {checkpointer.checkpoint_save_path}"
    )
    model = build_model(config["model"]["name"], config["model"]["config"])
    checkpointer.load_model(model, step)
    torch.save(model.state_dict(), args.out)
    params = sum(p.numel() for p in model.parameters())
    print(f"step {step}: {params / 1e9:.2f}B params -> {args.out}")


if __name__ == "__main__":
    main()
