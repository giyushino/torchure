"""
the training entrypoint -- single gpu and multi gpu are the same code path,
world_size just differs.

single gpu (the env setdefaults below stand in for what torchrun would set):

    uv run -m torchure.train.train --config configs/qwen3_dense_climbmix.json

multi gpu:

    uv run torchrun --nproc-per-node 8 -m torchure.train.train \
        --config configs/qwen3_dense_climbmix_ddp.json

benchmark mode (fixed step count, no checkpointing) -- where CHANGES.md
numbers come from:

    uv run -m torchure.train.train --config configs/qwen3_dense_climbmix.json --steps 20
"""

import argparse
import os

from datetime import timedelta

import torch
import torch.distributed as dist

from torchure.train.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the train config json")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="benchmark mode: run exactly N steps (no checkpointing) and print "
             "tps/peak mem. this is what CHANGES.md numbers come from; omit to "
             "run the config's full schedule",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # defaults for single gpu launches without torchrun
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    # nccl's default p2p policy falls back to slow SHM across the host bridge
    # on the a40 box; allowing p2p is 8.4x all_reduce bandwidth (see DDP.md)
    os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # before init, or every rank's nccl context lands on gpu 0
    torch.cuda.set_device(local_rank)
    dist.init_process_group(timeout=timedelta(minutes=10))

    try:
        trainer = Trainer(
            args.config,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            benchmark=args.steps is not None,
        )
        if args.steps is not None:
            trainer.train_n_step_test(args.steps)
        else:
            trainer.train()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
