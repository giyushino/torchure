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
    # set defaults for single gpu python launches,
    # maybe remove later
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    # nccl's default p2p policy stops at the host bridge, which on the a40
    # box turns every cross-pair ring hop into 1.3 GB/s SHM staging; p2p is
    # actually fine there at any distance (26 GB/s intra-numa, 13 cross), and
    # allowing it is worth 8.4x on all_reduce bus bandwidth (see DDP.md).
    # setdefault, so a box where distant p2p really is broken can override.
    os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # bind to my gpu BEFORE init: nccl allocates communicator state on
    # the current device, and this keeps rank0's gpu from collecting
    # everyone's contexts
    torch.cuda.set_device(local_rank)
    # no rank/world args: env:// rendezvous reads them from torchrun.
    # explicit timeout so a wedged collective fails instead of hanging.
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
