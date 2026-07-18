"""
single gpu training entrypoint

    python -m torchure.train.train --config configs/qwen3_dense.json

once distributed this is what you'd launch under torchrun; the Trainer
already carries rank/world_size so the launcher just fills them in.
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(world_size)

    # bind to my gpu BEFORE init: nccl allocates communicator state on
    # the current device, and this keeps rank0's gpu from collecting
    # everyone's contexts
    torch.cuda.set_device(local_rank)
    # no rank/world args: env:// rendezvous reads them from torchrun.
    # explicit timeout so a wedged collective fails instead of hanging.
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))

    try:
        trainer = Trainer(args.config, rank=rank, local_rank=local_rank, world_size=world_size)
        trainer.train()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()




if __name__ == "__main__":
    main()
