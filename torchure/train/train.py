"""
single gpu training entrypoint

    python -m torchure.train.train --config configs/qwen3_dense.json

once distributed this is what you'd launch under torchrun; the Trainer
already carries rank/world_size so the launcher just fills them in.
"""

import argparse
import os

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
    trainer = Trainer(args.config, rank=rank, local_rank=local_rank, world_size=world_size)
    trainer.train()




if __name__ == "__main__":
    main()
