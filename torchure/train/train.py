"""
single gpu training entrypoint

    python -m torchure.train.train --config configs/qwen3_dense.json

once distributed this is what you'd launch under torchrun; the Trainer
already carries rank/world_size so the launcher just fills them in.
"""

import argparse

from torchure.train.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the train config json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = Trainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
