"""
perplexity of a checkpoint over freshly streamed batches of its training set.

no held-out split exists for climbmix (the data config only declares
"train"), so this is in-distribution perplexity, not a real held-out
number -- a progress/sanity signal, not a benchmark. the seed is bumped
from the training config's so the shuffle buffer doesn't hand back the
exact stream order the run trained on (streaming shuffle can't guarantee
zero overlap without an actual held-out split).

    uv run -m torchure.inference.eval_ppl \
        --config configs/qwen3_dense_climbmix_ddp.json --step 42000
"""
import argparse
import math

import torch

from torchure.dataloader.builder import build_dataloader
from torchure.inference.common import load_config, load_model_and_tokenizer
from torchure.loss.cross_entropy import cross_entropy_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--num-batches", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(config, args.step, device)

    data_cfg = dict(config["data"])
    data_cfg["seed"] = data_cfg["seed"] + 1
    data_cfg["num_workers"] = 1  # persistent_workers=True in build_dataloader needs > 0
    ignore_index = config["objective"]["config"]["ignore_index"]
    loader = build_dataloader(data_cfg, tokenizer, ignore_index, dp_rank=0, dp_size=1)

    losses = []
    it = iter(loader)
    with torch.no_grad():
        for i in range(args.num_batches):
            batch = next(it)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
            loss = cross_entropy_loss(labels, logits.float(), ignore_index)
            losses.append(loss.item())
            running = sum(losses) / len(losses)
            print(f"batch {i + 1}/{args.num_batches} || loss={loss.item():.4f} "
                  f"|| running avg loss={running:.4f} || running ppl={math.exp(running):.2f}")

    avg_loss = sum(losses) / len(losses)
    n_tokens = len(losses) * data_cfg["micro_batch_size"] * data_cfg["seq_len"]
    print(f"\n{len(losses)} batches ({n_tokens} tokens) || avg loss={avg_loss:.4f} "
          f"|| perplexity={math.exp(avg_loss):.2f}")


if __name__ == "__main__":
    main()
