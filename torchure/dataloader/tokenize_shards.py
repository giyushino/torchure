"""
tokenize a huggingface text dataset into a token-shard source
(torchure/dataloader/shards.py) for the mixture loader.

    uv run -m torchure.dataloader.tokenize_shards \\
        --dataset OptimalScale/ClimbMix --tokenizer Qwen/Qwen3-0.6B \\
        --eos-token "<|endoftext|>" --max-tokens 2e9 --out data/climbmix_qwen3

then train from the directory directly, or upload it and point the mix
script at the url:

    hf upload <org>/climbmix-qwen3 data/climbmix_qwen3 --repo-type dataset
    SOURCES = {"web": "hf://datasets/<org>/climbmix-qwen3"}

docs are written in stream order. the loader shuffles at block granularity,
so a dataset sorted by topic should be fine, but don't rely on doc order
within a 16MB block being random.

re-run after a tokenizer change: the source records its tokenizer and the
loader refuses a mismatch.
"""

import argparse
import os
import time

import datasets
import numpy as np
from tokenizers import Tokenizer

from torchure.dataloader.shards import write_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="hf dataset name")
    p.add_argument("--config", default=None, help="hf subset name")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--eos-token", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=float, default=None, help="stop after about this many")
    p.add_argument("--shard-tokens", type=float, default=2.5e8,
                   help="tokens per .bin file (1GB at uint32)")
    p.add_argument("--batch-docs", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos_token)
    assert eos_id is not None, f"{args.eos_token!r} not in {args.tokenizer}"
    vocab = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab <= 2**16 else np.uint32
    max_tokens = None if args.max_tokens is None else int(args.max_tokens)
    shard_tokens = int(args.shard_tokens)

    os.makedirs(args.out, exist_ok=True)
    assert not os.path.exists(os.path.join(args.out, "index.json")), (
        f"{args.out} already holds a complete source"
    )
    dataset = datasets.load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    files: list[dict] = []
    current = None
    current_tokens = 0
    total_tokens = total_docs = 0
    t0 = time.time()

    def open_next():
        nonlocal current, current_tokens
        if current is not None:
            current.close()
            files[-1]["num_tokens"] = current_tokens
        path = f"{len(files):06d}.bin"
        files.append({"path": path, "num_tokens": 0})
        current = open(os.path.join(args.out, path), "wb")
        current_tokens = 0

    open_next()
    for batch in dataset.iter(batch_size=args.batch_docs):
        # same call as the hf path's packer, so both see identical tokens
        encodings = tokenizer.encode_batch(batch[args.text_field])
        lengths = np.array([len(e.ids) + 1 for e in encodings])
        flat = np.empty(int(lengths.sum()), dtype=dtype)
        ends = np.cumsum(lengths)
        for enc, end, n in zip(encodings, ends, lengths, strict=True):
            flat[end - n:end - 1] = enc.ids
            flat[end - 1] = eos_id
        # docs never straddle files
        if current_tokens and current_tokens + len(flat) > shard_tokens:
            open_next()
        flat.tofile(current)
        current_tokens += len(flat)
        total_tokens += len(flat)
        total_docs += len(encodings)
        if total_docs % (args.batch_docs * 50) == 0:
            rate = total_tokens / (time.time() - t0)
            print(f"{total_docs:,} docs, {total_tokens / 1e9:.3f}B tokens, {rate / 1e6:.1f}M tok/s")
        if max_tokens is not None and total_tokens >= max_tokens:
            break
    current.close()
    files[-1]["num_tokens"] = current_tokens

    write_index(
        args.out, dtype=np.dtype(dtype).name, tokenizer=args.tokenizer, eos_id=eos_id,
        files=files,
        extra={"num_docs": total_docs,
               "made_from": {"dataset": args.dataset, "config": args.config,
                             "split": args.split, "text_field": args.text_field}},
    )
    print(f"done: {total_docs:,} docs, {total_tokens:,} tokens in {len(files)} files -> {args.out}")


if __name__ == "__main__":
    main()
