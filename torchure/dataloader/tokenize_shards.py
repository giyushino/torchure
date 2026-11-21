"""
tokenize a raw source (sources.py: jsonl, jsonl.zst/.gz or parquet, local or
over http) into a token-shard source (shards.py). optional: the loader
tokenizes raw sources while training. a token-shard copy is for data you'll
reuse across many runs or epochs and don't want to tokenize again.

    uv run -m torchure.dataloader.tokenize_shards \\
        --source hf://datasets/OptimalScale/ClimbMix --tokenizer Qwen/Qwen3-0.6B \\
        --eos-token "<|endoftext|>" --max-tokens 2e9 --out data/climbmix_qwen3

then train from the directory directly, or upload it and point a source at the url:

    hf upload <org>/climbmix-qwen3 data/climbmix_qwen3 --repo-type dataset
    SOURCES = {"web": "hf://datasets/<org>/climbmix-qwen3"}

docs are written in the source's block order. the loader shuffles at block
granularity, so a dataset sorted by topic should be fine, but don't rely on
doc order within a 16MB block being random.

re-run after a tokenizer change: the source records its tokenizer and the
loader refuses a mismatch.
"""

import argparse
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tokenizers import Tokenizer

from torchure.dataloader.shards import write_index
from torchure.dataloader.sources import RawSource, RowConfig, open_source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True,
                   help="a location: local path, http(s) url or hf://datasets/<org>/<name>")
    p.add_argument("--files", default=None, help="glob over the source's files")
    p.add_argument("--text-field", default="text")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--eos-token", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=float, default=None, help="stop after about this many")
    p.add_argument("--shard-tokens", type=float, default=2.5e8,
                   help="tokens per .bin file (1GB at uint32)")
    p.add_argument("--threads", type=int, default=16, help="blocks fetched and tokenized at once")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos_token)
    assert eos_id is not None, f"{args.eos_token!r} not in {args.tokenizer}"
    max_tokens = None if args.max_tokens is None else int(args.max_tokens)
    shard_tokens = int(args.shard_tokens)

    os.makedirs(args.out, exist_ok=True)
    assert not os.path.exists(os.path.join(args.out, "index.json")), (
        f"{args.out} already holds a complete source"
    )
    spec = {"path": args.source, "files": args.files, "text_field": args.text_field}
    source = open_source(spec, RowConfig(1, tokenizer, eos_id), block_mb=16, min_blocks=1)
    assert isinstance(source, RawSource) and not source.images, (
        f"{args.source}: expected raw text (jsonl / parquet)"
    )
    dtype = np.dtype(source.dtype)

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

    def tokenize(block) -> tuple[np.ndarray, int]:
        texts = source.records(block)[args.text_field]
        return source.tokenize(texts), len(texts)

    def in_order(pool: ThreadPoolExecutor):
        """blocks' results in block order, at most 2 x threads in flight or waiting."""
        pending: deque = deque()
        for block in source.blocks:
            pending.append(pool.submit(tokenize, block))
            if len(pending) >= 2 * args.threads:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()

    open_next()
    with ThreadPoolExecutor(args.threads) as pool:
        for flat, n_docs in in_order(pool):
            # docs never straddle files
            if current_tokens and current_tokens + len(flat) > shard_tokens:
                open_next()
            flat.tofile(current)
            current_tokens += len(flat)
            total_tokens += len(flat)
            total_docs += n_docs
            rate = total_tokens / (time.time() - t0)
            print(f"{total_docs:,} docs, {total_tokens / 1e9:.3f}B tokens, "
                  f"{rate / 1e6:.1f}M tok/s", end="\r")
            if max_tokens is not None and total_tokens >= max_tokens:
                pool.shutdown(wait=False, cancel_futures=True)
                break
    current.close()
    files[-1]["num_tokens"] = current_tokens

    write_index(
        args.out, dtype=dtype.name, tokenizer=args.tokenizer, eos_id=eos_id, files=files,
        extra={"num_docs": total_docs,
               "made_from": {"source": args.source, "files": args.files,
                             "text_field": args.text_field}},
    )
    print(f"\ndone: {total_docs:,} docs, {total_tokens:,} tokens in {len(files)} files -> {args.out}")


if __name__ == "__main__":
    main()
