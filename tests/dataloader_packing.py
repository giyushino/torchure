"""
benchmark + correctness suite for the packing strategies in
torchure.dataloader.packers (ListPacker / NumpyPacker / TensorPacker).

two things matter and they're kept separate:

  correctness -- every strategy must emit *identical* blocks for identical
                 input. a fast packer that drops or reorders tokens is not a
                 win, so timings are only trusted after this passes.

  speed       -- pack() only. tokenization is shared across strategies and
                 would dominate the measurement, so the benchmark feeds
                 pre-tokenized synthetic docs straight into pack(). warmup +
                 repeats + median/stdev keep single-call noise from driving the
                 comparison (the reason a timing *decorator* is the wrong tool:
                 it measures one call and leaks timing into the return value).

  run benchmark:    PYTHONPATH=. python tests/dataloader_packing.py
  run correctness:  PYTHONPATH=. python tests/dataloader_packing.py --check
                    (or `pytest tests/dataloader_packing.py` if pytest is added)
"""

from __future__ import annotations

import gc
import random
import sys
import time

from collections import Counter
from dataclasses import dataclass
from statistics import median, pstdev

import torch

from torchure.dataloader.packers import BESTFIT_PACKERS, PACKERS


EOS_ID = 0
SEQ_LEN = 2048


def make_corpus(
    n_docs: int = 2000,
    *,
    seed: int = 0,
    min_len: int = 16,
    max_len: int = 1024,
    vocab_size: int = 32000,
) -> list[list[int]]:
    """
    a fixed, seeded corpus of pre-tokenized docs with varying lengths, so every
    run and every strategy sees the exact same input. ids start at 1 to keep the
    eos separator (0) distinguishable from real tokens.
    """
    rng = random.Random(seed)
    return [
        [rng.randrange(1, vocab_size) for _ in range(rng.randint(min_len, max_len))]
        for _ in range(n_docs)
    ]


def build_packers(seq_len: int = SEQ_LEN, eos_id: int = EOS_ID):
    # tokenizer=None: pack() never touches it and the benchmark feeds
    # pre-tokenized docs directly, so no real tokenizer is needed.
    return {name: cls(None, seq_len, eos_id) for name, cls in PACKERS.items()}


def build_bestfit_packers(seq_len: int = SEQ_LEN, eos_id: int = EOS_ID):
    return {name: cls(None, seq_len, eos_id) for name, cls in BESTFIT_PACKERS.items()}


# ---------------------------------------------------------------- correctness

def _check_correctness(n_docs: int = 500) -> None:
    docs = make_corpus(n_docs=n_docs)
    packers = build_packers()
    outs = {name: p.pack(docs) for name, p in packers.items()}

    # all strategies agree with the first one
    ref_name, ref = next(iter(outs.items()))
    for name, blocks in outs.items():
        assert blocks == ref, f"{name} disagrees with {ref_name}"

    # every block is exactly seq_len -- no ragged tails leak through
    for name, blocks in outs.items():
        for block in blocks:
            assert len(block) == SEQ_LEN, f"{name} emitted a ragged block"

    # the tensor path (pack_to_tensor) must agree too -- same blocks, just
    # materialized straight to a (n_blocks, seq_len) tensor.
    tensors = {name: p.pack_to_tensor(docs) for name, p in packers.items()}
    ref_t = tensors[ref_name]
    for name, t in tensors.items():
        assert torch.equal(t, ref_t), f"{name} tensor path disagrees with {ref_name}"

    # empty input is handled on both paths
    for name, p in packers.items():
        assert p.pack([]) == [], f"{name} mishandles empty input"
        assert p.pack_to_tensor([]).shape == (0, SEQ_LEN), f"{name} bad empty tensor"


def _check_bestfit(n_docs: int = 500) -> None:
    """
    best-fit emits different blocks than the stream packers (that's the point),
    so it gets invariants instead of block-equality:

      - every block is exactly seq_len (the collator stacks fixed-length rows);
      - no truncation: every real token survives. the corpus has no eos_id (ids
        start at 1, eos is 0), so eos in the output is only separator/padding --
        the multiset of non-eos tokens must equal the input's exactly. the
        stream packers fail this on purpose (they drop each batch's tail).
    """
    docs = make_corpus(n_docs=n_docs)
    for name, p in build_bestfit_packers().items():
        blocks = p.pack(docs)
        for block in blocks:
            assert len(block) == SEQ_LEN, f"{name} emitted a ragged block"

        orig = Counter(t for d in docs for t in d)
        out = Counter(t for b in blocks for t in b if t != EOS_ID)
        assert out == orig, f"{name} lost or altered tokens (truncation)"

        # tensor path agrees with the list path, and empty input is handled.
        assert torch.equal(p.pack_to_tensor(docs), torch.tensor(blocks, dtype=torch.long)), \
            f"{name} tensor path disagrees with list path"
        assert p.pack([]) == [], f"{name} mishandles empty input"
        assert p.pack_to_tensor([]).shape == (0, SEQ_LEN), f"{name} bad empty tensor"


# pytest entry points (used only if pytest is installed; harmless otherwise)
def test_strategies_agree():
    _check_correctness()


def test_bestfit_conserves_tokens():
    _check_bestfit()


def test_empty_input():
    for name, p in build_packers().items():
        assert p.pack([]) == [], f"{name} mishandles empty input"


# ------------------------------------------------------------------ benchmark

@dataclass
class BenchResult:
    name: str
    median_s: float
    min_s: float
    stdev_s: float
    n_blocks: int


def benchmark(name, fn, *, warmup: int = 3, repeats: int = 10) -> BenchResult:
    out = None
    for _ in range(warmup):
        out = fn()

    # disable gc so a collection mid-run doesn't land in one sample's timing;
    # restore whatever state we found it in.
    samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = fn()
            samples.append(time.perf_counter() - t0)
    finally:
        if gc_was_enabled:
            gc.enable()

    return BenchResult(
        name=name,
        median_s=median(samples),
        min_s=min(samples),
        stdev_s=pstdev(samples) if len(samples) > 1 else 0.0,
        n_blocks=len(out) if out is not None else 0,
    )


def run_benchmark(n_docs: int = 2000) -> None:
    docs = make_corpus(n_docs=n_docs)
    n_tokens = sum(len(d) for d in docs)
    print(f"corpus: {n_docs} docs, {n_tokens:,} tokens, seq_len={SEQ_LEN}\n")

    # correctness gate -- timings are meaningless if the strategies disagree.
    _check_correctness()
    _check_bestfit()

    packers = build_packers()

    # case 1: `.map`-time packing. output is lists (Arrow needs them); the
    # collator's torch.tensor is a constant across strategies, so it's excluded.
    _print_table(
        "case 1: pack() -> lists  (.map-time; collator tensorize is constant)",
        [benchmark(name, lambda p=p: p.pack(docs)) for name, p in packers.items()],
    )

    # case 2: collate-time packing. each strategy materializes the final
    # (n_blocks, seq_len) tensor its natural way -- this folds in the
    # tensorization and lets numpy/tensor skip the list round-trip.
    _print_table(
        "case 2: pack_to_tensor() -> torch  (collate-time; tensorize included)",
        [benchmark(name, lambda p=p: p.pack_to_tensor(docs)) for name, p in packers.items()],
    )

    # best-fit is timed on its own: it emits a different (non-dropping, padded)
    # block set, so the n_blocks and "vs best" ratio aren't comparable to the
    # stream packers above -- the value here is best-fit vs itself over time.
    _print_table(
        "case 3: pack() -> lists  (best-fit; different output, not comparable above)",
        [benchmark(name, lambda p=p: p.pack(docs)) for name, p in build_bestfit_packers().items()],
    )


def _print_table(title: str, results: list[BenchResult]) -> None:
    results.sort(key=lambda r: r.median_s)
    best = results[0].median_s
    print(title)
    print(f"{'strategy':<10}{'median(ms)':>12}{'min(ms)':>10}{'stdev(ms)':>11}{'vs best':>10}")
    for r in results:
        print(
            f"{r.name:<10}{r.median_s * 1e3:>12.2f}{r.min_s * 1e3:>10.2f}"
            f"{r.stdev_s * 1e3:>11.2f}{r.median_s / best:>9.2f}x"
        )
    print(f"blocks emitted: {results[0].n_blocks}\n")


def main() -> None:
    if "--check" in sys.argv:
        _check_correctness()
        _check_bestfit()
        print("correctness: all strategies agree, blocks well-formed, empty ok")
        print("best-fit: blocks exactly seq_len, no truncation, empty ok")
        return
    run_benchmark()


if __name__ == "__main__":
    main()
