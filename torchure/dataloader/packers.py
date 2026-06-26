"""
file mostly written by claude, except for listpacker and
the best fit

packing strategies for the dataloader .map step, behind one shared interface so
they're interchangeable and benchmarkable.

a packer takes a map-batch of docs, concatenates their token ids into a single
stream with an eos between docs, then slices the stream into back-to-back
seq_len blocks. every emitted row is exactly seq_len -- no padding, no attention
mask downstream. the tail remainder (< seq_len) of each map-batch is dropped.
carrying that remainder into the next batch is a separate algorithm (different
output), so it lives in its own class rather than as a "variant" here.

the variants below differ only in *how* they concatenate + slice (python lists
vs numpy vs torch). given the same docs they emit identical blocks, which the
benchmark suite asserts before trusting any timing. tokenization is shared in
the base __call__, so pack() -- the only thing that differs -- is the only axis
under test.

all packers are top-level classes (not closures) so DataLoader workers can
pickle them, which forkserver/spawn require (py3.14 default on linux).
"""

from __future__ import annotations

import numpy as np
import torch

from tokenizers import Tokenizer

from torchure.utils import Node


class Packer:
    """
    base packer: owns the shared tokenize step and the interface. subclasses
    implement pack(), the concatenate-and-slice that the benchmark compares.

    tokenizer is optional so the benchmark can construct a packer and exercise
    pack() on pre-tokenized docs without standing up a real tokenizer.
    """
    def __init__(
        self,
        tokenizer: Tokenizer | None,
        seq_len: int,
        eos_id: int,
        verbose: bool = False,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = eos_id
        # the "lost N tokens" accounting prints inside the hot path, so it's off
        # by default and never runs during a benchmark.
        self.verbose = verbose

    def __call__(self, examples: dict[str, list]) -> dict[str, list]:
        encodings = self.tokenizer.encode_batch(examples["text"])
        docs = [encoding.ids for encoding in encodings]
        return {"input_ids": self.pack(docs)}

    def pack(self, docs: list[list[int]]) -> list[list[int]]:
        """
        concatenate docs (eos-separated) and slice into seq_len blocks, as
        nested python lists. this is the `.map`-time path: output must be
        Arrow-serializable, so it's always lists regardless of strategy.
        """
        raise NotImplementedError

    def pack_to_tensor(self, docs: list[list[int]]) -> torch.Tensor:
        """
        same blocks, but materialized straight into the (n_blocks, seq_len)
        tensor a collate-time packer would feed the model -- no list round-trip.

        the base impl models "list-packer + collator": build lists, then
        tensorize. numpy/tensor strategies override to skip the round-trip
        (from_numpy / pass-through), which is the whole point of benchmarking
        them against this path.
        """
        blocks = self.pack(docs)
        if not blocks:
            return torch.empty((0, self.seq_len), dtype=torch.long)
        return torch.tensor(blocks, dtype=torch.long)

    def _report_loss(self, n_tokens: int, n_blocks: int) -> None:
        if self.verbose:
            lost = n_tokens - n_blocks * self.seq_len
            print(f"{type(self).__name__}: lost {lost} tokens")

class ListPacker(Packer):
    """naive python: extend a flat list, then slice. the original implementation."""

    def pack(self, docs: list[list[int]]) -> list[list[int]]:
        stream: list[int] = []
        for ids in docs:
            stream.extend(ids)
            stream.append(self.eos_id)

        n_blocks = len(stream) // self.seq_len
        self._report_loss(len(stream), n_blocks)
        stream = stream[: n_blocks * self.seq_len]
        return [
            stream[i * self.seq_len: (i + 1) * self.seq_len]
            for i in range(n_blocks)
        ]

class ListPackerBestFit(Packer):
    """
    best-fit-decreasing packing (https://arxiv.org/pdf/2404.10830).

    each emitted block is one training sequence == one bin of capacity seq_len.
    a long doc is split into full seq_len blocks (emitted as-is) plus a < seq_len
    remainder; those remainders -- together with whole short docs -- are packed
    into bins by best-fit-decreasing so we waste as little bin capacity as
    possible (the paper's "fewer truncations" goal). a bin that doesn't fill up
    is padded to seq_len with eos_id.

    unlike the stream packers (ListPacker/NumpyPacker/TensorPacker) this does NOT
    drop the tail and does NOT emit identical blocks -- different output is the
    whole point. so it lives outside the equality group: the benchmark checks it
    against token-conservation invariants instead (see BESTFIT_PACKERS).

    a segment tree over remaining-space buckets (0..seq_len) gives O(log seq_len)
    best fit: leaf s holds s iff some open bin has exactly s free, internal nodes
    hold the max, so search(c) returns the smallest space >= c that fits.
    """

    def pack(self, docs: list[list[int]]) -> list[list[int]]:
        seq_len = self.seq_len

        full_blocks: list[list[int]] = []
        # remainders to bin-pack. each carries a trailing eos so multiple docs
        # sharing a bin stay delimited; sized including that eos. an exact
        # multiple has an empty remainder -> a lone [eos] that still terminates
        # the doc and costs almost nothing.
        items: list[list[int]] = []
        for ids in docs:
            n_full = len(ids) // seq_len
            for i in range(n_full):
                full_blocks.append(ids[i * seq_len: (i + 1) * seq_len])
            items.append(ids[n_full * seq_len:] + [self.eos_id])

        # "decreasing": place the big items first so they claim fresh bins and
        # the small ones slot into whatever gaps are left.
        items.sort(key=len, reverse=True)

        # segment tree indexed by remaining space; space_to_bin maps a remaining
        # space to the open bins that currently have it; bins holds contents.
        free = Node.build(0, seq_len)
        space_to_bin: dict[int, list[int]] = {}
        bins: list[list[int]] = []

        for item in items:
            size = len(item)  # in [1, seq_len]
            space = free.search(size)
            if space is None:
                # nothing open fits -> start a fresh bin.
                bin_id = len(bins)
                bins.append([])
                space = seq_len
            else:
                bin_id = space_to_bin[space].pop()
                if not space_to_bin[space]:
                    free.update(space, available=False)

            bins[bin_id].extend(item)
            new_space = space - size
            if new_space > 0:
                bucket = space_to_bin.setdefault(new_space, [])
                if not bucket:
                    free.update(new_space, available=True)
                bucket.append(bin_id)

        # pad each bin up to seq_len (the collator stacks fixed-length rows).
        for b in bins:
            if len(b) < seq_len:
                b.extend([self.eos_id] * (seq_len - len(b)))
        return full_blocks + bins


class NumpyPacker(Packer):
    """concatenate into one int64 array, reshape into (n_blocks, seq_len)."""
    # this makes no sense? why are we turning this into a list again

    def _blocks(self, docs: list[list[int]]) -> np.ndarray:
        eos = np.array([self.eos_id], dtype=np.int64)
        parts: list[np.ndarray] = []
        for ids in docs:
            parts.append(np.asarray(ids, dtype=np.int64))
            parts.append(eos)
        if not parts:
            return np.empty((0, self.seq_len), dtype=np.int64)
        stream = np.concatenate(parts)

        n_blocks = stream.shape[0] // self.seq_len
        self._report_loss(stream.shape[0], n_blocks)
        return stream[: n_blocks * self.seq_len].reshape(n_blocks, self.seq_len)

    def pack(self, docs: list[list[int]]) -> list[list[int]]:
        return self._blocks(docs).tolist()

    def pack_to_tensor(self, docs: list[list[int]]) -> torch.Tensor:
        # from_numpy shares the buffer -- no copy, no list round-trip.
        return torch.from_numpy(self._blocks(docs))


class TensorPacker(Packer):
    """concatenate into one torch tensor, view into (n_blocks, seq_len)."""

    def _blocks(self, docs: list[list[int]]) -> torch.Tensor:
        eos = torch.tensor([self.eos_id], dtype=torch.long)
        parts: list[torch.Tensor] = []
        for ids in docs:
            parts.append(torch.tensor(ids, dtype=torch.long))
            parts.append(eos)
        if not parts:
            return torch.empty((0, self.seq_len), dtype=torch.long)
        stream = torch.cat(parts)

        n_blocks = stream.shape[0] // self.seq_len
        self._report_loss(stream.shape[0], n_blocks)
        return stream[: n_blocks * self.seq_len].view(n_blocks, self.seq_len)

    def pack(self, docs: list[list[int]]) -> list[list[int]]:
        return self._blocks(docs).tolist()

    def pack_to_tensor(self, docs: list[list[int]]) -> torch.Tensor:
        return self._blocks(docs)


# registry so the builder and the benchmark refer to strategies by name. these
# all emit identical blocks for identical input, so the benchmark equality-gates
# them against each other.
PACKERS: dict[str, type[Packer]] = {
    "list": ListPacker,
    "numpy": NumpyPacker,
    "tensor": TensorPacker,
}

# best-fit is a different algorithm with different output (no dropped tail,
# padded bins), so it can't be equality-checked against the stream packers --
# the benchmark times it and verifies token-conservation invariants separately.
BESTFIT_PACKERS: dict[str, type[Packer]] = {
    "bestfit": ListPackerBestFit,
}



