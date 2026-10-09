"""
packing strategies for the dataloader .map step.

stream packers concatenate docs with an eos between them and slice into
back-to-back seq_len blocks (no padding; each map-batch's tail is dropped).
list/numpy/tensor variants emit identical blocks and differ only in speed.

best-fit only saves ~0.04% of tokens on climbmix, possibly because its docs
are short.

top-level classes so spawn workers can pickle them.
"""

from __future__ import annotations

import numpy as np
import torch

from tokenizers import Tokenizer

from torchure.utils import Node


class Packer:
    """
    shared tokenize step; subclasses implement pack(). tokenizer is optional
    so benchmarks can call pack() on pre-tokenized docs.
    """
    def __init__(
        self,
        tokenizer: Tokenizer | None,
        seq_len: int,
        eos_id: int,
        pad_id: int | None = None,
        verbose: bool = False,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = eos_id
        # pad must differ from eos so pad can be masked without masking doc
        # boundaries. stream packers never pad, so it defaults to eos.
        self.pad_id = pad_id if pad_id is not None else eos_id
        self.verbose = verbose

    def __call__(self, examples: dict[str, list]) -> dict[str, list]:
        encodings = self.tokenizer.encode_batch(examples["text"])
        docs = [encoding.ids for encoding in encodings]
        return {"input_ids": self.pack(docs)}

    def pack(self, docs: list[list[int]]) -> list[list[int]]:
        """seq_len blocks as nested lists (must be arrow-serializable for .map)."""
        raise NotImplementedError

    def pack_to_tensor(self, docs: list[list[int]]) -> torch.Tensor:
        """same blocks as a (n_blocks, seq_len) tensor, for collate-time packing."""
        blocks = self.pack(docs)
        if not blocks:
            return torch.empty((0, self.seq_len), dtype=torch.long)
        return torch.tensor(blocks, dtype=torch.long)

    def _report_loss(self, n_tokens: int, n_blocks: int) -> None:
        if self.verbose:
            lost = n_tokens - n_blocks * self.seq_len
            print(f"{type(self).__name__}: lost {lost} tokens")

class ListPacker(Packer):
    """extend a flat python list, then slice."""

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

    long docs emit full seq_len blocks as-is; their remainders and short docs
    are packed into seq_len bins, and unfilled bins are padded with pad_id.
    nothing is dropped, so the output differs from the stream packers.

    a segment tree over remaining space (0..seq_len) gives O(log seq_len) best
    fit: search(c) returns the smallest open space >= c.
    """

    def pack(self, docs: list[list[int]]) -> list[list[int]]:
        seq_len = self.seq_len

        full_blocks: list[list[int]] = []
        # remainders to bin-pack, each with a trailing eos
        items: list[list[int]] = []
        for ids in docs:
            n_full = len(ids) // seq_len
            for i in range(n_full):
                full_blocks.append(ids[i * seq_len: (i + 1) * seq_len])
            items.append(ids[n_full * seq_len:] + [self.eos_id])

        items.sort(key=len, reverse=True)

        # space_to_bin: remaining space -> open bins with exactly that much
        free = Node.build(0, seq_len)
        space_to_bin: dict[int, list[int]] = {}
        bins: list[list[int]] = []

        for item in items:
            size = len(item)  # in [1, seq_len]
            space = free.search(size)
            if space is None:
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

        for b in bins:
            if len(b) < seq_len:
                b.extend([self.pad_id] * (seq_len - len(b)))
        return full_blocks + bins


class NumpyPacker(Packer):
    """concatenate into one int64 array, reshape into (n_blocks, seq_len)."""

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


# emit identical blocks, so the benchmark checks them against each other
PACKERS: dict[str, type[Packer]] = {
    "list": ListPacker,
    "numpy": NumpyPacker,
    "tensor": TensorPacker,
}

# different output, checked with token-conservation invariants instead
BESTFIT_PACKERS: dict[str, type[Packer]] = {
    "bestfit": ListPackerBestFit,
}



