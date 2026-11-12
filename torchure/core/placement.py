"""
placement value types: how one tensor is laid out along ONE mesh axis.

a tensor's full layout is a `Placements` tuple with one entry per mesh axis,
so on a (dp_replicate, dp_shard) mesh fsdp's master weights are
`(Replicate(), Shard(0))`.

nothing here needs a process group: these are value types plus the purely
local math (local shape, local slice). everything that launches a collective
lives in core/dtensor.py's redistribute table.

the dataclasses are frozen so placements compare and hash by value:
redistribute dispatches on them, and they can key dicts.
"""

from dataclasses import dataclass
from typing import get_args

import torch

from torchure.core.collective import ReduceOpName


class Placement:
    """base class so `isinstance(p, Placement)` works in asserts."""

    __slots__ = ()


@dataclass(frozen=True)
class Shard(Placement):
    """
    the tensor is split along tensor dim `dim` across this mesh axis; the rank
    at coordinate c holds chunk c. the split must be even (no padding; see
    `padded_chunk` for the opt-in padded layout).

    `dim` must be non-negative: Shard(-1) and Shard(1) would describe the
    same layout of a 2-d tensor yet compare unequal.
    """

    dim: int

    def __post_init__(self) -> None:
        assert isinstance(self.dim, int) and not isinstance(self.dim, bool), (
            f"Shard(dim) must be an int, got {type(self.dim).__name__}"
        )
        assert self.dim >= 0, (
            f"Shard({self.dim}): negative dims are rejected, because Shard(-1) and "
            "Shard(ndim-1) would be the same layout but compare unequal"
        )

    def __repr__(self) -> str:
        return f"Shard({self.dim})"

    def local_numel_along_dim(self, global_size: int, group_size: int) -> int:
        """size of this rank's chunk along `dim`, asserting even division."""
        assert global_size % group_size == 0, (
            f"cannot shard dim of size {global_size} across {group_size} ranks: "
            "not divisible, and Shard does not pad"
        )
        return global_size // group_size

    def local_shape(self, global_shape: tuple[int, ...], group_size: int) -> tuple[int, ...]:
        """the per-rank shape of a tensor whose global shape is `global_shape`."""
        assert self.dim < len(global_shape), (
            f"Shard({self.dim}) but the tensor has only {len(global_shape)} dims"
        )
        local = list(global_shape)
        local[self.dim] = self.local_numel_along_dim(global_shape[self.dim], group_size)
        return tuple(local)

    def global_shape(self, local_shape: tuple[int, ...], group_size: int) -> tuple[int, ...]:
        """inverse of local_shape: what the ranks' chunks concatenate into."""
        assert self.dim < len(local_shape), (
            f"Shard({self.dim}) but the tensor has only {len(local_shape)} dims"
        )
        full = list(local_shape)
        full[self.dim] *= group_size
        return tuple(full)

    def local_slice(self, tensor: torch.Tensor, group_size: int, coord: int) -> torch.Tensor:
        """
        this rank's chunk of a full tensor (Replicate -> Shard, no comm).

        returns a VIEW: a caller that wants the full tensor freed must copy
        the chunk into storage of its own.
        """
        n = self.local_numel_along_dim(tensor.shape[self.dim], group_size)
        assert 0 <= coord < group_size, f"coord {coord} out of range for {group_size}"
        return tensor.narrow(self.dim, coord * n, n)


@dataclass(frozen=True)
class Replicate(Placement):
    """every rank along this axis holds the whole tensor."""

    def __repr__(self) -> str:
        return "Replicate()"


@dataclass(frozen=True)
class Partial(Placement):
    """
    every rank holds a partial contribution; the true value is the reduction
    of all ranks' local tensors under `op`. an unsharded gradient is this
    before it has been reduced (each rank computed it from its own
    microbatch), which makes `Partial -> Shard(0)` fsdp's grad reduce-scatter.

    Partial("sum") and Partial("avg") are not interchangeable: confusing them
    is a silent factor of group_size.
    """

    op: ReduceOpName = "sum"

    def __post_init__(self) -> None:
        legal = get_args(ReduceOpName)
        assert self.op in legal, f"Partial(op={self.op!r}) is not one of {legal}"

    def __repr__(self) -> str:
        return f"Partial({self.op})"


def padded_chunk(global_size: int, group_size: int, coord: int) -> tuple[int, int, int]:
    """
    the padded layout for a dim that need not divide evenly: every rank holds
    a chunk of `length = ceil(global / group)` rows so the collectives see
    equal sizes, of which only the first `valid` are real; the rest is zero
    padding. rank `coord` owns global rows [start, start + valid).

    returns (length, start, valid). trailing ranks may own fewer rows than
    `length`, or none. with even division this is exactly Shard's layout.
    used by fsdp and the checkpointer, which both know how to trim the padding.
    """
    assert 0 <= coord < group_size, f"coord {coord} out of range for {group_size}"
    length = -(-global_size // group_size)
    start = min(coord * length, global_size)
    valid = min(start + length, global_size) - start
    return length, start, valid


# one entry per mesh axis, in the mesh's own axis order
Placements = tuple[Placement, ...]


def local_shape(
    global_shape: tuple[int, ...], placements: Placements, sizes: tuple[int, ...]
) -> tuple[int, ...]:
    """
    per-rank shape given the global shape and one group size per axis.
    Replicate and Partial leave the shape alone; only Shard divides.
    """
    assert len(placements) == len(sizes), (
        f"{len(placements)} placements for {len(sizes)} mesh axes"
    )
    shape = tuple(global_shape)
    for placement, size in zip(placements, sizes, strict=True):
        if isinstance(placement, Shard):
            shape = placement.local_shape(shape, size)
    return shape


def global_shape(
    local: tuple[int, ...], placements: Placements, sizes: tuple[int, ...]
) -> tuple[int, ...]:
    """inverse of local_shape."""
    assert len(placements) == len(sizes), (
        f"{len(placements)} placements for {len(sizes)} mesh axes"
    )
    shape = tuple(local)
    for placement, size in zip(placements, sizes, strict=True):
        if isinstance(placement, Shard):
            shape = placement.global_shape(shape, size)
    return shape
