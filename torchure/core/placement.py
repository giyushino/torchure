"""
placement value types: how one tensor is laid out along ONE mesh axis.

a tensor's full layout is a `Placements` tuple with one entry per mesh axis,
so a 2-axis (dp_replicate, dp_shard) mesh describes fsdp's master weights as
`(Replicate(), Shard(0))`. it is a tuple even though fsdp v0 only ever shards
along one axis: making it a scalar now means hsdp and tp both rewrite every
call site.

the dividing line against core/dtensor.py: **anything here needs no process
group.** these are value types plus the pure-local math (local shape, local
slice) that happens to need no communication. every operation that launches a
collective lives in dtensor.py's redistribute table. that keeps this module
importable and unit-testable with no init_process_group and no mp.spawn --
tests/placement.py is the only test file in the tree that doesn't need a
distributed harness.

frozen dataclasses on purpose: `frozen=True` gives __eq__ and __hash__
together, which is what lets redistribute dispatch on placement values and
what lets phase 3 put them in checkpoint metadata dicts. every field is a
plain int/str for the same reason -- whatever serializer phase 3 picks will
work without touching this file.
"""

from dataclasses import dataclass

import torch

from torchure.core.collective import ReduceOpName

# kept in sync with collective._OPS by the assert in Partial.__post_init__;
# duplicated as a tuple here only so this module doesn't reach into a private
_LEGAL_OPS: tuple[str, ...] = ("sum", "avg", "max", "min")


class Placement:
    """base class so `isinstance(p, Placement)` works in asserts."""

    __slots__ = ()


@dataclass(frozen=True)
class Shard(Placement):
    """
    the tensor is split along tensor dim `dim` across this mesh axis; rank at
    coordinate c holds chunk c. v0 requires even division and says so loudly
    rather than padding.

    `dim` must be non-negative. it cannot be normalized at construction (the
    tensor's rank isn't known yet), so allowing negatives would make
    Shard(-1) and Shard(1) describe the same layout of a 2-d tensor while
    comparing unequal and hashing differently -- every dict keyed on
    placements would then silently hold two entries for one placement and the
    redistribute table would miss. cheap to forbid, expensive to debug.
    """

    dim: int

    def __post_init__(self) -> None:
        assert isinstance(self.dim, int) and not isinstance(self.dim, bool), (
            f"Shard(dim) must be an int, got {type(self.dim).__name__}"
        )
        assert self.dim >= 0, (
            f"Shard({self.dim}): negative dims are rejected because they can't be "
            "normalized without a tensor rank, and Shard(-1) != Shard(ndim-1) "
            "would silently split the placement's identity"
        )

    def __repr__(self) -> str:
        return f"Shard({self.dim})"

    def local_numel_along_dim(self, global_size: int, group_size: int) -> int:
        """size of this rank's chunk along `dim`, asserting even division."""
        assert global_size % group_size == 0, (
            f"cannot shard dim of size {global_size} across {group_size} ranks: "
            "not divisible. v0 does not pad -- the fix when a model needs it is "
            "to pad the shard to ceil(global/group) and trim after the gather, "
            "which also means the checkpoint has to record the unpadded size"
        )
        return global_size // group_size

    def local_shape(
        self, global_shape: tuple[int, ...], group_size: int
    ) -> tuple[int, ...]:
        """the per-rank shape of a tensor whose global shape is `global_shape`."""
        assert self.dim < len(global_shape), (
            f"Shard({self.dim}) but the tensor has only {len(global_shape)} dims"
        )
        local = list(global_shape)
        local[self.dim] = self.local_numel_along_dim(global_shape[self.dim], group_size)
        return tuple(local)

    def global_shape(
        self, local_shape: tuple[int, ...], group_size: int
    ) -> tuple[int, ...]:
        """inverse of local_shape: what the ranks' chunks concatenate into."""
        assert self.dim < len(local_shape), (
            f"Shard({self.dim}) but the tensor has only {len(local_shape)} dims"
        )
        full = list(local_shape)
        full[self.dim] *= group_size
        return tuple(full)

    def local_slice(
        self, tensor: torch.Tensor, group_size: int, coord: int
    ) -> torch.Tensor:
        """
        this rank's chunk of a full tensor. this is the Replicate -> Shard(d)
        cell of the redistribute table, and the only cell needing NO
        communication, which is why it lives here rather than in dtensor.py.

        returns a VIEW. callers that want to own the storage (fsdp's master
        shards do) must .contiguous().clone() it -- otherwise the full tensor
        stays alive behind the view and the sharding saves nothing.
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
    of all ranks' local tensors under `op`. this is what an unsharded gradient
    is before it has been reduced -- each rank computed it from its own
    microbatch -- which makes `Partial -> Shard(0)` exactly fsdp's grad
    reduce-scatter.

    `op` reuses collective.ReduceOpName rather than inventing a second
    vocabulary for reductions, and is validated here so a typo fails at
    construction instead of inside a collective. note Partial("sum") and
    Partial("avg") are NOT interchangeable: confusing them is a silent factor
    of group_size, which is the same bug class the reduce_scatter "avg" path
    already had.
    """

    op: ReduceOpName = "sum"

    def __post_init__(self) -> None:
        assert self.op in _LEGAL_OPS, (
            f"Partial(op={self.op!r}) is not one of {_LEGAL_OPS}"
        )

    def __repr__(self) -> str:
        return f"Partial({self.op})"


# one entry per mesh axis, in the mesh's own axis order.
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
