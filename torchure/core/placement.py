"""
placement types: how a logical tensor is laid out over one mesh axis.

value types on purpose -- frozen, hashable, comparable, printable. they end up
as dict keys in the redistribute table, in checkpoint metadata (phase 3 reads
them back to resolve a resharding load), and in assertion messages, so
`repr()` has to be something a human can read at 3am.

one placement per mesh axis the tensor is distributed over. a tensor sharded
over dp_shard and replicated over tp is
    (mesh_dims=("dp_shard", "tp"), placements=(Shard(0), Replicate()))

`Partial(op)` is "this rank holds a contribution, not a value": the logical
tensor is the reduction of every rank's local tensor under `op`. it is what a
grad is immediately after backward and before the reduce-scatter, and it is
the only placement whose local tensor is NOT a slice of the logical one.
"""

from dataclasses import dataclass
from typing import Union

from torchure.core.collective import ReduceOpName


@dataclass(frozen=True)
class Shard:
    """local tensor is the coordinate-th chunk of the logical tensor along `dim`."""

    dim: int


@dataclass(frozen=True)
class Replicate:
    """local tensor is the whole logical tensor; every rank on the axis agrees."""


@dataclass(frozen=True)
class Partial:
    """logical tensor == reduce(local tensors, op) across the axis."""

    op: ReduceOpName = "sum"


Placement = Union[Shard, Replicate, Partial]
