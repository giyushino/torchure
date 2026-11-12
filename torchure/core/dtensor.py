"""
DTensor: a bookkeeping wrapper over (local_tensor, mesh, placements) with an
explicit `redistribute`.

unlike pytorch's DTensor this is NOT a __torch_dispatch__ subclass: placements
are not propagated through ops. fsdp2, tp and checkpointing know exactly which
collective they want and say so; this wrapper only names layouts and converts
between them. the optimizer never sees a DTensor: params stay plain tensors,
with their placements recorded alongside (see fsdp2.py).

the redistribute table (each cell is one call into core/collective.py):

    Shard(d)   -> Replicate    all_gather(gather_dim=d)
    Replicate  -> Shard(d)     local slice, NO comm
    Partial(op)-> Replicate    all_reduce(op)
    Partial(op)-> Shard(d)     reduce_scatter(op, scatter_dim=d)
    Shard(i)   -> Shard(j)     all_to_all(split_dim=j, concat_dim=i)

there are no transitions INTO Partial: Replicate -> Partial would have to zero
every rank but an arbitrary one. Partial arises only from local computation
(an unreduced gradient), so it enters through `from_local`.

tests: uv run tests/dtensor.py --world-size 4   (gloo/cpu)
"""

import torch

from torchure.core import placement as _placement
from torchure.core.collective import (
    MeshLike,
    all_gather,
    all_reduce,
    all_to_all,
    reduce_scatter,
)
from torchure.core.placement import (
    Partial,
    Placement,
    Placements,
    Replicate,
    Shard,
)


class DTensor:
    """
    (local_tensor, mesh, placements). `placements` has one entry per axis in
    `mesh_dims`, the axes this tensor is laid out over. naming them (rather
    than using every mesh axis) spares a tensor sharded only over
    ("dp_shard",) from carrying Replicate() entries for the rest.
    """

    __slots__ = ("_local", "_mesh", "_mesh_dims", "_placements")

    def __init__(
        self,
        local: torch.Tensor,
        mesh: MeshLike,
        mesh_dims: tuple[str, ...],
        placements: Placements,
    ):
        assert len(mesh_dims) == len(placements), (
            f"{len(placements)} placements for axes {mesh_dims}"
        )
        assert all(isinstance(p, Placement) for p in placements), (
            f"placements must be Placement instances, got {placements}"
        )
        # two axes sharding the same tensor dim would make the gathered order
        # depend on which axis is gathered first. nothing needs it
        shard_dims = [p.dim for p in placements if isinstance(p, Shard)]
        assert len(shard_dims) == len(set(shard_dims)), (
            f"two mesh axes shard the same tensor dim in {placements}; "
            "not supported (the gather order would be ambiguous)"
        )
        self._local = local
        self._mesh = mesh
        self._mesh_dims = tuple(mesh_dims)
        self._placements = tuple(placements)

    @staticmethod
    def from_local(
        local: torch.Tensor,
        mesh: MeshLike,
        mesh_dims: tuple[str, ...],
        placements: Placements,
    ) -> "DTensor":
        """
        wrap a local tensor the caller asserts is already laid out this way.
        no communication, and no check that the ranks agree.
        """
        return DTensor(local, mesh, mesh_dims, placements)

    def to_local(self) -> torch.Tensor:
        """the raw local shard. no comm."""
        return self._local

    @property
    def placements(self) -> Placements:
        return self._placements

    @property
    def mesh_dims(self) -> tuple[str, ...]:
        return self._mesh_dims

    @property
    def mesh(self) -> MeshLike:
        return self._mesh

    @property
    def dtype(self) -> torch.dtype:
        return self._local.dtype

    @property
    def device(self) -> torch.device:
        return self._local.device

    @property
    def shape(self) -> torch.Size:
        """the GLOBAL shape. Partial and Replicate don't change it."""
        sizes = tuple(self._mesh.size(d) for d in self._mesh_dims)
        return torch.Size(
            _placement.global_shape(tuple(self._local.shape), self._placements, sizes)
        )

    def full_tensor(self) -> torch.Tensor:
        """
        the whole logical tensor on every rank: redistribute to Replicate on
        every axis and unwrap. a real collective (an all_reduce for a Partial).
        """
        return self.redistribute((Replicate(),) * len(self._mesh_dims)).to_local()

    def __repr__(self) -> str:
        return (
            f"DTensor(local={tuple(self._local.shape)}, global={tuple(self.shape)}, "
            f"dims={self._mesh_dims}, placements={self._placements})"
        )

    def redistribute(self, placements: Placements) -> "DTensor":
        """
        return a new DTensor with the requested placements, launching one
        collective per axis whose placement changes, in `mesh_dims` order.
        """
        assert len(placements) == len(self._mesh_dims), (
            f"{len(placements)} placements for axes {self._mesh_dims}"
        )
        local = self._local
        for dim_name, src, dst in zip(self._mesh_dims, self._placements, placements, strict=True):
            if src != dst:
                local = self._transition(local, dim_name, src, dst)
        return DTensor(local, self._mesh, self._mesh_dims, tuple(placements))

    def _transition(
        self, local: torch.Tensor, dim_name: str, src: Placement, dst: Placement
    ) -> torch.Tensor:
        """one cell of the table, for one mesh axis."""
        size = self._mesh.size(dim_name)

        # on a size-1 axis every transition is the identity: skip the collective
        if size == 1:
            return local

        if isinstance(src, Shard) and isinstance(dst, Replicate):
            return all_gather(local, self._mesh, dim_name, gather_dim=src.dim)

        if isinstance(src, Replicate) and isinstance(dst, Shard):
            coord = self._mesh.coordinate(dim_name)
            return dst.local_slice(local, size, coord).contiguous()

        if isinstance(src, Partial) and isinstance(dst, Replicate):
            # all_reduce is in place; clone so the caller's tensor isn't mutated
            return all_reduce(local.clone(), self._mesh, dim_name, src.op)

        if isinstance(src, Partial) and isinstance(dst, Shard):
            return reduce_scatter(local, self._mesh, dim_name, src.op, scatter_dim=dst.dim)

        if isinstance(src, Shard) and isinstance(dst, Shard):
            # cut the local chunk along the TARGET dim, trade row-for-row, and
            # glue the received pieces back along the SOURCE dim
            return all_to_all(local, self._mesh, dim_name, split_dim=dst.dim, concat_dim=src.dim)

        if isinstance(dst, Partial):
            raise NotImplementedError(
                f"{src} -> {dst} on axis {dim_name!r}: there are no transitions INTO "
                "Partial. it comes from local computation, via from_local"
            )

        raise NotImplementedError(f"redistribute {src} -> {dst} on axis {dim_name!r}")


def distribute_tensor(
    full: torch.Tensor,
    mesh: MeshLike,
    mesh_dims: tuple[str, ...],
    placements: Placements,
) -> DTensor:
    """
    shard a full tensor that every rank already holds identically. sharding
    is a local slice, so this launches no collectives, and does NOT check
    that the ranks really agree on `full`.
    """
    dt = DTensor(full, mesh, mesh_dims, (Replicate(),) * len(mesh_dims))
    return dt.redistribute(placements)
