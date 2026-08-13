"""
DTensor: a bookkeeping wrapper over (local_tensor, mesh, placements) with an
explicit `redistribute`.

**this is deliberately NOT a __torch_dispatch__ subclass with sharding
propagation.** pytorch's DTensor infers placements through every aten op;
that's a compiler-sized project, and nothing downstream here needs it --
fsdp2, tp, cp and checkpointing all know exactly which collective they want
and are better off saying so. this wrapper exists to name layouts, convert
between them, and carry the metadata a sharded checkpoint will need.

consequence, and the reason fsdp2.py reads the way it does: **the optimizer
never sees a DTensor.** params stay plain torch.Tensors at runtime, with
their placements recorded alongside in fsdp2's per-param state. fused AdamW,
the torch._foreach_* calls in core/grad_helper.py, and every `p.grad` access
keep working untouched. if real op propagation is ever wanted, that decision
is what has to be revisited first.

the redistribute table (each cell is one call into core/collective.py):

    Shard(d)   -> Replicate    all_gather(gather_dim=d)
    Replicate  -> Shard(d)     local slice, NO comm
    Partial(op)-> Replicate    all_reduce(op)
    Partial(op)-> Shard(d)     reduce_scatter(op, scatter_dim=d)
    Shard(i)   -> Shard(j)     all_to_all(split_dim=j, concat_dim=i)

transitions INTO Partial are not implemented: Replicate -> Partial has no
canonical answer (you'd have to zero every rank but one, and which one is
arbitrary), and nothing needs it. Partial arises only from local computation
-- an unreduced gradient -- so it is produced by `from_local`, never by a
redistribute.

run tests with: uv run tests/dtensor.py --world-size 4   (gloo/cpu)
"""

import torch

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
from torchure.core import placement as _placement


class DTensor:
    """
    (local_tensor, mesh, placements). `placements` has one entry per axis in
    `mesh_dims`, which is the explicit axis-name tuple this tensor is laid out
    over -- passing the names rather than reading them off the mesh keeps a
    tensor sharded only over ("dp_shard",) from having to carry Replicate()
    entries for tp/pp/cp axes it doesn't care about.
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
        # two axes sharding the SAME tensor dim would make the gathered order
        # depend on which axis is gathered first, and neither order is
        # obviously right. nothing in fsdp/hsdp needs it (hsdp shards on
        # dp_shard and replicates on dp_replicate), so reject it rather than
        # pick an order nobody verified.
        shard_dims = [p.dim for p in placements if isinstance(p, Shard)]
        assert len(shard_dims) == len(set(shard_dims)), (
            f"two mesh axes shard the same tensor dim in {placements}; "
            "not supported in v0 (the gather order would be ambiguous)"
        )
        self._local = local
        self._mesh = mesh
        self._mesh_dims = tuple(mesh_dims)
        self._placements = tuple(placements)

    # ---- construction / access -------------------------------------------

    @staticmethod
    def from_local(
        local: torch.Tensor,
        mesh: MeshLike,
        mesh_dims: tuple[str, ...],
        placements: Placements,
    ) -> "DTensor":
        """
        wrap an already-correctly-laid-out local tensor. NO communication and
        no validation that the ranks agree -- this is an assertion by the
        caller about what the local tensor already is.
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

    def _sizes(self) -> tuple[int, ...]:
        return tuple(self._mesh.size(d) for d in self._mesh_dims)

    @property
    def shape(self) -> torch.Size:
        """the GLOBAL shape. Partial and Replicate don't change it."""
        return torch.Size(
            _placement.global_shape(
                tuple(self._local.shape), self._placements, self._sizes()
            )
        )

    def full_tensor(self) -> torch.Tensor:
        """
        the whole logical tensor on every rank: redistribute to Replicate on
        every axis and unwrap. this is the ground truth the tests compare
        against, and it is a real collective -- note that for a Partial it is
        an all_reduce, not a gather.
        """
        return self.redistribute((Replicate(),) * len(self._mesh_dims)).to_local()

    def __repr__(self) -> str:
        return (
            f"DTensor(local={tuple(self._local.shape)}, global={tuple(self.shape)}, "
            f"dims={self._mesh_dims}, placements={self._placements})"
        )

    # ---- the redistribute table ------------------------------------------

    def redistribute(self, placements: Placements) -> "DTensor":
        """
        return a new DTensor with the requested placements, launching one
        collective per axis that actually changes. axes whose placement is
        already the target are skipped entirely, which is what makes a size-1
        axis (and a no-op redistribute) genuinely free.

        axes are processed in `mesh_dims` order. that is well defined here
        because the ctor rejects two axes sharding the same tensor dim, so no
        two transitions can interact.
        """
        assert len(placements) == len(self._mesh_dims), (
            f"{len(placements)} placements for axes {self._mesh_dims}"
        )
        local = self._local
        current = list(self._placements)

        for i, dim_name in enumerate(self._mesh_dims):
            src, dst = current[i], placements[i]
            if src == dst:
                continue
            local = self._transition(local, dim_name, src, dst)
            current[i] = dst

        return DTensor(local, self._mesh, self._mesh_dims, tuple(current))

    def _transition(
        self, local: torch.Tensor, dim_name: str, src: Placement, dst: Placement
    ) -> torch.Tensor:
        """one cell of the table, for one mesh axis."""
        size = self._mesh.size(dim_name)

        # a size-1 axis can't move any data: every transition between
        # placements is the identity, except Partial->anything, which is still
        # the identity because there is exactly one contributor. skipping the
        # collective here is what keeps dp_shard=1 free.
        if size == 1:
            return local

        if isinstance(src, Shard) and isinstance(dst, Replicate):
            return all_gather(local, self._mesh, dim_name, gather_dim=src.dim)

        if isinstance(src, Replicate) and isinstance(dst, Shard):
            # the one cell with no comm
            coord = self._mesh.coordinate(dim_name)
            return dst.local_slice(local, size, coord).contiguous()

        if isinstance(src, Partial) and isinstance(dst, Replicate):
            # all_reduce is in-place on its input and returns it; clone first
            # so redistribute never mutates the caller's tensor (same contract
            # reduce_scatter follows)
            return all_reduce(local.clone(), self._mesh, dim_name, src.op)

        if isinstance(src, Partial) and isinstance(dst, Shard):
            return reduce_scatter(
                local, self._mesh, dim_name, src.op, scatter_dim=dst.dim
            )

        if isinstance(src, Shard) and isinstance(dst, Shard):
            # currently split along src.dim, want split along dst.dim: cut the
            # local chunk along the TARGET dim, trade row-for-row, and glue
            # the received pieces back along the SOURCE dim.
            return all_to_all(
                local, self._mesh, dim_name, split_dim=dst.dim, concat_dim=src.dim
            )

        if isinstance(dst, Partial):
            raise NotImplementedError(
                f"{src} -> {dst} on axis {dim_name!r}: transitions INTO Partial "
                "are not defined (you would have to zero all ranks but an "
                "arbitrary one). Partial comes from local computation, via "
                "from_local, never from a redistribute"
            )

        raise NotImplementedError(f"redistribute {src} -> {dst} on axis {dim_name!r}")


def distribute_tensor(
    full: torch.Tensor,
    mesh: MeshLike,
    mesh_dims: tuple[str, ...],
    placements: Placements,
) -> DTensor:
    """
    take a full tensor that every rank already holds identically and produce
    the DTensor with the requested placements. sharding is a local slice, so
    this launches no collectives -- it does NOT check that the ranks really do
    agree on `full` (fsdp broadcasts before calling this; see fsdp2.py).
    """
    dt = DTensor(full, mesh, mesh_dims, (Replicate(),) * len(mesh_dims))
    return dt.redistribute(placements)
