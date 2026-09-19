"""
DTensor: a bookkeeping wrapper over (local_tensor, mesh, placements) with an
explicit `redistribute`.

deliberately NOT a `__torch_dispatch__` subclass with sharding propagation for
every aten op (roadmap 2.1 makes this decision explicitly). pytorch's DTensor
does full op dispatch; that is a compiler-sized project, and nothing here needs
it -- fsdp2, tp, cp and the sharded checkpointer all know exactly which
collective they want and when. this class exists to make "which placement is
this tensor in" a thing you can assert on, test against a ground truth, and
write into checkpoint metadata.

**the runtime consequence, stated once so it doesn't get re-litigated:** fsdp2
params stay plain `torch.Tensor` / `nn.Parameter` at runtime. fused adamw,
`torch._foreach_*`, and `clip_grad_norm` keep working untouched, and nothing in
the hot path unwraps. DTensor is used at the edges: constructing shards at
init, checkpoint save/load metadata, and tests that need `full_tensor()` as
ground truth. if that ever stops being enough, the note to leave behind is:
every `p.grad`, every foreach call, and the whole clip path grows an unwrap.

the redistribute table (implement + test every cell against `full_tensor()`):

    from -> to              collective
    Shard(d)   -> Replicate all_gather(gather_dim=d)
    Replicate  -> Shard(d)  local slice, NO comm
    Partial    -> Replicate all_reduce(op)
    Partial    -> Shard(d)  reduce_scatter(op, scatter_dim=d)
    Shard(i)   -> Shard(j)  all_to_all(split_dim=j, concat_dim=i)
    X          -> X         no-op
    Replicate  -> Partial   ILLEGAL (would need to invent a decomposition)

divisibility: the collectives assert `shape[dim] % size == 0` and do not pad.
every qwen3 param's dim 0 is divisible by 8, so v0 asserts loudly and moves on.
the first model with an awkward vocab or head count needs padding, and it goes
*here* -- pad the local shard to the max shard size at `from_local`, carry the
logical shape in `self.shape`, and trim in `full_tensor()`. don't discover this
at 3am.

run tests with: uv run tests/dtensor.py --world-size 4  (gloo/cpu)
"""

from __future__ import annotations

import torch

from torchure.core.collective import MeshLike
from torchure.core.placement import Placement


class DTensor:
    def __init__(
        self,
        local_tensor: torch.Tensor,
        mesh: MeshLike,
        mesh_dims: tuple[str, ...],
        placements: tuple[Placement, ...],
        shape: torch.Size | None = None,
    ):
        """
        `mesh_dims` names the axes this tensor is distributed over, positionally
        paired with `placements` -- our Mesh is a named dict, not an ordered
        n-d object, so the pairing has to be carried explicitly rather than
        inferred from mesh rank order.

        `shape` is the LOGICAL (full) shape. it is derivable from local_tensor
        + placements today, but it stops being derivable the moment padding
        lands, and it is what checkpoint metadata wants. store it.
        """
        raise NotImplementedError

    # --- construction -----------------------------------------------------

    @staticmethod
    def from_local(
        local_tensor: torch.Tensor,
        mesh: MeshLike,
        mesh_dims: tuple[str, ...],
        placements: tuple[Placement, ...],
    ) -> DTensor:
        """
        wrap an already-local tensor, asserting nothing about its contents.
        no comm. this is the cheap constructor fsdp uses once per param at init.
        """
        raise NotImplementedError

    @staticmethod
    def from_full(
        full_tensor: torch.Tensor,
        mesh: MeshLike,
        mesh_dims: tuple[str, ...],
        placements: tuple[Placement, ...],
    ) -> DTensor:
        """
        every rank holds the identical full tensor; slice out this rank's part.
        no comm (that's the point -- fsdp2 v0 inits full weights on every rank
        with a fixed seed and slices, which is bit-identical by construction and
        keeps init_weights' residual scaling a whole-tensor operation).

        trap: `.contiguous()` the slice. a `Shard(d>0)` slice is a view with the
        original stride, and handing that to a collective either copies
        silently or asserts, depending on the op.
        """
        raise NotImplementedError

    # --- accessors --------------------------------------------------------

    def to_local(self) -> torch.Tensor:
        """this rank's local tensor. no comm, no copy -- the same object."""
        raise NotImplementedError

    def full_tensor(self) -> torch.Tensor:
        """
        the logical tensor, materialized on every rank. this is
        `redistribute(all Replicate).to_local()` and it is the ground truth
        every test in tests/dtensor.py compares against. never call it in the
        training loop.
        """
        raise NotImplementedError

    # --- the interesting one ----------------------------------------------

    def redistribute(self, placements: tuple[Placement, ...]) -> DTensor:
        """
        move to `placements` (same mesh_dims, positionally). returns a NEW
        DTensor; `self` is untouched.

        implementation notes:
        - do one axis at a time, in mesh_dims order, and keep it a flat
          dispatch over (src, dst) pairs -- no clever multi-axis fusion. the
          table is small and the whole value of this class is that each cell
          is individually testable.
        - `Partial -> Shard(d)` is the one fsdp's backward actually runs, and
          the one that has to accept `async_op`. it isn't exposed here (this
          API is sync); fsdp2 calls `reduce_scatter` directly for the
          overlapped path and uses this only to check itself in tests.
        - `Replicate -> Partial` must raise, not silently divide by group size.
        """
        raise NotImplementedError

    # --- ergonomics -------------------------------------------------------

    def __repr__(self) -> str:
        """
        `DTensor(shape=(151936, 1024), dp_shard=Shard(0), local=(37984, 1024))`
        -- placements and both shapes, because every bug in this file looks
        like a shape mismatch three frames later.
        """
        raise NotImplementedError

    def placement_for(self, mesh_dim: str) -> Placement:
        """placement on a named axis; Replicate() for axes not in mesh_dims."""
        raise NotImplementedError


if __name__ == "__main__":
    print("dtensor: see tests/dtensor.py")
