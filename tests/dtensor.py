"""
redistribute-table tests for torchure/core/dtensor.py.

run (gloo/cpu, no gpus needed):
    uv run tests/dtensor.py --world-size 4

every cell is checked against full_tensor() ground truth, which each rank can
compute locally because the data is a deterministic function of the
coordinate. the 2-axis cases run only at 4 ranks (2x2), where they also pin
that a transition on one axis leaves the other axis alone -- the property
hsdp depends on.
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchure.core.dtensor import DTensor, distribute_tensor
from torchure.core.mesh import Mesh
from torchure.core.placement import Partial, Replicate, Shard


class Skip(Exception):
    """raised by a test to report SKIP (not a failure) with a reason."""


def full(shape, device="cpu") -> torch.Tensor:
    """the same tensor on every rank: arange, so slices are recognisable."""
    numel = 1
    for s in shape:
        numel *= s
    return torch.arange(numel, dtype=torch.float32, device=device).reshape(shape)


def rank_pattern(shape, coord: int) -> torch.Tensor:
    numel = 1
    for s in shape:
        numel *= s
    return (torch.arange(numel, dtype=torch.float32) * (coord + 1)).reshape(shape)


# --------------------------------------------------------------------------
# single-axis cells
# --------------------------------------------------------------------------


def test_shard_to_replicate(mesh, dim):
    g, c = mesh.size(dim), mesh.coordinate(dim)
    x = full((4 * g, 3))
    dt = distribute_tensor(x, mesh, (dim,), (Shard(0),))
    assert tuple(dt.to_local().shape) == (4, 3)
    assert tuple(dt.shape) == (4 * g, 3), "global shape must survive sharding"
    torch.testing.assert_close(dt.to_local(), x[4 * c : 4 * c + 4])
    torch.testing.assert_close(dt.full_tensor(), x)


def test_shard_nonzero_dim(mesh, dim):
    g, c = mesh.size(dim), mesh.coordinate(dim)
    x = full((3, 4 * g))
    dt = distribute_tensor(x, mesh, (dim,), (Shard(1),))
    torch.testing.assert_close(dt.to_local(), x[:, 4 * c : 4 * c + 4])
    torch.testing.assert_close(dt.full_tensor(), x)


def test_replicate_to_shard_is_local(mesh, dim):
    # the one cell with no communication: it must work with the collectives
    # untouched, which is implicitly checked by it not hanging when only some
    # ranks would have participated
    g, c = mesh.size(dim), mesh.coordinate(dim)
    x = full((2 * g, 5))
    dt = DTensor.from_local(x, mesh, (dim,), (Replicate(),))
    shard = dt.redistribute((Shard(0),))
    torch.testing.assert_close(shard.to_local(), x[2 * c : 2 * c + 2])
    assert shard.to_local().is_contiguous(), "local slice must be made contiguous"


def test_partial_to_replicate(mesh, dim):
    # Partial -> Replicate is an ALL_REDUCE, not a gather: the local tensor is
    # already full-shaped, it just isn't reduced yet
    g, c = mesh.size(dim), mesh.coordinate(dim)
    local = rank_pattern((4, 3), c)
    dt = DTensor.from_local(local, mesh, (dim,), (Partial("sum"),))
    assert tuple(dt.shape) == (4, 3), "Partial does not change the global shape"
    torch.testing.assert_close(dt.full_tensor(), rank_pattern((4, 3), 0) * (g * (g + 1) / 2))
    # and it must not have mutated the caller's local tensor
    torch.testing.assert_close(local, rank_pattern((4, 3), c), rtol=0, atol=0)

    avg = DTensor.from_local(rank_pattern((4, 3), c), mesh, (dim,), (Partial("avg"),))
    torch.testing.assert_close(avg.full_tensor(), rank_pattern((4, 3), 0) * ((g + 1) / 2))


def test_partial_to_shard(mesh, dim):
    # this cell IS fsdp's gradient reduce-scatter
    g, c = mesh.size(dim), mesh.coordinate(dim)
    dt = DTensor.from_local(rank_pattern((4 * g, 3), c), mesh, (dim,), (Partial("avg"),))
    sharded = dt.redistribute((Shard(0),))
    expected = rank_pattern((4 * g, 3), 0) * ((g + 1) / 2)
    torch.testing.assert_close(sharded.to_local(), expected[4 * c : 4 * c + 4])
    # ... and gathering it back gives the all_reduced value: the identity fsdp
    # is built on, one level up from tests/collective.py
    torch.testing.assert_close(sharded.full_tensor(), expected)


def test_shard_to_shard(mesh, dim):
    g = mesh.size(dim)
    if g == 1:
        raise Skip("Shard(i)->Shard(j) is a no-op at 1 rank")
    x = full((2 * g, 3 * g))
    dt = distribute_tensor(x, mesh, (dim,), (Shard(0),))
    moved = dt.redistribute((Shard(1),))
    assert moved.placements == (Shard(1),)
    assert tuple(moved.shape) == (2 * g, 3 * g), "global shape is layout-invariant"
    c = mesh.coordinate(dim)
    torch.testing.assert_close(moved.to_local(), x[:, 3 * c : 3 * c + 3])
    torch.testing.assert_close(moved.full_tensor(), x)

    # and back again
    back = moved.redistribute((Shard(0),))
    torch.testing.assert_close(back.to_local(), x[2 * c : 2 * c + 2])


def test_round_trip_bit_exact(mesh, dim):
    g = mesh.size(dim)
    x = torch.randn(4 * g, 7, generator=torch.Generator().manual_seed(0))
    dt = distribute_tensor(x, mesh, (dim,), (Shard(0),))
    torch.testing.assert_close(dt.full_tensor(), x, rtol=0, atol=0)


def test_noop_redistribute(mesh, dim):
    # matching placements must short-circuit -- this is what makes a size-1
    # axis and a redundant redistribute genuinely free
    c, g = mesh.coordinate(dim), mesh.size(dim)
    dt = distribute_tensor(full((4 * g, 3)), mesh, (dim,), (Shard(0),))
    same = dt.redistribute((Shard(0),))
    assert same.to_local() is dt.to_local(), "no-op redistribute must not copy"


def test_into_partial_rejected(mesh, dim):
    if mesh.size(dim) == 1:
        raise Skip("every transition is the identity at 1 rank")
    dt = distribute_tensor(full((4 * mesh.size(dim), 3)), mesh, (dim,), (Shard(0),))
    try:
        dt.redistribute((Partial("sum"),))
    except NotImplementedError:
        return
    raise AssertionError("Shard -> Partial should be rejected, not guessed at")


def test_indivisible_rejected(mesh, dim):
    if mesh.size(dim) == 1:
        raise Skip("everything divides by 1")
    try:
        distribute_tensor(full((4 * mesh.size(dim) + 1, 3)), mesh, (dim,), (Shard(0),))
    except AssertionError:
        return
    raise AssertionError("indivisible shape must assert, not truncate")


def test_double_shard_same_dim_rejected(mesh, dim):
    # two axes sharding one tensor dim makes the gather order ambiguous
    try:
        DTensor.from_local(full((4, 3)), mesh, (dim, dim), (Shard(0), Shard(0)))
    except AssertionError:
        return
    raise AssertionError("two axes on one tensor dim should be rejected")


# --------------------------------------------------------------------------
# two-axis (hsdp-shaped) cells -- 4 ranks only
# --------------------------------------------------------------------------


def test_two_axis(mesh, dim):
    if dist.get_world_size() != 4:
        raise Skip("needs exactly 4 ranks (2x2 mesh)")
    m = Mesh({"dp_replicate": 2, "dp_shard": 2})
    dims = ("dp_replicate", "dp_shard")
    c = m.coordinate("dp_shard")

    # what fsdp's master weights are: replicated across dp_replicate, sharded
    # across dp_shard
    x = full((8, 3))
    dt = distribute_tensor(x, m, dims, (Replicate(), Shard(0)))
    assert tuple(dt.to_local().shape) == (4, 3)
    assert tuple(dt.shape) == (8, 3)
    torch.testing.assert_close(dt.to_local(), x[4 * c : 4 * c + 4])
    torch.testing.assert_close(dt.full_tensor(), x)

    # what an fsdp gradient is on the way down: Partial on both axes -> the
    # reduce_scatter/all_reduce pair hsdp performs
    coord_flat = dist.get_rank()
    local = rank_pattern((8, 3), coord_flat)
    grad = DTensor.from_local(local, m, dims, (Partial("avg"), Partial("avg")))
    reduced = grad.redistribute((Replicate(), Shard(0)))
    expected = sum(rank_pattern((8, 3), r) for r in range(4)) / 4
    torch.testing.assert_close(reduced.to_local(), expected[4 * c : 4 * c + 4])

    # a transition on one axis must leave the other alone
    assert reduced.placements == (Replicate(), Shard(0))


TESTS = [
    test_shard_to_replicate,
    test_shard_nonzero_dim,
    test_replicate_to_shard_is_local,
    test_partial_to_replicate,
    test_partial_to_shard,
    test_shard_to_shard,
    test_round_trip_bit_exact,
    test_noop_redistribute,
    test_into_partial_rejected,
    test_indivisible_rejected,
    test_double_shard_same_dim_rejected,
    test_two_axis,
]


def _worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29541")
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    mesh, dim = Mesh({"dp": world_size}), "dp"
    failed = []
    for test in TESTS:
        try:
            test(mesh, dim)
            msg = "PASS"
        except Skip as e:
            msg = f"SKIP: {e}"
        except AssertionError as e:
            msg = f"FAIL: {e}"
            failed.append(test.__name__)
        except Exception as e:  # noqa: BLE001
            msg = f"ERROR: {type(e).__name__}: {e}"
            failed.append(test.__name__)
        dist.barrier()
        if rank == 0:
            print(f"[{test.__name__}] {msg}")
    dist.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    mp.spawn(_worker, args=(args.world_size,), nprocs=args.world_size)
