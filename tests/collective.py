"""
correctness tests for torchure/core/collective.py.

run (gloo/cpu, no gpus needed, works on a single-gpu box):
    uv run tests/collective.py --world-size 4
or under torchrun (required for nccl, needs one gpu per rank):
    uv run torchrun --nproc-per-node 2 tests/collective.py --backend nccl

the tests are implemented; the collectives in torchure/core/collective.py
are the stubs. a test goes green when its collective honors the contract in
its docstring (in-place vs new tensor, coordinate vs global rank, etc.).

value scheme used throughout (see rank_pattern): rank at coordinate c holds
    x_c = arange(numel).reshape(shape) * (c + 1)
so with group size g the expected results are closed-form:
    sum over group -> arange * g*(g+1)/2
    avg            -> arange * (g+1)/2
    max            -> arange * g
distinct per-rank values + closed-form expectations means a wrong group,
a dropped rank, or a double-reduce all produce visibly wrong numbers.
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchure.core import collective as C


class Skip(Exception):
    """raised by a test to report SKIP (not a failure) with a reason."""


class FlatMesh:
    """
    1-d MeshLike stand-in over the WORLD group so collectives are testable
    before core/mesh.py exists. accepts any dim name.
    """

    def size(self, dim: str) -> int:
        return dist.get_world_size()

    def get_group(self, dim: str) -> dist.ProcessGroup:
        return dist.group.WORLD

    def coordinate(self, dim: str) -> int:
        return dist.get_rank()


def rank_pattern(shape, coord: int, device, dtype=torch.float32) -> torch.Tensor:
    """arange(numel).reshape(shape) * (coord + 1), on `device`."""
    numel = 1
    for s in shape:
        numel *= s
    return (torch.arange(numel, dtype=dtype, device=device) * (coord + 1)).reshape(shape)


# --------------------------------------------------------------------------
# tests. each takes (mesh, dim, device); AssertionError = failure,
# Skip = skipped. values are small-int fp32, so sums are exact and
# assert_close's default fp32 tolerances are more than enough.
# --------------------------------------------------------------------------


def test_all_reduce_sum(mesh, dim, device):
    c, g = mesh.coordinate(dim), mesh.size(dim)
    x = rank_pattern((4, 3), c, device)
    out = C.all_reduce(x, mesh, dim, "sum")
    expected = rank_pattern((4, 3), 0, device) * (g * (g + 1) / 2)
    torch.testing.assert_close(out, expected)
    # the in-place contract: all_reduce returns its input, no new allocation
    assert out is x, "all_reduce must operate in place and return the input tensor"


def test_all_reduce_avg(mesh, dim, device):
    # exercises the gloo emulation path (sum + local divide) -- the whole
    # reason op is a string
    c, g = mesh.coordinate(dim), mesh.size(dim)
    x = rank_pattern((4, 3), c, device)
    out = C.all_reduce(x, mesh, dim, "avg")
    expected = rank_pattern((4, 3), 0, device) * ((g + 1) / 2)
    torch.testing.assert_close(out, expected)


def test_all_reduce_max(mesh, dim, device):
    c, g = mesh.coordinate(dim), mesh.size(dim)
    x = rank_pattern((4, 3), c, device)
    out = C.all_reduce(x, mesh, dim, "max")
    expected = rank_pattern((4, 3), 0, device) * g
    torch.testing.assert_close(out, expected)


def test_broadcast(mesh, dim, device):
    # src=1, deliberately not 0, to catch coordinate-vs-global-rank confusion
    c, g = mesh.coordinate(dim), mesh.size(dim)
    if g < 2:
        raise Skip("needs group size >= 2")
    if c == 1:
        x = rank_pattern((4, 3), 1, device)
    else:
        x = torch.full((4, 3), -999.0, device=device)
    out = C.broadcast(x, mesh, dim, src=1)
    torch.testing.assert_close(out, rank_pattern((4, 3), 1, device))


def test_all_gather(mesh, dim, device):
    c, g = mesh.coordinate(dim), mesh.size(dim)
    x = rank_pattern((2, 3), c, device)

    out = C.all_gather(x, mesh, dim, gather_dim=0)
    expected = torch.cat([rank_pattern((2, 3), k, device) for k in range(g)], dim=0)
    assert out.shape == (2 * g, 3), f"gather_dim=0 shape {tuple(out.shape)} != {(2 * g, 3)}"
    torch.testing.assert_close(out, expected)

    out = C.all_gather(x, mesh, dim, gather_dim=-1)
    expected = torch.cat([rank_pattern((2, 3), k, device) for k in range(g)], dim=-1)
    assert out.shape == (2, 3 * g), f"gather_dim=-1 shape {tuple(out.shape)} != {(2, 3 * g)}"
    torch.testing.assert_close(out, expected)

    # non-contiguous input: (3, 2) pattern transposed to a (2, 3) view.
    # the wrapper may either handle it (result must be correct) or reject it
    # loudly; only silent corruption is a failure.
    nc = rank_pattern((3, 2), c, device).t()
    assert not nc.is_contiguous()
    try:
        out = C.all_gather(nc, mesh, dim, gather_dim=0)
    except Exception:
        pass  # rejecting non-contiguous inputs is an accepted contract
    else:
        expected = torch.cat(
            [rank_pattern((3, 2), k, device).t() for k in range(g)], dim=0
        )
        torch.testing.assert_close(
            out, expected, msg="all_gather silently corrupted a non-contiguous input"
        )


def test_reduce_scatter(mesh, dim, device):
    c, g = mesh.coordinate(dim), mesh.size(dim)
    x = rank_pattern((4 * g, 3), c, device)
    out = C.reduce_scatter(x, mesh, dim, "sum", scatter_dim=0)
    summed = rank_pattern((4 * g, 3), 0, device) * (g * (g + 1) / 2)
    assert out.shape == (4, 3), f"shape {tuple(out.shape)} != (4, 3)"
    torch.testing.assert_close(out, summed[4 * c : 4 * c + 4])

    # the loud-assert contract: shape[scatter_dim] % g != 0 must raise
    if g > 1:
        bad = rank_pattern((4 * g + 1, 3), c, device)
        try:
            C.reduce_scatter(bad, mesh, dim, "sum", scatter_dim=0)
        except Exception:
            pass
        else:
            raise AssertionError(
                "reduce_scatter accepted a shape not divisible by group size"
            )


def test_fsdp_identity(mesh, dim, device):
    # the identity fsdp is built on:
    # all_gather(reduce_scatter(x)) == all_reduce(x)
    c, g = mesh.coordinate(dim), mesh.size(dim)
    lhs = C.all_gather(
        C.reduce_scatter(rank_pattern((4 * g, 3), c, device), mesh, dim, "sum"),
        mesh,
        dim,
        gather_dim=0,
    )
    rhs = C.all_reduce(rank_pattern((4 * g, 3), c, device), mesh, dim, "sum")
    torch.testing.assert_close(lhs, rhs)


def test_all_to_all(mesh, dim, device):
    # encode (sender, chunk) in the values: rank at coord i holds a (g, K)
    # tensor whose row j is full(i*100 + j). after the exchange, rank i must
    # hold row j == full(j*100 + i): chunk i from every sender j, ordered by
    # sender.
    c, g = mesh.coordinate(dim), mesh.size(dim)
    K = 3
    x = torch.cat(
        [torch.full((1, K), float(c * 100 + j), device=device) for j in range(g)]
    )
    out = C.all_to_all(x, mesh, dim, split_dim=0, concat_dim=0)
    expected = torch.cat(
        [torch.full((1, K), float(j * 100 + c), device=device) for j in range(g)]
    )
    torch.testing.assert_close(out, expected)

    # involution: exchanging again restores the original
    back = C.all_to_all(out, mesh, dim, split_dim=0, concat_dim=0)
    torch.testing.assert_close(back, x)


def test_ring_send_recv(mesh, dim, device):
    # a deadlocked implementation fails via the process-group timeout in
    # _worker rather than hanging forever
    c, g = mesh.coordinate(dim), mesh.size(dim)
    x = rank_pattern((4,), c, device)
    recv = C.ring_send_recv(x, mesh, dim)
    torch.testing.assert_close(recv, rank_pattern((4,), (c - 1) % g, device))
    assert recv is not x, "ring_send_recv must return a newly allocated tensor"

    # g hops around the ring brings every tensor back home
    t = x
    for _ in range(g):
        t = C.ring_send_recv(t, mesh, dim)
    torch.testing.assert_close(t, x)


def test_async_matches_sync(mesh, dim, device):
    # correctness of the (tensor, work) handle path; overlap itself is
    # measured later in the ddp profiler trace, not asserted here
    c = mesh.coordinate(dim)
    res = C.all_reduce(rank_pattern((4, 3), c, device), mesh, dim, "sum", async_op=True)
    assert isinstance(res, tuple) and len(res) == 2, (
        "async_op=True must return (tensor, work)"
    )
    out, work = res
    work.wait()
    sync = C.all_reduce(rank_pattern((4, 3), c, device), mesh, dim, "sum")
    torch.testing.assert_close(out, sync)


def test_subgroup_isolation(mesh, dim, device):
    # THE group-construction test: all_reduce along one dim of a 2-d mesh
    # must only combine ranks in that dim's subgroup. runs against the real
    # core/mesh.py, not the FlatMesh passed in.
    try:
        # adjust this import to whatever core/mesh.py ends up exporting
        from torchure.core.mesh import init_mesh
    except ImportError:
        raise Skip("core/mesh.py not implemented yet")
    if dist.get_world_size() != 4:
        raise Skip("needs exactly 4 ranks (2x2 mesh); rerun with --world-size 4")

    m = init_mesh({"dp": 2, "tp": 2})
    rank = dist.get_rank()
    # powers of two: every subset of ranks has a unique sum, so any wrong
    # grouping is unambiguous
    world_sum = sum(2.0**r for r in range(4))
    for d in ("tp", "dp"):
        members = dist.get_process_group_ranks(m.get_group(d))
        expected = sum(2.0**r for r in members)
        assert rank in members, f"rank {rank} not in its own {d} group {members}"
        assert expected != world_sum, f"{d} group is the whole world: {members}"
        out = C.all_reduce(torch.full((4,), 2.0**rank, device=device), m, d, "sum")
        torch.testing.assert_close(out, torch.full((4,), expected, device=device))


TESTS = [
    test_all_reduce_sum,
    test_all_reduce_avg,
    test_all_reduce_max,
    test_broadcast,
    test_all_gather,
    test_reduce_scatter,
    test_fsdp_identity,
    test_all_to_all,
    test_ring_send_recv,
    test_async_matches_sync,
    test_subgroup_isolation,
]


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


def _worker(rank: int, world_size: int, backend: str, only: str | None = None):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    # short timeout so a deadlocked collective fails the run instead of
    # hanging the terminal (gloo enforces this; nccl needs it too).
    dist.init_process_group(
        backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    mesh, dim = FlatMesh(), "dp"
    tests = [t for t in TESTS if only is None or only in t.__name__]
    if not tests:
        raise SystemExit(f"--only {only!r} matched no tests")
    failed = []
    for test in tests:
        try:
            test(mesh, dim, device)
            err = None
        except NotImplementedError:
            err = "NOT IMPLEMENTED"
        except Skip as e:
            err = f"SKIP: {e}"
        except AssertionError as e:
            err = f"FAIL: {e}"
        dist.barrier()
        if rank == 0:
            print(f"[{test.__name__}] {err or 'PASS'}")
        if err and not err.startswith("SKIP"):
            failed.append(test.__name__)
    dist.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--only", help="run only tests whose name contains this substring")
    args = parser.parse_args()

    if "RANK" in os.environ:  # launched under torchrun
        _worker(
            int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), args.backend, args.only
        )
    else:
        mp.spawn(
            _worker, args=(args.world_size, args.backend, args.only), nprocs=args.world_size
        )
