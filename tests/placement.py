"""
unit tests for torchure/core/placement.py.

run:
    uv run tests/placement.py

the only test file in the tree with no distributed harness: placement types
touch no process group by construction, so this is plain python and runs in
milliseconds. if a test here ever needs mp.spawn, something that belongs in
dtensor.py has leaked into placement.py.
"""

import torch

from torchure.core.placement import (
    Partial,
    Placement,
    Replicate,
    Shard,
    global_shape,
    local_shape,
)


def test_value_semantics():
    # equality and hashing have to travel together: redistribute dispatches on
    # placement values and phase 3 puts them in checkpoint metadata dicts
    assert Shard(0) == Shard(0)
    assert Shard(0) != Shard(1)
    assert Replicate() == Replicate()
    assert Shard(0) != Replicate()
    assert Partial("sum") == Partial("sum")
    # the factor-of-group_size bug class: these must NOT compare equal
    assert Partial("sum") != Partial("avg")

    table = {Shard(0): "a", Replicate(): "b", Partial("avg"): "c"}
    assert table[Shard(0)] == "a"
    assert table[Replicate()] == "b"
    assert table[Partial("avg")] == "c"
    assert len({Shard(0), Shard(0), Replicate()}) == 2

    assert all(isinstance(p, Placement) for p in (Shard(1), Replicate(), Partial()))
    assert Partial().op == "sum", "a gradient is a sum by default"


def test_repr():
    # these end up in assert messages and checkpoint metadata
    assert repr(Shard(2)) == "Shard(2)"
    assert repr(Replicate()) == "Replicate()"
    assert repr(Partial("avg")) == "Partial(avg)"


def test_negative_dim_rejected():
    # Shard(-1) and Shard(ndim-1) would describe the same layout while hashing
    # differently, silently splitting one placement's identity in two
    for bad in (-1, -2):
        try:
            Shard(bad)
        except AssertionError:
            continue
        raise AssertionError(f"Shard({bad}) should have been rejected")


def test_bad_partial_op_rejected():
    try:
        Partial("mean")  # not one of sum/avg/max/min
    except AssertionError:
        return
    raise AssertionError("Partial('mean') should have been rejected")


def test_local_and_global_shape():
    s = Shard(0)
    assert s.local_shape((8, 3), 4) == (2, 3)
    assert s.global_shape((2, 3), 4) == (8, 3)
    assert Shard(1).local_shape((3, 8), 2) == (3, 4)
    assert Shard(1).global_shape((3, 4), 2) == (3, 8)

    # multi-axis: only Shard divides, and each axis divides its own dim
    placements = (Replicate(), Shard(0))
    assert local_shape((8, 3), placements, (2, 4)) == (2, 3)
    assert global_shape((2, 3), placements, (2, 4)) == (8, 3)

    # Partial leaves the shape alone -- it's an unreduced full tensor
    assert local_shape((8, 3), (Partial("avg"),), (4,)) == (8, 3)
    assert global_shape((8, 3), (Partial("avg"),), (4,)) == (8, 3)

    # round trip
    for shape, group in (((8, 3), 4), ((12,), 3), ((6, 5, 2), 2)):
        assert Shard(0).global_shape(Shard(0).local_shape(shape, group), group) == shape


def test_indivisible_asserts_loudly():
    # v0 policy: assert, never pad, never silently truncate
    try:
        Shard(0).local_shape((7, 3), 2)
    except AssertionError as e:
        assert "not divisible" in str(e), f"unhelpful message: {e}"
        return
    raise AssertionError("indivisible shard should have raised")


def test_local_slice():
    s = Shard(0)
    x = torch.arange(12).reshape(6, 2)
    assert torch.equal(s.local_slice(x, 3, 0), x[0:2])
    assert torch.equal(s.local_slice(x, 3, 1), x[2:4])
    assert torch.equal(s.local_slice(x, 3, 2), x[4:6])

    # concatenating every rank's slice in coordinate order rebuilds the whole
    # tensor -- this is the invariant all_gather(gather_dim=0) relies on
    rebuilt = torch.cat([s.local_slice(x, 3, c) for c in range(3)], dim=0)
    assert torch.equal(rebuilt, x)

    # dim 1 works the same way
    y = torch.arange(12).reshape(2, 6)
    assert torch.equal(Shard(1).local_slice(y, 2, 1), y[:, 3:6])

    # a slice is a VIEW: callers that want to own the storage must clone, and
    # fsdp's master shards do exactly that
    assert s.local_slice(x, 3, 0)._base is not None

    try:
        s.local_slice(x, 3, 3)  # coord out of range
    except AssertionError:
        pass
    else:
        raise AssertionError("out-of-range coordinate should have raised")


TESTS = [
    test_value_semantics,
    test_repr,
    test_negative_dim_rejected,
    test_bad_partial_op_rejected,
    test_local_and_global_shape,
    test_indivisible_asserts_loudly,
    test_local_slice,
]


if __name__ == "__main__":
    failed = []
    for test in TESTS:
        try:
            test()
            print(f"[{test.__name__}] PASS")
        except AssertionError as e:
            print(f"[{test.__name__}] FAIL: {e}")
            failed.append(test.__name__)
        except Exception as e:  # noqa: BLE001
            print(f"[{test.__name__}] ERROR: {type(e).__name__}: {e}")
            failed.append(test.__name__)
    raise SystemExit(1 if failed else 0)
