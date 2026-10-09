"""
tests for torchure/parallelism/fsdp2.py.

run (gloo/cpu, no gpus needed):
    uv run tests/fsdp2.py --world-size 4

same discipline as tests/data_parallel.py: mp.spawn, gloo, a tiny REAL qwen3
rather than an mlp (the tied embedding and the rmsnorm/rope structure are what
break), a deterministic batch every rank recomputes locally so no gathers are
needed for the comparison, and a TESTS list swept over parameters.

everything here runs under MixedPrecisionPolicy.fp32() -- gloo's bf16 support
is patchy, and a parity test that has to loosen its tolerance to accommodate
the comm dtype isn't testing what it claims to. the bf16 comm path is a gpu
concern, measured in CHANGES.md rather than asserted here.

===========================================================================
this file is written milestone by milestone
===========================================================================

ACTIVE: shard construction. no forward, no hooks, no autograd -- just "the
model is sharded and nothing was lost". these fail until FSDPUnit._build_shards
exists, which is the point.

STILL OWED (see notes/FSDP2_SCOPE.md for the full list):
  - forward parity + storage-actually-freed between units
  - grad parity vs single process, and the ddp cross-check (same model, same
    batch, ddp and fsdp grads must agree -- the test that catches
    reduce-scatter scaling and ordering mistakes)
  - multi-step loss curve vs single process (optimizer-state sharding bugs are
    invisible to a single backward)
  - grad accumulation, with the collective count asserted
  - clip_grad_norm with shard_dims=("dp_shard",) vs the single-process norm
  - hsdp: dp_replicate=2 x dp_shard=2 grads == dp=4 ddp grads
  - prefetch order: recorded order == module execution order, exactly reversed
    in backward

===========================================================================
the private surface these tests reach into
===========================================================================

deliberate, and documented here so a design change has one place to look:

    fsdp.units          list[FSDPUnit], in construction order
    unit.params         the unit's _ParamInfo entries (.param is the tensor)

reaching in is the whole point for the invariants that are invisible from the
outside -- an fsdp that gathers twice, or never frees, still trains perfectly
and just doesn't do its job. tests/data_parallel.py reaches into ddp._works for
the same reason.
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torchure.core.collective import all_gather
from torchure.core.mesh import Mesh
from torchure.models.qwen3.qwen3 import Qwen3
from torchure.parallelism.fsdp2 import FSDP, MixedPrecisionPolicy

# head_dim 16 is the tightest dim-0 in the model (q_norm/k_norm are 16 wide),
# so it's what bounds the world size this suite can run at: at --world-size 8
# each rank owns 2 elements of them, at 16 the divisibility assert fires. that
# is the intended behaviour, not a test bug -- v0 asserts rather than pads.
CFG = {
    "num_layers": 2,
    "num_heads": 2,
    "num_kv_heads": 1,
    "emb_dim": 32,
    "head_dim": 16,
    "vocab_size": 64,
}
SEQ = 16


def build_model(seed: int = 0) -> Qwen3:
    torch.manual_seed(seed)
    model = Qwen3(**CFG)
    model.init_weights()
    return model


def global_batch(n_seqs: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    return torch.randint(0, CFG["vocab_size"], (n_seqs, SEQ), generator=g)


def loss_fn(model: Qwen3, ids: torch.Tensor) -> torch.Tensor:
    # plain logits + CE (next-token shift). the fused-CE path consumes
    # lm_head.weight AFTER the model's forward returns, which is the root
    # unit's whole reason for existing -- it gets its own test once the
    # forward path lands, not this one.
    logits = model(ids)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, CFG["vocab_size"]), ids[:, 1:].reshape(-1)
    )


def unit_params(unit) -> list[torch.nn.Parameter]:
    """the tensors a unit owns. one place to fix if _ParamInfo changes shape."""
    return [info.param for info in unit.params]


def fsdp_fp32(model, mesh, **kwargs) -> FSDP:
    return FSDP(model, mesh, shard_dim="dp_shard", policy=MixedPrecisionPolicy.fp32(), **kwargs)


# --------------------------------------------------------------------------
# milestone 1: shard construction
# --------------------------------------------------------------------------


def test_unit_partition(mesh):
    """
    every param lands in exactly one unit, and the tie survives sharding.

    the tie assert is the load-bearing one. `lm_head.weight is
    token_emb.weight` is ONE Parameter object under two attributes, so
    `p.data = shard` propagates to both for free -- but rebinding
    `module._parameters[name]` instead silently breaks the tie, and you get two
    independently gathered copies of the largest weight in the model with grads
    flowing to only one. that failure trains fine and quietly diverges.

    note `named_parameters()` dedups by default (lm_head.weight never appears
    in it), so the count comparison below would pass even with the tie broken.
    the `is` check is what actually catches it.
    """
    model = build_model()
    expected = {id(p) for p in model.parameters()}

    fsdp = fsdp_fp32(model, mesh)

    seen: dict[int, str] = {}
    for unit in fsdp.units:
        for p in unit_params(unit):
            assert id(p) not in seen, (
                f"param claimed by both {seen[id(p)]} and {unit.name} -- a "
                "double-claimed param is gathered twice and its grad "
                "reduce-scattered twice"
            )
            seen[id(p)] = unit.name

    missing = expected - set(seen)
    assert not missing, (
        f"{len(missing)} param(s) in no unit at all -- they get no gather and "
        "no reduce-scatter, so they drift between ranks silently"
    )
    assert not (set(seen) - expected), "unit claims a param the model doesn't have"

    assert model.lm_head.weight is model.token_emb.weight, (
        "the embedding tie was broken by sharding: lm_head.weight and "
        "token_emb.weight are no longer the same object"
    )
    # and both must be in the SAME unit -- separate units double-gather one
    # storage even when the objects are still identical
    emb_units = {u.name for u in fsdp.units if any(
        p is model.token_emb.weight for p in unit_params(u)
    )}
    assert len(emb_units) == 1, f"tied embedding appears in units {emb_units}"


def test_shard_roundtrip(mesh):
    """
    all_gather of every rank's shard reconstructs the original weights.

    bit-exact (`rtol=0, atol=0`): sharding is a copy, there is no arithmetic
    here to be tolerant of. a tolerance would hide an off-by-one in the offset
    math, which is exactly the bug this is for.

    gathers `p.data` rather than reaching into shard_flat, so it holds whether
    or not the shards are backed by one flat buffer per unit.
    """
    reference = build_model()
    full = {name: p.detach().clone() for name, p in reference.named_parameters()}

    model = build_model()
    fsdp_fp32(model, mesh)

    for name, p in model.named_parameters():
        ref = full[name]
        assert p.shape[0] * mesh.size("dp_shard") == ref.shape[0], (
            f"param {name}: local shard dim-0 {p.shape[0]} x {mesh.size('dp_shard')} "
            f"!= full {ref.shape[0]}"
        )
        gathered = all_gather(p.detach().contiguous(), mesh, "dp_shard", gather_dim=0)
        try:
            torch.testing.assert_close(gathered, ref, rtol=0, atol=0)
        except AssertionError as e:
            raise AssertionError(f"param {name} did not round-trip: {e}") from None


def test_shard_memory(mesh):
    """
    per-rank param numel is total/dp_shard. this is the reason the phase
    exists, it's free to check on cpu, and it catches a param that quietly
    never got sharded -- which every correctness test above would still pass.
    """
    shard_size = mesh.size("dp_shard")
    before = sum(p.numel() for p in build_model().parameters())

    model = build_model()
    fsdp_fp32(model, mesh)
    after = sum(p.numel() for p in model.parameters())

    assert after == before // shard_size, (
        f"per-rank params {after} != {before}/{shard_size} = {before // shard_size} "
        f"({before - after * shard_size} elements unsharded)"
    )


def test_noop_at_size_one(mesh_replicate_only):
    """
    dp_shard=1 must be a TOTAL no-op, the way DDP is at dp=1: no units, no
    buffers, no collectives, and shard_dims=() so the trainer's checkpoint
    guard lets the run through (at size 1 rank 0's tensors really are the
    model).
    """
    model = build_model()
    before = [p.detach().clone() for p in model.parameters()]

    fsdp = fsdp_fp32(model, mesh_replicate_only)

    assert fsdp.shard_dims == (), f"shard_dims {fsdp.shard_dims} != () at dp_shard=1"
    assert not fsdp.units, f"{len(fsdp.units)} unit(s) built at dp_shard=1"
    for p, ref in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(p, ref, rtol=0, atol=0, msg="params touched at dp_shard=1")
    fsdp.sync()  # must not hang or raise


TESTS = [
    ("test_noop_at_size_one", test_noop_at_size_one, "replicate"),
    ("test_unit_partition", test_unit_partition, "shard"),
    ("test_shard_roundtrip", test_shard_roundtrip, "shard"),
    ("test_shard_memory", test_shard_memory, "shard"),
]


def _worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )

    # both meshes are built up front: dist.new_group is collective, so every
    # rank must create every group in the same order regardless of which tests
    # end up running.
    shard_mesh = Mesh({"dp_replicate": 1, "dp_shard": world_size})
    shard_mesh.flatten(("dp_replicate", "dp_shard"), "dp")
    replicate_mesh = Mesh({"dp_replicate": world_size, "dp_shard": 1})
    replicate_mesh.flatten(("dp_replicate", "dp_shard"), "dp")
    meshes = {"shard": shard_mesh, "replicate": replicate_mesh}

    failed = []
    for name, test, which in TESTS:
        try:
            test(meshes[which])
            err = None
        except AssertionError as e:
            err = f"FAIL: {e}"
        except NotImplementedError:
            err = "TODO: not implemented yet"
        except Exception as e:
            err = f"ERROR: {type(e).__name__}: {e}"
        dist.barrier()
        if rank == 0:
            print(f"[{name}] {err or 'PASS'}")
        # NotImplementedError is the expected state of an unwritten milestone,
        # not a failure -- the suite goes red on real breakage only
        if err and not err.startswith("TODO"):
            failed.append(name)
    dist.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    assert args.world_size >= 2, "fsdp at dp_shard=1 is a no-op; use >= 2 ranks"
    mp.spawn(_worker, args=(args.world_size,), nprocs=args.world_size)
