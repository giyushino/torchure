"""
parity tests for torchure/parallelism/data_parallel.py.

run (gloo/cpu, no gpus needed):
    uv run tests/data_parallel.py --world-size 4

the invariant (roadmap phase 1 exit criterion): N ranks x bs=1 with ddp grad
sync must produce the same grads as 1 process x bs=N -- averaging per-rank
mean-loss grads equals the global mean-loss grad when shards are equal sized.
the batch is deterministic, so every rank recomputes the single-process
reference locally and no gathers are needed for the comparison.

the model is a tiny real Qwen3, not an mlp, so the tied embedding (one param,
two grad contributions, AccumulateGrad must fire exactly once) and the
rmsnorm/rope structure are exercised. bucket_mb is swept to hit all three
bucket shapes: one flat bucket, many flat buckets, and single-param
in-place buckets (cap=0 makes every param "oversized").
"""

import argparse
import os
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torchure.core.mesh import Mesh
from torchure.models.qwen3.qwen3 import Qwen3
from torchure.parallelism.data_parallel import DDP

CFG = {
    "num_layers": 2,
    "num_heads": 2,
    "num_kv_heads": 1,
    "emb_dim": 32,
    "head_dim": 16,
    "vocab_size": 64,
}
SEQ = 16


def build_model(seed: int) -> Qwen3:
    torch.manual_seed(seed)
    model = Qwen3(**CFG)
    model.init_weights()
    return model


def global_batch(world: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    return torch.randint(0, CFG["vocab_size"], (world, SEQ), generator=g)


def loss_fn(model: Qwen3, ids: torch.Tensor) -> torch.Tensor:
    # plain logits + CE (next-token shift); the fused-CE path is objective
    # machinery, not what's under test here
    logits = model(ids)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, CFG["vocab_size"]), ids[:, 1:].reshape(-1)
    )


def test_replicate(mesh, dim):
    # different seed per rank; the ddp ctor broadcast must overwrite every
    # rank with dp-coord-0's (== seed 0's) weights, bitwise
    model = build_model(seed=dist.get_rank())
    DDP(model, mesh, dim)
    reference = build_model(seed=0)
    for (name, p), (_, p_ref) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        torch.testing.assert_close(p, p_ref, rtol=0, atol=0, msg=f"param {name} not replicated")


def test_grad_parity(mesh, dim, overlap: bool, bucket_mb: float):
    model = build_model(seed=dist.get_rank())
    ddp = DDP(model, mesh, dim, bucket_mb=bucket_mb, overlap=overlap)

    reference = build_model(seed=0)
    batch = global_batch(mesh.size(dim))
    loss_fn(reference, batch).backward()

    local = batch[mesh.coordinate(dim)].unsqueeze(0)
    for _ in range(2):  # two full cycles: catches rearm/handle-clearing bugs
        loss_fn(model, local).backward()
        ddp.sync()
        for (name, p), (_, p_ref) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            try:
                torch.testing.assert_close(p.grad, p_ref.grad, rtol=1e-4, atol=1e-6)
            except AssertionError as e:
                raise AssertionError(f"param {name}: {e}") from None
        for p in model.parameters():
            p.grad = None  # what optimizer.zero_grad(set_to_none=True) does


TESTS = [
    ("test_replicate", test_replicate),
    # 25MB cap: the whole tiny model lands in ONE flat bucket
    ("test_parity_one_bucket", partial(test_grad_parity, overlap=True, bucket_mb=25)),
    # 8KB cap: many flat buckets + the embedding goes single-param/in-place
    ("test_parity_many_buckets", partial(test_grad_parity, overlap=True, bucket_mb=0.008)),
    # 0 cap: every param is "oversized" -> all in-place, no flat buffers
    ("test_parity_inplace", partial(test_grad_parity, overlap=True, bucket_mb=0.0)),
    # the roadmap 1.1 correctness baseline / overlap ablation path
    ("test_parity_no_overlap", partial(test_grad_parity, overlap=False, bucket_mb=0.008)),
]


def _worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29532")
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    mesh, dim = Mesh({"dp": world_size}), "dp"
    failed = []
    for name, test in TESTS:
        try:
            test(mesh, dim)
            err = None
        except AssertionError as e:
            err = f"FAIL: {e}"
        except Exception as e:
            err = f"ERROR: {type(e).__name__}: {e}"
        dist.barrier()
        if rank == 0:
            print(f"[{name}] {err or 'PASS'}")
        if err:
            failed.append(name)
    dist.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    mp.spawn(_worker, args=(args.world_size,), nprocs=args.world_size)
