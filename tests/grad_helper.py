"""
tests for torchure/core/grad_helper.py -- global grad-norm clipping.

run (gloo/cpu, no gpus needed):
    uv run tests/grad_helper.py --world-size 2

two things are under test and they fail in very different ways:

- the LOCAL path (shard_dims=()), which is what pure DDP uses: grads are
  replicas post-sync, so every rank already holds the whole gradient vector and
  the norm is a local computation. checked against
  `torch.nn.utils.clip_grad_norm_` on a real (tiny) Qwen3, so the tied
  embedding -- one param reached through two modules, and `model.parameters()`
  must yield it once or the norm double-counts it -- is exercised.

- the SHARDED path (shard_dims=("dp",)), which nothing in the trainer uses yet
  but fsdp2/tp inherit directly: each rank holds a slice of the gradient
  vector, so the norm needs sum-of-p-th-powers reduced across the axis. norms
  themselves are NOT additive; the test pins a known vector split across ranks
  against the true norm of the unsplit vector, which catches the two obvious
  wrong implementations (all_reduce the norms with "sum", or with "max").

`test_below_threshold_is_noop` is a regression test: `clip_coef` was once
clamped to `max_norm` instead of `1.0`, which silently AMPLIFIED every
gradient that did not need clipping whenever max_norm > 1 (and shrank it when
max_norm < 1). it is correct at exactly max_norm == 1.0, which is the value
every config ships, so it passed casual testing.
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from torchure.core.grad_helper import clip_grad_norm
from torchure.core.mesh import Mesh
from torchure.models.qwen3.qwen3 import Qwen3

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


def backward_once(model: Qwen3, scale: float = 1.0) -> None:
    """populate .grad on every param. `scale` moves the norm above/below 1."""
    g = torch.Generator().manual_seed(1234)
    ids = torch.randint(0, CFG["vocab_size"], (2, SEQ), generator=g)
    logits = model(ids)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, CFG["vocab_size"]), ids[:, 1:].reshape(-1)
    )
    (loss * scale).backward()


def params_with_grads(grads: list[torch.Tensor]) -> list[nn.Parameter]:
    """bare params carrying the given grads -- no model, exact known norm."""
    out = []
    for g in grads:
        p = nn.Parameter(torch.zeros_like(g))
        p.grad = g.clone()
        out.append(p)
    return out


def test_matches_torch_reference(mesh, dim):
    """local path == torch's, on a real model incl. the tied embedding."""
    for max_norm in (0.1, 1.0, 5.0):
        mine, ref = build_model(), build_model()
        backward_once(mine)
        backward_once(ref)

        got = clip_grad_norm(mine.parameters(), None, max_norm)
        want = torch.nn.utils.clip_grad_norm_(ref.parameters(), max_norm)

        torch.testing.assert_close(
            got, want, msg=f"max_norm={max_norm}: returned norm differs from torch"
        )
        for (name, p), (_, p_ref) in zip(
            mine.named_parameters(), ref.named_parameters(), strict=True
        ):
            try:
                torch.testing.assert_close(p.grad, p_ref.grad, rtol=1e-5, atol=1e-7)
            except AssertionError as e:
                raise AssertionError(f"max_norm={max_norm}, param {name}: {e}") from None


def test_below_threshold_is_noop(mesh, dim):
    """norm < max_norm must leave grads BITWISE untouched, for every max_norm.

    regression: clamping clip_coef to max_norm instead of 1.0 scaled grads by
    max_norm here -- a 10x amplification at max_norm=10, invisible at 1.0.
    """
    grads = [torch.tensor([0.03, 0.04]), torch.zeros(3)]  # norm == 0.05
    for max_norm in (0.5, 1.0, 2.0, 10.0):
        params = params_with_grads(grads)
        before = [p.grad.clone() for p in params]
        total = clip_grad_norm(params, None, max_norm)

        torch.testing.assert_close(total, torch.tensor(0.05), rtol=1e-5, atol=1e-8)
        for i, (p, b) in enumerate(zip(params, before, strict=True)):
            torch.testing.assert_close(
                p.grad, b, rtol=0, atol=0,
                msg=f"max_norm={max_norm}: grad {i} was scaled by "
                    f"{(p.grad / b.clamp(min=1e-30))[0].item():.3f} though "
                    f"norm 0.05 < max_norm -- clip_coef must clamp to 1.0",
            )


def test_clips_to_max_norm(mesh, dim):
    """norm > max_norm: returns the PRE-clip norm, leaves post-clip norm at max."""
    params = params_with_grads([torch.tensor([3.0, 4.0])])  # norm == 5
    total = clip_grad_norm(params, None, 1.0)

    torch.testing.assert_close(total, torch.tensor(5.0))
    post = torch.linalg.vector_norm(params[0].grad)
    torch.testing.assert_close(post, torch.tensor(1.0), rtol=1e-5, atol=1e-6)
    # direction preserved: clipping is a rescale, not a per-element clamp
    torch.testing.assert_close(params[0].grad, torch.tensor([0.6, 0.8]), rtol=1e-5, atol=1e-6)


def test_max_norm_none_measures_only(mesh, dim):
    """max_norm=None -> return the norm for logging, touch nothing."""
    params = params_with_grads([torch.tensor([3.0, 4.0])])
    before = params[0].grad.clone()
    total = clip_grad_norm(params, None, None)

    torch.testing.assert_close(total, torch.tensor(5.0))
    torch.testing.assert_close(params[0].grad, before, rtol=0, atol=0)


def test_no_grads_asserts(mesh, dim):
    """a rank with no grads must fail loudly, not skip its collective.

    silently returning would deadlock the sharded path: this rank skips its
    all_reduce while every other rank enters one.
    """
    raised = False
    try:
        clip_grad_norm([nn.Parameter(torch.zeros(4))], None, 1.0)
    except AssertionError:
        raised = True
    assert raised, "clip_grad_norm must assert when no param carries a grad"


def test_sharded_norm(mesh, dim):
    """sharded path: the norm of the concatenation, not of the local slice.

    rank r holds one chunk of `full`. the returned norm must equal
    ||full||, and each rank's shard must come back scaled by max_norm/||full||.
    """
    world = mesh.size(dim)
    coord = mesh.coordinate(dim)
    full = torch.arange(1, 4 * world + 1, dtype=torch.float32)
    true_norm = full.norm()
    max_norm = (true_norm / 2).item()  # force clipping, expect a 0.5x scale

    shard = full.chunk(world)[coord]
    params = params_with_grads([shard])
    total = clip_grad_norm(params, mesh, max_norm, shard_dims=(dim,))

    torch.testing.assert_close(
        total, true_norm, rtol=1e-5, atol=1e-6,
        msg=f"sharded norm wrong: got {total.item()}, want {true_norm.item()} "
            f"(local slice norm is {shard.norm().item()}; summing NORMS across "
            f"ranks would give a different wrong answer than summing squares)",
    )
    torch.testing.assert_close(params[0].grad, shard * (max_norm / true_norm), rtol=1e-5, atol=1e-6)


def test_replicated_grads_are_not_reduced(mesh, dim):
    """the DDP case, and the trap the roadmap's 1.3 wording invites.

    post-sync DDP grads are REPLICAS: every rank already holds the whole
    gradient vector, so the local norm is the global norm and shard_dims must
    stay empty. all_reduce-ing squares over dp here would return
    sqrt(world) x the true norm and silently over-clip by that factor.
    """
    grads = [torch.tensor([3.0, 4.0]), torch.tensor([12.0, 0.0])]  # ||.|| == 13
    params = params_with_grads(grads)  # identical on every rank
    total = clip_grad_norm(params, mesh, 100.0)  # shard_dims=() by default

    torch.testing.assert_close(
        total, torch.tensor(13.0), rtol=1e-6, atol=1e-7,
        msg=f"replicated grads must not be reduced: got {total.item()}, want 13.0 "
            f"(a spurious sum over dp would give {13.0 * mesh.size(dim) ** 0.5:.3f})",
    )


TESTS = [
    ("test_matches_torch_reference", test_matches_torch_reference),
    ("test_below_threshold_is_noop", test_below_threshold_is_noop),
    ("test_clips_to_max_norm", test_clips_to_max_norm),
    ("test_max_norm_none_measures_only", test_max_norm_none_measures_only),
    ("test_no_grads_asserts", test_no_grads_asserts),
    ("test_sharded_norm", test_sharded_norm),
    ("test_replicated_grads_are_not_reduced", test_replicated_grads_are_not_reduced),
]


def _worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
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
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()
    mp.spawn(_worker, args=(args.world_size,), nprocs=args.world_size)
