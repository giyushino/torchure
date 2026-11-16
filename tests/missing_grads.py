"""
grad sync when some params get no grad, for ddp and fsdp.

run (gloo/cpu, no gpus needed):
    uv run tests/missing_grads.py --world-size 4

a param can end a backward with no grad: an moe expert no token picked, or a
param the forward never touched. ranks see different data, so they can
disagree on WHICH params those are. the sync has to (a) not wait forever on
the missing grads, (b) count a missing grad as zeros in the average, and (c)
keep every rank's collectives paired although the ranks notice at different
moments. (c) failing is a hang or a shape error, not a wrong number.

the model is a toy with the three cases in it: a root param nothing uses, a
whole submodule ("expert") each block uses only on some ranks and some
microbatches, and ordinary params. the expected grad is the mean over ranks
of each rank's local grad, zeros where it had none; every rank can compute
all of them locally because the model and data are deterministic.
"""

import argparse
import copy
import os
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from torchure.core.mesh import Mesh
from torchure.core.placement import padded_chunk
from torchure.parallelism.data_parallel import DDP
from torchure.parallelism.fsdp2 import FSDP

DIM = 8
LAYERS = 3


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(DIM, DIM)
        self.expert = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor, use_expert: bool) -> torch.Tensor:
        x = x + torch.tanh(self.lin(x))
        if use_expert:
            x = x + torch.tanh(self.expert(x))
        return x


class Sparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(DIM, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.head = nn.Linear(DIM, 1)
        self.unused = nn.Parameter(torch.ones(DIM))

    def forward(self, x: torch.Tensor, pattern: list[bool]) -> torch.Tensor:
        x = self.inp(x)
        for block, use in zip(self.blocks, pattern, strict=True):
            x = block(x, use)
        return self.head(x).pow(2).mean()


def build_model() -> Sparse:
    torch.manual_seed(0)
    return Sparse()


def batch_of(rank: int, micro: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(100 * rank + micro)
    return torch.randn(4, DIM, generator=g)


def pattern_of(rank: int, micro: int) -> list[bool]:
    """which blocks use their expert: differs by rank, block and microbatch."""
    return [(rank + layer + micro) % 2 == 0 for layer in range(LAYERS)]


def expected_grads(world: int, microbatches: int) -> dict[str, torch.Tensor]:
    """mean over ranks of the accumulated local grads, zeros for missing ones."""
    base = build_model()
    total = {name: torch.zeros_like(p) for name, p in base.named_parameters()}
    for rank in range(world):
        model = copy.deepcopy(base)
        for micro in range(microbatches):
            loss = model(batch_of(rank, micro), pattern_of(rank, micro)) / microbatches
            loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                total[name] += p.grad / world
    return total


def run_step(model: Sparse, parallel, microbatches: int) -> None:
    rank = dist.get_rank()
    parallel.requires_sync = False
    for micro in range(microbatches):
        if micro == microbatches - 1:
            parallel.requires_sync = True
        loss = model(batch_of(rank, micro), pattern_of(rank, micro)) / microbatches
        loss.backward()
    parallel.sync()


def compare(model: Sparse, expected: dict, local_of) -> list[str]:
    errors = []
    for name, p in model.named_parameters():
        want = local_of(expected[name])
        if p.grad is None:
            errors.append(f"{name}: no grad after sync")
        elif p.grad.shape != want.shape:
            errors.append(f"{name}: shape {tuple(p.grad.shape)} != {tuple(want.shape)}")
        elif not torch.allclose(p.grad, want, rtol=1e-4, atol=1e-6):
            errors.append(f"{name}: max err {(p.grad - want).abs().max():.3e}")
    return errors


def test_ddp(mesh, dim, bucket_mb: float, microbatches: int):
    model = build_model()
    ddp = DDP(model, mesh, dim, bucket_mb=bucket_mb)
    expected = expected_grads(mesh.size(dim), microbatches)
    errors = []
    # two steps: the second catches bucket state the first left behind
    for step in range(2):
        run_step(model, ddp, microbatches)
        errors += [f"step{step} {e}" for e in compare(model, expected, lambda g: g)]
        for p in model.parameters():
            p.grad = None
    assert not errors, " | ".join(errors[:4])


def test_fsdp(mesh, dim, microbatches: int, prefetch: int):
    model = build_model()
    fsdp = FSDP(
        model, mesh, model.blocks, shard_dim=dim, replicate_dim=None, prefetch=prefetch
    )
    expected = expected_grads(mesh.size(dim), microbatches)

    def local(grad: torch.Tensor) -> torch.Tensor:
        # this rank's rows, zero-padded to the common shard length
        length, start, valid = padded_chunk(grad.shape[0], mesh.size(dim), mesh.coordinate(dim))
        out = grad.new_zeros((length, *grad.shape[1:]))
        out[:valid] = grad[start:start + valid]
        return out

    errors = []
    for step in range(2):
        run_step(model, fsdp, microbatches)
        errors += [f"step{step} {e}" for e in compare(model, expected, local)]
        if fsdp.unsharded_bytes_live():
            errors.append(f"step{step} a unit was left gathered after sync")
        for p in model.parameters():
            p.grad = None
    assert not errors, " | ".join(errors[:4])


TESTS = [
    ("ddp_one_bucket", partial(test_ddp, bucket_mb=25, microbatches=1)),
    ("ddp_many_buckets", partial(test_ddp, bucket_mb=0.0003, microbatches=1)),
    ("ddp_inplace", partial(test_ddp, bucket_mb=0.0, microbatches=1)),
    ("ddp_accum_many_buckets", partial(test_ddp, bucket_mb=0.0003, microbatches=3)),
    ("fsdp", partial(test_fsdp, microbatches=1, prefetch=1)),
    ("fsdp_no_prefetch", partial(test_fsdp, microbatches=1, prefetch=0)),
    ("fsdp_accum", partial(test_fsdp, microbatches=3, prefetch=1)),
]


def _worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29557")
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    mesh, dim = Mesh({"dp": world_size}), "dp"
    failed = []
    for name, test in TESTS:
        try:
            test(mesh, dim)
            msg = "PASS"
        except AssertionError as e:
            msg = f"FAIL: {e}"
        except Exception as e:  # noqa: BLE001
            msg = f"ERROR: {type(e).__name__}: {e}"
        verdicts = [None] * world_size
        dist.all_gather_object(verdicts, msg)
        bad = {r: v for r, v in enumerate(verdicts) if v != "PASS"}
        if rank == 0:
            print(f"[{name}] " + (" || ".join(f"rank{r}: {v}" for r, v in bad.items()) or "PASS"),
                  flush=True)
        if bad:
            failed.append(name)
    dist.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    mp.spawn(_worker, args=(args.world_size,), nprocs=args.world_size)
