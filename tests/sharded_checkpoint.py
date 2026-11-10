"""
the golden resumption test for sharded checkpointing (roadmap phase 3).

run (gloo/cpu, no gpus needed):
    uv run tests/sharded_checkpoint.py

each scenario is two separate launches with DIFFERENT world sizes / meshes:

    save    train 3 steps, checkpoint, train 3 more. alongside the checkpoint,
            write the ground truth as plain full tensors: model + adam state at
            the checkpoint, and the model after the last step.
    resume  build a differently-initialized model on a different mesh, load
            the checkpoint, and check (a) every rank's params and adam moments
            are BITWISE its slice of the ground truth, (b) training the same 3
            remaining steps lands on the same final model.

(a) is the resharding itself and has no excuse to be inexact. (b) is the
end-to-end claim -- nothing the run needs was left out of the checkpoint --
and is close rather than exact only because a different rank count reduces
grads in a different order.

the layouts crossed: fsdp at two shard counts, hsdp, and unsharded (which is
also the legacy `model.pt` on-disk format, in both directions).
"""

import os
import shutil
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torchure.checkpoint.checkpointer import Checkpointer, ShardLayout
from torchure.core import collective as C
from torchure.core.mesh import Mesh
from torchure.core.placement import Shard
from torchure.models.qwen3.qwen3 import Qwen3
from torchure.parallelism.data_parallel import DDP
from torchure.parallelism.fsdp2 import FSDP

CFG = {
    "num_layers": 2,
    "num_heads": 2,
    "num_kv_heads": 1,
    "emb_dim": 32,
    "head_dim": 16,
    "vocab_size": 64,
}
SEQ = 16
GLOBAL_BATCH = 4
CKPT_STEP = 3
END_STEP = 6

# (name, save mesh, resume mesh) as (dp_replicate, dp_shard)
SCENARIOS = [
    ("fsdp4 -> fsdp2", (1, 4), (1, 2)),
    ("fsdp2 -> fsdp4", (1, 2), (1, 4)),
    ("fsdp4 -> unsharded", (1, 4), (1, 1)),
    ("unsharded -> fsdp4", (1, 1), (1, 4)),
    ("hsdp2x2 -> fsdp4", (2, 2), (1, 4)),
    ("fsdp4 -> hsdp2x2", (1, 4), (2, 2)),
    ("fsdp4 -> fsdp4", (1, 4), (1, 4)),
]


def build(seed: int, mesh: Mesh):
    torch.manual_seed(seed)
    model = Qwen3(**CFG)
    model.init_weights()
    if mesh.size("dp_shard") > 1:
        parallel = FSDP(
            model, mesh, model.blocks,
            replicate_dim="dp_replicate" if mesh.size("dp_replicate") > 1 else None,
        )
    else:
        parallel = DDP(model, mesh, dim="dp")
    # built after sharding, so adam's moments are allocated sharded
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    return model, parallel, optimizer


def train_step(model, parallel, optimizer, mesh: Mesh, step: int) -> None:
    g = torch.Generator().manual_seed(1000 + step)
    batch = torch.randint(0, CFG["vocab_size"], (GLOBAL_BATCH, SEQ), generator=g)
    per_rank = GLOBAL_BATCH // mesh.size("dp")
    ids = batch.narrow(0, mesh.coordinate("dp") * per_rank, per_rank)
    logits = model(ids)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, CFG["vocab_size"]), ids[:, 1:].reshape(-1)
    )
    loss.backward()
    parallel.sync()
    optimizer.step()
    optimizer.zero_grad()


def full(tensor: torch.Tensor, sharded: bool, mesh: Mesh) -> torch.Tensor:
    """a collective when sharded: every rank must call it."""
    if not sharded:
        return tensor.clone()
    return C.all_gather(tensor.contiguous(), mesh, "dp_shard", gather_dim=0)


def full_state(model, parallel, optimizer, mesh: Mesh) -> dict:
    """model params and adam moments as plain unsharded tensors."""
    out = {}
    for name, p in model.named_parameters():
        sharded = parallel.shard_dim_of(p) is not None
        out[f"param/{name}"] = full(p.data, sharded, mesh)
        for key in ("exp_avg", "exp_avg_sq"):
            out[f"{key}/{name}"] = full(optimizer.state[p][key], sharded, mesh)
        out[f"step/{name}"] = optimizer.state[p]["step"].clone()
    return out


def local_expected(ref: torch.Tensor, sharded: bool, mesh: Mesh) -> torch.Tensor:
    if not sharded:
        return ref
    return Shard(0).local_slice(ref, mesh.size("dp_shard"), mesh.coordinate("dp_shard"))


def writes_model_state(parallel, mesh: Mesh) -> bool:
    # same rule as Trainer._writes_model_state
    return all(
        mesh.coordinate(axis) == 0
        for axis in ("dp_replicate", "dp_shard")
        if axis not in parallel.shard_dims
    )


def save_phase(mesh: Mesh, tmp: str) -> None:
    rank = dist.get_rank()
    model, parallel, optimizer = build(seed=rank, mesh=mesh)
    ckpt = Checkpointer(os.path.join(tmp, "run"))

    for step in range(CKPT_STEP):
        train_step(model, parallel, optimizer, mesh, step)

    if writes_model_state(parallel, mesh):
        layout = ShardLayout.of(model, parallel, mesh)
        ckpt.save_model(model, CKPT_STEP, layout)
        ckpt.save_optimizer(optimizer, CKPT_STEP, layout)
    at_ckpt = full_state(model, parallel, optimizer, mesh)

    for step in range(CKPT_STEP, END_STEP):
        train_step(model, parallel, optimizer, mesh, step)
    at_end = full_state(model, parallel, optimizer, mesh)

    C.barrier()
    if rank == 0:
        torch.save(at_ckpt, os.path.join(tmp, "ref_ckpt.pt"))
        torch.save(at_end, os.path.join(tmp, "ref_end.pt"))
        # one file per shard, not per rank: hsdp replicas must not all write
        n = mesh.size("dp_shard")
        files = sorted(os.listdir(os.path.join(tmp, "run", str(CKPT_STEP))))
        stems = ("model", "optimizer")
        if n == 1:
            expected = [f"{s}.pt" for s in stems]
        else:
            expected = [f"{s}_shard{k}.pt" for s in stems for k in range(n)]
            expected += [f"{s}_shards.json" for s in stems]
        assert files == sorted(expected), f"checkpoint dir has {files}"


def resume_phase(mesh: Mesh, tmp: str) -> None:
    rank = dist.get_rank()
    # a different seed than the save run, so a load that silently does
    # nothing can't pass
    model, parallel, optimizer = build(seed=100 + rank, mesh=mesh)
    ckpt = Checkpointer(os.path.join(tmp, "run"))
    layout = ShardLayout.of(model, parallel, mesh)
    ckpt.load_model(model, CKPT_STEP, layout)
    ckpt.load_optimizer(optimizer, CKPT_STEP, layout)

    errors = []

    def check(label, got, expected, exact):
        if tuple(got.shape) != tuple(expected.shape):
            errors.append(f"{label}: shape {tuple(got.shape)} != {tuple(expected.shape)}")
        elif exact and not torch.equal(got, expected):
            errors.append(f"{label}: not bitwise equal after load")
        elif not exact and not torch.allclose(got, expected, rtol=1e-3, atol=1e-5):
            errors.append(f"{label}: max abs diff {(got - expected).abs().max():.3e}")

    ref = torch.load(os.path.join(tmp, "ref_ckpt.pt"))
    for name, p in model.named_parameters():
        sharded = parallel.shard_dim_of(p) is not None
        check(f"load {name}", p.data, local_expected(ref[f"param/{name}"], sharded, mesh), True)
        for key in ("exp_avg", "exp_avg_sq"):
            check(f"load {key} {name}", optimizer.state[p][key],
                  local_expected(ref[f"{key}/{name}"], sharded, mesh), True)
        check(f"load step {name}", optimizer.state[p]["step"], ref[f"step/{name}"], True)

    # the tie must survive a load that fed both keys separately
    assert model.lm_head.weight is model.token_emb.weight

    for step in range(CKPT_STEP, END_STEP):
        train_step(model, parallel, optimizer, mesh, step)

    ref = torch.load(os.path.join(tmp, "ref_end.pt"))
    for name, p in model.named_parameters():
        sharded = parallel.shard_dim_of(p) is not None
        check(f"end {name}", p.data, local_expected(ref[f"param/{name}"], sharded, mesh), False)

    if errors:
        raise AssertionError(f"{len(errors)} mismatch(es): " + " | ".join(errors[:4]))


def _worker(rank: int, world_size: int, phase: str, spec: tuple, tmp: str, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    mesh = Mesh({"dp_replicate": spec[0], "dp_shard": spec[1]})
    mesh.flatten(("dp_replicate", "dp_shard"), "dp")
    try:
        (save_phase if phase == "save" else resume_phase)(mesh, tmp)
        msg = "PASS"
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
    # every rank's verdict: rank 0 owns the first slice of everything and is
    # the rank a resharding bug is most likely to leave correct
    verdicts = [None] * world_size
    dist.all_gather_object(verdicts, msg)
    dist.destroy_process_group()
    bad = {r: v for r, v in enumerate(verdicts) if v != "PASS"}
    if bad:
        if rank == 0:
            print(f"  {phase}: " + " || ".join(f"rank{r}: {v}" for r, v in bad.items()), flush=True)
        raise SystemExit(1)


def main() -> None:
    port = 29600
    failed = []
    for name, save_spec, resume_spec in SCENARIOS:
        tmp = tempfile.mkdtemp(prefix="torchure_shard_ckpt_")
        try:
            for phase, spec in (("save", save_spec), ("resume", resume_spec)):
                port += 1
                world = spec[0] * spec[1]
                mp.spawn(_worker, args=(world, phase, spec, tmp, port), nprocs=world)
            print(f"[{name}] PASS", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[{name}] FAIL", flush=True)
            failed.append(name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
