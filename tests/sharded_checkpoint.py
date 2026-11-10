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
also the legacy `model.pt` on-disk format, in both directions). 3 shards
divides none of the model's dims, so those scenarios cover padded shards.
tensor parallel adds a second shard axis: tp cuts some params along dim 0 or
1, and fsdp shards whatever is left on the rank along dim 0.
"""

import math
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
from torchure.core.placement import Shard, padded_chunk
from torchure.models.qwen3.qwen3 import Qwen3
from torchure.parallelism.data_parallel import DDP
from torchure.parallelism.fsdp2 import FSDP
from torchure.parallelism.tensor_parallel import TP

CFG = {
    "num_layers": 2,
    "num_heads": 4,
    "num_kv_heads": 2,
    "emb_dim": 32,
    "head_dim": 8,
    "ffn_dim": 64,
    "vocab_size": 64,
}
SEQ = 16
GLOBAL_BATCH = 12
CKPT_STEP = 3
END_STEP = 6

# (name, save mesh, resume mesh) as (dp_replicate, dp_shard[, tp])
SCENARIOS = [
    ("fsdp4 -> fsdp2", (1, 4), (1, 2)),
    ("fsdp2 -> fsdp4", (1, 2), (1, 4)),
    ("fsdp4 -> unsharded", (1, 4), (1, 1)),
    ("unsharded -> fsdp4", (1, 1), (1, 4)),
    ("hsdp2x2 -> fsdp4", (2, 2), (1, 4)),
    ("fsdp4 -> hsdp2x2", (1, 4), (2, 2)),
    ("fsdp4 -> fsdp4", (1, 4), (1, 4)),
    ("fsdp3 (padded) -> fsdp4", (1, 3), (1, 4)),
    ("fsdp4 -> fsdp3 (padded)", (1, 4), (1, 3)),
    ("fsdp3 (padded) -> unsharded", (1, 3), (1, 1)),
    ("unsharded -> fsdp3 (padded)", (1, 1), (1, 3)),
    ("tp2 -> unsharded", (1, 1, 2), (1, 1)),
    ("unsharded -> tp2", (1, 1), (1, 1, 2)),
    ("tp2 -> fsdp4", (1, 1, 2), (1, 4)),
    ("fsdp2 x tp2 -> fsdp4", (1, 2, 2), (1, 4)),
    ("fsdp4 -> fsdp2 x tp2", (1, 4), (1, 2, 2)),
    ("fsdp2 x tp2 -> tp2 x ddp2", (1, 2, 2), (2, 1, 2)),
    ("fsdp3 x tp2 (padded) -> fsdp2 x tp2", (1, 3, 2), (1, 2, 2)),
    ("fsdp2 x tp2 -> fsdp3 (padded)", (1, 2, 2), (1, 3)),
]


class NoTP:
    """stand-in so the helpers below never branch on whether tp is on."""

    shard_dims: tuple = ()

    def shard_dim_of(self, param):
        return None

    def sync(self) -> None:
        pass


def build(seed: int, mesh: Mesh):
    torch.manual_seed(seed)
    model = Qwen3(**CFG)
    model.init_weights()
    tp = NoTP()
    if "tp" in mesh.sizes:
        # the full weights must agree before tp keeps a slice of them
        for p in model.parameters():
            C.broadcast(p.data, mesh, "tp")
            C.broadcast(p.data, mesh, "dp")
        tp = TP(mesh, **model.tp_plan(mesh.size("tp")), sequence_parallel=True)
    if mesh.size("dp_shard") > 1:
        parallel = FSDP(
            model, mesh, model.blocks,
            replicate_dim="dp_replicate" if mesh.size("dp_replicate") > 1 else None,
        )
    else:
        parallel = DDP(model, mesh, dim="dp")
    # built after sharding, so adam's moments are allocated sharded
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    return model, parallel, tp, optimizer


def train_step(model, parallel, tp, optimizer, mesh: Mesh, step: int) -> None:
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
    tp.sync()
    optimizer.step()
    optimizer.zero_grad()


def full(tensor: torch.Tensor, param, parallel, tp, mesh: Mesh) -> torch.Tensor:
    """
    the unsharded tensor behind this rank's piece of something laid out like
    `param`. a collective when sharded: every rank must call it. trims shard
    padding.
    """
    if parallel.shard_dim_of(param) is not None:
        rows = parallel.full_shape_of(param)[0]
        tensor = C.all_gather(tensor.contiguous(), mesh, "dp_shard", gather_dim=0)[:rows]
    dim = tp.shard_dim_of(param)
    if dim is not None:
        tensor = C.all_gather(tensor.contiguous(), mesh, "tp", gather_dim=dim)
    return tensor.clone()


def full_state(model, parallel, tp, optimizer, mesh: Mesh) -> dict:
    """model params and adam moments as plain unsharded tensors."""
    out = {}
    for name, p in model.named_parameters():
        out[f"param/{name}"] = full(p.data, p, parallel, tp, mesh)
        for key in ("exp_avg", "exp_avg_sq"):
            out[f"{key}/{name}"] = full(optimizer.state[p][key], p, parallel, tp, mesh)
        out[f"step/{name}"] = optimizer.state[p]["step"].clone()
    return out


def local_expected(ref: torch.Tensor, param, parallel, tp, mesh: Mesh) -> torch.Tensor:
    """this rank's piece of a full tensor: tp's slice, then fsdp's zero-padded rows."""
    dim = tp.shard_dim_of(param)
    if dim is not None:
        ref = Shard(dim).local_slice(ref, mesh.size("tp"), mesh.coordinate("tp"))
    if parallel.shard_dim_of(param) is None:
        return ref
    length, start, valid = padded_chunk(
        ref.shape[0], mesh.size("dp_shard"), mesh.coordinate("dp_shard")
    )
    out = ref.new_zeros((length, *ref.shape[1:]))
    out[:valid] = ref[start:start + valid]
    return out


def writes_model_state(parallel, tp, mesh: Mesh) -> bool:
    # same rule as Trainer._writes_model_state
    sharded = parallel.shard_dims + tp.shard_dims
    return all(
        mesh.coordinate(axis) == 0
        for axis in ("dp_replicate", "dp_shard", "tp")
        if axis in mesh.sizes and axis not in sharded
    )


def layout_of(model, parallel, tp, mesh: Mesh) -> ShardLayout:
    return ShardLayout.of(model, parallel, mesh, tp=tp if tp.shard_dims else None)


def save_phase(mesh: Mesh, tmp: str) -> None:
    rank = dist.get_rank()
    model, parallel, tp, optimizer = build(seed=rank, mesh=mesh)
    ckpt = Checkpointer(os.path.join(tmp, "run"))

    for step in range(CKPT_STEP):
        train_step(model, parallel, tp, optimizer, mesh, step)

    layout = layout_of(model, parallel, tp, mesh)
    if writes_model_state(parallel, tp, mesh):
        ckpt.save_model(model, CKPT_STEP, layout)
        ckpt.save_optimizer(optimizer, CKPT_STEP, layout)
    at_ckpt = full_state(model, parallel, tp, optimizer, mesh)

    for step in range(CKPT_STEP, END_STEP):
        train_step(model, parallel, tp, optimizer, mesh, step)
    at_end = full_state(model, parallel, tp, optimizer, mesh)

    C.barrier()
    if rank == 0:
        torch.save(at_ckpt, os.path.join(tmp, "ref_ckpt.pt"))
        torch.save(at_end, os.path.join(tmp, "ref_end.pt"))
        # one file per shard cell, not per rank: replicas must not all write
        n = layout.num_shards
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
    model, parallel, tp, optimizer = build(seed=100 + rank, mesh=mesh)
    ckpt = Checkpointer(os.path.join(tmp, "run"))
    layout = layout_of(model, parallel, tp, mesh)
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

    def expected(ref, p):
        return local_expected(ref, p, parallel, tp, mesh)

    ref = torch.load(os.path.join(tmp, "ref_ckpt.pt"))
    for name, p in model.named_parameters():
        check(f"load {name}", p.data, expected(ref[f"param/{name}"], p), True)
        for key in ("exp_avg", "exp_avg_sq"):
            check(f"load {key} {name}", optimizer.state[p][key],
                  expected(ref[f"{key}/{name}"], p), True)
        check(f"load step {name}", optimizer.state[p]["step"], ref[f"step/{name}"], True)

    # the tie must survive a load that fed both keys separately
    assert model.lm_head.weight is model.token_emb.weight

    for step in range(CKPT_STEP, END_STEP):
        train_step(model, parallel, tp, optimizer, mesh, step)

    ref = torch.load(os.path.join(tmp, "ref_end.pt"))
    for name, p in model.named_parameters():
        check(f"end {name}", p.data, expected(ref[f"param/{name}"], p), False)

    if errors:
        raise AssertionError(f"{len(errors)} mismatch(es): " + " | ".join(errors[:4]))


def _worker(rank: int, world_size: int, phase: str, spec: tuple, tmp: str, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    axes = {"dp_replicate": spec[0], "dp_shard": spec[1]}
    if len(spec) > 2:
        axes["tp"] = spec[2]
    mesh = Mesh(axes)
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
    port = 29700 + os.getpid() % 200
    failed = []
    for name, save_spec, resume_spec in SCENARIOS:
        tmp = tempfile.mkdtemp(prefix="torchure_shard_ckpt_")
        try:
            for phase, spec in (("save", save_spec), ("resume", resume_spec)):
                port += 1
                world = math.prod(spec)
                mp.spawn(_worker, args=(world, phase, spec, tmp, port), nprocs=world)
            print(f"[{name}] PASS", flush=True)
        except Exception as e:  # noqa: BLE001
            # a worker that died outside its own try (mesh setup) reports here
            print(f"[{name}] FAIL  {str(e).strip().splitlines()[-1][:300]}", flush=True)
            failed.append(name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
