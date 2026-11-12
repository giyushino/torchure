"""
equivalence tests for torchure/parallelism/tensor_parallel.py.

run (gloo/cpu, no gpus needed):
    uv run tests/tensor_parallel.py

the invariant: tensor parallel changes where the math runs, never the math. a
tp run must match one process holding the full model -- same logits, and each
rank's grads and trained weights must be its slice of the full ones. every
rank rebuilds the single-process reference locally (same seed, same
deterministic batch), so nothing is gathered for the comparison.

a scenario is a mesh (dp x tp), what runs on the dp axis (ddp, or fsdp
sharding what tp left on each rank), and the two tp options: vocab parallel
(embedding / lm_head weight sharded by vocab rows) and sequence parallel
(activations between regions sharded along the sequence).

checks, per scenario:
    logits        forward matches, with grad and under no_grad
    grads         one backward: sharded, shared (q_norm/k_norm, and the block
                  norms under sequence parallel) and replicated
    act ckpt      the same with activation checkpointing on, where the
                  regions' collectives run again inside backward
    training      3 adamw steps with 2 accumulated microbatches each and grad
                  clipping; the clipped global norm must match too. with
                  vocab parallel the loss is the fused vocab-parallel one,
                  which never builds full logits
    checkpoint    save sharded (one file per tp x dp_shard cell), load back
                  sharded (bitwise) and unsharded into a full model
    meta init     build from the meta device, so no rank holds full weights:
                  replicas agree, shards differ, and the model the ranks hold
                  between them behaves like one full model
"""

import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torchure.checkpoint.checkpointer import UNSHARDED, Checkpointer, ShardLayout
from torchure.core import collective as C
from torchure.core.grad_helper import clip_grad_norm
from torchure.core.mesh import Mesh
from torchure.core.meta import materialize
from torchure.core.placement import Shard, padded_chunk
from torchure.loss.fused_linear_ce import fused_linear_cross_entropy
from torchure.models.qwen3.qwen3 import Qwen3
from torchure.parallelism.data_parallel import DDP
from torchure.parallelism.fsdp2 import FSDP
from torchure.parallelism.tensor_parallel import TP

CFG = {
    "num_layers": 2,
    "num_heads": 8,
    "num_kv_heads": 4,
    "emb_dim": 32,
    "head_dim": 8,
    "ffn_dim": 64,
    "vocab_size": 64,
}
SEQ = 16
MICRO = 2          # sequences per rank per microbatch
ACCUM = 2
STEPS = 3
MAX_NORM = 0.5     # small enough that clipping is active
IGNORE = -100


@dataclass(frozen=True)
class Scenario:
    dp: int
    tp: int
    fsdp: bool = False
    vocab: bool = False
    sp: bool = False
    batch_sp: bool = False     # sp shards the batch dim instead of the sequence
    overlap: str = "none"

    @property
    def name(self) -> str:
        opts = [o for o, on in (
            ("vocab", self.vocab), ("sp", self.sp), ("batch-sp", self.batch_sp),
            (f"overlap={self.overlap}", self.overlap != "none"),
        ) if on]
        return (
            f"{'fsdp' if self.fsdp else 'ddp'} dp={self.dp} x tp={self.tp}"
            + (f" + {' + '.join(opts)}" if opts else "")
        )


SCENARIOS = [
    Scenario(1, 2),
    Scenario(1, 4, vocab=True),
    Scenario(1, 2, sp=True),
    Scenario(2, 2, vocab=True, sp=True),
    Scenario(2, 2, fsdp=True),
    Scenario(2, 2, fsdp=True, vocab=True, sp=True),
    # 3 shards divide none of the dims: fsdp pads what tp left
    Scenario(3, 2, fsdp=True, vocab=True, sp=True),
    Scenario(1, 2, sp=True, batch_sp=True),
    # fused overlap ops: sequence-sharded (a layout copy each way) and
    # batch-sharded rows, ring lengths 2 and 4
    Scenario(1, 2, sp=True, overlap="backward"),
    Scenario(2, 2, vocab=True, sp=True, overlap="collective_matmul"),
    Scenario(1, 4, vocab=True, sp=True, overlap="collective_matmul"),
    Scenario(1, 2, sp=True, batch_sp=True, overlap="collective_matmul"),
    Scenario(2, 2, fsdp=True, vocab=True, sp=True, batch_sp=True, overlap="backward"),
    Scenario(3, 2, fsdp=True, vocab=True, sp=True, overlap="collective_matmul"),
]


def build_full(seed: int = 0) -> Qwen3:
    torch.manual_seed(seed)
    model = Qwen3(**CFG)
    model.init_weights()
    # init leaves norm weights at exactly 1; move them so their grads matter
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 1:
                p.add_(torch.randn_like(p) * 0.1)
    return model


def parallelize(model: Qwen3, mesh: Mesh, sc: Scenario, tp_only: bool = False):
    tp = TP(
        mesh,
        **model.tp_plan(sc.tp, vocab_parallel=sc.vocab),
        sequence_parallel=sc.sp,
        seq_dim=0 if sc.batch_sp else 1,
        overlap=sc.overlap,
    )
    if tp_only:
        return tp
    if sc.fsdp:
        par = FSDP(model, mesh, model.blocks, replicate_dim=None)
    else:
        par = DDP(model, mesh, dim="dp")
    return tp, par


def build_tp(mesh: Mesh, sc: Scenario, seed: int = 0):
    """full weights on every rank, then keep this rank's slices."""
    model = build_full(seed)
    tp, par = parallelize(model, mesh, sc)
    return model, tp, par


def build_tp_from_meta(mesh: Mesh, sc: Scenario):
    """the trainer's order: no rank ever holds the full weights."""
    with torch.device("meta"):
        model = Qwen3(**CFG)
    tp = parallelize(model, mesh, sc, tp_only=True)
    cpu = torch.device("cpu")
    if sc.fsdp:
        par = FSDP(model, mesh, model.blocks, replicate_dim=None, device=cpu)
    else:
        materialize(model, cpu)
    torch.manual_seed(100 + dist.get_rank())
    model.init_weights()
    if sc.fsdp:
        par.finish_init()
    tp.finish_init(model)
    if not sc.fsdp:
        par = DDP(model, mesh, dim="dp")
    return model, tp, par


def batch(step: int, micro: int, dp: int) -> torch.Tensor:
    """the global batch of one microbatch: (dp * MICRO, SEQ)."""
    g = torch.Generator().manual_seed(1000 + 10 * step + micro)
    return torch.randint(0, CFG["vocab_size"], (dp * MICRO, SEQ), generator=g)


def my_rows(ids: torch.Tensor, mesh: Mesh) -> torch.Tensor:
    return ids.narrow(0, mesh.coordinate("dp") * MICRO, MICRO)


def ce(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, CFG["vocab_size"]), ids[:, 1:].reshape(-1)
    )


def fused_loss(model: Qwen3, tp: TP, ids: torch.Tensor) -> torch.Tensor:
    """what ARObjective does under tp: no full logits anywhere."""
    hidden = tp.head_input(model(ids, return_final_hidden=True))
    labels = torch.full_like(ids, IGNORE)
    labels[:, :-1] = ids[:, 1:]
    # a chunk size that doesn't divide the rows, so the last chunk is short
    return fused_linear_cross_entropy(
        hidden, model.lm_head.weight, labels, IGNORE, 7, vocab_axis=tp.vocab_axis
    )


def local(ref: torch.Tensor, tp: TP, par, param, mesh: Mesh) -> torch.Tensor:
    """this rank's piece of a full tensor laid out like `param`."""
    dim = tp.shard_dim_of(param)
    if dim is not None:
        ref = Shard(dim).local_slice(ref, mesh.size("tp"), mesh.coordinate("tp"))
    if par.shard_dim_of(param) is not None:
        length, start, valid = padded_chunk(
            ref.shape[0], mesh.size("dp_shard"), mesh.coordinate("dp_shard")
        )
        out = ref.new_zeros((length, *ref.shape[1:]))
        out[:valid] = ref[start:start + valid]
        ref = out
    return ref


def gathered(tensor: torch.Tensor, tp: TP, par, param, mesh: Mesh) -> torch.Tensor:
    """collective: the full tensor from every rank's piece. inverse of local()."""
    if par.shard_dim_of(param) is not None:
        rows = par.full_shape_of(param)[0]
        tensor = C.all_gather(tensor.contiguous(), mesh, "dp_shard", gather_dim=0)[:rows]
    dim = tp.shard_dim_of(param)
    if dim is not None:
        tensor = C.all_gather(tensor.contiguous(), mesh, "tp", gather_dim=dim)
    return tensor.clone()


def sync_and_clip(model, tp: TP, par, mesh: Mesh, max_norm):
    par.sync()
    tp.sync()
    return clip_grad_norm(
        model.parameters(), mesh, max_norm,
        shard_dims_of=lambda p: par.shard_dims + tp.shard_dims_of(p),
    )


class Checks:
    def __init__(self):
        self.errors: list[str] = []

    def close(self, label, got, expected, rtol=1e-4, atol=1e-6):
        if tuple(got.shape) != tuple(expected.shape):
            self.errors.append(f"{label}: shape {tuple(got.shape)} != {tuple(expected.shape)}")
        elif not torch.allclose(got, expected, rtol=rtol, atol=atol):
            self.errors.append(f"{label}: max abs diff {(got - expected).abs().max():.3e}")

    def equal(self, label, got, expected):
        if tuple(got.shape) != tuple(expected.shape) or not torch.equal(got, expected):
            self.errors.append(f"{label}: not bitwise equal")

    def true(self, label, cond):
        if not cond:
            self.errors.append(label)


def compare_one_step(ref: Qwen3, model: Qwen3, tp: TP, par, mesh: Mesh, checks: Checks, tag: str):
    """logits and grads of one backward, tp model vs the full reference."""
    ids = batch(0, 0, mesh.size("dp"))
    ref.zero_grad()
    ref_logits = ref(ids)
    ce(ref_logits, ids).backward()

    mine = my_rows(ids, mesh)
    # one forward per backward: fsdp records unit order on the first one
    logits = model(mine)
    checks.close(f"{tag}logits", logits, my_rows(ref_logits, mesh), atol=1e-5)
    ce(logits, mine).backward()
    sync_and_clip(model, tp, par, mesh, None)
    for (name, p), (_, p_ref) in zip(
        model.named_parameters(), ref.named_parameters(), strict=True
    ):
        checks.close(f"{tag}grad {name}", p.grad, local(p_ref.grad, tp, par, p, mesh))
    model.zero_grad()

    with torch.no_grad():
        checks.close(
            f"{tag}no_grad logits", model(mine), my_rows(ref_logits.detach(), mesh), atol=1e-5
        )


def check_forward_and_grads(mesh: Mesh, sc: Scenario, checks: Checks, act_ckpt: bool) -> None:
    model, tp, par = build_tp(mesh, sc)
    model.activation_checkpointing = act_ckpt

    # the plan really did shard something, and by the right amount
    q = model.blocks[0].gqa.q_proj.weight
    o = model.blocks[0].gqa.o_proj.weight
    heads_width = CFG["num_heads"] * CFG["head_dim"]
    assert tp.full_shape_of(q) == (heads_width, CFG["emb_dim"]), tp.full_shape_of(q)
    assert tp.shard_dim_of(q) == 0 and tp.shard_dim_of(o) == 1
    assert o.shape[1] == heads_width // sc.tp, o.shape
    assert model.lm_head.weight is model.token_emb.weight
    assert (tp.shard_dim_of(model.lm_head.weight) == 0) == sc.vocab

    compare_one_step(
        build_full(), model, tp, par, mesh, checks, "act-ckpt " if act_ckpt else ""
    )


def train(model, optimizer, dp: int, mesh: Mesh | None, sc=None, tp=None, par=None) -> list[float]:
    """STEPS optimizer steps of ACCUM microbatches; returns the pre-clip grad norms."""
    norms = []
    for step in range(STEPS):
        for micro in range(ACCUM):
            ids = batch(step, micro, dp)
            if mesh is None:
                loss = ce(model(ids), ids)
            else:
                ids = my_rows(ids, mesh)
                par.requires_sync = micro == ACCUM - 1
                loss = fused_loss(model, tp, ids) if sc.vocab else ce(model(ids), ids)
            (loss / ACCUM).backward()
        if mesh is None:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
        else:
            norm = sync_and_clip(model, tp, par, mesh, MAX_NORM)
        norms.append(float(norm))
        optimizer.step()
        optimizer.zero_grad()
    return norms


def check_training_and_checkpoint(mesh: Mesh, sc: Scenario, checks: Checks, tmp: str) -> None:
    dp = mesh.size("dp")
    ref = build_full()
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-2, weight_decay=0.1)
    ref_norms = train(ref, ref_opt, dp, None)

    model, tp, par = build_tp(mesh, sc)
    # built after sharding, so adam's moments are allocated sharded
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    norms = train(model, optimizer, dp, mesh, sc, tp, par)

    assert max(ref_norms) > MAX_NORM, "clipping never engaged; the test would prove less"
    checks.close("grad norms", torch.tensor(norms), torch.tensor(ref_norms))
    for (name, p), (_, p_ref) in zip(
        model.named_parameters(), ref.named_parameters(), strict=True
    ):
        # loose: adam divides by sqrt(v), so fp noise in a near-zero grad
        # moves a weight by a visible fraction of lr. a wrong grad moves it by
        # ~lr per step (1e-2), and the grads themselves are checked tightly above
        checks.close(
            f"trained {name}", p.data, local(p_ref.data, tp, par, p, mesh), rtol=1e-3, atol=1e-3
        )

    # ---- checkpoint: one file per shard cell, written by one rank each
    rank = dist.get_rank()
    ckpt = Checkpointer(os.path.join(tmp, "run"))
    layout = ShardLayout.of(model, par, mesh, tp=tp)
    assert layout.num_shards == sc.tp * (sc.dp if sc.fsdp else 1), layout.counts
    sharded_axes = par.shard_dims + tp.shard_dims
    if all(mesh.coordinate(a) == 0 for a in ("dp_shard", "tp") if a not in sharded_axes):
        ckpt.save_model(model, 1, layout)
        ckpt.save_optimizer(optimizer, 1, layout)
    dist.barrier()
    if rank == 0:
        files = sorted(os.listdir(os.path.join(tmp, "run", "1")))
        expected = [
            f"{s}_shard{k}.pt" for s in ("model", "optimizer") for k in range(layout.num_shards)
        ]
        expected += ["model_shards.json", "optimizer_shards.json"]
        assert files == sorted(expected), f"checkpoint dir has {files}"

    # sharded load onto a differently-initialized model: bitwise
    fresh, fresh_tp, fresh_par = build_tp(mesh, sc, seed=7)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-2, weight_decay=0.1)
    fresh_layout = ShardLayout.of(fresh, fresh_par, mesh, tp=fresh_tp)
    ckpt.load_model(fresh, 1, fresh_layout)
    ckpt.load_optimizer(fresh_opt, 1, fresh_layout)
    for (name, p), (_, p_new) in zip(
        model.named_parameters(), fresh.named_parameters(), strict=True
    ):
        checks.equal(f"reload {name}", p_new.data, p.data)
        for key in ("exp_avg", "exp_avg_sq"):
            checks.equal(f"reload {key} {name}", fresh_opt.state[p_new][key], optimizer.state[p][key])

    # unsharded load into a full model: every rank reassembles the same thing
    whole = build_full(seed=7)
    whole_opt = torch.optim.AdamW(whole.parameters(), lr=1e-2, weight_decay=0.1)
    ckpt.load_model(whole, 1, UNSHARDED)
    ckpt.load_optimizer(whole_opt, 1, UNSHARDED)
    for (name, p), (_, p_ref) in zip(
        whole.named_parameters(), ref.named_parameters(), strict=True
    ):
        checks.close(f"unsharded {name}", p.data, p_ref.data, rtol=1e-3, atol=1e-3)
    for (name, p), (_, p_whole) in zip(
        model.named_parameters(), whole.named_parameters(), strict=True
    ):
        checks.equal(f"unsharded slice {name}", local(p_whole.data, tp, par, p, mesh), p.data)
        checks.equal(
            f"unsharded exp_avg slice {name}",
            local(whole_opt.state[p_whole]["exp_avg"], tp, par, p, mesh),
            optimizer.state[p]["exp_avg"],
        )


def check_meta_init(mesh: Mesh, sc: Scenario, checks: Checks) -> None:
    model, tp, par = build_tp_from_meta(mesh, sc)
    assert not any(p.is_meta for p in model.parameters())
    assert not any(b.is_meta for b in model.buffers())
    assert model.lm_head.weight is model.token_emb.weight

    # the full model the ranks hold between them
    ref = Qwen3(**CFG)
    for (name, p), (_, p_ref) in zip(
        model.named_parameters(), ref.named_parameters(), strict=True
    ):
        full = gathered(p.data, tp, par, p, mesh)
        checks.true(
            f"meta {name}: assembled shape {tuple(full.shape)} != {tuple(p_ref.shape)}",
            tuple(full.shape) == tuple(p_ref.shape),
        )
        if tuple(full.shape) != tuple(p_ref.shape):
            continue
        # every rank must have assembled the same full tensor
        for axis in ("dp", "tp"):
            agreed = full.clone()
            C.broadcast(agreed, mesh, axis, src=0)
            checks.equal(f"meta {name}: differs across {axis}", full, agreed)
        dim = tp.shard_dim_of(p)
        if dim is not None and p.dim() == 2:
            first, second = full.chunk(sc.tp, dim=dim)[:2]
            checks.true(f"meta {name}: tp shards are copies of each other", not torch.equal(first, second))
        with torch.no_grad():
            p_ref.copy_(full)
    if checks.errors:
        return
    compare_one_step(ref, model, tp, par, mesh, checks, "meta ")


def _worker(rank: int, world_size: int, sc: Scenario, tmp: str, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    mesh = Mesh({"dp_replicate": 1, "dp_shard": sc.dp, "tp": sc.tp})
    mesh.flatten(("dp_replicate", "dp_shard"), "dp")
    checks = Checks()
    try:
        check_forward_and_grads(mesh, sc, checks, act_ckpt=False)
        check_forward_and_grads(mesh, sc, checks, act_ckpt=True)
        check_training_and_checkpoint(mesh, sc, checks, tmp)
        check_meta_init(mesh, sc, checks)
        msg = "PASS"
        if checks.errors:
            msg = f"{len(checks.errors)} mismatch(es): " + " | ".join(checks.errors[:4])
    except Exception as e:  # noqa: BLE001
        import traceback
        msg = f"{type(e).__name__}: {e} @ {traceback.format_exc().splitlines()[-3].strip()}"
    # every rank's verdict: rank 0 holds the first slice of everything and is
    # the rank a slicing bug is most likely to leave correct
    verdicts = [None] * world_size
    dist.all_gather_object(verdicts, msg)
    dist.destroy_process_group()
    bad = {r: v for r, v in enumerate(verdicts) if v != "PASS"}
    if bad:
        if rank == 0:
            print("  " + " || ".join(f"rank{r}: {v}" for r, v in bad.items()), flush=True)
        raise SystemExit(1)


def main() -> None:
    port = 29900 + os.getpid() % 200
    failed = []
    for sc in SCENARIOS:
        tmp = tempfile.mkdtemp(prefix="torchure_tp_")
        port += 1
        world = sc.dp * sc.tp
        try:
            mp.spawn(_worker, args=(world, sc, tmp, port), nprocs=world)
            print(f"[{sc.name}] PASS", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[{sc.name}] FAIL", flush=True)
            failed.append(sc.name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
