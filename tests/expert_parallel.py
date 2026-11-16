"""
tests for torchure/parallelism/expert_parallel.py.

run (gloo/cpu, no gpus needed):
    uv run tests/expert_parallel.py

every ep rank routes its OWN batch, so the reference is one process holding
the full model: it runs each rank's batch and averages the losses. after the
sync every rank must hold its slice of that reference's grads -- the ep
slice of an expert weight, all of a dense one, and under fsdp the dp_shard
chunk of either. that one comparison covers the dispatch (forward and the
grads retracing it), the 1/ep scale on expert grads, and which axis each
kind of param is averaged over.

layouts: ep alone, ep x dp under ddp, ep x dp_shard under fsdp. then the
meta build (no rank ever holds every expert) and a sharded checkpoint
loading back into one unsharded model.
"""

import os
import shutil
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from torchure.checkpoint.checkpointer import Checkpointer, ShardLayout
from torchure.core.collective import all_gather, all_reduce
from torchure.core.mesh import build_mesh
from torchure.core.meta import materialize
from torchure.core.placement import Shard, padded_chunk
from torchure.models.moe import ExpertBank, sort_by_expert
from torchure.parallelism.data_parallel import DDP
from torchure.parallelism.expert_parallel import EP
from torchure.parallelism.fsdp2 import FSDP

DIM, HIDDEN, EXPERTS, TOP_K, LAYERS, TOKENS = 8, 12, 8, 2, 2, 24


def _act(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.tanh(gate) * up


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(DIM, DIM)
        self.router = nn.Linear(DIM, EXPERTS, bias=False)
        self.experts = ExpertBank(EXPERTS, DIM, HIDDEN, DIM, _act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + torch.tanh(self.lin(x))
        scores, indices = self.router(h).topk(TOP_K, dim=-1)
        weights = scores.softmax(dim=-1)
        token, slot, counts = sort_by_expert(indices, EXPERTS)
        routed = self.experts(h[token], counts) * weights[token, slot].unsqueeze(-1)
        return h + torch.zeros_like(h).index_add(0, token, routed)


class TinyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(DIM, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.head = nn.Linear(DIM, 1)

    def init_weights(self) -> None:
        for p in self.parameters():
            nn.init.normal_(p, std=0.3)

    def ep_plan(self) -> list[ExpertBank]:
        return [block.experts for block in self.blocks]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inp(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).pow(2).mean()


def build_model(seed: int = 0) -> TinyMoE:
    torch.manual_seed(seed)
    model = TinyMoE()
    model.init_weights()
    return model


def batch_of(data_rank: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + data_rank)
    return torch.randn(TOKENS, DIM, generator=g)


def parallelize(model: TinyMoE, mesh, strategy: str, device=None):
    """EP, then the data-parallel strategy, wired the way the trainer does it."""
    ep = EP(mesh, model.ep_plan())
    if strategy == "ddp":
        parallel = DDP(
            model, mesh, dim="dp",
            dim_of=lambda p: "dp_expert" if ep.owns(p) else "dp",
        )
    else:
        parallel = FSDP(
            model, mesh, model.blocks, shard_dim="dp_shard",
            replicate_dim="dp_replicate_ep",
            replicate_dim_of=lambda p: "dp_replicate" if ep.owns(p) else "dp_replicate_ep",
            device=device,
        )
    return ep, parallel


def local_piece(full: torch.Tensor, expert: bool, mesh, strategy: str) -> torch.Tensor:
    """what this rank holds of a full tensor: ep slice, then fsdp's padded chunk."""
    if expert:
        full = Shard(0).local_slice(full, mesh.size("ep"), mesh.coordinate("ep"))
    if strategy == "fsdp":
        length, start, valid = padded_chunk(
            full.shape[0], mesh.size("dp_shard"), mesh.coordinate("dp_shard")
        )
        piece = full.new_zeros((length, *full.shape[1:]))
        piece[:valid] = full[start:start + valid]
        return piece
    return full


def test_grad_parity(mesh, strategy: str):
    world = dist.get_world_size()
    reference = build_model()
    losses = [reference(batch_of(r)) for r in range(world)]
    (sum(losses) / world).backward()

    model = build_model()
    ep, parallel = parallelize(model, mesh, strategy)
    expert_names = {n for n, p in model.named_parameters() if ep.owns(p)}
    assert len(expert_names) == 3 * LAYERS

    errors = []
    me = mesh.coordinate("dp")
    for step in range(2):
        loss = model(batch_of(me))
        if not torch.allclose(loss, losses[me], rtol=1e-5, atol=1e-7):
            errors.append(f"step{step} loss {loss.item():.6f} != reference {losses[me].item():.6f}")
        loss.backward()
        parallel.sync()
        ep.sync()
        for (name, p), (_, ref) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            want = local_piece(ref.grad, name in expert_names, mesh, strategy)
            if p.grad is None or p.grad.shape != want.shape:
                errors.append(f"step{step} {name}: grad {None if p.grad is None else tuple(p.grad.shape)}")
            elif not torch.allclose(p.grad, want, rtol=1e-4, atol=1e-6):
                errors.append(f"step{step} {name}: max err {(p.grad - want).abs().max():.2e}")
        for p in model.parameters():
            p.grad = None
    assert not errors, " | ".join(errors[:4])


def test_meta_and_checkpoint(mesh, strategy: str, tmp: str):
    world = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cpu")
    with torch.device("meta"):
        model = TinyMoE()
    if strategy == "fsdp":
        ep, parallel = parallelize(model, mesh, strategy, device)
    else:
        ep = EP(mesh, model.ep_plan())
        materialize(model, device)
    # every rank inits its own shards on its own seed
    torch.manual_seed(100 + rank)
    model.init_weights()
    if strategy == "fsdp":
        parallel.finish_init()
    ep.finish_init(model)
    if strategy == "ddp":
        parallel = DDP(
            model, mesh, dim="dp", dim_of=lambda p: "dp_expert" if ep.owns(p) else "dp"
        )

    errors = []
    held = sum(p.numel() for p in model.parameters())
    full = sum(p.numel() for p in build_model().parameters())
    if held >= full:
        errors.append(f"rank holds {held} of {full} params: nothing was sharded")
    for name, p in model.named_parameters():
        peers = all_gather(p.data.unsqueeze(0).contiguous(), mesh, "ep")
        same = all(torch.equal(peers[0], other) for other in peers[1:])
        if ep.owns(p) and same:
            errors.append(f"{name}: every ep rank holds the same experts")
        if not ep.owns(p) and not same:
            errors.append(f"{name}: dense param differs across ep ranks")

    # the loss the mesh computes, averaged over every data rank
    with torch.no_grad():
        loss = model(batch_of(mesh.coordinate("dp")))
    if strategy == "fsdp":
        # a forward with no backward leaves units gathered
        parallel.reshard()
    total = all_reduce(loss.detach().clone(), mesh, "dp", "sum") / world

    # save sharded, load into ONE unsharded model, and ask it the same question
    ckpt = Checkpointer(os.path.join(tmp, f"{strategy}_{world}"), rank=rank, world_size=world)
    layout = ShardLayout.of(model, parallel, mesh, ep=ep)
    writes = all(
        mesh.coordinate(axis) == 0
        for axis in ("dp_replicate", "dp_shard")
        if axis not in parallel.shard_dims
    )
    if writes:
        ckpt.save_model(model, 1, layout)
    ckpt.commit(1)
    dist.barrier()
    whole = TinyMoE()
    ckpt.load_model(whole, 1)
    with torch.no_grad():
        want = sum(whole(batch_of(r)) for r in range(world)) / world
    if not torch.allclose(total, want, rtol=1e-5, atol=1e-7):
        errors.append(f"mesh loss {total.item():.6f} != reassembled model's {want.item():.6f}")
    assert not errors, " | ".join(errors[:4])


KIMI_CFG = {
    "num_layers": 4, "emb_dim": 32, "num_heads": 2, "head_dim": 8,
    "qk_nope_dim": 8, "qk_rope_dim": 4, "v_head_dim": 8,
    "q_lora_rank": 8, "kv_lora_rank": 8, "vocab_size": 64,
    "latent_dim": 16, "moe_dim": 8, "num_experts": 8, "experts_per_token": 2,
    "shared_dim": 16, "dense_dim": 32, "attn_res_block_size": 2,
}


def test_kimi(mesh, strategy: str):
    """
    the real moe model under ep, with activation checkpointing on: every
    block's forward (dispatch included) is re-run inside backward. grads,
    and the expert load the balancing rule sees, must match one process.
    """
    from torchure.models.kimi_k31.kimi_k31 import KimiK31

    def build() -> KimiK31:
        torch.manual_seed(0)
        model = KimiK31(**KIMI_CFG)
        model.init_weights()
        model.activation_checkpointing = True
        return model

    def ids_of(data_rank: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(2000 + data_rank)
        return torch.randint(0, KIMI_CFG["vocab_size"], (2, 12), generator=g)

    def loss_of(model, ids):
        return model(ids).float().logsumexp(-1).mean()

    world = dist.get_world_size()
    reference = build()
    (sum(loss_of(reference, ids_of(r)) for r in range(world)) / world).backward()
    want_load = torch.stack([moe.load for moe in reference.moe_layers()])

    model = build()
    ep, parallel = parallelize(model, mesh, strategy)
    expert_names = {n for n, p in model.named_parameters() if ep.owns(p)}
    loss_of(model, ids_of(mesh.coordinate("dp"))).backward()
    parallel.sync()
    ep.sync()

    errors = []
    for (name, p), (_, ref) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        want = local_piece(ref.grad, name in expert_names, mesh, strategy)
        if p.grad is None or p.grad.shape != want.shape:
            errors.append(f"{name}: grad {None if p.grad is None else tuple(p.grad.shape)}")
        elif not torch.allclose(p.grad, want, rtol=1e-3, atol=1e-6):
            errors.append(f"{name}: max err {(p.grad - want).abs().max():.2e}")
    load = torch.stack([moe.load for moe in model.moe_layers()])
    load = all_reduce(load.clone(), mesh, "dp", "sum")
    if not torch.equal(load, want_load):
        errors.append("expert load summed over the mesh != the reference's")
    stats = model.balance_step(lambda t: all_reduce(t, mesh, "dp", "sum"))
    bias = model.moe_layers()[0].router_bias
    peers = all_gather(bias.unsqueeze(0).contiguous(), mesh, "dp")
    if not all(torch.equal(peers[0], other) for other in peers[1:]):
        errors.append("router bias diverged across ranks after balance_step")
    if not (stats["moe/min_load"] <= 1.0 <= stats["moe/max_load"]):
        errors.append(f"implausible balance stats {stats}")
    assert not errors, " | ".join(errors[:4])


LAYOUTS = [
    ("ep2 ddp", {"ep": 2}, "ddp"),
    ("ep4 ddp", {"ep": 4}, "ddp"),
    ("dp2 x ep2 ddp", {"dp": 2, "ep": 2}, "ddp"),
    ("dp_shard2 x ep2 fsdp", {"dp_shard": 2, "ep": 2}, "fsdp"),
    ("dp_replicate2 x dp_shard2 x ep2 fsdp (hsdp)", {"dp_replicate": 2, "dp_shard": 2, "ep": 2}, "fsdp"),
]


def _worker(rank: int, world_size: int, spec: dict, strategy: str, label: str, tmp: str):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29563")
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    mesh = build_mesh(spec)
    failed = False
    tests = [
        ("grad parity", lambda: test_grad_parity(mesh, strategy)),
        ("meta build + checkpoint", lambda: test_meta_and_checkpoint(mesh, strategy, tmp)),
        ("kimi k3.1 + activation checkpointing", lambda: test_kimi(mesh, strategy)),
    ]
    for name, test in tests:
        try:
            test()
            msg = "PASS"
        except AssertionError as e:
            msg = f"FAIL: {e}"
        except Exception as e:  # noqa: BLE001
            msg = f"ERROR: {type(e).__name__}: {e}"
        verdicts = [None] * world_size
        dist.all_gather_object(verdicts, msg)
        bad = {r: v for r, v in enumerate(verdicts) if v != "PASS"}
        if rank == 0:
            print(f"[{label}] {name}: "
                  + (" || ".join(f"rank{r}: {v}" for r, v in bad.items()) or "PASS"), flush=True)
        failed = failed or bool(bad)
    dist.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="torchure_ep_test_")
    try:
        for label, spec, strategy in LAYOUTS:
            world = 1
            for size in spec.values():
                world *= size
            mp.spawn(_worker, args=(world, spec, strategy, label, tmp), nprocs=world)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
