"""
parity + invariant tests for torchure/parallelism/fsdp2.py.

run (gloo/cpu, no gpus needed):
    uv run tests/fsdp2.py --world-size 4
    uv run tests/fsdp2.py --world-size 4 --backend nccl   (on a 4-gpu box)

the model is a tiny real Qwen3, not an mlp: the tied embedding (one storage,
two modules, and the weight consumed by the objective AFTER forward returns)
and the rmsnorm structure are exactly what an fsdp implementation breaks on.

two families of test here, and the second matters more:

- **parity** -- N ranks under fsdp must produce the single-process grads, and
  the same grads ddp produces. same invariant as tests/data_parallel.py.
- **invariants invisible from the outside** -- an fsdp that never frees its
  gathered buffers, or gathers a unit twice, still trains perfectly and simply
  does not do its job. those are asserted directly: storage sizes mid-forward,
  collective counts, and adam state numel.
"""

import argparse
import os
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torchure.core import collective as C
from torchure.core.grad_helper import clip_grad_norm
from torchure.core.mesh import Mesh
from torchure.core.placement import Replicate, Shard
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


class Skip(Exception):
    pass


def build_model(seed: int, device) -> Qwen3:
    torch.manual_seed(seed)
    model = Qwen3(**CFG)
    model.init_weights()
    return model.to(device)


def global_batch(n_seqs: int, device) -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    return torch.randint(0, CFG["vocab_size"], (n_seqs, SEQ), generator=g).to(device)


def loss_fn(model: Qwen3, ids: torch.Tensor) -> torch.Tensor:
    logits = model(ids)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, CFG["vocab_size"]), ids[:, 1:].reshape(-1)
    )


def fused_loss_fn(model: Qwen3, ids: torch.Tensor) -> torch.Tensor:
    """
    the path the trainer actually runs: the objective pulls
    model.lm_head.weight AFTER model.forward has returned. any design that
    reshards unit 0 on a forward-post hook hands this a freed weight.
    """
    from torchure.objectives.autoregressive import ARObjective

    obj = ARObjective(ignore_index=-100, fused_linear_ce=True, ce_chunk_size=8)
    return obj.compute_loss(model, {"input_ids": ids, "labels": ids})


def sharded_ref_grad(ref_param: torch.Tensor, mesh, dim) -> torch.Tensor:
    """the slice of a single-process grad this rank is responsible for."""
    return Shard(0).local_slice(ref_param, mesh.size(dim), mesh.coordinate(dim))


def collect(errors: list, label: str, got, expected, rtol=1e-4, atol=1e-6) -> None:
    """
    record a mismatch instead of raising.

    raising from inside a loop that still has collectives left is a DEADLOCK,
    not a failure: the rank that raises stops participating while every other
    rank blocks in the next all_gather until the process-group timeout. so
    every comparison inside a multi-cycle test collects here and the test
    raises once, after the last collective has run on every rank.
    """
    if got is None:
        errors.append(f"{label}: no grad")
        return
    if tuple(got.shape) != tuple(expected.shape):
        errors.append(
            f"{label}: shape {tuple(got.shape)} != sharded {tuple(expected.shape)} "
            "-- the reduce_scatter didn't land"
        )
        return
    try:
        torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        # keep the "Mismatched elements" and "Greatest absolute difference"
        # lines: the count vs the magnitude is what distinguishes a real bug
        # (systematic, large) from fp associativity (one element, tiny)
        lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
        detail = " ".join(ln for ln in lines if ln.startswith(("Mismatched", "Greatest")))
        errors.append(f"{label}: {detail or lines[0]}")


def finish(errors: list) -> None:
    if errors:
        raise AssertionError(f"{len(errors)} mismatch(es): " + " | ".join(errors[:4]))


def check_grads(errors, model, reference, mesh, dim, tag="", rtol=1e-4, atol=1e-6):
    for (name, p), (_, p_ref) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        collect(
            errors,
            f"{tag}{name}",
            p.grad,
            sharded_ref_grad(p_ref.grad, mesh, dim),
            rtol=rtol,
            atol=atol,
        )


# --------------------------------------------------------------------------
# parity
# --------------------------------------------------------------------------


def test_shard_and_replicate(mesh, dim, device):
    """
    ctor must (a) make every rank agree on the full weights before slicing --
    nothing in this repo seeds torch, so ranks init DIFFERENT models -- and
    (b) leave params pointing at their shard, which is what makes the
    optimizer allocate sharded state.
    """
    model = build_model(seed=dist.get_rank(), device=device)
    fsdp = FSDP(model, mesh, shard_dim=dim, replicate_dim=None)
    reference = build_model(seed=0, device=device)
    g = mesh.size(dim)

    for (name, p), (_, p_ref) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        expected = Shard(0).local_slice(p_ref.data, g, mesh.coordinate(dim))
        assert p.shape == expected.shape, (
            f"param {name}: {tuple(p.shape)} is not the sharded {tuple(expected.shape)}"
        )
        torch.testing.assert_close(
            p.data, expected, rtol=0, atol=0,
            msg=f"param {name} is not rank 0's weights (broadcast missing?)",
        )

    total = sum(p.numel() for p in reference.parameters())
    assert fsdp.sharded_numel() == total // g, (
        f"this rank owns {fsdp.sharded_numel()} of {total} params, expected "
        f"{total // g} -- the memory win is exactly this number"
    )
    assert fsdp.unsharded_bytes_live() == 0, (
        "gathered buffers hold memory before any forward has run"
    )
    assert fsdp.placements_of("norm.weight") == (Shard(0),)


def test_grad_parity(mesh, dim, device, loss=loss_fn):
    """N ranks x bs=1 under fsdp == 1 process x bs=N, sharded."""
    model = build_model(seed=dist.get_rank(), device=device)
    fsdp = FSDP(model, mesh, shard_dim=dim, replicate_dim=None)

    reference = build_model(seed=0, device=device)
    batch = global_batch(mesh.size(dim), device)
    loss(reference, batch).backward()

    local = batch[mesh.coordinate(dim)].unsqueeze(0)
    errors = []
    for cycle in range(2):  # two cycles: catches re-arm / buffer-reuse bugs
        loss(model, local).backward()
        fsdp.sync()
        check_grads(errors, model, reference, mesh, dim, tag=f"cycle{cycle} ")
        for p in model.parameters():
            p.grad = None
    finish(errors)


def test_grad_parity_fused_ce(mesh, dim, device):
    """
    the same parity, through ARObjective's fused linear+CE -- which reads
    model.lm_head.weight after forward returns. this is the single most
    likely thing to break, so it gets its own case.
    """
    model = build_model(seed=dist.get_rank(), device=device)
    fsdp = FSDP(model, mesh, shard_dim=dim, replicate_dim=None)
    reference = build_model(seed=0, device=device)
    batch = global_batch(mesh.size(dim), device)
    fused_loss_fn(reference, batch).backward()

    local = batch[mesh.coordinate(dim)].unsqueeze(0)
    fused_loss_fn(model, local).backward()
    fsdp.sync()
    errors = []
    check_grads(errors, model, reference, mesh, dim, rtol=1e-3, atol=1e-5)
    finish(errors)


def test_matches_ddp(mesh, dim, device):
    """
    ddp and fsdp are supposed to compute the identical thing. this is the
    test that catches reduce-scatter ordering and scaling mistakes, because
    both paths have to agree with each other and neither is the reference.
    """
    fsdp_model = build_model(seed=dist.get_rank(), device=device)
    fsdp = FSDP(fsdp_model, mesh, shard_dim=dim, replicate_dim=None)

    ddp_model = build_model(seed=dist.get_rank(), device=device)
    ddp = DDP(ddp_model, mesh, dim)

    batch = global_batch(mesh.size(dim), device)
    local = batch[mesh.coordinate(dim)].unsqueeze(0)

    loss_fn(fsdp_model, local).backward()
    fsdp.sync()
    loss_fn(ddp_model, local).backward()
    ddp.sync()

    errors = []
    for (name, p), (_, p_ddp) in zip(
        fsdp_model.named_parameters(), ddp_model.named_parameters(), strict=True
    ):
        collect(errors, f"fsdp!=ddp {name}", p.grad,
                sharded_ref_grad(p_ddp.grad, mesh, dim), rtol=1e-5, atol=1e-7)
    finish(errors)


def test_grad_accumulation(mesh, dim, device, microbatches: int):
    """K microbatches x N ranks == 1 process x bs=(N*K), still sharded."""
    world = mesh.size(dim)
    model = build_model(seed=dist.get_rank(), device=device)
    fsdp = FSDP(model, mesh, shard_dim=dim, replicate_dim=None)

    reference = build_model(seed=0, device=device)
    batch = global_batch(world * microbatches, device)
    loss_fn(reference, batch).backward()

    base = mesh.coordinate(dim) * microbatches
    errors = []
    for cycle in range(2):
        fsdp.requires_sync = False
        for i in range(microbatches - 1):
            (loss_fn(model, batch[base + i].unsqueeze(0)) / microbatches).backward()
        fsdp.requires_sync = True
        (loss_fn(model, batch[base + microbatches - 1].unsqueeze(0)) / microbatches).backward()
        fsdp.sync()
        check_grads(errors, model, reference, mesh, dim, tag=f"cycle{cycle} ")
        for p in model.parameters():
            p.grad = None
    finish(errors)


def test_multi_step(mesh, dim, device):
    """
    5 optimizer steps against a single-process reference. catches
    optimizer-state sharding bugs a single-backward test can't see: adam's
    moments are per-shard, and if the update is right the sharded weights must
    stay equal to the reference's slice step after step.
    """
    model = build_model(seed=dist.get_rank(), device=device)
    fsdp = FSDP(model, mesh, shard_dim=dim, replicate_dim=None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    reference = build_model(seed=0, device=device)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-2)

    g = mesh.size(dim)
    errors = []
    for step in range(5):
        batch = global_batch(g, device) + step  # a different batch each step
        batch = batch % CFG["vocab_size"]

        ref_loss = loss_fn(reference, batch)
        ref_loss.backward()

        loss = loss_fn(model, batch[mesh.coordinate(dim)].unsqueeze(0))
        loss.backward()
        fsdp.sync()

        # 1. grads, TIGHT, and BEFORE either optimizer steps -- this is the
        #    part a sharding bug breaks, and checking it here means the
        #    optimizer never gets a chance to launder the error into something
        #    that looks like rounding.
        # step 0 ONLY. at step 0 both models hold identical weights, so the
        # grads are comparable at test_grad_parity's tolerance. from step 1 the
        # two weight sets have legitimately drifted apart (see the weight check
        # below), so the grads are evaluated at different points and demanding
        # 1e-6 of them is demanding zero drift -- which is not what this test
        # is for. on nccl that shows up immediately: bs=4 and bs=1 backwards
        # pick different cuBLAS reduction orders, and by step 3 the grad gap is
        # ~2e-6 while the weights and the loss curve still track fine.
        if step == 0:
            check_grads(errors, model, reference, mesh, dim, tag=f"step{step} ",
                        rtol=1e-4, atol=1e-6)

        ref_opt.step()
        ref_opt.zero_grad()
        opt.step()
        opt.zero_grad()

        # 2. weights, at a tolerance adam's update actually permits. the
        #    update is lr * m/(sqrt(v)+eps) -- a NORMALIZED quantity, so for an
        #    element whose grad is near zero the ratio is ill-conditioned and a
        #    1e-9 grad difference (pure fp32 associativity: the reduce_scatter
        #    sums this rank's shard in a different order than the single-process
        #    reduction) can move the weight by a good fraction of lr. observed:
        #    exactly 1 element in 2048 drifting ~1.4e-5 with lr=1e-2, while the
        #    grads above stay equal to 1e-7. atol=1e-4 is still 100x tighter
        #    than any real sharding bug, which misplaces a whole shard and shows
        #    up at the O(1e-2) scale of the weights themselves.
        for (name, p), (_, p_ref) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            collect(errors, f"step{step} weight {name}", p.data,
                    Shard(0).local_slice(p_ref.data, g, mesh.coordinate(dim)),
                    rtol=1e-4, atol=1e-4)

        # 3. the loss curve, which is the roadmap's actual stated criterion.
        #    `loss` is THIS rank's bs=1 loss, not the global one -- the mean
        #    across dp is what equals the single-process bs=N loss, which is
        #    why the trainer all-reduces it before logging. comparing the local
        #    loss directly would "fail" on a completely correct run.
        mean_loss = C.all_reduce(loss.detach().clone(), mesh, dim, "avg")
        if abs(mean_loss.item() - ref_loss.item()) > 1e-4:
            errors.append(
                f"step{step} mean loss {mean_loss.item():.6f} != reference "
                f"{ref_loss.item():.6f}"
            )
    finish(errors)

    # the memory win, stated as a number: adam keeps two moments per SHARDED
    # element, so state numel must be 2x the sharded param count, not 2x total
    state_numel = sum(
        v.numel()
        for s in opt.state.values()
        for v in s.values()
        if torch.is_tensor(v) and v.dim() > 0
    )
    total = sum(p.numel() for p in reference.parameters())
    assert state_numel == 2 * (total // g), (
        f"adam state is {state_numel} elements; expected {2 * (total // g)} "
        f"(2 moments x {total}/{g}). optimizer state did not shard"
    )


def test_clip_grad_norm(mesh, dim, device):
    """the sharded norm must equal the single-process norm."""
    model = build_model(seed=dist.get_rank(), device=device)
    fsdp = FSDP(model, mesh, shard_dim=dim, replicate_dim=None)
    reference = build_model(seed=0, device=device)

    batch = global_batch(mesh.size(dim), device)
    loss_fn(reference, batch).backward()
    loss_fn(model, batch[mesh.coordinate(dim)].unsqueeze(0)).backward()
    fsdp.sync()

    ref_norm = clip_grad_norm(reference.parameters(), mesh, max_norm=None, shard_dims=())
    got = clip_grad_norm(model.parameters(), mesh, max_norm=None, shard_dims=(dim,))
    torch.testing.assert_close(
        got, ref_norm, rtol=1e-4, atol=1e-6,
        msg="sharded grad norm != single-process norm; shard_dims wrong?",
    )

    # and clipping writes through: post-clip grads must still match
    clip_grad_norm(reference.parameters(), mesh, max_norm=0.1, shard_dims=())
    clip_grad_norm(model.parameters(), mesh, max_norm=0.1, shard_dims=(dim,))
    errors = []
    check_grads(errors, model, reference, mesh, dim, tag="post-clip ")
    finish(errors)


def test_hsdp(mesh, dim, device):
    """dp_replicate=2 x dp_shard=2 grads == dp=4 ddp grads."""
    if dist.get_world_size() != 4:
        raise Skip("needs exactly 4 ranks (2x2 mesh)")
    m = Mesh({"dp_replicate": 2, "dp_shard": 2})

    model = build_model(seed=dist.get_rank(), device=device)
    fsdp = FSDP(model, m, shard_dim="dp_shard", replicate_dim="dp_replicate")

    reference = build_model(seed=0, device=device)
    batch = global_batch(4, device)
    loss_fn(reference, batch).backward()

    # under hsdp each rank still owns one sequence of the global batch
    local = batch[dist.get_rank()].unsqueeze(0)
    loss_fn(model, local).backward()
    fsdp.sync()

    errors = []
    for (name, p), (_, p_ref) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        collect(errors, f"hsdp {name}", p.grad,
                Shard(0).local_slice(p_ref.grad, 2, m.coordinate("dp_shard")))
    finish(errors)

    assert fsdp.placements_of("norm.weight") == (Replicate(), Shard(0))


# --------------------------------------------------------------------------
# the invariants that are invisible from the outside
# --------------------------------------------------------------------------


def test_storage_actually_freed(mesh, dim, device):
    """
    an fsdp that gathers correctly and never frees still trains perfectly and
    saves nothing. assert the storage size DURING a forward, between units --
    after the whole step everything is free either way.
    """
    model = build_model(seed=0, device=device)
    fsdp = FSDP(model, mesh, shard_dim=dim, replicate_dim=None)

    seen = []

    def probe(module, args):
        # fires at each block's forward-pre, before that block is gathered
        seen.append(fsdp.unsharded_bytes_live())

    handles = [b.register_forward_pre_hook(probe) for b in model.blocks]
    loss_fn(model, global_batch(1, device)).backward()
    fsdp.sync()
    for h in handles:
        h.remove()

    root_bytes = sum(
        st.unsharded.numel() * st.unsharded.element_size()
        for st in fsdp._units[0].params
    )
    block_bytes = sum(
        st.unsharded.numel() * st.unsharded.element_size()
        for st in fsdp._units[1].params
    )
    # at block i's forward-pre: unit 0 is still live (it must be -- lm_head is
    # consumed after forward returns), block i is being gathered by the
    # prefetch, and every EARLIER block has been freed. so the live total must
    # never grow with depth.
    assert seen[0] <= root_bytes + 2 * block_bytes, (
        f"{seen[0]} bytes live at the first block; expected at most unit 0 plus "
        "the prefetch window"
    )
    assert seen == sorted(seen, reverse=True) or len(set(seen)) == 1, (
        f"live gathered bytes grow with depth ({seen}) -- blocks are not being "
        "freed after their forward"
    )
    assert fsdp.unsharded_bytes_live() == 0, (
        f"{fsdp.unsharded_bytes_live()} bytes still held after backward; every "
        "unit must be resharded once its grads are reduced"
    )


def test_collective_counts(mesh, dim, device):
    """
    exactly one all_gather per param per forward, one per param per backward
    for blocks (unit 0 is kept unsharded so it is NOT re-gathered), and one
    reduce_scatter per param per step. gathering twice, or not prefetching at
    all, is otherwise invisible.
    """
    model = build_model(seed=0, device=device)
    fsdp = FSDP(model, mesh, shard_dim=dim, replicate_dim=None)

    counts = {"all_gather": 0, "reduce_scatter": 0}
    real_ag, real_rs = C.all_gather, C.reduce_scatter

    import torchure.parallelism.fsdp2 as fsdp_mod

    def counting_ag(*a, **kw):
        counts["all_gather"] += 1
        return real_ag(*a, **kw)

    def counting_rs(*a, **kw):
        counts["reduce_scatter"] += 1
        return real_rs(*a, **kw)

    fsdp_mod.all_gather, fsdp_mod.reduce_scatter = counting_ag, counting_rs
    try:
        loss_fn(model, global_batch(1, device)).backward()
        fsdp.sync()
    finally:
        fsdp_mod.all_gather, fsdp_mod.reduce_scatter = real_ag, real_rs

    n_params = len(fsdp._states)
    n_block_params = sum(len(u.params) for u in fsdp._units[1:])
    expected_ag = n_params + n_block_params  # forward all, backward blocks only
    assert counts["all_gather"] == expected_ag, (
        f"{counts['all_gather']} all_gathers, expected {expected_ag} "
        f"({n_params} in forward + {n_block_params} re-gathered in backward)"
    )
    assert counts["reduce_scatter"] == n_params, (
        f"{counts['reduce_scatter']} reduce_scatters, expected {n_params} (one "
        "per param per step)"
    )


def test_prefetch_order(mesh, dim, device):
    """recorded unit order must be module execution order, once."""
    model = build_model(seed=0, device=device)
    fsdp = FSDP(model, mesh, shard_dim=dim, replicate_dim=None)
    loss_fn(model, global_batch(1, device)).backward()
    fsdp.sync()
    assert fsdp._order_checked, "execution order was never pinned"
    assert fsdp._exec_order == list(range(len(fsdp._units))), (
        f"forward order {fsdp._exec_order} != static order; prefetch would "
        "silently degrade to none"
    )


def test_noop_at_size_one(mesh, dim, device):
    """dp_shard=1 must register no hooks, allocate nothing, launch nothing."""
    model = build_model(seed=0, device=device)
    before = [p.data_ptr() for p in model.parameters()]
    solo = Mesh({"dp_shard": dist.get_world_size()}) if False else None
    del solo

    class OneMesh:
        def size(self, d):
            return 1

        def get_group(self, d):
            return dist.group.WORLD

        def coordinate(self, d):
            return 0

    fsdp = FSDP(model, OneMesh(), shard_dim="dp_shard", replicate_dim=None)
    assert fsdp._units == [] and fsdp._states == [], "size-1 fsdp built state"
    assert [p.data_ptr() for p in model.parameters()] == before, (
        "size-1 fsdp moved a parameter's storage"
    )
    assert not model._forward_pre_hooks, "size-1 fsdp registered a hook"
    loss_fn(model, global_batch(1, device)).backward()
    fsdp.sync()  # must be a no-op, not an assert about units never firing
    for p in model.parameters():
        assert p.grad is not None and p.grad.shape == p.shape, (
            "size-1 fsdp changed the grad shape"
        )


def test_tied_embedding(mesh, dim, device):
    """
    embed and head must land in ONE unit, share one storage through every
    swap, and the shared weight's grad must be the SUM of both contributions.
    """
    model = build_model(seed=0, device=device)
    fsdp = FSDP(model, mesh, shard_dim=dim, replicate_dim=None)

    assert model.lm_head.weight is model.token_emb.weight
    owners = {
        st.name: u.index for u in fsdp._units for st in u.params
    }
    assert "token_emb.weight" in owners and owners["token_emb.weight"] == 0
    # exactly one state for the tied storage, not two
    assert sum(1 for st in fsdp._states if st.param is model.token_emb.weight) == 1

    reference = build_model(seed=0, device=device)
    batch = global_batch(mesh.size(dim), device)
    loss_fn(reference, batch).backward()
    loss_fn(model, batch[mesh.coordinate(dim)].unsqueeze(0)).backward()
    fsdp.sync()

    errors = []
    collect(errors, "tied token_emb.weight (sum of embedding + head paths)",
            model.token_emb.weight.grad,
            sharded_ref_grad(reference.token_emb.weight.grad, mesh, dim))
    finish(errors)
    # both modules still see one storage after a full cycle
    assert model.lm_head.weight.data_ptr() == model.token_emb.weight.data_ptr()


TESTS = [
    ("test_shard_and_replicate", test_shard_and_replicate),
    ("test_grad_parity", test_grad_parity),
    ("test_grad_parity_fused_ce", test_grad_parity_fused_ce),
    ("test_matches_ddp", test_matches_ddp),
    ("test_accum_2", partial(test_grad_accumulation, microbatches=2)),
    ("test_accum_3", partial(test_grad_accumulation, microbatches=3)),
    ("test_multi_step", test_multi_step),
    ("test_clip_grad_norm", test_clip_grad_norm),
    ("test_hsdp", test_hsdp),
    ("test_storage_actually_freed", test_storage_actually_freed),
    ("test_collective_counts", test_collective_counts),
    ("test_prefetch_order", test_prefetch_order),
    ("test_noop_at_size_one", test_noop_at_size_one),
    ("test_tied_embedding", test_tied_embedding),
]


def _worker(rank: int, world_size: int, backend: str):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29551")
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    mesh, dim = Mesh({"dp": world_size}), "dp"

    failed = []
    for name, test in TESTS:
        try:
            test(mesh, dim, device)
            msg = "PASS"
        except Skip as e:
            msg = f"SKIP: {e}"
        except AssertionError as e:
            msg = f"FAIL: {e}"
            failed.append(name)
        except Exception as e:  # noqa: BLE001
            msg = f"ERROR: {type(e).__name__}: {e}"
            failed.append(name)
        # gather every rank's verdict, not just rank 0's. a sharded bug is
        # frequently rank-dependent -- rank 0 owns the first slice of every
        # param and is the one rank a broadcast-from-0 bug leaves correct --
        # so reporting only rank 0 hides exactly the failures worth catching.
        verdicts = [None] * world_size
        dist.all_gather_object(verdicts, msg)
        if rank == 0:
            bad = {r: v for r, v in enumerate(verdicts) if v != "PASS" and not v.startswith("SKIP")}
            if bad:
                print(f"[{name}] " + " || ".join(f"rank{r}: {v}" for r, v in bad.items()), flush=True)
            else:
                print(f"[{name}] {verdicts[0]}", flush=True)
        if any(v != "PASS" and not v.startswith("SKIP") for v in verdicts):
            if name not in failed:
                failed.append(name)
    dist.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    args = parser.parse_args()
    mp.spawn(_worker, args=(args.world_size, args.backend), nprocs=args.world_size)
