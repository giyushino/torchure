"""
save -> load roundtrip for torchure/checkpoint/checkpointer.py.

exercises the layout the trainer writes (<run>/<step>/{model,optimizer,
scheduler,dataloader}.pt), then restores every component into freshly
built replicas and checks they match: params exact, optimizer moments
exact, scheduler position/lr exact, and the dataloader resumes from the
next unseen batch rather than batch 0.

also covers resume discovery: latest()/valid_step() must only surface
steps carrying a .complete marker, so a run that died mid-checkpoint
falls back to the last whole one instead of loading a torn dir.

cpu-only, no gpu or dataset download needed:
    uv run tests/checkpointer.py
"""

import os
import shutil
import tempfile

import torch
import torch.nn as nn

from torchdata.stateful_dataloader import StatefulDataLoader

from torchure.checkpoint.checkpointer import Checkpointer
from torchure.optimizer.scheduler import WarmupStableDecayScheduler


def build_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8))


def build_state(seed: int):
    model = build_model(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupStableDecayScheduler(
        optimizer, total_steps=100, warmup_ratio=0.1, decay_ratio=0.1
    )
    loader = StatefulDataLoader(list(range(64)), batch_size=4, num_workers=0)
    return model, optimizer, scheduler, loader


def train_steps(model, optimizer, scheduler, n: int) -> None:
    torch.manual_seed(42)
    for _ in range(n):
        model(torch.randn(4, 8)).pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()


def check_roundtrip(tmp: str) -> None:
    ckpt = Checkpointer(os.path.join(tmp, "run"))
    step = 5

    # source state: a model trained a few steps + a partially consumed loader,
    # so every component has non-trivial state to roundtrip
    model, optimizer, scheduler, loader = build_state(seed=0)
    it = iter(loader)
    for _ in range(3):
        next(it)
    train_steps(model, optimizer, scheduler, step)

    ckpt.save_model(model, step)
    ckpt.save_optimizer(optimizer, step)
    ckpt.save_scheduler(scheduler, step)
    ckpt.save_dataloader(loader, step)

    for name in ("model.pt", "optimizer.pt", "scheduler.pt", "dataloader_dp0.pt"):
        path = os.path.join(tmp, "run", str(step), name)
        assert os.path.isfile(path), f"missing {path}"

    # re-saving the same step must overwrite, not crash on an existing dir
    ckpt.save_model(model, step)

    # what the original loader would yield next; the restored one must match
    expected_next = next(it)

    # fresh replicas with deliberately different init, then restore
    model2, optimizer2, scheduler2, loader2 = build_state(seed=1)
    ckpt.load_model(model2, step)
    ckpt.load_optimizer(optimizer2, step)
    ckpt.load_scheduler(scheduler2, step)
    ckpt.load_dataloader(loader2, step)

    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), model2.state_dict().items(), strict=True
    ):
        assert k1 == k2, f"state_dict key mismatch: {k1} vs {k2}"
        torch.testing.assert_close(v1, v2)

    s1, s2 = optimizer.state_dict(), optimizer2.state_dict()
    assert s1["param_groups"] == s2["param_groups"]
    for pid, st in s1["state"].items():
        for key, val in st.items():
            torch.testing.assert_close(val, s2["state"][pid][key])

    assert scheduler2.last_epoch == scheduler.last_epoch
    assert scheduler2.get_last_lr() == scheduler.get_last_lr()

    resumed_next = next(iter(loader2))
    torch.testing.assert_close(resumed_next, expected_next)

    # trainer state: a plain dict (step, rng, config snapshot), no
    # state_dict holder. cpu-only here; the trainer adds cuda_rng on
    # real runs. rng restore must reproduce the exact next draw.
    torch.manual_seed(7)
    trainer_state = {
        "step": step,
        "cpu_rng": torch.get_rng_state(),
        "config": {"run_name": "test", "lr": 1e-3},
    }
    expected_draw = torch.randn(4)
    ckpt.save_trainer(trainer_state, step)
    loaded = ckpt.load_trainer(step)
    assert loaded["step"] == step
    assert loaded["config"] == trainer_state["config"]
    torch.set_rng_state(loaded["cpu_rng"])
    torch.testing.assert_close(torch.randn(4), expected_draw)


def check_completion_marker(tmp: str) -> None:
    """
    a step dir exists from the moment its first file lands, so the dir
    alone proves nothing. each rank commits a marker after its last file --
    a step missing any rank's marker is one we died partway through, and
    resume discovery has to skip it.
    """
    ckpt = Checkpointer(os.path.join(tmp, "marker_run"))
    model, optimizer, scheduler, loader = build_state(seed=0)

    def write_all(step: int) -> None:
        ckpt.save_model(model, step)
        ckpt.save_optimizer(optimizer, step)
        ckpt.save_scheduler(scheduler, step)
        ckpt.save_dataloader(loader, step)

    assert ckpt.latest() is None, "empty run dir has nothing to resume from"

    # every file on disk but no marker: still not resumable. this is the
    # window between the last save and the marker write
    write_all(10)
    assert not ckpt.is_complete(10)
    assert ckpt.latest() is None, "unmarked step must not be picked up"
    assert ckpt.valid_step(10) is None

    ckpt.commit(10)
    assert ckpt.is_complete(10)
    assert ckpt.latest() == 10
    assert ckpt.valid_step(10) == 10

    # step 20 dies after model.pt -- the torn write the marker exists for.
    # auto-resume has to fall back to 10 rather than load a half-written 20
    ckpt.save_model(model, 20)
    assert not ckpt.is_complete(20)
    assert ckpt.latest() == 10, "torn newest step must fall back to the last complete one"
    # and naming the torn step explicitly must be refused, not honored
    assert ckpt.valid_step(20) is None, "explicit resume must reject a torn step too"

    # finishing 20 promotes it
    write_all(20)
    ckpt.commit(20)
    assert ckpt.latest() == 20
    assert ckpt.valid_step(20) == 20

    # marking an already-marked step is a no-op, not an error
    ckpt.commit(20)
    assert ckpt.latest() == 20

    # a step that was never written at all
    assert not ckpt.is_complete(30)
    assert ckpt.valid_step(30) is None

    # non-numeric entries in the run dir must not break the scan
    os.makedirs(os.path.join(tmp, "marker_run", "scratch"), exist_ok=True)
    assert ckpt.latest() == 20


def check_per_rank_markers(tmp: str) -> None:
    """a step is complete only once EVERY rank of the run has committed."""
    path = os.path.join(tmp, "rank_run")
    model, _, _, _ = build_state(seed=0)
    r0 = Checkpointer(path, rank=0, world_size=2)
    r1 = Checkpointer(path, rank=1, world_size=2)

    r0.save_model(model, 5)
    r0.commit(5)
    assert r0.latest() is None, "rank 1 hasn't committed; step must not be resumable"
    r1.save_rng({"cpu_rng": torch.get_rng_state()}, 5, dp_rank=1)
    r1.commit(5)
    assert r0.latest() == 5 and r1.latest() == 5

    # re-saving the step withdraws that rank's marker until it commits again
    r0.save_model(model, 5)
    assert r0.latest() is None, "a step being rewritten must not read as complete"
    r0.commit(5)
    assert r0.latest() == 5

    # checkpoints written before per-rank markers carry a single .complete
    os.makedirs(os.path.join(path, "7"))
    open(os.path.join(path, "7", ".complete"), "w").close()
    assert r0.latest() == 7

    # per-dp-rank rng: each rank gets its own stream back, and a rank the
    # checkpoint has nothing for gets None rather than someone else's
    torch.manual_seed(11)
    state = torch.get_rng_state()
    expected = torch.randn(4)
    r1.save_rng({"cpu_rng": state}, 9, dp_rank=1)
    r1.commit(9)
    torch.set_rng_state(r1.load_rng(9, dp_rank=1)["cpu_rng"])
    torch.testing.assert_close(torch.randn(4), expected)
    assert r1.load_rng(9, dp_rank=3) is None


def check_async(tmp: str) -> None:
    """
    async save must write the state as it was at save time, not as it is when
    the background thread gets to it, and must not mark the step before then.
    """
    path = os.path.join(tmp, "async_run")
    ckpt = Checkpointer(path, async_save=True)
    model, optimizer, scheduler, loader = build_state(seed=0)
    train_steps(model, optimizer, scheduler, 3)
    want_model = {k: v.clone() for k, v in model.state_dict().items()}
    want_moments = {
        i: st["exp_avg"].clone() for i, st in optimizer.state_dict()["state"].items()
    }

    ckpt.save_model(model, 3)
    ckpt.save_optimizer(optimizer, 3)
    ckpt.save_scheduler(scheduler, 3)
    ckpt.save_dataloader(loader, 3)
    assert not os.path.exists(os.path.join(path, "3", "model.pt")), (
        "async save_* must only snapshot; nothing is written before commit"
    )
    assert ckpt.latest() is None
    ckpt.commit(3)
    # training carries on, mutating params and adam state in place, while
    # (or before) the writer thread runs
    train_steps(model, optimizer, scheduler, 3)
    ckpt.wait()
    assert ckpt.latest() == 3

    model2, optimizer2, scheduler2, _ = build_state(seed=1)
    ckpt.load_model(model2, 3)
    ckpt.load_optimizer(optimizer2, 3)
    for k, v in model2.state_dict().items():
        assert torch.equal(v, want_model[k]), f"{k}: async save captured later state"
    for i, st in optimizer2.state_dict()["state"].items():
        assert torch.equal(st["exp_avg"], want_moments[i]), "adam state captured late"

    # a failed background write surfaces on the next wait(), not silently,
    # and leaves the step unmarked
    ckpt.save_model(model, 4)
    os.makedirs(os.path.join(path, "4", "model.pt"))  # the write target is a dir
    ckpt.commit(4)
    try:
        ckpt.wait()
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed async save did not raise from wait()")
    assert ckpt.latest() == 3


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="torchure_ckpt_test_")
    try:
        check_roundtrip(tmp)
        check_completion_marker(tmp)
        check_per_rank_markers(tmp)
        check_async(tmp)
        print("all checkpointer checks passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
