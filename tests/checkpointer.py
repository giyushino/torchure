"""
save -> load roundtrip for torchure/checkpoint/checkpointer.py.

exercises the layout the trainer writes (<run>/<step>/{model,optimizer,
scheduler,dataloader}.pt), then restores every component into freshly
built replicas and checks they match: params exact, optimizer moments
exact, scheduler position/lr exact, and the dataloader resumes from the
next unseen batch rather than batch 0.

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
from torchure.optimizer.scheduler import WarmupStableDecaySchedulder


def build_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8))


def build_state(seed: int):
    model = build_model(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupStableDecaySchedulder(
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


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="torchure_ckpt_test_")
    try:
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

        for name in ("model.pt", "optimizer.pt", "scheduler.pt", "dataloader.pt"):
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

        print("all checkpointer roundtrip checks passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
