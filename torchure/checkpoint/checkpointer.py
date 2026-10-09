"""
unsharded checkpointing: checkpoints/<run>/<step>/ with one torch.save file
per component. load_* applies state in place onto a freshly built trainer.

TODO: torch.distributed.checkpoint for sharded state, same save_*/load_* api
"""
import os

import torch
import torch.nn as nn


class Checkpointer:
    def __init__(self, checkpoint_save_path: str):
        self.checkpoint_save_path = checkpoint_save_path
        os.makedirs(self.checkpoint_save_path, exist_ok=True)
    
    def latest(self) -> int | None:
        return max(self._steps(), default=None)

    def valid_step(self, step: int) -> int | None:
        return step if step in self._steps() else None

    def _steps(self) -> list[int]:
        return [
            int(d) for d in os.listdir(self.checkpoint_save_path)
            if d.isdigit() and self.is_complete(int(d))
        ]

    def _step_dir(self, step: int, create: bool = False) -> str:
        step_dir = os.path.join(self.checkpoint_save_path, str(step))
        if create:
            os.makedirs(step_dir, exist_ok=True)
        return step_dir

    def _save(self, obj, step: int, name: str) -> None:
        torch.save(obj.state_dict(), os.path.join(self._step_dir(step, create=True), name))

    def _load(self, step: int, name: str, weights_only: bool = True):
        # cpu: load_state_dict copies into the existing tensors anyway
        return torch.load(
            os.path.join(self._step_dir(step), name),
            map_location="cpu",
            weights_only=weights_only,
        )

    def save_model(self, model: nn.Module, step: int) -> None:
        self._save(model, step, "model.pt")

    def save_optimizer(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        self._save(optimizer, step, "optimizer.pt")

    def save_scheduler(self, scheduler: torch.optim.lr_scheduler.LRScheduler, step: int) -> None:
        self._save(scheduler, step, "scheduler.pt")

    def save_dataloader(self, dataloader, step: int, dp_rank: int = 0) -> None:
        self._save(dataloader, step, f"dataloader_dp{dp_rank}.pt")

    def save_trainer(self, state: dict, step: int) -> None:
        # already a plain dict, so it bypasses _save
        torch.save(state, os.path.join(self._step_dir(step, create=True), "trainer.pt"))

    def load_trainer(self, step: int) -> dict:
        return self._load(step, "trainer.pt")

    def load_model(self, model: nn.Module, step: int) -> None:
        model.load_state_dict(self._load(step, "model.pt"))

    def load_optimizer(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        optimizer.load_state_dict(self._load(step, "optimizer.pt"))

    def load_scheduler(self, scheduler: torch.optim.lr_scheduler.LRScheduler, step: int) -> None:
        scheduler.load_state_dict(self._load(step, "scheduler.pt"))

    def load_dataloader(self, dataloader, step: int, dp_rank: int = 0) -> None:
        # weights_only rejects the hf dataset state; we wrote this file ourselves
        dataloader.load_state_dict(
            self._load(step, f"dataloader_dp{dp_rank}.pt", weights_only=False)
        )

    def _complete_path(self, step: int) -> str:
        return os.path.join(self._step_dir(step), ".complete")

    def mark_complete(self, step: int) -> None:
        open(self._complete_path(step), "w").close()

    def is_complete(self, step: int) -> bool:
        return os.path.exists(self._complete_path(step))

