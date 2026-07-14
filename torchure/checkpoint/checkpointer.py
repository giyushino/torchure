"""
checkpointing for the single gpu case: one directory per step
(checkpoints/<run>/<step>/) with one torch.save file per component.
load_* mirrors save_* and applies the state in place, so resume is
"build everything like a fresh run, then load_* each piece".

distributed later means swapping the torch.save calls for
torch.distributed.checkpoint (sharded save/load); the per-step
directory layout and the save_*/load_* surface stay the same so the
trainer doesn't churn.
"""
import os

import torch
import torch.nn as nn


class Checkpointer:
    def __init__(self, checkpoint_save_path: str):
        self.checkpoint_save_path = checkpoint_save_path
        os.makedirs(self.checkpoint_save_path, exist_ok=True)

    def _step_dir(self, step: int, create: bool = False) -> str:
        step_dir = os.path.join(self.checkpoint_save_path, str(step))
        if create:
            os.makedirs(step_dir, exist_ok=True)
        return step_dir

    def _save(self, obj, step: int, name: str) -> None:
        torch.save(obj.state_dict(), os.path.join(self._step_dir(step, create=True), name))

    def _load(self, step: int, name: str, weights_only: bool = True):
        # map_location="cpu": load_state_dict copies into the existing
        # (possibly cuda) params/state, so deserializing straight to gpu
        # would just spike memory for no benefit.
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

    def save_dataloader(self, dataloader, step: int) -> None:
        self._save(dataloader, step, "dataloader.pt")

    def load_model(self, model: nn.Module, step: int) -> None:
        model.load_state_dict(self._load(step, "model.pt"))

    def load_optimizer(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        optimizer.load_state_dict(self._load(step, "optimizer.pt"))

    def load_scheduler(self, scheduler: torch.optim.lr_scheduler.LRScheduler, step: int) -> None:
        scheduler.load_state_dict(self._load(step, "scheduler.pt"))

    def load_dataloader(self, dataloader, step: int) -> None:
        # dataloader state carries arbitrary python from the hf dataset /
        # worker states, which weights_only rejects; we wrote this file
        # ourselves so full unpickling is fine.
        dataloader.load_state_dict(self._load(step, "dataloader.pt", weights_only=False))
