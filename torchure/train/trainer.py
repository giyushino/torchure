"""
training orchestrator

lifecycle is split into explicit phases so that moving to
distributed/parallelism later means filling in _parallelize and
_init_weights rather than reordering the constructor:

    build -> parallelize -> init/materialize -> optimizer -> data

for a single gpu run _parallelize is a no-op and world_size == 1.
"""

import json
import torch
import torch.nn as nn

from torchure.dataloader.dataloader import DataLoader
from torchure.dataloader.utils import build_dataloader
from torchure.models.utils import build_model
from torchure.objectives.utils import build_objective
from torchure.optim.utils import build_optimizer


def load_json(train_config_path: str) -> dict:
    with open(train_config_path, 'r') as file:
        return json.load(file)


class Trainer:
    def __init__(self, train_config_path: str, rank: int, local_rank: int, world_size: int):
        self.config = load_json(train_config_path)
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)  # have the process own this specific GPU

        # phased setup -- order matters (see module docstring)
        self.model = self._build_model()
        self.model = self._parallelize(self.model)
        self._init_weights(self.model)
        self.optimizer = self._build_optimizer(self.model)
        self.objective = self._build_objective()
        self.dataloader = self._build_dataloader()

    def _build_model(self) -> nn.Module:
        # for single gpu right now, when we want to do
        # sharding, we need this to be an empty init
        # where we actually don't init any weights yet
        model_cfg = self.config["model"]
        return build_model(model_cfg["name"], model_cfg["config"])

    def _parallelize(self, model: nn.Module) -> nn.Module:
        """
        apply tensor/expert/context parallel + fsdp2 here, driven by the
        device mesh + a per-model plan in torchure/parallelism/.

        single gpu: no-op. fill in once core/mesh.py + parallelism/ exist.
        """
        return model

    def _init_weights(self, model: nn.Module) -> None:
        # single gpu again, when we init for sharding
        # we won't have any weights
        # model.to_empty(device=self.device)
        model.to(self.device)

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        optim_cfg = self.config["optim"]
        return build_optimizer(model, optim_cfg["name"], optim_cfg["config"])

    def _build_objective(self):
        obj_cfg = self.config["objective"]
        return build_objective(obj_cfg["name"], obj_cfg["config"])

    def _build_dataloader(self) -> DataLoader:
        return build_dataloader(self.config["data"], self.rank, self.world_size)

    def train_step(self, batch) -> torch.Tensor:
        """
        one optimization step. rough shape:
            move batch to device -> objective.compute_loss(model, batch)
            -> backward -> optimizer.step -> optimizer.zero_grad
        return the loss (detached) for logging.

        TODO: implement. grad accumulation / clipping / mixed precision slot
        in here later.
        """
        raise NotImplementedError

    def train(self) -> None:
        """
        TODO: main loop over self.dataloader for the configured number of
        steps/epochs, calling self.train_step and logging.
        """
        raise NotImplementedError



