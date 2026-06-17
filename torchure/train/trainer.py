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
    def __init__(self, train_config_path: str):
        self.config = load_json(train_config_path)

        # single gpu runtime context. when you go distributed these come
        # from the launcher (torchrun) + a device mesh built in core/mesh.py
        self.rank = 0
        self.world_size = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # phased setup -- order matters (see module docstring)
        self.model = self._build_model()
        self.model = self._parallelize(self.model)
        self._init_weights(self.model)
        self.optimizer = self._build_optimizer(self.model)
        self.objective = self._build_objective()
        self.dataloader = self._build_dataloader()

    # --- phases -----------------------------------------------------------

    def _build_model(self) -> nn.Module:
        # structure only -- no device placement, no weight init here, so the
        # model can be sharded before it is ever materialized.
        cfg = self.config["model"]
        return build_model(cfg["name"], cfg["config"])

    def _parallelize(self, model: nn.Module) -> nn.Module:
        """
        apply tensor/expert/context parallel + fsdp2 here, driven by the
        device mesh + a per-model plan in torchure/parallelism/.

        single gpu: no-op. fill in once core/mesh.py + parallelism/ exist.
        """
        return model

    def _init_weights(self, model: nn.Module) -> None:
        """
        materialize + initialize parameters, then place on device.

        TODO single gpu: model.to(self.device) is enough for now.
        TODO distributed: materialize from meta device + rank-aware seeding.
        """
        raise NotImplementedError

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return build_optimizer(model, self.config["optimizer"])

    def _build_objective(self):
        obj = self.config["objective"]
        return build_objective(obj["name"], obj.get("config", {}))

    def _build_dataloader(self) -> DataLoader:
        return build_dataloader(self.config["data"], self.rank, self.world_size)

    # --- train loop -------------------------------------------------------

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
