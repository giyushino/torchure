"""
training orchestrator

lifecycle is split into explicit phases so that moving to
distributed/parallelism later means filling in _parallelize and
_init_weights rather than reordering the constructor:

    build -> parallelize -> init/materialize -> optimizer -> data

for a single gpu run _parallelize is a no-op and world_size == 1.

TODO:
remove the decorators for profiling... just keep that now for
ease of use then move to different profiling methods
"""

import json

import torch
import torch.nn as nn
import torchdata
from tokenizers import Tokenizer

from torchure.checkpoint.checkpointer import Checkpointer
from torchure.dataloader.builder import build_dataloader
from torchure.models.builder import build_model
from torchure.objectives.builder import build_objective
from torchure.optimizer.builder import build_optimizer, build_scheduler
from torchure.utils import record_time, debug_time, get_project_dir


PROJECT_DIR = get_project_dir()


def load_json(train_config_path: str) -> dict:
    with open(train_config_path, 'r') as file:
        return json.load(file)


class Trainer:
    def __init__(self, train_config_path: str, rank: int, local_rank: int, world_size: int):
        # we might want to add some asserts that
        # makes sure that the model and tokenizer
        # vocab sizes are the same 
        self.config = load_json(train_config_path)
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)  # have the process own this specific GPU
        
        self.ignore_index = self.config["objective"]["config"]["ignore_index"]
        self.tokenizer = Tokenizer.from_pretrained(self.config["data"]["tokenizer"])
        self.model = self._build_model()
        self.model = self._parallelize(self.model)
        self._init_weights(self.model)
        self.objective = self._build_objective()
        self.optimizer = self._build_optimizer(self.model)
        self.scheduler = self._build_scheduler(self.optimizer)
        # self.resume = self.config["resume_training"]
        # make the dataloader iterable
        self.dataloader = self._build_dataloader()
        self.dataloader_iter = iter(self.dataloader)
        self.checkpointer_path = f"{PROJECT_DIR}/checkpoints/{self.config['run_name']}"
        self.checkpointer = Checkpointer(self.checkpointer_path)

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
        # maybe we should make an ABC and require models
        # to have init_weight as a function 
        model.to(self.device)
        model.init_weights()

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        optim_cfg = self.config["optimizer"]
        return build_optimizer(model, optim_cfg["name"], optim_cfg["config"])

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        # currently we set last_epoch to -1 by default for a fresh training
        # run, we need to change this when we do resumption
        scheduler_cfg = self.config["optimizer"]["scheduler"]
        return build_scheduler(optimizer, scheduler_cfg)

    def _build_objective(self):
        obj_cfg = self.config["objective"]
        return build_objective(obj_cfg["name"], obj_cfg["config"])
    
    @debug_time
    def _build_dataloader(self) -> torchdata.stateful_dataloader.StatefulDataLoader:
        # rank/world_size are global for now; swap to the dp mesh coords
        # once core/mesh.py exists so TP/CP groups share a batch.
        return build_dataloader(
            self.config["data"], self.tokenizer, self.ignore_index, self.rank, self.world_size
        )
    
    @debug_time
    def get_batch(self) -> dict[str, torch.Tensor]:
        curr_batch = next(self.dataloader_iter)
        # https://docs.pytorch.org/docs/2.12/notes/cuda.html#cuda-memory-pinning
        # note that this only works since we set pin_memory true in the constructor
        return {k: v.to(self.device, non_blocking=True) for k, v in curr_batch.items()}

    def checkpoint(self, step: int) -> None:
        self.checkpointer.save_model(self.model, step)
        self.checkpointer.save_dataloader(self.dataloader, step)
        self.checkpointer.save_optimizer(self.optimizer, step)
        return

    @record_time
    def train_step_test(self) -> torch.Tensor:
            batch = self.get_batch()
            #print(batch)
            # cast to bf16 so we can take advantage of sdpa
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = self.objective.compute_loss(self.model, batch)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            return loss.detach()

    @debug_time
    def train_n_step_test(self, n_steps: int) -> None:
        for step in range(n_steps):
            loss, time = self.train_step_test()
            print(f"{step=} || {loss=} || tps={self.config['data']['seq_len']/time}")

    @debug_time
    def train_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        one optimization step. rough shape:
            move batch to device -> objective.compute_loss(model, batch)
            -> backward -> optimizer.step -> optimizer.zero_grad
        return the loss (detached) for logging.

        TODO: implement. grad accumulation / clipping / mixed precision slot
        in here later.
        """

        loss = self.objective.compute_loss(self.model, batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def train(self) -> None:
        """
        TODO: main loop over self.dataloader for the configured number of
        steps/epochs, calling self.train_step and logging.
        """


if __name__ == "__main__":
    test = Trainer(f"{PROJECT_DIR}/configs/qwen3_dense_climbmix.json", 0, 0, 1)
    loss = test.train_n_step_test(100)
    print(f"{loss=}")
     

