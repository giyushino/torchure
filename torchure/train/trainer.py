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
import os

import torch
import torch.nn as nn
import torchdata
from tokenizers import Tokenizer

from torchure.checkpoint.checkpointer import Checkpointer
from torchure.dataloader.builder import build_dataloader
from torchure.dataloader.prefetcher import CUDAPrefetcher
from torchure.core.mesh import Mesh
from torchure.models.builder import build_model
from torchure.objectives.builder import build_objective
from torchure.optimizer.builder import build_optimizer, build_scheduler
from torchure.utils import record_time, debug_time, get_project_dir


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# allow tf32 for the fp32 matmuls that autocast leaves alone (rope, some
# reductions) and for cudnn. bf16 autocast already covers the big matmuls, so
# this is a small but free win and is orthogonal to distributed.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

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
        self.mesh = Mesh(self.config["mesh"])
        
        self.ignore_index = self.config["objective"]["config"]["ignore_index"]
        self.tokenizer = Tokenizer.from_pretrained(self.config["data"]["tokenizer"])

        self.model = self._build_model()
        self._init_weights(self.model)
        self.model = self._parallelize(self.model)
        self._compile(self.model)

        self.objective = self._build_objective()
        self.optimizer = self._build_optimizer(self.model)
        self.scheduler = self._build_scheduler(self.optimizer)
        self.num_train_steps = self.config["optimizer"]["scheduler"]["total_steps"]

        self.dataloader = self._build_dataloader()

        self.checkpointer_path = f"{PROJECT_DIR}/checkpoints/{self.config['run_name']}"
        self.checkpointer = Checkpointer(self.checkpointer_path)
        self.resume = self.config["checkpointing"]["resume"]
        self.save_steps = self.config["checkpointing"]["save_steps"]
        self.start_step = self._resume()

        # make the dataloader iterable, has to be after
        # checkpointing resumption logic 
        self.dataloader_iter = iter(self.dataloader)
        # prefetch batch N+1's host->device copy on a side stream while step N
        # computes; see torchure/dataloader/prefetcher.py.
        self.prefetcher = CUDAPrefetcher(self.dataloader_iter, self.device)

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

    def _compile(self, model: nn.Module) -> None:
        """
        compile per transformer block rather than the whole model.

        this is the same granularity FSDP2 wants: each block becomes one
        compiled region, so the graph is captured once and reused for all
        identical blocks (fast warmup, no giant whole-model graph), and it
        composes with per-block FSDP wrapping/activation-checkpointing later.
        gated by config so eager stays available for debugging.
        """
        if not self.config.get("compile", True):
            return
        # "default" is the sweet spot here; "max-autotune-no-cudagraphs" buys a
        # few % more at the cost of a much longer warmup (which multiplies once
        # every rank compiles), so it's opt-in via config.
        mode = self.config.get("compile_mode", "default")
        for block in model.blocks:
            block.compile(mode=mode)
        # the final norm otherwise runs eager with a bf16 input vs fp32 weight
        # dtype mismatch (no fused kernel); compiling it is free and keeps the
        # block-level granularity FSDP2 wants.
        model.norm.compile(mode=mode)

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
        return build_dataloader(
            self.config["data"], self.tokenizer, self.ignore_index, 
            self.mesh.coordinate("dp"), self.mesh.size("dp")
        )
    
    @debug_time
    def get_batch_prefetcher(self) -> dict[str, torch.Tensor]:
        # the prefetcher already queued this batch's copy on a side stream last
        # step; this waits for it and kicks off the next one. pin_memory=True on
        # the DataLoader is what makes the async copy actually overlap.
        return next(self.prefetcher)
    
    def get_batch(self) -> dict[str, torch.Tensor]:
        curr_batch = next(self.dataloader_iter)
        # https://docs.pytorch.org/docs/2.12/notes/cuda.html#cuda-memory-pinning
        # note that this only works since we set pin_memory true in the constructor
        return {k: v.to(self.device, non_blocking=True) for k, v in curr_batch.items()}
    
    @debug_time
    def checkpoint(self, step: int) -> None:
        self.checkpointer.save_model(self.model, step)
        self.checkpointer.save_dataloader(self.dataloader, step)
        self.checkpointer.save_optimizer(self.optimizer, step)
        self.checkpointer.save_scheduler(self.scheduler, step)
        # written AFTER completing `step`, so _resume returns step + 1. rng
        # capture is what keeps resume exact once anything samples per step
        # (dropout, masked diffusion); the config snapshot is provenance for
        # init_from and future drift warnings.
        self.checkpointer.save_trainer(
            {
                "step": step,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device),
                "config": self.config,
            },
            step,
        )

    def _resume(self) -> int:
        if self.resume is None:
            return 0
        if self.resume == "auto":
            step = self.checkpointer.latest()
            if step is None:
                return 0
        elif isinstance(self.resume, int):
            step = self.checkpointer.valid_step(self.resume)
            assert step is not None, "non valid step to resume from"
        else:
            raise ValueError("invalid value for resume, must be either None, 'auto', or valid integer training step")


        self.checkpointer.load_model(self.model, step)
        self.checkpointer.load_optimizer(self.optimizer, step)
        self.checkpointer.load_scheduler(self.scheduler, step)
        self.checkpointer.load_dataloader(self.dataloader, step)
        state = self.checkpointer.load_trainer(step)
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], self.device)
        return state["step"] + 1
    

    @record_time
    def train_step(self) -> torch.Tensor:
            batch = self.get_batch_prefetcher()
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
        tokens_per_step = self.config["data"]["seq_len"] * self.config["data"]["batch_size"]
        for step in range(n_steps):
            loss, time = self.train_step()
            print(f"{step=} || {loss=} || tps={tokens_per_step / time}")
        print(f"peak cuda mem: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    def train(self) -> None:
        """
        we currently operate in a step regime, not epoch
        we could switch, but maybe no need
        """
        tokens_per_step = self.config["data"]["seq_len"] * self.config["data"]["batch_size"]
        for step in range(self.start_step, self.num_train_steps):
            loss, time = self.train_step()
            if (step + 1) % self.save_steps == 0 and step != 0:
                self.checkpoint(step)
            print(f"{step=} || {loss=} || tps={tokens_per_step / time}")

        print(f"peak cuda mem: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    test = Trainer(f"{PROJECT_DIR}/configs/qwen3_dense_climbmix.json", 0, 0, 1)
    loss = test.train_n_step_test(1000)
    print(f"{loss=}")
     

