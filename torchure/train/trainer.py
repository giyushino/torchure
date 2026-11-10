"""
training orchestrator: build -> init -> parallelize -> compile -> optimizer -> data

TODO: replace the @debug_time profiling decorators with a real profiler
"""

import atexit
import json
import os
import signal
import time
from datetime import datetime

import torch
import torch.nn as nn
import torchdata
from tokenizers import Tokenizer

from torchure.checkpoint.checkpointer import Checkpointer, ShardLayout
from torchure.core.collective import all_reduce
from torchure.core.grad_helper import clip_grad_norm
from torchure.core.mesh import Mesh
from torchure.dataloader.builder import build_dataloader, shutdown_workers
from torchure.dataloader.prefetcher import CUDAPrefetcher
from torchure.models.builder import build_model
from torchure.objectives.builder import build_objective
from torchure.optimizer.builder import build_optimizer, build_scheduler
from torchure.parallelism.data_parallel import DDP
from torchure.parallelism.fsdp2 import FSDP
from torchure.utils import debug_time, get_project_dir


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# tf32 for the fp32 matmuls bf16 autocast leaves alone (rope, some reductions)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

PROJECT_DIR = get_project_dir()

# config strings -> dtypes for fsdp's mixed precision. None means "no cast":
# gathered params keep the fp32 master dtype (what an fp32 debug run wants).
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    None: None,
    "none": None,
}

def load_json(train_config_path: str) -> dict:
    with open(train_config_path, 'r') as file:
        return json.load(file)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Trainer:
    def __init__(
        self,
        train_config_path: str,
        rank: int,
        local_rank: int,
        world_size: int,
        benchmark: bool = False,
    ):
        # TODO: assert model and tokenizer vocab sizes match
        self.config = load_json(train_config_path)
        self.rank = rank
        self.world_size = world_size
        self.seed = self.config.get("seed", 67)
        torch.manual_seed(self.seed)
        # benchmark mode (train.py --steps): fixed-step tps run, no checkpoints or wandb
        self.benchmark = benchmark
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        self.mesh = self._build_mesh()
        self.dp_size = self.mesh.size("dp")

        # read here, not in _parallelize, because it also gates checkpointing below
        self.parallelism = self.config.get("parallelism", "ddp")

        self.ignore_index = self.config["objective"]["config"]["ignore_index"]
        self.tokenizer = Tokenizer.from_pretrained(self.config["data"]["tokenizer"])

        # ddp: build -> init -> replicate. fsdp: build on the meta device ->
        # shard -> init the shards, so no rank ever holds the full weights.
        self.model = self._build_model()
        if self.parallelism == "fsdp":
            self.model = self._parallelize(self.model)
            self._init_shards(self.model)
        else:
            self._init_weights(self.model)
            self.model = self._parallelize(self.model)
        self.model.activation_checkpointing = self.config.get("activation_checkpointing", False)
        # init ran on the shared seed so every rank built the same model; from
        # here each dp rank draws its own stream (dropout, diffusion masks).
        # tp/cp peers share a dp coordinate, so they stay in lockstep.
        torch.manual_seed(self.seed + 1 + self.mesh.coordinate("dp"))
        self._compile(self.model)

        self.objective = self._build_objective()
        self.optimizer = self._build_optimizer(self.model)
        self.scheduler = self._build_scheduler(self.optimizer)
        self.num_train_steps = self.config["optimizer"]["scheduler"]["total_steps"]
        self.max_norm = self.config["optimizer"].get("max_norm")

        self.dataloader = self._build_dataloader()
        self.global_batch_size = self.config["data"]["global_batch_size"]
        self.micro_batch_size = self.config["data"]["micro_batch_size"]

        # depends only on config + mesh, so every rank runs the same number of
        # microbatches and the collectives stay in lockstep
        seqs_per_backward = self.micro_batch_size * self.dp_size
        assert self.global_batch_size % seqs_per_backward == 0, (
            f"global_batch_size {self.global_batch_size} must be divisible by "
            f"micro_batch_size {self.micro_batch_size} x dp {self.dp_size} "
            f"= {seqs_per_backward}"
        )
        self.microbatches_per_step = self.global_batch_size // seqs_per_backward

        # checkpoint dir is deliberately NOT timestamped: a relaunch must land in
        # the same dir for resume="auto" to find it (and ranks would disagree on
        # datetime.now()). only the wandb display name gets the timestamp.
        time_launched = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = f"{self.config['run_name']}_{time_launched}"
        self.checkpointer_path = f"{PROJECT_DIR}/checkpoints/{self.config['run_name']}"
        self.checkpointer = Checkpointer(
            self.checkpointer_path,
            rank=rank,
            world_size=world_size,
            # snapshot to cpu, write in a background thread while training runs
            async_save=self.config["checkpointing"].get("async_save", True),
        )
        self.resume = self.config["checkpointing"]["resume"]
        self.save_steps = self.config["checkpointing"]["save_steps"]
        # logging is the only host sync in the step loop
        self.log_every = self.config.get("log_every", 10)
        # false for "continue" runs that change the schedule: loading the old
        # scheduler state would overwrite total_steps/warmup/decay with the old values
        self.resume_scheduler = self.config["checkpointing"].get("resume_scheduler", True)
        # restoring stream position replays everything consumed in each worker's
        # current shard, which can be slow. false skips that and reshuffles instead.
        self.resume_dataloader = self.config["checkpointing"].get("resume_dataloader", True)
        # set by _resume so wandb reattaches to the existing run
        self.resume_wandb_id: str | None = None
        # the batch the prefetcher held at the checkpoint, set by _resume
        self._resumed_batch: dict[str, torch.Tensor] | None = None
        self.start_step = self._resume()
        self._setup_wandb()

        # must come after _resume restores the dataloader state
        self.dataloader_iter = iter(self.dataloader)
        # workers ignore SIGTERM (see builder._ignore_sigterm), so they must be
        # torn down explicitly on every exit path or the process hangs at exit
        atexit.register(shutdown_workers, self.dataloader_iter)
        self.prefetcher = CUDAPrefetcher(
            self.dataloader_iter, self.device, first_batch=self._resumed_batch
        )

        # set by the SIGTERM handler train() installs; see _stop_requested
        self._sigterm = False


    def _setup_wandb(self) -> None:
        """rank 0 only. runs after _resume so it can reattach to the checkpointed run id."""
        self.wandb_run = None
        wandb_cfg = self.config.get("wandb", {})
        if not wandb_cfg.get("enabled", False) or self.rank != 0 or self.benchmark:
            return
        assert HAS_WANDB, (
            "wandb.enabled is true in the config but wandb isn't installed: "
            "uv sync --extra wandb"
        )

        self.wandb_run = wandb.init(
            project=wandb_cfg.get("project", "torchure"),
            entity=wandb_cfg.get("entity"),
            name=self.run_name,
            id=self.resume_wandb_id,
            resume="must" if self.resume_wandb_id else None,
            config=self.config,
        )

    def _log_metrics(
        self, loss: float, grad_norm: float, step: int, tokens_per_step: int, step_time: float
    ) -> None:
        # `step` is the count of completed steps, same as checkpoint(). wandb
        # silently drops logs at step <= the last one, so these must agree.
        if self.wandb_run is None:
            return
        self.wandb_run.log(
            {
                "train/loss": loss,
                "train/grad_norm": grad_norm,
                "train/lr": self.scheduler.get_last_lr()[0],
                "train/tokens": step * tokens_per_step,
                "perf/tps": tokens_per_step / step_time,
                "perf/step_time": step_time,
                "perf/peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
            step=step,
        )

    def _build_model(self) -> nn.Module:
        model_cfg = self.config["model"]
        if self.parallelism == "fsdp":
            # shapes only: fsdp allocates this rank's shards, nothing more
            with torch.device("meta"):
                return build_model(model_cfg["name"], model_cfg["config"])
        return build_model(model_cfg["name"], model_cfg["config"])

    def _parallelize(self, model: nn.Module) -> nn.Module:
        """
        attach the data-parallel strategy as `self.parallel`. the trainer only
        uses `requires_sync`, `sync()`, and `shard_dims`, so train_step never
        branches on the strategy.

        runs before _compile so the grad hooks stay outside the compiled regions.
        """
        if self.parallelism == "ddp":
            ddp_cfg = self.config.get("ddp", {})
            self.parallel = DDP(
                model,
                self.mesh,
                dim="dp",
                bucket_mb=ddp_cfg.get("bucket_mb", 25),
            )
        elif self.parallelism == "fsdp":
            # FSDP at dp_shard=1 is a full no-op, so it would silently skip
            # the dp_replicate grad sync too. pure replication is ddp's job.
            assert self.mesh.size("dp_shard") > 1, (
                f"parallelism 'fsdp' needs dp_shard > 1 (mesh {self.config['mesh']}); "
                "use parallelism 'ddp' for pure replication"
            )
            fsdp_cfg = self.config.get("fsdp", {})
            self.parallel = FSDP(
                model,
                self.mesh,
                # one unit per block, same granularity as _compile
                units=model.blocks,
                shard_dim="dp_shard",
                # hsdp: reduce-scatter on dp_shard, then all_reduce on dp_replicate
                replicate_dim="dp_replicate" if self.mesh.size("dp_replicate") > 1 else None,
                param_dtype=_DTYPES[fsdp_cfg.get("param_dtype", "bfloat16")],
                reduce_dtype=_DTYPES[fsdp_cfg.get("reduce_dtype", "float32")],
                prefetch=fsdp_cfg.get("prefetch", 1),
                device=self.device,
            )
        else:
            raise ValueError(
                f"unknown parallelism {self.parallelism!r}; expected 'ddp' or 'fsdp'"
            )
        return model

    def _compile(self, model: nn.Module) -> None:
        """
        compile per transformer block: one graph reused across identical blocks,
        matching fsdp's per-block granularity.
        """
        if not self.config.get("compile", True):
            return
        mode = self.config.get("compile_mode", "default")
        # compile `forward`, not the module: module.compile() would also trace
        # the module hooks (fsdp's gathers), which must stay eager
        for block in model.blocks:
            block.forward = torch.compile(block.forward, mode=mode)
        # gives the final norm a fused kernel for bf16 input x fp32 weight
        model.norm.compile(mode=mode)

    def _init_weights(self, model: nn.Module) -> None:
        model.to(self.device)
        model.init_weights()
        # don't cast norm weights to bf16: bf16 spacing at 1.0 (~0.004) is far
        # larger than an adamw update (~lr), so the gains would never move

    def _init_shards(self, model: nn.Module) -> None:
        """
        init a model fsdp materialized from meta: each rank fills only its own
        shards. seeded per shard coordinate, so shards differ from each other
        while hsdp replicas of one shard agree.
        """
        torch.manual_seed(self.seed + self.mesh.coordinate("dp_shard"))
        model.init_weights()
        self.parallel.finish_init()

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        optim_cfg = self.config["optimizer"]
        return build_optimizer(model, optim_cfg["name"], optim_cfg["config"])

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        scheduler_cfg = self.config["optimizer"]["scheduler"]
        return build_scheduler(optimizer, scheduler_cfg)

    def _build_objective(self):
        obj_cfg = self.config["objective"]
        return build_objective(obj_cfg["name"], obj_cfg["config"])
    
    @debug_time
    def _build_dataloader(self) -> torchdata.stateful_dataloader.StatefulDataLoader:
        return build_dataloader(
            self.config["data"], self.tokenizer, self.ignore_index, 
            self.mesh.coordinate("dp"), self.mesh.size("dp"),
            # so a job-wide SIGTERM can't kill the workers before the
            # checkpoint; __init__ registers the matching teardown
            shield_sigterm=True,
        )

    def _build_mesh(self) -> Mesh:
        """
        canonicalize the config's mesh spec and add the flattened `dp` axis.
        `dp: N` is shorthand for `dp_shard: N`.

        the spec is rebuilt in a fixed axis order because order matters: the
        last axis gets stride 1 (tp/cp on adjacent ranks), and mesh.flatten
        requires its axes in spec order.
        """
        # copy: self.config["mesh"] is saved in checkpoints and checked on resume
        mesh_cfg = dict(self.config["mesh"])
        assert not ("dp" in mesh_cfg and ("dp_replicate" in mesh_cfg or "dp_shard" in mesh_cfg)), (
            f"mesh spec {self.config['mesh']}: `dp` is shorthand for `dp_shard`; "
            "don't mix it with the explicit axes"
        )

        if "dp" in mesh_cfg:
            mesh_cfg["dp_shard"] = mesh_cfg.pop("dp")

        AXIS_ORDER = ("pp", "dp_replicate", "dp_shard", "cp", "tp", "ep")

        spec = {}
        for axis in AXIS_ORDER:
            size = mesh_cfg.pop(axis, 1)
            # size-1 axes cost world_size new_group calls, so drop them, except
            # the dp axes, which flatten() needs
            if size > 1 or axis in ("dp_replicate", "dp_shard"):
                spec[axis] = size

        mesh = Mesh(spec)
        mesh.flatten(("dp_replicate", "dp_shard"), "dp")
        return mesh


    def get_batch_prefetcher(self) -> dict[str, torch.Tensor]:
        return next(self.prefetcher)
    
    def get_batch(self) -> dict[str, torch.Tensor]:
        curr_batch = next(self.dataloader_iter)
        # non_blocking relies on pin_memory=True in the dataloader
        return {k: v.to(self.device, non_blocking=True) for k, v in curr_batch.items()}
    
    def _shard_layout(self) -> ShardLayout:
        """how this run shards model/optimizer state, for the checkpointer."""
        return ShardLayout.of(self.model, self.parallel, self.mesh)

    def _writes_model_state(self) -> bool:
        """
        one rank per shard writes model/optimizer state: the one at coordinate
        0 on every dp axis the state is replicated (not sharded) over. that is
        rank 0 alone under ddp, and the dp_replicate=0 row under fsdp/hsdp.
        """
        return all(
            self.mesh.coordinate(axis) == 0
            for axis in ("dp_replicate", "dp_shard")
            if axis not in self.parallel.shard_dims
        )

    @debug_time
    def checkpoint(self, step: int) -> None:
        # one rank per shard writes model/optimizer state, rank 0 the rest;
        # every rank writes its own dataloader and rng state. no barrier: each
        # rank commits its own marker and the step is complete when all have
        if self._writes_model_state():
            layout = self._shard_layout()
            self.checkpointer.save_model(self.model, step, layout)
            self.checkpointer.save_optimizer(self.optimizer, step, layout)
        if self.rank == 0:
            self.checkpointer.save_scheduler(self.scheduler, step)
            self.checkpointer.save_trainer(
                {
                    "step": step,
                    "config": self.config,
                    "wandb_id": self.wandb_run.id if self.wandb_run else None,
                },
                step,
            )
        dp_rank = self.mesh.coordinate("dp")
        self.checkpointer.save_rng(
            {
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device),
            },
            step,
            dp_rank,
        )
        self.checkpointer.save_dataloader(self.dataloader, step, dp_rank)
        # the loader is one batch ahead of training; that batch is saved here
        self.checkpointer.save_prefetcher(self.prefetcher.state_dict(), step, dp_rank)
        self.checkpointer.commit(step)

    def _on_sigterm(self, signum, frame) -> None:
        """
        first SIGTERM: checkpoint and exit at the next log point. a second one
        means "now": put the default handler back and die.

        dataloader workers ignore SIGTERM, so a signal sent to the whole
        process group still leaves the run able to reach that log point.
        """
        if self._sigterm:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)
        self._sigterm = True

    def _stop_requested(self) -> bool:
        """
        collective: has any rank been sent SIGTERM? ranks get the signal at
        different moments, so they agree on it here, at a point every rank
        reaches on the same step. only called at log points, which already
        sync with the host.
        """
        flag = torch.tensor(float(self._sigterm), device=self.device)
        if self.dp_size > 1:
            all_reduce(flag, self.mesh, "dp", "max")
        return flag.item() > 0
            
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


        state = self.checkpointer.load_trainer(step)
        # model/optimizer state reshards onto any mesh, but the per-rank
        # dataloader files would be misread under a different dp layout
        assert state["config"]["mesh"] == self.config["mesh"] or not self.resume_dataloader, (
            f"resume across a mesh change (checkpoint {state['config']['mesh']}, "
            f"config {self.config['mesh']}) can't restore the per-rank dataloader "
            "state; set checkpointing.resume_dataloader to false to reshuffle"
        )
        layout = self._shard_layout()
        self.checkpointer.load_model(self.model, step, layout)
        # load_state_dict overwrites param_groups wholesale; keep this config's
        # hparams and take only per-param state from the checkpoint
        config_hparams = [
            {k: v for k, v in group.items() if k != "params"}
            for group in self.optimizer.param_groups
        ]
        self.checkpointer.load_optimizer(self.optimizer, step, layout)
        # a resumed scheduler owns lr
        lr_keys = ("lr", "initial_lr") if self.resume_scheduler else ()
        for group, hparams in zip(self.optimizer.param_groups, config_hparams, strict=True):
            group.update({k: v for k, v in hparams.items() if k not in lr_keys})

        if self.resume_scheduler:
            self.checkpointer.load_scheduler(self.scheduler, step)
        else:
            # fresh scheduler from this config, fast-forwarded to `step`, with
            # its lr applied now so the first resumed step doesn't use the step-0 lr
            self.scheduler.last_epoch = step
            lrs = self.scheduler.get_lr()
            for group, lr in zip(self.optimizer.param_groups, lrs, strict=True):
                group["lr"] = lr
            self.scheduler._last_lr = lrs
        if self.resume_dataloader:
            dp_rank = self.mesh.coordinate("dp")
            self.checkpointer.load_dataloader(self.dataloader, step, dp_rank)
            # older checkpoints have none: they resume one batch ahead
            held = self.checkpointer.load_prefetcher(step, dp_rank)
            if held is not None:
                self._resumed_batch = held["next_batch"]
        # each dp rank resumes its own stream. older checkpoints carry only
        # rank 0's, in trainer.pt. a rank with neither (resume onto a larger
        # mesh) keeps the seed-derived stream it already has.
        rng = self.checkpointer.load_rng(step, self.mesh.coordinate("dp"))
        if rng is None and "cpu_rng" in state:
            rng = state
        if rng is not None:
            torch.set_rng_state(rng["cpu_rng"])
            torch.cuda.set_rng_state(rng["cuda_rng"], self.device)
        # .get: older checkpoints have no wandb_id
        self.resume_wandb_id = state.get("wandb_id")
        return state["step"]

    def micro_train_step(self) -> torch.Tensor:
        batch = self.get_batch_prefetcher()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = self.objective.compute_loss(self.model, batch) / self.microbatches_per_step

        loss.backward()
        return loss.detach()

    def train_step(self) -> dict[str, torch.Tensor]:
        loss = torch.zeros((), device=self.device)
        self.parallel.requires_sync = False
        for _ in range(self.microbatches_per_step - 1):
            loss += self.micro_train_step()

        # only the last microbatch syncs grads (fsdp ignores this flag)
        self.parallel.requires_sync = True
        loss += self.micro_train_step()
        self.parallel.sync()
        grad_norm = clip_grad_norm(
            self.model.parameters(), self.mesh, self.max_norm,
            shard_dims=self.parallel.shard_dims,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # rank-local loss; _run_steps averages over dp once per log window
        return {"loss": loss, "grad_norm": grad_norm}

    def _run_steps(self, start: int, end: int, checkpoint: bool) -> None:
        """
        step loop shared by train() and the --steps benchmark. metrics stay on
        device and only sync with the host at log points, so the cpu can run
        ahead of the gpu. step time is averaged over the log window and
        excludes checkpoint saves.
        """
        tokens_per_step = self.config["data"]["seq_len"] * self.global_batch_size
        loss_sum = torch.zeros((), device=self.device)
        gnorm_sum = torch.zeros((), device=self.device)
        window = 0
        t0 = time.perf_counter()

        for step in range(start, end):
            metrics = self.train_step()
            loss_sum += metrics["loss"]
            gnorm_sum += metrics["grad_norm"]
            window += 1

            done = step + 1
            save = checkpoint and (done % self.save_steps == 0 or done == end)
            if save or done % self.log_every == 0 or done == end:
                stop = checkpoint and self._stop_requested()
                save = save or stop
                if self.dp_size > 1:
                    all_reduce(loss_sum, self.mesh, "dp", "avg")
                loss = loss_sum.item() / window
                gnorm = gnorm_sum.item() / window
                step_time = (time.perf_counter() - t0) / window
                if self.rank == 0:
                    self._log_metrics(loss, gnorm, done, tokens_per_step, step_time)
                    print(f"step={done} || loss={loss:.4f} || gnorm={gnorm:.3f} "
                        f"|| lr={self.scheduler.get_last_lr()[0]:.2e} "
                        f"|| tps={tokens_per_step / step_time:.0f}")
                loss_sum.zero_()
                gnorm_sum.zero_()
                window = 0
                if save:
                    self.checkpoint(done)
                if stop:
                    if self.rank == 0:
                        print(f"SIGTERM: checkpointed step {done}, exiting")
                    break
                t0 = time.perf_counter()

        if self.rank == 0:
            print(f"peak cuda mem: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    @debug_time
    def train_n_step_test(self, n_steps: int) -> None:
        self._run_steps(0, n_steps, checkpoint=False)

    def train(self) -> None:
        signal.signal(signal.SIGTERM, self._on_sigterm)
        try:
            self._run_steps(self.start_step, self.num_train_steps, checkpoint=True)
        finally:
            # past the loop nothing answers the flag, so SIGTERM kills again
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            try:
                # the last async save must be on disk before the process exits
                self.checkpointer.wait()
            finally:
                # so a crash marks the wandb run crashed instead of leaving it "running"
                if self.wandb_run is not None:
                    self.wandb_run.finish()

