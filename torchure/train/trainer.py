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
from datetime import datetime

import torch
import torch.nn as nn
import torchdata
from tokenizers import Tokenizer

from torchure.checkpoint.checkpointer import Checkpointer
from torchure.core.collective import all_reduce, barrier
from torchure.core.grad_helper import clip_grad_norm
from torchure.core.mesh import Mesh
from torchure.dataloader.builder import build_dataloader
from torchure.dataloader.prefetcher import CUDAPrefetcher
from torchure.models.builder import build_model
from torchure.objectives.builder import build_objective
from torchure.optimizer.builder import build_optimizer, build_scheduler
from torchure.parallelism.data_parallel import DDP
from torchure.parallelism.fsdp2 import FSDP
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
        # we might want to add some asserts that
        # makes sure that the model and tokenizer
        # vocab sizes are the same
        self.config = load_json(train_config_path)
        self.rank = rank
        self.world_size = world_size
        # benchmark mode (train.py --steps) is a fixed-step tps measurement,
        # not an experiment: no checkpoints, and no wandb run either
        self.benchmark = benchmark
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)  # have the process own this specific GPU
        self.mesh = self._build_mesh()
        self.dp_size = self.mesh.size("dp")

        # which data-parallel strategy _parallelize builds. read here rather
        # than inside it because it also decides whether checkpointing is
        # allowed at all (see the guard below).
        self.parallelism = self.config.get("parallelism", "ddp")

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
        self.max_norm = self.config["optimizer"].get("max_norm")

        self.dataloader = self._build_dataloader()
        self.global_batch_size = self.config["data"]["global_batch_size"]
        self.micro_batch_size = self.config["data"]["micro_batch_size"]

        # one backward consumes micro_batch_size seqs on each of the dp ranks
        # simultaneously. identical on every rank (config + mesh only, no
        # rank-local state), which is what keeps the collective count per step
        # in lockstep across ranks.
        seqs_per_backward = self.micro_batch_size * self.dp_size
        assert self.global_batch_size % seqs_per_backward == 0, (
            f"global_batch_size {self.global_batch_size} must be divisible by "
            f"micro_batch_size {self.micro_batch_size} x dp {self.dp_size} "
            f"= {seqs_per_backward}"
        )
        self.microbatches_per_step = self.global_batch_size // seqs_per_backward

        
        # two different names on purpose. the checkpoint dir is keyed on the
        # config name alone so a relaunch lands in the SAME directory and
        # resume="auto" can find it -- a timestamp here would hand every
        # launch a fresh empty dir and silently restart from step 0 (and,
        # under ddp, give each rank its own datetime.now() and its own dir).
        # the timestamp belongs on the wandb display name, where per-launch
        # is exactly what you want.
        time_launched = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = f"{self.config['run_name']}_{time_launched}"
        self.checkpointer_path = f"{PROJECT_DIR}/checkpoints/{self.config['run_name']}"
        self.checkpointer = Checkpointer(self.checkpointer_path)
        self.resume = self.config["checkpointing"]["resume"]
        self.save_steps = self.config["checkpointing"]["save_steps"]
        # sharded save/load is roadmap phase 3. until it lands, checkpoint()
        # has rank 0 write its own tensors as if they were the whole model --
        # under fsdp those are one rank's shards, and the file only announces
        # itself as corrupt at resume. fail at startup rather than at the first
        # save_steps boundary, hours in. benchmark mode never checkpoints, so
        # `--steps` stays the way to measure an fsdp run before phase 3.
        assert self.benchmark or not self.parallel.shard_dims, (
            f"parallelism={self.parallelism!r} shards model state across "
            f"{self.parallel.shard_dims} and the checkpointer is still unsharded "
            "(roadmap phase 3). run with --steps (benchmark mode, no "
            "checkpointing) until sharded checkpointing lands."
        )
        # a "continue" run that grows total_steps past the checkpointed
        # schedule wants the freshly-built scheduler's warmup/decay, not the
        # old one -- loading the old scheduler state would clobber
        # total_steps/warmup_ratio/decay_ratio right back to the prior run's
        # values (LRScheduler.load_state_dict is a blind __dict__.update)
        self.resume_scheduler = self.config["checkpointing"].get("resume_scheduler", True)
        # restoring a streaming IterableDataset's exact position means every
        # dataloader worker replays (downloads + discards) everything it already
        # consumed in its current shard -- for a resume deep into training that
        # can take a long time with zero visible progress. false trades exact
        # data-order continuity for an immediate start: the stream re-shuffles
        # from the top instead of picking up mid-shard.
        self.resume_dataloader = self.config["checkpointing"].get("resume_dataloader", True)
        # set by _resume when it finds a checkpoint: the wandb run this
        # experiment was already logging to, so we reattach instead of
        # splitting one loss curve across a new run per restart
        self.resume_wandb_id: str | None = None
        self.start_step = self._resume()
        self._setup_wandb()

        # make the dataloader iterable, has to be after
        # checkpointing resumption logic 
        self.dataloader_iter = iter(self.dataloader)
        # prefetch batch N+1's host->device copy on a side stream while step N
        # computes; see torchure/dataloader/prefetcher.py.
        self.prefetcher = CUDAPrefetcher(self.dataloader_iter, self.device)


    def _setup_wandb(self) -> None:
        """
        rank 0 only: every rank calling init would open world_size duplicate
        runs racing over the same name. runs after _resume so we can hand
        wandb the run id we recovered from the checkpoint.
        """
        self.wandb_run = None
        wandb_cfg = self.config.get("wandb", {})
        if not wandb_cfg.get("enabled", False) or self.rank != 0 or self.benchmark:
            return
        assert HAS_WANDB, (
            "wandb.enabled is true in the config but wandb isn't installed: "
            "uv sync --extra wandb"
        )

        # the full config doubles as the run's searchable hyperparameters, so
        # two runs can be diffed in the ui instead of by digging up the json
        self.wandb_run = wandb.init(
            project=wandb_cfg.get("project", "torchure"),
            entity=wandb_cfg.get("entity"),
            name=self.run_name,
            id=self.resume_wandb_id,
            resume="must" if self.resume_wandb_id else None,
            config=self.config,
        )

    def _log_metrics(self, metrics: dict, step: int, tokens_per_step: int, time: float) -> None:
        # `step` is the 1-based count of completed steps, matching what
        # checkpoint() writes. wandb DROPS any log at a step <= the last one
        # recorded, so a resumed run whose numbering disagrees with the
        # pre-crash run loses steps silently -- keep the two conventions equal.
        if self.wandb_run is None:
            return
        # .item() syncs on the device, but the print below already does via
        # __format__ on a cuda tensor, so this costs nothing extra
        self.wandb_run.log(
            {
                "train/loss": metrics["loss"].item(),
                "train/grad_norm": metrics["grad_norm"].item(),
                "train/lr": self.scheduler.get_last_lr()[0],
                "train/tokens": step * tokens_per_step,
                "perf/tps": tokens_per_step / time,
                "perf/step_time": time,
                "perf/peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
            step=step,
        )

    def _build_model(self) -> nn.Module:
        # for single gpu right now, when we want to do
        # sharding, we need this to be an empty init
        # where we actually don't init any weights yet
        model_cfg = self.config["model"]
        return build_model(model_cfg["name"], model_cfg["config"])

    def _parallelize(self, model: nn.Module) -> nn.Module:
        """
        select the data-parallel strategy and attach it as `self.parallel`.
        tensor/expert/context parallel slot in here later, driven by the same
        mesh and composing with whichever of these is chosen.

        the trainer talks to the result through three things only --
        `requires_sync`, `sync()`, and `shard_dims` -- so train_step never
        branches on which strategy is live. that's the whole abstraction; it
        stays this thin until something needs more.

        runs before _compile on purpose: the grad hooks sit on params at the
        AccumulateGrad boundary, outside the compiled regions. at dp=1 (ddp) or
        dp_shard=1 (fsdp) the ctor is a complete no-op -- no hooks, no buffers,
        no groups touched.
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
            fsdp_cfg = self.config.get("fsdp", {})
            self.parallel = FSDP(
                model,
                self.mesh,
                shard_dim="dp_shard",
                # hsdp: reduce-scatter along dp_shard, then all_reduce the
                # sharded grads along dp_replicate. None when there is nothing
                # to replicate over, so the ctor doesn't build a second path
                # for a size-1 axis.
                replicate_dim="dp_replicate" if self.mesh.size("dp_replicate") > 1 else None,
                prefetch_depth=fsdp_cfg.get("prefetch_depth", 2),
            )
        else:
            raise ValueError(
                f"unknown parallelism {self.parallelism!r}; expected 'ddp' or 'fsdp'"
            )
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
        # norm weights stay fp32 ON PURPOSE. they used to be cast to bf16 here
        # to kill the "bf16 input vs fp32 weight" mismatch warning, but that
        # made them bf16 *parameters*, so AdamW updated them in bf16: bf16
        # spacing at 1.0 is 2^-8 ~ 0.0039 while a step's update is ~lr = 3e-4,
        # so every update rounded back to the same value and the gains sat
        # frozen at their init for the whole run -- silently, since the loss
        # curve only degrades slightly. the mismatch is handled the right way
        # instead: _compile covers the blocks and the final norm, and compile
        # generates a fused kernel that takes the fp32 weight directly.

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

    def _build_mesh(self) -> Mesh:
        """
        canonicalize the config's mesh spec, then register the flattened `dp`
        axis every consumer downstream uses (dataloader split, loss reduce,
        ddp's grad sync).

        `dp: N` is shorthand for `dp_shard: N` -- sharding is the interesting
        axis, and plain ddp is `dp_replicate: N` spelled out.

        the spec is REBUILT in a fixed axis order rather than passed through,
        because dict order is load-bearing twice over: Mesh gives the last axis
        stride 1 (so tp/cp innermost puts their groups on adjacent ranks, i.e.
        the fast links), and mesh.flatten asserts its axes follow spec order --
        which is what makes coordinate("dp") equal the row-major index
        all_gather along that axis relies on. neither should be something a
        config author can get wrong by typing the keys in a different order.
        """
        # COPY: both pops below would otherwise mutate self.config["mesh"] in
        # place and leave it empty. that dict is snapshotted into the
        # checkpoint by save_trainer and compared by _resume's mesh-change
        # guard, so mutating it writes "mesh": {} into every checkpoint and
        # turns that guard into {} == {} -- vacuously true, silently gone.
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
            # size-1 axes are not free -- Mesh._group_for yields one group PER
            # RANK for them, so each unused axis costs world_size new_group
            # calls (22 vs 6 at 4 ranks with all six declared). the two dp axes
            # are the exception: flatten() needs both to exist, even at size 1.
            if size > 1 or axis in ("dp_replicate", "dp_shard"):
                spec[axis] = size

        mesh = Mesh(spec)
        mesh.flatten(("dp_replicate", "dp_shard"), "dp")
        return mesh

    
    # @debug_time
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
        # model/optimizer/scheduler are replicas under ddp: rank 0 writes the
        # single copy (sharded state is roadmap phase 3). dataloader position
        # is per dp shard, so every dp rank writes its own file.
        if self.rank == 0:
            self.checkpointer.save_model(self.model, step)
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
                    # so a resume reattaches to this run instead of opening a
                    # new one and cutting the loss curve in half
                    "wandb_id": self.wandb_run.id if self.wandb_run else None,
                },
                step,
            )
        self.checkpointer.save_dataloader(self.dataloader, step, self.mesh.coordinate("dp"))
        # fence: no rank moves on (or exits) while writers are mid-file
        barrier()
        if self.rank == 0:
            self.checkpointer.mark_complete(step)
            
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
        # a different dp size would silently misread the per-rank dataloader
        # files (subset of shards, wrong global batch) -- refuse instead
        assert state["config"]["mesh"] == self.config["mesh"], (
            f"resume across a mesh change is unsupported: checkpoint has "
            f"{state['config']['mesh']}, config has {self.config['mesh']}"
        )
        self.checkpointer.load_model(self.model, step)
        self.checkpointer.load_optimizer(self.optimizer, step)
        if self.resume_scheduler:
            self.checkpointer.load_scheduler(self.scheduler, step)
        else:
            # keep this run's freshly-built scheduler (its total_steps /
            # warmup_ratio / decay_ratio come from the current config, not
            # the checkpointed run's) but still pick up the curve at the
            # step the prior run left off, matching the numbering
            # load_scheduler would otherwise have restored
            self.scheduler.last_epoch = step
        if self.resume_dataloader:
            self.checkpointer.load_dataloader(self.dataloader, step, self.mesh.coordinate("dp"))
        # rank 0's rng lands on every rank; harmless while nothing samples
        # per step, revisit with per-rank seed derivation (roadmap 0.3)
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], self.device)
        # .get, not [...]: checkpoints written before wandb landed have no
        # such key, and they should still resume
        self.resume_wandb_id = state.get("wandb_id")
        return state["step"]

    def micro_train_step(self) -> torch.Tensor:
        batch = self.get_batch_prefetcher()
        # cast to bf16 so we can take advantage of sdpa
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = self.objective.compute_loss(self.model, batch) / self.microbatches_per_step

        loss.backward()
        return loss.detach()

    @record_time
    def train_step(self) -> dict[str, torch.Tensor]:
        loss = torch.zeros((), device=self.device)
        self.parallel.requires_sync = False
        for _ in range(self.microbatches_per_step - 1):
            loss += self.micro_train_step()

        # the last microbatch is the one whose hooks launch the bucket
        # all-reduces, over the fully accumulated grads. (fsdp accepts and
        # ignores this: it reduce-scatters every microbatch into the sharded
        # fp32 grad rather than holding an unsharded grad across them.)
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

        if self.dp_size > 1:
            all_reduce(loss, self.mesh, "dp", "avg")

        return {"loss": loss, "grad_norm": grad_norm}

    @debug_time
    def train_n_step_test(self, n_steps: int) -> None:
        # global_batch_size already spans every dp rank (and, once roadmap 1.3
        # lands, every accumulated microbatch), so tokens/step is just it x seq
        tokens_per_step = self.config["data"]["seq_len"] * self.global_batch_size
        for step in range(n_steps):
            metrics, time = self.train_step()
            if self.rank == 0:
                print(f"{step=} || loss={metrics['loss']:.4f} || gnorm={metrics['grad_norm']:.3f} "
                    f"|| lr={self.scheduler.get_last_lr()[0]:.2e} || tps={tokens_per_step / time:.0f}")

        if self.rank == 0:
            print(f"peak cuda mem: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


    def train(self) -> None:
        """
        we currently operate in a step regime, not epoch
        we could switch, but maybe no need
        """
        tokens_per_step = self.config["data"]["seq_len"] * self.global_batch_size
        try:
            for step in range(self.start_step, self.num_train_steps):
                metrics, time = self.train_step()
                if ((step + 1) % self.save_steps == 0 and step != 0) or (step + 1) == self.num_train_steps:
                    self.checkpoint(step + 1)
                if self.rank == 0:
                    self._log_metrics(metrics, step + 1, tokens_per_step, time)
                    print(f"{step=} || loss={metrics['loss']:.4f} || gnorm={metrics['grad_norm']:.3f} "
                        f"|| lr={self.scheduler.get_last_lr()[0]:.2e} || tps={tokens_per_step / time:.0f}")

            if self.rank == 0:
                print(f"peak cuda mem: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        finally:
            # in a finally so a crash marks the run crashed, instead of
            # leaving it "running" in the ui until wandb times it out
            if self.wandb_run is not None:
                self.wandb_run.finish()

# no __main__ here on purpose: a Trainer can't be built without a process group
# (Mesh asserts on it), so launching this file directly always crashed. the one
# entrypoint is torchure/train/train.py, which does the rendezvous first --
# including for the single-gpu benchmark path (--steps).

