"""
training orchestrator: build -> init -> parallelize -> compile -> optimizer -> data

TODO: replace the @debug_time profiling decorators with a real profiler
"""

import atexit
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
from torchure.core.meta import materialize
from torchure.dataloader.builder import build_dataloader, shutdown_workers
from torchure.dataloader.prefetcher import CUDAPrefetcher
from torchure.models.builder import build_model
from torchure.objectives.builder import build_objective
from torchure.optimizer.builder import build_optimizer, build_scheduler
from torchure.parallelism.data_parallel import DDP
from torchure.parallelism.fsdp2 import FSDP
from torchure.parallelism.tensor_parallel import TP
from torchure.utils import debug_time, get_project_dir, load_json

try:
    import wandb
except ImportError:
    wandb = None

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# tf32 for the fp32 matmuls bf16 autocast leaves alone (rope, some reductions)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

PROJECT_DIR = get_project_dir()

# config strings -> dtypes for fsdp's mixed precision. None means "no cast":
# gathered params keep the fp32 master dtype (what an fp32 debug run wants)
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    None: None,
    "none": None,
}

# fixed, because order matters: the last axis gets stride 1 (tp peers on
# adjacent ranks), and mesh.flatten needs its axes in spec order
MESH_AXIS_ORDER = ("pp", "dp_replicate", "dp_shard", "cp", "tp", "ep")


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
        # the mesh drops size-1 axes, so no tp axis means no tensor parallel
        self.tp_size = self.mesh.sizes.get("tp", 1)

        # "ddp" | "fsdp"
        self.parallelism = self.config.get("parallelism", "ddp")
        # set by _build_model: the data-parallel strategy (DDP | FSDP) and the
        # tensor-parallel one, if any
        self.parallel: DDP | FSDP
        self.tp: TP | None = None

        self.ignore_index = self.config["objective"]["config"]["ignore_index"]
        self.tokenizer = Tokenizer.from_pretrained(self.config["data"]["tokenizer"])

        self.model = self._build_model()
        self.model.activation_checkpointing = self.config.get("activation_checkpointing", False)
        # init ran on the shared seed so every rank built the same model; from
        # here each dp rank draws its own stream (dropout, diffusion masks).
        # tp/cp peers share a dp coordinate, so they stay in lockstep.
        torch.manual_seed(self.seed + 1 + self.mesh.coordinate("dp"))
        self._compile(self.model)

        obj_cfg = self.config["objective"]
        self.objective = build_objective(obj_cfg["name"], obj_cfg["config"])
        # the fused loss reads the head weight directly, bypassing tp's hooks
        self.objective.tp = self.tp
        optim_cfg = self.config["optimizer"]
        self.optimizer = build_optimizer(self.model, optim_cfg["name"], optim_cfg["config"])
        self.scheduler = build_scheduler(self.optimizer, optim_cfg["scheduler"])
        self.num_train_steps = optim_cfg["scheduler"]["total_steps"]
        self.max_norm = optim_cfg.get("max_norm")

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
        dataloader_iter = iter(self.dataloader)
        # workers ignore SIGTERM (see builder._ignore_sigterm), so they must be
        # torn down explicitly on every exit path or the process hangs at exit
        atexit.register(shutdown_workers, dataloader_iter)
        self.prefetcher = CUDAPrefetcher(
            dataloader_iter, self.device, first_batch=self._resumed_batch
        )

        # set by the SIGTERM handler train() installs; see _stop_requested
        self._sigterm = False
        self._sigterm_at = 0.0

    def _setup_wandb(self) -> None:
        """rank 0 only. runs after _resume so it can reattach to the checkpointed run id."""
        self.wandb_run = None
        wandb_cfg = self.config.get("wandb", {})
        if not wandb_cfg.get("enabled", False) or self.rank != 0 or self.benchmark:
            return
        assert wandb is not None, (
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
        """
        build, initialize and parallelize the model; sets self.parallel and
        self.tp.

        plain ddp: build -> init -> replicate. anything sharded (fsdp, tp or
        both): build on the meta device (shapes only) -> shard -> init the
        shards, so no rank ever holds the full weights. tp cuts first; fsdp
        then shards what tp left on the rank, and ddp replicates it.
        """
        model_cfg = self.config["model"]
        if self.parallelism != "fsdp" and self.tp_size == 1:
            model = build_model(model_cfg["name"], model_cfg["config"])
            model.to(self.device)
            model.init_weights()
            self._parallelize(model)
            return model

        with torch.device("meta"):
            model = build_model(model_cfg["name"], model_cfg["config"])
        self.tp = self._tensor_parallelize(model)
        if self.parallelism == "fsdp":
            self._parallelize(model)  # allocates this rank's shards
            self._init_shards(model)
        else:
            materialize(model, self.device)
            self._init_shards(model)
            self._parallelize(model)  # ddp, replicating what tp left
        return model

    def _parallelize(self, model: nn.Module) -> None:
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

    def _tensor_parallelize(self, model: nn.Module) -> TP | None:
        """
        split the model's weights along the mesh's tp axis. the model says
        which modules split which way. the model is still on meta here, so
        this only rewrites shapes.

        config `tp`: `vocab_parallel` (default true) shards the embedding /
        lm_head weight by vocab rows; `sequence_parallel` (default true)
        shards the activations between tp regions along the sequence (or the
        batch, `sp_dim: "batch"`); `overlap` ("none" | "backward" |
        "collective_matmul", see parallelism/tp_overlap.py) hides the sp
        collectives behind the region's matmuls.
        """
        if self.tp_size == 1:
            return None
        tp_cfg = self.config.get("tp", {})
        sequence_parallel = tp_cfg.get("sequence_parallel", True)
        seq_len = self.config["data"]["seq_len"]
        assert not sequence_parallel or seq_len % self.tp_size == 0, (
            f"sequence parallel needs seq_len {seq_len} divisible by tp {self.tp_size}"
        )
        plan = model.tp_plan(self.tp_size, vocab_parallel=tp_cfg.get("vocab_parallel", True))
        # the dim sp shards activations along between regions: "seq" (1) or
        # "batch" (0). batch keeps each rank's chunk contiguous, so gathers and
        # scatters need no layout copies; it needs micro_batch % tp == 0
        sp_dim = tp_cfg.get("sp_dim", "seq")
        assert sp_dim in ("seq", "batch"), f"tp.sp_dim {sp_dim!r}: expected 'seq' or 'batch'"
        if sp_dim == "batch":
            micro = self.config["data"]["micro_batch_size"]
            assert micro % self.tp_size == 0, (
                f"tp.sp_dim 'batch' needs micro_batch_size {micro} divisible by tp {self.tp_size}"
            )
        return TP(
            self.mesh, **plan, dim="tp", sequence_parallel=sequence_parallel,
            seq_dim=0 if sp_dim == "batch" else 1,
            overlap=tp_cfg.get("overlap", "none"),
        )

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

    def _init_shards(self, model: nn.Module) -> None:
        """
        init a model materialized from meta: each rank fills only its own
        shards. seeded per (dp_shard, tp) cell, so shards differ from each
        other; the finish_init calls then make everything that is replicated
        (hsdp replicas, params tp doesn't cut, buffers) agree.
        """
        shard = self.mesh.coordinate("dp_shard") if self.parallelism == "fsdp" else 0
        cell = shard * self.tp_size
        if self.tp is not None:
            cell += self.mesh.coordinate("tp")
        torch.manual_seed(self.seed + cell)
        model.init_weights()
        if self.parallelism == "fsdp":
            self.parallel.finish_init()
        if self.tp is not None:
            self.tp.finish_init(model)

    @debug_time
    def _build_dataloader(self) -> torchdata.stateful_dataloader.StatefulDataLoader:
        return build_dataloader(
            self.config["data"], self.tokenizer, self.ignore_index,
            self.mesh.coordinate("dp"), self.dp_size,
            # so a job-wide SIGTERM can't kill the workers before the
            # checkpoint; __init__ registers the matching teardown
            shield_sigterm=True,
        )

    def _build_mesh(self) -> Mesh:
        """
        canonicalize the config's mesh spec (into MESH_AXIS_ORDER) and add the
        flattened `dp` axis. `dp: N` is shorthand for `dp_shard: N`.
        """
        # copy: self.config["mesh"] is saved in checkpoints and checked on resume
        mesh_cfg = dict(self.config["mesh"])
        assert not ("dp" in mesh_cfg and ("dp_replicate" in mesh_cfg or "dp_shard" in mesh_cfg)), (
            f"mesh spec {self.config['mesh']}: `dp` is shorthand for `dp_shard`; "
            "don't mix it with the explicit axes"
        )

        if "dp" in mesh_cfg:
            mesh_cfg["dp_shard"] = mesh_cfg.pop("dp")

        spec = {}
        for axis in MESH_AXIS_ORDER:
            size = mesh_cfg.pop(axis, 1)
            # size-1 axes cost world_size new_group calls, so drop them, except
            # the dp axes, which flatten() needs
            if size > 1 or axis in ("dp_replicate", "dp_shard"):
                spec[axis] = size

        mesh = Mesh(spec)
        mesh.flatten(("dp_replicate", "dp_shard"), "dp")
        return mesh

    def _shard_layout(self) -> ShardLayout:
        """how this run shards model/optimizer state, for the checkpointer."""
        return ShardLayout.of(self.model, self.parallel, self.mesh, tp=self.tp)

    def _writes_model_state(self) -> bool:
        """
        one rank per shard writes model/optimizer state: the one at coordinate
        0 on every axis the state is replicated (not sharded) over. that is
        rank 0 alone under ddp, the dp_replicate=0 row under fsdp/hsdp, and
        with tp on, one rank per tp coordinate of those.
        """
        shard_dims = self.parallel.shard_dims + (self.tp.shard_dims if self.tp else ())
        return all(
            self.mesh.coordinate(axis) == 0
            for axis in ("dp_replicate", "dp_shard", "tp")
            if axis in self.mesh.sizes and axis not in shard_dims
        )

    def _writes_dp_state(self) -> bool:
        """
        rng, dataloader and prefetcher state is one file per dp coordinate.
        tp peers share that coordinate and hold identical state, so only the
        first of them writes it.
        """
        return self.tp_size == 1 or self.mesh.coordinate("tp") == 0

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
        if self._writes_dp_state():
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
        first SIGTERM: checkpoint and exit at the next log point. a later one
        means "now": put the default handler back and die.

        one request often arrives twice within moments: a signal to the whole
        process group reaches this rank directly and again when torchrun
        forwards its own copy. repeats inside a short window count as one.

        dataloader workers ignore SIGTERM, so a signal sent to the whole
        process group still leaves the run able to reach that log point.
        """
        now = time.monotonic()
        if self._sigterm and now - self._sigterm_at > 5.0:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)
        if not self._sigterm:
            self._sigterm_at = now
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
        if self.tp_size > 1:
            all_reduce(flag, self.mesh, "tp", "max")
        return flag.item() > 0

    def _resume_step(self) -> int | None:
        """
        the checkpoint step to resume from, per `checkpointing.resume`: null
        (start fresh), "auto" (the latest complete step, if any) or an
        explicit step. None means start fresh.
        """
        if self.resume is None:
            return None
        if self.resume == "auto":
            return self.checkpointer.latest()
        if isinstance(self.resume, int):
            step = self.checkpointer.valid_step(self.resume)
            assert step is not None, (
                f"checkpointing.resume={self.resume}: no complete checkpoint at that "
                f"step in {self.checkpointer_path}"
            )
            return step
        raise ValueError(
            f"checkpointing.resume={self.resume!r}: expected null, 'auto' or a training step"
        )

    def _resume(self) -> int:
        """restore everything from a checkpoint; returns the step to start at."""
        step = self._resume_step()
        if step is None:
            return 0

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
        self._resume_optimizer(step, layout)

        dp_rank = self.mesh.coordinate("dp")
        if self.resume_dataloader:
            self.checkpointer.load_dataloader(self.dataloader, step, dp_rank)
            # older checkpoints have none: they resume one batch ahead
            held = self.checkpointer.load_prefetcher(step, dp_rank)
            if held is not None:
                self._resumed_batch = held["next_batch"]
        # each dp rank resumes its own stream. older checkpoints carry only
        # rank 0's, in trainer.pt. a rank with neither (resume onto a larger
        # mesh) keeps the seed-derived stream it already has.
        rng = self.checkpointer.load_rng(step, dp_rank)
        if rng is None and "cpu_rng" in state:
            rng = state
        if rng is not None:
            torch.set_rng_state(rng["cpu_rng"])
            torch.cuda.set_rng_state(rng["cuda_rng"], self.device)
        # .get: older checkpoints have no wandb_id
        self.resume_wandb_id = state.get("wandb_id")
        return state["step"]

    def _resume_optimizer(self, step: int, layout: ShardLayout) -> None:
        """
        optimizer and scheduler state. per-param state (adam moments) always
        comes from the checkpoint and hparams from this config, except lr,
        which belongs to the scheduler when that is resumed too.
        """
        # load_state_dict overwrites param_groups wholesale, so save this
        # config's hparams and put them back
        config_hparams = [
            {k: v for k, v in group.items() if k != "params"}
            for group in self.optimizer.param_groups
        ]
        self.checkpointer.load_optimizer(self.optimizer, step, layout)
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

    def _grad_shard_dims(self, param: nn.Parameter) -> tuple[str, ...]:
        """the mesh axes `param`'s grad is sharded over (for the global grad norm)."""
        dims = self.parallel.shard_dims
        return dims if self.tp is None else dims + self.tp.shard_dims_of(param)

    def micro_train_step(self) -> torch.Tensor:
        batch = next(self.prefetcher)
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
        if self.tp is not None:
            # after the dp sync: it reduces these same grad tensors in the background
            self.tp.sync()
        grad_norm = clip_grad_norm(
            self.model.parameters(), self.mesh, self.max_norm,
            shard_dims_of=self._grad_shard_dims,
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
                    print(
                        f"step={done} || loss={loss:.4f} || gnorm={gnorm:.3f} "
                        f"|| lr={self.scheduler.get_last_lr()[0]:.2e} "
                        f"|| tps={tokens_per_step / step_time:.0f}"
                    )
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
