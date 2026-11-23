"""
training orchestrator: build -> init -> parallelize -> compile -> optimizer -> data

TODO: replace @debug_time with a real profiler wrapper
"""

import atexit
import os
import signal
import time
from datetime import datetime

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from torchure.checkpoint.checkpointer import Checkpointer, ShardLayout
from torchure.core.collective import all_reduce
from torchure.core.grad_helper import clip_grad_norm
from torchure.config import load_config, validate
from torchure.core.mesh import Mesh, build_mesh
from torchure.core.meta import materialize
from torchure.dataloader.builder import build_dataloader
from torchure.dataloader.mixture import MixtureLoader
from torchure.dataloader.prefetcher import CUDAPrefetcher
from torchure.models.builder import build_model
from torchure.objectives.builder import build_objective
from torchure.optimizer.builder import build_optimizer, build_scheduler
from torchure.parallelism.data_parallel import DDP
from torchure.parallelism.expert_parallel import EP
from torchure.parallelism.fsdp2 import FSDP
from torchure.parallelism.tensor_parallel import TP
from torchure.utils import checkpoint_dir, debug_time

try:
    import wandb
except ImportError:
    wandb = None

# i think we can remove this soon, forgot what version
# of pytorch this is no longer needed for
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# tf32 for the fp32 matmuls bf16 autocast leaves alone (rope, some reductions)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# config dtype names (fsdp param / reduce dtype, ddp reduce dtype)
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    None: None,
    "none": None,
}

class Trainer:
    def __init__(
        self,
        train_config_path: str,
        rank: int,
        local_rank: int,
        world_size: int,
        benchmark: bool = False,
    ):
        self.config = load_config(train_config_path)
        validate(self.config, world_size)
        self.rank = rank
        self.world_size = world_size
        self.seed = self.config["seed"]
        torch.manual_seed(self.seed)

        self.benchmark = benchmark
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        self.mesh = self._build_mesh()
        self.dp_size = self.mesh.size("dp")
        # the mesh drops size-1 axes, default to 1
        self.tp_size = self.mesh.sizes.get("tp", 1)
        self.ep_size = self.mesh.sizes.get("ep", 1)

        # "ddp" | "fsdp"
        self.parallelism = self.config["parallelism"]
        # set by _build_model
        self.parallel: DDP | FSDP
        self.tp: TP | None = None
        self.ep: EP | None = None

        self.ignore_index = self.config["objective"]["config"]["ignore_index"]
        self.tokenizer = Tokenizer.from_pretrained(self.config["data"]["tokenizer"])

        self.model = self._build_model()
        self.model.activation_checkpointing = self.config["activation_checkpointing"]
        self.eos_id = self.tokenizer.token_to_id(self.config["data"]["eos_token"])
        if self.config["doc_mask"]:
            assert self.model.supports_doc_mask, (
                f"doc_mask: model {self.config['model']['name']!r} doesn't support document masking"
            )
            # layers.doc_mask is causal, and finds documents by eos in the
            # (possibly noised) input ids
            assert getattr(self.model, "causal", True), (
                "doc_mask is causal-only; it can't be combined with causal: false"
            )
            self.model.doc_mask_eos = self.eos_id
        # each data rank has own random stream, useful for random
        # dropout and masking. tp peers share dp coord
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
        assert self.dataloader.eos_id == self.eos_id, (
            f"data sources end docs with token {self.dataloader.eos_id}, but "
            f"data.eos_token is {self.eos_id}"
        )
        self.global_batch_size = self.config["data"]["global_batch_size"]
        self.micro_batch_size = self.config["data"]["micro_batch_size"]

        # same on every rank, so every rank issues the same collectives
        seqs_per_backward = self.micro_batch_size * self.dp_size
        assert self.global_batch_size % seqs_per_backward == 0, (
            f"global_batch_size {self.global_batch_size} must be divisible by "
            f"micro_batch_size {self.micro_batch_size} x dp {self.dp_size} "
            f"= {seqs_per_backward}"
        )
        self.microbatches_per_step = self.global_batch_size // seqs_per_backward

        # only the wandb name is timestamped: the checkpoint dir must be the
        # same on every rank and every relaunch, or resume="auto" can't find it
        time_launched = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = f"{self.config['run_name']}_{time_launched}"
        self.checkpointer_path = checkpoint_dir(self.config)
        self.checkpointer = Checkpointer(
            self.checkpointer_path,
            rank=rank,
            world_size=world_size,
            # snapshot to cpu, write in a background thread while training runs
            async_save=self.config["checkpointing"]["async_save"],
        )
        self.resume = self.config["checkpointing"]["resume"]
        self.save_steps = self.config["checkpointing"]["save_steps"]
        # complete checkpoints to keep besides the one being written; None = all
        self.keep_last_n = self.config["checkpointing"]["keep_last_n"]
        # the step loop syncs with the host only at log points and SIGTERM
        # checks; the checks default to the log points
        self.log_every = self.config["log_every"]
        self.stop_check_every = (
            self.config["checkpointing"]["stop_check_every"] or self.log_every
        )
        # false to resume under a new schedule: the saved scheduler state would
        # bring back the old total_steps / warmup / decay
        self.resume_scheduler = self.config["checkpointing"]["resume_scheduler"]
        # restoring the stream position replays each worker's current shard up
        # to where it was, which can be slow; false reshuffles instead
        self.resume_dataloader = self.config["checkpointing"]["resume_dataloader"]
        # set by _resume so wandb reattaches to the existing run
        self.resume_wandb_id: str | None = None
        # the batch the prefetcher held at the checkpoint, set by _resume
        self._resumed_batch: dict[str, torch.Tensor] | None = None
        self.start_step = self._resume()
        self._setup_wandb()

        # must come after _resume restores the dataloader state. the loader
        # never runs out: each source starts its next epoch itself
        atexit.register(self.dataloader.close)
        self.prefetcher = CUDAPrefetcher(
            self.dataloader, self.device, first_batch=self._resumed_batch
        )

        # set by the SIGTERM handler train() installs; see _stop_requested
        self._sigterm = False
        self._sigterm_at = 0.0

    def _setup_wandb(self) -> None:
        """rank 0 only. runs after _resume so it can reattach to the checkpointed run id."""
        self.wandb_run = None
        wandb_cfg = self.config["wandb"]
        if not wandb_cfg["enabled"] or self.rank != 0 or self.benchmark:
            return
        assert wandb is not None, (
            "wandb.enabled is true in the config but wandb isn't installed: "
            "uv sync --extra wandb"
        )

        self.wandb_run = wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg["entity"],
            name=self.run_name,
            id=self.resume_wandb_id,
            resume="must" if self.resume_wandb_id else None,
            config=self.config,
        )

    def _log_metrics(
        self, loss: float, grad_norm: float, step: int, tokens_per_step: int, step_time: float,
        extra: dict[str, float],
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
                # the schedule's mix at this step
                **({f"mix/{name}": w for name, w in self.dataloader.weights(step).items()}
                   if len(self.dataloader.names) > 1 else {}),
                **extra,
            },
            step=step,
        )
    
    @debug_time
    def _build_model(self) -> nn.Module:
        """
        build, initialize and parallelize the model. sets self.parallel,
        self.tp and self.ep.

        plain ddp: build -> init -> replicate.
        anything sharded (fsdp, tp, ep): build on the meta device (shapes only)
        -> shard -> init the shards, so no rank ever holds the full weights. tp
        and ep cut first; then fsdp shards what's left on the rank, or ddp
        replicates it (bad for heterogeneous).
        """
        model_cfg = self.config["model"]
        # plain ddp
        if self.parallelism != "fsdp" and self.tp_size == 1 and self.ep_size == 1:
            model = build_model(model_cfg["name"], model_cfg["config"])
            model.to(self.device)
            model.init_weights()
            self._parallelize(model)
            return model

        with torch.device("meta"):
            model = build_model(model_cfg["name"], model_cfg["config"])
        self.tp = self._tensor_parallelize(model)
        self.ep = self._expert_parallelize(model)
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
        attach DDP or FSDP as `self.parallel`. both expose `requires_sync`,
        `sync()` and `shard_dims`, so train_step doesn't branch on which.

        runs before _compile, so the grad hooks stay outside compiled code.
        """
        if self.parallelism == "ddp":
            self.parallel = DDP(
                model,
                self.mesh,
                dim="dp",
                bucket_mb=self.config["ddp"]["bucket_mb"],
                reduce_dtype=_DTYPES[self.config["ddp"]["reduce_dtype"]],
                # under ep, `dp` includes ep; expert weights average over
                # dp_expert instead (see _expert_axis)
                dim_of=self._expert_axis("dp", "dp_expert"),
            )
        elif self.parallelism == "fsdp":
            # at dp_shard=1 fsdp does nothing, including the dp_replicate
            # sync; pure replication is ddp's job
            assert self.mesh.size("dp_shard") > 1, (
                f"parallelism 'fsdp' needs dp_shard > 1 (mesh {self.config['mesh']}); "
                "use parallelism 'ddp' for pure replication"
            )
            fsdp_cfg = self.config["fsdp"]
            # the axis holding copies of the same shard: dp_replicate, plus ep
            # for dense params (ep ranks each hold a full copy of those)
            replicate_dim = "dp_replicate_ep" if self.ep is not None else "dp_replicate"
            self.parallel = FSDP(
                model,
                self.mesh,
                # one unit per block, same granularity as _compile
                units=model.blocks,
                shard_dim="dp_shard",
                # hsdp: reduce-scatter on dp_shard, then all_reduce on dp_replicate
                replicate_dim=replicate_dim if self.mesh.size(replicate_dim) > 1 else None,
                replicate_dim_of=self._expert_axis(replicate_dim, "dp_replicate"),
                param_dtype=_DTYPES[fsdp_cfg["param_dtype"]],
                reduce_dtype=_DTYPES[fsdp_cfg["reduce_dtype"]],
                prefetch=fsdp_cfg["prefetch"],
                reshard_after_forward=fsdp_cfg["reshard_after_forward"],
                device=self.device,
            )
        else:
            raise ValueError(
                f"unknown parallelism {self.parallelism!r}; expected 'ddp' or 'fsdp'"
            )

    def _expert_axis(self, dense: str, expert: str):
        """
        a `param -> axis` function telling ddp / fsdp which axis to average each
        grad over: `expert` (the dp axes without ep) for an expert weight, which
        lives on one ep rank, and `dense` for everything else. returns None
        without ep, where every param uses the same axis.
        """
        if self.ep is None:
            return None
        ep = self.ep
        return lambda param: expert if ep.owns(param) else dense

    def _expert_parallelize(self, model: nn.Module) -> EP | None:
        """
        split the model's expert banks along the mesh's ep axis. the model is
        still on meta here, so this only rewrites shapes.
        """
        if self.ep_size == 1:
            return None
        banks = model.ep_plan()
        assert banks, (
            f"mesh.ep={self.ep_size} but {type(model).__name__} has no expert banks: "
            "expert parallel needs a moe model"
        )
        return EP(self.mesh, banks, dim="ep")

    def _tensor_parallelize(self, model: nn.Module) -> TP | None:
        """
        split the model's weights along the mesh's tp axis. the model says
        which modules split which way. the model is still on meta here, so
        this only rewrites shapes.

        config `tp`: `vocab_parallel` shards the embedding / lm_head by vocab
        rows; `sequence_parallel` shards activations between tp regions along
        `sp_dim`; `overlap` hides the sp collectives behind matmuls
        (parallelism/tp_overlap.py).
        """
        if self.tp_size == 1:
            return None
        tp_cfg = self.config["tp"]
        sequence_parallel = tp_cfg["sequence_parallel"]
        plan = model.tp_plan(self.tp_size, vocab_parallel=tp_cfg["vocab_parallel"])
        # no regions (kimi shards only the vocab): nothing for sp to shard between
        sequence_parallel = sequence_parallel and bool(plan["regions"])
        # "batch" keeps each rank's chunk contiguous (no copies around the
        # collectives) but needs micro_batch_size % tp == 0
        sp_dim = tp_cfg["sp_dim"]
        return TP(
            self.mesh, **plan, dim="tp", sequence_parallel=sequence_parallel,
            seq_dim=0 if sp_dim == "batch" else 1,
            overlap=tp_cfg["overlap"] if sequence_parallel else "none",
        )
    
    @debug_time
    def _compile(self, model: nn.Module) -> None:
        """
        compile each block's forward: identical blocks share one graph, and
        the boundaries match fsdp's units.
        """
        if not self.config["compile"]:
            return
        mode = self.config["compile_mode"]
        # `forward`, not the module: module.compile() would also trace fsdp's
        # hooks, which must stay eager
        for block in model.blocks:
            block.forward = torch.compile(block.forward, mode=mode)
        # a fused kernel for the norm's bf16 input x fp32 weight
        model.norm.compile(mode=mode)

    def _init_shards(self, model: nn.Module) -> None:
        """
        init a model materialized from meta. each rank fills its own shards,
        seeded per (dp_shard, tp, ep) cell so different shards get different
        values. the finish_init calls then copy everything meant to be
        replicated (hsdp replicas, uncut params, buffers) from one rank.
        """
        cell = self.mesh.coordinate("dp_shard") if self.parallelism == "fsdp" else 0
        cell = cell * self.tp_size + (self.mesh.coordinate("tp") if self.tp else 0)
        cell = cell * self.ep_size + (self.mesh.coordinate("ep") if self.ep else 0)
        torch.manual_seed(self.seed + cell)
        model.init_weights()
        if self.parallelism == "fsdp":
            self.parallel.finish_init()
        if self.tp is not None:
            self.tp.finish_init(model)
        if self.ep is not None:
            self.ep.finish_init(model)

    @debug_time
    def _build_dataloader(self) -> MixtureLoader:
        return build_dataloader(
            self.config["data"], self.tokenizer, self.ignore_index,
            self.mesh.coordinate("dp"), self.dp_size,
            total_steps=self.num_train_steps,
            # a vlm's image spec, for caption sources
            vision=(
                (self.model.image_size, self.model.num_image_tokens)
                if self.model.num_image_tokens is not None else None
            ),
        )
    
    @debug_time
    def _build_mesh(self) -> Mesh:
        """the config's mesh plus the derived data axis `dp` (core/mesh.py)."""
        return build_mesh(self.config["mesh"])

    def _shard_layout(self) -> ShardLayout:
        """how this run shards model/optimizer state, for the checkpointer."""
        return ShardLayout.of(self.model, self.parallel, self.mesh, tp=self.tp, ep=self.ep)

    def _writes_model_state(self) -> bool:
        """
        whether this rank writes model/optimizer state. one rank per distinct
        shard does: the one at coordinate 0 on every axis the state is
        replicated over. under ddp that is rank 0 alone; fsdp, tp and ep each
        add their shard axis.
        """
        shard_dims = (
            self.parallel.shard_dims
            + (self.tp.shard_dims if self.tp else ())
            + (self.ep.shard_dims if self.ep else ())
        )
        return all(
            self.mesh.coordinate(axis) == 0
            for axis in ("dp_replicate", "dp_shard", "tp", "ep")
            if axis in self.mesh.sizes and axis not in shard_dims
        )

    def _writes_dp_state(self) -> bool:
        """
        whether this rank writes rng, dataloader and prefetcher state, which is
        one file per dp coordinate. tp peers hold identical copies, so only tp
        coordinate 0 writes.
        """
        return self.tp_size == 1 or self.mesh.coordinate("tp") == 0

    @debug_time
    def checkpoint(self, step: int) -> None:
        # no barrier: each rank writes a done-marker after its files, and the
        # step is complete once every rank's marker exists
        if self.rank == 0 and self.keep_last_n is not None:
            # prune before saving: if this save dies, keep_last_n complete
            # checkpoints are still there
            self.checkpointer.prune(self.keep_last_n, keep=(step,))
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
            # the prefetcher holds one batch training hasn't seen yet; save it
            # with the current data pass
            held = self.prefetcher.state_dict()
            self.checkpointer.save_prefetcher(held, step, dp_rank)
        self.checkpointer.commit(step)

    def _on_sigterm(self, signum, frame) -> None:
        """
        first SIGTERM: checkpoint and exit at the next stop check. a second one
        more than 5s later: die now. the 5s window is there because one kill
        often arrives twice (directly, and forwarded by torchrun).
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
        has any rank been sent SIGTERM? a collective: ranks receive the signal
        at different times, so they vote here and all stop on the same step.
        costs a host sync, so it runs every `stop_check_every` steps.
        """
        flag = torch.tensor(float(self._sigterm), device=self.device)
        if self.dp_size > 1:
            all_reduce(flag, self.mesh, "dp", "max")
        if self.tp_size > 1:
            all_reduce(flag, self.mesh, "tp", "max")
        return flag.item() > 0

    def _resume_step(self) -> int | None:
        """
        the step to resume from per `checkpointing.resume` (null, "auto" or a
        step), or None to start fresh.
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
        # model/optimizer state reshards onto any mesh; the per-dp-rank
        # dataloader files don't
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
            # None in older checkpoints: they resume one batch ahead
            held = self.checkpointer.load_prefetcher(step, dp_rank)
            if held is not None:
                self._resumed_batch = held["next_batch"]
            self.checkpointer.load_dataloader(self.dataloader, step, dp_rank)
        else:
            # a fresh shuffle, not the start of the run's own order again
            # (which the same seed would replay). it can still overlap what
            # the run already saw
            self.dataloader.reseed(self.dataloader.seed + step)
        # each dp rank's own stream (older checkpoints: rank 0's, in
        # trainer.pt). a rank with none, after growing the mesh, keeps its seed
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
        per-param state (adam moments) comes from the checkpoint, hparams
        from this config. lr comes from the checkpoint when the scheduler is
        resumed, else from this config's schedule at `step`.
        """
        # load_state_dict replaces param_groups, hparams included; keep ours
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
            # fast-forward this config's schedule to `step` and apply its lr
            # now, or the first resumed step would use the step-0 lr
            self.scheduler.last_epoch = step
            lrs = self.scheduler.get_lr()
            for group, lr in zip(self.optimizer.param_groups, lrs, strict=True):
                group["lr"] = lr
            self.scheduler._last_lr = lrs

    def _grad_shard_dims(self, param: nn.Parameter) -> tuple[str, ...]:
        """the mesh axes `param`'s grad is sharded over (for the global grad norm)."""
        dims = self.parallel.shard_dims
        if self.tp is not None:
            dims = dims + self.tp.shard_dims_of(param)
        if self.ep is not None:
            dims = dims + self.ep.shard_dims_of(param)
        return dims

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
            # after parallel.sync(): ddp may still be all-reducing these grads
            self.tp.sync()
        if self.ep is not None:
            self.ep.sync()
        grad_norm = clip_grad_norm(
            self.model.parameters(), self.mesh, self.max_norm,
            shard_dims_of=self._grad_shard_dims,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # loss is this rank's; _run_steps averages it over dp at log points
        return {"loss": loss, "grad_norm": grad_norm, **self._balance_experts()}

    def _balance_experts(self) -> dict[str, torch.Tensor]:
        """
        moe router-bias balancing, once per optimizer step. expert loads are
        summed over all data ranks so every rank applies the same update.
        returns balance metrics on device ({} for a dense model).
        """
        def reduce(load: torch.Tensor) -> torch.Tensor:
            if self.dp_size > 1:
                all_reduce(load, self.mesh, "dp", "sum")
            return load

        return self.model.balance_step(reduce)

    def _run_steps(self, start: int, end: int, checkpoint: bool) -> list[tuple[int, int, float]]:
        """
        step loop for train() and the --steps benchmark. metrics stay on the
        gpu until a log point so the cpu can run ahead. step time is averaged
        over the log window and excludes checkpoint saves.

        returns the log windows as (last step, steps, seconds).
        """
        tokens_per_step = self.config["data"]["seq_len"] * self.global_batch_size
        loss_sum = torch.zeros((), device=self.device)
        gnorm_sum = torch.zeros((), device=self.device)
        # any other train_step metrics (moe balance)
        extra_sums: dict[str, torch.Tensor] = {}
        window = 0
        windows: list[tuple[int, int, float]] = []
        t0 = time.perf_counter()

        for step in range(start, end):
            metrics = self.train_step()
            loss_sum += metrics.pop("loss")
            gnorm_sum += metrics.pop("grad_norm")
            for name, value in metrics.items():
                extra_sums[name] = extra_sums.get(name, 0) + value
            window += 1

            done = step + 1
            save = checkpoint and (done % self.save_steps == 0 or done == end)
            log = done % self.log_every == 0 or done == end
            # a stop turns this step into a log + save point
            stop = False
            if checkpoint and (save or log or done % self.stop_check_every == 0):
                stop = self._stop_requested()
            if save or log or stop:
                save = save or stop
                if self.dp_size > 1:
                    all_reduce(loss_sum, self.mesh, "dp", "avg")
                loss = loss_sum.item() / window
                gnorm = gnorm_sum.item() / window
                extra = {name: total.item() / window for name, total in extra_sums.items()}
                elapsed = time.perf_counter() - t0
                windows.append((done, window, elapsed))
                step_time = elapsed / window
                if self.rank == 0:
                    self._log_metrics(loss, gnorm, done, tokens_per_step, step_time, extra)
                    print(
                        f"step={done} || loss={loss:.4f} || gnorm={gnorm:.3f} "
                        f"|| lr={self.scheduler.get_last_lr()[0]:.2e} "
                        f"|| tps={tokens_per_step / step_time:.0f}"
                        + "".join(f" || {name}={value:.3g}" for name, value in extra.items())
                    )
                loss_sum.zero_()
                gnorm_sum.zero_()
                extra_sums.clear()
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
        return windows

    @debug_time
    def train_n_step_test(self, n_steps: int, warmup_steps: int | None = None) -> None:
        """
        benchmark: n_steps without checkpointing, then the mean tok/s over
        the log windows that start after `warmup_steps` (compile, allocator
        and clock warmup; default min(20, n_steps // 2)). compare runs on
        this number, with the same warmup and step count, interleaved
        (notes/CHANGES.md).
        """
        if warmup_steps is None:
            warmup_steps = min(20, n_steps // 2)
        windows = self._run_steps(0, n_steps, checkpoint=False)
        steady = [(steps, secs) for done, steps, secs in windows if done - steps >= warmup_steps]
        if self.rank != 0:
            return
        if not steady:
            print(f"no log window starts after step {warmup_steps}: no steady-state tps")
            return
        steps = sum(s for s, _ in steady)
        tokens = self.config["data"]["seq_len"] * self.global_batch_size * steps
        print(
            f"steady tps (steps {n_steps - steps + 1}-{n_steps}): "
            f"{tokens / sum(t for _, t in steady):.0f}"
        )

    def train(self) -> None:
        signal.signal(signal.SIGTERM, self._on_sigterm)
        try:
            self._run_steps(self.start_step, self.num_train_steps, checkpoint=True)
        finally:
            # nothing checks the flag after the loop, so SIGTERM kills again
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            try:
                # the last async save must finish before exit
                self.checkpointer.wait()
            finally:
                # so a crash shows as crashed in wandb, not "running"
                if self.wandb_run is not None:
                    self.wandb_run.finish()
