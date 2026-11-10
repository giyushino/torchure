"""
checkpointing: checkpoints/<run>/<step>/ with one torch.save file per
component. load_* applies state in place onto a freshly built trainer.

model and optimizer state may be sharded (fsdp). a `ShardLayout` says how THIS
run lays params out; with the default layout nothing is sharded and the files
are the plain `model.pt` / `optimizer.pt` this module has always written.

sharded save (num_shards = N > 1): the rank at shard coordinate c writes
`model_shard{c}.pt` / `optimizer_shard{c}.pt`, its local state_dict as is.
coordinate 0 also writes `{model,optimizer}_shards.json`, recording N and
which tensors are split along which dim. everything not listed there is
replicated and read from shard 0.

uneven shards are stored as fsdp holds them, zero-padded to equal length; the
metadata records each sharded tensor's true size so load can trim the padding
and re-pad for the current shard count.

load reshards: a checkpoint saved at any N (including the unsharded N=1) loads
onto any current layout. files are opened with mmap, and each rank reads only
the byte range of each tensor it owns, so no rank ever materializes the full
model to take its slice.

optimizer state follows its param: a state tensor shaped like the param
(adam's exp_avg / exp_avg_sq) is sharded the same way; anything else (the
`step` counter) is replicated.

completion: `commit(step)` writes this rank's `.done{rank}of{world}` marker
after its last file. a step is complete once every rank's marker is there, so
no collective fences the save (a legacy `.complete` file still counts).

async (`async_save=True`): save_* snapshot state to cpu and return; commit()
hands the snapshot to a background thread that writes the files and then the
marker, while training continues. at most one save is in flight: commit() and
wait() block on the previous one and re-raise anything it failed with. call
wait() before exiting.
"""
import copy
import json
import os
import re
import threading
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from torchure.core.placement import padded_chunk


@dataclass(frozen=True)
class ShardLayout:
    """
    how the current run shards params: `dims` maps id(param) -> the tensor dim
    it is split along and `sizes` -> its true unsharded length along that dim
    (a local shard may be padded), for sharded params only. this rank holds
    chunk `coord` of `num_shards`. the default is the unsharded layout.
    """

    num_shards: int = 1
    coord: int = 0
    dims: dict[int, int] = field(default_factory=dict)
    sizes: dict[int, int] = field(default_factory=dict)

    @classmethod
    def of(cls, model: nn.Module, parallel, mesh) -> "ShardLayout":
        """
        the layout a parallelism strategy (DDP / FSDP: anything exposing
        `shard_dims` and `shard_dim_of`) gives `model` on this rank.
        """
        if not parallel.shard_dims:
            return cls()
        assert len(parallel.shard_dims) == 1, (
            f"one shard axis supported, got {parallel.shard_dims}"
        )
        axis = parallel.shard_dims[0]
        dims, sizes = {}, {}
        for param in model.parameters():
            dim = parallel.shard_dim_of(param)
            if dim is not None:
                dims[id(param)] = dim
                sizes[id(param)] = parallel.full_shape_of(param)[dim]
        return cls(mesh.size(axis), mesh.coordinate(axis), dims, sizes)


UNSHARDED = ShardLayout()


def _read(
    pieces: list[torch.Tensor],
    saved_dim: int | None,
    saved_size: int | None,
    want_dim: int | None,
    layout: ShardLayout,
) -> torch.Tensor:
    """
    one tensor of the current layout, from its saved pieces.

    `pieces` concatenate along `saved_dim` into the full tensor (a single
    piece when it was saved replicated), followed by any shard padding past
    `saved_size`. returns the full tensor when `want_dim` is None, else this
    rank's chunk along `want_dim`, zero-padded to the common chunk length.
    always an owning copy: the inputs are mmap views, and a view handed to
    load_state_dict's deepcopy would drag its whole backing storage along.
    """
    if saved_dim is None:
        pieces, saved_size = pieces[:1], None

    if want_dim is None or (len(pieces) > 1 and saved_dim != want_dim):
        # the whole tensor: wanted as is, or saved along one dim and wanted
        # along another, which leaves no range to pick out
        full = pieces[0] if len(pieces) == 1 else torch.cat(pieces, saved_dim)
        if saved_size is not None:
            full = full.narrow(saved_dim, 0, saved_size)
        if want_dim is None:
            return full.clone()
        pieces, saved_size = [full], None

    total = saved_size
    if total is None:
        total = sum(p.shape[want_dim] for p in pieces)
    length, start, valid = padded_chunk(total, layout.num_shards, layout.coord)
    end = start + valid

    out, offset = [], 0
    for piece in pieces:
        n = piece.shape[want_dim]
        lo, hi = max(start, offset), min(end, offset + n)
        if lo < hi:
            out.append(piece.narrow(want_dim, lo - offset, hi - lo))
        offset += n
    if valid < length:
        shape = list(pieces[0].shape)
        shape[want_dim] = length - valid
        out.append(pieces[0].new_zeros(shape))
    return out[0].clone() if len(out) == 1 else torch.cat(out, want_dim)


def _snapshot(obj):
    """a cpu copy of `obj` that later training steps can't mutate."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _snapshot(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_snapshot(v) for v in obj)
    return copy.deepcopy(obj)


_DONE = re.compile(r"^\.done(\d+)of(\d+)$")


class Checkpointer:
    def __init__(
        self,
        checkpoint_save_path: str,
        rank: int = 0,
        world_size: int = 1,
        async_save: bool = False,
    ):
        self.checkpoint_save_path = checkpoint_save_path
        self.rank = rank
        self.world_size = world_size
        self.async_save = async_save
        os.makedirs(self.checkpoint_save_path, exist_ok=True)
        self._begun: int | None = None          # step whose files are being written
        self._queue: list[tuple[str, object]] = []   # async: (path, snapshot) not yet written
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
    
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

    def _put(self, state, step: int, name: str) -> None:
        """write `state` to <step>/<name>: now, or at commit() when async."""
        step_dir = self._step_dir(step, create=True)
        if self._begun != step:
            # re-saving over an old step dir: drop this rank's markers first,
            # so the dir can't read as complete while its files are rewritten
            self._begun = step
            for entry in os.listdir(step_dir):
                m = _DONE.match(entry)
                if (m and int(m.group(1)) == self.rank) or (
                    entry == ".complete" and self.rank == 0
                ):
                    os.remove(os.path.join(step_dir, entry))
        path = os.path.join(step_dir, name)
        if self.async_save:
            self._queue.append((path, _snapshot(state)))
        else:
            torch.save(state, path)

    def _save(self, obj, step: int, name: str) -> None:
        self._put(obj.state_dict(), step, name)

    def commit(self, step: int) -> None:
        """
        this rank is done saving `step`: mark it, after every file is on disk.
        async, this starts the background write and returns.
        """
        marker = os.path.join(
            self._step_dir(step, create=True), f".done{self.rank}of{self.world_size}"
        )
        self._begun = None
        if not self.async_save:
            open(marker, "w").close()
            return
        self.wait()  # back-pressure: one save in flight
        jobs, self._queue = self._queue, []

        def write() -> None:
            try:
                for path, state in jobs:
                    torch.save(state, path)
                open(marker, "w").close()
            except BaseException as e:  # noqa: BLE001
                self._error = e

        # not a daemon: a save that has started finishes even if training dies
        self._thread = threading.Thread(target=write, name=f"ckpt-{step}")
        self._thread.start()

    def wait(self) -> None:
        """block until the in-flight async save is on disk; raise if it failed."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError("async checkpoint save failed") from error

    def _load(self, step: int, name: str, weights_only: bool = True, mmap: bool = False):
        # cpu: load_state_dict copies into the existing tensors anyway
        return torch.load(
            os.path.join(self._step_dir(step), name),
            map_location="cpu",
            weights_only=weights_only,
            mmap=mmap,
        )

    def _save_shard(
        self, state: dict, step: int, stem: str, layout: ShardLayout, dims: dict, sizes: dict
    ) -> None:
        """this rank's shard of `state`, plus the metadata if it is shard 0."""
        step_dir = self._step_dir(step, create=True)
        if layout.num_shards == 1:
            self._put(state, step, f"{stem}.pt")
            return
        self._put(state, step, f"{stem}_shard{layout.coord}.pt")
        if layout.coord == 0:
            # tiny, so written inline even when async; the marker gates it
            meta = {"num_shards": layout.num_shards, "dims": dims, "sizes": sizes}
            with open(os.path.join(step_dir, f"{stem}_shards.json"), "w") as f:
                json.dump(meta, f, indent=1)

    def _load_shards(self, step: int, stem: str) -> tuple[list[dict], dict, dict]:
        """every shard's saved state (mmapped), and the saved dims and sizes."""
        meta_path = os.path.join(self._step_dir(step), f"{stem}_shards.json")
        if not os.path.exists(meta_path):
            return [self._load(step, f"{stem}.pt", mmap=True)], {}, {}
        with open(meta_path) as f:
            meta = json.load(f)
        shards = [
            self._load(step, f"{stem}_shard{k}.pt", mmap=True)
            for k in range(meta["num_shards"])
        ]
        # no sizes: written before padding existed, so nothing is padded
        return shards, meta["dims"], meta.get("sizes", {})

    def save_model(self, model: nn.Module, step: int, layout: ShardLayout = UNSHARDED) -> None:
        # remove_duplicate=False: a tied weight is two state_dict keys
        named = [
            (name, p)
            for name, p in model.named_parameters(remove_duplicate=False)
            if id(p) in layout.dims
        ]
        dims = {name: layout.dims[id(p)] for name, p in named}
        sizes = {name: layout.sizes[id(p)] for name, p in named}
        self._save_shard(model.state_dict(), step, "model", layout, dims, sizes)

    def save_optimizer(
        self, optimizer: torch.optim.Optimizer, step: int, layout: ShardLayout = UNSHARDED
    ) -> None:
        state = optimizer.state_dict()
        params = [p for group in optimizer.param_groups for p in group["params"]]
        dims: dict[str, dict[str, int]] = {}
        sizes: dict[str, int] = {}
        for idx, param_state in state["state"].items():
            param = params[idx]
            if id(param) not in layout.dims:
                continue
            followers = {
                key: layout.dims[id(param)]
                for key, value in param_state.items()
                if torch.is_tensor(value) and value.shape == param.shape
            }
            if followers:
                dims[str(idx)] = followers
                sizes[str(idx)] = layout.sizes[id(param)]
        self._save_shard(state, step, "optimizer", layout, dims, sizes)

    def save_scheduler(self, scheduler: torch.optim.lr_scheduler.LRScheduler, step: int) -> None:
        self._save(scheduler, step, "scheduler.pt")

    def save_dataloader(self, dataloader, step: int, dp_rank: int = 0) -> None:
        self._save(dataloader, step, f"dataloader_dp{dp_rank}.pt")

    def save_trainer(self, state: dict, step: int) -> None:
        # already a plain dict, so it bypasses _save
        self._put(state, step, "trainer.pt")

    def save_rng(self, state: dict, step: int, dp_rank: int = 0) -> None:
        self._put(state, step, f"rng_dp{dp_rank}.pt")

    def save_prefetcher(self, state: dict, step: int, dp_rank: int = 0) -> None:
        self._put(state, step, f"prefetcher_dp{dp_rank}.pt")

    def load_prefetcher(self, step: int, dp_rank: int = 0) -> dict | None:
        """the batch this dp rank had prefetched but not consumed, if saved."""
        name = f"prefetcher_dp{dp_rank}.pt"
        if not os.path.exists(os.path.join(self._step_dir(step), name)):
            return None
        return self._load(step, name)

    def load_rng(self, step: int, dp_rank: int = 0) -> dict | None:
        """this dp rank's rng state, or None if the checkpoint has none for it."""
        name = f"rng_dp{dp_rank}.pt"
        if not os.path.exists(os.path.join(self._step_dir(step), name)):
            return None
        return self._load(step, name)

    def load_trainer(self, step: int) -> dict:
        return self._load(step, "trainer.pt")

    def load_model(self, model: nn.Module, step: int, layout: ShardLayout = UNSHARDED) -> None:
        shards, saved_dims, saved_sizes = self._load_shards(step, "model")
        want_dims = {
            name: layout.dims[id(p)]
            for name, p in model.named_parameters(remove_duplicate=False)
            if id(p) in layout.dims
        }
        state = {
            name: _read(
                [shard[name] for shard in shards],
                saved_dims.get(name),
                saved_sizes.get(name),
                want_dims.get(name),
                layout,
            )
            for name in shards[0]
        }
        model.load_state_dict(state)

    def load_optimizer(
        self, optimizer: torch.optim.Optimizer, step: int, layout: ShardLayout = UNSHARDED
    ) -> None:
        shards, saved_dims, saved_sizes = self._load_shards(step, "optimizer")
        params = [p for group in optimizer.param_groups for p in group["params"]]
        state: dict = {}
        for idx, saved in shards[0]["state"].items():
            param = params[idx]
            want_dim = layout.dims.get(id(param))
            # the param's unsharded shape, to spot state that follows it when
            # the checkpoint itself was saved unsharded
            full_shape = list(param.shape)
            if want_dim is not None:
                full_shape[want_dim] = layout.sizes[id(param)]
            state[idx] = {}
            for key, value in saved.items():
                if not torch.is_tensor(value):
                    state[idx][key] = value
                    continue
                saved_dim = saved_dims.get(str(idx), {}).get(key)
                follows = saved_dim is not None or list(value.shape) == full_shape
                state[idx][key] = _read(
                    [shard["state"][idx][key] for shard in shards],
                    saved_dim,
                    saved_sizes.get(str(idx)),
                    want_dim if follows else None,
                    layout,
                )
        optimizer.load_state_dict(
            {"state": state, "param_groups": shards[0]["param_groups"]}
        )

    def load_scheduler(self, scheduler: torch.optim.lr_scheduler.LRScheduler, step: int) -> None:
        scheduler.load_state_dict(self._load(step, "scheduler.pt"))

    def load_dataloader(self, dataloader, step: int, dp_rank: int = 0) -> None:
        # weights_only rejects the hf dataset state; we wrote this file ourselves
        dataloader.load_state_dict(
            self._load(step, f"dataloader_dp{dp_rank}.pt", weights_only=False)
        )

    def is_complete(self, step: int) -> bool:
        """every rank of the run that wrote `step` has committed it."""
        step_dir = self._step_dir(step)
        if not os.path.isdir(step_dir):
            return False
        entries = os.listdir(step_dir)
        if ".complete" in entries:  # legacy single marker
            return True
        done: dict[int, set[int]] = {}
        for entry in entries:
            m = _DONE.match(entry)
            if m:
                done.setdefault(int(m.group(2)), set()).add(int(m.group(1)))
        return any(ranks == set(range(world)) for world, ranks in done.items())
