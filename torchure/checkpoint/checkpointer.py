"""
checkpointing: checkpoints/<run>/<step>/ with one torch.save file per
component. load_* applies state in place onto a freshly built trainer.

model and optimizer state may be sharded (fsdp). a `ShardLayout` says how THIS
run lays params out; with the default layout nothing is sharded and the files
are the plain `model.pt` / `optimizer.pt` this module has always written.

sharded save: a layout has one or more shard AXES (fsdp's dp_shard, tensor
parallel's tp, or both), and the ranks form a grid over them. the rank at grid
cell k (row-major) writes `model_shard{k}.pt` / `optimizer_shard{k}.pt`, its
local state_dict as is. cell 0 also writes `{model,optimizer}_shards.json`:
the grid, each tensor's unsharded shape, and its `splits` -- which axes cut
it, along which tensor dim, outermost first. a tensor an axis doesn't cut is
replicated along that axis and read from coordinate 0.

every piece therefore owns a BOX of the full tensor. uneven shards are stored
as fsdp holds them, zero-padded to equal length; the recorded shape says
where the real rows end.

load reshards: a checkpoint saved on any grid (including the unsharded one)
loads onto any current layout, by copying the overlap of each saved box with
the box this rank wants. files are opened with mmap, so each rank reads only
the byte ranges it owns and no rank materializes the full model.

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
import itertools
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from torchure.core.placement import padded_chunk


Splits = tuple[tuple[int, int], ...]   # ((shard axis index, tensor dim), ...) outermost first


@dataclass(frozen=True)
class ShardLayout:
    """
    how the current run shards params. `counts` is the number of shards along
    each shard axis and `coords` this rank's cell. for sharded params only,
    `splits` maps id(param) -> the (axis, tensor dim) cuts applied to it,
    outermost first, and `shapes` -> its unsharded shape (a local shard may
    be padded). the default is the unsharded layout.
    """

    counts: tuple[int, ...] = ()
    coords: tuple[int, ...] = ()
    splits: dict[int, Splits] = field(default_factory=dict)
    shapes: dict[int, tuple[int, ...]] = field(default_factory=dict)

    @property
    def num_shards(self) -> int:
        return math.prod(self.counts)

    @property
    def index(self) -> int:
        """this rank's cell, row-major: which shard file it writes."""
        return _flat(self.coords, self.counts)

    @classmethod
    def of(cls, model: nn.Module, parallel, mesh, tp=None) -> "ShardLayout":
        """
        the layout `model` has on this rank under a data-parallel strategy
        (DDP / FSDP) and, optionally, tensor parallel on top. both expose
        `shard_dims`, `shard_dim_of` and `full_shape_of`. tp cuts first: fsdp
        shards whatever tp left on the rank.
        """
        sources = [s for s in (tp, parallel) if s is not None and s.shard_dims]
        if not sources:
            return cls()
        for source in sources:
            assert len(source.shard_dims) == 1, (
                f"one shard axis per strategy supported, got {source.shard_dims}"
            )
        axes = [source.shard_dims[0] for source in sources]
        splits, shapes = {}, {}
        for param in model.parameters():
            cuts, shape = [], None
            for axis, source in enumerate(sources):
                dim = source.shard_dim_of(param)
                if dim is None:
                    continue
                cuts.append((axis, dim))
                if shape is None:
                    # the outermost strategy that cuts it saw it whole
                    shape = tuple(source.full_shape_of(param))
            if cuts:
                splits[id(param)] = tuple(cuts)
                shapes[id(param)] = shape
        return cls(
            tuple(mesh.size(a) for a in axes),
            tuple(mesh.coordinate(a) for a in axes),
            splits,
            shapes,
        )


UNSHARDED = ShardLayout()


def _flat(coords, counts) -> int:
    index = 0
    for coord, count in zip(coords, counts, strict=True):
        index = index * count + coord
    return index


def _box(shape, splits, counts, coords) -> tuple[list[int], list[int], list[int]]:
    """
    the part of a tensor of `shape` held at grid cell `coords`: per tensor
    dim, (first index, number of real entries, stored length). the stored
    length exceeds the real one where the innermost cut pads.
    """
    lo, size, stored = [0] * len(shape), list(shape), list(shape)
    for axis, dim in splits:
        length, start, valid = padded_chunk(size[dim], counts[axis], coords[axis])
        lo[dim] += start
        size[dim] = valid
        stored[dim] = length
    return lo, size, stored


def _read(
    shards: list[dict],
    pick,
    shape: tuple[int, ...],
    saved_splits: Splits,
    saved_counts: tuple[int, ...],
    want_splits: Splits,
    layout: ShardLayout,
) -> torch.Tensor:
    """
    one tensor of the current layout, from its saved pieces. `pick(shard)`
    fetches the tensor from one saved shard file. returns this rank's box of
    it (the whole tensor when `want_splits` is empty), zero-padded to the
    stored length. always an owning copy: the inputs are mmap views, and a
    view handed to load_state_dict's deepcopy would drag its whole backing
    storage along.
    """
    want_lo, want_size, want_stored = _box(shape, want_splits, layout.counts, layout.coords)
    cut_axes = {axis for axis, _ in saved_splits}
    # along an axis that doesn't cut this tensor every cell holds the same
    # copy; read coordinate 0
    cells = itertools.product(
        *(range(n) if axis in cut_axes else (0,) for axis, n in enumerate(saved_counts))
    )
    out = None
    for cell in cells:
        lo, size, _ = _box(shape, saved_splits, saved_counts, cell)
        a = [max(x, y) for x, y in zip(lo, want_lo, strict=True)]
        b = [
            min(x + n, y + m)
            for x, n, y, m in zip(lo, size, want_lo, want_size, strict=True)
        ]
        if any(hi <= low for low, hi in zip(a, b, strict=True)):
            continue
        piece = pick(shards[_flat(cell, saved_counts)])
        src = piece[tuple(slice(low - x, hi - x) for low, hi, x in zip(a, b, lo, strict=True))]
        if list(src.shape) == want_stored:
            return src.clone()
        if out is None:
            out = piece.new_zeros(want_stored)
        out[tuple(slice(low - x, hi - x) for low, hi, x in zip(a, b, want_lo, strict=True))] = src
    if out is None:
        # this rank's box is all padding
        out = pick(shards[0]).new_zeros(want_stored)
    return out


def _legacy_meta(meta: dict, shards: list[dict], pick_of) -> dict:
    """
    the single-axis metadata written before layouts had more than one axis
    (`num_shards`, `dims`, optional `sizes`), in the current form.
    `pick_of(name, key)` fetches a tensor from a shard.
    """
    out = {"counts": [meta["num_shards"]], "splits": {}, "shapes": {}}
    if any(isinstance(v, dict) for v in meta["dims"].values()):
        out["followers"] = {}
    for name, dims in meta["dims"].items():
        keys = dims if isinstance(dims, dict) else {None: dims}
        dim = next(iter(keys.values()))
        sample = [pick_of(name, next(iter(keys)))(shard) for shard in shards]
        shape = list(sample[0].shape)
        # no size recorded: written before padding existed, so nothing is padded
        shape[dim] = meta.get("sizes", {}).get(name, sum(p.shape[dim] for p in sample))
        out["splits"][name] = [[0, dim]]
        out["shapes"][name] = shape
        if isinstance(dims, dict):
            out["followers"][name] = list(dims)
    return out


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

    def _save_shard(self, state: dict, step: int, stem: str, layout: ShardLayout, meta: dict) -> None:
        """this rank's shard of `state`, plus the metadata if it is cell 0."""
        step_dir = self._step_dir(step, create=True)
        if layout.num_shards == 1:
            self._put(state, step, f"{stem}.pt")
            return
        self._put(state, step, f"{stem}_shard{layout.index}.pt")
        if layout.index == 0:
            # tiny, so written inline even when async; the marker gates it
            meta = {"counts": list(layout.counts), **meta}
            with open(os.path.join(step_dir, f"{stem}_shards.json"), "w") as f:
                json.dump(meta, f, indent=1)

    def _load_shards(self, step: int, stem: str, pick_of) -> tuple[list[dict], dict]:
        """every shard's saved state (mmapped), and the saved metadata."""
        meta_path = os.path.join(self._step_dir(step), f"{stem}_shards.json")
        if not os.path.exists(meta_path):
            return [self._load(step, f"{stem}.pt", mmap=True)], {"counts": [], "splits": {}, "shapes": {}}
        with open(meta_path) as f:
            meta = json.load(f)
        count = meta["num_shards"] if "num_shards" in meta else math.prod(meta["counts"])
        shards = [self._load(step, f"{stem}_shard{k}.pt", mmap=True) for k in range(count)]
        if "num_shards" in meta:
            meta = _legacy_meta(meta, shards, pick_of)
        return shards, meta

    def save_model(self, model: nn.Module, step: int, layout: ShardLayout = UNSHARDED) -> None:
        # remove_duplicate=False: a tied weight is two state_dict keys
        named = [
            (name, p)
            for name, p in model.named_parameters(remove_duplicate=False)
            if id(p) in layout.splits
        ]
        meta = {
            "splits": {name: layout.splits[id(p)] for name, p in named},
            "shapes": {name: layout.shapes[id(p)] for name, p in named},
        }
        self._save_shard(model.state_dict(), step, "model", layout, meta)

    def save_optimizer(
        self, optimizer: torch.optim.Optimizer, step: int, layout: ShardLayout = UNSHARDED
    ) -> None:
        state = optimizer.state_dict()
        params = [p for group in optimizer.param_groups for p in group["params"]]
        meta: dict = {"splits": {}, "shapes": {}, "followers": {}}
        for idx, param_state in state["state"].items():
            param = params[idx]
            if id(param) not in layout.splits:
                continue
            followers = [
                key for key, value in param_state.items()
                if torch.is_tensor(value) and value.shape == param.shape
            ]
            if followers:
                meta["splits"][str(idx)] = layout.splits[id(param)]
                meta["shapes"][str(idx)] = layout.shapes[id(param)]
                meta["followers"][str(idx)] = followers
        self._save_shard(state, step, "optimizer", layout, meta)

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
        shards, meta = self._load_shards(
            step, "model", lambda name, key: (lambda shard: shard[name])
        )
        want = {
            name: layout.splits[id(p)]
            for name, p in model.named_parameters(remove_duplicate=False)
            if id(p) in layout.splits
        }
        state = {}
        for name, value in shards[0].items():
            state[name] = _read(
                shards,
                lambda shard, name=name: shard[name],
                tuple(meta["shapes"].get(name, value.shape)),
                tuple(map(tuple, meta["splits"].get(name, ()))),
                tuple(meta["counts"]),
                want.get(name, ()),
                layout,
            )
        model.load_state_dict(state)

    def load_optimizer(
        self, optimizer: torch.optim.Optimizer, step: int, layout: ShardLayout = UNSHARDED
    ) -> None:
        shards, meta = self._load_shards(
            step, "optimizer",
            lambda idx, key: (lambda shard: shard["state"][int(idx)][key]),
        )
        params = [p for group in optimizer.param_groups for p in group["params"]]
        state: dict = {}
        for idx, saved in shards[0]["state"].items():
            param = params[idx]
            want = layout.splits.get(id(param), ())
            # the param's unsharded shape, to spot state that follows it when
            # the checkpoint itself was saved unsharded
            full_shape = tuple(layout.shapes.get(id(param), param.shape))
            followers = meta.get("followers", {}).get(str(idx), ())
            state[idx] = {}
            for key, value in saved.items():
                if not torch.is_tensor(value):
                    state[idx][key] = value
                    continue
                if key in followers:
                    shape = tuple(meta["shapes"][str(idx)])
                    saved_splits = tuple(map(tuple, meta["splits"][str(idx)]))
                    follows = True
                else:
                    shape, saved_splits = tuple(value.shape), ()
                    follows = shape == full_shape
                state[idx][key] = _read(
                    shards,
                    lambda shard, idx=idx, key=key: shard["state"][idx][key],
                    shape,
                    saved_splits,
                    tuple(meta["counts"]),
                    want if follows else (),
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
