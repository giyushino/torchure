"""
a data mixture over pre-tokenized sources, scheduled by a python script.

the mix script (config `data.mix`) is plain python:

    SOURCES = {"web": "hf://datasets/me/climbmix-qwen3", "code": "/nvme/stack-qwen3"}

    def weights(step: int, total_steps: int) -> dict[str, float]:
        return {"web": 0.8, "code": 0.2} if step < 0.9 * total_steps else {"web": 0.5, "code": 0.5}

weights are relative (normalized here); a source left out gets 0. the
function is called once per optimizer step and must be a pure function of
its arguments: every rank calls it independently and a resume calls it
again, so anything else (randomness, wall clock, files) desyncs ranks.

per step:
  1. weights(step) * global_batch_size is added to a per-source credit; the
     step's row counts are the credits rounded (largest remainder) so they
     sum to exactly global_batch_size. the rounding error carries over, so a
     source at weight 0.001 still gets its share over time instead of
     rounding to 0 every step.
  2. the global batch's source labels are shuffled with rng(seed, step), and
     rank r's micro batch m reads positions [(m*dp + r)*b, +b). every rank
     computes the whole (cheap) plan, so the ranks need no communication.
  3. each rank reads each source from its own stream (SourceStream): the
     source's blocks are permuted per epoch, dealt round-robin to the dp
     ranks (disjoint, so each rank downloads 1/dp of the data), and rows are
     shuffled within windows of `window_blocks` blocks. the next window
     downloads in the background while the current one is consumed.

state is a handful of integers per source plus the credits, so resume is
O(1): it refetches the current window and nothing else.

runs in the trainer's process: no tokenization, so there is nothing for
worker processes to parallelize, and process workers would each download
the same windows. fetching is threads.
"""

import importlib.util
import zlib
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

from torchure.dataloader.shards import TokenSource


def load_mix(path: str):
    """the mix script as a module, checked for SOURCES and weights()."""
    spec = importlib.util.spec_from_file_location("torchure_mix", path)
    assert spec is not None and spec.loader is not None, f"can't load mix script {path!r}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(getattr(module, "SOURCES", None), dict) and module.SOURCES, (
        f"{path}: needs SOURCES = {{name: location}}"
    )
    assert callable(getattr(module, "weights", None)), (
        f"{path}: needs def weights(step, total_steps) -> {{name: weight}}"
    )
    return module


def normalize(raw: dict, names: list[str], step: int) -> np.ndarray:
    """weights() output as a probability vector in `names` order."""
    unknown = set(raw) - set(names)
    assert not unknown, f"mix weights at step {step} name unknown sources {sorted(unknown)}"
    w = np.array([float(raw.get(name, 0.0)) for name in names], dtype=np.float64)
    assert np.all(np.isfinite(w)) and np.all(w >= 0), f"mix weights at step {step}: {raw}"
    assert w.sum() > 0, f"mix weights at step {step} are all zero"
    return w / w.sum()


def allocate(credit: np.ndarray, weights: np.ndarray, total: int) -> tuple[np.ndarray, np.ndarray]:
    """
    (non-negative counts summing to `total`, credit carried forward). the
    credit is what each source is owed (negative: it got ahead); it stays
    within about one row per source.
    """
    owed = credit + weights * total
    counts = np.maximum(np.floor(owed), 0).astype(np.int64)
    short = total - int(counts.sum())
    if short > 0:
        # most owed first; ties to the earlier source
        order = np.argsort(-(owed - counts), kind="stable")
        counts[order[:short]] += 1
    elif short < 0:
        # least owed first, among sources that have rows to give back
        order = [i for i in np.argsort(owed - counts, kind="stable") if counts[i] > 0]
        counts[order[:-short]] -= 1
    return counts, owed - counts


def _rng(*keys: int) -> np.random.Generator:
    return np.random.default_rng([int(k) for k in keys])


def _source_key(name: str) -> int:
    """stable per-name seed, so adding a source doesn't reshuffle the others."""
    return zlib.crc32(name.encode())


class SourceStream:
    """
    one rank's endless, shuffled row stream over one source. position is
    (epoch, window, pos): the window's rows are permuted, pos indexes that
    permutation.
    """

    def __init__(self, source: TokenSource, name: str, seed: int, dp_rank: int, dp_size: int,
                 window_blocks: int, pool: ThreadPoolExecutor):
        self.source = source
        self.key = _source_key(name)
        self.seed = seed
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.window_blocks = window_blocks
        self.pool = pool
        self.epoch = 0
        self.window = 0
        self.pos = 0
        # (epoch, window) -> futures of that window's blocks
        self._pending: dict[tuple[int, int], list[Future]] = {}
        self._rows: np.ndarray | None = None      # current window, (n, seq_len)
        self._order: np.ndarray | None = None     # permutation of its rows
        self._blocks_epoch: int | None = None
        self._blocks: list[int] = []

    def _my_blocks(self, epoch: int) -> list[int]:
        """this rank's blocks in epoch order: a per-epoch permutation, dealt round-robin."""
        if self._blocks_epoch != epoch:
            perm = _rng(self.seed, self.key, epoch).permutation(len(self.source.blocks))
            self._blocks = perm[self.dp_rank::self.dp_size].tolist()
            self._blocks_epoch = epoch
        return self._blocks

    def _num_windows(self, epoch: int) -> int:
        n = len(self._my_blocks(epoch))
        return -(-n // self.window_blocks)

    def _next(self, epoch: int, window: int) -> tuple[int, int]:
        window += 1
        if window >= self._num_windows(epoch):
            return epoch + 1, 0
        return epoch, window

    def _submit(self, epoch: int, window: int) -> None:
        if (epoch, window) in self._pending:
            return
        blocks = self._my_blocks(epoch)
        mine = blocks[window * self.window_blocks:(window + 1) * self.window_blocks]
        self._pending[(epoch, window)] = [
            self.pool.submit(self.source.fetch, self.source.blocks[b]) for b in mine
        ]

    def _load(self) -> None:
        """make the current window resident and start downloading the next."""
        self._submit(self.epoch, self.window)
        futures = self._pending.pop((self.epoch, self.window))
        self._rows = np.concatenate([f.result() for f in futures])
        self._order = _rng(
            self.seed, self.key, self.epoch, self.dp_rank, self.window
        ).permutation(len(self._rows))
        self._submit(*self._next(self.epoch, self.window))

    def take(self) -> np.ndarray:
        if self._rows is None:
            self._load()
        row = self._rows[self._order[self.pos]]
        self.pos += 1
        if self.pos == len(self._order):
            self.epoch, self.window = self._next(self.epoch, self.window)
            self.pos = 0
            self._rows = self._order = None
        return row

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "window": self.window, "pos": self.pos}

    def load_state_dict(self, state: dict) -> None:
        for futures in self._pending.values():
            for f in futures:
                f.cancel()
        self._pending.clear()
        self.epoch, self.window, self.pos = state["epoch"], state["window"], state["pos"]
        self._rows = self._order = None


class MixtureLoader:
    """
    yields this dp rank's micro batches ({"input_ids", "labels"}, int64,
    (micro_batch_size, seq_len)), microbatches_per_step of them per optimizer
    step, forever. state_dict / load_state_dict like a StatefulDataLoader.
    """

    def __init__(self, data_cfg: dict, dp_rank: int, dp_size: int, total_steps: int,
                 tokenizer: str | None = None, pin_memory: bool = True):
        self.mix_path = data_cfg["mix"]
        self.mix = load_mix(self.mix_path)
        with open(self.mix_path) as f:
            self.mix_text = f.read()
        self.names = list(self.mix.SOURCES)
        self.seq_len = data_cfg["seq_len"]
        self.global_batch = data_cfg["global_batch_size"]
        self.micro = data_cfg["micro_batch_size"]
        self.dp_rank, self.dp_size = dp_rank, dp_size
        assert self.global_batch % (self.micro * dp_size) == 0
        self.micro_per_step = self.global_batch // (self.micro * dp_size)
        self.total_steps = total_steps
        self.seed = data_cfg["seed"]
        self.pin_memory = pin_memory and torch.cuda.is_available()

        self.pool = ThreadPoolExecutor(
            max_workers=data_cfg["fetch_threads"], thread_name_prefix="mix-fetch"
        )
        self.sources = {
            name: TokenSource(location, self.seq_len, data_cfg["block_mb"], min_blocks=dp_size)
            for name, location in self.mix.SOURCES.items()
        }
        for name, src in self.sources.items():
            assert tokenizer is None or src.tokenizer == tokenizer, (
                f"source {name!r} was tokenized with {src.tokenizer!r}, "
                f"the config's tokenizer is {tokenizer!r}"
            )
        eos = {src.eos_id for src in self.sources.values()}
        assert len(eos) == 1, f"sources disagree on eos_id: {eos}"
        self.streams = {
            name: SourceStream(src, name, self.seed, dp_rank, dp_size,
                               data_cfg["window_blocks"], self.pool)
            for name, src in self.sources.items()
        }

        # micro batches this rank has yielded
        self.micro_index = 0
        # credit carried into `_next_step`, the next step to be planned
        self._credit = np.zeros(len(self.names))
        self._next_step = 0
        # the last planned step: its labels and the credit carried into it
        self._labels_step: int | None = None
        self._labels: np.ndarray | None = None
        self._labels_credit = self._credit

    def weights(self, step: int) -> dict[str, float]:
        """the normalized mix at `step`, for logging."""
        w = normalize(self.mix.weights(step, self.total_steps), self.names, step)
        return dict(zip(self.names, w.tolist(), strict=True))

    def _plan(self, step: int) -> np.ndarray:
        """source index of each of the step's global_batch_size rows."""
        if step == self._labels_step:
            return self._labels
        assert step == self._next_step, f"planning step {step}, expected {self._next_step}"
        w = normalize(self.mix.weights(step, self.total_steps), self.names, step)
        self._labels_credit = self._credit
        counts, self._credit = allocate(self._credit, w, self.global_batch)
        labels = np.repeat(np.arange(len(self.names)), counts)
        self._labels = _rng(self.seed, step).permutation(labels)
        self._labels_step = step
        self._next_step = step + 1
        return self._labels

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        step, m = divmod(self.micro_index, self.micro_per_step)
        labels = self._plan(step)
        start = (m * self.dp_size + self.dp_rank) * self.micro
        rows = [self.streams[self.names[s]].take() for s in labels[start:start + self.micro]]
        input_ids = torch.from_numpy(np.stack(rows).astype(np.int64))
        if self.pin_memory:
            input_ids = input_ids.pin_memory()
        self.micro_index += 1
        # labels unshifted, like the hf path's Collator
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    def _shape(self) -> dict:
        return {"dp_size": self.dp_size, "global_batch_size": self.global_batch,
                "micro_batch_size": self.micro, "seq_len": self.seq_len}

    def state_dict(self) -> dict:
        step = self.micro_index // self.micro_per_step
        # credit carried into `step`: mid-step it's already planned
        credit = self._labels_credit if step == self._labels_step else self._credit
        return {
            "micro_index": self.micro_index,
            "credit": dict(zip(self.names, credit.tolist(), strict=True)),
            "streams": {name: s.state_dict() for name, s in self.streams.items()},
            "mix_text": self.mix_text,
            **self._shape(),
        }

    def load_state_dict(self, state: dict) -> None:
        for key, value in self._shape().items():
            assert state[key] == value, (
                f"mixture state has {key}={state[key]}, this run has {value}"
            )
        if state["mix_text"] != self.mix_text:
            print(f"note: mix script {self.mix_path} changed since the checkpoint; "
                  "the new schedule applies from here")
        self.micro_index = state["micro_index"]
        # a source added since the checkpoint starts fresh; one removed is dropped
        self._credit = np.array([state["credit"].get(name, 0.0) for name in self.names])
        self._next_step = self.micro_index // self.micro_per_step
        self._labels_step = self._labels = None
        for name, stream in self.streams.items():
            if name in state["streams"]:
                stream.load_state_dict(state["streams"][name])

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
