"""
the dataloader: a mixture of sources, scheduled by a python script. one
source is the special case `data.source`, a mix of one at weight 1.

sources (sources.py) are raw jsonl / parquet, tokenized on the fly, or
pre-tokenized token shards, each a local path, an http(s) url or
hf://datasets/<org>/<name>[/<subdir>]. the mix script (config `data.mix`)
is plain python:

    SOURCES = {
        "web": "hf://datasets/OptimalScale/ClimbMix",
        "edu": {"path": "hf://datasets/HuggingFaceFW/fineweb-edu", "files": "sample/10BT/*"},
        "code": "/nvme/stack-qwen3",   # token shards (tokenize_shards.py)
    }

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

no torch DataLoader. by default (`workers: 0`) blocks are fetched,
tokenized and decoded on `fetch_threads` threads in the trainer's process
(tokenizers and PIL release the gil for their heavy parts); the trainer's
thread only gathers rows into a batch. with `workers: N`, N processes each
fetch whole blocks and send back the rows: decoding images holds the gil
long enough to slow the training loop itself (notes/DATA_LOADING.md), text
tokenizing doesn't. either way a block is fetched once, by one rank.
"""

import importlib.util
import json
import multiprocessing
import os
import signal
import types
import zlib
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import torch

from torchure.dataloader.sources import RowConfig, Vision, open_source


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
    permutation. rows are dicts of arrays (input_ids, and labels and
    pixel_values for caption sources).
    """

    def __init__(self, source, name: str, seed: int, dp_rank: int, dp_size: int,
                 window_blocks: int, submit: Callable[[object], Future],
                 prefetch_windows: int = 1):
        self.source = source
        self.key = _source_key(name)
        self.seed = seed
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.window_blocks = window_blocks
        self.prefetch_windows = prefetch_windows
        # block -> future of source.fetch(block), on a thread or a worker process
        self.submit = submit
        self.epoch = 0
        self.window = 0
        self.pos = 0
        # (epoch, window) -> futures of that window's blocks
        self._pending: dict[tuple[int, int], list[Future]] = {}
        # current window: its blocks' rows, unconcatenated (a window is up to
        # hundreds of MB; copying it would stall the training thread)
        self._parts: list[dict[str, np.ndarray]] | None = None
        self._order: np.ndarray | None = None              # permutation of its rows
        self._where: np.ndarray | None = None              # window row -> (part, row)
        self._blocks: dict[int, list[int]] = {}            # epoch -> this rank's blocks

    def _my_blocks(self, epoch: int) -> list[int]:
        """this rank's blocks in epoch order: a per-epoch permutation, dealt round-robin."""
        if epoch not in self._blocks:
            perm = _rng(self.seed, self.key, epoch).permutation(len(self.source.blocks))
            self._blocks[epoch] = perm[self.dp_rank::self.dp_size].tolist()
            # prefetching crosses at most one epoch boundary
            for old in [e for e in self._blocks if e < epoch - 1]:
                del self._blocks[old]
        return self._blocks[epoch]

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
            self.submit(self.source.blocks[b]) for b in mine
        ]

    def _load(self) -> None:
        """
        make the current window resident and keep the next prefetch_windows
        downloading. a window can come out empty (raw text blocks shorter than
        seq_len); it's skipped, as if its rows had been consumed.
        """
        skipped = 0
        while True:
            self._submit(self.epoch, self.window)
            futures = self._pending.pop((self.epoch, self.window))
            parts = [f.result() for f in futures]
            ahead = (self.epoch, self.window)
            for _ in range(self.prefetch_windows):
                ahead = self._next(*ahead)
                self._submit(*ahead)
            n = sum(len(p["input_ids"]) for p in parts)
            if n:
                break
            skipped += 1
            assert skipped <= self._num_windows(self.epoch) + 1, (
                f"{self.source.location}: a whole epoch of this rank's blocks yielded "
                f"no rows of seq_len {self.source.seq_len}"
            )
            self.epoch, self.window = self._next(self.epoch, self.window)
        sizes = [len(p["input_ids"]) for p in parts]
        self._parts = parts
        self._where = np.stack([np.repeat(np.arange(len(parts)), sizes),
                                np.concatenate([np.arange(k) for k in sizes])], axis=1)
        self._order = _rng(
            self.seed, self.key, self.epoch, self.dp_rank, self.window
        ).permutation(n)

    def take(self) -> dict[str, np.ndarray]:
        if self._parts is None:
            self._load()
        part, i = self._where[self._order[self.pos]]
        row = {k: v[i] for k, v in self._parts[part].items()}
        self.pos += 1
        if self.pos == len(self._order):
            self.epoch, self.window = self._next(self.epoch, self.window)
            self.pos = 0
            self._parts = self._order = self._where = None
        return row

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "window": self.window, "pos": self.pos}

    def load_state_dict(self, state: dict) -> None:
        for futures in self._pending.values():
            for f in futures:
                f.cancel()
        self._pending.clear()
        self.epoch, self.window, self.pos = state["epoch"], state["window"], state["pos"]
        self._parts = self._order = self._where = None


# a worker process's sources, set once by _worker_init
_WORKER_SOURCES: dict | None = None


def _worker_init(sources: dict) -> None:
    global _WORKER_SOURCES
    # the trainer answers a job-wide SIGTERM by checkpointing at its next log
    # point and needs batches until then; MixtureLoader.close() kills us
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    # parallelism comes from the processes; a rayon pool per worker would be
    # hundreds of threads each
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_SOURCES = sources


def _worker_fetch(name: str, block) -> dict[str, np.ndarray]:
    return _WORKER_SOURCES[name].fetch(block)


def _single_source(step: int, total_steps: int) -> dict[str, float]:
    return {"data": 1.0}


class MixtureLoader:
    """
    yields this dp rank's micro batches, microbatches_per_step of them per
    optimizer step, forever: {"input_ids", "labels"} int64 (micro_batch_size,
    seq_len), plus uint8 "pixel_values" (micro_batch_size, 3, S, S) from
    caption sources (the model normalizes them). labels are unshifted; shifting is the
    objective's job. state_dict / load_state_dict for checkpoints.

    data_cfg needs exactly one of `mix` (a mix script) or `source` (one
    source spec, see sources.parse_spec). `tokenizer` (a tokenizers.Tokenizer)
    is needed for raw sources; it's loaded from data_cfg["tokenizer"] if not
    given. `vision` is the vlm's image spec, for caption sources.
    """

    def __init__(self, data_cfg: dict, dp_rank: int, dp_size: int, total_steps: int,
                 tokenizer=None, ignore_index: int = -100, vision: Vision | None = None,
                 pin_memory: bool = True):
        if data_cfg.get("mix") is not None:
            self.mix_path = data_cfg["mix"]
            self.mix = load_mix(self.mix_path)
            with open(self.mix_path) as f:
                self.mix_text = f.read()
        else:
            spec = data_cfg["source"]
            assert spec is not None, "data needs `source` or `mix`"
            self.mix_path = None
            self.mix = types.SimpleNamespace(SOURCES={"data": spec}, weights=_single_source)
            self.mix_text = json.dumps(spec, sort_keys=True)
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

        tokenizer_name = data_cfg.get("tokenizer")
        if tokenizer is None and tokenizer_name is not None:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_pretrained(tokenizer_name)

        def token(key: str) -> int | None:
            if tokenizer is None or data_cfg.get(key) is None:
                return None
            id_ = tokenizer.token_to_id(data_cfg[key])
            assert id_ is not None, f"data.{key} {data_cfg[key]!r} not in the tokenizer vocab"
            return id_

        rows = RowConfig(self.seq_len, tokenizer, token("eos_token"), token("pad_token"),
                         ignore_index, vision)
        self.pool = ThreadPoolExecutor(
            max_workers=data_cfg["fetch_threads"], thread_name_prefix="mix-fetch"
        )
        # opened in parallel: listing a dataset and reading parquet footers is
        # a few seconds of round trips per source
        opened = self.pool.map(
            lambda spec: open_source(spec, rows, data_cfg["block_mb"], dp_size, tokenizer_name),
            self.mix.SOURCES.values(),
        )
        self.sources = dict(zip(self.names, opened, strict=True))
        eos = {src.eos_id for src in self.sources.values()}
        assert len(eos) == 1, f"sources disagree on eos_id: {eos}"
        self.eos_id = eos.pop()
        keys = {src.keys for src in self.sources.values()}
        assert len(keys) == 1, f"sources yield different rows {keys}: can't mix text and images"
        self.keys = keys.pop()
        self.workers = None
        if data_cfg.get("workers", 0) > 0:
            # spawn, not fork: the parent has threads (fetch pool, tokenizers)
            self.workers = ProcessPoolExecutor(
                data_cfg["workers"], mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_init, initargs=(self.sources,),
            )

        def submitter(name: str, src) -> Callable[[object], Future]:
            if self.workers is not None:
                return lambda block: self.workers.submit(_worker_fetch, name, block)
            return lambda block: self.pool.submit(src.fetch, block)

        self.streams = {
            name: SourceStream(src, name, self.seed, dp_rank, dp_size,
                               getattr(src, "window_blocks", None) or data_cfg["window_blocks"],
                               submitter(name, src), data_cfg.get("prefetch_windows", 1))
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

    def reseed(self, seed: int) -> None:
        """a different shuffle (block order, windows, step plans). before the first batch only."""
        assert self.micro_index == 0, "reseed before iterating"
        self.seed = seed
        for stream in self.streams.values():
            stream.seed = seed

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
        self.micro_index += 1

        def stacked(key: str, dtype: torch.dtype) -> torch.Tensor:
            """rows[key] gathered straight into one (pinned) tensor: one copy."""
            first = rows[0][key]
            out = torch.empty((len(rows), *first.shape), dtype=dtype, pin_memory=self.pin_memory)
            np.stack([r[key] for r in rows], out=out.numpy(), casting="unsafe")
            return out

        batch = {"input_ids": stacked("input_ids", torch.int64)}
        # text rows: every token is a target. caption rows bring their own
        if "labels" in self.keys:
            batch["labels"] = stacked("labels", torch.int64)
        else:
            batch["labels"] = torch.empty_like(batch["input_ids"], pin_memory=self.pin_memory)
            batch["labels"].copy_(batch["input_ids"])
        if "pixel_values" in self.keys:
            # uint8: normalizing is the vision model's job (siglip and clip
            # differ), on the gpu, after a 4x smaller host->device copy
            batch["pixel_values"] = stacked("pixel_values", torch.uint8)
        return batch

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
            print(f"note: data.{'mix' if self.mix_path else 'source'} "
                  f"({self.mix_path or self.mix_text}) changed since the checkpoint; "
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
        if self.workers is not None:
            # they ignore SIGTERM (see _worker_init), so kill them outright
            processes = list((getattr(self.workers, "_processes", None) or {}).values())
            self.workers.shutdown(wait=False, cancel_futures=True)
            for process in processes:
                if process.is_alive():
                    process.kill()
            for process in processes:
                process.join(timeout=5)
            self.workers = None
