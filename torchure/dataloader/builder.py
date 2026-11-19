"""
dataloader builder

TODO: numpy packing in the collator benchmarks faster, but packing isn't a
throughput bottleneck yet
"""

import signal

import datasets
import torch

from datasets.distributed import split_dataset_by_node
from tokenizers import Tokenizer
from torchdata.stateful_dataloader import StatefulDataLoader

from torchure.dataloader.mixture import MixtureLoader
from torchure.dataloader.packers import ListPacker


class Collator:
    """
    stack pre-packed seq_len rows into a batch. labels are an unshifted clone;
    shifting is the objective's job. ignore_index is kept for sft masking later.

    top-level class so spawn workers can pickle it.
    """
    def __init__(self, tokenizer: Tokenizer, ignore_index: int):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, examples):
        input_ids = torch.tensor([ex["input_ids"] for ex in examples], dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def _ignore_sigterm(worker_id: int) -> None:
    """
    a SIGTERM aimed at the job usually reaches the whole process group,
    workers included. the trainer answers it by checkpointing at its next log
    point and needs batches until then, so workers sit it out.

    the price: nothing can SIGTERM them at exit either, which is how the
    dataloader and multiprocessing reap stragglers. whoever iterates the
    loader must call shutdown_workers() before the process exits.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def shutdown_workers(loader_iter) -> None:
    """
    stop a loader iterator's workers for good: the iterator's own graceful
    shutdown first, then SIGKILL for any that are still alive. safe to call
    twice, and on a single-process iterator.
    """
    close = getattr(loader_iter, "close", None)  # MixtureLoader's fetch threads
    if close is not None:
        close()
    shutdown = getattr(loader_iter, "_shutdown_workers", None)
    if shutdown is not None:
        try:
            shutdown()
        except Exception:  # noqa: BLE001 -- best effort, the kill below is what counts
            pass
    for worker in getattr(loader_iter, "_workers", []):
        if worker.is_alive():
            worker.kill()


def _check_shards(num_shards: int, dp_size: int, num_workers: int) -> None:
    """
    warn about shard counts that quietly cost throughput. split_dataset_by_node
    hands each rank whole shards only when the rank count divides the shard
    count; otherwise EVERY rank streams the whole dataset and keeps one
    example in dp_size. and a rank with fewer shards than workers leaves the
    extra workers idle.
    """
    if num_shards % dp_size != 0:
        print(
            f"WARNING: {num_shards} dataset shards don't divide over dp={dp_size} ranks. "
            f"every rank will read and tokenize the whole stream and drop "
            f"{dp_size - 1}/{dp_size} of it. use a shard count that is a multiple of {dp_size}"
        )
    elif num_workers > num_shards // dp_size:
        print(
            f"WARNING: each rank gets {num_shards // dp_size} shard(s) but has "
            f"{num_workers} dataloader workers; the extra workers will sit idle"
        )


def build_dataloader(
    data_cfg, tokenizer: Tokenizer, ignore_id: int, dp_rank: int, dp_size: int,
    shield_sigterm: bool = False, total_steps: int | None = None,
    vision: tuple[int, int] | None = None,
):
    """
    `data.mix` set: a MixtureLoader over pre-tokenized sources (mixture.py),
    which needs total_steps for the schedule. otherwise the hf path below.
    """
    if data_cfg["mix"] is not None:
        assert total_steps is not None, "a mix schedule needs total_steps"
        assert data_cfg["image_field"] is None, "data.mix doesn't support image-text rows"
        return MixtureLoader(data_cfg, dp_rank, dp_size, total_steps, data_cfg["tokenizer"])
    return build_hf_dataloader(
        data_cfg, tokenizer, ignore_id, dp_rank, dp_size, shield_sigterm, vision
    )


def build_hf_dataloader(
    data_cfg, tokenizer: Tokenizer, ignore_id: int, dp_rank: int, dp_size: int,
    shield_sigterm: bool = False, vision: tuple[int, int] | None = None,
):
    """
    a streaming hf dataset -> shuffle -> this dp rank's shards -> tokenize and
    pack into seq_len rows -> micro batches.

    shield_sigterm: workers ignore SIGTERM (see _ignore_sigterm). only for a
    caller that calls shutdown_workers() before exit; off by default so a
    plain script can't hang at exit.

    vision: (image_size, num_image_tokens) of a vlm. needed when the config
    sets data.image_field, which switches to image-caption rows (dataloader/vlm.py).
    """
    dataset = stream_dataset(data_cfg, dp_rank, dp_size)

    # eos and pad must be distinct ids (see packers.py)
    eos_id = tokenizer.token_to_id(data_cfg["eos_token"])
    pad_id = tokenizer.token_to_id(data_cfg["pad_token"])
    assert eos_id is not None, f"eos_token {data_cfg['eos_token']!r} not in tokenizer vocab"
    assert pad_id is not None, f"pad_token {data_cfg['pad_token']!r} not in tokenizer vocab"

    if data_cfg["image_field"] is not None:
        from torchure.dataloader.vlm import ImageTextCollator

        assert vision is not None, "data.image_field is set, but the model has no vision tower"
        image_size, num_image_tokens = vision
        collate_fn = ImageTextCollator(
            tokenizer, ignore_id, data_cfg["seq_len"], eos_id, pad_id,
            image_size, num_image_tokens,
        )
        return make_loader(dataset, data_cfg, collate_fn, shield_sigterm)

    packer = ListPacker(tokenizer, data_cfg["seq_len"], eos_id, pad_id)
    dataset = dataset.map(
        packer, batched=True,
        batch_size=data_cfg["pack_batch"],
        remove_columns=dataset.column_names,
    )
    return make_loader(dataset, data_cfg, Collator(tokenizer, ignore_id), shield_sigterm)


def stream_dataset(data_cfg, dp_rank: int, dp_size: int):
    """the configured hf dataset, streamed, shuffled, and cut to this dp rank's shards."""
    # an explicit split returns an IterableDataset (not a Dict), which
    # split_dataset_by_node needs. `config` is the hf subset name, or None.
    # TODO: doesn't handle hf datasets stored as json
    dataset = datasets.load_dataset(
        data_cfg["name"],
        data_cfg.get("config"),
        split=data_cfg.get("split", "train"),
        streaming=True,
    )
    if dp_rank == 0:
        print(f"num shards: {dataset.num_shards}")
        _check_shards(dataset.num_shards, dp_size, data_cfg["num_workers"])

    # rename_column rather than a lambda .map, since lambdas don't pickle to workers
    for field, name in ((data_cfg["text_field"], "text"), (data_cfg["image_field"], "image")):
        if field is not None and field != name:
            dataset = dataset.rename_column(field, name)

    dataset = dataset.shuffle(buffer_size=data_cfg["shuffle_buffer"], seed=data_cfg["seed"])
    # tp/cp peers share a dp_rank, so they read the same shards
    return split_dataset_by_node(dataset, rank=dp_rank, world_size=dp_size)


def make_loader(dataset, data_cfg, collate_fn, shield_sigterm: bool) -> StatefulDataLoader:
    # micro batches only; grad accumulation is the trainer's job
    num_workers = data_cfg["num_workers"]
    return StatefulDataLoader(
        dataset,
        batch_size=data_cfg["micro_batch_size"],
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=_ignore_sigterm if shield_sigterm else None,
        persistent_workers=num_workers > 0,
        # fork can deadlock on fsspec's background threads inherited from the parent
        multiprocessing_context="spawn" if num_workers > 0 else None,
        pin_memory=True,
    )
