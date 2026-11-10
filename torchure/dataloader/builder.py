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
    shutdown = getattr(loader_iter, "_shutdown_workers", None)
    if shutdown is not None:
        try:
            shutdown()
        except Exception:  # noqa: BLE001 -- best effort, the kill below is what counts
            pass
    for worker in getattr(loader_iter, "_workers", []):
        if worker.is_alive():
            worker.kill()


def build_dataloader(
    data_cfg, tokenizer: Tokenizer, ignore_id: int, dp_rank: int, dp_size: int,
    shield_sigterm: bool = False,
):
    # shield_sigterm: workers ignore SIGTERM (see _ignore_sigterm). only for a
    # caller that calls shutdown_workers() before exit; off by default so a
    # plain script can't hang at exit.
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

    # rename_column rather than a lambda .map, since lambdas don't pickle to workers
    text_field = data_cfg["text_field"]
    if text_field != "text":
        dataset = dataset.rename_column(text_field, "text")

    dataset = dataset.shuffle(buffer_size=data_cfg["shuffle_buffer"], seed=data_cfg["seed"])
    # tp/cp peers share a dp_rank, so they read the same shards
    dataset = split_dataset_by_node(dataset, rank=dp_rank, world_size=dp_size)

    # eos and pad must be distinct ids (see packers.py)
    eos_id = tokenizer.token_to_id(data_cfg["eos_token"])
    pad_id = tokenizer.token_to_id(data_cfg["pad_token"])
    assert eos_id is not None, f"eos_token {data_cfg['eos_token']!r} not in tokenizer vocab"
    assert pad_id is not None, f"pad_token {data_cfg['pad_token']!r} not in tokenizer vocab"

    packer = ListPacker(tokenizer, data_cfg["seq_len"], eos_id, pad_id)
    dataset = dataset.map(
        packer, batched=True,
        batch_size=data_cfg.get("pack_batch", 1000),
        remove_columns=dataset.column_names,
    )

    # micro batches only; grad accumulation is the trainer's job
    num_workers = data_cfg["num_workers"]
    return StatefulDataLoader(
        dataset,
        batch_size=data_cfg["micro_batch_size"],
        num_workers=num_workers,
        collate_fn=Collator(tokenizer, ignore_id),
        worker_init_fn=_ignore_sigterm if shield_sigterm else None,
        persistent_workers=num_workers > 0,
        # fork can deadlock on fsspec's background threads inherited from the parent
        multiprocessing_context="spawn" if num_workers > 0 else None,
        pin_memory=True
    )



