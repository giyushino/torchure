"""
build helper for the dataloader, colocated with DataLoader
note that testing shows numpy is faster than torch or
python arrs, so we should move packing into the collator

but this doesn't seem to be bottleneck for throughput
so leave for now

we also need to consider if we want to use unique ids
for padding, but with our current naive list packer,
we do not need to take this into consideration 
"""

import datasets
import torch

from datasets.distributed import split_dataset_by_node
from tokenizers import Tokenizer
from torchdata.stateful_dataloader import StatefulDataLoader

# packing strategies live in packers.py so they can be swapped/benchmarked
# independently; see tests/dataloader_packing.py. ListPacker is the original
# python-list implementation.
from torchure.dataloader.packers import ListPacker


class Collator:
    """
    stack pre-packed rows into a batch. the Packer (.map step) already emits
    fixed-length seq_len blocks with no padding, so there's nothing to pad,
    truncate, or mask here -- the collate just stacks ids and clones labels.

    every position is a real token, so labels are a straight clone (no
    ignore_index in the pretrain path). label *shifting* stays the objective's
    job. ignore_index is kept on the constructor for SFT prompt-masking later.

    a top-level class (not a closure) so DataLoader workers can pickle it,
    which forkserver/spawn start methods require (py3.14 default on linux).
    """
    def __init__(self, tokenizer: Tokenizer, ignore_index: int):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, examples):
        input_ids = torch.tensor([ex["input_ids"] for ex in examples], dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def build_dataloader(data_cfg, tokenizer: Tokenizer, ignore_id: int,dp_rank: int, dp_size: int):
    # streaming on by default. an explicit split is required so we get an
    # IterableDataset back, not an IterableDatasetDict (no .num_shards, and
    # split_dataset_by_node rejects it). `config` is the dataset's subset
    # name (e.g. "en" for c4); pass None for datasets without subsets.
    # TODO: a dataset registry once per-dataset split/field quirks pile up.
    dataset = datasets.load_dataset(
        data_cfg["name"],
        data_cfg.get("config"),
        split=data_cfg.get("split", "train"),
        streaming=True,
    )
    print(f"num shards: {dataset.num_shards}")

    # normalize whatever the source calls its text column to "text" so the
    # collate_fn stays dataset-agnostic. cheap: applied lazily per element.
    text_field = data_cfg["text_field"]
    if text_field != "text":
        dataset = dataset.map(lambda ex: {"text": ex[text_field]})

    dataset = dataset.shuffle(buffer_size=data_cfg["shuffle_buffer"], seed=data_cfg["seed"])
    # split by the data-parallel mesh coordinate: ranks in the same TP/CP
    # group share a dp_rank and so deterministically read the same shards.
    dataset = split_dataset_by_node(dataset, rank=dp_rank, world_size=dp_size)

    # configure batch tokenization once; mutating here keeps the collate_fn
    # stateless and avoids reconfiguring per batch. pad with a real vocab id

    # for the pad tokens, maybe just set this to eos implicity
    pad_token = data_cfg["pad_token"]
    pad_id = tokenizer.token_to_id(pad_token)

    packer = ListPacker(tokenizer, data_cfg["seq_len"], pad_id)
    dataset = dataset.map(
        packer, batched=True,
        batch_size=data_cfg.get("pack_batch", 1000),
        remove_columns=dataset.column_names,
    )

    #return torch.utils.data.DataLoader(
    return StatefulDataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        collate_fn=Collator(tokenizer, ignore_id),
        persistent_workers=True,
        pin_memory=True
    )



