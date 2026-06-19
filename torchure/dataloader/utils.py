"""
build helper for the dataloader, colocated with DataLoader
"""

import datasets
import torch

from datasets.distributed import split_dataset_by_node
from tokenizers import Tokenizer


class Packer:
    """
    streaming sequence packer for the .map(batched=True) step.

    tokenizes a map-batch of docs, concatenates their ids into one stream with
    an eos between docs, then slices into back-to-back seq_len blocks. every
    emitted row is exactly seq_len, so there is no padding and no attention mask
    downstream. the remainder (< seq_len) at the tail of each map-batch is
    dropped.

    top-level class (not a closure) so DataLoader workers can pickle it, which
    forkserver/spawn require (py3.14 default on linux).

    currently we're throwing away a fair number of tokens away, we should
    carry the discard tokens forward to the next batch
    """
    def __init__(self, tokenizer: Tokenizer, seq_len: int, eos_id: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = eos_id

    def __call__(self, examples: dict[str, list]) -> dict[str, list]:
        # this might be faster to turn into 
        # numpy array and then reshape
        encodings = self.tokenizer.encode_batch(examples["text"])
        stream = []

        for encoding in encodings:
            stream.extend(encoding.ids)
            stream.append(self.eos_id)

        n_blocks = len(stream) // self.seq_len
        # measure the remainder before we slice the stream down to full blocks
        print(f"lost {len(stream) - n_blocks * self.seq_len} tokens")
        stream = stream[: n_blocks * self.seq_len]

        blocks = [
            stream[i * self.seq_len: (i + 1) * self.seq_len]
            for i in range(n_blocks)
        ]

        return {"input_ids": blocks}


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
    pad_token = data_cfg["pad_token"]
    pad_id = tokenizer.token_to_id(pad_token)

    packer = Packer(tokenizer, data_cfg["seq_len"], pad_id)
    dataset = dataset.map(
        packer, batched=True,
        batch_size=data_cfg.get("pack_batch", 1000),
        remove_columns=dataset.column_names,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        collate_fn=Collator(tokenizer, ignore_id),
        persistent_workers=True,
        pin_memory=True
    )



