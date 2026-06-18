"""
build helper for the dataloader, colocated with DataLoader
"""

import datasets
import torch

from datasets.distributed import split_dataset_by_node
from tokenizers import Tokenizer


class Collator:
    """
    batch-tokenize the canonical "text" field. padding/truncation are
    configured on the tokenizer in build_dataloader, so every encoding
    in the batch comes out the same length and stacks cleanly.

    input_ids are padded with a real vocab id (see build_dataloader); labels
    copy input_ids but set padded positions to ignore_index so the loss skips
    them. label *shifting* is still the objective's job.

    a top-level class (not a closure) so DataLoader workers can pickle it,
    which forkserver/spawn start methods require (py3.14 default on linux).
    """
    def __init__(self, tokenizer: Tokenizer, ignore_index: int):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, examples):
        texts = [ex["text"] for ex in examples]
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = torch.tensor([e.ids for e in encodings], dtype=torch.long)
        # attention_mask is 1 for real tokens, 0 for padding
        attn = torch.tensor([e.attention_mask for e in encodings], dtype=torch.bool)
        labels = input_ids.masked_fill(~attn, self.ignore_index)
        return {"input_ids": input_ids, "labels": labels, "attn_mask": attn}


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
    # (Qwen3 has no <pad>, so we reuse eos); ignore_id only ever lands in
    # labels, never in input_ids.
    pad_token = data_cfg["pad_token"]
    pad_id = tokenizer.token_to_id(pad_token)
    tokenizer.enable_truncation(max_length=data_cfg["seq_len"])
    tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        collate_fn=Collator(tokenizer, ignore_id),
        persistent_workers=True,
        pin_memory=True
    )



