"""
image-caption batches for the vlm (models/vlm/vlm.py). one image per row, no
packing:

    <|vision_start|> <|image_pad|> x num_image_tokens <|vision_end|> caption <eos> <pad>...

labels are the unshifted ids with ignore_index on the image span and the
padding, so only the caption (and its eos) is scored. right padding needs no
attention mask under causal attention: nothing before a pad attends to it.

needs pillow (`uv sync --extra vision`) for hf's image decoding.
"""

import numpy as np
import torch
from tokenizers import Tokenizer


class ImageTextCollator:
    """top-level class so spawn workers can pickle it."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        ignore_index: int,
        seq_len: int,
        eos_id: int,
        pad_id: int,
        image_size: int,
        num_image_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.image_size = image_size

        def token(text: str) -> int:
            id_ = tokenizer.token_to_id(text)
            assert id_ is not None, f"{text!r} not in the tokenizer vocab"
            return id_

        self.prefix = [
            token("<|vision_start|>"),
            *[token("<|image_pad|>")] * num_image_tokens,
            token("<|vision_end|>"),
        ]
        assert seq_len > len(self.prefix), (
            f"seq_len {seq_len} leaves no room for text after {len(self.prefix)} image tokens"
        )

    def pixels(self, image) -> torch.Tensor:
        """PIL image -> (3, S, S) float in [-1, 1] (siglip's normalization)."""
        from PIL import Image

        image = image.convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BICUBIC
        )
        x = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
        return x.permute(2, 0, 1).float().div(127.5).sub(1.0)

    def __call__(self, examples):
        captions = self.tokenizer.encode_batch(
            [ex["text"] for ex in examples], add_special_tokens=False
        )
        ids = torch.full((len(examples), self.seq_len), self.pad_id, dtype=torch.long)
        labels = torch.full_like(ids, self.ignore_index)
        n_prefix = len(self.prefix)
        for row, caption in enumerate(captions):
            seq = (self.prefix + caption.ids + [self.eos_id])[: self.seq_len]
            ids[row, : len(seq)] = torch.tensor(seq)
            labels[row, n_prefix: len(seq)] = ids[row, n_prefix: len(seq)]

        pixel_values = torch.stack([self.pixels(ex["image"]) for ex in examples])
        return {"input_ids": ids, "labels": labels, "pixel_values": pixel_values}
