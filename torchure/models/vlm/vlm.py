"""
qwen3 + siglip vlm, llava-style: vision tower -> mlp projector -> image
features replace the `image_token_id` placeholders in the token embeddings,
then the qwen3 decoder runs as usual (1d rope over the whole sequence).

    [<|vision_start|>, <|image_pad|> x num_image_tokens, <|vision_end|>, text...]

`pixel_shuffle` r > 1 folds each r x r patch neighbourhood into one token
(idefics3 / smolvlm), cutting the image's share of the sequence by r^2.

the vision tower and projector sit in fsdp's root unit (only the decoder
blocks are units) and are not compiled. tensor parallel is not supported:
tp's embedding hooks shard the sequence before the image features land.
"""

import torch
import torch.nn as nn

from torchure.models.qwen3.qwen3 import Qwen3
from torchure.models.vlm.siglip import SiglipVisionTower


class Projector(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, grid: int, pixel_shuffle: int):
        super().__init__()
        assert grid % pixel_shuffle == 0, "pixel_shuffle must divide the patch grid"
        self.grid = grid
        self.r = pixel_shuffle
        self.fc1 = nn.Linear(vision_dim * pixel_shuffle**2, text_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(text_dim, text_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 1:
            # (N, g*g, D) -> (N, (g/r)^2, r*r*D): each r x r window becomes one token
            N, _, D = x.shape
            g, r = self.grid // self.r, self.r
            x = x.view(N, g, r, g, r, D).permute(0, 1, 3, 2, 4, 5).reshape(N, g * g, r * r * D)
        return self.fc2(self.act(self.fc1(x)))


class Qwen3VL(Qwen3):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        emb_dim: int,
        head_dim: int,
        vocab_size: int,
        image_token_id: int,
        # SiglipVisionTower kwargs
        vision: dict,
        pixel_shuffle: int = 1,
        ffn_dim: int | None = None,
        norm_eps: float = 1e-6,
    ):
        super().__init__(
            num_layers, num_heads, num_kv_heads, emb_dim, head_dim, vocab_size,
            ffn_dim, norm_eps,
        )
        self.image_token_id = image_token_id
        self.vision = SiglipVisionTower(**vision)
        self.projector = Projector(self.vision.emb_dim, emb_dim, self.vision.grid, pixel_shuffle)
        # placeholders per image; the dataloader reads these two off the model
        self.num_image_tokens = (self.vision.grid // pixel_shuffle) ** 2
        self.image_size = self.vision.grid * self.vision.patch_embedding.stride[0]

    def init_weights(self, std: float = 0.02) -> None:
        super().init_weights(std)
        self.vision.init_weights(std)

    def tp_plan(self, tp_size: int, vocab_parallel: bool = True):
        raise NotImplementedError("tensor parallel is not supported for the vlm yet")

    def forward(
        self,
        x: torch.Tensor,
        pixel_values: torch.Tensor | None = None,  # (num_images, 3, H, W)
        **kwargs,
    ):
        if pixel_values is None:
            return super().forward(x, **kwargs)

        embeds = self.token_emb(x)
        # (num_images, num_image_tokens, emb_dim) -> the placeholders, in order
        features = self.projector(self.vision(pixel_values))
        slots = (x == self.image_token_id).unsqueeze(-1)
        embeds = embeds.masked_scatter(slots, features.to(embeds.dtype))
        return super().forward(x, inputs_embeds=embeds, **kwargs)
