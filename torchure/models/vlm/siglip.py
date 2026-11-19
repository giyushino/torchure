"""
siglip vision tower (Zhai et al. 2023): a plain pre-norm vit with no cls
token. the defaults are siglip-base-patch16-224.

module names follow hf's SiglipVisionModel (patch_embedding, layer_norm1,
self_attn.q_proj, mlp.fc1, post_layernorm, ...) so loading pretrained
weights is a key rename, not a re-layout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiglipAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        def heads(t):
            return t.view(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, N, D))


class SiglipMLP(nn.Module):
    def __init__(self, emb_dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SiglipBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, ffn_dim: int, norm_eps: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(emb_dim, eps=norm_eps)
        self.self_attn = SiglipAttention(emb_dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(emb_dim, eps=norm_eps)
        self.mlp = SiglipMLP(emb_dim, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class SiglipVisionTower(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        emb_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be a multiple of patch_size"
        self.grid = image_size // patch_size
        self.emb_dim = emb_dim
        self.patch_embedding = nn.Conv2d(3, emb_dim, patch_size, stride=patch_size)
        self.position_embedding = nn.Embedding(self.grid**2, emb_dim)
        self.layers = nn.ModuleList(
            SiglipBlock(emb_dim, num_heads, ffn_dim, norm_eps) for _ in range(num_layers)
        )
        self.post_layernorm = nn.LayerNorm(emb_dim, eps=norm_eps)

    @torch.no_grad()
    def init_weights(self, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(module.weight, std=std)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=std)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) in [-1, 1] -> (N, grid * grid, emb_dim) patch features."""
        x = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        x = x + self.position_embedding.weight
        for layer in self.layers:
            x = layer(x)
        return self.post_layernorm(x)
