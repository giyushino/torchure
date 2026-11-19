"""
building blocks the model files share: rope, swiglu, the position-id rule,
document masking.

attribute names inside these modules are part of every model's state_dict
(`ffn.gate_proj.weight`, ...), so renaming one breaks existing checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    create_mask,
    flex_attention,
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[a, b, c, d, e, f, g, h] -> [-e, -f, -g, -h, a, b, c, d]"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """
    rope cos/sin tables. plain rope by default; a subclass overrides
    `frequencies` for another schedule (olmo's yarn).

    `inv_freq` is a non-persistent buffer filled by `init_buffers`, so a
    model built on the meta device gets it from the model's `init_weights`.
    """

    def __init__(self, head_dim: int, base: float = 10_000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        self.head_dim = head_dim
        self.base = base
        # multiplies cos and sin (yarn's attention factor); 1 for plain rope
        self.scale = 1.0
        self.register_buffer("inv_freq", torch.empty(head_dim // 2), False)
        # cos/sin per (seq_len, device) for the packed, no-mask training path
        self._cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self.init_buffers()

    def frequencies(self, device: torch.device) -> tuple[torch.Tensor, float]:
        """(inv_freq, scale)."""
        exponent = torch.arange(0, self.head_dim, 2, device=device).float()
        return 1.0 / (self.base ** (exponent / self.head_dim)), 1.0

    @torch.no_grad()
    def init_buffers(self) -> None:
        """(re)compute inv_freq in place, e.g. after a meta-device build."""
        inv_freq, self.scale = self.frequencies(self.inv_freq.device)
        self.inv_freq.copy_(inv_freq)
        self._cache.clear()

    def _cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, S, 1) * (1, 1, D/2) -> (B, S, D/2)
        freqs = position_ids.float()[:, :, None] * self.inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        # (B, 1, S, D), fp32; apply_rotary casts to the Q/K dtype
        cos = emb.cos()[:, None, :, :]
        sin = emb.sin()[:, None, :, :]
        if self.scale != 1.0:
            cos, sin = cos * self.scale, sin * self.scale
        return cos, sin

    def forward(
        self, position_ids: torch.Tensor, cacheable: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cacheable: position_ids is arange(seq_len) broadcast over the batch
        if not cacheable:
            return self._cos_sin(position_ids)

        # cache a single row (1, 1, S, D) so it broadcasts over any batch size
        key = (position_ids.shape[1], position_ids.device)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._cos_sin(position_ids[:1])
            self._cache[key] = cached
        return cached


def positions(
    ids: torch.Tensor, attention_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    """
    (position_ids, bool attention mask or None, cacheable) for a batch of
    token ids (B, S). with a mask only non-pad tokens are numbered, so left
    padding works with rope. cacheable: the positions are a plain arange.
    """
    B, S = ids.shape
    if attention_mask is None:
        return torch.arange(S, device=ids.device).unsqueeze(0).expand(B, -1), None, True
    attention_mask = attention_mask.bool()
    position_ids = attention_mask.long().cumsum(dim=1) - 1
    return position_ids.masked_fill(~attention_mask, 0), attention_mask, False


def doc_mask(
    ids: torch.Tensor, eos_id: int, windows=(None,)
) -> tuple[torch.Tensor, dict[int | None, BlockMask]]:
    """
    document masking for packed rows: each token attends only to earlier
    tokens of its own document, and positions restart at 0 per document.
    a document ends at (and includes) an eos; a row starting mid-document
    numbers that fragment from 0 too.

    returns (position_ids, {window: BlockMask}) for each sliding window size
    in `windows` (None = no window). built once per forward, outside the
    compiled blocks, and shared by every layer.
    """
    B, S = ids.shape
    starts = torch.ones_like(ids, dtype=torch.bool)
    starts[:, 1:] = ids[:, :-1] == eos_id
    doc = starts.cumsum(dim=1)
    idx = torch.arange(S, device=ids.device).expand(B, S)
    position_ids = idx - torch.where(starts, idx, 0).cummax(dim=1).values

    def mask_for(window):
        def mask_mod(b, h, q, kv):
            keep = (doc[b, q] == doc[b, kv]) & (q >= kv)
            return keep if window is None else keep & (q - kv < window)
        return create_block_mask(mask_mod, B, None, S, S, device=ids.device, _compile=True)

    return position_ids, {w: mask_for(w) for w in windows}


# compiled once: eager flex_attention materializes the full score matrix
_flex_attention = torch.compile(flex_attention, dynamic=False)


def masked_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, block_mask: BlockMask
) -> torch.Tensor:
    """flex attention under a doc_mask BlockMask; (B, H, S, D) in and out, gqa ok."""
    gqa = Q.size(1) != K.size(1)
    if Q.device.type == "cpu":
        # compiled flex backward fails on cpu (torch 2.10): same mask, dense, through sdpa
        B, _, S, _ = Q.shape
        dense = create_mask(block_mask.mask_mod, B, 1, S, S, device=Q.device)
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=dense, enable_gqa=gqa)
    return _flex_attention(Q, K, V, block_mask=block_mask, enable_gqa=gqa)


def swiglu_ffn_dim(emb_dim: int) -> int:
    """8/3 * emb_dim rounded up to a multiple of 256."""
    return 256 * ((int(8 * emb_dim / 3) + 255) // 256)


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
