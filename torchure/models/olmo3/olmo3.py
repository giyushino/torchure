"""
self contained olmo 3 (dense)

faithful to allenai/olmo-3-7b (config.json + hf transformers modeling_olmo3):
  - post-norm blocks: attn/mlp outputs are rmsnormed *after* the sublayer,
    then added to the residual (olmo 2/3 style, not llama pre-norm)
  - 3:1 sliding-window (4096) : full attention, repeating; the last layer is
    full when num_layers % 4 == 0
  - yarn rope (theta 500k, factor 8 from 8192 -> 65536, beta_fast 32,
    beta_slow 1, attention_factor 0.1*ln(8)+1)
  - swiglu mlp, no biases anywhere, untied lm_head

deliberate deviations (so the trainer's tp/sp works, see tp_plan):
  - qk-norm is per-head rmsnorm(head_dim), applied after split_heads. hf
    norms the flattened (heads*head_dim) q/k jointly; per-head is the form
    the olmo 3 paper describes ("normalized per head") and the only form
    that stays correct when heads are sharded over tp.
  - sliding window is an explicit bool mask on top of sdpa instead of a
    flash-attention window argument; same math, more memory at long seq.

defaults are the real 7b shape. trainer contract, same as qwen3: `blocks`,
`norm`, `lm_head.weight`, `init_weights()`, `tp_plan()`,
`activation_checkpointing`, forward(..., return_final_hidden=True).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


def _yarn_inv_freq(
    head_dim: int,
    base: float,
    factor: float,
    original_max: int,
    beta_fast: float,
    beta_slow: float,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """hf `_compute_yarn_parameters`, truncate=True, mscale=1."""

    def correction_dim(num_rotations: float) -> float:
        return head_dim * math.log(original_max / (num_rotations * 2 * math.pi)) / (
            2 * math.log(base)
        )

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), head_dim - 1)
    if low == high:
        high += 0.001

    pos_freqs = base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    inv_extrapolation = 1.0 / pos_freqs
    inv_interpolation = 1.0 / (factor * pos_freqs)
    ramp = torch.clamp(
        (torch.arange(head_dim // 2, device=device).float() - low) / (high - low), 0, 1
    )
    inv_freq = inv_interpolation * ramp + inv_extrapolation * (1 - ramp)
    attention_factor = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
    return inv_freq, attention_factor


class Olmo3RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float = 500_000,
        factor: float = 8.0,
        original_max: int = 8192,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        self.head_dim = head_dim
        self.base = base
        self.factor = factor
        self.original_max = original_max
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.attention_factor = 1.0
        self.register_buffer("inv_freq", torch.empty(head_dim // 2), False)
        self._cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self.init_buffers()

    @torch.no_grad()
    def init_buffers(self) -> None:
        inv_freq, factor = _yarn_inv_freq(
            self.head_dim, self.base, self.factor, self.original_max,
            self.beta_fast, self.beta_slow, self.inv_freq.device,
        )
        self.inv_freq.copy_(inv_freq)
        self.attention_factor = factor
        self._cache.clear()

    def _cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.float()[:, :, None] * self.inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[:, None, :, :] * self.attention_factor
        sin = emb.sin()[:, None, :, :] * self.attention_factor
        return cos, sin

    def forward(
        self, position_ids: torch.Tensor, cacheable: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not cacheable:
            return self._cos_sin(position_ids)
        key = (position_ids.shape[1], position_ids.device)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._cos_sin(position_ids[:1])
            self._cache[key] = cached
        return cached


class Olmo3Attention(nn.Module):
    """mha/gqa with per-head qk-norm; sliding or full, set per layer."""

    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0 and num_heads >= num_kv_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.emb_dim = emb_dim
        self.sliding_window = sliding_window

        q_out = num_heads * head_dim
        kv_out = num_kv_heads * head_dim
        self.q_proj = nn.Linear(emb_dim, q_out, bias=False)
        self.k_proj = nn.Linear(emb_dim, kv_out, bias=False)
        self.v_proj = nn.Linear(emb_dim, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, emb_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=norm_eps)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, width = x.shape
        return x.reshape(B, S, width // self.head_dim, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, heads, S, _ = x.shape
        return x.transpose(1, 2).reshape(B, S, heads * self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Q = self.q_norm(self.split_heads(self.q_proj(x)))
        K = self.k_norm(self.split_heads(self.k_proj(x)))
        V = self.split_heads(self.v_proj(x))
        Q = apply_rotary(Q, cos, sin)
        K = apply_rotary(K, cos, sin)

        S = Q.size(-2)
        window = self.sliding_window
        if attention_mask is None and (window is None or S <= window):
            out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=True)
            return self.o_proj(self.combine_heads(out))

        # explicit mask: causal (& sliding) (& padding). True = attend.
        causal = torch.ones(S, S, dtype=torch.bool, device=Q.device).tril()
        if window is not None and S > window:
            idx = torch.arange(S, device=Q.device)
            causal &= (idx[:, None] - idx[None, :]) < window
        attn_mask = causal[None, None].expand(Q.size(0), 1, S, S)
        if attention_mask is not None:
            attn_mask = attn_mask & attention_mask.bool()[:, None, None, :]
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, enable_gqa=True)
        return self.o_proj(self.combine_heads(out))


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Olmo3TransformerBlock(nn.Module):
    """post-norm: sublayer -> rmsnorm -> + residual (twice)."""

    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        num_kv_heads: int,
        head_dim: int,
        ffn_dim: int,
        sliding_window: int | None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn = Olmo3Attention(
            num_heads, emb_dim, num_kv_heads, head_dim, sliding_window, norm_eps
        )
        self.ffn = SwiGLU(emb_dim, ffn_dim)
        self.post_attn_norm = nn.RMSNorm(emb_dim, eps=norm_eps)
        self.post_ffn_norm = nn.RMSNorm(emb_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.post_attn_norm(self.attn(x, cos, sin, attention_mask))
        x = x + self.post_ffn_norm(self.ffn(x))
        return x


def _layer_types(num_layers: int, sliding_window: int | None) -> list[bool]:
    """is_sliding per layer: 3 sliding : 1 full, repeating (hf layer_types)."""
    if sliding_window is None:
        return [False] * num_layers
    return [(i + 1) % 4 != 0 for i in range(num_layers)]


class Olmo3(nn.Module):
    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: int = 32,
        emb_dim: int = 4096,
        head_dim: int = 128,
        vocab_size: int = 100_278,
        ffn_dim: int | None = 11_008,
        sliding_window: int | None = 4096,
        norm_eps: float = 1e-6,
        rope_base: float = 500_000,
        rope_factor: float = 8.0,
        rope_original_max: int = 8192,
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 256 * ((int(8 * emb_dim / 3) + 255) // 256)
        self.emb_dim = emb_dim
        self.sliding_window = sliding_window
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.rope = Olmo3RotaryEmbedding(
            head_dim, base=rope_base, factor=rope_factor, original_max=rope_original_max
        )
        self.blocks = nn.ModuleList(
            Olmo3TransformerBlock(
                num_heads, emb_dim, num_kv_heads, head_dim, ffn_dim,
                sliding_window if is_sliding else None, norm_eps,
            )
            for is_sliding in _layer_types(num_layers, sliding_window)
        )
        self.norm = nn.RMSNorm(emb_dim, eps=norm_eps)
        # olmo 3 does not tie embeddings
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.activation_checkpointing = False

    def init_weights(self, std: float = 0.02) -> None:
        """same contract as qwen3: every param/buffer set, elementwise."""
        self.rope.init_buffers()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.RMSNorm) and module.weight is not None:
                nn.init.ones_(module.weight)

        residual_scale = (2 * len(self.blocks)) ** -0.5
        for block in self.blocks:
            block.attn.o_proj.weight.data.mul_(residual_scale)
            block.ffn.down_proj.weight.data.mul_(residual_scale)

    def tp_plan(self, tp_size: int, vocab_parallel: bool = True) -> dict[str, list[nn.Module]]:
        """
        same vocabulary as qwen3.tp_plan. attention (q/k/v column, o row)
        and mlp (gate/up column, down row) are the two regions per block;
        per-head q/k norms are shared; the post-norms sit between regions,
        so they are seq_local under sequence parallel.

        untied head: embedding and lm_head are different params, each
        vocab-sharded on its own (the tp engine's tied-weight dedup simply
        never triggers).
        """
        attn = self.blocks[0].attn
        assert attn.num_kv_heads % tp_size == 0, (
            f"tp={tp_size} must divide num_kv_heads={attn.num_kv_heads}"
        )
        plan: dict[str, list[nn.Module]] = {
            "column": [], "row": [], "regions": [], "shared": [],
            "embeddings": [self.token_emb],
            "heads": [self.lm_head],
            "vocab": [self.token_emb, self.lm_head] if vocab_parallel else [],
            "seq_local": [self.norm],
        }
        for block in self.blocks:
            attn, ffn = block.attn, block.ffn
            plan["column"] += [attn.q_proj, attn.k_proj, attn.v_proj, ffn.gate_proj, ffn.up_proj]
            plan["row"] += [attn.o_proj, ffn.down_proj]
            plan["regions"] += [attn, ffn]
            plan["shared"] += [attn.q_norm, attn.k_norm]
            plan["seq_local"] += [block.post_attn_norm, block.post_ffn_norm]
        return plan

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_final_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        B, S = x.shape
        if attention_mask is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
            cacheable = True
        else:
            attention_mask = attention_mask.bool()
            position_ids = attention_mask.long().cumsum(dim=1) - 1
            position_ids = position_ids.masked_fill(~attention_mask, 0)
            cacheable = False

        x = self.token_emb(x)
        cos, sin = self.rope(position_ids, cacheable=cacheable)
        hidden_states: list[torch.Tensor] = [x] if return_hidden_states else []

        use_checkpoint = self.activation_checkpointing and torch.is_grad_enabled()
        for block in self.blocks:
            if use_checkpoint:
                x = checkpoint(block, x, cos, sin, attention_mask, use_reentrant=False)
            else:
                x = block(x, cos, sin, attention_mask)
            if return_hidden_states:
                hidden_states.append(x)

        if return_final_hidden:
            return self.norm(x)

        logits = self.lm_head(self.norm(x))
        if return_hidden_states:
            return logits, hidden_states
        return logits


if __name__ == "__main__":
    # the real olmo-3-7b shape
    model = Olmo3()
    total = sum(p.numel() for p in model.parameters())
    print(f"olmo3-7b shape: {total / 1e9:.2f}B params, {len(model.blocks)} layers")
    sliding = sum(b.attn.sliding_window is not None for b in model.blocks)
    print(f"sliding layers: {sliding}, full layers: {len(model.blocks) - sliding}")

