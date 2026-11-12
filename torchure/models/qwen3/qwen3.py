"""
self contained dense qwen3

TODO: maybe remove the rope cache, it doesn't seem to give any speedup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[a, b, c, d, e, f, g, h] -> [-e, -f, -g, -h, a, b, c, d]"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: int = 1_000_000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"

        self.head_dim = head_dim
        self.base = base
        self.register_buffer("inv_freq", torch.empty(head_dim // 2), False)

        # cos/sin per (seq_len, device) for the packed, no-mask training path
        self._cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self.init_buffers()

    @torch.no_grad()
    def init_buffers(self) -> None:
        """(re)compute inv_freq in place, e.g. after a meta-device build."""
        exponent = torch.arange(0, self.head_dim, 2, device=self.inv_freq.device).float()
        self.inv_freq.copy_(1.0 / (self.base ** (exponent / self.head_dim)))
        self._cache.clear()

    def _cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, S, 1) * (1, 1, D/2) -> (B, S, D/2)
        freqs = position_ids.float()[:, :, None] * self.inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)

        # (B, 1, S, D), fp32; apply_rotary casts to the Q/K dtype
        cos = emb.cos()[:, None, :, :]
        sin = emb.sin()[:, None, :, :]

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


class Qwen3GroupQueryAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        num_kv_heads: int = 8,
        head_dim: int | None = None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0 and num_heads >= num_kv_heads, \
            "num_heads must be a positive multiple of num_kv_heads"

        self.num_heads = num_heads
        self.emb_dim = emb_dim
        # Qwen3 decouples head_dim from emb_dim / num_heads
        self.head_dim = head_dim if head_dim is not None else emb_dim // num_heads
        self.num_kv_heads = num_kv_heads

        q_out = num_heads * self.head_dim
        kv_out = num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(emb_dim, q_out, bias=False)
        self.k_proj = nn.Linear(emb_dim, kv_out, bias=False)
        self.v_proj = nn.Linear(emb_dim, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, emb_dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)

    # head counts are read off the tensor, not self.num_heads: under tensor
    # parallel each rank holds only its slice of the heads

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, heads * head_dim) -> (B, heads, S, head_dim)
        B, S, width = x.shape
        return x.reshape(B, S, width // self.head_dim, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, heads, S, head_dim) -> (B, S, heads * head_dim)
        B, heads, S, _ = x.shape
        return x.transpose(1, 2).reshape(B, S, heads * self.head_dim)

    def scaled_self_attention(self, Q, K, V, attention_mask):
        # no mask (packed training): is_causal lets sdpa use the flash kernel
        if attention_mask is None:
            return F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=True)

        # padding: is_causal can't be combined with a mask, so build one bool
        # mask (True = attend)
        S = Q.size(-2)
        causal = torch.ones(S, S, dtype=torch.bool, device=Q.device).tril()
        key_keep = attention_mask[:, None, None, :].bool()      # (B,1,1,S)
        attn_mask = causal[None, None] & key_keep               # (B,1,S,S)
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, enable_gqa=True)

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

        attn_output = self.scaled_self_attention(Q, K, V, attention_mask)
        return self.o_proj(self.combine_heads(attn_output))


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3TransformerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        num_kv_heads: int = 8,
        head_dim: int | None = None,
        ffn_dim: int | None = None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.gqa = Qwen3GroupQueryAttention(num_heads, emb_dim, num_kv_heads, head_dim, norm_eps)
        self.norm1 = nn.RMSNorm(emb_dim, eps=norm_eps)
        self.norm2 = nn.RMSNorm(emb_dim, eps=norm_eps)

        if ffn_dim is None:
            # 8/3 * emb_dim rounded up to a multiple of 256
            ffn_dim = 256 * ((int(8 * emb_dim / 3) + 255) // 256)
        self.ffn = SwiGLU(emb_dim, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.gqa(self.norm1(x), cos, sin, attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class Qwen3(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        emb_dim: int,
        head_dim: int,
        vocab_size: int,
        ffn_dim: int | None = None,
        # explicit: nn.RMSNorm's default eps depends on input dtype
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.rope = Qwen3RotaryEmbedding(head_dim)
        self.blocks = nn.ModuleList(
            Qwen3TransformerBlock(num_heads, emb_dim, num_kv_heads, head_dim, ffn_dim, norm_eps)
            for _ in range(num_layers)
        )
        self.norm = nn.RMSNorm(emb_dim, eps=norm_eps)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        # recompute each block's activations in backward instead of storing
        # them: trades roughly one extra forward for most activation memory
        self.activation_checkpointing = False

    def init_weights(self, std: float = 0.02) -> None:
        """
        gpt/llama-style init. projections that write into the residual stream
        (o_proj, down_proj) are scaled by 1/sqrt(2 * num_layers) so variance
        doesn't compound with depth.

        initializes ALL state, buffers included, so it is also what gives a
        model built on the meta device its values. every init here is
        elementwise, so it stays correct when fsdp hands it shards.
        """
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
            block.gqa.o_proj.weight.data.mul_(residual_scale)
            block.ffn.down_proj.weight.data.mul_(residual_scale)

    def tp_plan(self, tp_size: int, vocab_parallel: bool = True) -> dict[str, list[nn.Module]]:
        """
        which modules tensor parallel splits and how (see
        parallelism/tensor_parallel.py for the vocabulary). two regions per
        block: attention (q/k/v column, o row) and the mlp (gate/up column,
        down row).

        q/k/v split by whole heads, and a kv head must stay with the q heads
        that share it, so tp has to divide num_kv_heads. q_norm/k_norm are one
        weight shared by every head: replicated, but each rank only sees its
        own heads.

        vocab_parallel shards the tied embedding / lm_head weight by vocab
        rows; otherwise it is replicated. the norms on the residual stream
        are always replicated.
        """
        gqa = self.blocks[0].gqa
        assert gqa.num_kv_heads % tp_size == 0, (
            f"tp={tp_size} must divide num_kv_heads={gqa.num_kv_heads}: shards "
            "hold whole kv heads together with their q heads"
        )
        plan: dict[str, list[nn.Module]] = {
            "column": [], "row": [], "regions": [], "shared": [],
            "embeddings": [self.token_emb],
            "heads": [self.lm_head],
            "vocab": [self.token_emb, self.lm_head] if vocab_parallel else [],
            "seq_local": [self.norm],
        }
        for block in self.blocks:
            gqa, ffn = block.gqa, block.ffn
            plan["column"] += [gqa.q_proj, gqa.k_proj, gqa.v_proj, ffn.gate_proj, ffn.up_proj]
            plan["row"] += [gqa.o_proj, ffn.down_proj]
            plan["regions"] += [gqa, ffn]
            plan["shared"] += [gqa.q_norm, gqa.k_norm]
            plan["seq_local"] += [block.norm1, block.norm2]
        return plan

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_final_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        B, S = x.shape

        # number only non-pad tokens so left padding works with rope
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

        # for fused lm_head + cross-entropy (torchure/loss/fused_linear_ce.py)
        if return_final_hidden:
            return self.norm(x)

        logits = self.lm_head(self.norm(x))
        if return_hidden_states:
            return logits, hidden_states

        return logits


if __name__ == "__main__":
    config = {
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "emb_dim": 1024,
        "head_dim": 128,
        "vocab_size": 151_936,
    }

    model = Qwen3(**config)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
