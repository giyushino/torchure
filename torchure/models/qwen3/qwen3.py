"""
self contained qwen3 arch llm
dense qwen
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    tensor = [a, b, c, d, e, f, g, h]
    rotate_half(tensor) = [-e, -f, -g, -h, a, b, c, d]
    """
    x1 = x[...,:x.shape[-1] // 2]
    x2 = x[...,x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


class Qwen3RotaryEmbedding(nn.Module):
    """
    The goal of this is to encode position by
    rotating the Q and K vectors inside attention
    """
    def __init__(self, head_dim: int, base: int = 1_000_000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"

        # inv_freq gives us theta, which determines how much
        # we should rotate the Q K vectors in attention
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, False)

        # cache for the packed-training path: with no padding mask every step
        # sees position_ids = arange(seq_len), so cos/sin are identical across
        # steps. cache them per (seq_len, device) so the eager rope math (which
        # runs outside the per-block compiled regions) happens once, not every
        # forward. keyed by seq_len; the padded/eval path recomputes.
        self._cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # add new dimension to end of position ids, get (B, S, 1)
        # add 2 new dimentions to inv_freq, get (1, 1, D/2)
        freqs = position_ids.float()[:, :, None] * self.inv_freq[None, None, :]

        # freqs shape: (B, S, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)

        # cos/sin shape: (B, 1, S, D), broadcast over heads. kept in fp32 here;
        # apply_rotary casts to the Q/K dtype at use-time.
        cos = emb.cos()[:, None, :, :]
        sin = emb.sin()[:, None, :, :]

        return cos, sin

    def forward(
        self, position_ids: torch.Tensor, cacheable: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # `cacheable` is set by the caller only when position_ids is the plain
        # arange(seq_len) broadcast over the batch (the packed, no-mask training
        # path), so the result depends solely on seq_len and can be reused.
        if not cacheable:
            return self._cos_sin(position_ids)

        key = (position_ids.shape[1], position_ids.device)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._cos_sin(position_ids)
            self._cache[key] = cached
        return cached


class Qwen3GroupQueryAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        num_kv_heads: int = 8,
        head_dim: int | None = None,
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

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        # self.rope = Qwen3RotaryEmbedding(self.head_dim)

    def split_heads_old(self, x: torch.Tensor) -> torch.Tensor:
        # reshape input tensor for multihead attention
        # x is (batch_size, seq_length, emb_dim)
        B, S, _ = x.shape

        # each head should see a slice of the embeddings
        # the new shape is (batch_size, num_heads, seq_length, head_dim)
        return x.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
     
    def split_heads(self, x):
        n_heads = x.shape[-1] // self.head_dim
        return x.reshape(*x.shape[:2], n_heads, self.head_dim).transpose(1, 2)
    
    def split_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, num_heads, S, head_dim) 
        B, _, S, _ = x.shape
        return x.transpose(1, 2).reshape(B, S, self.num_heads * self.head_dim)

    def scaled_self_attention_naive(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor | None
    ):

        # Q matrix contains what each token is looking for
        # K matrix contains how each token should be matched
        # V matrix contains the information each token can contribute

        # Q and K are (batch_size, num_heads, seq_length, head_dim)
        # for matrix mult, we'll take the transpose of K, where we
        # can simply swap head_dim and seq_length
        # divide by sqrt of head dim to reduce variance  
        # matmul automatically broadcasts over the first two dims
        # shape is now (batch_size, num_heads, seq_length, seq_length)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask_value = torch.finfo(attn_scores.dtype).min

        S = Q.size(-2)
        causal_mask = torch.triu(
            torch.ones(S, S, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, mask_value)

        # this about the padding tokens
        if attention_mask is not None:
            key_padding_mask = ~attention_mask[:, None, None, :].bool()
            attn_scores = attn_scores.masked_fill(key_padding_mask, mask_value)

        attn_probs = torch.softmax(attn_scores, dim=-1)
    
        # this final matrix mult mixes information between the embeddings
        # of a token with the tokens it attended to
        return torch.matmul(attn_probs, V)

    def scaled_self_attention(self, Q, K, V, attention_mask):
        # SDPA applies 1/sqrt(head_dim) scaling itself, so don't pre-scale Q@Kᵀ.
  
        # fast path (packing / training): pure causal. is_causal=True lets SDPA
        # use the flash kernel and never materialize the S×S score matrix.
        if attention_mask is None:
            return F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=True)
  
        # padding present (eval / left-pad generation): can't combine is_causal
        # with a key-padding mask, so fold both into one boolean mask.
        # bool semantics: True = "attend to this position".
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
        K = self.k_norm(self.split_kv_heads(self.k_proj(x)))
        V = self.split_kv_heads(self.v_proj(x))

        Q = apply_rotary(Q, cos, sin)
        K = apply_rotary(K, cos, sin)

        # GQA !! 
        # repeat_factor = self.num_heads // self.num_kv_heads
        # K = K.repeat_interleave(repeat_factor, dim=1)
        # V = V.repeat_interleave(repeat_factor, dim=1)

        attn_output = self.scaled_self_attention(Q, K, V, attention_mask)

        return self.o_proj(self.combine_heads(attn_output))


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )

    
class Qwen3TransformerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        num_kv_heads: int = 8,
        head_dim: int | None = None,
    ):
        super().__init__()
        self.gqa = Qwen3GroupQueryAttention(num_heads, emb_dim, num_kv_heads, head_dim)
        self.norm1 = nn.RMSNorm(emb_dim)
        self.norm2 = nn.RMSNorm(emb_dim)

        ffn_dim = int(emb_dim * 8/ 3) # hard coded in qwen3
        self.ffn  = SwiGLU(emb_dim, ffn_dim)

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
        vocab_size: int
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.rope = Qwen3RotaryEmbedding(head_dim)
        self.blocks = nn.ModuleList(
            Qwen3TransformerBlock(num_heads, emb_dim, num_kv_heads, head_dim)
            for _ in range(num_layers)
        )
        self.norm = nn.RMSNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def init_weights(self, std: float = 0.02) -> None:
        """
        gpt/llama-style init. without this, pytorch's default per-layer init
        lets residual-stream variance compound with depth and the logits blow
        up (fresh CE ~900 instead of ~ln(vocab)).

        the key piece is scaling the projections that *write into* the residual
        stream (attn o_proj, ffn down_proj) by 1/sqrt(2 * num_layers) so the
        added variance per layer stays bounded as depth grows.
        """
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

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        B, S = x.shape

        # keep left-padded batches compatible with RoPE by numbering only the
        # non-pad tokens.
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

        for block in self.blocks:
            x = block(x, cos, sin, attention_mask)
            if return_hidden_states:
                hidden_states.append(x)
        
        # (batch_size, seq_lenth, vocab_size)
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
    #566_632_448

#    input_ids = torch.randint(0, config["vocab_size"], (2, 10))
#    logits = model(input_ids)
#    print(logits.shape)
