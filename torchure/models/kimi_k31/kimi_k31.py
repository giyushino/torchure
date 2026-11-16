"""
self contained kimi k3.1, scaled down to ~4b total params

the real kimi-k3.1-pro is 2.8t total / 104b active: 93 layers at hidden
7168, 69 kda + 24 gated (dsa) mla layers, 896 routed experts (top-16) +
2 shared in a 3584-wide latent moe, situ-glu everywhere, block attention
residuals (block size 12), 163840 vocab, untied head. this file keeps the
architecture and scales every width/depth/count down (see KimiK31 for the
scaled defaults; `__main__` prints the exact total / active counts):

  - hybrid attention, 3 kda : 1 mla repeating, final layer mla (k3 uses
    1-based full layers 4, 8, ..., 92, 93)
  - kda (kimi delta attention): causal short conv (k=4) + silu on q/k/v,
    l2-normed q/k, per-channel forget gate alpha in (0, 1) with a -5 log
    floor, sigmoid beta write gate, recurrent gated delta rule
        S <- (I - beta k k^T) diag(alpha) S + beta k v^T,  o = S^T q
    gated rmsnorm on the output. no rope in kda (nope).
  - mla: low-rank q/kv compression, nope (128) + rope (64) qk dims, v 128,
    per-head sigmoid output gate. optional dsa: a learned indexer scores
    positions and each query attends to its top-k (off by default; at the
    scaled context it is an inference optimisation, not a quality lever)
  - latent moe: hidden -> latent (0.5x) -> routed situ experts -> latent
    norm happens on the way in -> up to hidden; shared experts run on the
    hidden stream in parallel. sigmoid router, top-k renormalised,
    aux-free (a zero-init router bias buffer is kept for the noaux_tc
    bias-update rule; nothing here updates it yet, so routing is purely
    the learned sigmoid for now)
  - situ-glu: beta1*tanh(g/beta1) * sigmoid(g) * beta2*tanh(u/beta2),
    beta1=4, beta2=25, so outputs are bounded by 100
  - block attention residuals: sublayer outputs sum into a per-block
    partial; before each sublayer the input is a softmax mix (learned
    per-sublayer pseudo-query w over rmsnormed sources) of the embedding,
    the completed block sums, and the current partial
  - first layer is dense (first_k_dense_replace=1 in the real config)

pure-pytorch references, deliberately: the kda recurrence is a sequential
scan (compiler-disabled, like the tp collectives; checkpointed in segments
so its autograd memory stays bounded, see _kda_scan) and the moe dispatch is
a per-expert loop. fused chunkwise-kda / grouped-moe kernels are the
follow-up, not something to fake here. expect kda layers to be slow at
seq 4096 until then; the scaled config in configs/ turns compile off.

trainer contract, same as qwen3: `blocks`, `norm`, `lm_head.weight`,
`init_weights()`, `tp_plan()`, `activation_checkpointing`,
forward(..., return_final_hidden=True). tp is vocab-only (see tp_plan):
the current tp engine shards (b, s, d) regions, and neither the kda scan,
mla's latent bottleneck, nor expert dispatch fit that vocabulary.
fsdp/dp over `blocks` is the intended scaling path (ep is still a stub).
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


class KimiRotaryEmbedding(nn.Module):
    """plain rope for the mla rope dims only (kda is nope)."""

    def __init__(self, head_dim: int, base: float = 10_000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        self.head_dim = head_dim
        self.base = base
        self.register_buffer("inv_freq", torch.empty(head_dim // 2), False)
        self._cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self.init_buffers()

    @torch.no_grad()
    def init_buffers(self) -> None:
        exponent = torch.arange(0, self.head_dim, 2, device=self.inv_freq.device).float()
        self.inv_freq.copy_(1.0 / (self.base ** (exponent / self.head_dim)))
        self._cache.clear()

    def _cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.float()[:, :, None] * self.inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[:, None, :, :], emb.sin()[:, None, :, :]

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


def situ_glu(gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    return (beta * torch.tanh(gate / beta) * torch.sigmoid(gate)) * (
        linear_beta * torch.tanh(up / linear_beta)
    )


class SiTUMLP(nn.Module):
    def __init__(self, in_dim: int, ffn_dim: int, out_dim: int, beta: float, linear_beta: float):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta
        self.gate_proj = nn.Linear(in_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(in_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            situ_glu(self.gate_proj(x), self.up_proj(x), self.beta, self.linear_beta)
        )


# timesteps per checkpointed segment of the kda scan
KDA_SCAN_CHUNK = 64


def _kda_segment(
    state: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    alpha: torch.Tensor, beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """the recurrence over one run of timesteps: (final state, outputs)."""
    outs = []
    for t in range(q.shape[1]):
        k_t, v_t = k[:, t], v[:, t]                       # (B, H, D)
        state = state * alpha[:, t].unsqueeze(-1)         # per-channel decay
        pred = (state * k_t.unsqueeze(-1)).sum(dim=-2)    # S^T k_t
        delta = beta[:, t].unsqueeze(-1) * (v_t - pred)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * q[:, t].unsqueeze(-1)).sum(dim=-2))
    return state, torch.stack(outs, dim=1)


@torch.compiler.disable
def _kda_scan(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    alpha: torch.Tensor, beta: torch.Tensor,
    chunk: int = KDA_SCAN_CHUNK,
) -> torch.Tensor:
    """
    recurrent gated delta rule. q/k/v (B, S, H, D), alpha (B, S, H, D) in
    (0, 1), beta (B, S, H). state S is (B, H, D, D), kept in fp32.

    autograd would keep several (B, H, D, D) tensors for EVERY timestep,
    gigabytes per sequence per layer at seq 4096. so with grad on, the scan
    runs in segments of `chunk` steps under activation checkpointing: only
    the state at each segment boundary is kept, and a segment's steps are
    recomputed when backward reaches it. memory goes from O(S) states to
    O(S / chunk + chunk), for one extra forward of the scan.
    """
    B, S, H, D = q.shape
    state = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    q = q.float() * (D ** -0.5)
    k, v, alpha, beta = k.float(), v.float(), alpha.float(), beta.float()
    use_checkpoint = torch.is_grad_enabled() and S > chunk and any(
        t.requires_grad for t in (q, k, v, alpha, beta)
    )
    outs = []
    for s in range(0, S, chunk):
        args = (state, *(t[:, s:s + chunk] for t in (q, k, v, alpha, beta)))
        if use_checkpoint:
            state, out = checkpoint(_kda_segment, *args, use_reentrant=False)
        else:
            state, out = _kda_segment(*args)
        outs.append(out)
    return torch.cat(outs, dim=1)


class KimiDeltaAttention(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        norm_eps: float = 1e-5,
        conv_kernel: int = 4,
        gate_lower_bound: float = -5.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.gate_lower_bound = gate_lower_bound
        width = num_heads * head_dim

        self.q_proj = nn.Linear(emb_dim, width, bias=False)
        self.k_proj = nn.Linear(emb_dim, width, bias=False)
        self.v_proj = nn.Linear(emb_dim, width, bias=False)
        self.o_proj = nn.Linear(width, emb_dim, bias=False)
        # depthwise causal short convs on q/k/v
        self.q_conv = nn.Conv1d(width, width, conv_kernel, groups=width, bias=False)
        self.k_conv = nn.Conv1d(width, width, conv_kernel, groups=width, bias=False)
        self.v_conv = nn.Conv1d(width, width, conv_kernel, groups=width, bias=False)
        # per-channel forget gate (full-rank, as in the real config)
        self.gate_proj = nn.Linear(emb_dim, width, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(width))
        self.A_log = nn.Parameter(torch.full((num_heads, head_dim), math.log(8.0)))
        self.beta_proj = nn.Linear(emb_dim, num_heads, bias=False)
        # gated output norm
        self.out_gate_proj = nn.Linear(emb_dim, width, bias=False)
        self.out_norm = nn.RMSNorm(head_dim, eps=norm_eps)

    def _conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        # x (B, S, W) -> causal depthwise conv -> (B, S, W)
        pad = conv.kernel_size[0] - 1
        y = F.conv1d(
            F.pad(x.transpose(1, 2), (pad, 0)), conv.weight, groups=conv.groups
        )
        return y.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # kda is nope and has no padding mask in the recurrent form; the
        # trainer's packed path (attention_mask=None) is the supported one.
        B, S, _ = x.shape
        H, D = self.num_heads, self.head_dim

        def heads(y: torch.Tensor) -> torch.Tensor:
            return y.reshape(B, S, H, D)

        q = heads(F.silu(self._conv(self.q_conv, self.q_proj(x))))
        k = heads(F.silu(self._conv(self.k_conv, self.k_proj(x))))
        v = heads(self._conv(self.v_conv, self.v_proj(x)))
        q = F.normalize(q.float(), p=2, dim=-1).to(x.dtype)
        k = F.normalize(k.float(), p=2, dim=-1).to(x.dtype)

        gate = self.gate_proj(x).reshape(B, S, H, D)
        log_alpha = -torch.exp(self.A_log)[None, None] * F.softplus(
            gate.float() + self.dt_bias.reshape(1, 1, H, D)
        )
        alpha = torch.exp(log_alpha.clamp_min(self.gate_lower_bound))
        beta = torch.sigmoid(self.beta_proj(x))

        out = _kda_scan(q, k, v, alpha, beta).to(x.dtype)  # (B, S, H, D)
        out = self.out_norm(out.reshape(B, S, H, D))
        out = out * torch.sigmoid(self.out_gate_proj(x).reshape(B, S, H, D))
        return self.o_proj(out.reshape(B, S, H * D))


class MultiHeadLatentAttention(nn.Module):
    """gated mla; optional learned dsa indexer restricting each query's keys."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        qk_nope_dim: int = 128,
        qk_rope_dim: int = 64,
        v_head_dim: int = 128,
        q_lora_rank: int = 512,
        kv_lora_rank: int = 256,
        norm_eps: float = 1e-5,
        dsa_topk: int | None = None,
        dsa_index_heads: int = 8,
        dsa_index_dim: int = 64,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qk_nope_dim = qk_nope_dim
        self.qk_rope_dim = qk_rope_dim
        self.v_head_dim = v_head_dim
        self.dsa_topk = dsa_topk

        self.q_down = nn.Linear(emb_dim, q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(q_lora_rank, eps=norm_eps)
        self.q_up = nn.Linear(q_lora_rank, num_heads * (qk_nope_dim + qk_rope_dim), bias=False)
        self.kv_down = nn.Linear(emb_dim, kv_lora_rank + qk_rope_dim, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank, eps=norm_eps)
        self.kv_up = nn.Linear(kv_lora_rank, num_heads * (qk_nope_dim + v_head_dim), bias=False)
        self.o_proj = nn.Linear(num_heads * v_head_dim, emb_dim, bias=False)
        self.out_gate = nn.Linear(emb_dim, num_heads, bias=False)

        if dsa_topk is not None:
            self.index_q = nn.Linear(emb_dim, dsa_index_heads * dsa_index_dim, bias=False)
            self.index_k = nn.Linear(emb_dim, dsa_index_dim, bias=False)
            self.index_k_norm = nn.RMSNorm(dsa_index_dim, eps=norm_eps)
            self.dsa_index_heads = dsa_index_heads
            self.dsa_index_dim = dsa_index_dim

    def _dsa_mask(self, x: torch.Tensor, S: int) -> torch.Tensor:
        """(B, 1, S, S) bool mask: causal and within each query's top-k."""
        B = x.size(0)
        qi = self.index_q(x).reshape(B, S, self.dsa_index_heads, self.dsa_index_dim)
        qi = qi.transpose(1, 2)                                    # (B, Hi, S, Di)
        ki = self.index_k_norm(self.index_k(x))                    # (B, S, Di)
        scores = torch.einsum("bhid,bsd->bhis", qi.float(), ki.float())
        scores = scores.mean(dim=1) * (self.dsa_index_dim ** -0.5)  # (B, S, S)
        causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal[None], float("-inf"))
        topk = min(self.dsa_topk, S)
        picked = scores.topk(topk, dim=-1).indices                 # (B, S, topk)
        mask = torch.zeros(B, S, S, dtype=torch.bool, device=x.device)
        mask.scatter_(-1, picked, True)
        return (mask & causal[None])[:, None]

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        H = self.num_heads

        q = self.q_up(self.q_norm(self.q_down(x)))
        q = q.reshape(B, S, H, self.qk_nope_dim + self.qk_rope_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        kv_raw = self.kv_down(x)
        kv_latent, k_rope = kv_raw.split(
            [kv_raw.size(-1) - self.qk_rope_dim, self.qk_rope_dim], dim=-1
        )
        kv = self.kv_up(self.kv_norm(kv_latent))
        kv = kv.reshape(B, S, H, self.qk_nope_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = kv.split([self.qk_nope_dim, self.v_head_dim], dim=-1)

        # rope only on the rope slices; k_rope is one shared head, broadcast
        q_rope = apply_rotary(q_rope, cos, sin)
        k_rope = apply_rotary(k_rope.reshape(B, S, 1, self.qk_rope_dim).transpose(1, 2), cos, sin)
        Q = torch.cat([q_nope, q_rope], dim=-1)
        K = torch.cat([k_nope, k_rope.expand(-1, H, -1, -1)], dim=-1)

        scale = (self.qk_nope_dim + self.qk_rope_dim) ** -0.5
        if attention_mask is None and self.dsa_topk is None:
            out = F.scaled_dot_product_attention(Q, K, v, is_causal=True, scale=scale)
        else:
            causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
            attn_mask = causal[None, None].expand(B, 1, S, S)
            if self.dsa_topk is not None and S > self.dsa_topk:
                attn_mask = attn_mask & self._dsa_mask(x, S)
            if attention_mask is not None:
                attn_mask = attn_mask & attention_mask.bool()[:, None, None, :]
            out = F.scaled_dot_product_attention(Q, K, v, attn_mask=attn_mask, scale=scale)

        gate = torch.sigmoid(self.out_gate(x)).reshape(B, S, H, 1).transpose(1, 2)
        out = (out * gate).transpose(1, 2).reshape(B, S, H * self.v_head_dim)
        return self.o_proj(out)


class LatentMoE(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        latent_dim: int,
        moe_dim: int,
        num_experts: int,
        experts_per_token: int,
        num_shared_experts: int,
        shared_dim: int,
        beta: float,
        linear_beta: float,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.down_proj = nn.Linear(emb_dim, latent_dim, bias=False)
        self.latent_norm = nn.RMSNorm(latent_dim, eps=norm_eps)
        self.up_proj = nn.Linear(latent_dim, emb_dim, bias=False)
        self.router = nn.Linear(latent_dim, num_experts, bias=False)
        # noaux_tc bias: updated by a balancing rule, not by gradient
        self.register_buffer("router_bias", torch.zeros(num_experts))
        self.experts = nn.ModuleList(
            SiTUMLP(latent_dim, moe_dim, latent_dim, beta, linear_beta)
            for _ in range(num_experts)
        )
        self.shared_experts = nn.ModuleList(
            SiTUMLP(emb_dim, shared_dim, emb_dim, beta, linear_beta)
            for _ in range(num_shared_experts)
        )

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        z = self.latent_norm(self.down_proj(x)).reshape(B * S, -1)
        probs = torch.sigmoid(self.router(z).float())
        _, indices = (probs + self.router_bias.float()).topk(self.experts_per_token, dim=-1)
        # renormalise the *sigmoid* weights of the picked experts (the bias
        # only steers the choice, as in noaux_tc)
        picked = probs.gather(-1, indices)
        weights = picked / picked.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        out = torch.zeros_like(z)
        for e, expert in enumerate(self.experts):
            rows, slot = (indices == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            contrib = expert(z[rows]) * weights[rows, slot].unsqueeze(-1).to(z.dtype)
            out.index_add_(0, rows, contrib)

        result = self.up_proj(out.reshape(B, S, -1))
        for shared in self.shared_experts:
            result = result + shared(x)
        return result


def attn_res_mix(
    sources: list[torch.Tensor], query: torch.Tensor, norm: nn.RMSNorm
) -> torch.Tensor:
    """softmax over depth: h = sum_i alpha_i v_i, alpha from w . rmsnorm(v)."""
    stacked = torch.stack(sources)                       # (N, B, S, D)
    scores = (norm(stacked) * query).sum(dim=-1)         # (N, B, S)
    alpha = scores.float().softmax(dim=0).to(stacked.dtype)
    return (alpha.unsqueeze(-1) * stacked).sum(dim=0)


class KimiK31Block(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        attn: nn.Module,
        ffn: nn.Module,
        use_attn_res: bool,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.norm1 = nn.RMSNorm(emb_dim, eps=norm_eps)
        self.norm2 = nn.RMSNorm(emb_dim, eps=norm_eps)
        self.use_attn_res = use_attn_res
        self.attn_res_norm = nn.RMSNorm(emb_dim, eps=norm_eps)
        self.attn_res_q_attn = nn.Parameter(torch.zeros(emb_dim))
        self.attn_res_q_mlp = nn.Parameter(torch.zeros(emb_dim))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        completed: tuple[torch.Tensor, ...] = (),
        has_partial: bool = True,
    ) -> torch.Tensor:
        if not self.use_attn_res:
            x = x + self.attn(self.norm1(x), cos, sin, attention_mask)
            x = x + self.ffn(self.norm2(x))
            return x

        # x is the current block's partial sum (zeros at a block boundary)
        sources = list(completed) + ([x] if has_partial else [])
        h = attn_res_mix(sources, self.attn_res_q_attn, self.attn_res_norm)
        partial = x + self.attn(self.norm1(h), cos, sin, attention_mask)
        h = attn_res_mix([*completed, partial], self.attn_res_q_mlp, self.attn_res_norm)
        return partial + self.ffn(self.norm2(h))


def _is_mla_layer(idx: int, num_layers: int) -> bool:
    """1-based layers 4, 8, ... plus the final layer are mla; the rest kda."""
    return (idx + 1) % 4 == 0 or idx == num_layers - 1


class KimiK31(nn.Module):
    def __init__(
        self,
        num_layers: int = 28,
        emb_dim: int = 2048,
        num_heads: int = 16,
        head_dim: int = 128,
        qk_nope_dim: int = 128,
        qk_rope_dim: int = 64,
        v_head_dim: int = 128,
        q_lora_rank: int = 512,
        kv_lora_rank: int = 256,
        vocab_size: int = 163_840,
        latent_dim: int = 1024,
        moe_dim: int = 512,
        num_experts: int = 56,
        experts_per_token: int = 4,
        num_shared_experts: int = 1,
        shared_dim: int = 1024,
        dense_dim: int = 8192,
        first_dense_layers: int = 1,
        situ_beta: float = 4.0,
        situ_linear_beta: float = 25.0,
        norm_eps: float = 1e-5,
        rope_base: float = 10_000,
        attn_res: bool = True,
        attn_res_block_size: int = 6,
        dsa_topk: int | None = None,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_attn_res = attn_res
        self.attn_res_block_size = attn_res_block_size
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.rope = KimiRotaryEmbedding(qk_rope_dim, base=rope_base)

        blocks = []
        for i in range(num_layers):
            if _is_mla_layer(i, num_layers):
                attn: nn.Module = MultiHeadLatentAttention(
                    emb_dim, num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
                    q_lora_rank, kv_lora_rank, norm_eps, dsa_topk=dsa_topk,
                )
            else:
                attn = KimiDeltaAttention(emb_dim, num_heads, head_dim, norm_eps)
            if i < first_dense_layers:
                ffn: nn.Module = SiTUMLP(
                    emb_dim, dense_dim, emb_dim, situ_beta, situ_linear_beta
                )
            else:
                ffn = LatentMoE(
                    emb_dim, latent_dim, moe_dim, num_experts, experts_per_token,
                    num_shared_experts, shared_dim, situ_beta, situ_linear_beta, norm_eps,
                )
            blocks.append(KimiK31Block(emb_dim, attn, ffn, attn_res, norm_eps))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(emb_dim, eps=norm_eps)
        self.attn_res_norm = nn.RMSNorm(emb_dim, eps=norm_eps)
        self.attn_res_q_final = nn.Parameter(torch.zeros(emb_dim))
        # kimi does not tie embeddings
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
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.RMSNorm) and module.weight is not None:
                nn.init.ones_(module.weight)
        for name, param in self.named_parameters():
            if "attn_res_q" in name or name == "attn_res_q_final":
                nn.init.zeros_(param)
            elif name.endswith("A_log"):
                param.data.fill_(math.log(8.0))
            elif name.endswith("dt_bias"):
                nn.init.zeros_(param)
        for name, buf in self.named_buffers():
            if name.endswith("router_bias"):
                buf.zero_()

        residual_scale = (2 * len(self.blocks)) ** -0.5
        for block in self.blocks:
            block.attn.o_proj.weight.data.mul_(residual_scale)
            ffn = block.ffn
            if isinstance(ffn, LatentMoE):
                ffn.up_proj.weight.data.mul_(residual_scale)
                for expert in (*ffn.experts, *ffn.shared_experts):
                    expert.down_proj.weight.data.mul_(residual_scale)
            else:
                ffn.down_proj.weight.data.mul_(residual_scale)

    def tp_plan(self, tp_size: int, vocab_parallel: bool = True) -> dict[str, list[nn.Module]]:
        """
        vocab only. the kda scan, mla's latent bottleneck and expert
        dispatch do not fit the tp engine's (b, s, d) region vocabulary,
        so block weights stay replicated over tp and dp/fsdp does the
        sharding. sequence parallel must be off for this model (there are
        no regions for it to shard between); the trainer config for kimi
        has to say so explicitly.
        """
        return {
            "column": [], "row": [], "regions": [], "shared": [],
            "embeddings": [self.token_emb],
            "heads": [self.lm_head],
            "vocab": [self.token_emb, self.lm_head] if vocab_parallel else [],
            "seq_local": [],
        }

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

        emb = self.token_emb(x)
        cos, sin = self.rope(position_ids, cacheable=cacheable)
        hidden_states: list[torch.Tensor] = [emb] if return_hidden_states else []
        use_checkpoint = self.activation_checkpointing and torch.is_grad_enabled()

        if not self.use_attn_res:
            h = emb
            for block in self.blocks:
                if use_checkpoint:
                    h = checkpoint(block, h, cos, sin, attention_mask, use_reentrant=False)
                else:
                    h = block(h, cos, sin, attention_mask)
                if return_hidden_states:
                    hidden_states.append(h)
        else:
            completed: list[torch.Tensor] = [emb]
            partial = torch.zeros_like(emb)
            for i, block in enumerate(self.blocks):
                has_partial = (i % self.attn_res_block_size) != 0
                args = (partial, cos, sin, attention_mask, tuple(completed), has_partial)
                if use_checkpoint:
                    partial = checkpoint(block, *args, use_reentrant=False)
                else:
                    partial = block(*args)
                if return_hidden_states:
                    hidden_states.append(partial)
                if (i + 1) % self.attn_res_block_size == 0 or i == len(self.blocks) - 1:
                    completed.append(partial)
                    partial = torch.zeros_like(emb)
            h = attn_res_mix(completed, self.attn_res_q_final, self.attn_res_norm)

        if return_final_hidden:
            return self.norm(h)

        logits = self.lm_head(self.norm(h))
        if return_hidden_states:
            return logits, hidden_states
        return logits


if __name__ == "__main__":
    with torch.device("meta"):
        model = KimiK31()
    total = sum(p.numel() for p in model.parameters())
    # active: everything except the token embedding and the routed
    # experts a token does not pick
    inactive = model.token_emb.weight.numel()
    for block in model.blocks:
        ffn = block.ffn
        if isinstance(ffn, LatentMoE):
            per_expert = sum(p.numel() for p in ffn.experts[0].parameters())
            inactive += per_expert * (len(ffn.experts) - ffn.experts_per_token)
    mla = sum(isinstance(b.attn, MultiHeadLatentAttention) for b in model.blocks)
    print(f"kimi-k3.1 scaled: {total / 1e9:.2f}B total, "
          f"{(total - inactive) / 1e9:.2f}B active (excl. token embedding)")
    print(f"layers: {len(model.blocks)} ({len(model.blocks) - mla} kda + {mla} mla)")


