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
    aux-free: a router bias buffer steers the top-k choice and is moved
    by a balancing rule, not by gradient (see balance_step)
  - situ-glu: beta1*tanh(g/beta1) * sigmoid(g) * beta2*tanh(u/beta2),
    beta1=4, beta2=25, so outputs are bounded by 100
  - block attention residuals: sublayer outputs sum into a per-block
    partial; before each sublayer the input is a softmax mix (learned
    per-sublayer pseudo-query w over rmsnormed sources) of the embedding,
    the completed block sums, and the current partial
  - first layer is dense (first_k_dense_replace=1 in the real config)

pure pytorch throughout. the kda recurrence is solved chunkwise (64 steps in
closed form at a time, see _kda_chunk; compiler-disabled, like the tp
collectives) and the routed experts are one stacked bank driven by a sort and
a grouped matmul (models/moe.py). the scaled config in configs/ turns compile
off.

trainer contract, same as qwen3: `blocks`, `norm`, `lm_head.weight`,
`init_weights()`, `tp_plan()`, `activation_checkpointing`,
forward(..., return_final_hidden=True). tp is vocab-only (see tp_plan):
the current tp engine shards (b, s, d) regions, and neither the kda scan,
mla's latent bottleneck, nor expert dispatch fit that vocabulary. the
scaling paths are fsdp/dp over `blocks` and expert parallel over the routed
experts (`ep_plan`, parallelism/expert_parallel.py).

load balancing is aux-loss free: `balance_step` (called by the trainer after
every optimizer step) nudges each layer's router bias toward the experts
that got fewer tokens than their share, and reports the spread.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torchure.models.layers import RotaryEmbedding, apply_rotary, positions
from torchure.models.moe import ExpertBank, GroupedLinear, balance_stats, sort_by_expert


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


# timesteps the kda scan solves in closed form at once
KDA_CHUNK = 64


def _kda_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    log_alpha: torch.Tensor, beta: torch.Tensor,
) -> torch.Tensor:
    """
    the gated delta rule one timestep at a time, exactly as written:

        S <- (I - beta k k^T) diag(alpha) S + beta k v^T,   o = S^T q

    q/k/v (B, S, H, D), log_alpha (B, S, H, D) <= 0, beta (B, S, H). slow and
    memory hungry under autograd; `_kda_scan` is what the model runs, and is
    tested against this.
    """
    B, S, H, D = q.shape
    state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    q = q * (D ** -0.5)
    alpha = log_alpha.exp()
    outs = []
    for t in range(S):
        k_t, v_t = k[:, t], v[:, t]                       # (B, H, D)
        state = state * alpha[:, t].unsqueeze(-1)         # per-channel decay
        pred = (state * k_t.unsqueeze(-1)).sum(dim=-2)    # S^T k_t
        delta = beta[:, t].unsqueeze(-1) * (v_t - pred)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * q[:, t].unsqueeze(-1)).sum(dim=-2))
    return torch.stack(outs, dim=1)


def _kda_chunk(
    state: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    log_alpha: torch.Tensor, beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    C timesteps of the recurrence in closed form: (state after, outputs).
    q/k/v/log_alpha (B, H, C, D), beta (B, H, C), state (B, H, D, D).

    every step adds a rank-1 write k_t u_t^T to the decayed state, with
    u_t = beta_t (v_t - S_{t-1}'^T k_t). unrolling, with g_t the log decay
    accumulated up to step t and w[t, i] = exp(g_t - g_i) the per-channel
    decay a write at step i has suffered by step t:

        S_t = exp(g_t) S_0 + sum_{i<=t} (w[t, i] k_i) u_i^T

    substituting that into u_t gives a unit lower-triangular system for all
    C of them at once,

        (I + diag(beta) tril(A, -1)) U = diag(beta) (V - (exp(g) K) S_0),
        A[t, i] = sum_d k_t[d] w[t, i, d] k_i[d]

    and the outputs and the final state follow from U. w is formed pairwise,
    exp of a difference that is always <= 0, rather than as exp(g_t) /
    exp(g_i): a channel that decays hard would overflow the division.
    """
    g = log_alpha.cumsum(dim=2)                                   # (B, H, C, D)
    # w[t, i, d] for i <= t; the i > t half is masked off below
    w = (g.unsqueeze(3) - g.unsqueeze(2)).clamp(max=0).exp()      # (B, H, C, C, D)
    decay = g.exp()

    a = torch.einsum("bhtd,bhtid,bhid->bhti", k, w, k).tril(-1)
    system = a * beta.unsqueeze(-1)                               # rows scaled by beta_t
    rhs = beta.unsqueeze(-1) * (v - (k * decay) @ state)
    u = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)

    aq = torch.einsum("bhtd,bhtid,bhid->bhti", q, w, k).tril()
    out = (q * decay) @ state + aq @ u
    carried = (w[:, :, -1] * k).transpose(-1, -2) @ u             # writes, decayed to the end
    return decay[:, :, -1].unsqueeze(-1) * state + carried, out


@torch.compiler.disable
def _kda_scan(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    log_alpha: torch.Tensor, beta: torch.Tensor,
    chunk: int = KDA_CHUNK,
    checkpoint_chunks: bool = True,
) -> torch.Tensor:
    """
    the gated delta rule over a sequence, `chunk` timesteps at a time (see
    _kda_chunk). q/k/v (B, S, H, D), log_alpha (B, S, H, D) <= 0, beta
    (B, S, H). the state is (B, H, D, D), kept in fp32.

    S / chunk python steps instead of S. each chunk builds a (C, C, D)
    pairwise-decay tensor per head, so memory grows with the chunk size;
    with grad on, chunks run under activation checkpointing by default, so
    only the state at each chunk boundary is kept for backward, at the price
    of running each chunk a second time.
    """
    B, S, H, D = q.shape
    state = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    # (B, S, H, ...) -> (B, H, S, ...)
    q = q.float().transpose(1, 2) * (D ** -0.5)
    k, v, log_alpha = (t.float().transpose(1, 2) for t in (k, v, log_alpha))
    beta = beta.float().transpose(1, 2)
    use_checkpoint = checkpoint_chunks and torch.is_grad_enabled() and S > chunk and any(
        t.requires_grad for t in (q, k, v, log_alpha, beta)
    )
    outs = []
    for s in range(0, S, chunk):
        args = (state, *(t[:, :, s:s + chunk] for t in (q, k, v, log_alpha, beta)))
        if use_checkpoint:
            state, out = checkpoint(_kda_chunk, *args, use_reentrant=False)
        else:
            state, out = _kda_chunk(*args)
        outs.append(out)
    return torch.cat(outs, dim=2).transpose(1, 2)


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
        log_alpha = log_alpha.clamp_min(self.gate_lower_bound)
        beta = torch.sigmoid(self.beta_proj(x))

        out = _kda_scan(q, k, v, log_alpha, beta).to(x.dtype)  # (B, S, H, D)
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
        self.beta = beta
        self.linear_beta = linear_beta
        self.down_proj = nn.Linear(emb_dim, latent_dim, bias=False)
        self.latent_norm = nn.RMSNorm(latent_dim, eps=norm_eps)
        self.up_proj = nn.Linear(latent_dim, emb_dim, bias=False)
        self.router = nn.Linear(latent_dim, num_experts, bias=False)
        # noaux_tc bias: moved by KimiK31.balance_step, not by gradient
        self.register_buffer("router_bias", torch.zeros(num_experts))
        # tokens each expert got since the last balance_step
        self.register_buffer("load", torch.zeros(num_experts), persistent=False)
        self.experts = ExpertBank(num_experts, latent_dim, moe_dim, latent_dim, self._situ)
        self.shared_experts = nn.ModuleList(
            SiTUMLP(emb_dim, shared_dim, emb_dim, beta, linear_beta)
            for _ in range(num_shared_experts)
        )

    def _situ(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return situ_glu(gate, up, self.beta, self.linear_beta)

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

        # the (token, expert) pairs, grouped by expert
        token, slot, counts = sort_by_expert(indices, self.num_experts)
        if torch.is_grad_enabled():
            # only passes that train: not eval, and not the no-grad first
            # pass of an activation-checkpointed block (its re-run counts)
            self.load += counts
        routed = self.experts(z[token], counts)                    # (N * k, latent)
        routed = routed * weights[token, slot].unsqueeze(-1).to(routed.dtype)
        out = torch.zeros_like(z).index_add_(0, token, routed.to(z.dtype))

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
        router_bias_rate: float = 1e-3,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        # how far balance_step moves a router bias per optimizer step
        self.router_bias_rate = router_bias_rate
        self.use_attn_res = attn_res
        self.attn_res_block_size = attn_res_block_size
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        # plain rope, on the mla rope dims only (kda is nope)
        self.rope = RotaryEmbedding(qk_rope_dim, base=rope_base)

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
            elif isinstance(module, GroupedLinear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
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
            if name.endswith(("router_bias", "load")):
                buf.zero_()

        residual_scale = (2 * len(self.blocks)) ** -0.5
        for block in self.blocks:
            block.attn.o_proj.weight.data.mul_(residual_scale)
            ffn = block.ffn
            if isinstance(ffn, LatentMoE):
                ffn.up_proj.weight.data.mul_(residual_scale)
                ffn.experts.down.weight.data.mul_(residual_scale)
                for expert in ffn.shared_experts:
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

    def moe_layers(self) -> list[LatentMoE]:
        return [block.ffn for block in self.blocks if isinstance(block.ffn, LatentMoE)]

    def ep_plan(self) -> list[ExpertBank]:
        """
        the expert banks expert parallel shards across the mesh's ep axis.
        everything else (routers, shared experts, attention) stays dense.
        """
        return [moe.experts for moe in self.moe_layers()]

    @torch.no_grad()
    def balance_step(self, reduce=None) -> dict[str, torch.Tensor]:
        """
        aux-loss-free load balancing, once per optimizer step: every
        layer's router bias moves `router_bias_rate` toward the experts that
        got fewer tokens than the mean since the last call, away from the
        ones that got more. returns the spread (see moe.balance_stats).

        `reduce(tensor)` sums a tensor in place over every rank that saw
        different data, so all ranks apply the same update and the bias
        stays replicated.
        """
        layers = self.moe_layers()
        if not layers:
            return {}
        load = torch.stack([moe.load for moe in layers])
        if reduce is not None:
            load = reduce(load)
        step = self.router_bias_rate * torch.sign(load.mean(dim=-1, keepdim=True) - load)
        for moe, delta in zip(layers, step, strict=True):
            moe.router_bias += delta.to(moe.router_bias.dtype)
            moe.load.zero_()
        return balance_stats(load)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_final_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        position_ids, attention_mask, cacheable = positions(x, attention_mask)

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
            per_expert = sum(p.numel() for p in ffn.experts.parameters()) // ffn.num_experts
            inactive += per_expert * (ffn.num_experts - ffn.experts_per_token)
    mla = sum(isinstance(b.attn, MultiHeadLatentAttention) for b in model.blocks)
    print(f"kimi-k3.1 scaled: {total / 1e9:.2f}B total, "
          f"{(total - inactive) / 1e9:.2f}B active (excl. token embedding)")
    print(f"layers: {len(model.blocks)} ({len(model.blocks) - mla} kda + {mla} mla)")


