"""
cpu tests for the olmo3 and kimi-k3.1 model files.

run: uv run tests/models.py

covers the trainer contract both models have to satisfy (blocks / norm /
lm_head.weight / init_weights / tp_plan / activation_checkpointing /
return_final_hidden), the architecture invariants that make them not just
qwen3 with new names (olmo post-norm + 3:1 sliding:full + yarn; kimi
kda/mla hybrid + latent moe + situ-glu + block attn-res), and causality
of the kda scan / sliding window, which are the two places a silent
future leak would still train.
"""

import math

import torch
import torch.nn.functional as F

from torchure.core.meta import materialize
from torchure.models.builder import build_model
from torchure.models.kimi_k31.kimi_k31 import (
    KimiK31,
    LatentMoE,
    MultiHeadLatentAttention,
    SiTUMLP,
    _kda_reference,
    _kda_scan,
    situ_glu,
)
from torchure.models.olmo3.olmo3 import Olmo3

OLMO_CFG = {
    "num_layers": 8, "num_heads": 4, "num_kv_heads": 2, "emb_dim": 64,
    "head_dim": 16, "vocab_size": 128, "ffn_dim": 128, "sliding_window": 4,
}
KIMI_CFG = {
    "num_layers": 8, "emb_dim": 64, "num_heads": 4, "head_dim": 16,
    "qk_nope_dim": 16, "qk_rope_dim": 8, "v_head_dim": 16,
    "q_lora_rank": 32, "kv_lora_rank": 16, "vocab_size": 128,
    "latent_dim": 32, "moe_dim": 32, "num_experts": 8,
    "experts_per_token": 2, "num_shared_experts": 1, "shared_dim": 32,
    "dense_dim": 128, "attn_res_block_size": 3,
}
SEQ = 12


def build(cls, cfg, seed=0):
    torch.manual_seed(seed)
    model = cls(**cfg)
    model.init_weights()
    return model


def check_trainer_contract(model, vocab, name):
    ids = torch.randint(0, vocab, (2, SEQ))
    logits = model(ids)
    assert logits.shape == (2, SEQ, vocab), (name, logits.shape)
    assert torch.isfinite(logits).all(), name

    hidden = model(ids, return_final_hidden=True)
    assert hidden.shape == (2, SEQ, model.emb_dim), (name, hidden.shape)

    logits2, states = model(ids, return_hidden_states=True)
    assert len(states) == len(model.blocks) + 1, name
    assert torch.allclose(logits, logits2), name

    mask = torch.ones(2, SEQ, dtype=torch.bool)
    mask[0, -3:] = False
    masked = model(ids, attention_mask=mask)
    assert torch.isfinite(masked).all(), name

    loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), ids[:, 1:].reshape(-1))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, (name, missing)
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p.grad).all()]
    assert not bad, (name, bad)
    model.zero_grad(set_to_none=True)

    model.activation_checkpointing = True
    out = model(ids)
    out.float().mean().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    ), name
    model.activation_checkpointing = False
    model.zero_grad(set_to_none=True)
    print(f"{name}: trainer contract ok (loss {loss.item():.3f})")


def check_meta_init(cls, cfg, name):
    with torch.device("meta"):
        model = cls(**cfg)
    materialize(model, torch.device("cpu"))
    model.init_weights()
    for n, p in model.named_parameters():
        assert not p.is_meta and torch.isfinite(p).all(), (name, n)
    for n, b in model.named_buffers():
        assert not b.is_meta and torch.isfinite(b).all(), (name, n)
    print(f"{name}: meta build + materialize + init_weights ok")


def test_olmo():
    model = build(Olmo3, OLMO_CFG)
    check_trainer_contract(model, OLMO_CFG["vocab_size"], "olmo3")
    check_meta_init(Olmo3, OLMO_CFG, "olmo3")

    sliding = [b.attn.sliding_window is not None for b in model.blocks]
    assert sliding == [True, True, True, False] * 2, sliding
    assert model.lm_head.weight is not model.token_emb.weight
    assert abs(model.rope.attention_factor - (0.1 * math.log(8.0) + 1.0)) < 1e-6
    plan = model.tp_plan(2)
    assert len(plan["regions"]) == 2 * len(model.blocks)
    assert len(plan["column"]) == 5 * len(model.blocks)
    assert plan["vocab"] == [model.token_emb, model.lm_head]

    # sliding locality: a token 6 back must not move layer-0's output at t
    # (window 4), while the full-attention layer further down may see it.
    layer0 = build(Olmo3, OLMO_CFG, seed=3).blocks[0]
    torch.manual_seed(3)
    x = torch.randn(1, SEQ, OLMO_CFG["emb_dim"])
    pos = torch.arange(SEQ).unsqueeze(0)
    cos, sin = model.rope(pos)
    changed = x.clone()
    changed[0, 0] += 5.0
    out_a = layer0(x, cos, sin)
    out_b = layer0(changed, cos, sin)
    assert torch.allclose(out_a[0, -1], out_b[0, -1], atol=1e-5), "sliding leak"
    assert not torch.allclose(out_a[0, 1], out_b[0, 1]), "sliding window inert"
    print("olmo3: 3:1 sliding:full, untied head, yarn, tp_plan, sliding locality ok")


def test_kimi():
    model = build(KimiK31, KIMI_CFG)
    check_trainer_contract(model, KIMI_CFG["vocab_size"], "kimi_k31")
    check_meta_init(KimiK31, KIMI_CFG, "kimi_k31")

    attn_types = [isinstance(b.attn, MultiHeadLatentAttention) for b in model.blocks]
    assert attn_types == [False, False, False, True, False, False, False, True], attn_types
    assert isinstance(model.blocks[0].ffn, SiTUMLP), "first layer must be dense"
    assert all(isinstance(b.ffn, LatentMoE) for b in model.blocks[1:])
    assert model.lm_head.weight is not model.token_emb.weight

    # situ-glu is bounded by beta * linear_beta = 100, whatever the inputs
    big = situ_glu(torch.full((4,), 1e4), torch.full((4,), -1e4), 4.0, 25.0)
    assert big.abs().max() <= 100.0

    plan = model.tp_plan(2)
    assert plan["regions"] == [] and plan["vocab"] == [model.token_emb, model.lm_head]

    # attn-res off must also work (plain pre-norm residual path)
    plain = build(KimiK31, {**KIMI_CFG, "attn_res": False}, seed=1)
    assert torch.isfinite(plain(torch.randint(0, 128, (1, SEQ)))).all()
    # dsa indexer on must also work
    sparse = build(KimiK31, {**KIMI_CFG, "dsa_topk": 3}, seed=1)
    assert torch.isfinite(sparse(torch.randint(0, 128, (1, SEQ)))).all()

    # kda causality: tokens after t cannot change outputs at or before t
    block = build(KimiK31, KIMI_CFG, seed=5).blocks[0]
    torch.manual_seed(5)
    x = torch.randn(1, SEQ, KIMI_CFG["emb_dim"])
    pos = torch.arange(SEQ).unsqueeze(0)
    cos, sin = model.rope(pos)
    changed = x.clone()
    changed[0, 8:] += 5.0
    out_a = block(x, cos, sin, None, (x,), True)
    out_b = block(changed, cos, sin, None, (x,), True)
    assert torch.allclose(out_a[0, :8], out_b[0, :8], atol=1e-5), "kda future leak"
    assert not torch.allclose(out_a[0, 8:], out_b[0, 8:]), "kda change inert"
    print("kimi_k31: hybrid kda/mla, latent moe, situ bound, attn-res, dsa, kda causality ok")


def test_kda_chunkwise():
    """the chunkwise closed form is the step-by-step recurrence, values and grads."""
    torch.manual_seed(0)
    B, S, H, D = 2, 150, 3, 8
    q = F.normalize(torch.randn(B, S, H, D), dim=-1)
    k = F.normalize(torch.randn(B, S, H, D), dim=-1)
    v = torch.randn(B, S, H, D)
    beta = torch.rand(B, S, H)
    # mild decay, and the worst case: half the channels pinned at the -5
    # floor, where exp(g_t) / exp(g_i) would overflow within one chunk
    mild = -2 * torch.rand(B, S, H, D)
    harsh = torch.where(torch.rand(B, S, H, D) < 0.5, torch.full((B, S, H, D), -5.0), mild * 1e-3)
    for name, log_alpha in (("mild", mild), ("floor", harsh)):
        ref = _kda_reference(q.double(), k.double(), v.double(), log_alpha.double(), beta.double())
        for chunk in (7, 32, 64, 1000):  # ragged tail, sub-blocks, the default, one chunk
            out = _kda_scan(q, k, v, log_alpha, beta, chunk=chunk)
            err = (out.double() - ref).abs().max().item()
            assert err < 5e-5, f"kda {name} chunk={chunk}: max err {err:.2e}"

    ins = [t.clone().requires_grad_() for t in (q, k, v, mild, beta)]

    def grads(fn):
        for t in ins:
            t.grad = None
        fn(*ins).pow(2).sum().backward()
        return [t.grad.clone() for t in ins]

    want = grads(_kda_reference)
    for chunk in (16, 64):
        got = grads(lambda *a, c=chunk: _kda_scan(*a, chunk=c))
        for g, w in zip(got, want, strict=True):
            assert torch.allclose(g, w, rtol=1e-3, atol=1e-5), f"kda chunk={chunk}: grads differ"
    print("kimi_k31: chunkwise kda == step-by-step recurrence (values, grads, decay floor) ok")


def test_moe_dispatch():
    """the sorted, grouped expert bank is the obvious per-expert loop."""
    torch.manual_seed(0)
    moe = LatentMoE(
        emb_dim=16, latent_dim=8, moe_dim=12, num_experts=6, experts_per_token=2,
        num_shared_experts=1, shared_dim=8, beta=4.0, linear_beta=25.0,
    )
    # a bias that keeps experts 4 and 5 from ever being picked
    moe.router_bias[4:] = -10.0
    x = torch.randn(2, 9, 16, requires_grad=True)
    out = moe(x)

    z = moe.latent_norm(moe.down_proj(x)).reshape(18, -1)
    probs = torch.sigmoid(moe.router(z))
    _, indices = (probs + moe.router_bias).topk(2, dim=-1)
    picked = probs.gather(-1, indices)
    weights = picked / picked.sum(dim=-1, keepdim=True)
    bank = moe.experts
    naive = torch.zeros_like(z)
    for n in range(18):
        for slot in range(2):
            e = indices[n, slot]
            hidden = situ_glu(bank.gate.weight[e] @ z[n], bank.up.weight[e] @ z[n], 4.0, 25.0)
            naive[n] += weights[n, slot] * (bank.down.weight[e] @ hidden)
    want = moe.up_proj(naive.reshape(2, 9, -1)) + moe.shared_experts[0](x)
    assert torch.allclose(out, want, atol=1e-5), "grouped dispatch != per-expert loop"

    assert moe.load.sum() == 18 * 2 and (moe.load[4:] == 0).all()
    out.sum().backward()
    # an expert no token picked still leaves its params with a grad (zeros)
    for linear in (bank.gate, bank.up, bank.down):
        assert linear.weight.grad is not None and (linear.weight.grad[4:] == 0).all()
        assert linear.weight.grad[:4].abs().sum() > 0
    with torch.no_grad():
        moe(x)
    assert moe.load.sum() == 18 * 2, "a no-grad forward must not count toward the load"

    # the balancing rule moves the bias toward the starved experts
    model = build(KimiK31, KIMI_CFG, seed=3)
    layer = model.moe_layers()[0]
    layer.load.copy_(torch.tensor([9.0, 0, 3, 3, 3, 3, 3, 0]))
    stats = model.balance_step()
    rate = model.router_bias_rate
    assert torch.allclose(layer.router_bias, rate * torch.tensor([-1.0, 1, 0, 0, 0, 0, 0, 1]))
    assert layer.load.sum() == 0 and stats["moe/dead_experts"] >= 2
    assert stats["moe/max_load"] == 3.0 and stats["moe/min_load"] == 0.0
    print("kimi_k31: grouped expert dispatch, load counting, bias balancing ok")


def test_registry_and_counts():
    for name, cfg in (("olmo3", OLMO_CFG), ("kimi_k31", KIMI_CFG)):
        model = build_model(name, cfg)
        assert len(model.blocks) == cfg["num_layers"], name
    with torch.device("meta"):
        olmo = Olmo3()
        kimi = KimiK31()
    olmo_total = sum(p.numel() for p in olmo.parameters()) / 1e9
    assert 7.0 < olmo_total < 7.6, olmo_total
    assert sum(b.attn.sliding_window is not None for b in olmo.blocks) == 24
    kimi_total = sum(p.numel() for p in kimi.parameters()) / 1e9
    assert 3.5 < kimi_total < 4.5, kimi_total
    print(f"registry ok; defaults: olmo3 {olmo_total:.2f}B, kimi_k31 {kimi_total:.2f}B")


if __name__ == "__main__":
    test_olmo()
    test_kimi()
    test_kda_chunkwise()
    test_moe_dispatch()
    test_registry_and_counts()
    print("all model tests passed")
