"""
cpu tests for the experimental architectures: bidirectional qwen3 + the
masked diffusion objective, the looped transformer, and the qwen3 + siglip vlm.

run: uv run tests/experimental_archs.py

the single-process tests check what makes each arch what it is (attention
really is bidirectional, the loop really reuses weights, image features
really land in the placeholders). the last test runs the looped model and
the vlm under fsdp on 2 gloo ranks against single-process grads (the looped
core is one fsdp unit applied many times per forward, and the vlm's vision
tower lives in the root unit, both shapes fsdp never saw before), plus the
weighted fused loss on the vocab-parallel path.
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image

from torchure.core.mesh import Mesh
from torchure.core.meta import materialize
from torchure.core.placement import Shard
from torchure.dataloader.vlm import ImageTextCollator
from torchure.loss.fused_linear_ce import fused_linear_cross_entropy
from torchure.models.looped.looped import LoopedQwen3
from torchure.models.qwen3.qwen3 import Qwen3
from torchure.models.vlm.vlm import Qwen3VL
from torchure.objectives.autoregressive import ARObjective
from torchure.objectives.masked_diffusion import MaskedDiffusionObjective
from torchure.parallelism.fsdp2 import FSDP

VOCAB = 64
SEQ = 12
QWEN = {"num_heads": 4, "num_kv_heads": 2, "emb_dim": 32, "head_dim": 8, "vocab_size": VOCAB}
LOOPED = {**QWEN, "prelude_layers": 1, "core_layers": 2, "num_loops": 3, "coda_layers": 1}
VISION = {
    "image_size": 16, "patch_size": 4, "emb_dim": 24, "num_layers": 2,
    "num_heads": 2, "ffn_dim": 48,
}
IMAGE_TOKEN = VOCAB - 1
VLM = {**QWEN, "num_layers": 2, "image_token_id": IMAGE_TOKEN, "vision": VISION, "pixel_shuffle": 2}


def build(cls, cfg, seed=0):
    torch.manual_seed(seed)
    model = cls(**cfg)
    model.init_weights()
    return model


def ids(batch=2, seed=1):
    return torch.randint(0, VOCAB - 2, (batch, SEQ), generator=torch.Generator().manual_seed(seed))


def grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


# --- diffusion -------------------------------------------------------------


def test_bidirectional_attention():
    """a future token moves earlier outputs only when causal=False."""
    x = ids()
    y = x.clone()
    y[:, -1] = (y[:, -1] + 1) % VOCAB
    for causal in (True, False):
        model = build(Qwen3, {**QWEN, "num_layers": 2, "causal": causal})
        moved = (model(x)[:, :-1] - model(y)[:, :-1]).abs().max()
        assert (moved > 1e-4) == (not causal), (causal, moved)
        # the padding-mask path must agree with the flash path on a full mask
        full = torch.ones_like(x, dtype=torch.bool)
        torch.testing.assert_close(model(x, attention_mask=full), model(x), rtol=1e-4, atol=1e-5)


def test_diffusion_noise():
    obj = MaskedDiffusionObjective(mask_token_id=VOCAB - 1, fused_linear_ce=False)
    x = torch.randint(0, VOCAB - 2, (64, 256))
    labels = x.clone()
    labels[:, :8] = obj.ignore_index  # e.g. padding: never masked, never scored
    torch.manual_seed(0)
    noised, targets, t = obj.noise(x, labels)
    masked = noised == obj.mask_token_id
    assert not masked[:, :8].any(), "ignored positions must stay clean"
    assert torch.equal(targets != obj.ignore_index, masked), "targets exactly on the masked positions"
    assert torch.equal(noised[~masked], x[~masked])
    # the masked fraction of each row tracks its t
    frac = masked[:, 8:].float().mean(1)
    assert (frac - t.squeeze(1)).abs().max() < 0.15, (frac, t)


def test_diffusion_loss():
    """the loss is the 1/t-weighted masked nll over all tokens, and trains every param."""
    model = build(Qwen3, {**QWEN, "num_layers": 2, "causal": False})
    obj = MaskedDiffusionObjective(mask_token_id=VOCAB - 1, fused_linear_ce=False)
    x = ids()
    batch = {"input_ids": x, "labels": x}

    torch.manual_seed(3)
    loss = obj.compute_loss(model, batch)
    torch.manual_seed(3)
    noised, targets, t = obj.noise(x, x)
    nll = F.cross_entropy(model(noised).flatten(0, 1), targets.flatten(), reduction="none")
    expected = (nll.view_as(x) / t).sum() / x.numel()
    torch.testing.assert_close(loss, expected)

    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_diffusion_generate():
    """every mask gets filled, `gen_len / steps` per step, and the prompt is never touched."""
    from torchure.inference.diffusion_generate import diffusion_generate

    mask_id = VOCAB - 1
    model = build(Qwen3, {**QWEN, "num_layers": 2, "causal": False})
    revealed = []
    model.register_forward_pre_hook(lambda m, args: revealed.append(int((args[0] == mask_id).sum())))
    prompt = ids(batch=2)[:, :4]
    out = diffusion_generate(model, prompt, gen_len=8, steps=4, mask_id=mask_id, temperature=1.0)
    assert out.shape == (2, 8) and (out != mask_id).all()
    # masks left at each of the 4 forwards, both rows: 8, 6, 4, 2
    assert revealed == [16, 12, 8, 4], revealed


# --- looped ----------------------------------------------------------------


def test_looped_reuses_weights():
    """
    LoopedQwen3 == a qwen3 whose block list repeats the core, same weights.
    also: params count unique blocks only, and the state_dict has no
    duplicate (aliased) entries.
    """
    model = build(LoopedQwen3, LOOPED)
    p, c, n = LOOPED["prelude_layers"], LOOPED["core_layers"], LOOPED["num_loops"]
    assert len(model.transformer_blocks()) == p + c + LOOPED["coda_layers"]
    assert model.depth == p + c * n + LOOPED["coda_layers"]

    sd = model.state_dict()
    ptrs = [v.data_ptr() for k, v in sd.items() if k != "lm_head.weight"]
    assert len(ptrs) == len(set(ptrs)), "a tensor appears twice in the state_dict"

    blocks = model.transformer_blocks()
    unrolled = blocks[:p] + blocks[p:p + c] * n + blocks[p + c:]
    flat = Qwen3(num_layers=len(unrolled), **QWEN)
    flat.load_state_dict(
        {k: v for k, v in sd.items() if not k.startswith("blocks.")}, strict=False
    )
    flat.blocks = torch.nn.ModuleList(unrolled)
    x = ids()
    torch.testing.assert_close(model(x), flat(x))

    # more loops at inference: a different (deeper) function, same weights
    model.core.num_loops = n + 2
    assert (model(x) - flat(x)).abs().max() > 1e-4


def test_looped_checkpointing():
    """per-iteration activation checkpointing gives the same loss and grads."""
    x = ids()
    out = {}
    for ckpt in (False, True):
        model = build(LoopedQwen3, LOOPED)
        model.activation_checkpointing = ckpt
        loss = F.cross_entropy(model(x)[:, :-1].flatten(0, 1), x[:, 1:].flatten())
        loss.backward()
        out[ckpt] = (loss.detach(), grads(model))
    torch.testing.assert_close(out[True][0], out[False][0])
    for name, g in out[False][1].items():
        torch.testing.assert_close(out[True][1][name], g, msg=name)


def test_looped_meta_build():
    """the trainer's sharded path: build on meta, materialize, init."""
    with torch.device("meta"):
        model = LoopedQwen3(**LOOPED)
    materialize(model, torch.device("cpu"))
    model.init_weights()
    assert torch.isfinite(model(ids())).all()
    # residual init scales with the unrolled depth, not the unique block count
    o = model.transformer_blocks()[0].gqa.o_proj.weight
    expected = 0.02 * (2 * model.depth) ** -0.5
    assert abs(o.std().item() - expected) / expected < 0.2, (o.std().item(), expected)


# --- vlm -------------------------------------------------------------------


def vlm_batch(n_images=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    model_tokens = (VISION["image_size"] // VISION["patch_size"] // VLM["pixel_shuffle"]) ** 2
    x = torch.randint(0, VOCAB - 2, (n_images, SEQ), generator=g)
    x[:, 1:1 + model_tokens] = IMAGE_TOKEN
    pixels = torch.rand(n_images, 3, VISION["image_size"], VISION["image_size"], generator=g) * 2 - 1
    return x, pixels


def test_vlm_image_tokens():
    model = build(Qwen3VL, VLM)
    x, pixels = vlm_batch()
    assert model.num_image_tokens == 4 and model.image_size == VISION["image_size"]

    out = model(x, pixel_values=pixels)
    # image 1 changed: row 0 is untouched, row 1 changes from the image span on
    other = pixels.clone()
    other[1] = -other[1]
    diff = (model(x, pixel_values=other) - out).abs().amax(-1)
    assert diff[0].max() < 1e-6, "an image leaked into another row"
    assert diff[1, 0] < 1e-6 and diff[1, 1:].min() > 1e-6, diff[1]

    loss = ARObjective(fused_linear_ce=False).compute_loss(
        model, {"input_ids": x, "labels": x, "pixel_values": pixels}
    )
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_vlm_text_only():
    """no pixel_values: plain qwen3 forward."""
    model = build(Qwen3VL, VLM)
    x = ids()
    assert model(x).shape == (2, SEQ, VOCAB)


def test_image_text_collator():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    eos, pad = tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|fim_pad|>")
    collate = ImageTextCollator(tok, -100, 16, eos, pad, image_size=8, num_image_tokens=4)
    images = [Image.new("RGB", (20, 10), (255, 0, 0)), Image.new("L", (5, 5), 0)]
    batch = collate([
        {"image": images[0], "text": "a red square"},
        {"image": images[1], "text": "a very long caption " * 10},
    ])
    ids_, labels, pixels = batch["input_ids"], batch["labels"], batch["pixel_values"]
    assert pixels.shape == (2, 3, 8, 8)
    assert pixels[0, 0].eq(1).all() and pixels[0, 1].eq(-1).all() and pixels[1].eq(-1).all()

    caption = tok.encode("a red square", add_special_tokens=False).ids
    n = 6 + len(caption) + 1  # vision_start + 4 pads + vision_end, caption, eos
    assert ids_[0, :6].tolist() == collate.prefix
    assert ids_[0, 6:n].tolist() == caption + [eos]
    assert ids_[0, n:].eq(pad).all()
    assert labels[0, :6].eq(-100).all() and labels[0, n:].eq(-100).all()
    assert torch.equal(labels[0, 6:n], ids_[0, 6:n])
    # truncated row: no eos, no pad, every text position scored
    assert labels[1, 6:].ne(-100).all() and ids_[1, -1] != pad


# --- fsdp ------------------------------------------------------------------


def _fsdp_worker(rank: int, world_size: int, port: int, failures):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=60))
    mesh = Mesh({"dp": world_size})
    errors = []

    def loss_fn(model, x, extra):
        return ARObjective(fused_linear_ce=False).compute_loss(
            model, {"input_ids": x, "labels": x, **extra}
        )

    x_text = ids(batch=world_size, seed=7)
    x_vlm, pixels = vlm_batch(world_size, seed=7)
    cases = [
        ("looped", LoopedQwen3, LOOPED, x_text, {}),
        ("vlm", Qwen3VL, VLM, x_vlm, {"pixel_values": pixels}),
    ]
    for name, cls, cfg, x, extra in cases:
        reference = build(cls, cfg)
        loss_fn(reference, x, extra).backward()

        model = build(cls, cfg)
        fsdp = FSDP(model, mesh, model.blocks, shard_dim="dp", replicate_dim=None)
        local = {k: v[rank:rank + 1] for k, v in extra.items()}
        for _ in range(2):  # twice: catches buffers not re-armed after a step
            loss_fn(model, x[rank:rank + 1], local).backward()
            fsdp.sync()
            for (pname, p), (_, p_ref) in zip(
                model.named_parameters(), reference.named_parameters(), strict=True
            ):
                expected = Shard(0).local_slice(p_ref.grad, world_size, rank)
                if p.grad is None or not torch.allclose(p.grad, expected, rtol=1e-4, atol=1e-6):
                    errors.append(f"{name}.{pname}")
            for p in model.parameters():
                p.grad = None
    _check_weighted_vocab_parallel(mesh, rank, world_size, errors)
    if errors:
        failures.append(f"rank {rank}: mismatch in {sorted(set(errors))[:6]}")
    dist.destroy_process_group()


def _check_weighted_vocab_parallel(mesh, rank, world_size, errors):
    """
    the fused loss's per-token weights (the diffusion objective's 1/t) through
    the vocab-parallel path, which runs uncompiled on cpu, against the plain
    weighted nll.
    """
    g = torch.Generator().manual_seed(11)
    hidden = torch.randn(24, 16, generator=g)
    head = torch.randn(VOCAB, 16, generator=g) * 0.1
    labels = torch.randint(0, VOCAB, (24,), generator=g)
    labels[::3] = -100
    weights = torch.rand(24, generator=g) * 4

    h_ref, w_ref = hidden.clone().requires_grad_(), head.clone().requires_grad_()
    nll = F.cross_entropy(h_ref @ w_ref.t(), labels, reduction="none")
    ((nll * weights).sum() / (labels != -100).sum()).backward()

    rows = VOCAB // world_size
    h = hidden.clone().requires_grad_()
    w = head[rank * rows:(rank + 1) * rows].clone().requires_grad_()
    loss = fused_linear_cross_entropy(
        h, w, labels, chunk_size=8, vocab_axis=(mesh, "dp"), weights=weights
    )
    loss.backward()
    expected = (nll * weights).sum() / (labels != -100).sum()
    for name, got, want in [
        ("loss", loss, expected),
        ("grad_hidden", h.grad, h_ref.grad),
        ("grad_head", w.grad, w_ref.grad[rank * rows:(rank + 1) * rows]),
    ]:
        if not torch.allclose(got, want, rtol=1e-4, atol=1e-5):
            errors.append(f"weighted_vocab_parallel.{name}")


def test_distributed():
    world_size = 2
    failures = mp.Manager().list()
    mp.spawn(_fsdp_worker, args=(world_size, 29563, failures), nprocs=world_size)
    assert not failures, list(failures)


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"[{name}] PASS", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAIL: {type(e).__name__}: {e}", flush=True)
            failed.append(name)
    raise SystemExit(1 if failed else 0)
