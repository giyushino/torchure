"""
document masking (models/layers.py doc_mask): a packed row under doc_mask
must give the same logits as running each of its documents on its own,
for qwen3 and for olmo3 (sliding windows shorter than the row).

    uv run tests/doc_mask.py          # cuda if available, else cpu

on cpu, attention runs the dense-mask sdpa fallback (layers.masked_attention);
the flex kernel itself is only exercised on cuda.
"""

import torch

from torchure.models.layers import doc_mask
from torchure.models.olmo3.olmo3 import Olmo3
from torchure.models.qwen3.qwen3 import Qwen3

EOS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# flex attention blocks are 128 wide: docs of these lengths straddle blocks
DOC_LENS = [[100, 1, 250, 161], [512], [37, 300, 175]]
S = 512


def make_rows(vocab: int, gen: torch.Generator) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
    rows, docs = [], []
    for lens in DOC_LENS:
        assert sum(lens) == S
        row_docs = []
        for n in lens:
            doc = torch.randint(EOS + 1, vocab, (n,), generator=gen)
            doc[-1] = EOS
            row_docs.append(doc)
        # the last doc of a row is cut off: no trailing eos
        row_docs[-1][-1] = EOS + 1
        rows.append(torch.cat(row_docs))
        docs.append(row_docs)
    return torch.stack(rows), docs


def test_positions() -> None:
    ids = torch.tensor([[5, 5, EOS, 5, EOS, EOS, 5, 5]])
    pos, _ = doc_mask(ids.to(DEVICE), EOS)
    assert pos.tolist() == [[0, 1, 2, 0, 1, 0, 0, 1]], pos.tolist()
    print("ok  positions restart after every eos")


def check_model(name: str, model: torch.nn.Module, tol: float) -> None:
    gen = torch.Generator().manual_seed(0)
    model = model.to(DEVICE)
    vocab = model.lm_head.out_features
    ids, docs = make_rows(vocab, gen)
    ids = ids.to(DEVICE)
    with torch.no_grad(), torch.autocast(DEVICE, torch.bfloat16, enabled=DEVICE == "cuda"):
        model.doc_mask_eos = EOS
        packed = model(ids).float()
        model.doc_mask_eos = None
        plain = model(ids).float()
        worst = 0.0
        for r, row_docs in enumerate(docs):
            start = 0
            for doc in row_docs:
                alone = model(doc[None].to(DEVICE)).float()[0]
                got = packed[r, start:start + len(doc)]
                worst = max(worst, (got - alone).abs().max().item())
                start += len(doc)
    scale = packed.abs().max().item()
    assert worst <= tol * scale, f"{name}: max diff {worst:.3g} vs logit scale {scale:.3g}"
    # sanity: masking changes the multi-doc rows
    assert (packed[0] - plain[0]).abs().max() > 10 * tol * scale, f"{name}: doc mask had no effect"
    # a single-doc row is plain causal
    assert (packed[1] - plain[1]).abs().max() <= tol * scale, f"{name}: one-doc row changed"
    print(f"ok  {name}: packed + doc_mask == each doc alone (max diff {worst:.2e}, scale {scale:.1f})")


def main() -> None:
    torch.manual_seed(0)
    tol = 2e-2 if DEVICE == "cuda" else 1e-4
    test_positions()
    qwen = Qwen3(num_layers=2, num_heads=4, num_kv_heads=2, emb_dim=128, head_dim=32,
                 vocab_size=512)
    qwen.init_weights()
    check_model("qwen3", qwen, tol)
    olmo = Olmo3(num_layers=4, num_heads=4, num_kv_heads=2, emb_dim=128, head_dim=32,
                 vocab_size=512, ffn_dim=256, sliding_window=128)
    olmo.init_weights()
    check_model("olmo3 (window 128)", olmo, tol)
    print("all doc mask tests passed")


if __name__ == "__main__":
    main()
