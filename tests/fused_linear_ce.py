"""
correctness check: chunked fused linear+CE vs plain lm_head + F.cross_entropy.

compares loss, grad wrt hidden, and grad wrt the head weight under the same
bf16-matmul / fp32-softmax precision recipe the trainer uses (autocast).
run: uv run tests/fused_linear_ce.py
"""

import torch
import torch.nn.functional as F

from torchure.loss.fused_linear_ce import fused_linear_cross_entropy


def reference(hidden, weight, labels, ignore_index):
    logits = hidden @ weight.to(hidden.dtype).t()
    V = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, V).float(), labels.reshape(-1), ignore_index=ignore_index
    )


def run_case(B, S, D, V, chunk_size, ignore_frac, device="cuda"):
    torch.manual_seed(0)
    hidden_ref = torch.randn(B, S, D, device=device, dtype=torch.bfloat16) * 0.5
    weight_ref = torch.randn(V, D, device=device, dtype=torch.float32) * 0.02
    labels = torch.randint(0, V, (B, S), device=device)
    labels[torch.rand(B, S, device=device) < ignore_frac] = -100

    h1 = hidden_ref.clone().requires_grad_(True)
    w1 = weight_ref.clone().requires_grad_(True)
    loss_ref = reference(h1, w1, labels, -100)
    loss_ref.backward()

    h2 = hidden_ref.clone().requires_grad_(True)
    w2 = weight_ref.clone().requires_grad_(True)
    loss_fused = fused_linear_cross_entropy(h2, w2, labels, -100, chunk_size)
    loss_fused.backward()

    def rel(a, b):
        return ((a - b).float().norm() / b.float().norm().clamp_min(1e-12)).item()

    print(
        f"B={B} S={S} D={D} V={V} chunk={chunk_size} ign={ignore_frac}: "
        f"loss ref={loss_ref.item():.6f} fused={loss_fused.item():.6f} "
        f"| rel dh={rel(h2.grad, h1.grad):.2e} dw={rel(w2.grad, w1.grad):.2e}"
    )
    assert abs(loss_ref.item() - loss_fused.item()) < 2e-3 * max(1.0, loss_ref.item())
    assert rel(h2.grad, h1.grad) < 3e-2   # bf16 grads
    assert rel(w2.grad, w1.grad) < 3e-2


if __name__ == "__main__":
    run_case(2, 128, 64, 1000, chunk_size=64, ignore_frac=0.1)       # uneven chunks
    run_case(2, 256, 128, 5000, chunk_size=512, ignore_frac=0.0)     # one chunk
    run_case(2, 4096, 1024, 151936, chunk_size=1024, ignore_frac=0.05)  # real shape
    print("all fused linear CE checks passed")
