"""
chunked fused linear + cross-entropy (Liger-style, pure torch + compile).

why: with vocab_size ~152k, materializing full logits for CE is the single
biggest activation in the step: (B=2, S=4096, V=151936) is 2.5 GB in bf16, and
F.cross_entropy upcasts to fp32 (another 5 GB) and saves logits-sized state for
backward. that memory bounds the per-GPU batch size, which matters even more
under FSDP/distributed than it does here.

how: flatten to N = B*S rows and process them in chunks. per chunk:
    logits = h @ Wᵀ  (bf16 GEMM, chunk-sized)
    loss   = logsumexp - target logit   (fp32)
    and, because d(loss_sum)/d(logits) = softmax - onehot is known in closed
    form, compute grad_hidden and grad_W *in the forward pass* and stash them.
backward then just scales the stashed grads by grad_output / n_valid, in place,
and hands the same buffers straight out. nothing logits-sized ever exists for
more than one chunk at a time, and autograd never stores logits.

cost: 3 chunk GEMMs (logits, grad_h, grad_W) vs 3 for the unfused path
(logits fwd, grad_h, grad_W bwd) -- so compute is the same; we save the full
logits materialization, the fp32 upcast of it, and the softmax re-read in
backward. per-chunk softmax math is torch.compile'd into a few fused kernels.

labels use ignore_index (-100) semantics identical to F.cross_entropy with
reduction="mean".
"""

import torch


@torch.compile(dynamic=False)
def _chunk_fwd(
    hidden: torch.Tensor,      # (C, D) activation dtype (bf16 under autocast)
    weight: torch.Tensor,      # (V, D) pre-cast to hidden.dtype
    labels: torch.Tensor,      # (C,) long
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = (hidden @ weight.t()).float()              # (C, V) fp32
    valid = labels != ignore_index                       # (C,)
    safe = torch.where(valid, labels, 0).unsqueeze(1)    # (C, 1) gather-safe

    lse = torch.logsumexp(logits, dim=-1)                # (C,)
    target = logits.gather(1, safe).squeeze(1)           # (C,)
    loss_sum = ((lse - target) * valid).sum()            # fp32 scalar, sum-reduced

    # d(loss_sum)/d(logits) = softmax - onehot on valid rows, 0 on ignored rows
    probs = torch.exp(logits - lse.unsqueeze(1))
    probs = probs * valid.unsqueeze(1)
    probs.scatter_add_(1, safe, -valid.float().unsqueeze(1))

    grad_logits = probs.to(hidden.dtype)                 # (C, V)
    grad_hidden = grad_logits @ weight                   # (C, D)
    # keep the chunk grad_W in bf16: the GEMM accumulates in fp32 internally,
    # and the caller's `+=` into the fp32 buffer upcasts on the fly. an explicit
    # .float() here would burn an extra full read+write of a (V, D) tensor per
    # chunk for no precision gain.
    grad_weight = grad_logits.t() @ hidden               # (V, D)
    return loss_sum, grad_hidden, grad_weight


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, ignore_index, chunk_size):
        N, _ = hidden.shape
        w = weight.to(hidden.dtype)  # cast once, not per chunk

        loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
        grad_hidden = torch.empty_like(hidden)
        grad_weight = (
            torch.zeros_like(weight, dtype=torch.float32)
            if weight.requires_grad else None
        )

        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            chunk_loss, gh, gw = _chunk_fwd(
                hidden[s:e], w, labels[s:e], ignore_index
            )
            loss_sum += chunk_loss
            grad_hidden[s:e] = gh
            if grad_weight is not None:
                grad_weight += gw

        # mean over non-ignored tokens, matching F.cross_entropy(reduction="mean")
        n_valid = (labels != ignore_index).sum().clamp_min(1)
        # stashed as plain ctx attributes rather than save_for_backward: these
        # are forward *intermediates* (neither inputs nor outputs of this
        # Function), so there is no version-counter contract to honor -- which
        # is what lets backward scale them in place and drop them eagerly.
        ctx.n_valid = n_valid
        ctx.grad_hidden = grad_hidden
        ctx.grad_weight = grad_weight
        return loss_sum / n_valid

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.n_valid is None:
            # the buffers are scaled in place and released below, so there is
            # nothing left to differentiate a second time. save_for_backward
            # would have raised this for us; since we stash on ctx, say it
            # ourselves rather than dying on a NoneType further down.
            raise RuntimeError(
                "fused_linear_cross_entropy: backward called twice. the stashed "
                "grads are consumed in place on the first call, so retain_graph "
                "does not work through this op"
            )
        grad_hidden, grad_weight = ctx.grad_hidden, ctx.grad_weight
        scale = grad_out / ctx.n_valid
        # drop ctx's references first: once the returned tensors are consumed by
        # the engine, nothing else holds these buffers and the allocator can
        # reuse them immediately instead of at graph teardown.
        ctx.grad_hidden = ctx.grad_weight = ctx.n_valid = None

        # in place, both of them. the buffers were built in forward for this one
        # backward and nobody else aliases them, so `out = buf * scale` would
        # just allocate a second copy of each. that matters here: grad_weight is
        # a full fp32 (V, D) -- 594 MiB at qwen3-0.6B's 151936 vocab -- so the
        # out-of-place version doubles it at exactly the moment the rest of
        # backward's grads are also live.
        grad_hidden.mul_(scale.to(grad_hidden.dtype))
        if grad_weight is not None:
            grad_weight.mul_(scale)
        return grad_hidden, grad_weight, None, None, None


def fused_linear_cross_entropy(
    hidden: torch.Tensor,        # (B, S, D) or (N, D)
    weight: torch.Tensor,        # (V, D) lm_head weight
    labels: torch.Tensor,        # (B, S) or (N,), already shifted
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    if hidden.dim() == 3:
        hidden = hidden.reshape(-1, hidden.shape[-1])
        labels = labels.reshape(-1)
    return _FusedLinearCE.apply(hidden, weight, labels, ignore_index, chunk_size)
