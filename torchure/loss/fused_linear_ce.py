"""
chunked fused linear + cross-entropy (Liger-style, pure torch + compile).

with a ~152k vocab, full logits are the biggest activation in the step
(2.5 GB bf16 at B=2, S=4096, plus the fp32 upcast in F.cross_entropy).

rows are processed in chunks. since d(loss)/d(logits) = softmax - onehot, each
chunk computes its loss AND grad_hidden / grad_W in the forward pass; backward
just scales the stashed grads in place. nothing logits-sized outlives a chunk,
and the GEMM count is the same as the unfused path.

matches F.cross_entropy(reduction="mean", ignore_index=...).
"""

import torch

from torchure.core.collective import all_reduce


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
    # grad_W is accumulated by the caller with one addmm
    return loss_sum, grad_hidden, grad_logits


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, ignore_index, chunk_size):
        N, _ = hidden.shape
        w = weight.to(hidden.dtype)

        loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
        grad_hidden = torch.empty_like(hidden)
        grad_weight = (
            torch.zeros_like(weight, dtype=torch.float32)
            if weight.requires_grad else None
        )

        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            chunk_loss, gh, grad_logits = _chunk_fwd(
                hidden[s:e], w, labels[s:e], ignore_index
            )
            loss_sum += chunk_loss
            grad_hidden[s:e] = gh
            if grad_weight is not None:
                # bf16 GEMM accumulating straight into the fp32 buffer: no
                # per-chunk bf16 (V, D) partial and no lossy rounding
                torch.addmm(
                    grad_weight, grad_logits.t(), hidden[s:e],
                    out_dtype=torch.float32, out=grad_weight,
                )
            # otherwise it stays alive through the next chunk, at peak memory
            del grad_logits

        n_valid = (labels != ignore_index).sum().clamp_min(1)
        # plain ctx attributes, not save_for_backward, so backward can scale
        # them in place and drop them early
        ctx.n_valid = n_valid
        ctx.grad_hidden = grad_hidden
        ctx.grad_weight = grad_weight
        return loss_sum / n_valid

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.n_valid is None:
            raise RuntimeError(
                "fused_linear_cross_entropy: backward called twice. the stashed "
                "grads are consumed in place on the first call, so retain_graph "
                "does not work through this op"
            )
        grad_hidden, grad_weight = ctx.grad_hidden, ctx.grad_weight
        scale = grad_out / ctx.n_valid
        # drop ctx refs so the buffers free as soon as autograd consumes them
        ctx.grad_hidden = ctx.grad_weight = ctx.n_valid = None

        # in place: grad_weight is a full fp32 (V, D), a copy would double it
        grad_hidden.mul_(scale.to(grad_hidden.dtype))
        if grad_weight is not None:
            grad_weight.mul_(scale)
        return grad_hidden, grad_weight, None, None, None


# ---- vocab-parallel variant ------------------------------------------------
#
# the weight is this rank's slice of the vocab rows (tensor parallel), so the
# logits are a slice too and the softmax needs every rank's. per chunk:
#
#     local logits -> row max           --all_reduce(max)--> global max
#     sum of exp(logits - max), and the
#     target logit where this rank owns
#     the label (0 elsewhere)            --all_reduce(sum)--> logsumexp, target
#
# two all_reduces of a few floats per row. the grad wrt the local logits is
# then local, and grad_hidden is a partial sum, all_reduced once at the end.
# the three pieces between the collectives are compiled on cuda, like _chunk_fwd.


def _vp_logits(hidden, weight):
    logits = (hidden @ weight.t()).float()               # (C, V/tp) fp32
    return logits, logits.max(dim=-1).values


def _vp_stats(logits, row_max, safe, owned):
    sum_exp = torch.exp(logits - row_max.unsqueeze(1)).sum(dim=-1)
    target = logits.gather(1, safe).squeeze(1) * owned
    return torch.stack([sum_exp, target])                # (2, C)


def _vp_grads(logits, lse, target, valid, safe, owned, weight):
    loss_sum = ((lse - target) * valid).sum()
    # softmax - onehot, restricted to this rank's vocab rows
    probs = torch.exp(logits - lse.unsqueeze(1)) * valid.unsqueeze(1)
    probs.scatter_add_(1, safe, -owned.unsqueeze(1))
    grad_logits = probs.to(weight.dtype)
    return loss_sum, grad_logits @ weight, grad_logits


_VP_COMPILED: dict = {}


def _vp(fn, device: torch.device):
    if device.type != "cuda":
        return fn
    if fn not in _VP_COMPILED:
        _VP_COMPILED[fn] = torch.compile(fn, dynamic=False)
    return _VP_COMPILED[fn]


class _VocabParallelFusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, ignore_index, chunk_size, mesh, axis):
        N, _ = hidden.shape
        w = weight.to(hidden.dtype)
        rows = weight.shape[0]
        device = hidden.device

        valid_all = labels != ignore_index
        local = labels - mesh.coordinate(axis) * rows
        owned_all = valid_all & (local >= 0) & (local < rows)
        safe_all = torch.where(owned_all, local, 0).unsqueeze(1)

        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        grad_hidden = torch.empty_like(hidden)
        grad_weight = (
            torch.zeros_like(weight, dtype=torch.float32)
            if weight.requires_grad else None
        )

        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            valid = valid_all[s:e].float()
            owned = owned_all[s:e].float()
            safe = safe_all[s:e]

            logits, row_max = _vp(_vp_logits, device)(hidden[s:e], w)
            all_reduce(row_max, mesh, axis, "max")
            stats = _vp(_vp_stats, device)(logits, row_max, safe, owned)
            all_reduce(stats, mesh, axis, "sum")
            lse = row_max + torch.log(stats[0])

            chunk_loss, gh, grad_logits = _vp(_vp_grads, device)(
                logits, lse, stats[1], valid, safe, owned, w
            )
            loss_sum += chunk_loss
            grad_hidden[s:e] = gh
            if grad_weight is not None and device.type == "cuda":
                torch.addmm(
                    grad_weight, grad_logits.t(), hidden[s:e],
                    out_dtype=torch.float32, out=grad_weight,
                )
            elif grad_weight is not None:
                # the out_dtype overload is cuda-only
                grad_weight.addmm_(grad_logits.t().float(), hidden[s:e].float())
            del logits, grad_logits

        # each rank holds the part of grad_hidden that flows through its rows
        all_reduce(grad_hidden, mesh, axis, "sum")

        n_valid = valid_all.sum().clamp_min(1)
        ctx.n_valid = n_valid
        ctx.grad_hidden = grad_hidden
        ctx.grad_weight = grad_weight
        return loss_sum / n_valid

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.n_valid is None:
            raise RuntimeError(
                "fused_linear_cross_entropy: backward called twice. the stashed "
                "grads are consumed in place on the first call, so retain_graph "
                "does not work through this op"
            )
        grad_hidden, grad_weight = ctx.grad_hidden, ctx.grad_weight
        scale = grad_out / ctx.n_valid
        ctx.grad_hidden = ctx.grad_weight = ctx.n_valid = None

        grad_hidden.mul_(scale.to(grad_hidden.dtype))
        if grad_weight is not None:
            grad_weight.mul_(scale)
        return grad_hidden, grad_weight, None, None, None, None, None


def fused_linear_cross_entropy(
    hidden: torch.Tensor,        # (B, S, D) or (N, D)
    weight: torch.Tensor,        # (V, D) lm_head weight
    labels: torch.Tensor,        # (B, S) or (N,), already shifted
    ignore_index: int = -100,
    chunk_size: int = 1024,
    vocab_axis: tuple | None = None,
) -> torch.Tensor:
    """
    vocab_axis=(mesh, axis): `weight` is this rank's slice of the vocab rows
    along that mesh axis (tensor parallel). `hidden` and `labels` are still
    the full, replicated ones, and every rank returns the same full loss.
    """
    if hidden.dim() == 3:
        hidden = hidden.reshape(-1, hidden.shape[-1])
        labels = labels.reshape(-1)
    # the final norm returns fp32 even under autocast, and autocast doesn't
    # reach inside the autograd.Function, so cast here or everything runs in fp32
    if torch.is_autocast_enabled(hidden.device.type):
        hidden = hidden.to(torch.get_autocast_dtype(hidden.device.type))
    if vocab_axis is not None:
        mesh, axis = vocab_axis
        return _VocabParallelFusedLinearCE.apply(
            hidden, weight, labels, ignore_index, chunk_size, mesh, axis
        )
    return _FusedLinearCE.apply(hidden, weight, labels, ignore_index, chunk_size)
