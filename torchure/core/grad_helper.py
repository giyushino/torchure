"""
grad operations
"""
import torch
from torchure.core.collective import all_reduce


def clip_grad_norm(params, mesh, max_norm, shard_dims=()) -> torch.Tensor:
    """
    clips grads in place and returns the pre-clip global grad norm.
    max_norm=None measures without clipping.
    """
    norm_type = 2.0
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        assert grads, "clip_grad_norm got no grads; backward didn't run, or all params are frozen"
    norms = torch._foreach_norm(grads, norm_type)
    total = torch.linalg.vector_norm(torch.stack(norms), norm_type)

    if shard_dims:
        total = total ** norm_type
        for dim in shard_dims:
            all_reduce(total, mesh, dim, "sum")
        total = total ** (1.0 / norm_type)

    if max_norm is None:
        return total

    clip_coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    torch._foreach_mul_(grads, clip_coef)
    return total
    
