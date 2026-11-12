"""
gradient operations that have to know how grads are laid out across the mesh.
"""

import torch

from torchure.core.collective import all_reduce


def clip_grad_norm(params, mesh, max_norm, shard_dims=(), shard_dims_of=None) -> torch.Tensor:
    """
    clips grads in place and returns the pre-clip global grad norm.
    max_norm=None measures without clipping.

    a sharded grad's norm is spread over the ranks of its mesh axes; a
    replicated one must be counted once. `shard_dims` are the axes EVERY grad
    is sharded over (fsdp). when params differ (tensor parallel shards some
    and replicates others), pass `shard_dims_of(param) -> axes` instead.
    """
    norm_type = 2.0
    params = [p for p in params if p.grad is not None]
    grads = [p.grad for p in params]
    assert grads, "clip_grad_norm got no grads; backward didn't run, or all params are frozen"
    norms = torch._foreach_norm(grads, norm_type)

    if shard_dims_of is None:
        groups = {tuple(shard_dims): norms}
    else:
        groups = {}
        for param, norm in zip(params, norms, strict=True):
            groups.setdefault(tuple(shard_dims_of(param)), []).append(norm)

    # one squared norm per distinct sharding, each reduced over its own axes.
    # sorted so every rank launches the collectives in the same order
    total = 0.0
    for axes in sorted(groups):
        squared = torch.linalg.vector_norm(torch.stack(groups[axes]), norm_type) ** norm_type
        for dim in axes:
            all_reduce(squared, mesh, dim, "sum")
        total = total + squared
    total = total ** (1.0 / norm_type)

    if max_norm is None:
        return total

    clip_coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    torch._foreach_mul_(grads, clip_coef)
    return total
