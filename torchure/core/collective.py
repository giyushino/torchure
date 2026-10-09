"""
mesh-aware collective wrappers over torch.distributed. the only module that
calls torch.distributed comm ops directly.

https://www.aleksagordic.com/blog/collective-operations
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html

- callers pass (tensor, mesh, dim_name), never a ProcessGroup.
- async_op=True returns (tensor, work); don't read the tensor until work.wait().
- all_reduce / broadcast are in place and return the input. the others
  return a new tensor.
- op is a string so "avg" can be emulated on gloo (no ReduceOp.AVG): divide
  locally first, then "sum". pre-dividing means no fixup after an async wait.
- no autograd. if tp needs these inside compiled blocks, swap the internals
  to torch.distributed._functional_collectives behind the same signatures.

tests: uv run tests/collective.py  (gloo/cpu, no gpus needed)

┌────────────────┬───────────────────┬─────────────────────────────────────────┐
│   collective   │   per-rank size   │                   why                   │
├────────────────┼───────────────────┼─────────────────────────────────────────┤
│ all_reduce     │ same (in place)   │ combines values, doesn't move ownership │
├────────────────┼───────────────────┼─────────────────────────────────────────┤
│ broadcast      │ same (in place)   │ overwrites with src's values            │
├────────────────┼───────────────────┼─────────────────────────────────────────┤
│ all_gather     │ × g               │ you keep yours and gain everyone else's │
├────────────────┼───────────────────┼─────────────────────────────────────────┤
│ reduce_scatter │ ÷ g               │ everything combined, you keep one shard │
├────────────────┼───────────────────┼─────────────────────────────────────────┤
│ all_to_all     │ same (new tensor) │ trade, row-for-row — a transpose        │
├────────────────┼───────────────────┼─────────────────────────────────────────┤
│ ring_send_recv │ same (new tensor) │ one chunk out, one chunk in             │
└────────────────┴───────────────────┴─────────────────────────────────────────┘
"""

from typing import Literal, Protocol

import torch
import torch.distributed as dist

ReduceOpName = Literal["sum", "avg", "max", "min"]

_OPS = {
    "sum": dist.ReduceOp.SUM,
    "avg": dist.ReduceOp.AVG,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
}


class MeshLike(Protocol):
    """what collectives need from core/mesh.py (or any test stand-in)."""

    def size(self, dim: str) -> int:
        """number of ranks along `dim`."""
        ...

    def get_group(self, dim: str) -> dist.ProcessGroup:
        """the ProcessGroup this rank belongs to along `dim`."""
        ...

    def coordinate(self, dim: str) -> int:
        """this rank's index within its `dim` group, in [0, size(dim))."""
        ...


def all_reduce(
    tensor: torch.Tensor,
    mesh: MeshLike,
    dim: str,
    op: ReduceOpName = "sum",
    *,
    async_op: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dist.Work]:
    """
    reduce `tensor` across the `dim` group, in place; every rank in the group
    ends up with the same reduced values. returns the input tensor.
    """
    group = mesh.get_group(dim)
    if op == "avg" and dist.get_backend(group) != "nccl":
        tensor /= mesh.size(dim)
        op = "sum"

    work = dist.all_reduce(tensor, _OPS[op], group=group, async_op=async_op)
    return (tensor, work) if async_op else tensor


def broadcast(
    tensor: torch.Tensor,
    mesh: MeshLike,
    dim: str,
    src: int = 0,
    *,
    async_op: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dist.Work]:
    """
    copy `tensor` from coordinate `src` within the dim group (not a global
    rank) to every rank in the group, in place.
    """
    group = mesh.get_group(dim)
    work = dist.broadcast(tensor, src=None, group=group, async_op=async_op, group_src=src)
    return (tensor, work) if async_op else tensor


def all_gather(
    tensor: torch.Tensor,
    mesh: MeshLike,
    dim: str,
    gather_dim: int = 0,
    *,
    async_op: bool = False,
    output_tensor: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dist.Work]:
    """
    gather every rank's `tensor` and concatenate along `gather_dim`, ordered
    by coordinate. out.shape[gather_dim] == tensor.shape[gather_dim] * size.
    all ranks get the same result.

    `output_tensor` lets the caller own a reusable destination buffer (fsdp2's
    per-unit unshard buffer); the result aliases it. only for gather_dim == 0.
    """
    group = mesh.get_group(dim)
    group_size = mesh.size(dim)
    input_tensor = tensor.movedim(gather_dim, 0).contiguous()
    expected_shape = (group_size * input_tensor.size(0), *input_tensor.size()[1:])

    if output_tensor is None:
        output_tensor = torch.empty(
            expected_shape, dtype=tensor.dtype, device=tensor.device
        )
    else:
        assert gather_dim == 0, (
            f"caller-provided output_tensor requires gather_dim=0, got {gather_dim}; "
            "the buffer is in stacked-along-0 layout, not the returned view's layout"
        )
        assert tuple(output_tensor.shape) == expected_shape, (
            f"output_tensor shape {tuple(output_tensor.shape)} != expected {expected_shape}"
        )
        assert output_tensor.dtype == tensor.dtype, (
            f"output_tensor dtype {output_tensor.dtype} != input dtype {tensor.dtype}"
        )
        assert output_tensor.device == tensor.device, (
            f"output_tensor device {output_tensor.device} != input device {tensor.device}"
        )
        assert output_tensor.is_contiguous(), "output_tensor must be contiguous"

    work = dist.all_gather_into_tensor(
        output_tensor=output_tensor, input_tensor=input_tensor, 
        group=group, async_op = async_op
    )
    out = output_tensor.movedim(0, gather_dim)
    return (out, work) if async_op else out


def reduce_scatter(
    tensor: torch.Tensor,
    mesh: MeshLike,
    dim: str,
    op: ReduceOpName = "sum",
    scatter_dim: int = 0,
    *,
    async_op: bool = False,
    output_tensor: torch.Tensor | None = None
) -> torch.Tensor | tuple[torch.Tensor, dist.Work]:
    """
    reduce across the group, then each rank keeps only its coordinate-th
    shard along `scatter_dim`. returns a new tensor of shape
    tensor.shape with shape[scatter_dim] // size.

    the defining identity (tested): for any x,
        all_gather(reduce_scatter(x)) == all_reduce(x)
    which is exactly fsdp's decomposition.

    requires shape[scatter_dim] % size == 0 (no padding).
    """
    group = mesh.get_group(dim)
    group_size = mesh.size(dim)
    assert tensor.shape[scatter_dim] % group_size == 0, f"shape[{scatter_dim}]={tensor.shape[scatter_dim]} not divisible by group size {group_size}"

    # gloo avg emulation. must happen before .contiguous() (which may copy),
    # and out of place: unlike all_reduce, the input must not be mutated
    if op == "avg" and dist.get_backend(group) != "nccl":
        tensor = tensor / group_size
        op = "sum"

    input_tensor = tensor.movedim(scatter_dim, 0).contiguous()
    expected_shape = (input_tensor.size(0) // group_size, *input_tensor.size()[1:])
    if output_tensor is None:
        output_tensor = torch.empty(
            expected_shape, dtype=tensor.dtype, device=tensor.device
        )
    else:
        assert scatter_dim == 0, (
            f"caller-provided output_tensor requires gather_dim=0, got {scatter_dim}; "
            "the buffer is in stacked-along-0 layout, not the returned view's layout"
        )
        assert tuple(output_tensor.shape) == expected_shape, (
            f"output_tensor shape {tuple(output_tensor.shape)} != expected {expected_shape}"
        )
        assert output_tensor.dtype == tensor.dtype, (
            f"output_tensor dtype {output_tensor.dtype} != input dtype {tensor.dtype}"
        )
        assert output_tensor.device == tensor.device, (
            f"output_tensor device {output_tensor.device} != input device {tensor.device}"
        )
        assert output_tensor.is_contiguous(), "output_tensor must be contiguous"

    work = dist.reduce_scatter_tensor(output=output_tensor, input=input_tensor, op=_OPS[op], group=group, async_op=async_op)
    out = output_tensor.movedim(0, scatter_dim)
    return (out, work) if async_op else out


def all_to_all(
    tensor: torch.Tensor,
    mesh: MeshLike,
    dim: str,
    split_dim: int = 0,
    concat_dim: int = 0,
    *,
    async_op: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dist.Work]:
    """
    split `tensor` into size equal chunks along `split_dim`; chunk j is sent
    to coordinate j; the size received chunks (ordered by sender coordinate)
    are concatenated along `concat_dim`. returns a new tensor.

    even splits only. ep will eventually need uneven splits.
    """
    group = mesh.get_group(dim)
    group_size = mesh.size(dim)
    assert tensor.shape[split_dim] % group_size == 0
    assert not async_op, "all_to_all is sync-only (v0)"
    input_tensor = tensor.movedim(split_dim, 0).contiguous()
    output_tensor = torch.empty_like(input_tensor, dtype=tensor.dtype, device=tensor.device)
    dist.all_to_all_single(output=output_tensor, input=input_tensor, group=group)
    chunks = output_tensor.chunk(group_size, dim=0)
    chunks = [c.movedim(0, split_dim) for c in chunks]
    return torch.cat(chunks, dim=concat_dim)


def ring_send_recv(
    tensor: torch.Tensor,
    mesh: MeshLike,
    dim: str,
    *,
    async_op: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dist.Work]:
    """
    simultaneously send `tensor` to coordinate+1 and receive from
    coordinate-1 (mod size) along `dim`. returns the received tensor
    (newly allocated, same shape/dtype as the input).

    sync-only for now; cp overlap would return the recv req as the work handle.
    """
    assert not async_op, "ring_send_recv is sync-only (v0)"
    group = mesh.get_group(dim)
    group_size = mesh.size(dim)
    coord = mesh.coordinate(dim)
    if group_size == 1:
        # backends don't support send/recv to self
        return tensor.clone()

    # P2POp wants global ranks
    dst = dist.get_global_rank(group, (coord + 1) % group_size)
    src = dist.get_global_rank(group, (coord - 1) % group_size)

    send_tensor = tensor.contiguous()
    recv_tensor = torch.empty_like(send_tensor)
    # batched: blocking send-then-recv on every rank would deadlock
    reqs = dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, send_tensor, dst, group=group),
            dist.P2POp(dist.irecv, recv_tensor, src, group=group),
        ]
    )
    for req in reqs:
        req.wait()
    return recv_tensor


def barrier(mesh: MeshLike | None = None, dim: str | None = None) -> None:
    """
    block until all ranks arrive (all ranks of the dim group; the whole
    world if mesh is None).
    """
    dist.barrier(group=None if mesh is None else mesh.get_group(dim))

