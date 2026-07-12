"""
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
mesh-aware collective wrappers over torch.distributed.

the only module in the repo that calls torch.distributed comm ops directly;
ddp grad sync, fsdp2, dtensor redistribute, and cp ring attention all speak
these functions instead of raw ProcessGroups.

decisions baked into the signatures (so the call sites don't churn later):

- callers pass (tensor, mesh, dim_name), never a ProcessGroup. MeshLike below
  is the contract core/mesh.py must satisfy; collectives are implementable
  and testable before the real mesh exists (see FlatMesh in
  tests/collective.py).
- sync call returns the result tensor. async_op=True returns (tensor, work);
  the tensor is NOT safe to read until work.wait().
- all_reduce / broadcast are IN-PLACE on the input and return it (the ddp
  grad path wants zero extra allocations). all_gather / reduce_scatter /
  all_to_all allocate and return a NEW tensor (the output shape differs
  anyway).
- op is a lowercase string, not dist.ReduceOp, so "avg" can be emulated on
  gloo (which has no ReduceOp.AVG): all_reduce "sum", then divide by
  mesh.size(dim) locally. nccl uses ReduceOp.AVG directly.
- no autograd through these: they run on grads/params/buffers outside any
  compiled graph. when tp needs collectives *inside* compiled blocks, swap
  the internals to torch.distributed._functional_collectives behind these
  same signatures -- they are functional-shaped on purpose.

run tests with: uv run tests/collective.py  (gloo/cpu, no gpus needed)

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

# shared by all_reduce / reduce_scatter. "avg" is resolved per-backend at the
# call site (nccl has ReduceOp.AVG, gloo needs sum + local divide).
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
        """the ProcessGroup this rank belongs to along `dim`.

        (mesh.py note for later: dist.new_group is itself a collective --
        ALL ranks must call it for EVERY group, including groups this rank
        is not a member of, in the same order.)
        """
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
    # 1. resolve WHERE: mesh dim name -> the ProcessGroup this rank reduces in
    group = mesh.get_group(dim)

    # 2. resolve WHAT: "avg" needs a per-backend decision, everything else is
    #    a straight enum lookup
    if op == "avg" and dist.get_backend(group) != "nccl":
        # no ReduceOp.AVG here: sum, then divide locally. the divide must
        # happen after the sum completes, so no async on this path (v0);
        # revisit with a Work wrapper if a consumer ever needs it.
        assert not async_op, "async 'avg' unsupported on backends without ReduceOp.AVG"
        dist.all_reduce(tensor, _OPS["sum"], group=group)
        tensor /= mesh.size(dim)
        return tensor

    # 3. communicate: the backend runs the actual ring/tree exchange and
    #    mutates `tensor` in place on every rank
    work = dist.all_reduce(tensor, _OPS[op], group=group, async_op=async_op)

    # 4. no fix-up needed on this path
    # 5. return the input tensor (in-place contract), plus the handle if async
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
    copy `tensor` from the rank at coordinate `src` WITHIN the dim group
    (not a global rank) to every rank in the group, in place.

    impl note: dist.broadcast wants a GLOBAL rank -- convert with
    dist.get_global_rank(group, src). passing the coordinate straight
    through happens to work on a 1-d mesh and silently breaks on any real
    one; tests/collective.py uses src=1 to catch exactly this.
    """
    # what
    group = mesh.get_group(dim)
    # where
    work = dist.broadcast(tensor, src=None, group=group, async_op=async_op, group_src=src)
    return (tensor, work) if async_op else tensor



def all_gather(
    tensor: torch.Tensor,
    mesh: MeshLike,
    dim: str,
    gather_dim: int = 0,
    *,
    async_op: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dist.Work]:
    """
    gather every rank's `tensor` and concatenate along `gather_dim`, ordered
    by coordinate. out.shape[gather_dim] == tensor.shape[gather_dim] * size.
    returns a new tensor; all ranks get the same result.

    impl notes:
    - wrap dist.all_gather_into_tensor (single preallocated output, one
      kernel), not the python-list variant.
    - it only stacks along dim 0: for gather_dim != 0, movedim before and
      after. inputs must be contiguous -- either .contiguous() or assert,
      never silently corrupt.
    """
    group = mesh.get_group(dim)
    group_size = mesh.size(dim)
    input_tensor = tensor.movedim(gather_dim, 0).contiguous()
    output_tensor = torch.empty(
        (group_size * input_tensor.size(0), *input_tensor.size()[1:]), 
        dtype=tensor.dtype, device=tensor.device
    )

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
) -> torch.Tensor | tuple[torch.Tensor, dist.Work]:
    """
    reduce across the group, then each rank keeps only its coordinate-th
    shard along `scatter_dim`. returns a new tensor of shape
    tensor.shape with shape[scatter_dim] // size.

    the defining identity (tested): for any x,
        all_gather(reduce_scatter(x)) == all_reduce(x)
    which is exactly fsdp's decomposition.

    v0: require tensor.shape[scatter_dim] % size == 0 and assert loudly;
    no padding. wrap dist.reduce_scatter_tensor; same movedim/contiguity
    story as all_gather. same gloo-"avg" caveat as all_reduce.
    """
    group = mesh.get_group(dim)
    group_size = mesh.size(dim)
    assert tensor.shape[scatter_dim] % group_size == 0, f"shape[{scatter_dim}]={tensor.shape[scatter_dim]} not divisible by group size {group_size}"
    input_tensor = tensor.movedim(scatter_dim, 0).contiguous()
    output_tensor = torch.empty((input_tensor.size(0) // group_size, *input_tensor.size()[1:]), 
                                dtype=tensor.dtype, device=tensor.device
    )

    if op == "avg" and dist.get_backend(group) != "nccl":
        # no ReduceOp.AVG here: sum, then divide locally. the divide must
        # happen after the sum completes, so no async on this path (v0);
        # revisit with a Work wrapper if a consumer ever needs it.
        assert not async_op, "async 'avg' unsupported on backends without ReduceOp.AVG"
        dist.reduce_scatter_tensor(output=output_tensor, input=input_tensor, op=_OPS["sum"], group=group)
        out = output_tensor.movedim(0, scatter_dim)
        out /= group_size
        return out

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

    this is Shard(i)->Shard(j) redistribution and the ep token shuffle.

    v0: even splits only (shape[split_dim] % size == 0), wrap
    dist.all_to_all_single. ep will eventually need uneven
    input_splits/output_splits -- add those kwargs when a consumer exists,
    not now.
    """
    group = mesh.get_group(dim)
    group_size = mesh.size(dim)
    assert tensor.shape[split_dim] % group_size == 0
    assert async_op == False
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

    this is the cp ring-attention primitive (and the building block of the
    educational ring all-reduce).

    impl notes:
    - every rank doing send() then recv() deadlocks on an unbuffered
      backend. use dist.batch_isend_irecv([P2POp(dist.isend, ...),
      P2POp(dist.irecv, ...)]) and wait the returned reqs, or stagger by
      coordinate parity.
    - P2POp also wants GLOBAL ranks: same dist.get_global_rank conversion
      as broadcast.
    """
    raise NotImplementedError


def barrier(mesh: MeshLike | None = None, dim: str | None = None) -> None:
    """
    block until all ranks arrive (all ranks of the dim group; the whole
    world if mesh is None). for fencing checkpoint saves and test output --
    don't sprinkle it through the training loop.
    """
    raise NotImplementedError


