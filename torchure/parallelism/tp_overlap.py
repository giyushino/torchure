"""
tp comm/compute overlap: fused autograd ops that own both a region's
sequence-parallel collective and the matmuls next to it, so the collective
can run while those matmuls do. only meaningful under sequence parallel.

    column(x_local, Ws)   all_gather(x) along the sp dim, then y_i = x_full @ W_i^T
                          for every column linear of the region
    row(h, W)             y = h @ W^T, then reduce_scatter(y) along the sp dim

the plain path (tensor_parallel.py's _Gather/_Scatter around ordinary
linears) runs each collective with nothing behind it. two modes here:

    "backward"            column backward: dX = sum_i dY_i W_i, launch the
                          reduce_scatter of dX async, compute the weight grads
                          dW_i = dY_i^T X_full (which don't need it) while it
                          flies, then wait. forward and the row op are the
                          plain collective + matmul, fused into one node.

    "collective_matmul"   additionally split every collective into tp ring
                          steps over the token chunks and pipeline them with
                          the matmuls (megatron tp_comm_overlap / pytorch
                          async tp):
                            all_gather + matmul    compute on the chunk held
                                                   while the next one arrives
                            matmul + reduce_scatter compute the partial for the
                                                   next chunk while the running
                                                   sum travels the ring

layout. everything works on "rows": the activation with its sp dim moved to
the front and flattened to (tokens, features), so rank k's chunk of the full
tensor is the contiguous block of rows [k*n, (k+1)*n). with the sp dim at 0
(sharding the batch) or a batch of 1, those moves are free views; sharding
the sequence of a batch > 1 costs a copy each way, as the plain path does.

precision. matmuls run in the autocast dtype (what F.linear under autocast
would do); x is gathered in that dtype (half the bytes of the plain path,
which gathers the fp32 norm output and casts after). grads of x are
reduce-scattered in x's own dtype, as the plain path does.

the ops take and return grads for the weights as autograd inputs, so the
grad-accumulate hooks ddp and fsdp hang off the params fire as usual.
"""

import torch

from torchure.core.collective import MeshLike, all_gather, reduce_scatter, ring_send_recv

MODES = ("none", "backward", "collective_matmul")


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    return x.dtype


def _to_rows(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Size]:
    """x with `dim` moved to the front, as (rows, features); and that moved shape."""
    moved = x.movedim(dim, 0)
    return moved.reshape(-1, moved.shape[-1]), moved.shape


def _from_rows(rows: torch.Tensor, moved_shape, dim: int) -> torch.Tensor:
    """inverse of _to_rows; moved_shape's last dim is replaced by rows'."""
    return rows.view(*moved_shape[:-1], rows.shape[-1]).movedim(0, dim)


def _full_shape(moved_shape, g: int) -> tuple[int, ...]:
    return (moved_shape[0] * g, *moved_shape[1:])


class _Ring:
    """rank / size along the tp axis and the chunk views of a rows tensor."""

    def __init__(self, mesh: MeshLike, axis: str):
        self.mesh, self.axis = mesh, axis
        self.g = mesh.size(axis)
        self.r = mesh.coordinate(axis)

    def chunk(self, rows: torch.Tensor, k: int | None) -> torch.Tensor:
        """row chunk k of a full rows tensor; all of it for k=None."""
        if k is None:
            return rows
        n = rows.shape[0] // self.g
        return rows[k * n: (k + 1) * n]

    def send_recv(self, tensor: torch.Tensor, out: torch.Tensor | None = None):
        return ring_send_recv(tensor, self.mesh, self.axis, async_op=True, output_tensor=out)


def _gather_matmul(ring: _Ring, x_rows, weights, overlap: bool):
    """X_full = all_gather(x_rows), Y_i = X_full @ W_i^T; returns (X_full, [Y_i])."""
    n = x_rows.shape[0]
    X = x_rows.new_empty(ring.g * n, x_rows.shape[1])
    Ys = [x_rows.new_empty(ring.g * n, w.shape[0]) for w in weights]
    if not overlap:
        all_gather(x_rows, ring.mesh, ring.axis, 0, output_tensor=X)
        for Y, W in zip(Ys, weights, strict=True):
            torch.mm(X, W.t(), out=Y)
        return X, Ys

    # at step s this rank holds chunk (r - s) and receives chunk (r - s - 1)
    k = ring.r
    cur = ring.chunk(X, k)
    cur.copy_(x_rows)
    for s in range(ring.g):
        work = None
        if s < ring.g - 1:
            nxt = (ring.r - s - 1) % ring.g
            _, work = ring.send_recv(cur, ring.chunk(X, nxt))
        for Y, W in zip(Ys, weights, strict=True):
            torch.mm(cur, W.t(), out=ring.chunk(Y, k))
        if work is not None:
            work.wait()
            k, cur = nxt, ring.chunk(X, nxt)
    return X, Ys


def _matmul_scatter(ring: _Ring, partial_of, overlap: bool, between=None):
    """
    reduce_scatter of a (g*n, d) partial sum: partial_of(k) returns this
    rank's partial for row chunk k, partial_of(None) the whole thing.
    `between` runs while the last collective is in flight (weight grads).
    """
    if not overlap:
        full = partial_of(None)
        out, work = reduce_scatter(full, ring.mesh, ring.axis, "sum", 0, async_op=True)
        if between is not None:
            between()
        work.wait()
        return out

    # at step s this rank adds its partial for chunk (r - 1 - s) to the running
    # sum from rank r-1 and passes it on; the last step lands on chunk r.
    # `sent` holds each in-flight send buffer until its wait: dropping it
    # earlier would let the allocator hand its memory to the next matmul
    work = recv = sent = None
    for s in range(ring.g):
        acc = partial_of((ring.r - 1 - s) % ring.g)
        if s == ring.g - 1 and between is not None:
            between()
        if work is not None:
            work.wait()
            acc += recv
        if s < ring.g - 1:
            sent = acc
            recv, work = ring.send_recv(sent)
    return acc


def _partial_sum(pairs):
    """sum of G @ W over the (G, W) pairs whose G isn't None."""
    out = None
    for G, W in pairs:
        if G is not None:
            out = G @ W if out is None else out.addmm_(G, W)
    return out


class _Column(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mesh, axis, dim, overlap, *weights):
        ring = _Ring(mesh, axis)
        cdtype = _compute_dtype(x)
        x_rows, moved = _to_rows(x.to(cdtype), dim)
        Ws = [w.to(cdtype) for w in weights]
        X, Ys = _gather_matmul(ring, x_rows, Ws, overlap)

        ctx.save_for_backward(X, *Ws)
        ctx.ring, ctx.dim, ctx.overlap = ring, dim, overlap
        ctx.moved, ctx.x_dtype = moved, x.dtype
        ctx.w_dtypes = [w.dtype for w in weights]
        full = _full_shape(moved, ring.g)
        return (_from_rows(X, full, dim), *(_from_rows(Y, full, dim) for Y in Ys))

    @staticmethod
    def backward(ctx, grad_x_full, *grad_ys):
        X, *Ws = ctx.saved_tensors
        ring, dim = ctx.ring, ctx.dim
        cdtype = X.dtype
        Gs = [None if g is None else _to_rows(g.to(cdtype), dim)[0] for g in grad_ys]
        Gx = None if grad_x_full is None else _to_rows(grad_x_full, dim)[0]
        dWs: list[torch.Tensor | None] = [None] * len(Ws)

        def partial_of(k):
            p = _partial_sum(
                (None if G is None else ring.chunk(G, k), W) for G, W in zip(Gs, Ws, strict=True)
            )
            rows = X.shape[0] if k is None else X.shape[0] // ring.g
            p = X.new_zeros(rows, X.shape[1], dtype=ctx.x_dtype) if p is None else p.to(ctx.x_dtype)
            if Gx is not None:
                p += ring.chunk(Gx, k).to(ctx.x_dtype)
            return p

        def weight_grads():
            for i, G in enumerate(Gs):
                if G is not None:
                    dWs[i] = (G.t() @ X).to(ctx.w_dtypes[i])

        dx_rows = _matmul_scatter(ring, partial_of, ctx.overlap, weight_grads)
        dx = _from_rows(dx_rows, ctx.moved, dim)
        return (dx, None, None, None, None, *dWs)


class _Row(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, mesh, axis, dim, overlap, weight):
        ring = _Ring(mesh, axis)
        cdtype = _compute_dtype(h)
        H, moved = _to_rows(h.to(cdtype), dim)
        W = weight.to(cdtype)

        out = _matmul_scatter(ring, lambda k: ring.chunk(H, k) @ W.t(), overlap)
        ctx.save_for_backward(H, W)
        ctx.ring, ctx.dim, ctx.overlap = ring, dim, overlap
        ctx.moved, ctx.h_dtype, ctx.w_dtype = moved, h.dtype, weight.dtype
        local = (moved[0] // ring.g, *moved[1:-1], W.shape[0])
        return _from_rows(out, local, dim)

    @staticmethod
    def backward(ctx, grad):
        H, W = ctx.saved_tensors
        ring, dim = ctx.ring, ctx.dim
        G_local = _to_rows(grad.to(H.dtype), dim)[0]
        G, dH = _gather_matmul(ring, G_local, [W.t()], ctx.overlap)
        dW = (G.t() @ H).to(ctx.w_dtype)
        dh = _from_rows(dH[0], ctx.moved, dim).to(ctx.h_dtype)
        return dh, None, None, None, None, dW


@torch.compiler.disable
def column(x, weights, mesh: MeshLike, axis: str, dim: int, overlap: bool):
    """(x_full, *outputs): the gathered input and each column linear's output."""
    return _Column.apply(x, mesh, axis, dim, overlap, *weights)


@torch.compiler.disable
def row(h, weight, mesh: MeshLike, axis: str, dim: int, overlap: bool):
    """the row linear's output, reduced and scattered along `dim`."""
    return _Row.apply(h, mesh, axis, dim, overlap, weight)
