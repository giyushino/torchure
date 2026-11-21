"""
ddp: every rank along the mesh's dp axis holds a full replica of the model.
grads are averaged across the axis after backward, which keeps the replicas
in lockstep.

- params are grouped into buckets in reverse `model.parameters()` order
  (roughly backward order), so all-reduces launch from grad hooks while
  backward is still running.
- a bucket's grads are copied into one flat buffer that is all-reduced in
  place; `.grad` is repointed at views of it. a param at or over the bucket
  cap (e.g. the tied embedding) is all-reduced directly through its `.grad`,
  with no staging copy.
- buckets launch in bucket order, on every rank, as each becomes ready. a
  param that got no grad this step (an moe expert no token picked, a param
  the forward never used) leaves its bucket unready; `sync()` launches what
  is left, with zeros for the missing grads. ranks may differ in which grads
  are missing: the fixed order keeps their collectives paired.
- `sync()` waits on every reduce; call it between backward and
  optimizer.step.
- `requires_sync` is the no_sync switch for grad accumulation: False for the
  first K-1 microbatches, True for the last.
- `reduce_dtype` (e.g. bf16) all-reduces a cast copy of each bucket and
  copies the average back into the fp32 grads at `sync()`: half the bytes on
  the wire, at the cost of rounding the average. None reduces in the grad
  dtype.
- `dim_of(param)` overrides the axis for individual params: expert
  parallel's expert weights are replicated over fewer ranks than the rest of
  the model. None (or an axis of size 1) means the param has no replicas to
  sync with.

one DDP instance per model: hooks are registered once and never removed.
"""

import torch
import torch.nn as nn

from collections.abc import Callable

from torchure.core.collective import MeshLike, all_reduce, broadcast


class _Bucket:
    """
    params whose grads are all-reduced together. with more than one param the
    grads live in `flat`, each param's `.grad` being one of its `views`; a
    lone param is reduced through its own `.grad`.
    """

    def __init__(self, params: list[nn.Parameter], dim: str):
        self.params = params
        self.dim = dim              # the mesh axis these grads are averaged over
        self.pending = len(params)  # params still owing a grad this backward
        self.flat = None
        self.views = None
        if len(params) > 1:
            numel = sum(p.numel() for p in params)
            self.flat = torch.empty(numel, dtype=params[0].dtype, device=params[0].device)
            self.views = []
            offset = 0
            for p in params:
                self.views.append(self.flat[offset: offset + p.numel()].view_as(p))
                offset += p.numel()


class DDP:
    def __init__(
        self,
        model: nn.Module,
        mesh: MeshLike,
        dim="dp",
        bucket_mb=25,
        dim_of: Callable[[nn.Parameter], str | None] | None = None,
        reduce_dtype: torch.dtype | None = None,
    ):
        self.mesh = mesh
        self.dim = dim
        self.group_size = mesh.size(dim)
        self._dim_of_param = dim_of
        self.reduce_dtype = reduce_dtype
        # (work, reduced tensor, grad to copy it back into or None if in place)
        self._works: list[tuple] = []
        self._next = 0  # index of the next bucket to launch this step
        self._buckets: list[_Bucket] = []
        self.requires_sync = True

        self._replicate(model)
        params = [
            param for param in model.parameters()
            if param.requires_grad and self._axis(param) is not None
        ]
        if not params:
            return
        params.reverse()
        self._buckets = self._build_buckets(params, int(bucket_mb * 1024 * 1024))
        self._bucket_of = {p: b for b in self._buckets for p in b.params}
        for param in params:
            param.register_post_accumulate_grad_hook(self._on_grad_ready)

    def _axis(self, param: nn.Parameter | None = None) -> str | None:
        """the mesh axis `param` (default: any param) has replicas along, if any."""
        dim = self.dim
        if param is not None and self._dim_of_param is not None:
            dim = self._dim_of_param(param)
        return dim if dim is not None and self.mesh.size(dim) > 1 else None

    def _build_buckets(self, params: list[nn.Parameter], max_bucket_bytes: int) -> list[_Bucket]:
        """
        greedy buckets of up to max_bucket_bytes, in the given order (reverse
        parameter order, so roughly the order backward produces grads). a new
        bucket also starts where dtype or reduce axis changes.
        """
        buckets = []
        run = []
        run_bytes = 0

        for param in params:
            n_bytes = param.numel() * param.element_size()
            if run and (
                run_bytes + n_bytes > max_bucket_bytes
                or param.dtype != run[0].dtype
                or self._axis(param) != self._axis(run[0])
            ):
                buckets.append(_Bucket(run, self._axis(run[0])))
                run = []
                run_bytes = 0
            run.append(param)
            run_bytes += n_bytes

        if run:
            buckets.append(_Bucket(run, self._axis(run[0])))
        return buckets

    def _on_grad_ready(self, param: nn.Parameter) -> None:
        if not self.requires_sync:
            return

        self._bucket_of[param].pending -= 1
        self._launch_ready()

    def _launch_ready(self, force: bool = False) -> None:
        """
        launch buckets strictly in bucket order, stopping at the first one
        still owed a grad. `force` (from sync) launches the rest regardless.
        """
        while self._next < len(self._buckets):
            bucket = self._buckets[self._next]
            if bucket.pending > 0 and not force:
                return
            self._launch(bucket)
            self._next += 1

    def _launch(self, bucket: _Bucket) -> None:
        bucket.pending = len(bucket.params)
        # no grad this step: this rank contributes zeros to the average
        for param in bucket.params:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
        if bucket.flat is None:
            grad = bucket.params[0].grad
        else:
            torch._foreach_copy_(bucket.views, [param.grad for param in bucket.params])
            for param, view in zip(bucket.params, bucket.views, strict=True):
                param.grad = view
            grad = bucket.flat
        if self.reduce_dtype is not None and grad.dtype != self.reduce_dtype:
            wire = grad.to(self.reduce_dtype)
            _, work = all_reduce(wire, self.mesh, bucket.dim, "avg", async_op=True)
            self._works.append((work, wire, grad))
        else:
            _, work = all_reduce(grad, self.mesh, bucket.dim, "avg", async_op=True)
            self._works.append((work, grad, None))

    @property
    def shard_dims(self) -> tuple[str, ...]:
        """mesh axes param/grad state is sharded over; none for ddp."""
        return ()

    def shard_dim_of(self, param: nn.Parameter) -> int | None:
        """the tensor dim `param` is sharded along; ddp shards nothing."""
        return None

    def full_shape_of(self, param: nn.Parameter) -> tuple[int, ...]:
        return tuple(param.shape)

    def _replicate(self, model: nn.Module):
        for param in model.parameters():
            if self._axis(param) is not None:
                broadcast(param.detach(), self.mesh, self._axis(param), src=0)
        if self._axis() is not None:
            for buf in model.buffers():
                broadcast(buf, self.mesh, self.dim, src=0)

    def sync(self) -> None:
        """launch any buckets still owed a grad, then wait for every reduce."""
        if not self._buckets:
            return
        assert self.requires_sync, (
            "sync() with requires_sync False: it must be True for the last "
            "microbatch of an accumulation step, so its backward launches the reduces"
        )
        self._launch_ready(force=True)
        for work, wire, grad in self._works:
            work.wait()
            if grad is not None:
                grad.copy_(wire)
        self._works.clear()
        self._next = 0
