"""
ddp: replicate the model along the mesh's dp axis and keep replicas in
lockstep by averaging grads across the axis after every backward.

- buckets follow reverse `model.parameters()` order (roughly backward order),
  so all-reduces launch from grad hooks while backward is still running.
- small params share a flat buffer per bucket: grads are copied in, `.grad`
  is repointed at views of it, and the buffer is all-reduced in place.
- a param at or over the bucket cap (e.g. the tied embedding) is all-reduced
  directly through its `.grad`, with no staging copy.
- `sync()` waits on all handles; call it between backward and optimizer.step.
- `requires_sync` is the no_sync switch for grad accumulation: False for the
  first K-1 microbatches, True for the last.
- buckets are all-reduced in bucket order, on every rank, whenever each
  becomes ready. a param that got no grad this step (an moe expert no token
  picked, a param the forward never used) leaves its bucket unready; `sync()`
  launches what is left with zeros for the missing grads. ranks may differ in
  which grads are missing: the fixed order keeps their collectives paired.

one DDP instance per model: hooks are registered once and never removed.
"""

import torch
import torch.nn as nn

from torchure.core.collective import MeshLike, all_reduce, broadcast


class _Bucket:
    """
    params whose grads are all-reduced together. with more than one param the
    grads live in `flat`, each param's `.grad` being one of its `views`; a
    lone param is reduced through its own `.grad`.
    """

    def __init__(self, params: list[nn.Parameter]):
        self.params = params
        self.pending = len(params)  # params still owing a grad this backward
        self.launched = False       # all-reduced already this step
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
    def __init__(self, model: nn.Module, mesh: MeshLike, dim="dp", bucket_mb=25):
        self.mesh = mesh
        self.dim = dim
        self.group_size = mesh.size(dim)
        self._works = []
        self._next = 0  # index of the next bucket to launch this step
        self.requires_sync = True

        if self.group_size == 1:
            return

        self._replicate(model)
        params = [param for param in model.parameters() if param.requires_grad]
        params.reverse()
        self._buckets = self._build_buckets(params, int(bucket_mb * 1024 * 1024))
        self._bucket_of = {p: b for b in self._buckets for p in b.params}
        for param in params:
            param.register_post_accumulate_grad_hook(self._on_grad_ready)

    def _build_buckets(self, params: list[nn.Parameter], max_bucket_bytes: int) -> list[_Bucket]:
        """params should already be reversed."""
        buckets = []
        run = []
        run_bytes = 0

        for param in params:
            n_bytes = param.numel() * param.element_size()
            if run and (run_bytes + n_bytes > max_bucket_bytes or param.dtype != run[0].dtype):
                buckets.append(_Bucket(run))
                run = []
                run_bytes = 0
            run.append(param)
            run_bytes += n_bytes

        if run:
            buckets.append(_Bucket(run))
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
        _, work = all_reduce(grad, self.mesh, self.dim, "avg", async_op=True)
        self._works.append(work)

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
        for tensor in [*model.parameters(), *model.buffers()]:
            broadcast(tensor.detach(), self.mesh, self.dim, src=0)

    def sync(self) -> None:
        if self.group_size == 1:
            return
        assert self.requires_sync, (
            "sync() with requires_sync False: it must be True for the last "
            "microbatch of an accumulation step, so its backward launches the reduces"
        )
        # buckets still owed a grad (see the module docstring)
        self._launch_ready(force=True)
        for work in self._works:
            work.wait()
        self._works.clear()
        self._next = 0
