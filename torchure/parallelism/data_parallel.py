"""
ddp: replicate the model along the mesh's dp axis and keep replicas in
lockstep by averaging grads across the axis after every backward.

design (roadmap phase 1.2 -- overlap version, since grad sync on a pcie-only
box is the whole ballgame):

- buckets in reverse `model.parameters()` order, which approximates backward
  completion order, so early buckets fill (and their all-reduce launches)
  while the rest of backward is still computing. bucket fill order is the
  hook firing order, which is identical across ranks (same graph, same
  engine schedule), so nccl sees the same collective order everywhere.
- a bucket of small params shares one preallocated flat buffer: grads are
  copied in with a single foreach launch, `.grad` is repointed at views of
  the buffer, and the buffer is all-reduced. the reduced values are then
  already what the optimizer reads -- no copy back out.
- a param at or over the bucket cap gets a bucket of its own and is
  all-reduced in place via its `.grad` directly: no buffer, no copy. this is
  the tied-embedding grad (622MB fp32 at qwen3-0.6B), where a staging copy
  would cost real time and transient memory.
- `sync()` (call between backward and optimizer.step) waits on all handles.
  there is no non-overlap mode: launching from hooks is the only thing that
  makes grad sync affordable on this box, so it is not a knob.

zero_grad interplay: the trainer's `optimizer.zero_grad()` (set_to_none=True)
drops the buffer views, so each backward the engine allocates fresh grads and
the hook copies them in. set_to_none=False also works (the engine then
accumulates straight into the views and the copy degenerates to a no-op);
what does NOT work is multiple backwards per sync -- grad accumulation needs
no_sync semantics (roadmap 1.3), not built yet, and sync() asserts loudly if
a bucket never filled.

one DDP instance per model: hooks are registered once and never removed.
"""

import torch
import torch.nn as nn

from torchure.core.collective import MeshLike, all_reduce, broadcast


class _Bucket:
    def __init__(self, params: list[nn.Parameter]):
        self.params = params
        self.pending = len(params)
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
        self.requires_sync = True

        if self.group_size == 1:
            return

        self._replicate(model)
        params = [param for param in model.parameters() if param.requires_grad]
        params.reverse()
        # 1024 * 1024 is the number of bytes in mb
        self._buckets = self._build_buckets(params, int(bucket_mb * 1024 * 1024))
        self._bucket_of = {p: b for b in self._buckets for p in b.params}
        for param in params:
            param.register_post_accumulate_grad_hook(self._on_grad_ready)

    def _build_buckets(self, params: list[nn.Parameter], max_bucket_bytes: int) -> list[_Bucket]:
        """
        params should already be revesed when passed into this function
        """
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

        bucket = self._bucket_of[param]
        bucket.pending -= 1
        if bucket.pending == 0:
            self._launch(bucket)

    def _launch(self, bucket: _Bucket) -> None:
        bucket.pending = len(bucket.params)
        if bucket.flat is None:
            grad = bucket.params[0].grad
        else:
            torch._foreach_copy_(bucket.views, [param.grad for param in bucket.params])
            for param, view in zip(bucket.params, bucket.views, strict=True):
                param.grad = view
            grad = bucket.flat
        _, work = all_reduce(grad, self.mesh, self.dim, "avg", async_op=True)
        self._works.append(work)
         
    def _replicate(self, model: nn.Module):
        for tensor in [*model.parameters(), *model.buffers()]:
            broadcast(tensor.detach(), self.mesh, self.dim, src=0)

    def sync(self) -> None:
        if self.group_size == 1:
            return
        assert len(self._works) == len(self._buckets), (
            f"only {len(self._works)}/{len(self._buckets)} grad buckets launched -- "
            "a param got no grad this backward (frozen params / grad "
            "accumulation need no_sync semantics, not built yet)"
        )
        for work in self._works:
            work.wait()
        self._works.clear()


if __name__ == "__main__":
    print("1")
