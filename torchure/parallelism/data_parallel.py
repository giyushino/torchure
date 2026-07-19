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
- `sync()` (call between backward and optimizer.step) waits on all handles;
  with overlap=False it instead launches every bucket back-to-back there --
  the roadmap 1.1 correctness baseline, kept as the ablation knob for
  measuring what overlap buys.

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
    """one grad-sync unit: either a run of small params sharing a flat
    buffer, or a single large param reduced in place through its .grad."""

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
                self.views.append(self.flat[offset : offset + p.numel()].view_as(p))
                offset += p.numel()


class DDP:
    def __init__(
        self,
        model: nn.Module,
        mesh: MeshLike,
        dim: str = "dp",
        bucket_mb: float = 25,
        overlap: bool = True,
    ):
        self.mesh = mesh
        self.dim = dim
        self.group_size = mesh.size(dim)
        self.overlap = overlap
        self._works = []
        if self.group_size == 1:
            # size-1 dp: replication and grad sync are both no-ops, and no
            # hooks means literally zero per-step overhead
            return

        self._replicate(model)

        # reverse order ~= backward completion order (embedding last: its
        # grad only completes at the very end of backward, tied lm_head or
        # not, so it always reduces with nothing left to overlap)
        params = [p for p in model.parameters() if p.requires_grad]
        params.reverse()
        self._buckets = self._build_buckets(params, int(bucket_mb * (1 << 20)))
        self._bucket_of = {p: b for b in self._buckets for p in b.params}
        if overlap:
            for p in params:
                p.register_post_accumulate_grad_hook(self._on_grad_ready)

    def _replicate(self, model: nn.Module) -> None:
        # ranks agree on the initial replica even if rng drifted; per-tensor
        # broadcasts are ~µs each and this runs once, so no flattening
        for t in [*model.parameters(), *model.buffers()]:
            broadcast(t.detach(), self.mesh, self.dim, src=0)

    def _build_buckets(self, params: list[nn.Parameter], cap_bytes: int) -> list[_Bucket]:
        buckets: list[_Bucket] = []
        run: list[nn.Parameter] = []
        run_bytes = 0

        def close_run():
            nonlocal run, run_bytes
            if run:
                buckets.append(_Bucket(run))
                run, run_bytes = [], 0

        for p in params:
            nbytes = p.numel() * p.element_size()
            if nbytes >= cap_bytes:
                # oversized param: own bucket, reduced in place (no copy)
                close_run()
                buckets.append(_Bucket([p]))
                continue
            if run and (run_bytes + nbytes > cap_bytes or p.dtype != run[0].dtype):
                close_run()
            run.append(p)
            run_bytes += nbytes
        close_run()
        return buckets

    def _on_grad_ready(self, param: nn.Parameter) -> None:
        bucket = self._bucket_of[param]
        bucket.pending -= 1
        if bucket.pending == 0:
            self._launch(bucket)

    def _launch(self, bucket: _Bucket) -> None:
        bucket.pending = len(bucket.params)  # rearm for the next backward
        if bucket.flat is None:
            grad = bucket.params[0].grad
        else:
            # one foreach launch instead of a copy kernel per param, then
            # repoint .grad at the buffer views so the reduced values are
            # what the optimizer reads
            torch._foreach_copy_(bucket.views, [p.grad for p in bucket.params])
            for p, v in zip(bucket.params, bucket.views, strict=True):
                p.grad = v
            grad = bucket.flat
        _, work = all_reduce(grad, self.mesh, self.dim, "avg", async_op=True)
        self._works.append(work)

    def sync(self) -> None:
        """wait for every bucket's grad all-reduce. call between
        loss.backward() and optimizer.step()."""
        if self.group_size == 1:
            return
        if not self.overlap:
            nograds = sum(p.grad is None for b in self._buckets for p in b.params)
            assert nograds == 0, f"{nograds} params got no grad this backward"
            for bucket in self._buckets:
                self._launch(bucket)  # back-to-back on the comm stream
        assert len(self._works) == len(self._buckets), (
            f"only {len(self._works)}/{len(self._buckets)} grad buckets launched -- "
            "a param got no grad this backward (frozen params / grad "
            "accumulation need no_sync semantics, not built yet)"
        )
        for work in self._works:
            work.wait()
        self._works.clear()
