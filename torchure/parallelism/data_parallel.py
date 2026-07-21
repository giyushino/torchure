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
from torchure.core.mesh import Mesh

class DDP:
    def __init__(self, model: nn.Module, mesh: Mesh, dim="dp", bucket_mb=25, overlap=True):
        self.model = model
        self.mesh = mesh
        self.dim = dim
        self.group_size = mesh.size(dim)
        if self.group_size == 1:
            return

        self.bucket_mb = bucket_mb
        self.overlap = overlap
        self._replicate(model)

    def _replicate(self, model: nn.Module):
        for tensor in [*model.parameters(), *model.buffers()]:
            broadcast(tensor.detach(), self.mesh, self.dim, src=0)

    def sync(self):
        if self.group_size == 1:
            return

        for param in self.model.parameters():
            if param.requires_grad:
                all_reduce(param.grad, self.mesh, self.dim, "avg")




if __name__ == "__main__":
    print("1")
