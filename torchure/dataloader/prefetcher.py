"""
host->device prefetching for the training loop.

the prefetcher always holds one batch the loop hasn't consumed, so the
dataloader's own state is one batch ahead of training. a checkpoint saves that
held batch (`state_dict`) and resume hands it back as `first_batch`, so the
resumed run sees every batch exactly once.
"""

import torch


class CUDAPrefetcher:
    """
    copies batch N+1 to the gpu on a side stream while batch N computes.
    needs pin_memory=True on the dataloader, or the copy silently blocks.
    """
    def __init__(
        self,
        loader_iter,
        device: torch.device,
        first_batch: dict[str, torch.Tensor] | None = None,
    ):
        self.loader_iter = loader_iter
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_batch: dict[str, torch.Tensor] | None = None
        # first_batch: the held batch from a checkpoint; it goes out before
        # anything is pulled from the (already advanced) loader
        self._preload(first_batch)

    def state_dict(self) -> dict:
        """the batch pulled from the loader but not yet handed to training."""
        if self.next_batch is None:
            return {"next_batch": None}
        torch.cuda.current_stream().wait_stream(self.stream)
        return {"next_batch": {k: v.cpu() for k, v in self.next_batch.items()}}

    def _preload(self, batch: dict[str, torch.Tensor] | None = None) -> None:
        if batch is None:
            try:
                batch = next(self.loader_iter)
            except StopIteration:
                self.next_batch = None
                return
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: v.to(self.device, non_blocking=True) for k, v in batch.items()
            }

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self.next_batch is None:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        # so the allocator doesn't free them while compute still uses them
        for v in batch.values():
            v.record_stream(torch.cuda.current_stream())
        self._preload()
        return batch
