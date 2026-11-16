"""
host->device prefetching for the training loop.

the prefetcher always holds one batch the loop hasn't consumed, so the
dataloader's own state is one batch ahead of training. a checkpoint saves that
held batch (`state_dict`) and resume hands it back as `first_batch`, so the
resumed run sees every batch exactly once.

a loader that runs out is restarted through `restart` (the next pass over the
data), so a run longer than the dataset keeps going. each dp rank restarts on
its own, whenever its own shards run out.
"""

from collections.abc import Callable, Iterator

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
        restart: Callable[[], Iterator] | None = None,
    ):
        self.loader_iter = loader_iter
        # called when the loader runs out; returns the iterator of the next
        # pass. None: the prefetcher just ends (StopIteration)
        self.restart = restart
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
                if self.restart is None:
                    self.next_batch = None
                    return
                self.loader_iter = self.restart()
                try:
                    batch = next(self.loader_iter)
                except StopIteration:
                    raise RuntimeError(
                        "the dataloader yielded no batch on a fresh pass: this "
                        "rank's share of the dataset is smaller than one micro batch"
                    ) from None
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
