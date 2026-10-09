"""
host->device prefetching for the training loop.

TODO: checkpoint save/load skips one batch if we naively resume at N + 1
"""

import torch


class CUDAPrefetcher:
    """
    copies batch N+1 to the gpu on a side stream while batch N computes.
    needs pin_memory=True on the dataloader, or the copy silently blocks.
    """
    def __init__(self, loader_iter, device: torch.device):
        self.loader_iter = loader_iter
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_batch: dict[str, torch.Tensor] | None = None
        self._preload()

    def _preload(self) -> None:
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
