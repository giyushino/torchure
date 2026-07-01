"""
host->device prefetching for the training loop.

the DataLoader (with num_workers > 0) already prefetches collated CPU batches
into a queue, so next(iter) usually pops a ready batch. what's still serial is
the host->device copy: by default it runs on the compute stream and blocks the
next step. CUDAPrefetcher issues that copy on a side stream so it overlaps the
previous step's kernels, and only makes the compute stream wait right before it
needs the batch.
"""

import torch


class CUDAPrefetcher:
    """
    overlaps the host->device copy of batch N+1 with the compute of batch N.

    the copy runs on a dedicated CUDA stream so it doesn't serialize behind the
    default (compute) stream. right before we hand a batch to the caller we make
    the compute stream wait on the copy stream, and record_stream the tensors so
    the caching allocator doesn't recycle their memory while the async copy is
    still in flight.

    requires the source tensors to be in pinned memory (pin_memory=True on the
    DataLoader) -- otherwise .to(non_blocking=True) silently falls back to a
    blocking copy and nothing overlaps.
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
        # queue the copy on the side stream; returns immediately on the host.
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: v.to(self.device, non_blocking=True) for k, v in batch.items()
            }

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self.next_batch is None:
            raise StopIteration
        # ensure the copy issued on self.stream is complete before compute uses it.
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        # these tensors were produced on self.stream; tell the allocator the
        # compute stream now owns them so their memory isn't freed too early.
        for v in batch.values():
            v.record_stream(torch.cuda.current_stream())
        # kick off the copy for the *following* batch, then return the current one.
        self._preload()
        return batch
