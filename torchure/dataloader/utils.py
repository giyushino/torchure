"""
build helper for the dataloader, colocated with DataLoader
"""

from torchure.dataloader.dataloader import DataLoader


def build_dataloader(data_config: dict, rank: int = 0, world_size: int = 1) -> DataLoader:
    """
    TODO: build the dataloader from data_config. takes rank/world_size now
    (single gpu defaults) so adding per-rank sharding for data parallel
    later is a signature you already have, not a rewrite.
    """
    raise NotImplementedError
