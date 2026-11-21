"""
dataloader builder. there is one loader, MixtureLoader (mixture.py): a single
dataset is `data.source`, a mix of several is `data.mix`. no torch DataLoader
and no hf `datasets`.
"""

from tokenizers import Tokenizer

from torchure.dataloader.mixture import MixtureLoader
from torchure.dataloader.sources import Vision


def build_dataloader(
    data_cfg: dict, tokenizer: Tokenizer, ignore_id: int, dp_rank: int, dp_size: int,
    total_steps: int, vision: tuple[int, int] | None = None,
) -> MixtureLoader:
    """
    vision: (image_size, num_image_tokens) of a vlm, for caption sources
    (a source with image_field set).
    """
    return MixtureLoader(
        data_cfg, dp_rank, dp_size, total_steps, tokenizer=tokenizer, ignore_index=ignore_id,
        vision=Vision(*vision) if vision is not None else None,
    )
