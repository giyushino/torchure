"""
autoregressive next-token objective
"""

import torch

import torch.nn as nn

from torchure.loss.cross_entropy import cross_entropy_loss
from torchure.loss.fused_linear_ce import fused_linear_cross_entropy


class ARObjective():
    def __init__(
        self,
        ignore_index: int = -100,
        fused_linear_ce: bool = True,
        ce_chunk_size: int = 4096,
    ):
        # fused_linear_ce needs the model to expose lm_head.weight and accept
        # return_final_hidden
        self.ignore_index = ignore_index
        self.fused_linear_ce = fused_linear_ce
        self.ce_chunk_size = ce_chunk_size

    def compute_loss(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.fused_linear_ce:
            hidden = model(batch["input_ids"], return_final_hidden=True)
            # next-token shift; the last position has no target
            labels = torch.full_like(batch["labels"], self.ignore_index)
            labels[:, :-1] = batch["labels"][:, 1:]
            return fused_linear_cross_entropy(
                hidden,
                model.lm_head.weight,
                labels,
                self.ignore_index,
                self.ce_chunk_size,
            )

        logits = model(batch["input_ids"])
        return cross_entropy_loss(batch["labels"], logits, self.ignore_index)

if __name__ == "__main__":
    test = ARObjective()
    print("works")
