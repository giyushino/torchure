"""
normally for AR models we don't need to have
this objective wrapper, but if we want to expand
to different model archs / learning objectives, 
it's good to have this abstraction? maybe we can change
"""

import torch

import torch.nn as nn

from torchure.loss.cross_entropy import cross_entropy_loss

class ARObjective():
    def __init__(self, ignore_index: int=-100):
        self.ignore_index = ignore_index

    def compute_loss(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = model(batch["input_ids"])
        return cross_entropy_loss(batch["labels"], logits, self.ignore_index)

if __name__ == "__main__":
    test = ARObjective()
    print("works")




