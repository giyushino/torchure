"""
registry + build helper for optimizers

optimizers are a swappable, config-selected component like objectives, so
they get the same registry treatment instead of being special-cased in the
trainer.
"""

import torch.nn as nn
import torch.optim as optim

OPTIMIZER_REGISTRY = {
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
}


def build_optimizer(model: nn.Module, optim_name: str, optim_config: dict) -> optim.Optimizer:
    """
    for now, we can just do a very native global optim, ie have the same
    optimizer settings for each param. maybe want to implement a more fine 
    grained optimizer wrapper class for different layers
    """
    return OPTIMIZER_REGISTRY[optim_name](model.parameters(), **optim_config)
