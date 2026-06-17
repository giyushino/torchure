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


def build_optimizer(model: nn.Module, optim_config: dict) -> optim.Optimizer:
    """
    NOTE: must be called AFTER the model is parallelized/wrapped. once
    sharded the parameters change identity (become DTensors / get
    flat-grouped by fsdp), so an optimizer built earlier closes over the
    wrong tensors.

    TODO: pop the optimizer "name" from optim_config, look it up in
    OPTIMIZER_REGISTRY, and build it over model.parameters() with the
    remaining kwargs (lr, weight_decay, betas, ...).
    """
    raise NotImplementedError
