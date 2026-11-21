"""
registry + build helpers for optimizers and schedulers.

weight decay applies only to the weights of linear, embedding, conv and
expert-bank layers. biases, norm weights and bare nn.Parameters (gates,
learned queries, log-rates) are exempt, whatever their shape.
"""

import torch.nn as nn
import torch.optim as optim

from torchure.models.moe import GroupedLinear

from torchure.optimizer.scheduler import WarmupStableDecayScheduler, CosineAnnealingScheduler


_DECAYED_MODULES = (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.Conv3d, GroupedLinear)

OPTIMIZER_REGISTRY = {
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
}

SCHEDULER_REGISTRY = {
    "wsd": WarmupStableDecayScheduler,
    "cosine": CosineAnnealingScheduler,
}


def build_optimizer(model: nn.Module, optim_name: str, optim_config: dict) -> optim.Optimizer:
    cfg = dict(optim_config)
    weight_decay = cfg.pop("weight_decay", 0.0)

    # by module type, not p.dim(): a 2-D bare parameter (a per-head gate
    # rate, say) is not a matrix to regularize
    decayed = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, _DECAYED_MODULES) and module.weight is not None
    }
    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if id(p) in decayed]
    nodecay_params = [p for p in params if id(p) not in decayed]
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    return OPTIMIZER_REGISTRY[optim_name](param_groups, **cfg)


def build_scheduler(optimizer: optim.Optimizer, scheduler_config: dict) -> optim.lr_scheduler.LRScheduler:
    scheduler_cfg = dict(scheduler_config)
    scheduler_type = scheduler_cfg.pop("type")
    scheduler_cfg["optimizer"] = optimizer
    return SCHEDULER_REGISTRY[scheduler_type](**scheduler_cfg)
