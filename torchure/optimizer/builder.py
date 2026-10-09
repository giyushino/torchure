"""
registry + build helpers for optimizers and schedulers. weight decay applies
only to >=2D params; biases and norm weights are exempt.
"""

import torch.nn as nn
import torch.optim as optim

from torchure.optimizer.scheduler import WarmupStableDecaySchedulder, CosineAnnealingScheduler


OPTIMIZER_REGISTRY = {
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
}

SCHEDULER_REGISTRY = {
    "wsd": WarmupStableDecaySchedulder,
    "cosine": CosineAnnealingScheduler
}


def build_optimizer(model: nn.Module, optim_name: str, optim_config: dict) -> optim.Optimizer:
    cfg = dict(optim_config)
    weight_decay = cfg.pop("weight_decay", 0.0)

    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    nodecay_params = [p for p in params if p.dim() < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    return OPTIMIZER_REGISTRY[optim_name](param_groups, **cfg)


def build_scheduler(optimizer: optim.Optimizer, scheduler_config: dict) -> optim.lr_scheduler.LRScheduler:
    scheduler_cfg = dict(scheduler_config)
    schedulder_type = scheduler_cfg["type"]
    if schedulder_type == "cosine":
        raise NotImplementedError("cosine scheduler is a stub (get_lr returns 0); use wsd")

    scheduler_cfg.pop("type")
    scheduler_cfg["optimizer"] = optimizer
    return SCHEDULER_REGISTRY[schedulder_type](**scheduler_cfg)

