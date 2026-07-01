"""
registry + build helpers for optimizers (and their schedulers)

optimizers are a swappable, config-selected component like objectives, so they
get the same registry treatment instead of being special-cased in the trainer.

on top of registry selection, params are split into two groups: weight decay
applies only to >=2D tensors (matmuls, embeddings), while 1D params (biases,
norm/scale weights) are exempted -- the standard decoupled-weight-decay
convention. everything else in optim_config (lr, betas, fused, ...) is passed
straight through, so it stays config-driven per optimizer.
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
    # weight decay is applied per param group, not globally, so pull it out of
    # the flat config and hand each group its own value below. copy first so we
    # don't mutate the caller's config dict.
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
    scheduler_cfg.pop("type")
    scheduler_cfg["optimizer"] = optimizer
    return SCHEDULER_REGISTRY[schedulder_type](**scheduler_cfg)

