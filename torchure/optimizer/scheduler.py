"""
learning rate scheduler
functions are not typed, maybe we should change
https://docs.pytorch.org/docs/2.12/generated/torch.optim.lr_scheduler.LambdaLR.html
"""
import math

import torch

from torch.optim.lr_scheduler import LRScheduler

class WarmupStableDecaySchedulder(LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_ratio, decay_ratio, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio
        self.decay_ratio = decay_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        step = self.last_epoch
        decay_phase_start = self.total_steps * (1 - self.decay_ratio)

        warmup_steps = max(1, round(self.total_steps * self.warmup_ratio))
        # strict <: with the +1 numerator, step W-1 already hits scale 1.0;
        # letting step == W in here would overshoot to (W+1)/W
        if step < warmup_steps:
            scale = (step + 1) / warmup_steps

        elif step >= decay_phase_start:
            num_decay_steps = self.decay_ratio * self.total_steps
            scale = 1 - ((step - decay_phase_start) / num_decay_steps)
            scale = max(0, scale) # clamp to 0

        else:
            scale = 1.0

        return [base_lr * scale for base_lr in self.base_lrs] 


# we'll just have warmup regardless
class CosineAnnealingScheduler(LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_ratio, min_lr_ratio, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        step = self.last_epoch
        warmup_steps = max(1, round(self.total_steps * self.warmup_ratio))

        if step < warmup_steps:
            scale = (step + 1) / warmup_steps
        
        else:
            progress = (step - warmup_steps) / max(1, self.total_steps - warmup_steps)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))

        return [base_lr * scale for base_lr in self.base_lrs]


class TrollScheduler(LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        import random
        return [base_lr * random.random() for base_lr in self.base_lrs] 
