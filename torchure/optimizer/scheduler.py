"""
learning rate scheduler
functions are not typed, maybe we should change
https://docs.pytorch.org/docs/2.12/generated/torch.optim.lr_scheduler.LambdaLR.html
"""
import torch

from torch.optim.lr_scheduler import LRScheduler

class WarmupStableDecaySchedulder(LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_ratio, decay_ratio, last_epoch = -1):
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio
        self.decay_ratio = decay_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        step = self.last_epoch
        decay_phase_start = self.total_steps * (1 - self.decay_ratio)

        warmup_steps = max(1, round(self.total_steps * self.warmup_ratio))
        if step <= warmup_steps:
            scale = (step + 1) / warmup_steps

        elif step >= decay_phase_start:
            num_decay_steps = self.decay_ratio * self.total_steps
            scale = 1 - ((step - decay_phase_start) / num_decay_steps)
            scale = max(0, scale) # clamp to 0

        else:
            scale = 1.0

        return [base_lr * scale for base_lr in self.base_lrs] 


class CosineAnnealingScheduler(LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_ratio, decay_phase, last_epoch):
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio
        self.decay_phase = decay_phase
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        scale = 0 # temp
        return [base_lr * scale for base_lr in self.base_lrs] 
