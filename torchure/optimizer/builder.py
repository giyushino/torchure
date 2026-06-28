import torch

def build_optimizer(model, lr: int, weight_decay: float, betas: tuple[float | torch.Tensor, float | torch.Tensor], use_fused: bool):
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    
    # larger dim tensors have weight decay
    decay_params = [p for p in params.values() if p.dim() >= 2]
    nodecay_params = [p for p in params.values() if p.dim() <  2]
    param_groups = [
        {"params": decay_params,   "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=1e-8, fused=use_fused)
    return optimizer
