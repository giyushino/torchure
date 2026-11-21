"""
masked diffusion objective (MDLM, Sahoo et al. 2024 / LLaDA, Nie et al. 2025).

forward process: each sequence draws a noise level t ~ U(eps, 1) and every
token is independently replaced by [MASK] with probability t. the model sees
the noised sequence with bidirectional attention (model config
`causal: false`) and predicts the original token at every masked position.

loss, an upper bound on the sequence nll:

    E_t [ (1/t) * sum over masked i of -log p(x_i | x_t) ] / num_tokens

no next-token shift: position i predicts its own token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchure.loss.fused_linear_ce import fused_linear_cross_entropy
from torchure.objectives.autoregressive import model_inputs


class MaskedDiffusionObjective:
    def __init__(
        self,
        mask_token_id: int,
        ignore_index: int = -100,
        # floor on t, so 1/t stays bounded
        eps: float = 1e-3,
        fused_linear_ce: bool = True,
        ce_chunk_size: int = 4096,
    ):
        self.mask_token_id = mask_token_id
        self.ignore_index = ignore_index
        self.eps = eps
        self.fused_linear_ce = fused_linear_ce
        self.ce_chunk_size = ce_chunk_size
        # set by the trainer, as for ARObjective
        self.tp = None

    def noise(
        self, ids: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(noised ids, targets with ignore_index off the masked positions, t per row)."""
        B, S = ids.shape
        t = self.eps + (1 - self.eps) * torch.rand(B, 1, device=ids.device)
        masked = (torch.rand(B, S, device=ids.device) < t) & (labels != self.ignore_index)
        noised = torch.where(masked, self.mask_token_id, ids)
        targets = torch.where(masked, labels, self.ignore_index)
        return noised, targets, t

    def compute_loss(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch["labels"]
        noised, targets, t = self.noise(batch["input_ids"], labels)
        num_tokens = (labels != self.ignore_index).sum().clamp_min(1)
        weights = (1 / t).expand_as(targets)

        if self.fused_linear_ce:
            hidden = model(noised, return_final_hidden=True, **model_inputs(batch))
            if self.tp is not None:
                hidden = self.tp.head_input(hidden)
            # the fused loss averages over the masked tokens; rescale so the
            # result averages over all tokens instead
            num_masked = (targets != self.ignore_index).sum().clamp_min(1)
            return fused_linear_cross_entropy(
                hidden,
                model.lm_head.weight,
                targets,
                self.ignore_index,
                self.ce_chunk_size,
                vocab_axis=self.tp.vocab_axis if self.tp is not None else None,
                weights=weights * (num_masked / num_tokens),
            )

        logits = model(noised, **model_inputs(batch))
        nll = F.cross_entropy(
            logits.flatten(0, 1).float(), targets.flatten(),
            ignore_index=self.ignore_index, reduction="none",
        )
        return (nll * weights.flatten()).sum() / num_tokens
