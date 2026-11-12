import torch
import torch.nn.functional as F


# TODO: move shifting out of the loss so other objectives can reuse it
def cross_entropy_loss(
    labels: torch.Tensor, logits: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """
    next-token cross entropy. `labels` are the unshifted input ids, shifted
    here: position i is scored against token i+1, and the last position
    (which has no target) is ignored.
    """
    B, S, V = logits.shape
    targets = torch.full_like(labels, ignore_index)
    targets[:, :-1] = labels[:, 1:]
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=ignore_index)
