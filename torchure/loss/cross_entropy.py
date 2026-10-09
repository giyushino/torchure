import torch
import torch.nn.functional as F

# TODO: move shifting out of the loss so other objectives can reuse it
def cross_entropy_loss_naive(tokenized_input: torch.Tensor, logits: torch.Tensor, ignore_index: int) -> torch.Tensor:
    B, S, V = logits.shape
    shifted_logits = logits[:, :-1 ,:]
    shifted_labels = tokenized_input[:, 1:]
    return F.cross_entropy(
        shifted_logits.reshape(-1, V),
        shifted_labels.reshape(-1),
        ignore_index=ignore_index
    )

def cross_entropy_loss(tokenized_input: torch.Tensor, logits: torch.Tensor, ignore_index: int) -> torch.Tensor:
    B, S, V = logits.shape
    labels = torch.full_like(tokenized_input, ignore_index)
    labels[:, 0:S-1] = tokenized_input[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        ignore_index=ignore_index
    )

if __name__ == "__main__":
    test_input = torch.randint(1, 1000, [2, 100])
    test_logits = torch.randn(2, 100, 2500)
    loss1 = cross_entropy_loss_naive(test_input, test_logits, -100)
    loss2 = cross_entropy_loss(test_input, test_logits, -100)
    print(loss1, loss2)
    # naive: tensor(8.2090) 0.020412254030816257
    # optimized: tensor(8.2090) 0.017415661015547812


