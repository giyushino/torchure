import torch
import torch.nn.functional as F


def cross_entropy_loss(tokenized_input: torch.Tensor, logits: torch.Tensor, ignore_index: int) -> torch.Tensor:
    B, S, V = logits.shape
    # for the seq length, throw away the last token
    # since we don't have a ground truth for that
    shifted_logits = logits[:, :-1 ,:]
    
    # for the ground truth, remove the first token
    # since we don't have any context for it
    shifted_labels = tokenized_input[:, 1:]

    # we're missing the tokenizer padding right
    # now, not sure how to set this. probably
    # pass it as a param
    return F.cross_entropy(
        shifted_logits.reshape(-1, V),
        shifted_labels.reshape(-1),
        ignore_index=ignore_index
    )


if __name__ == "__main__":
    test_input = torch.randint(1, 1000, [2, 100])
    test_logits = torch.randn(2, 100, 2500)
    loss = cross_entropy_loss(test_input, test_logits, -100)
    print(loss)
