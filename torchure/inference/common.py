"""
shared model/tokenizer/checkpoint loading for one-off inference scripts
(perplexity eval, chat repl). these run outside the Trainer/DDP path
entirely -- single process, straight off a checkpoint dir on disk.
"""
from tokenizers import Tokenizer

from torchure.checkpoint.checkpointer import Checkpointer
from torchure.models.builder import build_model
from torchure.config import load_config  # noqa: F401 -- re-exported for the scripts
from torchure.utils import checkpoint_dir


def load_model_and_tokenizer(config: dict, step: int, device: str):
    tokenizer = Tokenizer.from_pretrained(config["data"]["tokenizer"])
    model = build_model(config["model"]["name"], config["model"]["config"])

    # the default (unsharded) layout: a checkpoint saved sharded by fsdp
    # and/or tp is reassembled from its shard files into the full weights
    checkpointer = Checkpointer(checkpoint_dir(config))
    assert checkpointer.valid_step(step) is not None, (
        f"no complete checkpoint at step {step} in {checkpointer.checkpoint_save_path}"
    )
    checkpointer.load_model(model, step)

    model.to(device).eval()
    return model, tokenizer
