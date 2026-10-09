"""
shared model/tokenizer/checkpoint loading for one-off inference scripts
(perplexity eval, chat repl). these run outside the Trainer/DDP path
entirely -- single process, straight off a checkpoint dir on disk.
"""
import json
import os

import torch

from tokenizers import Tokenizer

from torchure.models.builder import build_model
from torchure.utils import get_project_dir


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def load_model_and_tokenizer(config: dict, step: int, device: str):
    tokenizer = Tokenizer.from_pretrained(config["data"]["tokenizer"])
    model = build_model(config["model"]["name"], config["model"]["config"])

    ckpt_dir = os.path.join(get_project_dir(), "checkpoints", config["run_name"], str(step))
    state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    model.to(device).eval()
    return model, tokenizer
