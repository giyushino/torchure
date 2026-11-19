import torch.nn as nn

from torchure.models.kimi_k31.kimi_k31 import KimiK31
from torchure.models.olmo3.olmo3 import Olmo3
from torchure.models.qwen3.qwen3 import Qwen3

MODEL_REGISTRY = {
    "qwen3_dense": Qwen3,
    "olmo3": Olmo3,
    "kimi_k31": KimiK31,
}


def build_model(model_name: str, model_config: dict) -> nn.Module:
    assert model_name in MODEL_REGISTRY, (
        f"model {model_name!r} is not implemented; registry: {list(MODEL_REGISTRY)}"
    )
    return MODEL_REGISTRY[model_name](**model_config)
