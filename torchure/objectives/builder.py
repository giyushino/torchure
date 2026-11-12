"""
objective registry. objectives own the loss, so models stay decoupled from
loss functions.
"""

from torchure.objectives.autoregressive import ARObjective

OBJECTIVE_REGISTRY = {
    "AR": ARObjective,
}


def build_objective(objective_name: str, objective_config: dict):
    assert objective_name in OBJECTIVE_REGISTRY, "objective not in registry"
    return OBJECTIVE_REGISTRY[objective_name](**objective_config)
