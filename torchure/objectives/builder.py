"""
registry to store the objectives
we can have this because we want to
decouple the models from the loss
functions
"""
from torchure.objectives.autoregressive import ARObjective

OBJECTIVE_REGISTRY = {
    "AR": ARObjective,
}


def build_objective(objective_name: str, objective_config: dict):
    assert objective_name in OBJECTIVE_REGISTRY, "objective not in registry"
    return OBJECTIVE_REGISTRY[objective_name](**objective_config)
