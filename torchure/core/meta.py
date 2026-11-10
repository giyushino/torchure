"""
turning a model built on the `meta` device (shapes only, no storage) into a
real one without ever allocating more than the caller asks for.
"""

from collections.abc import Callable

import torch
import torch.nn as nn


def materialize(
    model: nn.Module,
    device: torch.device,
    local_shape: Callable[[nn.Parameter], tuple[int, ...]] | None = None,
) -> dict[int, tuple[int, ...]]:
    """
    give a meta model real, UNINITIALIZED storage on `device`. each param is
    allocated at `local_shape(param)` (default: its own shape), buffers at
    their full shape. run the model's init afterwards.

    a meta Parameter can't be repointed at device storage, so each one is
    replaced, once, wherever it is registered -- which keeps a tied weight
    tied. returns {id(new param): the meta param's shape}.
    """
    new: dict[int, nn.Parameter] = {}
    shapes: dict[int, tuple[int, ...]] = {}
    for module in model.modules():
        for name, old in module._parameters.items():
            if old is None:
                continue
            if id(old) not in new:
                shape = tuple(old.shape) if local_shape is None else local_shape(old)
                param = nn.Parameter(
                    torch.empty(shape, dtype=old.dtype, device=device),
                    requires_grad=old.requires_grad,
                )
                new[id(old)] = param
                shapes[id(param)] = tuple(old.shape)
            module._parameters[name] = new[id(old)]
        for name, buf in module._buffers.items():
            if buf is not None:
                module._buffers[name] = torch.empty(buf.shape, dtype=buf.dtype, device=device)
    return shapes
