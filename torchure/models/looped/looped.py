"""
looped (depth-recurrent) qwen3: a prelude of ordinary blocks, a core of
blocks applied `num_loops` times with shared weights, then a coda.

    tokens -> prelude -> [core] x num_loops -> coda -> lm_head

effective depth is prelude + core * num_loops + coda at the parameter count
of prelude + core + coda (cf. "Scaling up Test-Time Compute with Latent
Reasoning", Geiping et al. 2025, minus their random-depth training and state
injection). the loop count can be changed after training (`model.core.num_loops`)
to spend more or less compute per token.

the core is ONE module in `self.blocks`, so fsdp gathers its weights once and
keeps them for every iteration, and compile traces the unrolled loop as one
graph.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from torchure.models.qwen3.qwen3 import Qwen3, Qwen3TransformerBlock


class LoopedCore(nn.Module):
    """`layers` applied in order, `num_loops` times."""

    def __init__(self, layers: list[Qwen3TransformerBlock], num_loops: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.num_loops = num_loops
        # set per forward by LoopedQwen3.run_block
        self.activation_checkpointing = False

    def forward(self, x, cos, sin, attention_mask=None):
        for _ in range(self.num_loops):
            for layer in self.layers:
                # checkpoint per block application: checkpointing the whole
                # core would recompute (and hold) every iteration at once
                if self.activation_checkpointing:
                    x = checkpoint(layer, x, cos, sin, attention_mask, use_reentrant=False)
                else:
                    x = layer(x, cos, sin, attention_mask)
        return x


class LoopedQwen3(Qwen3):
    def __init__(
        self,
        prelude_layers: int,
        core_layers: int,
        num_loops: int,
        coda_layers: int,
        num_heads: int,
        num_kv_heads: int,
        emb_dim: int,
        head_dim: int,
        vocab_size: int,
        ffn_dim: int | None = None,
        norm_eps: float = 1e-6,
        causal: bool = True,
    ):
        num_layers = prelude_layers + core_layers + coda_layers
        super().__init__(
            num_layers, num_heads, num_kv_heads, emb_dim, head_dim, vocab_size,
            ffn_dim, norm_eps, causal,
        )
        blocks = list(self.blocks)
        prelude = blocks[:prelude_layers]
        core = blocks[prelude_layers:prelude_layers + core_layers]
        coda = blocks[prelude_layers + core_layers:]
        self.blocks = nn.ModuleList([*prelude, LoopedCore(core, num_loops), *coda])
        # an index, not a submodule attribute: registering the core twice
        # would duplicate its weights in the state_dict
        self.core_index = prelude_layers
        self.depth = prelude_layers + core_layers * num_loops + coda_layers

    @property
    def core(self) -> LoopedCore:
        return self.blocks[self.core_index]

    def transformer_blocks(self) -> list[Qwen3TransformerBlock]:
        out = []
        for block in self.blocks:
            out += list(block.layers) if block is self.core else [block]
        return out

    def run_block(self, block: nn.Module, x: torch.Tensor, *args) -> torch.Tensor:
        if block is self.core:
            block.activation_checkpointing = (
                self.activation_checkpointing and torch.is_grad_enabled()
            )
            return block(x, *args)
        return super().run_block(block, x, *args)
