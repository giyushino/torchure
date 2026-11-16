"""
mixture-of-experts building blocks, shared by any moe model.

the experts of one layer are a BANK: three weights stacked on a leading
expert dim, `(num_experts, out, in)`, instead of num_experts separate
modules. that buys three things:

- dispatch is one sort + one grouped matmul per projection, not a python
  loop of masked selects over the experts.
- every backward gives the bank's params a grad, even when some expert got
  no tokens (its rows are just zero).
- the expert dim is dim 0, which is the dim both fsdp and expert parallel
  shard. an expert-parallel rank simply holds a slice of the bank.

the flow, for N tokens each routed to k experts:

    route      -> expert ids (N, k) and mixing weights (N, k)
    sort_by_expert -> the N*k (token, expert) pairs ordered by expert, and
                  how many each expert got
    bank(x, counts) -> every pair's expert output, same order
    combine    -> weighted sum back onto the N tokens

`bank(x, counts)` is the one step expert parallel replaces: with `bank.remote`
set (parallelism/expert_parallel.py) the pairs travel to the rank holding
their expert and the outputs travel back.
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedLinear(nn.Module):
    """num_groups bias-free linears as one weight (num_groups, out, in)."""

    def __init__(self, num_groups: int, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_groups, out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)

    @property
    def num_groups(self) -> int:
        # read off the weight: expert parallel leaves this rank a slice
        return self.weight.shape[0]

    def forward(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        x (M, in), rows grouped: the first counts[0] rows belong to group 0,
        the next counts[1] to group 1, ... returns (M, out).
        """
        return grouped_linear(x, self.weight, counts)


def grouped_linear(x: torch.Tensor, weight: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """`x[group g's rows] @ weight[g].T` for every group, rows staying in place."""
    weight = weight.to(x.dtype)
    if x.is_cuda and x.dtype == torch.bfloat16:
        # jagged grouped gemm: one kernel for all groups (bf16 cuda only)
        offsets = counts.cumsum(0).to(torch.int32)
        return F.grouped_mm(x, weight.transpose(1, 2), offs=offsets)
    out = []
    start = 0
    for group, count in enumerate(counts.tolist()):
        if count:
            out.append(x[start:start + count] @ weight[group].t())
            start += count
    if not out:
        return x.new_zeros((0, weight.shape[1]))
    return torch.cat(out)


class ExpertBank(nn.Module):
    """
    num_experts gated mlps, `down(act(gate(x), up(x)))`, as three grouped
    linears. `act(gate, up)` is the model's glu.

    `num_experts` is the layer's TOTAL expert count and never changes; under
    expert parallel this rank's weights hold only `num_local` of them and
    `remote` carries rows to the ranks holding the others.
    """

    def __init__(
        self,
        num_experts: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.num_experts = num_experts
        self.out_dim = out_dim
        self.act = act
        self.gate = GroupedLinear(num_experts, in_dim, hidden_dim)
        self.up = GroupedLinear(num_experts, in_dim, hidden_dim)
        self.down = GroupedLinear(num_experts, hidden_dim, out_dim)
        # set by expert parallel: remote(bank, x, counts) -> outputs
        self.remote: Callable | None = None

    @property
    def num_local(self) -> int:
        return self.gate.num_groups

    def local(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """apply this rank's experts; `counts` is per LOCAL expert."""
        return self.down(self.act(self.gate(x, counts), self.up(x, counts)), counts)

    def forward(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """x (M, in) sorted by expert, counts (num_experts,) -> (M, out)."""
        if self.remote is not None:
            return self.remote(self, x, counts)
        return self.local(x, counts)


def sort_by_expert(
    indices: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    indices (N, k): the experts each token is routed to. returns the N*k
    (token, slot) pairs ordered by expert, as (token index, slot index), and
    the number of pairs per expert (num_experts,).
    """
    flat = indices.reshape(-1)
    # stable: pairs of one expert stay in token order, so the layout (and
    # with it the result) doesn't depend on the sort's tie-breaking
    order = flat.argsort(stable=True)
    k = indices.shape[1]
    counts = torch.bincount(flat, minlength=num_experts)
    return order // k, order % k, counts


def balance_stats(load: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    how evenly tokens spread over the experts. `load` (layers, experts) is
    the number of tokens each expert got. shares are relative to uniform, so
    1.0 everywhere is perfect balance; the worst layer is reported.
    """
    share = load / load.mean(dim=-1, keepdim=True).clamp_min(1e-9)
    return {
        "moe/min_load": share.min(),
        "moe/max_load": share.max(),
        "moe/dead_experts": (load == 0).sum().float(),
    }
