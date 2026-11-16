"""
device mesh: the world's ranks arranged as an n-d grid with named axes, e.g.
{"dp": 2, "tp": 4} for 8 ranks. ranks are laid out row-major, so the last
axis varies fastest: its peers are adjacent ranks (stride 1).

along each axis a rank belongs to one ProcessGroup: the ranks that share its
coordinates on every OTHER axis. collectives (core/collective.py) address
those groups by axis name.

    spec {"dp": 2, "tp": 4}        strides {"dp": 4, "tp": 1}
    rank 5 -> dp coordinate 1, tp coordinate 1
    its tp group is ranks [4, 5, 6, 7], its dp group is ranks [1, 5]
"""

import math

import torch.distributed as dist

# fixed, because order matters: the last axis gets stride 1 (its peers are
# adjacent ranks), and flatten() needs its axes in spec order
AXIS_ORDER = ("pp", "dp_replicate", "dp_shard", "cp", "tp", "ep")


class Mesh:
    def __init__(self, spec: dict[str, int]):
        assert dist.is_initialized(), "init_process_group() must be called before building a Mesh"
        self.world_size = dist.get_world_size()
        assert self.world_size == math.prod(spec.values()), (
            f"mesh {spec} needs {math.prod(spec.values())} ranks, world size is {self.world_size}"
        )

        self.sizes = spec
        self.strides = self._build_strides()
        self.groups = self._build_groups()
        # coordinates along flattened axes, which strides can't express
        self._flat_coords: dict[str, int] = {}

    def _build_strides(self) -> dict[str, int]:
        """how far apart in rank two neighbours along each axis are."""
        strides = {}
        stride = 1
        for axis, size in reversed(self.sizes.items()):
            strides[axis] = stride
            stride *= size
        return strides

    def _groups_along(self, axis: str):
        """yield the member ranks of every group along `axis`."""
        size = self.size(axis)
        stride = self.strides[axis]
        block = size * stride  # ranks spanned by one full sweep of this axis
        for outer in range(self.world_size // block):
            for inner in range(stride):
                base = outer * block + inner
                yield [base + coord * stride for coord in range(size)]

    def _build_groups(self) -> dict[str, dist.ProcessGroup]:
        # new_group is collective: every rank creates every group, in the same
        # order, and keeps the ones it belongs to
        groups = {}
        rank = dist.get_rank()
        for axis in self.sizes:
            for members in self._groups_along(axis):
                group = dist.new_group(members)
                if rank in members:
                    groups[axis] = group
        return groups

    def flatten(self, axes: tuple[str, ...], name: str) -> None:
        """
        register a virtual axis `name` treating `axes` as one row-major axis,
        e.g. dp = (dp_replicate, dp_shard).

        `axes` must be in spec order so coordinate(name) matches all_gather
        order. collective: every rank must call this identically.
        """
        assert axes and len(set(axes)) == len(axes), f"bad axes {axes}"
        assert all(ax in self.strides for ax in axes), f"unknown axis in {axes}"
        spec_order = [ax for ax in self.sizes if ax in axes]
        assert list(axes) == spec_order, f"{axes} must follow spec order {spec_order}"
        assert name not in self.sizes, f"axis {name!r} already exists"

        # ranks with the same coords on the other axes form one flat group.
        # subtracting a rank's offset along `axes` leaves the group's first
        # rank, which keys the bucket
        buckets: dict[int, list[int]] = {}
        for r in range(self.world_size):
            offset = sum(
                ((r // self.strides[ax]) % self.sizes[ax]) * self.strides[ax]
                for ax in axes
            )
            buckets.setdefault(r - offset, []).append(r)

        rank = dist.get_rank()
        for members in buckets.values():  # same order on every rank
            group = dist.new_group(members)
            if rank in members:
                self.groups[name] = group
                self._flat_coords[name] = members.index(rank)

        self.sizes[name] = math.prod(self.sizes[ax] for ax in axes)

    def size(self, dim: str) -> int:
        return self.sizes[dim]

    def get_group(self, dim: str) -> dist.ProcessGroup:
        return self.groups[dim]

    def coordinate(self, dim: str) -> int:
        """this rank's index within its `dim` group."""
        if dim in self._flat_coords:
            return self._flat_coords[dim]
        return (dist.get_rank() // self.strides[dim]) % self.sizes[dim]


def build_mesh(spec: dict[str, int]) -> Mesh:
    """
    the mesh a training run uses, from a config's mesh spec: axes put in
    AXIS_ORDER, `dp: N` read as shorthand for `dp_shard: N`, and the derived
    axes added:

        dp      every rank that reads its own data: dp_replicate x dp_shard,
                and x ep under expert parallel (ep ranks are data-parallel
                for everything but the experts). the dataloader split, loss
                averaging and dense grad sync all run over it.

    and, only with ep > 1:

        dp_expert         dp_replicate x dp_shard: the ranks holding a
                          replica of the same experts
        dp_replicate_ep   dp_replicate x ep: the ranks holding a whole
                          replica of a dense fsdp shard
    """
    spec = dict(spec)
    assert not ("dp" in spec and ("dp_replicate" in spec or "dp_shard" in spec)), (
        f"mesh spec {spec}: `dp` is shorthand for `dp_shard`; don't mix it with the explicit axes"
    )
    if "dp" in spec:
        spec["dp_shard"] = spec.pop("dp")
    unknown = set(spec) - set(AXIS_ORDER)
    assert not unknown, f"mesh spec has unknown axes {sorted(unknown)}; known: {AXIS_ORDER}"

    ordered = {}
    for axis in AXIS_ORDER:
        size = spec.get(axis, 1)
        # size-1 axes cost world_size new_group calls, so drop them, except
        # the dp axes, which flatten() needs
        if size > 1 or axis in ("dp_replicate", "dp_shard"):
            ordered[axis] = size

    mesh = Mesh(ordered)
    if "ep" in ordered:
        mesh.flatten(("dp_replicate", "dp_shard", "ep"), "dp")
        mesh.flatten(("dp_replicate", "dp_shard"), "dp_expert")
        mesh.flatten(("dp_replicate", "ep"), "dp_replicate_ep")
    else:
        mesh.flatten(("dp_replicate", "dp_shard"), "dp")
    return mesh
