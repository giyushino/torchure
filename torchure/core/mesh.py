"""
mesh
"""

import torch.distributed as dist


def normalize_dp_spec(spec: dict[str, int]) -> dict[str, int]:
    """
    expand a config's mesh spec so the data-parallel axis is always the pair
    (dp_replicate, dp_shard), preserving the order of every other axis.

    **`dp: N` is shorthand for `dp_replicate: N`**, not dp_shard. that keeps
    every existing ddp config bit-identical: the two axes are laid out with
    dp_shard innermost, so at dp_shard=1 the dp_replicate groups are exactly
    the groups the old single `dp` axis produced, and DDP (which runs on the
    flattened `dp`) sees no change at all. an fsdp config asks for sharding
    explicitly with `dp_shard: N`.

    the flattened `dp` axis the dataloader and the loss all-reduce use is
    registered by the caller via mesh.flatten(("dp_replicate","dp_shard"),"dp")
    -- which is also why `dp` must not survive here as a real axis.
    """
    out: dict[str, int] = {}
    for axis, size in spec.items():
        if axis == "dp":
            assert "dp_replicate" not in spec and "dp_shard" not in spec, (
                "mesh spec has both `dp` and an explicit dp_replicate/dp_shard; "
                "use one or the other"
            )
            out["dp_replicate"] = size
            out["dp_shard"] = 1
        elif axis == "dp_replicate":
            out["dp_replicate"] = size
            out.setdefault("dp_shard", spec.get("dp_shard", 1))
        elif axis == "dp_shard":
            out.setdefault("dp_replicate", spec.get("dp_replicate", 1))
            out["dp_shard"] = size
        else:
            out[axis] = size

    assert "dp_shard" in out, (
        f"mesh spec {spec} has no data-parallel axis; expected one of "
        "`dp`, `dp_replicate`, `dp_shard`"
    )
    # dp_replicate must precede dp_shard: Mesh.flatten requires spec order, and
    # that order is what makes coordinate("dp") the row-major index the
    # dataloader splits on
    ordered = {}
    for axis in out:
        if axis == "dp_shard":
            continue
        ordered[axis] = out[axis]
        if axis == "dp_replicate":
            ordered["dp_shard"] = out["dp_shard"]
    return ordered


class Mesh:
    def __init__(self, spec: dict[str, int]):
        assert dist.is_initialized(), "init_process_group() must be called before building a Mesh"
        self.world_size = dist.get_world_size()

        stride_world_size = 1
        for i in spec.values():
            stride_world_size *= i
        assert self.world_size == stride_world_size, f"{self.world_size} is different than {stride_world_size}"

        self.sizes = spec
        self.strides = self._build_strides()
        self.groups = self._build_groups()
        self._flat_coords: dict[str, int] = {}

    def _build_strides(self) -> dict[str, int]:
        strides = {}
        stride_size = 1
        for axis, size in reversed(self.sizes.items()):
            strides[axis] = stride_size
            stride_size *= size

        return strides

    def _group_for(self, axis: str):
        """
        get every group for a given axis
        """
        size = self.size(axis)
        stride = self.strides[axis]
        block = size * stride
        for high in range(self.world_size // block): 
            for low in range(stride):
                base = high * block + low
                yield [base + c * stride for c in range(size)] 

    def _build_groups(self):
        groups = {}
        curr_rank = dist.get_rank()
        for axis in self.sizes:
            for members in self._group_for(axis):
                g = dist.new_group(members)
                if curr_rank in members:
                    groups[axis] = g

        return groups

    def flatten(self, axes: tuple[str, ...], name: str) -> None:
        """
        register a virtual axis `name` treating `axes` as one row-major
        super-axis; size()/coordinate()/get_group() then work on it like a
        real axis. built for "which batch shard am i": the dataloader wants
        one dp coordinate spanning (dp_replicate, dp_shard), shared by
        tp/cp/pp peers.

        `axes` must be real axes listed in spec (outer -> inner) order; that
        makes coordinate(name) == my index in the group's rank order, so
        all_gather order along the flat axis matches the coordinate.

        collective, like _build_groups: every rank must call this with the
        same arguments in the same order (dist.new_group is collective).
        """
        assert axes and len(set(axes)) == len(axes), f"bad axes {axes}"
        assert all(ax in self.strides for ax in axes), f"unknown axis in {axes}"
        # NB: self.strides iterates inner->outer (built reversed); use
        # self.sizes for spec order. virtual names in sizes can't collide
        # here -- the strides-membership assert above already excluded them.
        spec_order = [ax for ax in self.sizes if ax in axes]
        assert list(axes) == spec_order, f"{axes} must follow spec order {spec_order}"
        assert name not in self.sizes, f"axis {name!r} already exists"

        # bucket every global rank by its coords on the NON-flattened axes:
        # strip the flattened axes' contribution to the rank; ranks left with
        # the same base differ only along `axes`, i.e. form one flat group.
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
                # coord = position in group rank order; the spec-order assert
                # makes this equal the row-major mixed-radix coordinate, which
                # is the invariant all_gather along `name` relies on
                self._flat_coords[name] = members.index(rank)

        flat_size = 1
        for ax in axes:
            flat_size *= self.sizes[ax]
        self.sizes[name] = flat_size

    def size(self, dim: str) -> int:
        return self.sizes[dim]

    def get_group(self, dim: str) -> dist.ProcessGroup:
        return self.groups[dim]

    def coordinate(self, dim: str) -> int:
        if dim in self._flat_coords:
            return self._flat_coords[dim]
        return (dist.get_rank() // self.strides[dim]) % self.sizes[dim]


if __name__ == "__main__":
    mesh = Mesh({"pp":2, "dp":2, "tp":2})
    print("temp")
    
