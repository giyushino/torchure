"""
mesh
"""

import torch.distributed as dist


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

    def size(self, dim: str) -> int:        # extent of that axis
        return self.sizes[dim]

    def get_group(self, dim: str) -> dist.ProcessGroup:  # MY group along that axis
        return self.groups[dim]

    def coordinate(self, dim: str) -> int:  # MY index within that group
        return (dist.get_rank() // self.strides[dim]) % self.sizes[dim]


if __name__ == "__main__":
    mesh = Mesh({"pp":2, "dp":2, "tp":2})
    print("temp")
    
