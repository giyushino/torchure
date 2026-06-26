from __future__ import annotations

import os
import time

# this should change to
# just record everything
# slightly in a debug log 
def record_time(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        return result, t1 - t0
    return wrapper

def debug_time(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"{func.__name__} took {t1 - t0} seconds to run")
        return result
    return wrapper

def get_project_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return project_root

class Node:
    def __init__(self, low: int, high: int, value: int = 0):
        self.low = low
        self.high = high
        self.value = value
        self.left = None
        self.right = None

    @classmethod
    def build(cls, low: int, high: int) -> Node:
        node = cls(low, high)
        if low == high:
            return node
        mid = (low + high) // 2
        node.left = cls.build(low, mid)
        node.right = cls.build(mid + 1, high)
        return node

    def search(self, size: int):
        if self.value < size:
            return None
        if self.low == self.high:
            return self.low
        if self.left.value >= size:
            return self.left.search(size)
        return self.right.search(size)

    def update(self, capacity: int, available: bool):
        if self.low == self.high:
            self.value = capacity if available else 0
            return
        if capacity <= self.left.high:
            self.left.update(capacity, available)
        else:
            self.right.update(capacity, available)
        self.value = max(self.left.value, self.right.value)
        

if __name__ == "__main__":
    print(get_project_dir())

