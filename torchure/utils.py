import json
import os
import time


def debug_time(func):
    """print how long `func` took, on rank 0 only."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        if os.environ.get("RANK", "0") == "0":
            print(f"{func.__name__} took {time.perf_counter() - t0} seconds to run")
        return result
    return wrapper


def get_project_dir() -> str:
    """the repo root (the directory holding the `torchure` package)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
