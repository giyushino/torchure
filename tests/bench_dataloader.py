"""
loader-only throughput for one dp rank (no gpu, no model). not run in ci: it
reads real data. to load a box like a dp=N run, start N copies at once with
--rank 0..N-1 (notes/DATA_LOADING.md has the 4-rank numbers).

  max:    next() in a tight loop for --seconds: the rows/s the loader can deliver,
          and the cores it took (cpu time of this process and its workers / wall)
  paced:  next() every --pace seconds (a stand-in for the gpu step): the
          training thread's cpu and wall time per next()

    uv run tests/bench_dataloader.py --config configs/qwen3_dense_climbmix_ddp_h100.json \\
        --rank 0 --dp 4 --pace 0.27
    # a vlm config's caption source needs the model's image spec:
    uv run tests/bench_dataloader.py --config <vlm config> --vision 224 49 --pace 0.3
"""

import argparse
import json
import os
import statistics
import time

from tokenizers import Tokenizer

from torchure.config import load_config
from torchure.dataloader.builder import build_dataloader


def tree_cpu(pid: int | None = None) -> float:
    """cpu seconds so far of a process and its live descendants (linux /proc)."""
    pid = pid or os.getpid()
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        total = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
        for task in os.listdir(f"/proc/{pid}/task"):
            with open(f"/proc/{pid}/task/{task}/children") as f:
                total += sum(tree_cpu(int(child)) for child in f.read().split())
    except (FileNotFoundError, ProcessLookupError):
        return 0.0
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--warmup", type=int, default=5, help="batches before the timed loop")
    p.add_argument("--warmup-seconds", type=float, default=15,
                   help="and seconds: the first windows are fetched in a burst (every core)")
    p.add_argument("--pace", type=float, default=0.0)
    p.add_argument("--paced-batches", type=int, default=40)
    p.add_argument("--vision", type=int, nargs=2, default=None,
                   metavar=("IMAGE_SIZE", "NUM_IMAGE_TOKENS"))
    p.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                   help="override a data.* key, e.g. --set fetch_threads=16")
    p.add_argument("--out", default=None, help="also write the result json here")
    args = p.parse_args()

    data = dict(load_config(args.config)["data"])
    for kv in args.set:
        key, value = kv.split("=", 1)
        data[key] = json.loads(value)
    tokenizer = Tokenizer.from_pretrained(data["tokenizer"])

    t0 = time.perf_counter()
    loader = build_dataloader(data, tokenizer, -100, args.rank, args.dp, 10**6,
                              tuple(args.vision) if args.vision else None)
    batch = next(loader)
    first = time.perf_counter() - t0
    for _ in range(args.warmup):
        next(loader)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.warmup_seconds:
        next(loader)

    n = 0
    cpu, t0 = tree_cpu(), time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        next(loader)
        n += 1
    dt, cpu = time.perf_counter() - t0, tree_cpu() - cpu
    rows = batch["input_ids"].shape[0]
    result = {"rank": args.rank, "dp": args.dp, "first_batch_s": round(first, 2),
              "rows_per_s": round(n * rows / dt, 1),
              "tokens_per_s": round(n * batch["input_ids"].numel() / dt),
              "cores": round(cpu / dt, 2), "rows_per_core_s": round(n * rows / cpu, 1)}

    if args.pace > 0:
        cpu, wall = [], []
        for _ in range(args.paced_batches):
            time.sleep(args.pace)
            w, c = time.perf_counter(), time.thread_time()
            next(loader)
            cpu.append(time.thread_time() - c)
            wall.append(time.perf_counter() - w)
        wall.sort()
        result |= {"next_cpu_ms_median": round(1e3 * statistics.median(cpu), 2),
                   "next_wall_ms_median": round(1e3 * statistics.median(wall), 2),
                   "next_wall_ms_p90": round(1e3 * wall[int(0.9 * len(wall))], 2)}
    loader.close()
    print(json.dumps(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
