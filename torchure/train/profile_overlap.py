"""
comm/compute overlap profiler -- roadmap phase 1 exit criterion (c).

answers "how much of the grad all-reduce is actually hidden behind backward",
which an end-to-end ablation cannot: an ablation tells you overlapping is
faster than not overlapping, but a +11% delta looks the same whether you are
hiding 40% of the traffic or 95%.

capture a trace and print the table:

    uv run torchrun --nproc-per-node 8 -m torchure.train.profile_overlap \
        --config configs/qwen3_dense_climbmix_ddp.json --out traces/overlap_dp8.json

re-analyze one you already have (no gpus, no rendezvous):

    uv run -m torchure.train.profile_overlap --analyze traces/overlap_dp8.json

the chrome trace also loads directly in ui.perfetto.dev. two tracks matter
there: the compute stream and the nccl stream. put them side by side and the
hidden comm (nccl bars under a solid compute row) and the exposed tail (one
long all-reduce at the end of the step with an empty compute row beside it)
are visible without any of the arithmetic below.

method: GPU kernels split into comm (`ncclDevKernel_*`) and everything else,
each set's intervals unioned, the two unions intersected.

    hidden  = comm  n  compute      -- overlapped, effectively free
    exposed = comm --- compute      -- what ddp actually COSTS

exposed is the number to optimize. `traces/` is gitignored -- regenerate
rather than commit, they are ~11 MB each.

warmup matters: the first steps pay torch.compile and allocator churn, so
profiling them measures the wrong thing entirely. default 10 is past both.
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from torchure.train.trainer import Trainer

# cupti categories that represent actual device work
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to the train config json")
    parser.add_argument("--out", help="where to write the chrome trace")
    parser.add_argument("--analyze", help="skip capture, analyze this trace instead")
    parser.add_argument("--warmup", type=int, default=10, help="steps to run before profiling")
    parser.add_argument("--steps", type=int, default=2, help="steps to profile")
    return parser.parse_args()


def union(intervals: list[tuple[float, float]]) -> tuple[list[list[float]], float]:
    """merge overlapping [start, end) intervals; returns (merged, total length).

    unioned rather than summed because kernels from different streams overlap
    in time -- summing durations would double-count concurrent work and could
    exceed the wall clock.
    """
    if not intervals:
        return [], 0.0
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged, sum(end - start for start, end in merged)


def intersect_len(a: list[list[float]], b: list[list[float]]) -> float:
    """total length of the intersection of two already-merged interval lists."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(trace_path: str, n_steps: int) -> None:
    events = json.load(open(trace_path))["traceEvents"]
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in GPU_CATS]
    assert kernels, f"no gpu kernels in {trace_path} -- was ProfilerActivity.CUDA on?"

    comm, compute, per_kernel = [], [], []
    for e in kernels:
        start, dur = e["ts"], e.get("dur", 0.0)
        span = (start, start + dur)
        if "nccl" in e["name"].lower():
            comm.append(span)
            per_kernel.append(span)
        else:
            compute.append(span)

    comm_merged, comm_total = union(comm)
    comp_merged, comp_total = union(compute)
    hidden = intersect_len(comm_merged, comp_merged)
    exposed = comm_total - hidden

    all_merged, _ = union(comm + compute)
    wall = all_merged[-1][1] - all_merged[0][0]
    ms = lambda us: us / 1000.0 / n_steps  # noqa: E731 -- us total -> ms/step

    print(f"\n{trace_path}  ({n_steps} steps, {wall / 1000 / n_steps:.1f} ms/step)\n")
    print(f"{'':<32}{'ms/step':>10}{'share':>12}")
    print(f"{'compute kernels (union)':<32}{ms(comp_total):>10.1f}{100 * comp_total / wall:>11.1f}%")
    print(f"{'nccl all-reduce (union)':<32}{ms(comm_total):>10.1f}{100 * comm_total / wall:>11.1f}%")
    print(f"{'  hidden behind compute':<32}{ms(hidden):>10.1f}{100 * hidden / comm_total:>10.1f}% of comm")
    print(f"{'  EXPOSED (real ddp cost)':<32}{ms(exposed):>10.1f}{100 * exposed / wall:>11.1f}% of step")
    print(f"{'gpu idle (no kernel at all)':<32}{ms(wall - comp_total - exposed):>10.1f}")

    # per-kernel exposure separates the structural tail from the rest. the
    # tied-embedding grad is the LAST one ready (first layer, plus the lm_head
    # contribution), so its reduce has no compute left to hide behind -- that
    # part is unavoidable at any bucket size. exposure in the smaller buckets
    # is either bucketing/launch order or, once total comm exceeds the backward
    # window, plain bandwidth starvation.
    ranked = sorted(
        ((e - s, (e - s) - intersect_len([[s, e]], comp_merged)) for s, e in per_kernel),
        reverse=True,
    )
    print(f"\n{len(per_kernel) // n_steps} nccl kernels/step; largest by duration:")
    print(f"{'':<6}{'dur ms':>10}{'exposed ms':>12}{'exposed':>10}")
    for i, (dur, exp) in enumerate(ranked[:5]):
        print(f"{i:<6}{dur / 1000:>10.1f}{exp / 1000:>12.1f}{100 * exp / dur:>9.0f}%")

    biggest = ranked[0][0]
    tail = sum(exp for dur, exp in ranked if dur >= biggest * 0.9)
    print(f"\n{'exposed in the largest bucket(s)':<38}{ms(tail):>8.1f} ms/step "
          f"({100 * tail / exposed:.0f}%)  <- structural tail")
    print(f"{'exposed everywhere else':<38}{ms(exposed - tail):>8.1f} ms/step "
          f"({100 * (exposed - tail) / exposed:.0f}%)")

    streams = defaultdict(float)
    for e in kernels:
        streams[e.get("tid")] += e.get("dur", 0.0)
    print("\nper-stream device time (the tracks to compare in perfetto):")
    for tid, dur in sorted(streams.items(), key=lambda kv: -kv[1]):
        print(f"  stream {tid}: {ms(dur):.1f} ms/step")


def capture(args: argparse.Namespace) -> None:
    assert args.config and args.out, "--config and --out are required to capture"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # same p2p policy as train.py -- profiling the slow path would be useless
    os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(timeout=timedelta(minutes=10))
    try:
        trainer = Trainer(args.config, rank=rank, local_rank=local_rank, world_size=world_size)
        for _ in range(args.warmup):
            trainer.train_step()
        # fence: no rank starts profiling while another is still warming up,
        # or the trace catches a step that was waiting on a straggler
        torch.cuda.synchronize()
        dist.barrier()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.steps):
                trainer.train_step()
            torch.cuda.synchronize()

        if rank == 0:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            prof.export_chrome_trace(args.out)
            print(f"wrote {args.out}")
            analyze(args.out, args.steps)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.analyze:
        analyze(args.analyze, args.steps)
    else:
        capture(args)


if __name__ == "__main__":
    # the guard is load-bearing, not boilerplate: the dataloader's forkserver
    # re-runs this file to rebuild __main__ in every worker process, and
    # without it each worker re-enters main() and the run dies in a
    # BrokenPipeError with no useful traceback.
    main()
