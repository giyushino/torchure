# H100 DDP — first real-length run

goal: not a benchmark this time — an actual multi-hour pretraining run, on
the 4×H100 NVL box, to see the checkpointing/resume/wandb machinery survive
something longer than a 20-40 step smoke test and to have a checkpoint worth
actually looking at.

```
uv run torchrun --nproc-per-node 4 -m torchure.train.train \
    --config configs/qwen3_dense_climbmix_ddp.json
```

## sizing the run

target: ~8 hours wall clock. `configs/qwen3_dense_climbmix_ddp.json` came in
set up for an 8×A40 box (`mesh.dp: 8`, `global_batch_size: 16`), and
`optimizer.scheduler.total_steps: 100` — a benchmark leftover. worth knowing:
`total_steps` isn't just the LR schedule length, `trainer.py` reads it
straight into `num_train_steps`, which is the training loop's stop condition.
so the number that "sizes the run" lives in the scheduler block, not
somewhere that looks like a step-count knob.

benchmarked the actual target shape first (4×H100 NVL, `mesh.dp: 4`,
`global_batch_size: 32`, `micro_batch_size: 4`, `seq_len: 4096`) with
`--steps 40` rather than trusting the old A40/A6000 numbers in CHANGES.md —
different box, and CHANGES.md's own thermal-drift note says not to
cross-compare across boxes anyway. steady state (steps 20-39, past compile
warmup): **~196,100 tok/s** aggregate, peak mem 38.5/95.8 GiB per gpu. at
131,072 tok/step that's ~0.668 s/step.

8h target / 0.668 s/step ≈ 43,100 steps unfaded; derating ~3% for the
power-cap thermal drift CHANGES.md documents for sustained runs (short
benchmark runs never leave boost clock, long ones do) lands around 41,800.
picked **42,000** — also chosen to divide evenly into `save_steps`.

## config changes from the checked-in file

| key | before | after | why |
|---|---|---|---|
| `mesh.dp` | 8 | 4 | box has 4 gpus, not 8 |
| `data.global_batch_size` | 16 | 32 | keep effective batch/gpu sane at dp=4 |
| `data.micro_batch_size` | 2 | 4 | divides 32/(4×4)=2 microbatches/step |
| `optimizer.scheduler.total_steps` | 100 | 42000 | sizes the run, see above |
| `optimizer.scheduler.warmup_ratio` | 0.001 | 0.01 | 0.001×42000 ≈ 42 steps was too short for a from-scratch run; 420 is more typical |
| `checkpointing.save_steps` | 10000 | 6000 | 10000 > total_steps meant **zero checkpoints would have saved at all**; 6000 divides 42000 evenly (7 checkpoints, last one lands exactly at the end) |
| `run_name` | `climbmix_test_run_ddp` | `climbmix_long_test_run_ddp_h100` | distinguish from the benchmark-run checkpoints already sitting in `checkpoints/` |
| `wandb` | (missing) | `{"enabled": true, "project": "torchure", "entity": null}` | see below |

the `save_steps` one was the closest call to a real footgun: at the
as-checked-in settings (`total_steps: 100`, `save_steps: 10000`) a full run
would finish having saved nothing, silently, because `save_steps` only fires
on `(step+1) % save_steps == 0` — there's no unconditional save at the end of
`train()`.

## wandb

landed for real in `8e322a7` ("wired in wandb") + `2c77e72` ("wandb stuff") —
`trainer.py` now does `wandb.init` on rank 0 (skipped for `--steps` benchmark
mode) and logs the same metrics dict that already goes to stdout, plus stores
`wandb_id` in `trainer.pt` so resume reattaches to the same run instead of
forking a new one. package lives in `.venv`'s own site-packages (see gotcha
below), install via `uv sync --extra wandb` or `pip install wandb` into that
venv specifically.

**gotcha hit while checking this**: `.venv/bin/python` is a symlink to
`miniconda3/envs/fresh/bin/python3`, but it is a real venv (`pyvenv.cfg`,
`include-system-site-packages = false`) with its own `site-packages`
layered on top via `sys.path`. importing `wandb` through the raw conda
`fresh` interpreter path fails; importing it through `.venv/bin/python`
(same binary, different path) succeeds. `uv` itself wasn't on `$PATH` in
this session either, so `.venv/bin/python -m ...` directly, not `uv run`,
is what actually launched the run.

## the run

`wandb/run-20260730_224316-seyczvik`, project `torchure`.

- started 2026-07-30 22:43:16, finished 2026-07-31 06:54:46 — **~8h11m**,
  within ~10% of the 8h target derived above.
- exit code 0, clean shutdown, no OOM/crash/dmesg noise.
- loss: ~12.1 at step 0 → ~2.5-2.6 by step 42000 (WSD stable phase mostly
  sat in the 7.5-8 range early on before the decay phase pulled it down
  further in the last 10%).
- LR decayed to exactly 0.00e+00 at the final step per the WSD schedule.
- tps drifted from ~197k (early) down to ~187-189k (steps ~41990+) — the
  power-cap thermal decay CHANGES.md predicted, on a different box.
- all 7 checkpoints present and complete:
  `checkpoints/climbmix_long_test_run_ddp_h100/{6000,12000,...,42000}/`,
  each with `.complete` marker, `model.pt` (2.14 GiB fp32), `optimizer.pt`
  (4.28 GiB fp32 adamw state), `scheduler.pt`, `trainer.pt`, and
  `dataloader_dp{0,1,2,3}.pt`. ~6.5 GiB/checkpoint × 7 ≈ 45 GiB total,
  against 779 GiB free on `/home` at the time.

## post-run: does it know anything

no inference/eval code existed before this — added
`torchure/inference/{common,eval_ppl,chat}.py`, all standalone (no process
group, no Mesh, load straight off a `checkpoints/<run_name>/<step>/model.pt`
state dict).

**perplexity** (`eval_ppl.py`, 20 fresh batches, seed bumped off the
training config's so the streaming shuffle doesn't just replay the trained
stream — caveat: ClimbMix has no held-out split, so this is in-distribution,
not a real held-out number): **avg loss 2.64, perplexity ≈ 14.0** over
327,680 tokens. matches where the loss curve ended.

**chat repl** (`chat.py`, no chat template — base model, plain
continuation, no KV cache): fluent grammar, unreliable facts, exactly what
you'd expect from ~5.5B tokens into a 574M-param model. e.g. "The capital of
France is called St. Pierre..." — sentence-shaped, wrong. 100 tokens takes
~2.9s without a cache; decided not worth adding KV caching for now (GQA
changes the masking/repeat logic, needs correctness testing against the
no-cache path) unless this grows into an actual serving/eval focus rather
than "does the checkpoint look alive."

## chinchilla, for reference

574,029,824 params, 42000 × 32 × 4096 = 5.505B tokens trained ⇒ **9.6
tokens/param**, about 48% of the ~20:1 chinchilla-optimal ratio. reaching
20:1 at this param count needs ~11.48B tokens total, i.e. ~6B more (~another
~88,000 steps total, ~8.7 more hours at this box's measured throughput).
not a defect — chinchilla-optimal minimizes *training* compute, says
nothing about inference cost, and plenty of shipped small models are
trained well past 20:1 on purpose. just a data point for this run, not a
target it was supposed to hit.
