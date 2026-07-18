# Dev Log

## Day 0
got the scaffolding done, no real implementation

## Day 1
Work on getting single GPU training working

## Day ??? Jun 17
got the dataloader working, trained for 20 steps

## Day ??? Jun 18
Wired in the attn mask
add packing and remove attention mask for now
step=14 || loss=tensor(8.3120, device='cuda:0') || tps=17924.840979258188
--> mfu so low :(
step=36 || loss=tensor(8.0759, device='cuda:0') || tps=33741.09590184431
-> changing to sdpa which uses flash attention under the hood
doubles tps

## Day ??? Jul 16
mesh.flatten() implemented + tested (virtual axis over (dp_replicate, dp_shard)
so the dataloader gets one "which batch shard am i" coordinate; tp/cp peers
share it). all 12 collective tests green on gloo, nccl sweep still owed.

## Day ??? Jul 17
big day: distributed bootstrap + checkpointing/resume landed
- train.py wired for torchrun: env rank/local_rank/world_size,
  init_process_group with timeout, destroy in finally. trainer builds the
  mesh from config now
- decided: single gpu is NOT a special case, it's world_size=1. a process
  group always exists (still need the env setdefaults in train.py so plain
  python launch works, and drop the hardcoded "nccl" -> default backend is
  cpu:gloo,cuda:nccl)
- checkpointing: resume knob is None | "auto" | int. auto = latest in the
  run dir (full-state, atomic, crash recovery only). explicit int = rollback,
  errors if the step doesn't exist. branching/midtraining/SFT deliberately
  deferred to a future init_from (explicit path + per-component flags) --
  resume never carries semantic changes, that's what forks are for
- _resume() wired into __init__ BEFORE iterator/prefetcher creation (else the
  loaded dataloader position is silently ignored)
- save/load trainer state (trainer.pt): step + cpu/cuda rng + config
  snapshot. rng is dead weight today (no dropout, AR objective) but becomes
  load-bearing the moment anything samples per step (masked diffusion, sft).
  checkpointer roundtrip test extended, passes + ruff clean
- convention: checkpoint dir name = number of COMPLETED steps (save at
  step+1, _resume returns state["step"] directly, no +1 anywhere)
- designed but not built: metrics logger (jsonl+stdout always, wandb as
  optional sink), async checkpointing (sync snapshot to pinned cpu, write in
  a background thread, done-marker for atomicity)
- known warts: prefetcher one-batch skip on resume (noted in prefetcher.py),
  no done marker yet so latest() trusts partial dirs, train_n_step_test
  still calls the renamed train_step_test (bench harness broken)
