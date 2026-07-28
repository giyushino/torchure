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

## Day ??? Jul 27
review pass: two real bugs found (one silent numerics, one memory), plus the
doc drift that had piled up since the ddp merge.

### the norm weights were frozen the whole time (numerics, silent)

`_init_weights` cast every `nn.RMSNorm` weight to bf16. that was added back in
the compile round to kill the "bf16 input vs fp32 weight -> no fused kernel"
warning, and it did. what it also did, which nobody noticed: it made them bf16
**parameters**, and `_build_optimizer` runs *after* `_init_weights`, so AdamW
was holding bf16 params and doing bf16 updates.

bf16 has an 8-bit mantissa. spacing at 1.0 (where norm gains init) is
2^-8 = 0.0039. a step's update is on the order of lr = 3e-4. so every single
update rounded straight back to the value it started at — the gains sat at
1.0 for the entire run, deterministically, and the exp_avg/exp_avg_sq state
was bf16 garbage on top of that. no crash, no warning, just a slightly worse
loss curve, which is the worst way for a bug to present.

fix: drop the cast, norm weights stay fp32. the dtype mismatch it was papering
over is handled properly by `_compile`, which already covers the blocks *and*
the final norm — compile emits a fused kernel that consumes the fp32 weight
directly, so we get the kernel without paying in parameter precision. the
general rule to hold onto: **bf16 is a compute dtype, not a storage dtype for
anything the optimizer updates.** if a bf16 master weight is ever actually
wanted, it needs an fp32 master copy behind it, not a `.data` cast.

worth re-running the loss-curve comparison in CHANGES.md now that the gains
actually move — the numbers there were all measured with them frozen.

### freeing the fused-CE grads (memory, ~594 MiB + (N,D))

`_FusedLinearCE` computes grad_hidden and grad_weight in the forward and
stashes them; backward's only job is to scale by grad_output / n_valid. it was
doing that **out of place** (`grad_weight * scale`), which allocates a second
full fp32 (V, D) — at qwen3-0.6B's 151936 vocab that's another **594 MiB**,
plus a second (N, D) for grad_hidden. and it allocates it at the worst possible
moment: the peak of backward, when every other grad is also live.

(unit nit while we're here: 151936 x 1024 x 4 = 622 **MB** = 594 **MiB**.
data_parallel.py's docstring and DDP.md both call it "622 MiB" — harmless, it's
the same tensor and the same conclusion, but the tables mix the two units.)

the buffers are forward intermediates built for exactly one backward, and
nothing else aliases them, so scaling in place and handing the same buffer out
is free. two related changes:

- moved the stash from `save_for_backward` to plain ctx attributes. these are
  intermediates, not inputs/outputs of the Function, so there's no
  version-counter contract to violate — which is precisely what makes the
  in-place `mul_` legal here. (doing `mul_` on a `save_for_backward`'d *input*
  would rightly blow up.)
- null out the ctx references before returning, so the buffers die as soon as
  the engine consumes them rather than at graph teardown.
- second backward now raises a real message instead of dying on a NoneType.
  `save_for_backward` gave us "backward through the graph a second time" for
  free; stashing on ctx means saying it ourselves. retain_graph genuinely does
  not work through this op anymore, which is fine (nothing wants it) but should
  fail legibly.

measured, N=512 D=1024 V=151936 chunk=256, `max_memory_allocated` across
backward only:

| backward | peak |
|----------|------|
| out of place (before) | 594.5 MiB |
| in place (after) | 0.0 MiB |

594.5 MiB back, and it's exactly the fp32 (V, D) at 593.5 MiB plus the bf16
(N, D) at 1.0 MiB — i.e. the whole saving is the two copies not being made,
nothing else moved. grads bitwise identical before/after (`rtol=0, atol=0`),
and the cuda correctness cases still match the unfused reference.

generalizable: **any custom autograd Function that precomputes grads in
forward should scale them in place in backward.** the pattern of "stash, then
`return stashed * grad_out`" quietly doubles whatever you stashed, and if what
you stashed was sized by vocab it's the biggest allocation in the step.

related, not done: `ce_chunk_size` is 4096 in the configs while the wrapper's
own default is 1024. per chunk `_chunk_fwd` materializes an fp32 (chunk, V) —
4096 x 151936 x 4 = 2.5 GiB for logits and the same again for probs. dropping
to 1024 should give back ~3.7 GiB of the 23.88 GiB peak, which is per-gpu batch
size we can spend. wants an actual before/after measurement + a tps check
(smaller chunks = more kernel launches), so it's a CHANGES.md entry, not a
drive-by.

### fixes + doc drift

- `configs/qwen3_dense.json` was dead: no `mesh` key (KeyError in the Trainer
  ctor) and no `checkpointing` key, since neither existed when it was written.
  it's the config train.py's docstring pointed at.
- the README's run command (`uv run torchure/train/trainer.py`) had been broken
  since the mesh landed — Trainer can't be built without a process group.
  deleted trainer.py's `__main__` rather than repairing it: there should be one
  entrypoint, and it's train.py, which does the rendezvous first. the bench
  harness it used to call is now `train.py --steps N`, so CHANGES.md's workflow
  has a working command again.
- README claimed `core/` and `parallelism/` were empty; ROADMAP's "current
  state" table claimed collectives were NotImplementedError stubs and that
  fused-linear-CE wasn't in the tree. all stale by a month+. both updated.
- DDP.md still documents an `overlap` config knob with an ablation — that knob
  was removed before the merge (there is no non-overlap mode). left the note as
  the historical work log it is, but flagged it inline so nobody writes the
  config key expecting it to do something.
- still owed, unchanged: prefetcher one-batch skip on resume, checkpoint done
  marker, `tests/data_parallel.py` not in ci (it's gloo/cpu, it should be),
  no grad clipping, no metrics logger.
