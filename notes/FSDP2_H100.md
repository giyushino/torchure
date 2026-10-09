# FSDP2 on 4×H100 NVL — throughput, and how big a model actually fits

box: 4× H100 NVL (93.08 GiB usable HBM each, 400 W cap), nccl over
`NCCL_P2P_LEVEL=SYS`. all numbers below are `torch 2.10.0+cu128`,
`seq_len=4096`, ClimbMix, per-block `compile(mode="default")`, bf16 autocast,
fused AdamW, fp32 masters.

two questions: what does fsdp2 cost relative to ddp, and where is the ceiling.
the short version: **fsdp2 costs ~3.5% at a fixed batch shape, but that framing
is misleading — the memory it frees buys a bigger microbatch, and once you spend
it fsdp2 is *faster* than ddp at 4B and is the only option at all past ~5B. the
ceiling is 14B, and it is set by activation memory, not by parameters.**

## method

`train.py --steps N` on a generated config, mean tps over a fixed window past
compile warmup (window starts at step 8 of 20, or 6 of 15 for the slow runs).
per CHANGES.md's benchmark discipline: fixed windows, and the fsdp/ddp A/B at
0.6B was interleaved A/B/A/B rather than run in blocks. the two fsdp repeats
came in at 186,802 and 186,442 tps (0.19% apart) and the two ddp repeats at
194,134 and 193,921 (0.11%), so run-to-run noise is well under every effect
reported here.

MFU uses `6·N + 6·L·S·d_q` FLOPs/token (causal attention, tied head counted once
as the lm_head matmul) against 835 TFLOP/s dense bf16 per gpu.

a second harness, `memprobe.py` (scratch, not checked in), reproduces the
trainer's exact phase order with synthetic batches and reports memory per
phase. it agrees with the real trainer to 0.00 GiB at 8B (60.03 both ways) and
within 2.9 GiB at 12B, the difference being the dataloader/prefetcher.

## fsdp2 vs ddp at a fixed batch shape

| params | micro_bs | | tok/s | peak GiB | MFU |
|---|---|---|---|---|---|
| 0.57B | 4 | ddp | 194,028 | 38.48 | 28.2% |
| 0.57B | 4 | **fsdp2** | 186,622 | **28.75** | 27.1% |
| 1.72B | 4 | ddp | 103,064 | 71.43 | 36.2% |
| 1.72B | 4 | **fsdp2** | 99,556 | **47.47** | 35.0% |
| 4.02B | 1 | ddp | 34,692 | 91.41 | 28.8% |
| 4.02B | 1 | **fsdp2** | 30,161 | **39.94** | 25.1% |

fsdp2 is 3.8% / 3.4% / 13.1% slower. the first two are inside the roadmap's
"within ~10% of ddp" exit criterion; the 4B row is not, and the reason is the
next section.

memory goes the other way: −25%, −34%, −56%. the 4B ddp run sits at 91.41 of
93.08 GiB — 98% of the card. **ddp at 4B/micro_bs=2 OOMs, and ddp at 8B OOMs
before it completes a single step**, both verified rather than extrapolated. so
34,692 tok/s is not just ddp's number at this shape, it is ddp's *best possible*
number at 4B on this box, and 4B is roughly where ddp stops entirely: fp32
master + fp32 grad + two fp32 adam moments is 16 bytes/param undivided, which is
93 GiB at 5.8B params before a single activation.

## the real comparison: fsdp2 lets you spend the memory

every microbatch under this fsdp2 pays a full all-gather of every weight in the
forward and again in the backward, plus a reduce-scatter. ddp pays one
all-reduce per *step*, no matter how many microbatches. so fsdp2's overhead is
comm-per-microbatch against compute-per-microbatch, and it shrinks as micro_bs
grows:

| params | micro_bs | accum | tok/s | peak GiB | MFU |
|---|---|---|---|---|---|
| 4.02B | 1 | 2 | 30,142 | 39.94 | 25.1% |
| 4.02B | 2 | 2 | 38,734 | 57.03 | 32.2% |
| 4.02B | 4 | 1 | **44,465** | 83.08 | **37.0%** |
| 7.57B | 1 | 2 | 17,271 | 60.03 | 25.4% |
| 7.57B | 2 | 1 | **23,003** | 69.71 | **33.8%** |

4B at micro_bs=4 does **44,465 tok/s against ddp's best-possible 34,692** — 28%
*faster*, because ddp at 4B has no memory left to buy a larger microbatch with
(confirmed: ddp 4B at micro_bs=2 OOMs).
that is the honest headline: the fixed-shape A/B above understates fsdp2,
because holding the batch shape fixed is exactly the thing fsdp2 exists to stop
you having to do.

corollary for config-writing: **under fsdp2, prefer micro_batch_size over
grad-accumulation steps far more aggressively than under ddp.** accumulation is
nearly free in ddp and costs a full model gather per step in fsdp2.

## model-size scaling (fsdp2, micro_bs=1, accum=2)

| params | tok/s | peak GiB | MFU |
|---|---|---|---|
| 0.57B | 132,322 | 12.89 | 19.2% |
| 1.72B | 67,052 | 22.22 | 23.6% |
| 4.02B | 30,142 | 39.94 | 25.1% |
| 7.57B | 17,271 | 60.03 | 25.4% |
| 9.88B | 13,321 | 73.03 | 25.6% |
| 11.79B | 11,447 | 83.95 | 26.0% |
| 13.99B | **OOM** (reached 90.2 GiB) | — | — |

with `reduce_dtype: bfloat16` and `ce_chunk_size: 1024`:

| params | tok/s | peak GiB | MFU |
|---|---|---|---|
| 11.79B | 12,813 | 78.97 | 29.1% |
| 13.99B | 11,105 | 90.45 | 29.6% |

**`reduce_dtype: bfloat16` is a win on both axes** — at 11.79B it is 12% faster
*and* saves 5 GiB against fp32 reduction. the fp32 reduce upcasts each unit's
full unsharded grad before the reduce-scatter, which at these sizes is over a
GiB per block, allocated and freed 40 times a backward. the fsdp2 docstring
frames fp32 reduction as the safe default for accumulation error across ranks;
at dp_shard=4 that error is 4 terms, and the cost is measurable.

## where the memory goes

`memprobe.py` phases, all fsdp2, micro_bs=1:

| params | fp32 model on device | peak during sharding | after sharding | training peak |
|---|---|---|---|---|
| 4.02B | 14.98 | 18.78 | 3.75 | 38.49 |
| 7.57B | 28.19 | 35.34 | 7.05 | 60.03 |
| 9.88B | 36.82 | 46.12 | 9.21 | 73.03 |
| 11.79B | 43.92 | 55.03 | 10.98 | 81.05 |
| 13.99B | 52.12 | 65.31 | 13.03 | 90.45 |
| 18.90B | 70.40 | **88.20** | 17.60 | OOM at 91.45 |

(the 13.99B and 18.90B rows use `reduce_dtype: bfloat16` + `ce_chunk_size: 1024`;
fp32 reduction OOMs at both. the first three columns are unaffected by either —
they are fixed by param count alone.)

three constants fall out, and they are exact:

- **steady state is 4 bytes/param/gpu** at dp_shard=4: fp32 master + fp32 grad +
  two fp32 adam moments, each sharded 4 ways = 16/4. this is the column labelled
  "fp32 model on device" too, by coincidence — 4·N bytes either way.
- **sharding transiently costs 5 bytes/param.** `_init_weights` puts the whole
  fp32 model on every rank and `_make_state` clones each shard out while the
  full tensor is still alive, so the peak is 4·N + N. this is v0's documented
  "init full and slice" and it is a *second, independent* wall — see 18.90B
  above, which peaks at 88.20 GiB (95% of the card) before a single activation
  exists.
- **everything else — activations, gather buffers, the CE chunk — is 23–38 GiB**
  and it is what actually binds. at 13.99B it is 38.33 GiB against 52.12 GiB of
  param state.

that last term is dominated by activations, because **there is no activation
checkpointing anywhere in the repo** (the fsdp2 docstring lists the interaction
as untested; nothing calls `torch.utils.checkpoint`). halving `seq_len` to 2048
at 13.99B drops the term from 38.33 to 25.33 GiB, so ~26 GiB of it is
seq-proportional activation storage and ~12 GiB is fixed (gather buffers, the
`ce_chunk_size × vocab` fp32 logits chunk, fragmentation).

## the answer: how big can we go

| | params | why |
|---|---|---|
| comfortable | **~8B** | 69.71 GiB at micro_bs=2, 23,003 tok/s, 33.8% MFU. room for a longer seq or a bigger microbatch. |
| largest reasonable | **~12B** | 78.97 GiB with `reduce_dtype: bfloat16`, 12,813 tok/s. 85% of the card — real but not reckless. |
| hard ceiling | **14B** | 90.45 GiB = 97.2% of HBM, and only with `reduce_dtype: bfloat16` + `ce_chunk_size: 1024` + micro_bs=1. fp32 reduction at this size OOMs. |

14B runs, but I would not start a multi-day job on it: 2.6 GiB of headroom is
one fragmentation event or one config tweak away from dying at hour 30, and
micro_bs=1 pins it at 29.6% MFU. **12B with bf16 reduction is the largest size I
would actually launch**, and 8B is where the throughput/size tradeoff is best.

what would move the ceiling, in order of payoff:

1. **activation checkpointing** (~26 GiB at 14B). the single biggest lever, and
   the only one that touches the binding constraint. not measured here — but
   reading the hook structure, it is not a drop-in: fsdp2 registers its hooks on
   the block module itself, so wrapping that block in `checkpoint()` would have
   the forward-post reshard fire again during recompute and free the weights the
   recomputed backward is about to read. forcing `reshard_after_forward=False`
   on checkpointed units should avoid it, and the existing hook structure
   already supports that, but this needs a loss-equality test before anyone
   trusts it.
2. **meta-device init** (frees the 5 bytes/param transient). without it, nothing
   past ~18.6B can even be *constructed* on this box regardless of how small
   activations get — 18.90B already peaks at 88.20 GiB during sharding.
3. bf16 or fp8 optimizer moments (4 bytes/param → 3 or 2.5).

with 1 and 2 done, the estimate is ~18B on this box, at which point the
init-transient wall and the activation wall land in roughly the same place.

## incidental findings

- **prefetch works, and it is worth 14%.** 7.57B, micro_bs=2:

  | prefetch | s/step | tok/s | peak GiB |
  |---|---|---|---|
  | 0 | 1.6229 | 20,191 | 69.71 |
  | 1 (default) | 1.4227 | 23,032 | 69.71 |
  | 2 | 1.4038 | **23,343** | 69.71 |
  | 3 | 1.4146 | 23,165 | 69.71 |

  the same ablation at 13.99B gives 3.4572 → 2.9592 s/step, +16.8%. this is the
  phase's stated perf exit criterion (gather overlapping compute) and it is met
  *without* the dedicated comm stream the docstring lists as the next perf step
  — `ProcessGroupNCCL` already runs collectives on its own internal stream, so
  `async_op=True` + a deferred `work.wait()` is enough. the remaining headroom
  from owning a stream is therefore smaller than the docstring implies.

  `prefetch=1` captures nearly all of it; `prefetch=2` is a free further 1.3%
  and a reasonable new default; `prefetch=3` is already past the knee.
- **`prefetch=0` does not save a single byte** — 69.71 GiB at every setting
  above, and 90.45 GiB at 14B either way. the peak is set by activations, not
  gather buffers, so lowering prefetch to save memory is pure loss.
- the `#566_632_448` comment at `qwen3.py:355` is stale: it predates the
  ffn_dim 256-rounding, and that config is now 574,029,824 params.
- fsdp2 checkpointing is still gated off (phase 3), so none of these sizes can
  actually be checkpointed yet. that, not throughput, is what blocks a real run
  at 8B+.
