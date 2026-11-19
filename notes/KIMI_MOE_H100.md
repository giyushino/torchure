# kimi k3.1 4b + ddp regression on 4x h100 (2026-11-18)

all `train.py --steps N` on 4x H100 NVL, seq 4096, tps = steady-state log
window (aggregate over 4 gpus).

## ddp regression check (strict bucket order)

`qwen3_dense_climbmix_ddp_h100.json`, 40 steps: **196.7-197.1k tok/s**, peak
36.9 GiB. the H100_DDP.md number was ~196.1k / 38.5 GiB, so launching buckets
in order (for missing-grad safety) costs nothing measurable.

## kimi: where the time went

`kimi_k31_4b_climbmix.json` (28 layers, 21 kda, 56 experts top-4), global
batch 8, micro 2, activation checkpointing on, block compile off.

before: 4.6k tok/s. a 4-layer profile on one gpu was launch-bound (42k kernel
launches / step, cpu 1.3 s vs gpu 0.83 s), almost all of it the kda chunk
loop's small elementwise ops.

kda alone (B=2, S=4096, H=16, D=128, fwd+bwd, one gpu):

| chunk | eager ms | compiled ms | peak GiB (compiled) |
|---|---|---|---|
| 32  | 362 | 254 | 1.25 |
| 64  | 195 | 119 | 1.26 |
| 128 | 264 |  84 | 1.78 |

so `_kda_chunk` is now torch.compile'd on its own (the scan stays
compiler-disabled) and KDA_CHUNK = 128. matches `_kda_reference` to ~1e-6.

## kimi on 4 gpus after the kda change

| mesh | tok/s | peak GiB |
|---|---|---|
| fsdp dp_shard=4, before kda change | 4,615 | 20.5 |
| fsdp dp_shard=4 | **10,189** | 20.5 |
| fsdp dp_shard=4, block compile on | 10,606 | 20.5 |
| fsdp dp_shard=2 x ep=2 | 9,842 | 26.3 |

ep doesn't pay at this size: 56 small experts (moe_dim 512) inside one node.
dense params are sharded only 2-way under ep, hence the extra memory.

## 300-step run: balancing, sigterm, resume

fsdp dp_shard=4, save every 100, keep_last_n 2. SIGTERM to ONE worker at
step 150: all ranks agreed at the next check, checkpointed step 160, exited
0. relaunch with resume "auto" continued from 161, loss continuous
(5.70 -> 5.72), finished at 300. pruning left {160, 200, 300}: keep_last_n=2
complete ones besides the one being written, as designed.

load balance (27 moe layers x 56 experts = 1512):

| step | dead experts | max_load (worst layer) |
|---|---|---|
| 10  | 267 | 11 |
| 40  | 658 | 14 |
| 100 | 569 | 14 |
| 200 | 451 | 14 |
| 300 | 303 | 14 |

max_load 14 = 56/4: in the worst layer one expert is picked by every token.
the routers collapse in the first ~30 steps (lr 3e-4 after a 3-step warmup)
and router_bias_rate 1e-3 per step pulls them back at ~1-2 experts per
step. balancing works, but too slowly for short runs; worth trying a larger
rate (1e-2) or a longer warmup.

## balancing: router_bias_rate x warmup (300 steps, fsdp dp_shard=2, batch 8)

both with warmup_ratio 0.1 (30 steps; the run above had 3). metrics are the
worst layer, averaged over the 10-step log window.

| step | rate 1e-3: dead / max_load | rate 1e-2: dead / max_load |
|---|---|---|
| 20  | 4.6 / 8.6  | 0.2 / 5.3 |
| 60  | 0 / 9.8    | 0 / 2.9   |
| 150 | 3.5 / 10.0 | 0 / 5.0   |
| 300 | 0 / 10.6   | 0 / 2.5   |

the warmup is what stops the collapse (hundreds of dead experts -> a few).
the higher rate is what evens the load out: max_load stays ~10 at 1e-3 and
comes down to ~2.5 at 1e-2, min_load 0.005 -> 0.31. loss is the same
either way at this length (5.230 vs 5.234 at step 300).

## micro batch sweep (fsdp dp_shard=4, one microbatch per step, 15 steps)

| micro batch | act. ckpt | tok/s | peak GiB |
|---|---|---|---|
| 2  | on  | 10,189 | 20.5 |
| 4  | on  | 12,436 | 23.0 |
| 8  | on  | 13,286 | 31.2 |
| 16 | on  | 13,605 | 48.0 |
| 4  | off | OOM | >93 |
| 8  | off | OOM | >93 |

past micro 8 the gain is small (+2.4% for 16). without block checkpointing
even micro 4 runs out in forward: each kda layer keeps fp32 copies of
q/k/v/log_alpha plus a (B, H, D, D) fp32 state per chunk for backward, times
21 kda layers.

## the new kimi configs at micro batch 8 (100 steps, global batch 32)

router_bias_rate 1e-2, micro batch 8, 4 gpus.

| run | warmup | tok/s | peak GiB | dead experts (peak / step 100) | max_load @100 | loss @100 |
|---|---|---|---|---|---|---|
| fsdp_ep (dp_shard 2 x ep 2) | 10 steps | 13,016 | 36.9 | 129 / 49 | 13.8 | 5.545 |
| fsdp dp_shard 4 | 10 steps | 13,320 | 31.2 | 242 / 100 | 14 | 5.561 |
| fsdp dp_shard 4 | 30 steps | 13,330 | 31.2 | 0.1 / 0 | 2.5 | **5.468** |

warmup has to be long in STEPS: 10% of a 100-step schedule (10 steps) lets
the routers collapse again; 30 steps doesn't, and the loss is lower too.
the kimi config now uses warmup_ratio 0.3 (30 of its 100 steps).
(the w30 run's last two log windows dropped to ~6.5k tok/s when another
user's job started on gpu 0; 13.3k is the uncontended number.)

## kda, parallel over chunks

profile at micro batch 8, one gpu, fwd+bwd per component: kda attention 269
ms (x21 layers), mla 57 ms (x7), latent moe 18.5 ms (x27), dense ffn 28.5 ms
(x1). kda was ~70% of the step, and memory-bound: each chunk built a (C, C,
D) fp32 pairwise-decay tensor, ~1 GiB per chunk at micro 8.

_kda_scan now does all the state-independent work for every chunk at once
(_kda_intra): within a chunk, the decay between two 16-step sub-blocks
factors through the later one's first step, so those parts of the
triangular system are plain matmuls; only the 16x16 diagonal sub-blocks
use the pairwise form. what's left sequential is the state, two small
matmuls per chunk. matches _kda_reference to ~1e-6 (tests/models.py).

scan alone, B=8, S=4096, H=16, D=128, fwd+bwd:

| version | ms | peak GiB |
|---|---|---|
| per-chunk loop, chunk 128, compiled | 333 | 6.7 |
| parallel, eager | 213 | 25.8 |
| parallel, _kda_intra compiled (now) | **128** | 15.8 |

full model, fsdp dp_shard=4, micro 8: **13.3k -> 19.3k tok/s**, peak 31.2 ->
38.0 GiB, same loss curve. (the per-layer peak is transient: blocks are
checkpointed, so one layer holds it at a time.)
