# data mixing (11/19)

a second data path beside the hf stream: pre-tokenized token shards, read
by byte range, mixed per optimizer step by a python schedule. code:
`torchure/dataloader/{shards,mixture,tokenize_shards}.py`, tests:
`tests/mixture_loader.py` (ci), example: `mixes/smoke.py` +
`configs/qwen3_dense_mix_smoke.json`.

## why not the hf stream

- **resume.** a streamed jsonl shard has no seek, so resuming means
  re-downloading and re-parsing everything already consumed in the current
  shard (DATA_LOADING.md). `resume_dataloader: false` dodges it by giving up
  the stream position.
- **mixing.** `interleave_datasets(probabilities=...)` is static (no
  schedule), and every source would bring its own resume replay.
- **shuffle.** a 1000-doc shuffle buffer over a stream that may be sorted
  by topic (climbmix is clustered).

the fix isn't "do the http ourselves", it's a format with random access.
once rows are addressable by byte offset, resume, shuffling and mixing all
become arithmetic.

## format

a source is a directory or url prefix:

```
index.json   {"format": "torchure-tokens", "version": 1, "dtype": "uint32",
              "tokenizer": "Qwen/Qwen3-0.6B", "eos_id": 151643,
              "files": [{"path": "000000.bin", "num_tokens": ...}, ...]}
000000.bin   raw little-endian tokens, each doc followed by eos
```

uint16 when the vocab fits, else uint32 (qwen3's 151936 doesn't).
`index.json` is written last, so a half-written source is unreadable. the
tokenizer is recorded and the loader refuses a source whose tokenizer
differs from `data.tokenizer`.

at load time each file is cut into `seq_len` rows (the tail < seq_len is
dropped, ≤ one row per file), and rows into **blocks** of ~`block_mb`
(default 16MB = 1024 rows at seq_len 4096, uint32). a block is one range
request. a source with fewer blocks than dp ranks gets smaller blocks.

locations: a local path (pread), `http(s)://` (any server that honors
`Range`: r2, s3, a cdn), or `hf://datasets/<org>/<name>[/<subdir>]` (→ the
hub's resolve url; `HF_TOKEN` is sent if set and stripped on the redirect to
the cdn). reads retry 408/425/429/5xx and connection errors with capped
exponential backoff + jitter; a server that ignores Range is an error, not a
silent whole-file download. measured 11/19: a 16MB range read through the
hub's redirect takes ~0.62s (~25MB/s per connection); training needs
~1MB/step per run at 64×4096.

making a source:

```
uv run -m torchure.dataloader.tokenize_shards --dataset OptimalScale/ClimbMix \
    --tokenizer Qwen/Qwen3-0.6B --eos-token "<|endoftext|>" \
    --max-tokens 2e9 --out data/climbmix_qwen3
hf upload <org>/climbmix-qwen3 data/climbmix_qwen3 --repo-type dataset   # optional
```

tokenization uses the same `encode_batch` call as the hf path's packer.

## the mix script

config: `"data": {"name": null, "mix": "mixes/smoke.py"}` (exactly one of
`name` / `mix`). the script:

```python
SOURCES = {"climbmix": "hf://datasets/<org>/climbmix-qwen3", "fineweb_edu": "data/fwedu"}

def weights(step: int, total_steps: int) -> dict[str, float]:
    edu = 0.1 if step < 0.9 * total_steps else 0.5
    return {"climbmix": 1 - edu, "fineweb_edu": edu}
```

weights are relative; a source left out gets 0; unknown names, negatives
and all-zero are errors. **`weights` must be a pure function of its
arguments**: every rank calls it independently and a resume calls it
again. the normalized mix is logged to wandb as `mix/<name>`.

the script's text is saved in the loader state. changing it mid-run is
allowed (resume prints a note and the new schedule applies from the resumed
step), as is adding a source (it starts fresh) or removing one.

## per step

1. **counts.** `weights(step) × global_batch_size` is added to a per-source
   credit; the step's counts are the credit rounded (largest remainder,
   clamped ≥ 0) so they sum to exactly `global_batch_size`. the remainder
   carries, so a source at 0.1% still gets ~1 row every 16 steps at
   batch 64 instead of rounding to 0 forever. tested: cumulative counts stay
   within one row of the cumulative target per source.
2. **placement.** the step's G source labels are shuffled with
   `rng(seed, step)`; rank r's micro batch m reads positions
   `[(m·dp + r)·b, +b)`. every rank computes the whole plan (G ints), so
   there's no communication.
3. **rows.** each (rank, source) is a `SourceStream`: per epoch, the
   source's blocks are permuted with `rng(seed, crc32(name), epoch)` and
   dealt round-robin to ranks (disjoint: each rank downloads 1/dp of the
   data; tested that one epoch covers every row exactly once across ranks).
   rows are shuffled within windows of `window_blocks` (default 4) blocks.
   the next window downloads on `fetch_threads` threads while the current
   one is consumed. a source that runs out starts its next epoch on its own;
   the loader never ends.

seeds are per-name (`crc32`), so adding a source doesn't reshuffle the
others.

## state and resume

```
micro_index, credit carried into the current step,
per source: (epoch, window, pos), mix_text, shape (dp, G, b, seq_len)
```

resume rebuilds the current step's plan from the credit and refetches one
window per source: O(window), not O(position). tested: resuming mid-step,
on a step boundary and after an epoch wrap gives identical batches to an
uninterrupted run. the trainer's held-batch mechanism (prefetcher) works as
before. a different dp size or batch shape is refused: per-rank streams
don't remap (same constraint as the hf path; `resume_dataloader: false`).

## design choices

- **no worker processes.** there's no tokenization left, and process
  workers would each download the same windows. the loader runs in the
  trainer process; downloads are threads (the gil is released during
  socket reads). assembling a micro batch is a numpy gather: 0.3ms for
  16×4096 on real data.
- **why not a StatefulDataLoader + sampler.** the original plan was a
  map-style dataset with the mixing in a batch sampler. process workers
  break block locality (each would fetch every window), and with
  num_workers=0 the StatefulDataLoader adds nothing over `state_dict()` on
  the loader itself.
- **memory.** per rank: 2 × window_blocks × block_mb per source (current +
  prefetched window) = 128MB per source at the defaults.
- **shuffle quality.** rows are mixed within 4 random blocks (4096 rows,
  ~16M tokens at 4096) from anywhere in the source, and block order is a
  full per-epoch permutation. better than the hf path's 1000-doc buffer.
  docs inside a row are in file order.

## what this doesn't do yet

- **metric-driven schedules** (e.g. upweight the domain with the highest
  loss). `weights` only sees `(step, total_steps)`. adding live inputs
  means logging the realized weights per step and replaying them on resume,
  since the function would no longer be pure.
- **per-source loss.** the batch doesn't carry each row's source. adding a
  `source` tensor to the batch and a per-source reduction in the objective
  is the obvious next step for anyone tuning a mix.
- rows cut through docs, like the hf path's stream packer. `doc_mask`
  (TORCHTITAN_COMPARE.md) masks across the eos tokens already in each row.
- **multi-node dedupe.** blocks are rank-disjoint, so nothing is downloaded
  twice; a shared node-local cache would only help multi-epoch runs.

## gpu check (11/19)

4×h100 nvl, ddp, qwen3 0.6B, compile off, global batch 32×4096,
`mixes/smoke.py` over 60 steps (switch to 50/50 at step 54), checkpoint at
30. run B resumed from 30:

| step | run A loss | run B (resumed) |
|---|---|---|
| 35 | 7.2821 | 7.2821 |
| 45 | 7.2258 | 7.2258 |
| 55 | 7.1594 | 7.1594 |
| 60 | 7.3564 | 7.3563 |

~134k tok/s steady in both. wandb `mix/climbmix` / `mix/fineweb_edu` read
0.9/0.1 through step 50 and 0.5/0.5 from 55. the loss bump at 60 is the
distribution shift into fineweb-edu.

## throughput vs the hf path (11/19)

the gpu check above ran with compile off (~134k tok/s); that, not the
loader, is why it looked slow. same shape with compile on
(`qwen3_dense_climbmix_ddp_h100.json`, `--steps 60`, mean of steps 21-60),
run in both orders:

| run | hf stream | mixture |
|---|---|---|
| first pair (hf first) | 195,883 | 194,420 |
| second pair (mix first) | 195,028 | 194,549 |

the mixture is ~0.5% lower both times, about the same as the hf path's own
run-to-run spread (0.45%). the loader itself costs 0.06ms per micro batch
including pinning (0.12ms of a 670ms step, ~0.02%), so it can't account for
the gap; treat it as noise unless a longer run says otherwise. matches the
~196k recorded in H100_DDP.md.
