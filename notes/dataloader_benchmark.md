# loader throughput vs torch DataLoader, grain, zephon (11/22)

loader-only, cpu only, local files. the torchure loader is the branch
`claude/dataloader-optimizing` at `60fe526` (DATA_LOADING.md's "per-core
cost" section describes what it changed). this is the repeat of the
130-core comparison that section says was still to do, on a different box.

short version:

- with `workers`, torchure is the fastest of the four on text (20.5M tok/s
  at 32 workers; torch DataLoader 18.0M, grain 16.2M) and level with torch
  and grain on images (11-12k img/s at 32).
- per core it is at or above the others: 650-790k tok/s at n=8 against
  600-630k for torch and grain, ~390 img/s against ~385 with the same decode.
- fetch threads (the default, `workers: 0`) do not scale to 32: text costs
  2.3x the cpu per token, and images are slower at 32 threads than at 8.
- the default window shape leaves a quarter of the threads / workers idle.
- zephon stayed at ~2M tok/s and ~2k img/s per pipeline whatever its
  parallelism was set to.

## method

**box.** 2 × epyc 74F3: 48 cores, 96 hardware threads. shared: load average
19-35 from other users' jobs for the whole session, so "a core" below is a
hardware thread that often had a busy sibling. gpus hidden
(`CUDA_VISIBLE_DEVICES=""`).

**versions.** python 3.14.6, torch 2.10.0+cu130, tokenizers 0.23.1, numpy
2.4.6, pyarrow 24.0.0, pillow 13.0.0 (the repo's lock), plus grain 0.2.19,
zephon 0.2.1, transformers 5.23.0 (zephon's tokenize op loads the tokenizer
through it).

**data**, on local disk and in the page cache, the same files for every loader:

- text: the first 4GB of ClimbMix `part_0.jsonl` (1,486,000 docs, ~680
  tokens each), written twice: 64 jsonl shards of 64MB, and 64 parquet
  files with a `text` column in row groups of 4,000 docs.
- images: the first 16 `train-*` parquet files of `jxie/coco_captions`
  (49,824 rows, row groups of 100; 640×480 jpegs).

**the job.** every loader delivers the same batches as torch tensors:

- text: tokenize (Qwen3-0.6B, no special tokens), `<|endoftext|>` after
  each doc, concatenate, cut into rows of 4096; batches of 4:
  `input_ids`, `labels` int64 (4, 4096). the token ids of transformers'
  `AutoTokenizer` (zephon) and of `tokenizers` (the rest) were checked equal.
- images: decode, resize to 224px bicubic, uint8 chw; the caption in the
  vlm row layout (`<|vision_start|>` + 49 `<|image_pad|>` + `<|vision_end|>`
  + caption + eos, padded to 128, labels ignored on the image span and the
  padding); batches of 64: `input_ids`, `labels`, `pixel_values`.

**the loaders.**

| loader | pipeline | `n` is |
|---|---|---|
| torchure | `MixtureLoader`, one source, `workers: 0` | `fetch_threads` |
| torchure `workers` | same, `workers: n` | worker processes |
| torch | `DataLoader` over an `IterableDataset`: a worker takes every n-th row group, in shuffled order | `num_workers` |
| grain | `MapDataset.range(row groups).shuffle().repeat().map(read + tokenize)` → `ConcatThenSplitIterDataset` → `batch` → `mp_prefetch`; images: a `FlatMapIterDataset` of row group → rows | `mp_prefetch` workers |
| zephon | `Pipeline(StaticMixtureWorkSource).decode_text().tokenize().pack_flat(4096, algorithm="wrap").batch(4)`; images: `map_transform` | `max_workers` and the tokenize / map op's parallelism |

torch and grain run the same user code for the work itself
(`encode_batch_fast` in chunks, the same caption layout). zephon uses its
own tokenize and packing operators. everyone reads parquet; torchure was
also run on the jsonl shards, its production path for ClimbMix.

**images are decoded two ways** in torch, grain and zephon: `pil`, the
textbook path (full decode, `convert("RGB")`, PIL bicubic resize), and
`fast`, the branch's own (`draft()`, zero-copy pixels, torch's antialiased
bicubic: `sources._caption_rows`). torchure always uses its own. `fast` is
the loader comparison; `pil` shows what the decode change is worth.

**settings.**

- `TOKENIZERS_PARALLELISM=false` for everyone: one core per thread / worker.
  (torchure sets it itself, and turns the pool on for a cold stream's first
  window.)
- torchure: `window_blocks = n/2`, `prefetch_windows = 2`, so at most n
  blocks are in flight: the shape of the config defaults (8 threads, 4 × 2).
  rows marked `pw4` use `prefetch_windows: 4`. `block_mb` 16.
- torch: `prefetch_factor` 64 for text, 4 for images; text workers tokenize
  32 docs per call. at `prefetch_factor` 4 and a row group per call, 8
  workers used one core between them (0.7M tok/s, on a 200MB sample): the
  DataLoader takes batches from its workers in turn, and a worker that
  stops for seconds to tokenize stalls the rest.
- grain: `per_worker_buffer_size` 64 for text, for the same reason (rows
  marked `buf4` are the first runs, at 4), 4 for images.
- the fd soft limit is raised from 1024: torch and grain pass each
  prefetched batch as a shared-memory fd and both crash at 32 workers
  without it.

**measurement.** one loader per process, one process at a time. time to the
first batch; then `next()` in a tight loop for a warmup (15 s text, 10 s
images) and for the timed window (30 s text, 25 s images). reported: tokens
or images delivered per second; cores = cpu time of the process and all its
descendants over the window / wall; per core = delivered / cpu time. three
runs of every configuration, interleaved (all loaders, then again), median
and range below.

## text

tokens per second, parquet unless marked.

| n | loader | tok/s, median (range) | cores | tok/s per core | first batch |
|---|---|---|---|---|---|
| 8 | torchure `workers`, pw4 | 6.26M (5.85-6.31M) | 8.0 | 783k | 2.7 s |
| 8 | torchure | 5.45M (5.28-5.46M) | 6.8 | 786k | 2.5 s |
| 8 | torchure, pw4 | 5.38M (5.31-5.52M) | 7.9 | 682k | 2.6 s |
| 8 | torchure `workers` | 5.25M (4.99-5.53M) | 6.8 | 757k | 2.8 s |
| 8 | torch | 5.16M (5.11-5.43M) | 8.2 | 629k | 1.7 s |
| 8 | grain | 5.09M (4.72-5.17M) | 8.4 | 598k | 4.4 s |
| 8 | grain, buf4 | 4.76M (4.57-4.83M) | 7.8 | 602k | 4.5 s |
| 8 | grain, threads instead of `mp_prefetch` | 4.63M (4.56-4.72M) | 6.5 | 704k | 5.4 s |
| 8 | torchure, jsonl | 4.54M (4.49-4.60M) | 7.0 | 650k | 3.5 s |
| 8 | zephon, tokenize in processes | 2.15M (2.05-2.24M) | 6.2 | 347k | 5.2 s |
| 8 | zephon | 2.05M (1.97-2.08M) | 6.1 | 330k | 5.0 s |
| 32 | torchure `workers`, pw4 | **20.45M** (18.60-20.54M) | 31.4 | 643k | 8.0 s |
| 32 | torch | 18.02M (16.07-18.39M) | 32.5 | 555k | 2.0 s |
| 32 | grain | 16.16M (16.14-16.57M) | 31.7 | 513k | 5.5 s |
| 32 | torchure `workers` | 13.41M (12.78-13.51M) | 22.0 | 610k | 7.7 s |
| 32 | grain, buf4 | 12.39M (11.29-13.06M) | 21.8 | 569k | 5.6 s |
| 32 | torchure, pw4 | 10.03M (9.59-10.95M) | 28.1 | 357k | 8.3 s |
| 32 | torchure, jsonl | 9.30M (8.12-9.39M) | 20.1 | 461k | 9.9 s |
| 32 | torchure | 7.53M (6.42-9.66M) | 22.7 | 332k | 8.3 s |
| 32 | zephon | 1.72M (1.71-1.76M) | 6.2 | 275k | 20.7 s |

## images

images per second. `pil` was run at n=8 only.

| n | loader | decode | img/s, median (range) | cores | img/s per core | first batch |
|---|---|---|---|---|---|---|
| 8 | torchure `workers` | own | 3,188 (3,151-3,217) | 8.1 | 396 | 1.2 s |
| 8 | grain | fast | 3,121 (3,105-3,184) | 8.1 | 388 | 2.1 s |
| 8 | torch | fast | 3,102 (3,092-3,139) | 8.1 | 382 | 1.8 s |
| 8 | torchure | own | 2,558 (2,552-2,580) | 6.6 | 391 | 1.1 s |
| 8 | zephon | fast | 1,922 (1,917-2,067) | 5.7 | 335 | 1.5 s |
| 8 | torch | pil | 1,729 (1,720-1,745) | 8.1 | 213 | 1.9 s |
| 8 | grain | pil | 1,713 (1,711-1,739) | 8.1 | 213 | 2.5 s |
| 8 | zephon | pil | 1,434 (1,361-1,490) | 6.9 | 207 | 1.5 s |
| 32 | grain | fast | 11,933 (11,197-11,937) | 32.3 | 367 | 2.7 s |
| 32 | torch | fast | 11,619 (11,449-11,752) | 32.1 | 361 | 2.1 s |
| 32 | torchure `workers` | own | 11,084 (10,309-11,087) | 31.1 | 355 | 2.2 s |
| 32 | torchure | own | 2,013 (1,807-2,039) | 7.3 | 271 | 2.2 s |
| 32 | zephon | fast | 1,085 (975-1,398) | 5.3 | 207 | 4.9 s |

## what it shows

- **per-core cost.** at n=8 torchure tokenizes at 757-786k tok/s per core
  (parquet) against torch's 629k and grain's 598k; images are ~390 img/s per
  core for torchure, torch and grain alike once they share a decode. the
  11/21 gap (16-22k against 210-230k tok/s per core) is gone.
- **the image gain is the decode, not the loader.** `pil` → `fast` is 213 →
  ~385 img/s per core (1.8x) in torch and grain: the same function gives
  them the same speed.
- **threads stop scaling before 32.** text: 786k → 332k tok/s per core from
  8 to 32 threads, 7.5M tok/s on 22.7 cores, where 32 `workers` deliver
  13.4M (20.5M with pw4). images: 2,013 img/s at 32 threads, less than the
  2,558 at 8, on 7 cores. DATA_LOADING.md guessed threads would stop
  scaling "somewhere past 16"; they do. past 8 threads, use `workers`.
- **the default window shape idles threads / workers.** at most
  `window_blocks × prefetch_windows` blocks are in flight and a window waits
  for its slowest block, so ~6.8 of 8 and ~22 of 32 were busy.
  `prefetch_windows: 4` fills them: with workers, 5.25M → 6.26M tok/s at 8
  and 13.4M → 20.5M at 32, at the same per-core cost. on threads it adds
  cores but little throughput (682k vs 786k per core at 8).
- **jsonl costs ~17% more cpu per token than parquet** at n=8 (650k vs 786k
  tok/s per core): the same docs, read as json lines instead of an arrow
  column.
- **torch DataLoader and grain are close behind when tuned**, and reach
  their first batch sooner at 32 (2.0 s and 5.5 s against 8 s; torchure's
  8 s is this benchmark's `window_blocks: 16`, a first window of 16 row
  groups, not a property of the loader).
- **zephon did not scale here.** ~2M tok/s on 6-7 cores at n=8 and no more
  at 32, at about half the per-core rate of the others. tried at n=32,
  none above 1.85M tok/s: fetch / decode parallelism 8 and 16, the tokenize
  op in processes, `deterministic=False`, deeper queues
  (`prefetch_batches`, `op_queue_capacity` 512), `worker_allocation:
  "global"`. not tried: a pipeline per DataLoader worker
  (`to_torch_dataset`), which is how it would be spread over more cores.
  this may be this benchmark's configuration rather than zephon's limit.

## what it does not show

- **no training run.** loader-only, as DATA_LOADING.md warns: it measures
  the loader's cores, not what fetch threads do to a busy training thread.
- **throughput only.** the loaders do not do equal work around the job:
  torchure shuffles rows within windows and its state resumes in O(1);
  the torch and grain pipelines here shuffle row groups only and nothing
  was resumed; zephon keeps its deterministic bookkeeping.
- **tuned baselines.** torch and grain are shown with the buffer depths
  above. before that, torch text at n=8 ran at 0.7M tok/s on one core.
- **a shared box.** absolute numbers moved with the other users' load;
  compare within a table. per-core rates here are ~3x the laptop's in
  DATA_LOADING.md (220-240k tok/s per core).
- **n = 8 and 32 only**, one rank, local files: no hub reads, no dp > 1.
- **`fast` pixels differ from `pil`** by a few levels (DATA_LOADING.md).

## files

not in the repo: `/home/allanz/torchure-dlbench/` holds the worktree of the
branch with its venv, the data, and

- `bench/compare_loaders.py`: one loader, one run
  (`--task text|image --loader ... --n 8 --data DIR`; `--check` prints a
  batch summary instead)
- `bench/prep_text.py`: a jsonl file → the jsonl and parquet shards
- `bench/sweep_text.sh`, `sweep_text2.sh`, `sweep_rest.sh`,
  `sweep_image.sh`: the runs above
- `bench/summarize.py results/text.jsonl`: the tables, from
  `results/{text,image}.jsonl` (one json line per run)
- `bench/clean_shm.sh`: removes the semaphores and shared memory that
  killed torch / grain runs leave in `/dev/shm`
