# data loading (11/19)

one loader, no torch `DataLoader`, no hf `datasets`, nothing written to
disk. raw jsonl / parquet is read by byte range from local disk or over
http (including `hf://`), tokenized (or decoded, for images) on threads in
the trainer process, or in `data.workers` processes (image sources want
those), and mixed per step by `mixture.py`. one dataset is `data.source`; several are `data.mix`
(DATA_MIXING.md).

code: `torchure/dataloader/{sources,shards,mixture,builder}.py`.
tests: `tests/data_sources.py`, `tests/mixture_loader.py` (both ci).

```json
"data": {"source": "hf://datasets/OptimalScale/ClimbMix", ...}
"data": {"source": {"path": "hf://datasets/jxie/coco_captions", "files": "data/train-*",
                    "text_field": "caption", "image_field": "image", "window_blocks": 16}, ...}
"data": {"mix": "mixes/example_mix.py", ...}
```

## why the hf stream went

the old default path (`data.name`) was `datasets.load_dataset(streaming=True)`
→ shuffle buffer → `split_dataset_by_node` → `.map` packing → a
`StatefulDataLoader` with spawn workers. its problems:

- **resume replays the shard.** a streamed jsonl shard has no seek; hf's
  iterator resumes by re-downloading and re-parsing every record already
  consumed in the current shard (`ExamplesIterable.__iter__`'s `islice`).
  multi-GB, minutes.
- **`resume_dataloader: false` replayed the run's own data.** the loader
  was rebuilt with the same seed at epoch 0, so a "reshuffle" was the same
  order from the start. (now: a resume without loader state reseeds with
  `seed + step`; see the open problems for what that still doesn't solve.)
- **shard counts.** `split_dataset_by_node` hands ranks whole shards only
  when dp divides the shard count. coco's 182 shards over 4 ranks: every
  rank downloaded and decoded the *whole* stream and kept 1/4.
- **shuffle** was a 1000-doc buffer over a stream that may be sorted by
  topic.
- startup: 30s (climbmix) to 110-180s (coco) to the first batch.

the fix was random access, not a different stream: once a block of data is
addressable by a byte range, resume, rank splitting, shuffling and mixing
are arithmetic. jsonl turns out to be addressable without an index.

## source kinds (sources.py)

picked from the files at the location (a directory, one file, or
`hf://datasets/<org>/<name>[/<subdir>]`, listed through the hub's tree api):

| kind | block | resume refetches |
|---|---|---|
| `*.jsonl` | a ~`block_mb` byte range; owns the lines that **start** in it | one 16MB range |
| `*.jsonl.zst` / `.gz` | a whole file (compressed streams don't seek) | one file (dolma: ~85MB) |
| `*.parquet` | one row group, one range read (footers read once at startup) | one row group |
| `index.json` + `*.bin` | pre-tokenized rows (`tokenize_shards.py`, optional) | one range |

**jsonl without an index.** a block `[start, end)` reads from `start - 1`:
if that byte isn't a newline, the block starts mid-line and skips to after
the first newline (that line belongs to the previous block). then it reads
past `end` until its last line's newline. every line has exactly one block
(the one its first byte is in), whatever the block size; a line longer than
a block leaves the blocks inside it empty. tested for block sizes from 4KB
up, a file without a trailing newline, and lines spanning several blocks.

**text rows.** a block's docs → `encode_batch` (the same call
`tokenize_shards.py` uses) → each followed by eos → concatenated → cut into
`seq_len` rows; the block's tail is dropped (≤ 1 row per block: ~0.1% at
16MB jsonl blocks, ~0.2% at fineweb-edu's ~1M-token row groups). rows start
on a doc boundary at the start of each block.

**caption rows** (`image_field` set; parquet): one image + one caption per
row, unpacked: `<|vision_start|> <|image_pad|>×n <|vision_end|> caption eos
pad…`, labels ignore_index on the image span and padding (identical layout
to the old hf path's `ImageTextCollator`). images decode on the fetch threads
(or the `workers` processes) with PIL (it releases the gil: 1 → 16 threads scales 221 → 2,390 img/s);
jpegs use `draft()` (dct-domain 2/4/8x downscale before the bicubic resize,
+25%; pixels differ slightly from a full decode). the resize is torch's
(11/21, below). `pixel_values` stay
**uint8** through the loader and the host→device copy; the vision tower
normalizes (siglip and clip use different constants, so it's the model's
call): `SiglipVisionTower.forward` casts and applies `/127.5 - 1` when it
gets uint8.

**http.** range reads keep a connection per host per thread and cache each
file's redirect target. the hub redirects every `resolve/` url to a
presigned cdn url; a fresh connection plus that redirect was most of a read:

| 3.5MB range read from the hub | time |
|---|---|
| resolve url, new connection (urllib) | 0.38s |
| cdn url, new connection | 0.29s |
| cdn url, kept-alive connection | 0.08s |

a cached target that fails with 401/403/404 (expired signature) is dropped
and resolved again (tested against a server whose signatures expire).
`HF_TOKEN` is only sent to the host it was configured for.

## the loader (mixture.py)

per (rank, source), a `SourceStream`: blocks permuted per epoch, dealt
round-robin to ranks (disjoint), rows shuffled within windows of
`window_blocks` blocks, and `prefetch_windows` windows (default 2) fetched
ahead on `fetch_threads` threads. a window's blocks are not concatenated
(at coco's 16 blocks that's 240MB of pixels): rows are gathered by index
straight into one pinned tensor per key. an empty window (raw blocks
shorter than `seq_len`) is skipped. state stays `(epoch, window, pos)` per
source, since `fetch(block)` is deterministic: resume refetches one window.

## loader-only throughput (11/19)

the 4×h100 nvl box (384 cores, shared with another user's gpu job that
day), 4 loader processes at once (one per dp
rank, dp=4), each pulling batches in a tight loop for 60 s, then once every
0.27 s (text) / 0.3 s (vlm), the configs' 4-gpu step time, to time the
training thread's `next()`. data from the hub, cold. script:
`tests/bench_dataloader.py` (the old-loader runs used the same loop against
the pre-11/19 tree).

**text**, climbmix, `qwen3_dense_climbmix_ddp_h100.json` (micro 4 × 4096):

| loader | tok/s, all 4 ranks | first batch | `next()` cpu ms (median) |
|---|---|---|---|
| old: hf stream, 4 workers/rank | 1,620,507 | 31.5 s | 0.7 |
| new: jsonl byte ranges, 8 threads/rank | **9,577,825** | 6.4 s | 0.5 |

training needs ~240k tok/s at this shape (4 gpus): 40x headroom.

**vlm**, coco, `qwen3_vl_coco_ddp_h100.json` (micro 128, seq 128, 224px):

| loader | images/s, all 4 ranks | first batch | `next()` cpu ms (median) |
|---|---|---|---|
| old: hf stream + `ImageTextCollator`, 8 workers/rank | 670 | 111-183 s | 0.6 |
| new, first cut (float pixels, no keep-alive) | 1,279 | 4-16 s | 24.5 |
| new, + keep-alive/redirect cache, uint8 pixels, pinned gather | 3,095 | 4-9 s | 3.9 |
| new, + no window concat (threads) | 2,944 | 6-7 s | 4.7 |
| new, `workers: 8` processes per rank (final config) | **3,013** | 14 s | 4-15 |

the 4-gpu vlm run trains at ~1,690 images/s (EXPERIMENTAL_ARCHS.md), so the
new loader has ~1.75x headroom; the old one measured 670/s here but kept up
in that run, so the hub was likely slower today, which hits both alike. the
final two rows are the same code bar the concat change, i.e. noise of
~±5%. per rank, decode tops out around 16 threads (32 is slower: gil).

**what the training thread pays.** old: unpickling a batch the workers
built (~0.7 ms). new: gathering rows + pinning (0.5 ms text, ~4-5 ms vlm,
which copies 19MB of uint8 pixels). the new loader's p90 `next()` wall time
had spikes of 10-170 ms in the paced runs, while windows were still being
tokenized: python-side work on the fetch threads (json parsing, building
the token arrays) competes for the gil. see the end-to-end numbers for
whether training notices.

## end-to-end (11/19)

the box was shared that afternoon (another user's job on gpus 0-1), so
these are **2-gpu ddp on gpus 2-3**: the 4-gpu configs with `mesh.dp: 2`
(same per-gpu micro batch, so each rank's loader serves the same rate as in
the 4-gpu run), compile on, `--steps 60`, mean tok/s of steps 21-60, runs
interleaved old/new.

| run | old loader | new loader |
|---|---|---|
| text, `qwen3_dense_climbmix_ddp_h100` | 124,972 / 124,546 | 124,230 / 124,660 |
| vlm, `qwen3_vl_coco_ddp_h100`, new loader on threads | 124,102 / 124,336 | **105,974 / 87,178** |
| vlm, new loader with `workers: 8` | 124,100 | 124,566 / 124,469 |

- text: no difference (±0.3%, the run-to-run spread).
- **vlm on threads lost 15-30%**, with a 32k tok/s window in one run.
  loader-only it had 1.75x headroom, so this isn't the loader falling
  behind: decoding 500 images/s per rank on 16 threads in the trainer
  process holds the gil enough (PIL's python-side work, pyarrow
  conversion) to slow the training loop that launches the kernels. that's
  what `workers` is for: the same fetch code in N processes, rows back by
  pickle (~7 ms to unpickle a 100-image block, off the training thread).
  with it, the vlm matches the old loader (+0.3%, noise).
- the first 10 steps: the new loader's vlm runs averaged 66-89k tok/s
  against the old one's 16k, because the old loader takes minutes to
  produce the first batch.
- the loader-only benchmark didn't predict the threads regression: it
  measures the loader's thread, not what the loader does to a busy one.
  judge loader changes with an end-to-end run.

## per-core cost (11/21)

a comparison against torch's `StatefulDataLoader`, grain and zephon on a
130-core box put this loader's text at 16-22k tok/s per core against their
210-230k (it reached 2-3M tok/s by using ~130 cores), and everyone's images
at 240-270 img/s per core. what was wrong, and what changed
(branch `claude/dataloader-optimizing`):

- **text: tokenizers' thread pool.** every fetch thread called
  `encode_batch`, which fans one block out over every core (rayon), 8 of
  them at once. now each fetch thread (or worker) tokenizes its block on
  one core with the gil released, `TOKENIZERS_PARALLELISM=false`, which the
  loader sets for the process. the pool is turned back on only while a
  cold stream (startup, resume) waits on its first window
  (`sources.fetch_block`), so the first batch is no slower.
- **text: `encode_batch` → `encode_batch_fast`** (no offsets, which nothing
  read). with 8 threads tokenizing at once, offsets more than doubled the
  cpu per token (7.5 vs 3.4 cpu-s for the same docs with a tokenizer copy
  per thread; `_fast` is 2.6 either way). in chunks of 256 docs, so a
  16MB block's Encodings never all exist at once.
- **text: malloc arenas.** glibc gives each new thread the next arena
  round-robin once there are about as many arenas as cores. the fetch pool
  started its threads lazily, some before and some after the ~ncores
  threads rayon spawns for the first window, so on a small box fetch
  threads ended up sharing arenas and fought over the arena lock in every
  allocation for the rest of the run: 25,000 context switches/s per
  thread, 0.5-1.3M tok/s instead of 1.8M, in about one run in three. the
  fetch threads are now all started back to back in `__init__`.
- **images: the resize was half the cost.** per 640×480 coco jpeg, one
  core: decode with `draft` 2.7 ms (a full decode is 3.0: huffman decoding
  dominates, so there is little left to take), PIL bicubic resize
  320×240 → 224×224 2.1 ms, `np.asarray` 0.2 ms. torch's antialiased
  bicubic on uint8 is the same kernel vectorized: 0.6 ms. its output is
  within a few levels of PIL's on ~5% of values (max 5, mean 0.05 on coco),
  so **pixels are not bit-identical to before**. the decoded pixels reach
  torch without a copy (PIL lends its RGBX memory through arrow), which
  also removes most of the gil-held python per image.
- **workers are plain subprocesses** (`workers.py`), not a spawn
  `ProcessPoolExecutor`. a spawn child re-imports the trainer's `__main__`
  (torch and everything else) before it can decode anything; these import
  the source code, start while the parent is still listing files, and a
  cold stream's first window is fetched on the trainer's threads without
  waiting for them. rows come back as raw bytes on a pipe, read straight
  into the array with the gil released, instead of pickle. a worker exits
  when its pipe closes, so nothing is left to join at interpreter exit
  (the old pool could hang there after `close()` had killed its workers).
- **exit.** `close()` is also hooked ahead of the interpreter's thread
  join, and tells running fetches to stop at their next chunk. before, a
  trainer that hadn't closed the loader first worked off the queued
  prefetch at exit: 26 s for the text config on the laptop below.

loader-only, one rank, 8 fetch threads or `workers: 8`, local files, on a
14-core laptop (i9-13900H on battery, so absolute numbers are low and
"per core" mixes P and E cores; before/after runs interleaved, 3 each,
range shown). cores = cpu time of the process tree / wall.
`tests/bench_dataloader.py` now reports it.

| text, climbmix jsonl, 4 × 4096 | tok/s | cores | tok/s per core | first batch |
|---|---|---|---|---|
| threads, before | 1.13-1.22M | 12.1 | 93-104k | 2.0-2.2 s |
| threads, after | 1.71-1.81M | 7.7 | 221-233k | 1.6-1.9 s |
| `workers: 8`, before | 1.16-1.22M | 6.9 | 170-178k | 12.1-12.4 s |
| `workers: 8`, after | 1.37-1.44M | 7.8 | 175-183k | 1.9-2.1 s |

(`block_mb: 2` for these: at 16MB a window is 16M tokens and a 20 s run
only sees a few arrive. at the default 16MB, one 40 s run each: threads
1.15M → 1.84M tok/s, 85k → 238k per core, first batch 13.4 → 11.1 s;
workers first batch 37.5 → 12.7 s.)

| coco captions, 64 × 224px | img/s | cores | img/s per core | first batch |
|---|---|---|---|---|
| threads, before | 561-587 | 6.6 | 85-89 | 3.3-3.5 s |
| threads, after | 749-756 | 6.2 | 121-122 | 2.6 s |
| `workers: 8`, before | 668-674 | 8.1 | 82-83 | 17.9 s |
| `workers: 8`, after | 911-985 | 8.0 | 113-122 | 4.6 s |

resume (state saved after 20 batches, restored in a fresh process, time to
the first batch) is the cold start again: 37.0 → 12.2 s for text with
workers, 17.6 → 5.3 s for images with workers, unchanged on threads.

what this does not show:

- **no gpu run.** the end-to-end section's lesson stands: these are
  loader-only numbers, and the vlm-on-threads regression there came from
  the gil, which a loader-only run doesn't see. the 130-core comparison
  and a training run are still to be repeated on this branch.
- **text throughput per rank is now `fetch_threads` × ~220k tok/s**, not
  "every core": about 1.8M tok/s at the default 8. on the 130-core box
  that is less than the 2-3M the old loader reached by using all of it.
  a 4-gpu run needs ~60k tok/s per rank; raise `fetch_threads` (the python
  share of a block is ~5%, so expect threads to stop scaling somewhere past
  16; not measured) or use `workers` if a run ever needs more.
- images on threads still use only ~6 of 8 cores (presumably the gil:
  PIL's python-side header parsing, pyarrow conversion); that is what
  `workers` is for.

## open problems

- **`workers` is opt-in.** image sources want it (the vlm config sets 8);
  a vlm config without it trains 15-30% slower. it could default on for
  caption sources (`workers: null` already means 8 for image sources and 0
  for text in `MixtureLoader`; the config default is still 0).
- **the 4-gpu end-to-end check is still to do** (the box was shared).
- **resume without loader state** (`resume_dataloader: false`, e.g. after a
  mesh change) reseeds with `seed + step`: a new order, not the remainder
  of the old one, so it can repeat data the run saw. a mesh-independent
  position (a global row counter over a rank-independent block order) would
  let any dp size resume exactly.
- **hub listing.** a whole-repo `hf://` location lists the repo
  recursively at startup, per rank. dolma3's 6T mix is ~1 request per 1000
  files; point at a subdirectory or use `files` for big repos. parquet
  footers are read per file at startup (32 threads; coco's 182 files take
  ~6 s), every rank. a shared metadata cache would fix both.
- **compressed jsonl** is a whole file per block, so resume refetches up to
  one file (~85MB for dolma) and a window holds `window_blocks` files of
  tokens (whole-file sources default to 1 per window).
- **no per-source eval split**; `eval_ppl.py` still reshuffles the
  training data (seed + 1).
- `packers.py` (stream / best-fit packers) is no longer on the training
  path; it stays for `tests/dataloader_packing.py`'s comparisons.
