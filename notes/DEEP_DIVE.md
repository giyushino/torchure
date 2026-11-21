# torchure deep dive

A walk through the whole codebase: what each piece does, why it is built that
way, and how the pieces meet. Written against `main` at `b4d2a62`, plus the
VLM, which lives on `origin/claude/experimental-arch` (section 12).

Reading order follows a training step from the outside in. Each section names
the files it covers, so you can read the code alongside.

Contents

1. [The 30-second picture](#1-the-30-second-picture)
2. [Entry point and config](#2-entry-point-and-config)
3. [The trainer](#3-the-trainer)
4. [Core: mesh, collectives, placements](#4-core-mesh-collectives-placements)
5. [The dataloader](#5-the-dataloader)
6. [Models: shared layers, Qwen3, OLMo 3](#6-models-shared-layers-qwen3-olmo-3)
7. [Kimi K3.1: KDA, MLA, latent MoE, attention residuals](#7-kimi-k31-kda-mla-latent-moe-attention-residuals)
8. [The loss: fused linear + cross-entropy](#8-the-loss-fused-linear--cross-entropy)
9. [Parallelism: DDP, FSDP2, TP, EP](#9-parallelism-ddp-fsdp2-tp-ep)
10. [Optimizer, schedule, grad clipping](#10-optimizer-schedule-grad-clipping)
11. [Checkpointing and resume](#11-checkpointing-and-resume)
12. [The VLM (branch `claude/experimental-arch`)](#12-the-vlm-branch-claudeexperimental-arch)
13. [Inference and profiling](#13-inference-and-profiling)
14. [Tests and CI](#14-tests-and-ci)
15. [Loose ends found while reading](#15-loose-ends-found-while-reading)

---

## 1. The 30-second picture

```
train.py ── init_process_group ── Trainer(config)
                                     │
   load_config + validate            │  config.py
   build_mesh                        │  core/mesh.py
   build model (maybe on `meta`)     │  models/*
   TP / EP rewrite shapes            │  parallelism/tensor_parallel.py, expert_parallel.py
   DDP or FSDP attach hooks          │  parallelism/data_parallel.py, fsdp2.py
   init weights (per-shard seeds)    │
   torch.compile each block          │
   objective, optimizer, scheduler   │  objectives/, loss/, optimizer/
   MixtureLoader + CUDAPrefetcher    │  dataloader/*
   Checkpointer (+ resume)           │  checkpoint/checkpointer.py
                                     ▼
 train_step:  for each microbatch:  batch → autocast(bf16) → model → fused CE → backward
              parallel.sync() → tp.sync() → ep.sync() → clip → optimizer → scheduler
```

Three ideas run through everything:

- **One code path.** Single GPU is `world_size == 1`, not a special case. A
  process group and a mesh always exist; collectives over a size-1 axis are
  skipped.
- **Hooks, not wrappers.** DDP, FSDP, TP and EP never wrap the model or change
  its `forward`. They register module hooks and grad hooks, and (for sharding)
  replace a parameter's storage. The model files stay plain PyTorch.
- **The model declares, the engine acts.** A model exposes a small contract
  (`blocks`, `norm`, `lm_head.weight`, `init_weights()`, `tp_plan()`,
  optionally `ep_plan()` and `balance_step()`). The parallelism code reads no
  other attribute names.

Runtime dependencies are `torch==2.10`, `numpy`, `tokenizers`, `pyarrow`
(`pillow` and `wandb` are optional extras). There is no `datasets`, no torch
`DataLoader`, no Accelerate.

---

## 2. Entry point and config

### `torchure/train/train.py`

- Sets `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `LOCAL_RANK`, `WORLD_SIZE` with
  `setdefault`, so a plain `python -m torchure.train.train` behaves like a
  one-process `torchrun`.
- Sets `NCCL_P2P_LEVEL=SYS` by default. On some PCIe boxes NCCL otherwise
  falls back to shared memory across the host bridge (8.4x all-reduce
  bandwidth difference measured on an 8x A40 node).
- Calls `torch.cuda.set_device(local_rank)` **before**
  `dist.init_process_group`, otherwise every rank's NCCL context lands on
  GPU 0.
- Builds a `Trainer`, then runs `trainer.train()` or, with `--steps N`,
  `trainer.train_n_step_test(N)` (benchmark mode: no checkpointing, no
  wandb, prints throughput and peak memory).

### `torchure/config.py`

A run is one JSON file. `load_config(path)` does four things:

1. **Inheritance** (`_read`, `_merge`). `"base": "other.json"` is resolved
   relative to the file, recursively, with cycle detection. Dicts merge key
   by key; everything else replaces. `mesh` is the exception (`_WHOLE`): a
   variant's mesh replaces the base's whole, because merging two meshes would
   keep axes the variant meant to drop.
2. **Schema check** (`_check` against `SCHEMA`). Each schema leaf is a
   default value, `REQUIRED`, or `FREE` (a dict passed through as kwargs for
   something else). An unknown key is an error that names the closest known
   key via `difflib`.
3. **Kwarg check** (`_check_kwargs`). The `FREE` blocks (`model.config`,
   `objective.config`, `optimizer.scheduler`) are checked against the
   `__init__` signature of the class they will be passed to, so a typo in a
   model hyperparameter fails at load time, not after the mesh is built.
4. **Value check** (`_check_values`): enums, `source` xor `mix`,
   `global_batch_size % micro_batch_size == 0`, positive mesh sizes.

`validate(config, world_size)` is separate because it needs the launched rank
count. It checks everything that would otherwise show up as a hang or a shape
error at step 0:

- the mesh's product equals `world_size`;
- `pp` and `cp` are size 1 (not implemented);
- `parallelism: "fsdp"` needs `dp_shard > 1`;
- `tp > 1` and `ep > 1` together are rejected;
- `global_batch_size` divides by `micro_batch_size x data ranks`;
- sequence parallel needs `seq_len % tp == 0`; `sp_dim: "batch"` needs
  `micro_batch_size % tp == 0`;
- `num_experts % ep == 0`.

The result is a plain dict. It is saved inside every checkpoint and sent to
wandb.

Config keys worth knowing:

| key | meaning |
|---|---|
| `mesh` | axis sizes: `dp_replicate`, `dp_shard` (or `dp` as shorthand for `dp_shard`), `tp`, `ep` |
| `parallelism` | `"ddp"` or `"fsdp"`: how the dp axes hold the model |
| `ddp.bucket_mb`, `ddp.reduce_dtype` | bucket cap; dtype grads are all-reduced in |
| `fsdp.param_dtype`, `fsdp.reduce_dtype`, `fsdp.prefetch` | gathered-weight dtype, grad wire dtype, units gathered ahead |
| `tp.vocab_parallel`, `tp.sequence_parallel`, `tp.sp_dim`, `tp.overlap` | see section 9.3 |
| `compile`, `compile_mode` | per-block `torch.compile` |
| `activation_checkpointing` | recompute block activations in backward |
| `doc_mask` | attend within documents only |
| `data.*` | section 5 |
| `checkpointing.*` | section 11 |

---

## 3. The trainer

`torchure/train/trainer.py`. Module import time sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and enables TF32 for the
fp32 matmuls that bf16 autocast leaves alone.

### 3.1 Construction order

The order in `Trainer.__init__` matters; several steps depend on the one
before.

1. `load_config`, `validate`, `torch.manual_seed(seed)`.
2. `build_mesh(config["mesh"])`. `dp_size = mesh.size("dp")`; `tp_size` and
   `ep_size` default to 1 because the mesh drops size-1 axes.
3. Tokenizer (`Tokenizer.from_pretrained`).
4. `_build_model()` (below).
5. `model.activation_checkpointing`, `model.doc_mask_eos` set from config.
6. Reseed: `torch.manual_seed(seed + 1 + dp_coordinate)`. Every data rank
   gets its own random stream after init; TP peers share a dp coordinate and
   so share a stream.
7. `_compile(model)`: `block.forward = torch.compile(block.forward)` for each
   block, and `model.norm.compile()`. It compiles `forward`, not the module:
   `module.compile()` would trace FSDP's hooks, which must stay eager.
   Identical blocks share one graph, and the graph boundaries match FSDP's
   units.
8. Objective (and `objective.tp = self.tp`, because the fused loss bypasses
   the head module and so bypasses TP's head hooks).
9. Optimizer, scheduler.
10. Dataloader. Asserts the sources' eos id equals `data.eos_token`'s id.
11. `microbatches_per_step = global_batch_size / (micro_batch_size x dp_size)`.
12. `Checkpointer`, then `_resume()` (returns the step to start at).
13. wandb (rank 0 only, after `_resume` so it can reattach to the saved run
    id).
14. `CUDAPrefetcher`, **after** resume, because it pulls a batch immediately
    and the loader's position must already be restored.

### 3.2 Building the model: two paths

`_build_model` picks between:

**Plain DDP** (no FSDP, TP or EP): build on the GPU, `init_weights()`,
then `DDP(...)`, which broadcasts rank 0's weights to everyone.

**Anything sharded** (FSDP, TP or EP): no rank should ever hold the full
weights, so:

```
with torch.device("meta"): model = build_model(...)   # shapes only, no storage
tp = TP(mesh, **model.tp_plan(...))                   # rewrites shapes on meta
ep = EP(mesh, model.ep_plan())                        # rewrites shapes on meta
if fsdp:  FSDP(...)          # allocates only this rank's shards
          _init_shards()
else:     materialize(model) # core/meta.py: real storage at the (tp/ep-cut) shapes
          _init_shards()
          DDP(...)           # replicates what tp/ep left
```

TP and EP cut first; FSDP then shards whatever is left on the rank along
`dp_shard`.

`_init_shards` seeds each rank by its **shard cell**,

```
cell = ((dp_shard coord) * tp_size + tp coord) * ep_size + ep coord
torch.manual_seed(seed + cell)
```

so different shards of one weight get different random values, then runs
`model.init_weights()` on the shards, then calls `finish_init()` on FSDP, TP
and EP in that order. Each `finish_init` broadcasts whatever is supposed to
be *replicated* along its axis (HSDP replicas, params TP did not cut, dense
params under EP, buffers) from coordinate 0, so replicas agree regardless of
what each rank's RNG produced.

This is why every model's `init_weights` must be **elementwise**
(`normal_`, `zeros_`, `ones_`, a scalar `mul_`): it runs on a shard and
cannot depend on the full weight's shape.

`core/meta.py: materialize` replaces each meta `nn.Parameter` once, wherever
it is registered, so a tied weight stays tied, and returns
`{id(new param): original shape}`.

### 3.3 One optimizer step

```python
def train_step(self):
    self.parallel.requires_sync = False
    for _ in range(microbatches_per_step - 1):
        loss += self.micro_train_step()
    self.parallel.requires_sync = True      # only the last microbatch syncs (ddp)
    loss += self.micro_train_step()
    self.parallel.sync()                    # wait for grad collectives
    if self.tp: self.tp.sync()              # all_reduce grads of shared params
    if self.ep: self.ep.sync()              # divide expert grads by ep
    grad_norm = clip_grad_norm(params, mesh, max_norm, shard_dims_of=...)
    self.optimizer.step(); self.optimizer.zero_grad(); self.scheduler.step()
    return {"loss": loss, "grad_norm": grad_norm, **self._balance_experts()}
```

`micro_train_step` pulls a batch from the prefetcher, runs
`objective.compute_loss` under `torch.autocast(bfloat16)`, divides by
`microbatches_per_step`, and calls `backward()`.

Ordering constraints:

- `tp.sync()` comes after `parallel.sync()`: DDP may still be all-reducing
  those same grad tensors in the background until its `sync()` returns.
- Clipping comes after all three syncs, because only then is every `.grad`
  the final (sharded or replicated) gradient.

`_balance_experts` calls `model.balance_step(reduce)` if the model has one
(MoE router-bias balancing, section 7.4). `reduce` all-reduces expert loads
over `dp` so every rank applies the same bias update.

### 3.4 The step loop

`_run_steps` keeps loss, grad norm and extra metrics **on the GPU** and only
calls `.item()` at log points, so the CPU can queue work ahead of the GPU.
At a log / save / stop point it all-reduces the loss sum over `dp` (avg),
divides by the window length, prints and logs on rank 0, and resets.
Step time is averaged over the log window and excludes checkpoint saves.

`tokens_per_step = seq_len x global_batch_size`; `tps = tokens_per_step /
step_time`.

### 3.5 SIGTERM

`train()` installs `_on_sigterm`, which only sets a flag. Ranks receive the
signal at different times, so `_stop_requested()` all-reduces the flag with
`max` over `dp` (and `tp`): every rank sees the same answer on the same step.
That costs a host sync, so it runs every `stop_check_every` steps (default
`log_every`). A stop turns the current step into a log + save point, then
breaks. A second SIGTERM more than 5 s after the first restores the default
handler and kills the process; the 5 s window exists because one kill often
arrives twice (directly, and forwarded by torchrun).

`train()`'s `finally` waits for the last async checkpoint write and finishes
the wandb run, so a crash shows as crashed rather than "running".

---

## 4. Core: mesh, collectives, placements

### 4.1 `core/mesh.py`

The world's ranks are arranged as an n-d grid with named axes, laid out
row-major: the **last axis varies fastest**, so its peers are adjacent ranks.

```
spec {"dp": 2, "tp": 4}     strides {"dp": 4, "tp": 1}
rank 5 → dp coordinate 1, tp coordinate 1
its tp group is ranks [4,5,6,7]; its dp group is ranks [1,5]
```

- `strides[axis]` is how far apart in rank two neighbours along that axis
  are. `coordinate(axis) = (rank // stride) % size`.
- Along each axis a rank belongs to exactly one `ProcessGroup`: the ranks
  sharing its coordinates on every *other* axis. `dist.new_group` is itself
  collective, so every rank creates every group in the same order and keeps
  the ones it is in.
- `flatten(axes, name)` registers a **virtual axis** treating several axes as
  one row-major axis. It buckets ranks by "rank minus its offset along
  `axes`", creates one group per bucket, and records this rank's index in
  `_flat_coords` (strides cannot express a flattened coordinate).

`build_mesh(spec)` is what the trainer uses:

- puts axes in the fixed `AXIS_ORDER = (pp, dp_replicate, dp_shard, cp, tp, ep)`;
- reads `dp: N` as `dp_shard: N`;
- drops size-1 axes (each costs `world_size` `new_group` calls), except
  `dp_replicate` and `dp_shard`, which `flatten` needs;
- adds derived axes:

| axis | made of | used for |
|---|---|---|
| `dp` | `dp_replicate x dp_shard` (`x ep` under EP) | every rank that reads its own data: dataloader split, loss averaging, dense grad sync |
| `dp_expert` | `dp_replicate x dp_shard` (only with EP) | ranks holding a replica of the same experts |
| `dp_replicate_ep` | `dp_replicate x ep` (only with EP) | ranks holding a whole replica of a dense FSDP shard |

### 4.2 `core/collective.py`

The only module that calls `torch.distributed` communication ops. Callers
pass `(tensor, mesh, axis_name)`, never a `ProcessGroup`.

| collective | per-rank size | notes |
|---|---|---|
| `all_reduce` | same, in place | `"avg"` is emulated on gloo (divide locally, then sum) |
| `broadcast` | same, in place | `src` is a coordinate within the group, not a global rank |
| `all_gather` | x g | concatenates along `gather_dim`, ordered by coordinate; optional caller-owned `output_tensor` |
| `reduce_scatter` | / g | reduce, then keep your coordinate's shard along `scatter_dim` |
| `all_to_all` | same | even (a transpose) or uneven with `input_splits` / `output_splits` (EP's token dispatch) |
| `ring_send_recv` | same | send to coord+1, receive from coord-1, batched so it cannot deadlock |
| `barrier` | | |

`async_op=True` returns `(tensor, work)`; the tensor is garbage until
`work.wait()`.

The identity `all_gather(reduce_scatter(x)) == all_reduce(x)` is tested and
is exactly FSDP's decomposition of DDP's all-reduce.

`MeshLike` is a `Protocol` (`size`, `get_group`, `coordinate`) so tests can
pass a stand-in.

### 4.3 `core/placement.py` and `core/dtensor.py`

A **placement** says how one tensor is laid out along one mesh axis:

- `Shard(dim)`: split evenly along tensor dim `dim`; coordinate `c` holds
  chunk `c`. Negative dims are rejected so `Shard(-1)` and `Shard(1)` cannot
  describe the same layout yet compare unequal.
- `Replicate()`: every rank holds the whole tensor.
- `Partial(op)`: every rank holds a contribution; the true value is their
  reduction. An unreduced gradient is `Partial`.

They are frozen dataclasses: hashable, comparable by value. Pure local math
lives here (`local_shape`, `global_shape`, `local_slice`).

`padded_chunk(global_size, group_size, coord) -> (length, start, valid)` is
the opt-in padded layout for a dim that does not divide evenly: every rank
holds `ceil(global / group)` rows, of which the first `valid` are real. FSDP
and the checkpointer both use it.

`DTensor` is `(local_tensor, mesh, mesh_dims, placements)` with an explicit
`redistribute`. Unlike PyTorch's `DTensor` it is **not** a
`__torch_dispatch__` subclass: placements are not propagated through ops.
The redistribute table is one collective per cell:

| from → to | collective |
|---|---|
| `Shard(d)` → `Replicate` | `all_gather(gather_dim=d)` |
| `Replicate` → `Shard(d)` | local slice, no comm |
| `Partial(op)` → `Replicate` | `all_reduce(op)` |
| `Partial(op)` → `Shard(d)` | `reduce_scatter(op, scatter_dim=d)` |
| `Shard(i)` → `Shard(j)` | `all_to_all(split_dim=j, concat_dim=i)` |

There are no transitions *into* `Partial`.

In practice the training hot paths (FSDP, TP) call collectives directly
because they know exactly which one they want. `Shard` and `padded_chunk`
are used everywhere; `DTensor` itself is the vocabulary and the tested
reference, and the optimizer never sees one.

### 4.4 `core/grad_helper.py`

`clip_grad_norm` is in section 10.

---

## 5. The dataloader

Files: `dataloader/{builder,mixture,sources,shards,prefetcher,tokenize_shards}.py`.

There is one loader, `MixtureLoader`. A single dataset (`data.source`) is a
mix of one source at weight 1. The design goal was **random access**: once a
block of data is addressable by a byte range, resume, rank splitting,
shuffling and mixing all reduce to arithmetic.

### 5.1 Layers

```
MixtureLoader            per-step plan: which source each row comes from
  └─ SourceStream        one rank's endless shuffled row stream over one source
       └─ Source         blocks + fetch(block) → {key: array of rows}
            └─ Reader    byte-range reads: local pread or http Range
CUDAPrefetcher           host → device copy on a side stream, one batch ahead
```

### 5.2 `Reader` (`shards.py`)

Byte-range reads from a local path, any HTTP server that honours `Range`, or
`hf://datasets/<org>/<name>[/<subdir>]` (rewritten to the hub's
`resolve/main` URL by `resolve`).

Over HTTP:

- One kept-alive connection per (host, thread), stored in a
  `threading.local`.
- **Redirect cache.** The hub answers every resolve URL with a redirect to a
  presigned CDN URL. A fresh connection plus that redirect was most of a
  3.5 MB read (0.38 s vs 0.08 s). The target is cached per file; if the
  cached target returns 401/403/404 (signature expired) it is dropped and
  resolved again.
- `HF_TOKEN` is sent only to `huggingface.co`, never to the redirect target.
- Retries on 408/425/429/5xx and connection errors, with exponential backoff
  and jitter capped at a minute. A server that ignores `Range` (answers 200)
  is an error; a short body is retried.
- `__getstate__` drops the thread-local so a `Reader` can be pickled to
  worker processes.

### 5.3 Source kinds (`sources.py`, `shards.py`)

`open_source(spec, ...)` picks the kind from the files at the location. Every
source exposes the same surface:

```
source.blocks         descriptors; a pure function of the files, identical on every rank
source.fetch(block)   {key: array}, one row per leading index; deterministic
source.keys           ("input_ids",) or ("input_ids", "labels", "pixel_values")
source.eos_id
source.window_blocks  default blocks per shuffle window, or None
```

| kind | block | how it is read |
|---|---|---|
| `*.jsonl` (`JsonlSource`) | a ~`block_mb` byte range | see below |
| `*.jsonl.zst` / `.gz` (`CompressedJsonlSource`) | a whole file | compressed streams do not seek; `window_blocks = 1` |
| `*.parquet` (`ParquetSource`) | one row group | footers read once at startup (last 64 KB of each file, 32 threads); a row group is one range read covering just the needed columns |
| `index.json` + `*.bin` (`TokenSource`) | `block_mb` of pre-tokenized `seq_len` rows | written by `tokenize_shards.py` |

**JSONL without an index.** A block `[start, end)` owns the lines that
*start* in it. It reads from `start - 1` and skips to just after the first
newline it finds. If the block begins on a line boundary that newline is
byte `start - 1` itself and nothing is lost; otherwise the skipped partial
line belongs to the previous block. Then it reads past `end`, in growing
chunks, until its last line's newline. Every line therefore has exactly one owner
whatever the block size; a line longer than a block leaves the blocks inside
it empty.

**Parquet.** `_SpanFile` is a fake seekable file handed to pyarrow: reads
inside a preloaded byte span come from memory, anything else is a range
read. `_span(block)` computes the byte range covering the row group's column
chunks for just the fields being read.

**Text rows** (`RawSource.tokenize` / `pack`): the block's docs go through
`tokenizer.encode_batch`, each doc is followed by eos, everything is
concatenated and cut into `seq_len` rows. The block's tail (< `seq_len`
tokens) is dropped, about 0.1% of a 16 MB block. Token dtype is `uint16` if
the vocab fits, else `uint32`. Rows are **packed**: a row can hold several
documents and a document can span rows. No padding, no attention mask.

**Caption rows** (`image_field` set, parquet only), for a VLM. One image and
one caption per row, not packed:

```
<|vision_start|> <|image_pad|> x num_image_tokens <|vision_end|> caption eos pad...
```

`labels` are `ignore_index` over the image span and the padding. Images are
decoded with PIL on the fetch threads (PIL releases the GIL while decoding);
JPEGs use `draft()` to downscale by 2/4/8 inside the decoder before the
bicubic resize. `pixel_values` stay **uint8** `(3, S, S)`; normalizing is the
vision tower's job, on the GPU, after a 4x smaller host-to-device copy.

`RowConfig` carries what a raw source needs (`seq_len`, tokenizer, eos/pad
ids, `ignore_index`, and `Vision(image_size, num_image_tokens)`). The trainer
passes `vision` only if the model has a `num_image_tokens` attribute.

**Token shards** (`TokenSource`): `index.json` records dtype, tokenizer,
eos id and the files with their token counts. Rows are cut per file. The
loader refuses a source whose recorded tokenizer differs from the config's.
`tokenize_shards.py` produces these from any raw source: blocks are
tokenized on a thread pool, written in block order, docs never straddle
`.bin` files, and `index.json` is written last (via `os.replace`) so a source
is readable only once complete.

Both source families shrink their block size until there are at least
`dp_size` blocks, because every dp rank must own at least one.

### 5.4 `SourceStream`: one rank's stream over one source

Position is three integers: `(epoch, window, pos)`.

- **Blocks per epoch.** `perm = rng(seed, source_key, epoch).permutation(n_blocks)`;
  this rank takes `perm[dp_rank::dp_size]`. Ranks get disjoint blocks, so
  each downloads 1/dp of the data. `source_key` is `crc32(name)`, so adding a
  source to a mix does not reshuffle the others.
- **Windows.** The rank's blocks are grouped into windows of `window_blocks`.
  A window's rows are permuted with
  `rng(seed, source_key, epoch, dp_rank, window)`.
- **Prefetch.** `_load` makes the current window resident and submits the
  next `prefetch_windows` windows to the pool, crossing an epoch boundary if
  needed.
- **No concatenation.** A window can be hundreds of MB. Its blocks are kept
  as separate parts; `_where[row] = (part, row_in_part)` maps a window row to
  its block, and rows are gathered by index straight into the batch tensor.
- An empty window (raw blocks shorter than `seq_len`) is skipped; a whole
  epoch of empties is an assertion.

`take()` returns one row and advances. Because `fetch(block)` is
deterministic, `state_dict()` is just `{epoch, window, pos}` and resume
refetches one window: **O(1) resume**.

### 5.5 `MixtureLoader`: the per-step plan

A mix script is plain Python:

```python
SOURCES = {"climbmix": "hf://datasets/OptimalScale/ClimbMix",
           "fineweb_edu": "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT"}

def weights(step, total_steps):
    edu = 0.1 if step < 0.9 * total_steps else 0.5
    return {"climbmix": 1 - edu, "fineweb_edu": edu}
```

`weights` must be a pure function of its arguments: every rank calls it
independently and a resume calls it again.

Per optimizer step (`_plan(step)`):

1. **Counts.** `allocate(credit, weights, global_batch_size)`:
   `owed = credit + weights * total`; counts are `floor(owed)`, then the
   shortfall goes to the largest remainders so the counts sum to exactly
   `global_batch_size`. The rounding error is carried as `credit`, so a
   source at weight 0.001 still gets its share over time instead of rounding
   to zero every step.
2. **Order.** The step's source labels (`np.repeat(arange, counts)`) are
   shuffled with `rng(seed, step)`.
3. **Rank split.** Rank `r`'s micro batch `m` reads label positions
   `[(m * dp_size + r) * micro, + micro)`.

Every rank computes the whole (cheap) plan, so ranks need no communication
to agree on who reads what.

`__next__` takes `micro` rows from the named streams and gathers them into
**pinned** tensors with one copy (`np.stack(..., out=tensor.numpy())`):
`input_ids` int64, `labels` (a copy of `input_ids` for text; the source's
own for captions), and uint8 `pixel_values` for caption sources. Labels are
**unshifted**; the next-token shift is the objective's job.

**Threads vs processes.** By default (`workers: 0`) blocks are fetched,
tokenized and decoded on `fetch_threads` threads inside the trainer process;
`tokenizers` and PIL release the GIL for the heavy parts. With `workers: N`,
N spawned processes each fetch whole blocks and send rows back. That is
worth it for images (decoding held the GIL enough to slow the training loop
15-30%) and not for text. Workers ignore SIGTERM, since the trainer still
needs batches until its checkpoint; `close()` kills them.

**State.** `state_dict()` is `micro_index`, the credit carried into the
current step, each stream's `(epoch, window, pos)`, the mix script's text,
and the batch shape. `load_state_dict` asserts the shape matches, notes if
the mix changed (the new schedule applies from there), and tolerates sources
being added or removed. `reseed(seed)` gives a different shuffle; the trainer
uses `seed + step` when resuming without loader state.

### 5.6 `CUDAPrefetcher`

Holds one batch ahead. `_preload` copies batch N+1 to the GPU with
`non_blocking=True` on a side `torch.cuda.Stream` while batch N computes.
`__next__` waits on that stream, calls `record_stream` on each tensor (so the
allocator does not recycle them while compute still uses them), and preloads
the next.

Because it is one batch ahead, the loader's saved state is one batch ahead
of training. `state_dict()` returns the held batch; a checkpoint saves it and
resume hands it back as `first_batch`, so the resumed run sees every batch
exactly once.

### 5.7 Leftovers

`dataloader/packers.py` (`ListPacker`, `NumpyPacker`, `TensorPacker`,
`ListPackerBestFit`) is from the older HF `.map` pipeline. The loader does
not use it; only `tests/dataloader_packing.py` does. The best-fit packer
(segment tree over free space, best-fit-decreasing) is a reference
implementation: it saved only ~0.04% of tokens on ClimbMix.

---

## 6. Models: shared layers, Qwen3, OLMo 3

### 6.1 The trainer contract

Every model provides:

| member | used by |
|---|---|
| `blocks` (`nn.ModuleList`) | per-block compile, FSDP units |
| `norm` | compiled separately |
| `lm_head.weight` | the fused loss reads it directly |
| `init_weights()` | elementwise; also fills buffers after a meta build |
| `tp_plan(tp_size, vocab_parallel)` | tensor parallel |
| `activation_checkpointing` | set by the trainer |
| `forward(ids, attention_mask=None, return_hidden_states=False, return_final_hidden=False)` | `return_final_hidden=True` returns `norm(x)` and skips the head |
| `doc_mask_eos` (optional) | document masking |
| `ep_plan()`, `balance_step()` (optional) | MoE |
| `num_image_tokens`, `image_size` (optional) | VLM caption rows |

`models/builder.py` maps `model.name` to a class through `MODEL_REGISTRY`
(`qwen3_dense`, `olmo3`, `kimi_k31` on main).

### 6.2 `models/layers.py`

**RoPE.** `RotaryEmbedding` holds `inv_freq` as a *non-persistent* buffer
filled by `init_buffers()`, so a model built on `meta` gets it when
`init_weights()` runs. `frequencies()` returns `(inv_freq, scale)`; plain
RoPE returns `1 / base^(2i/d)` and scale 1, and a subclass overrides it.
`forward(position_ids, cacheable)` returns `(cos, sin)` shaped
`(B, 1, S, D)` in fp32; when positions are a plain `arange` it caches one
row per `(seq_len, device)`. `apply_rotary` is
`x * cos + rotate_half(x) * sin`.

**`positions(ids, attention_mask)`** returns `(position_ids, bool mask,
cacheable)`. With a mask, only non-pad tokens are numbered (cumsum of the
mask minus 1), so left padding works with RoPE.

**Document masking** (`doc_mask`). For packed rows, each token should attend
only to earlier tokens of its own document, and positions should restart per
document:

```python
starts[:, 1:] = ids[:, :-1] == eos_id        # a doc starts right after an eos
doc = starts.cumsum(1)                       # document index per token
position_ids = idx - cummax(where(starts, idx, 0))   # restart at each start
mask_mod(b, h, q, kv) = (doc[b,q] == doc[b,kv]) & (q >= kv) [& (q - kv < window)]
```

It returns `{window: BlockMask}` built with flex attention's
`create_block_mask`, once per forward, outside the compiled blocks, shared
by every layer. `masked_attention` runs a compiled `flex_attention` on GPU
(eager flex materializes the full score matrix) and falls back to a dense
mask through SDPA on CPU, where the compiled flex backward fails on torch
2.10. The README reports doc masking as *faster* on short-document data
(246k → 276k tok/s at 0.6B FSDP); the likely reason is that a `BlockMask`
lets flex attention skip fully masked blocks of the score matrix, though I
did not find that attribution stated in the notes.

**`SwiGLU`**: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
`swiglu_ffn_dim` is `8/3 * emb_dim` rounded up to a multiple of 256.
Attribute names inside these modules are part of every state_dict, so
renaming one breaks checkpoints.

### 6.3 Qwen3 (`models/qwen3/qwen3.py`)

A pre-norm decoder:

```
x = token_emb(ids)
for block:  x = x + gqa(norm1(x), cos, sin, mask)
            x = x + ffn(norm2(x))
logits = lm_head(norm(x))          # lm_head.weight is tied to token_emb.weight
```

- **Grouped-query attention.** `q_proj` produces `num_heads * head_dim`,
  `k_proj` / `v_proj` produce `num_kv_heads * head_dim`; `head_dim` is
  decoupled from `emb_dim / num_heads`. Per-head `RMSNorm(head_dim)` on Q and
  K (QK-norm) before RoPE. `split_heads` reads the head count **off the
  tensor width**, not `self.num_heads`, because under tensor parallel a rank
  holds only its slice of the heads.
- **Three attention paths** (`scaled_self_attention`):
  a `BlockMask` → flex attention (doc masking); no mask → SDPA with
  `is_causal=True, enable_gqa=True` (the flash kernel; packed training);
  a padding mask → an explicit bool causal-and-key mask (`is_causal` cannot be
  combined with a mask).
- **Init.** Normal(0, 0.02) for linears and embeddings, ones for RMSNorm,
  and `o_proj` / `down_proj` scaled by `1 / sqrt(2 * num_layers)` so
  residual variance does not compound with depth.
- **Activation checkpointing.** `torch.utils.checkpoint.checkpoint(block,
  ..., use_reentrant=False)` per block when grad is enabled: roughly one
  extra forward for most of the activation memory.
- **`tp_plan`.** Two regions per block: attention (q/k/v column, o row) and
  the MLP (gate/up column, down row). q/k/v split by whole heads and a KV
  head must stay with the Q heads that share it, so `tp` must divide
  `num_kv_heads`. `q_norm` / `k_norm` are `shared`; `norm1`, `norm2` and the
  final norm are `seq_local`; the tied embedding / head is `vocab` when
  `vocab_parallel`.

The base config is the 0.6B shape: 28 layers, 16 heads, 8 KV heads, width
1024, head dim 128, vocab 151,936.

### 6.4 OLMo 3 (`models/olmo3/olmo3.py`)

Faithful to `allenai/olmo-3-7b`, with two deliberate deviations.

- **Post-norm blocks**: `x = x + post_attn_norm(attn(x))`, then
  `x = x + post_ffn_norm(ffn(x))`. The sublayer output is normed, not its
  input.
- **3:1 sliding : full attention.** `_layer_types`: layer `i` is sliding
  unless `(i + 1) % 4 == 0`. Window 4096.
- **YaRN RoPE** (`_yarn_inv_freq`): frequencies are a ramp between
  extrapolation (`1 / pos_freqs`) and interpolation (`1 / (factor *
  pos_freqs)`), with `low` / `high` from `beta_fast` / `beta_slow`, and an
  attention factor `0.1 * ln(factor) + 1` multiplied into cos and sin (the
  `scale` slot of the base class). Base 500k, factor 8, original max 8192.
- **Untied head**, no biases anywhere.

Deviations:

- QK-norm is **per head** (`RMSNorm(head_dim)` after `split_heads`). HF norms
  the flattened `heads * head_dim` vector jointly. Per-head is what the
  paper describes and the only form that stays correct when heads are
  sharded over TP.
- The sliding window is an explicit bool mask on SDPA
  (`(q - kv) < window`), not a flash-attention window argument: same math,
  more memory at long sequences. With `S <= window` it takes the plain
  causal flash path.

With doc masking, `forward` builds one `BlockMask` per distinct window size
in use and hands every block the whole dict; each block picks its own.

`tp_plan` matches Qwen3's vocabulary; embedding and head are separate
params, each vocab-sharded on its own.

---

## 7. Kimi K3.1: KDA, MLA, latent MoE, attention residuals

`models/kimi_k31/kimi_k31.py`, `models/moe.py`. A scaled-down (~4B) version
of the Kimi K3.1 architecture: the structure is kept and every width, depth
and count is reduced.

Scaled defaults: 28 layers, hidden 2048, 16 heads. `_is_mla_layer`: 1-based
layers 4, 8, ... and the final layer are MLA, the rest KDA, giving 21 KDA +
7 MLA. The first layer has a dense `SiTUMLP`; the other 27 have a
`LatentMoE` with 56 experts, top-4, plus 1 shared expert.

### 7.1 SiTU-GLU

```
situ_glu(g, u) = (β1 * tanh(g / β1) * sigmoid(g)) * (β2 * tanh(u / β2))     β1 = 4, β2 = 25
```

A GLU whose gate and linear branches are both soft-clipped, so the output is
bounded by `β1 * β2 = 100`. `SiTUMLP` is `down(situ_glu(gate(x), up(x)))`.

### 7.2 KDA: Kimi Delta Attention

KDA is linear attention with a recurrent state instead of a softmax over
past tokens. Per head it keeps a `D x D` matrix `S` (a fast-weight
associative memory mapping keys to values) and applies the **gated delta
rule** at every timestep:

```
S ← (I − β_t k_t k_tᵀ) diag(α_t) S + β_t k_t v_tᵀ          o_t = Sᵀ q_t
```

Read it as three moves:

1. **Forget**: `diag(α_t) S` decays each key channel by `α_t ∈ (0, 1)`.
2. **Predict and correct**: `pred = Sᵀ k_t` is what the memory currently
   returns for this key. The write is `u_t = β_t (v_t − pred)`: only the
   *error* is stored (the delta rule), scaled by a write gate `β_t`.
3. **Write and read**: `S ← S + k_t u_tᵀ`, then `o_t = Sᵀ q_t`. The read
   happens after the write, so a token sees its own write.

**Inputs** (`KimiDeltaAttention.forward`):

- `q`, `k`, `v`: linear projection → depthwise **causal** conv (kernel 4,
  left-padded by 3) → SiLU on q and k (not v). q and k are then L2-normalized
  in fp32. Unit-norm keys keep `I − β k kᵀ` a contraction.
- **Forget gate**, per channel:
  `log α = −exp(A_log) * softplus(gate_proj(x) + dt_bias)`, clamped at
  `gate_lower_bound = −5`. `A_log` is a per-(head, channel) parameter
  initialized to `log 8`. Always ≤ 0, so `α ∈ (e⁻⁵, 1]`.
- **Write gate**: `β = sigmoid(beta_proj(x))`, one scalar per head.
- **Output**: per-head `RMSNorm`, multiplied by
  `sigmoid(out_gate_proj(x))` (a gated norm), then `o_proj`.

KDA uses **no positional encoding** (the `cos` / `sin` arguments are
ignored; order comes from the recurrence and the convs) and cannot apply a
padding mask, so only the packed path is supported.

**`_kda_reference`** is the rule exactly as written, one timestep at a time.
It is slow and memory hungry under autograd (it keeps a `(B, H, D, D)` state
per step) and exists as the oracle the fast path is tested against.

**`_kda_scan`: the chunked solve.** The sequence is cut into chunks of
`KDA_CHUNK = 64` steps. Within a chunk, let `g_t = Σ_{s ≤ t} log α_s` be the
log decay accumulated from the chunk's start, and `S0` the state entering
the chunk. The state just before step `t`'s write is

```
diag(e^{g_t}) S0 + Σ_{i<t} diag(e^{g_t − g_i}) k_i u_iᵀ
```

Substituting into `u_t = β_t (v_t − Sᵀ k_t)` gives a linear system in the
chunk's writes `U`:

```
(I + diag(β) A) U = diag(β) (V − (K ⊙ e^G) S0)
A[t, i] = Σ_d k_t[d] e^{g_t[d] − g_i[d]} k_i[d]     for i < t   (strictly lower triangular)
```

That is a **unit lower-triangular** system, solved with
`torch.linalg.solve_triangular(..., unitriangular=True)`. The right-hand
side is linear in `S0`, so `_kda_intra` solves for both parts at once by
stacking `[V, K ⊙ e^G]`:

```
U    = U0 − Wk · S0                         U0, Wk from the solve
out  = out0 + P · S0                        out0 = Aq · U0,  P = Q ⊙ e^G − Aq · Wk
S_end = diag(e^{g_end}) S0 + KLᵀ · U        KL = K ⊙ e^{g_end − G}
```

where `Aq` is the same kernel as `A` with `q_t` in place of `k_t`, **with**
the diagonal (a token reads its own write).

So everything that does not involve the state (`U0, Wk, out0, P, KL`) is
computed **for all chunks at once**, and only the state is carried
sequentially: two small matmuls per chunk in a Python loop
(`u = u0 − wk @ state; state = end_decay * state + klᵀ @ u`). The outputs
are then one batched matmul against every chunk's incoming state. The state
is kept in fp32.

**The overflow problem and the sub-block trick.** `A[t, i]` needs
`e^{g_t − g_i}` per channel. Forming it pairwise costs a `(C, C, D)` tensor.
Factoring it as `e^{g_t} · e^{−g_i}` would make it a plain matmul, but
`e^{−g_i}` overflows: with the −5 floor, `g` can reach −320 over 64 steps.
The code splits each chunk into sub-blocks of `KDA_SUB = 16`:

- **Across sub-blocks**, the decay is factored through the first step `r` of
  the later sub-block: `e^{g_t − g_r} · e^{g_r − g_i}`. Both exponents are
  ≤ 0, so both factors are ≤ 1 and it is a plain matmul.
- **Within a sub-block**, it is formed pairwise on a `16 x 16 x D` tensor
  (the exponent is clamped at 0), which never overflows.

Padding: the last chunk is zero-padded; a padded step has `k = β = 0` and
`log α = 0`, so it writes nothing and does not decay.

`_kda_intra` is `torch.compile`d on its own on GPU (the pairwise decay's
elementwise ops dominate eagerly: 213 ms → 128 ms fwd+bwd at B=8, S=4096,
H=16, D=128 on an H100), while `_kda_scan` around it is
`@torch.compiler.disable`d.

### 7.3 MLA: multi-head latent attention

`MultiHeadLatentAttention` is softmax attention with low-rank Q and KV
bottlenecks:

```
q  = q_up(q_norm(q_down(x)))                 emb → 512 → heads * (128 nope + 64 rope)
kv = kv_down(x)                              emb → 256 latent + 64 rope
k_nope, v = kv_up(kv_norm(latent))           256 → heads * (128 + 128)
```

- Position is applied only to the **RoPE slices**: `q_rope` per head, and
  `k_rope` as **one head shared by all heads** (it comes straight from
  `kv_down`, broadcast across heads). `Q = [q_nope, q_rope]`,
  `K = [k_nope, k_rope]`.
- Scale is `(qk_nope_dim + qk_rope_dim)^-0.5`.
- A per-head sigmoid **output gate** (`out_gate(x)`) multiplies the attention
  output before `o_proj`.
- **DSA** (optional, `dsa_topk`, off by default): a learned indexer
  (`index_q`, `index_k`) scores every (query, key) pair, averaged over index
  heads, and each query attends only to its causal top-k. At the scaled
  context it is an inference optimisation, not a quality lever.

The model's `RotaryEmbedding(qk_rope_dim)` exists only for these MLA slices.

### 7.4 Latent MoE (`LatentMoE`, `models/moe.py`)

```
z = latent_norm(down_proj(x))                  hidden 2048 → latent 1024, per token
probs = sigmoid(router(z))                     one score per expert, independent (not softmax)
indices = topk(probs + router_bias, k)         the bias only steers the CHOICE
weights = probs[indices] / sum(probs[indices]) mixing weights from the UNBIASED scores
routed = experts(z[token], counts)             (N*k, latent)
out = index_add over tokens of routed * weights
result = up_proj(out) + Σ shared_expert(x)     shared experts run on the hidden stream
```

**The expert bank.** `ExpertBank` stores a layer's experts as three weights
stacked on a leading expert dim, `(num_experts, out, in)` (`GroupedLinear`),
instead of `num_experts` modules. That buys:

- dispatch as one sort plus one grouped matmul per projection, not a Python
  loop of masked selects;
- a grad for the bank's params on every backward, even if some expert got no
  tokens (its rows are just zero);
- the expert dim is dim 0, which is what both FSDP and expert parallel
  shard.

`sort_by_expert(indices)` flattens the `(N, k)` choices, does a **stable**
argsort by expert id, and returns `(token index, slot index, counts per
expert)`. `grouped_linear(x, weight, counts)` applies
`x[group g's rows] @ weight[g].T`; on CUDA in bf16 it uses `F.grouped_mm`
(one jagged grouped GEMM kernel for all experts), otherwise a loop.

`ExpertBank.forward(x, counts)` is the single step expert parallel replaces:
if `bank.remote` is set, the rows travel to the rank holding their expert
(section 9.4); otherwise `bank.local` runs `down(act(gate(x), up(x)))`.

`LatentMoE.forward` is `@torch.compiler.disable`d (data-dependent sort and
counts).

**Aux-loss-free balancing.** There is no balancing loss. `router_bias` is a
*buffer*, added to the scores only to pick experts, and moved by a rule
rather than by gradient. Each forward (when grad is enabled, so not eval and
not the no-grad first pass of a checkpointed block) adds the per-expert pair
counts into a `load` buffer. After each optimizer step
`KimiK31.balance_step(reduce)`:

```
load = reduce(stack(load per layer))                       # summed over all data ranks
router_bias += router_bias_rate * sign(mean(load) − load)  # toward under-used experts
load.zero_()
```

and returns `balance_stats`: min and max load relative to uniform (worst
layer) and the dead-expert count. Summing over ranks first means every rank
applies the same update and the bias stays replicated.

### 7.5 Block attention residuals

With `attn_res=True` the model replaces the plain residual stream with a
learned mix over depth. Layers are grouped into residual blocks of
`attn_res_block_size` (6). The forward keeps:

- `completed`: `[emb, sum of block 1's sublayer outputs, sum of block 2's, ...]`
- `partial`: the running sum of sublayer outputs in the current block.

Before each sublayer the input is

```
attn_res_mix(sources, w, norm):   α = softmax over sources of ( w · rmsnorm(source) )
                                  h = Σ α_i * source_i
```

with a learned pseudo-query `w` per sublayer (`attn_res_q_attn`,
`attn_res_q_mlp`). Sublayer outputs accumulate into `partial`, not into the
mix. The final hidden state is a mix over `completed` with
`attn_res_q_final`. The queries are initialized to zero, so every mix starts
as a uniform average. At a block boundary `partial` is all zeros and is left
out of the sources (`has_partial=False`).

### 7.6 Parallelism for Kimi

`tp_plan` is **vocab only**: the KDA scan, MLA's latent bottleneck and
expert dispatch do not fit the TP engine's `(B, S, D)` region vocabulary, so
block weights stay replicated over `tp` and there are no regions (the
trainer turns sequence parallel off when a plan has none). The scaling paths
are FSDP over `blocks` and expert parallel over the routed experts
(`ep_plan()` returns each MoE layer's bank).

The configs set `compile: false` and `activation_checkpointing: true`.

---

## 8. The loss: fused linear + cross-entropy

Files: `objectives/autoregressive.py`, `loss/fused_linear_ce.py`,
`loss/cross_entropy.py`.

### 8.1 The objective

`ARObjective.compute_loss(model, batch)`:

- **Fused path** (default): `hidden = model(ids, return_final_hidden=True)`
  (the head is skipped), shift the labels by one (`labels[:, :-1] =
  batch["labels"][:, 1:]`, last position `ignore_index`), then
  `fused_linear_cross_entropy(hidden, model.lm_head.weight, labels, ...)`.
  If TP is on, `tp.head_input(hidden)` first gathers the sequence (what the
  head's hooks would have done), and `vocab_axis=tp.vocab_axis` selects the
  vocab-parallel variant.
- **Unfused path**: `logits = model(ids)`, then `cross_entropy_loss`, which
  does the same shift and calls `F.cross_entropy`. The inference scripts use
  this.

### 8.2 Why fuse

With a ~152k vocab, the full logits are the largest activation in the step:
`B * S * V` is 2.5 GB in bf16 at `B=2, S=4096`, plus the fp32 upcast inside
`F.cross_entropy`, plus the same again for the logits' gradient.

The trick: for cross-entropy, `d(loss) / d(logits) = softmax − onehot`. The
gradient is known in closed form **during the forward pass**. So the forward
can compute the loss *and* the gradients with respect to `hidden` and `W`
chunk by chunk, and nothing logits-sized has to survive a chunk.

### 8.3 `_FusedLinearCE` (single device / replicated head)

`forward(hidden (N, D), weight (V, D), labels (N,), ignore_index, chunk_size)`:

For each chunk of `chunk_size` rows, the compiled kernel `_chunk_fwd` does:

```python
logits = (hidden @ weight.T).float()                 # (C, V) fp32
valid  = labels != ignore_index
lse    = logsumexp(logits, -1)
target = logits.gather(1, safe_labels)
loss_sum = ((lse - target) * valid).sum()
probs  = exp(logits - lse) * valid                   # softmax on valid rows, 0 elsewhere
probs.scatter_add_(1, safe_labels, -valid)           # minus the one-hot
grad_logits = probs.to(hidden.dtype)                 # (C, V) bf16
grad_hidden = grad_logits @ weight                   # (C, D)
```

The caller writes `grad_hidden` into a preallocated `(N, D)` buffer and
accumulates the weight gradient with one

```python
torch.addmm(grad_weight, grad_logits.T, hidden_chunk, out_dtype=torch.float32, out=grad_weight)
```

That is a bf16 GEMM accumulating straight into an fp32 `(V, D)` buffer: no
per-chunk bf16 partial and no lossy rounding. `grad_logits` is deleted
before the next chunk so it does not sit at peak memory.

The forward returns `loss_sum / n_valid` and stashes `grad_hidden`,
`grad_weight` and `n_valid` as plain `ctx` attributes (not
`save_for_backward`), so backward can modify them in place and drop them
early.

`backward(grad_out)` is `_scaled_grads`: scale both stashed gradients **in
place** by `grad_out / n_valid` and return them. `grad_weight` is a full
fp32 `(V, D)`; a copy would double it. The `ctx` references are cleared so
the buffers free as soon as autograd consumes them. Consequence: backward
can run once; a second call raises a clear error (`retain_graph` does not
work through this op).

GEMM count is the same as the unfused path (one for logits, one for
`grad_hidden`, one for `grad_W`); only where they happen moves.

Two details in the wrapper `fused_linear_cross_entropy`:

- 3-d `hidden` / 2-d `labels` are flattened.
- The final norm returns fp32 even under autocast, and autocast does not
  reach inside an `autograd.Function`, so `hidden` is cast to the autocast
  dtype here; otherwise everything would run in fp32.

Because the returned `grad_weight` flows to `lm_head.weight` as an ordinary
autograd input, the grad-accumulate hooks DDP and FSDP hang off that
parameter fire as usual. This is also why FSDP keeps the root unit (which
holds the head weight) gathered from forward until its backward: code
outside `model.forward` is still reading it.

### 8.4 `_VocabParallelFusedLinearCE`

Under tensor parallel with `vocab_parallel`, `weight` is this rank's slice
of the vocab rows, so the logits are a slice and the softmax needs every
rank's. Per chunk:

```
local logits → row max                     ── all_reduce(max) ──▶ global max
Σ exp(logits − max), and the target logit
  where this rank owns the label (else 0)  ── all_reduce(sum) ──▶ logsumexp, target
```

Two all-reduces of a few floats per row. After that the gradient with
respect to the local logits is local (`softmax − onehot` restricted to this
rank's rows; the one-hot is subtracted only on the rank that owns the
label). `grad_hidden` on each rank is the part that flows through its rows,
a partial sum, all-reduced **once** at the end. `hidden` and `labels` are
still full and replicated, and every rank returns the same loss.

The three pieces between the collectives (`_vp_logits`, `_vp_stats`,
`_vp_grads`) are compiled on CUDA and run eagerly on CPU, where the
`out_dtype` overload of `addmm` is also unavailable.

---

## 9. Parallelism: DDP, FSDP2, TP, EP

All four share a small surface so the trainer and checkpointer never branch
on the strategy: `shard_dims`, `shard_dim_of(param)`,
`full_shape_of(param)`, `sync()`, and for DDP / FSDP `requires_sync`.

### 9.1 DDP (`parallelism/data_parallel.py`)

Replicate the model along `dp`; keep replicas in lockstep by averaging
gradients after every backward.

- **Construction.** `_replicate` broadcasts every param and buffer from
  coordinate 0. Trainable params are taken in **reverse**
  `model.parameters()` order (roughly the order backward produces grads) and
  packed greedily into buckets of up to `bucket_mb` (25 MB). A new bucket
  also starts where the dtype or the reduce axis changes.
- **Buckets.** A multi-param bucket owns one flat buffer with a view per
  param. A single-param bucket (a param at or over the cap, such as the tied
  embedding) is reduced directly through its `.grad`, with no staging copy.
- **Hooks.** Each param gets a `register_post_accumulate_grad_hook`. When a
  grad lands, the bucket's `pending` count drops and `_launch_ready` launches
  buckets **strictly in bucket order**, stopping at the first one still owed
  a grad.
- **Launch.** Copy the bucket's grads into the flat buffer
  (`torch._foreach_copy_`), repoint each `.grad` at its view, and
  `all_reduce(flat, "avg", async_op=True)`. The reduce runs on NCCL's stream
  while backward keeps computing: that is the comm/compute overlap.
- **`reduce_dtype`.** If set (e.g. bf16), a cast copy goes on the wire and
  the average is copied back into the fp32 grads at `sync()`: half the bytes
  for a rounding of the average.
- **`sync()`.** Force-launch any bucket still unready, wait on every handle,
  copy back wire results, reset.
- **Grad accumulation.** With `requires_sync = False` the hook returns
  immediately and grads simply accumulate locally. The last microbatch sets
  it `True`, so its backward launches the reduces on the accumulated grads:
  one all-reduce per *step*, however many microbatches.
- **Missing grads.** A param that got no grad this step (an MoE expert no
  token picked) leaves its bucket unready. `sync()` launches what is left
  with zeros for the missing grads. Ranks can differ in which grads are
  missing; the fixed bucket order keeps their collectives paired.
- **`dim_of(param)`.** Per-param override of the reduce axis. Under expert
  parallel, expert weights average over `dp_expert` and everything else over
  `dp`.

`shard_dims` is `()`: DDP shards nothing.

### 9.2 FSDP2 (`parallelism/fsdp2.py`)

Per-parameter `Shard(0)` along `dp_shard`; HSDP when a replicate axis is
also set.

**The core idea.** Each param keeps **one** `nn.Parameter` object whose
`.data` is swapped between two storages:

| storage | dtype | when it is `param.data` |
|---|---|---|
| `sharded` | fp32, this rank's rows (zero-padded) | between steps: what the optimizer sees |
| `full` (a view of `unsharded` without padding) | `param_dtype` (bf16) | during forward and backward |

Keeping one `Parameter` means tied weights stay tied, grad hooks register
once, and the optimizer never sees a wrapper type.

**Units.** The caller passes the modules to gather and reshard
independently (`units=model.blocks`). Everything *not* inside a passed unit
forms **unit 0, the root** (embedding, final norm, head). FSDP reads no
model attribute names. Units must not nest or share a param.

**Lifecycle of a block unit** (`_FREE → _GATHERING → _UNSHARDED → _FREE`):

| hook | action |
|---|---|
| forward-pre | wait for this unit's all-gather; point params at the full buffers; start gathering the next `prefetch` units |
| forward-post | reshard: point params back at shards and **resize the buffer's storage to 0** |
| backward-pre | re-gather; start gathering the previous `prefetch` units |
| grad hook, when the unit's last grad lands | `_post_backward`: reduce-scatter grads, reshard |

The root unit is gathered at forward and stays gathered until its own
backward, because code outside `model.forward` (the fused loss) still reads
the head weight.

**Freeing memory** (`_free_storage`). Dropping the Python reference would
not free a gathered weight: activations saved for backward keep the tensor
alive. `untyped_storage().resize_(0)` frees the bytes while keeping the same
tensor object and shape, which also keeps `torch.compile`'s guards stable
across steps.

**Gather.** `_start_gather` allocates the buffer storage, casts the fp32
shard to `param_dtype` if needed, and launches
`all_gather(..., async_op=True, output_tensor=st.unsharded)` straight into
the unit's buffer. It keeps the (possibly cast) source alive until the
gather completes. `_wait_gather` blocks, then sets `param.data = st.full`.
Prefetch means unit `i+1`'s gather runs while unit `i` computes. The forward
order of units is recorded on the first step and then pinned
(`_check_order`); a wrong order would silently turn prefetch off.

**Gradients** (`_post_backward`). For each param of the unit: take `.grad`
(or zeros if it got none), cast to `reduce_dtype`, zero-pad to the padded
length, and `reduce_scatter(grad, "avg", async_op=True)`. Before launching,
`_drain` waits on the *previous* unit's reduce-scatters, which had this
unit's whole backward to complete, so the wait is usually free and in-flight
unsharded grads are bounded to about one unit's worth. Results accumulate
into `st.sharded_grad` in fp32.

Grads are reduce-scattered **every microbatch**. Holding a full unsharded
grad per unit across an accumulation step is exactly the memory FSDP exists
to reclaim, so `requires_sync` is ignored. The cost: each microbatch pays a
gather in forward, a gather in backward and a reduce-scatter. This is why
the notes recommend a larger `micro_batch_size` over more accumulation
steps under FSDP.

**HSDP.** With a replicate axis, `_drain` chains a second collective: after
a reduce-scatter (avg over `dp_shard`) completes, it launches an async
`all_reduce(avg)` of the sharded result over the replicate axis, finishing
the ones from the unit before.

**`sync()`.** Close any unit left short, flush both pipelines, and publish
`param.grad = sharded_grad`. Afterwards every `.grad` is a sharded fp32
tensor the same shape as `param.data`, so the optimizer just works.

**Uneven shards.** A dim-0 size that does not divide by the shard count is
padded (`padded_chunk`): every rank's shard has `ceil(rows / shards)` rows,
zero past the rows it owns. The gathered buffer carries the padding at its
tail and `full` is a `narrow` view of the real rows. Padding rows get zero
grads, so they stay zero.

**Meta init.** From a meta model, `_materialize` allocates only this rank's
(padded) shard for each trainable param. The caller runs the model's init on
the shards and then `finish_init()`, which re-zeros the padding the init
wrote over and broadcasts shards along the replicate axis and buffers along
both.

**Activation checkpointing.** A checkpointed unit's forward is re-run inside
its backward. `unit.in_backward` makes the forward hooks stand down for that
re-run: the backward-pre hook already gathered the unit, and resharding
there would free weights the backward is about to read.

**Missing grads.** A unit's grads are reduced once every one of its params
has a grad. A unit left short is closed with zeros: a block unit when
backward moves on to another unit (`_close_open_units`), the rest at the
next forward or at `sync()` (`_close_backward`). A late close lands exactly
where an on-time one would, so every rank issues the same collectives in the
same order.

All hooks are `@torch.compiler.disable`d: they resize storages, which
inductor cannot compile. The graph break lands on a unit boundary, which is
also a compile boundary, so it costs nothing.

`shard_dims` is `(dp_shard,)`. `reshard()` exists for eval or generation: a
forward with no backward leaves the root gathered.

Not done yet (per the file): a dedicated comm stream, flat per-unit buffers.

### 9.3 Tensor parallel (`parallelism/tensor_parallel.py`, `tp_overlap.py`)

Split a layer's weights across the `tp` axis so each rank computes a slice
of the layer's features. TP peers share a dp coordinate and therefore see
the **same batch**.

**Two ways to split a linear `weight[out, in]`:**

| | weight | input | output | forward comm | backward |
|---|---|---|---|---|---|
| **column** | `Shard(0)` (output features) | replicated | this rank's slice | none | grad of input is a partial sum |
| **row** | `Shard(1)` (input features) | already feature-sharded | a partial sum of the full output | reduce | none |

They pair up as a **region**: a module whose input and output are replicated
and whose inside is sharded. Column linears first, then ops that never mix
features across the split (elementwise ops, per-head attention), then one
row linear. A region costs one collective in forward (its output) and one in
backward (the grad of its input).

For Qwen3's attention: q/k/v are column (each rank gets whole heads), the
attention itself is per head, and `o_proj` is row. For the MLP: gate/up are
column, the SiLU and product are elementwise, `down_proj` is row.

**The plan.** The model hands over lists of modules:

| key | meaning |
|---|---|
| `column`, `row` | linears to shard on dim 0 / dim 1 (row linears must have no bias) |
| `regions` | modules that get the two collectives |
| `shared` | modules *inside* a region whose params are replicated but see only this rank's features (per-head norms); their grads are partial sums |
| `embeddings`, `heads` | modules at the ends of the residual stream |
| `vocab` | the subset of those sharded by vocab rows |
| `seq_local` | replicated modules *between* regions (block norms); matters only under sequence parallel |

**Sharding** (`_shard`). On a meta model it just replaces the parameter with
one of the local shape. On real weights it keeps an owning copy of this
rank's slice. A tied weight reached twice is sharded once and both modules
point at the same new `Parameter`. Entries are tracked by `(module,
attribute)` rather than by parameter id, because materializing from meta
replaces the `Parameter` objects.

**The autograd functions.** Each pairs a forward collective with its
backward dual:

| function | forward | backward |
|---|---|---|
| `_ReduceGrad` | identity | `all_reduce(sum)` |
| `_ReduceOutput` | `all_reduce(sum)` | identity |
| `_Gather(partial)` | `all_gather(dim)` | `reduce_scatter(sum)` if consumers produce partial grads, else local slice |
| `_Scatter` | `reduce_scatter(sum, dim)` | `all_gather` |
| `_Slice` | local slice | `all_gather` |

All launch collectives and are `@torch.compiler.disable`d; a compiled block
splits into graphs around them.

**Plain TP** (no sequence parallel): region entry is `_ReduceGrad`, region
exit is `_ReduceOutput`. Between regions every rank holds all activations.

**Sequence parallel.** Between regions the activations are replicated, which
wastes memory and compute. SP shards them along the sequence (or batch)
dim instead: a region **gathers** the sequence on entry and
**reduce-scatters** it on exit, the same bytes on the wire as the two
all-reduces it replaces. Norms and residual adds between regions then run on
1/tp of the tokens. Their params (`seq_local`) get partial grads, so they
join `shared`.

`sp_dim: "batch"` shards dim 0 instead of dim 1: each rank's chunk is
contiguous (no copies around the collectives) but it needs
`micro_batch_size % tp == 0`.

**Vocab parallel.**

- *Embedding* (`_embedding_pre` / `_embedding_post`): subtract this rank's
  row offset from the ids, look up row 0 for ids it does not hold, and zero
  those outputs. The ranks' outputs sum to the full lookup, which is exactly
  a region exit: an all-reduce, or under SP a reduce-scatter that sums and
  shards the sequence in one collective.
- *Head*: the head is a column linear over the vocab. Called as a module it
  reduces the input's grad and gathers the logit slices into full logits.
  The memory-efficient path is the fused loss, which takes the local weight
  and never builds full logits (section 8.4).

`head_input(hidden)` gathers the sequence under SP; the head's hooks call
it, and a loss that bypasses the head module must call it itself (the
trainer's objective does).

The full table:

```
op                     forward              backward
region entry           identity             all_reduce
  under sp             all_gather(seq)      reduce_scatter(seq)
region exit            all_reduce           identity
  under sp             reduce_scatter(seq)  all_gather(seq)
embedding out, sp      (as region exit when vocab-sharded, else:)
                       local slice(seq)     all_gather(seq)
head input, sp         all_gather(seq)      local slice(seq)
head input, vocab      identity             all_reduce
head output, vocab     all_gather(vocab)    local slice(vocab)
```

**`sync()`.** One `all_reduce(sum)` over `tp` for the concatenated grads of
all `shared` params. Everything else replicated needs nothing: TP peers get
the same batch and the same activations, so they compute the same grads.
Nothing re-syncs those replicas, which relies on kernels being deterministic
across peers (as Megatron does).

**`finish_init`.** After a meta build, broadcast everything *not* sharded
over tp, buffers included, from tp coordinate 0.

#### Comm/compute overlap (`tp_overlap.py`)

Only under sequence parallel. The plain path runs each SP collective with
nothing behind it. Overlap fuses a collective with the matmuls next to it
into one autograd node:

```
column(x_local, Ws)   all_gather(x) along the sp dim, then y_i = x_full @ W_iᵀ for every column linear
row(h, W)             y = h @ Wᵀ, then reduce_scatter(y) along the sp dim
```

Wiring (in `TP.__init__` when `overlap != "none"`): the region's pre hook
(`_fused_region_pre`) runs **all** of the region's column linears at once
and parks the outputs in `_column_out`; each column linear's `forward` is
replaced by a function that just pops its parked output; the row linear's
`forward` becomes the fused matmul + reduce-scatter.

Everything works on "rows": the activation with its SP dim moved to the
front and flattened to `(tokens, features)`, so rank `k`'s chunk is a
contiguous block.

Two modes:

- **`"backward"`**: in the column op's backward, compute
  `dX = Σ dY_i W_i`, launch its reduce-scatter **async**, compute the weight
  grads `dW_i = dY_iᵀ X_full` (which do not need it) while it flies, then
  wait. Forward is the plain collective + matmul.
- **`"collective_matmul"`**: additionally split every collective into `tp`
  **ring steps** over token chunks and pipeline them with the matmuls
  (Megatron's `tp_comm_overlap`):
  - `_gather_matmul`: at step `s` this rank holds chunk `r − s`, starts
    receiving chunk `r − s − 1` from its ring neighbour, and computes the
    matmuls on the chunk it holds while the next one arrives.
  - `_matmul_scatter`: at step `s` this rank computes its partial for chunk
    `r − 1 − s`, adds the running sum received from rank `r − 1`, and passes
    it on; the last step lands on its own chunk `r`. The send buffer is held
    until its wait so the allocator cannot hand its memory to the next
    matmul.

Matmuls run in the autocast dtype, and `x` is gathered in that dtype (half
the bytes of the plain path, which gathers the fp32 norm output and casts
after). The ops return the weight grads as ordinary autograd outputs, so
DDP's and FSDP's grad hooks fire as usual.

### 9.4 Expert parallel (`parallelism/expert_parallel.py`)

Split a MoE layer's experts across the `ep` axis and **move the tokens to
the experts** rather than the expert weights to the tokens.

Each ep rank holds `num_experts / ep` experts of every bank (`Shard(0)` on
the expert dim) and its **own batch**: ep ranks are data-parallel for
everything that is not an expert. That is why the mesh's `dp` axis includes
`ep`.

`EP.__init__` shards `bank.gate`, `bank.up`, `bank.down` on dim 0 and sets
`bank.remote = self._remote`.

**Dispatch** (`_remote(bank, x, counts)`; `x` is this rank's rows sorted by
global expert id):

```
1. table = all_gather(counts)                    (sender, expert): who sends how many to whom
2. send  = counts summed per destination rank    experts are assigned to ranks in contiguous
   recv  = table[:, my experts] summed per sender   blocks, so rows sorted by expert are
                                                    already grouped by destination
3. arrived = all_to_all(x, send, recv)           uneven; rows go to the rank holding their expert
4. reorder arrivals from (sender, expert) to expert order; run bank.local once
5. undo the reorder; all_to_all back with the splits swapped
```

The exchange is `_Exchange`, an autograd function whose backward is the
reverse all-to-all, so gradients retrace the same route. Two all-to-alls per
layer in forward and two in backward. The split sizes must be Python ints,
so each layer costs one host sync (`.tolist()`).

**Gradient bookkeeping.** The loss is a mean over all data ranks, ep ranks
included.

- *Dense params*: every rank has a copy and a grad from its own batch, so
  DDP / FSDP average over `dp` (which includes ep), as usual.
- *Expert params*: an expert lives on one ep rank, and the return trip of
  the dispatch already delivered the grads from every ep peer's tokens: a
  **sum** over the ep group. `EP.sync()` multiplies expert grads by `1/ep` to
  make it a mean. DDP / FSDP then average only over the *other copies* of
  that expert: `dp_expert` (the dp axes without ep).

The trainer wires this with `_expert_axis(dense, expert)`, a
`param → axis` function: under DDP it is `dim_of` (`dp` vs `dp_expert`);
under FSDP it is `replicate_dim_of` (`dp_replicate_ep` for dense params,
since each ep rank holds a whole copy of a dense shard; `dp_replicate` for
expert params).

`finish_init` broadcasts everything that is not an expert weight, and all
buffers, from ep coordinate 0.

Measured (notes/KIMI_MOE_H100.md): EP does not pay at this size. 56 small
experts inside one node; `dp_shard=2 x ep=2` was slightly slower than
`dp_shard=4` and used more memory, because dense params are sharded only
2-way.

### 9.5 How the axes compose

| mesh | `parallelism` | what happens |
|---|---|---|
| `dp: 8` | ddp | 8 replicas, bucketed all-reduce over `dp` |
| `dp_shard: 4` | fsdp | every param split 4 ways on dim 0 |
| `dp_replicate: 2, dp_shard: 2` | fsdp | HSDP: reduce-scatter on `dp_shard`, then all-reduce on `dp_replicate` |
| `dp_shard: 2, tp: 2` | fsdp | TP cuts features; FSDP shards what TP left on each rank along dim 0 |
| `dp: 2, tp: 2` | ddp | TP cuts; DDP replicates what TP left |
| `dp_shard: 2, ep: 2` | fsdp | EP cuts experts; dense params sharded over `dp_shard`, replicated over `dp_replicate_ep` |

`tp > 1` together with `ep > 1` is rejected by `validate`.

---

## 10. Optimizer, schedule, grad clipping

### `optimizer/builder.py`

`build_optimizer` makes two param groups. Weight decay applies only to the
`.weight` of `nn.Linear`, `nn.Embedding`, `nn.Conv1d/2d/3d` and
`GroupedLinear`. Biases, norm weights and bare `nn.Parameter`s (gates,
learned queries, log-rates such as KDA's `A_log`) are exempt. The selection
is by **module type**, not `p.dim()`: a 2-d bare parameter is not a matrix
to regularize. `OPTIMIZER_REGISTRY` has `adamw` and `sgd`; the configs use
fused AdamW.

Under FSDP the optimizer is built *after* sharding, so its params and Adam
moments are sharded: 1/N of the optimizer state per rank.

### `optimizer/scheduler.py`

- `WarmupStableDecayScheduler` (`wsd`): linear warmup over
  `warmup_ratio * total_steps` (step `W−1` already reaches scale 1.0), hold,
  then linear decay to 0 over the last `decay_ratio * total_steps`.
- `CosineAnnealingScheduler` (`cosine`): linear warmup, then cosine decay to
  `min_lr_ratio * base_lr`.
- `TrollScheduler`: a random LR every step. Not registered; a joke.

### `core/grad_helper.py: clip_grad_norm`

The global grad norm has to know how grads are laid out. A sharded grad's
norm is spread over the ranks of its mesh axes; a replicated one must be
counted once.

```
norms  = torch._foreach_norm(grads)
groups = norms grouped by shard_dims_of(param)        # e.g. (), ("dp_shard",), ("dp_shard", "tp")
for axes in sorted(groups):                           # sorted: same collective order on every rank
    squared = ||group norms||²
    for dim in axes: all_reduce(squared, dim, "sum")
    total += squared
total = sqrt(total)
clip_coef = clamp(max_norm / (total + 1e-6), max=1);  grads *= clip_coef
```

The trainer's `_grad_shard_dims(param)` is
`parallel.shard_dims + tp.shard_dims_of(param) + ep.shard_dims_of(param)`.
`max_norm=None` measures without clipping. The pre-clip norm is returned for
logging.

---

## 11. Checkpointing and resume

`checkpoint/checkpointer.py`, plus `Trainer.checkpoint` / `_resume`.

### 11.1 What is on disk

```
<checkpointing.dir or repo/checkpoints>/<run_name>/<step>/
    model.pt | model_shard{k}.pt + model_shards.json
    optimizer.pt | optimizer_shard{k}.pt + optimizer_shards.json
    scheduler.pt
    trainer.pt                 {step, config, wandb_id}
    rng_dp{c}.pt               cpu + cuda rng state, per dp coordinate
    dataloader_dp{c}.pt        MixtureLoader.state_dict(), per dp coordinate
    prefetcher_dp{c}.pt        the one batch the prefetcher held
    .done{rank}of{world}       per-rank completion markers
```

The directory name is the number of **completed** steps, so resume returns
`state["step"]` with no `+1` anywhere. The run directory is not timestamped
(only the wandb name is): it must be the same on every rank and every
relaunch or `resume: "auto"` could not find it.

### 11.2 Who writes what

| file | writer |
|---|---|
| model / optimizer shards | `_writes_model_state()`: the rank at coordinate 0 on every axis the state is **replicated** over. DDP: rank 0 alone. FSDP, TP, EP each add their shard axis |
| scheduler, trainer | rank 0 |
| rng, dataloader, prefetcher | one per dp coordinate; with TP only the `tp == 0` peer writes |

Before saving, rank 0 prunes to `keep_last_n` complete checkpoints (prune
first: if this save dies, `keep_last_n` complete ones are still there).

### 11.3 Shard layouts

`ShardLayout.of(model, parallel, mesh, tp, ep)` describes how *this run*
lays params out:

- `counts`: shards along each shard axis (TP, then EP, then FSDP: TP and EP
  cut first);
- `coords`: this rank's cell; `index` is its row-major position, i.e. which
  shard file it writes;
- `splits[id(param)]`: the `(axis index, tensor dim)` cuts applied to the
  param, outermost first;
- `shapes[id(param)]`: its unsharded shape (from the outermost strategy that
  cut it, which saw it whole).

The rank at cell `k` writes its **local `state_dict` as is**. Cell 0 also
writes `*_shards.json`: the grid, each tensor's unsharded shape and its
splits. A tensor an axis does not cut is replicated along that axis and read
from coordinate 0.

### 11.4 Resharding on load

Every saved piece owns a **box** of the full tensor. `_box(shape, splits,
counts, coords)` applies `padded_chunk` per cut to get, per tensor dim,
`(first index, real entries, stored length)`.

`_read` builds one tensor of the *current* layout from its saved pieces: for
every saved cell, intersect the saved box with the box this rank wants, and
copy the overlap. So a checkpoint saved on **any** grid (including
unsharded) loads onto **any** current layout: FSDP ↔ HSDP ↔ DDP, a different
GPU count, with or without TP.

Files are opened with `torch.load(..., mmap=True)`, so a rank reads only the
byte ranges it owns and no rank materializes the full model. `_read` always
returns an owning copy, because handing an mmap view to `load_state_dict`'s
deepcopy would drag the whole backing storage along.

**Optimizer state follows its param.** A state tensor shaped like the param
(Adam's `exp_avg`, `exp_avg_sq`) is a "follower" and is sharded the same
way; anything else (the `step` counter) is replicated. On load, a tensor
saved whole is treated as following its param if it has the param's full
shape.

`_legacy_meta` upgrades the older single-axis metadata format.

### 11.5 Completion without a barrier

`commit(step)` writes this rank's `.done{rank}of{world}` marker after its
last file. A step is complete once markers for every rank of some world
size exist (`is_complete`); `latest()` ignores incomplete directories. No
collective fences the save. When re-saving over an existing step directory,
a rank first removes its own marker so the directory cannot read as complete
while its files are being rewritten.

### 11.6 Async save

With `async_save` (the default in the trainer), `save_*` snapshot state to
CPU (`_snapshot`: a detached copy training cannot mutate) and return.
`commit()` hands the queued snapshots to a background thread that writes the
files and then the marker, while training continues. At most one save is in
flight: `commit()` and `wait()` block on the previous one and re-raise
anything it failed with. The thread is not a daemon, so a save that has
started finishes even if training dies. The trainer calls `wait()` before
exiting.

### 11.7 Resume (`Trainer._resume`)

`checkpointing.resume` is `null`, `"auto"` (latest complete step) or an
explicit step (an error if it is not complete).

1. Load `trainer.pt`. If the mesh changed, `resume_dataloader` must be
   `false`: model and optimizer state reshard onto any mesh, but the
   per-dp-rank dataloader files do not.
2. `load_model` with the current layout.
3. `_resume_optimizer`: per-param state comes from the checkpoint,
   **hyperparameters from this config** (`load_state_dict` would otherwise
   bring back the old ones). If `resume_scheduler` is true the LR and
   scheduler state come from the checkpoint. If false, this config's
   schedule is fast-forwarded to `step` and its LR applied immediately, so
   the first resumed step does not use the step-0 LR.
4. Dataloader: restore the prefetcher's held batch and the loader state, or
   (with `resume_dataloader: false`) `reseed(seed + step)` for a fresh
   shuffle rather than a replay of the run's own order.
5. RNG: each dp rank restores its own stream; a rank with no file (the mesh
   grew) keeps its seed-derived stream.
6. `resume_wandb_id` so wandb reattaches with `resume="must"`.

Verified in the notes: a resume from step 30 matches the uninterrupted run's
losses to 4 decimals through a mix-schedule switch.

---

## 12. The VLM (branch `claude/experimental-arch`)

**The VLM model is not on `main`.** `main` has only the data side: caption
rows in `dataloader/sources.py`, the `vision` extra (`pillow`) in
`pyproject.toml`, and the trainer hook that passes
`(model.image_size, model.num_image_tokens)` to the loader when the model
has those attributes. The model itself is on
`origin/claude/experimental-arch` (7 commits ahead of main, alongside a
masked-diffusion objective and a looped transformer). To read it without
switching branches:

```bash
git show origin/claude/experimental-arch:torchure/models/vlm/vlm.py
git show origin/claude/experimental-arch:torchure/models/vlm/siglip.py
git show origin/claude/experimental-arch:notes/EXPERIMENTAL_ARCHS.md
```

### 12.1 Architecture

LLaVA-style: vision tower → MLP projector → image features replace
placeholder tokens in the text embeddings → the Qwen3 decoder runs as usual.

```
pixel_values (N, 3, 224, 224) uint8
   │  SiglipVisionTower
   ▼
(N, 196, 768) patch features
   │  Projector (2x2 pixel shuffle → fc1 → GELU → fc2)
   ▼
(N, 49, 1024) image tokens
   │  masked_scatter into the <|image_pad|> positions of token_emb(ids)
   ▼
Qwen3 decoder (1-d RoPE over the whole sequence) → hidden → fused CE on caption + eos
```

**`SiglipVisionTower`** (`models/vlm/siglip.py`). A plain pre-norm ViT with
no CLS token; defaults are siglip-base-patch16-224 (12 layers, width 768, 12
heads, MLP 3072).

- `patch_embedding`: `Conv2d(3, 768, 16, stride=16)`, giving a 14 x 14 grid
  of 196 patches.
- Learned `position_embedding` (`nn.Embedding(196, 768)`), added to every
  patch.
- Blocks: `LayerNorm → SiglipAttention → residual`, `LayerNorm → MLP
  (GELU tanh) → residual`. Attention is plain bidirectional SDPA with
  biases.
- `post_layernorm` at the end.
- **uint8 handling**: if `pixel_values` is uint8 (what the loader sends) it
  is cast and normalized as `x / 127.5 − 1`, i.e. SigLIP's mean = std = 0.5.
- Module names follow HF's `SiglipVisionModel` (`patch_embedding`,
  `layer_norm1`, `self_attn.q_proj`, `mlp.fc1`, `post_layernorm`), so loading
  pretrained weights would be a key rename rather than a re-layout. No
  pretrained loading exists yet.

**`Projector`** (`models/vlm/vlm.py`). With `pixel_shuffle = r > 1`, each
`r x r` patch neighbourhood is folded into one token (idefics3 / SmolVLM
style):

```python
x.view(N, g, r, g, r, D).permute(0, 1, 3, 2, 4, 5).reshape(N, g*g, r*r*D)   # g = grid // r
```

With `r = 2`: 196 patches of 768 → 49 tokens of 3072, cutting the image's
share of the sequence by 4. Then `fc1 (3072 → 1024) → GELU → fc2 (1024 →
1024)`.

**`Qwen3VL(Qwen3)`**. A subclass adding `vision`, `projector`, and:

- `num_image_tokens = (grid // pixel_shuffle)²` (49) and
  `image_size = grid * patch stride` (224). The dataloader reads these two
  off the model to build caption rows.
- `forward(x, pixel_values=None, **kwargs)`: with no pixels it is plain
  Qwen3. Otherwise

  ```python
  embeds   = self.token_emb(x)
  features = self.projector(self.vision(pixel_values))      # (N, 49, 1024)
  slots    = (x == self.image_token_id).unsqueeze(-1)
  embeds   = embeds.masked_scatter(slots, features.to(embeds.dtype))
  return super().forward(x, inputs_embeds=embeds, **kwargs)
  ```

  `masked_scatter` fills the placeholder positions in order, so image `i`'s
  49 features land in row `i`'s 49 `<|image_pad|>` slots. `x` is still
  passed so positions are computed from the ids.
- `init_weights` also initializes the tower.
- `tp_plan` raises `NotImplementedError`.

### 12.2 What the branch changes outside the VLM files

- **`Qwen3.forward` gains `inputs_embeds`**: if given, it replaces
  `token_emb(x)`.
- **Objectives pass extra batch keys to the model**:
  `model_inputs(batch)` returns everything except `input_ids` and `labels`,
  so `pixel_values` reaches `forward` as a keyword argument.
- `MODEL_REGISTRY` gains `qwen3_vl` (and `looped_qwen3`).
- The fused loss gains optional per-token `weights` (for masked diffusion,
  not the VLM).

### 12.3 Data path end to end

1. Config (`qwen3_vl_coco.json`): `data.source` is a dict with
   `image_field: "image"`, `text_field: "caption"`, `window_blocks: 16`;
   `seq_len: 128`; `workers: 8`; `image_token_id: 151655`
   (Qwen's own `<|image_pad|>`).
2. Trainer: `vision=(model.image_size, model.num_image_tokens)` goes to
   `build_dataloader`.
3. `ParquetSource` with `images=True`: one row group per block; `fetch`
   returns `input_ids`, `labels`, `pixel_values` (section 5.3). Row layout:

   ```
   <|vision_start|> <|image_pad|> x 49 <|vision_end|> caption <|endoftext|> <pad>...
   labels:  -100 over the 51-token image prefix and the padding
   ```

4. `MixtureLoader` stacks uint8 pixels into a pinned tensor;
   `CUDAPrefetcher` copies it to the GPU (4x smaller than float32).
5. `ARObjective` → `Qwen3VL.forward(ids, pixel_values=...,
   return_final_hidden=True)` → fused CE. Only caption + eos positions are
   scored.

### 12.4 Parallelism and limits

- **DDP** works and is what was run on GPUs.
- **FSDP**: only the decoder blocks are units, so the vision tower and
  projector sit in the **root unit**: gathered for the whole step and not
  compiled. Fine at siglip-base (86M); a larger tower would want its blocks
  as units. Tested only on gloo/CPU.
- **TP: not supported.** TP's embedding hooks shard the sequence before the
  image features are spliced in.
- One image per row, padded, no packing: about 45% of each 128-token row is
  padding at COCO caption lengths.
- No pretrained weights, so nothing yet shows the model uses the image
  beyond the CPU test (changing one image moves only its own row, and only
  from the image span on; grads reach every vision / projector param).

Measured on 4x H100 NVL, DDP (EXPERIMENTAL_ARCHS.md): 664M params; micro 64
gives 172k tok/s (~1,350 images/s); micro 128 gives 216k tok/s (~1,690
images/s). Token counts include padding and image tokens. With the loader on
threads instead of `workers: 8`, training was 15-30% slower from GIL
contention.

---

## 13. Inference and profiling

`inference/common.py: load_model_and_tokenizer` builds the model, and calls
`checkpointer.load_model(model, step)` with the default **unsharded** layout,
so a checkpoint saved sharded by FSDP / TP / EP is reassembled into full
weights by the same resharding load. Single process, no mesh.

- `inference/export.py`: write one plain `state_dict` file. CPU; needs host
  memory for the full weights.
- `inference/eval_ppl.py`: mean loss and perplexity over freshly streamed
  batches of the *training* set with the seed bumped by 1. There is no
  held-out split, so it is a progress signal, not a benchmark. Uses the
  unfused loss.
- `inference/chat.py`: a text-continuation REPL with temperature and top-k.
  A base LM with no chat template. **No KV cache**: every new token
  recomputes the full prefix.

`profiling/overlap.py` measures how much of the grad all-reduce hides behind
backward. It runs warmup steps, profiles a few `train_step`s with
`torch.profiler` (CPU + CUDA), exports a Chrome trace, and analyzes it: GPU
kernels are split into comm (`ncclDevKernel_*`) and compute; the *union* of
each is taken (kernels on different streams overlap in time, so summing
would double-count); then

```
hidden  = comm ∩ compute
exposed = comm − hidden        ← the number to optimize
```

It also ranks the largest NCCL kernels by exposed time. The tied-embedding
grad is ready last, so its reduce is always exposed: the "structural tail".
`--analyze trace.json` re-analyzes an existing trace with no GPUs.

---

## 14. Tests and CI

Each test is a standalone script (`uv run tests/<name>.py`), most on CPU
with the gloo backend and several ranks spawned locally.

| test | checks |
|---|---|
| `collective.py` | every collective, incl. `all_gather(reduce_scatter(x)) == all_reduce(x)` |
| `placement.py`, `dtensor.py` | placement math, the redistribute table |
| `data_parallel.py` | DDP grad parity and accumulation parity vs single process |
| `fsdp2.py` | FSDP parity and memory invariants |
| `tensor_parallel.py` | TP equivalence across ddp/fsdp x vocab/sp/overlap, meta init, checkpoint |
| `expert_parallel.py` | dispatch, grads, meta build, checkpoint |
| `missing_grads.py` | grad sync when some params get no grad, DDP + FSDP |
| `grad_helper.py` | sharded clip vs `torch.nn.utils.clip_grad_norm_` |
| `checkpointer.py`, `sharded_checkpoint.py` | roundtrip; resharding between grids |
| `mixture_loader.py` | schedule, rank split, resume, HTTP range reads |
| `data_sources.py` | JSONL byte ranges, zst/gz, parquet, captions, resume, HTTP |
| `dataloader_packing.py` | the legacy packers |
| `doc_mask.py` | a packed row equals each doc alone (Qwen3 + OLMo 3 windows) |
| `models.py` | model contracts (OLMo 3, Kimi), incl. `_kda_scan` vs `_kda_reference` |
| `fused_linear_ce.py` | fused loss vs `F.cross_entropy`; **needs CUDA**, not in CI |
| `bench_dataloader.py` | loader throughput benchmark |

`.github/workflows/ci.yml` runs `ruff check` and the CPU suites, installing
CPU torch from `pyproject.toml` (not `uv.lock`) to avoid multi-GB CUDA
wheels.

---

## 15. Loose ends found while reading

Small things that do not match, in case they are worth cleaning up. None
affect correctness as far as I can tell.

- `torchure/dataloader/mixing.py` is an empty, untracked file. `scripts/gmn.py`
  (a node-provisioning client) is also untracked.
- `dataloader/packers.py` is unused by the loader; only
  `tests/dataloader_packing.py` imports it. The README's layout line still
  lists "packing" under `dataloader/`.
- README's layout says `train/` holds the overlap profiler; it moved to
  `profiling/` in `b4d2a62`.
- `notes/KIMI_MOE_H100.md` says `_kda_chunk` is compiled and
  `KDA_CHUNK = 128`. The code compiles `_kda_intra` and has
  `KDA_CHUNK = 64` with `KDA_SUB = 16`.
- `KimiK31.tp_plan`'s docstring says the config must turn sequence parallel
  off explicitly; the trainer already does it automatically when a plan has
  no regions.
- Stale comments from the old HF loader: `Checkpointer.load_dataloader`
  ("weights_only rejects the hf dataset state"), `CUDAPrefetcher` ("needs
  pin_memory=True on the dataloader"; the loader now pins by itself), and
  `profiling/overlap.py`'s last line ("dataloader workers re-import
  `__main__`").
- `AXIS_ORDER` includes `pp` and `cp`, which `validate` rejects above
  size 1.
- The VLM depends on branch-only changes (section 12), so `main`'s
  `image_field` path cannot be exercised end to end with a model on `main`;
  `open_source` asserts "the model has no vision tower".
