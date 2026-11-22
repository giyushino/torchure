"""
train config: one json file, checked and completed before anything is built.

    config = load_config("configs/qwen3_dense_climbmix.json")

- `"base": "other.json"` (path relative to the file) starts from another
  config and overrides it key by key, so a variant states only what differs.
  `mesh` is the exception: a variant's mesh replaces the base's whole.
- every key is checked against SCHEMA. a key the schema doesn't know is an
  error naming the closest known key, instead of a silently ignored typo.
- optional keys get their defaults here, in one place. the trainer indexes
  the result directly.
- `validate(config, world_size)` holds the rules that need the launch's
  rank count; the trainer calls it before building anything.

the result is a plain dict (it is saved in checkpoints and sent to wandb).
"""

import difflib
import inspect
import json
import math
import os

REQUIRED = object()   # the key must be given
FREE = object()       # a dict passed through as is (kwargs for something else)

SCHEMA = {
    "run_name": REQUIRED,
    "seed": 67,
    "model": {"name": REQUIRED, "config": FREE},
    # axis sizes; see core/mesh.py build_mesh
    "mesh": FREE,
    # "ddp" | "fsdp": how the dp axes hold the model
    "parallelism": "ddp",
    # reduce_dtype: the dtype grads are all-reduced in ("float32" = as is)
    "ddp": {"bucket_mb": 25, "reduce_dtype": "float32"},
    # reshard_after_forward false keeps the gathered weights from forward to
    # backward: half the all_gathers for one gathered copy of the model in
    # memory (parallelism/fsdp2.py)
    "fsdp": {
        "param_dtype": "bfloat16",
        "reduce_dtype": "float32",
        "prefetch": 1,
        "reshard_after_forward": True,
    },
    "tp": {
        "vocab_parallel": True,
        "sequence_parallel": True,
        "sp_dim": "seq",
        "overlap": "none",
    },
    "compile": True,
    "compile_mode": "default",
    "activation_checkpointing": False,
    # attend within documents only (flex attention, positions restart per
    # doc); documents are split at data.eos_token. supported by qwen3_dense
    # and olmo3
    "doc_mask": False,
    "log_every": 10,
    "optimizer": {
        "name": REQUIRED,
        "config": FREE,
        "max_norm": None,
        "scheduler": FREE,
    },
    "objective": {"name": REQUIRED, "config": FREE},
    "data": {
        # exactly one of: `source`, one dataset (a location string, or
        # {"path", "files", "text_field", "image_field", "window_blocks"}),
        # or `mix`, a mix script over several (dataloader/{sources,mixture}.py,
        # notes/DATA_LOADING.md)
        "source": None,
        "mix": None,
        "tokenizer": REQUIRED,
        "eos_token": REQUIRED,
        "pad_token": REQUIRED,
        "seed": 42,
        "global_batch_size": REQUIRED,
        "micro_batch_size": REQUIRED,
        "seq_len": REQUIRED,
        # bytes per range request (token shards, jsonl), blocks per shuffle
        # window, windows downloading ahead of the current one, and the
        # threads that download / tokenize / decode
        "block_mb": 16,
        "window_blocks": 4,
        "prefetch_windows": 2,
        "fetch_threads": 8,
        # > 0: that many processes fetch / tokenize / decode instead of the
        # threads, keeping the gil free for the training loop. worth it for
        # images, not for text (notes/DATA_LOADING.md), so null picks 0 for
        # text sources and mixture.IMAGE_WORKERS for image sources
        "workers": None,
    },
    "checkpointing": {
        "save_steps": REQUIRED,
        # null | "auto" | <step>
        "resume": None,
        "resume_scheduler": True,
        "resume_dataloader": True,
        "async_save": True,
        # complete checkpoints to keep; null keeps all
        "keep_last_n": None,
        # root dir for runs; null = <repo>/checkpoints
        "dir": None,
        # steps between checks for a SIGTERM; null = log_every
        "stop_check_every": None,
    },
    "wandb": {"enabled": False, "project": "torchure", "entity": None},
}


class ConfigError(ValueError):
    pass


# blocks an override replaces whole: merging two meshes key by key would
# keep axes the variant meant to drop
_WHOLE = ("mesh",)


def _merge(base: dict, override: dict) -> dict:
    """`override` on top of `base`, dict by dict; anything else replaces."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key not in _WHOLE:
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read(path: str, seen: tuple[str, ...] = ()) -> dict:
    """the json at `path` with its `base` chain resolved."""
    path = os.path.abspath(path)
    if path in seen:
        raise ConfigError(f"config `base` cycle: {' -> '.join((*seen, path))}")
    with open(path) as f:
        config = json.load(f)
    base = config.pop("base", None)
    if base is None:
        return config
    base_path = os.path.join(os.path.dirname(path), base)
    return _merge(_read(base_path, (*seen, path)), config)


def _check(config: dict, schema: dict, where: str) -> dict:
    """`config` with defaults filled, or a ConfigError naming the problem."""
    if not isinstance(config, dict):
        raise ConfigError(f"{where or 'config'} must be an object, got {type(config).__name__}")
    for key in config:
        if key not in schema:
            close = difflib.get_close_matches(key, list(schema), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise ConfigError(
                f"unknown config key {where + key!r}{hint} (known: {sorted(schema)})"
            )
    out = {}
    for key, spec in schema.items():
        if isinstance(spec, dict):
            out[key] = _check(config.get(key, {}), spec, f"{where}{key}.")
        elif key not in config:
            if spec is REQUIRED:
                raise ConfigError(f"missing required config key {where + key!r}")
            out[key] = {} if spec is FREE else spec
        else:
            if spec is FREE and not isinstance(config[key], dict):
                raise ConfigError(f"{where + key!r} must be an object")
            out[key] = config[key]
    return out


def _check_kwargs(fn, kwargs: dict, what: str) -> None:
    """every key of `kwargs` must be a parameter of `fn`."""
    params = inspect.signature(fn).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return
    names = [n for n in params if n != "self"]
    for key in kwargs:
        if key not in names:
            close = difflib.get_close_matches(key, names, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise ConfigError(f"{what} has no option {key!r}{hint} (options: {names})")
    missing = [
        n for n in names
        if params[n].default is inspect.Parameter.empty and n not in kwargs
    ]
    if missing:
        raise ConfigError(f"{what} is missing required option(s) {missing}")


def load_config(path: str) -> dict:
    config = _check(_read(path), SCHEMA, "")

    # the free-form blocks are kwargs; check them against what takes them
    from torchure.models.builder import MODEL_REGISTRY
    from torchure.objectives.builder import OBJECTIVE_REGISTRY
    from torchure.optimizer.builder import SCHEDULER_REGISTRY

    name = config["model"]["name"]
    if name not in MODEL_REGISTRY:
        raise ConfigError(f"model {name!r} is not implemented; registry: {list(MODEL_REGISTRY)}")
    _check_kwargs(MODEL_REGISTRY[name].__init__, config["model"]["config"], f"model {name!r}")

    name = config["objective"]["name"]
    if name not in OBJECTIVE_REGISTRY:
        raise ConfigError(f"unknown objective {name!r}; registry: {list(OBJECTIVE_REGISTRY)}")
    _check_kwargs(
        OBJECTIVE_REGISTRY[name].__init__, config["objective"]["config"], f"objective {name!r}"
    )
    config["objective"]["config"].setdefault("ignore_index", -100)

    scheduler = config["optimizer"]["scheduler"]
    if scheduler.get("type") not in SCHEDULER_REGISTRY:
        raise ConfigError(
            f"optimizer.scheduler.type {scheduler.get('type')!r}: "
            f"expected one of {list(SCHEDULER_REGISTRY)}"
        )
    if "total_steps" not in scheduler:
        raise ConfigError("optimizer.scheduler.total_steps is required")
    _check_kwargs(
        SCHEDULER_REGISTRY[scheduler["type"]].__init__,
        {k: v for k, v in scheduler.items() if k != "type"} | {"optimizer": None},
        f"scheduler {scheduler['type']!r}",
    )

    _check_values(config)
    return config


def _check_values(config: dict) -> None:
    def expect(ok: bool, message: str) -> None:
        if not ok:
            raise ConfigError(message)

    expect(config["parallelism"] in ("ddp", "fsdp"),
           f"parallelism {config['parallelism']!r}: expected 'ddp' or 'fsdp'")
    expect(config["tp"]["sp_dim"] in ("seq", "batch"),
           f"tp.sp_dim {config['tp']['sp_dim']!r}: expected 'seq' or 'batch'")
    ckpt = config["checkpointing"]
    expect(ckpt["resume"] in (None, "auto") or isinstance(ckpt["resume"], int),
           f"checkpointing.resume {ckpt['resume']!r}: expected null, 'auto' or a step")
    expect(ckpt["keep_last_n"] is None or ckpt["keep_last_n"] >= 1,
           f"checkpointing.keep_last_n {ckpt['keep_last_n']}: expected null or >= 1")
    expect(ckpt["stop_check_every"] is None or ckpt["stop_check_every"] >= 1,
           "checkpointing.stop_check_every: expected null or >= 1")
    data = config["data"]
    expect((data["source"] is None) != (data["mix"] is None),
           "data needs exactly one of `source` (one dataset) or `mix` (a mix script)")
    expect(data["source"] is None or isinstance(data["source"], (str, dict)),
           "data.source must be a location string or an object with `path`")
    expect(data["mix"] is None or os.path.isfile(data["mix"]),
           f"data.mix {data['mix']!r}: no such file")
    expect(data["global_batch_size"] % data["micro_batch_size"] == 0,
           f"data.global_batch_size {data['global_batch_size']} must be a multiple of "
           f"data.micro_batch_size {data['micro_batch_size']}")
    for axis, size in config["mesh"].items():
        expect(isinstance(size, int) and size >= 1, f"mesh.{axis} must be a positive integer")


def validate(config: dict, world_size: int) -> None:
    """
    the rules that depend on how many ranks were launched. everything here
    would otherwise surface as a hang or a shape error at step 0.
    """
    def expect(ok: bool, message: str) -> None:
        if not ok:
            raise ConfigError(message)

    mesh = dict(config["mesh"])
    expect(not ("dp" in mesh and ("dp_replicate" in mesh or "dp_shard" in mesh)),
           f"mesh {mesh}: `dp` is shorthand for `dp_shard`; don't mix it with the explicit axes")
    if "dp" in mesh:
        mesh["dp_shard"] = mesh.pop("dp")
    ranks = math.prod(mesh.values())
    expect(ranks == world_size,
           f"mesh {config['mesh']} needs {ranks} ranks, but {world_size} were launched")

    tp, ep = mesh.get("tp", 1), mesh.get("ep", 1)
    for axis in ("pp", "cp"):
        expect(mesh.get(axis, 1) == 1, f"mesh.{axis} > 1: {axis} is not implemented yet")
    expect(config["parallelism"] != "fsdp" or mesh.get("dp_shard", 1) > 1,
           "parallelism 'fsdp' needs mesh.dp_shard > 1; use 'ddp' for pure replication")
    expect(tp == 1 or ep == 1, "tensor parallel together with expert parallel is not supported yet")

    data = config["data"]
    data_ranks = mesh.get("dp_replicate", 1) * mesh.get("dp_shard", 1) * ep
    per_backward = data["micro_batch_size"] * data_ranks
    expect(data["global_batch_size"] % per_backward == 0,
           f"data.global_batch_size {data['global_batch_size']} must be divisible by "
           f"micro_batch_size {data['micro_batch_size']} x {data_ranks} data ranks = {per_backward}")
    if tp > 1:
        if config["tp"]["sequence_parallel"]:
            expect(data["seq_len"] % tp == 0,
                   f"sequence parallel needs data.seq_len {data['seq_len']} divisible by tp {tp}")
        if config["tp"]["sp_dim"] == "batch":
            expect(data["micro_batch_size"] % tp == 0,
                   f"tp.sp_dim 'batch' needs data.micro_batch_size divisible by tp {tp}")
    if ep > 1:
        experts = config["model"]["config"].get("num_experts")
        expect(experts is None or experts % ep == 0,
               f"model num_experts {experts} must be divisible by mesh.ep {ep}")
