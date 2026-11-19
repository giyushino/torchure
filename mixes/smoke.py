"""
example mix (see notes/DATA_MIXING.md): mostly climbmix, annealing into
fineweb-edu over the last 10% of training, which is also wsd's decay phase.

the sources are local slices made by tokenize_shards.py:

    uv run -m torchure.dataloader.tokenize_shards --dataset OptimalScale/ClimbMix \
        --tokenizer Qwen/Qwen3-0.6B --eos-token "<|endoftext|>" \
        --max-tokens 3e7 --shard-tokens 1e7 --out data/smoke_climbmix
    uv run -m torchure.dataloader.tokenize_shards --dataset HuggingFaceFW/fineweb-edu \
        --config sample-10BT --tokenizer Qwen/Qwen3-0.6B --eos-token "<|endoftext|>" \
        --max-tokens 1e7 --shard-tokens 1e7 --out data/smoke_fwedu

a location can also be an http(s) url or hf://datasets/<org>/<name>.
"""

SOURCES = {
    "climbmix": "data/smoke_climbmix",
    "fineweb_edu": "data/smoke_fwedu",
}


def weights(step: int, total_steps: int) -> dict[str, float]:
    edu = 0.1 if step < 0.9 * total_steps else 0.5
    return {"climbmix": 1 - edu, "fineweb_edu": edu}
