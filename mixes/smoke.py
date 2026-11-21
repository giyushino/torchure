"""
example mix (see notes/DATA_MIXING.md): mostly climbmix, annealing into
fineweb-edu over the last 10% of training, which is also wsd's decay phase.

both stream straight from the hub and are tokenized while training: climbmix
is jsonl (byte-range blocks), fineweb-edu's 10BT sample is parquet
(row-group blocks). a location can also be a local directory or file, an
http(s) url to one file, or a token-shard directory made by
tokenize_shards.py.
"""

SOURCES = {
    "climbmix": "hf://datasets/OptimalScale/ClimbMix",
    "fineweb_edu": "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT",
}


def weights(step: int, total_steps: int) -> dict[str, float]:
    edu = 0.1 if step < 0.9 * total_steps else 0.5
    return {"climbmix": 1 - edu, "fineweb_edu": edu}
