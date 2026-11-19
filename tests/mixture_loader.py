"""
mixture loader (torchure/dataloader/mixture.py) on tiny synthetic token
sources. cpu only, no network: the http path runs against a local
range-capable server.

every token is unique across sources (source k's tokens are k * 10**8 +
position), so a row's first token says exactly which row it is.

    uv run tests/mixture_loader.py
"""

import functools
import http.server
import io
import json
import os
import tempfile
import threading

import numpy as np
import torch

from torchure.config import ConfigError, load_config
from torchure.dataloader.mixture import MixtureLoader, SourceStream, allocate, normalize
from torchure.dataloader.shards import TokenSource, write_index

SEQ_LEN = 16
TOKENIZER = "test/tokenizer"
EOS = 7
# 8 rows of 16 uint32 tokens per block
BLOCK_MB = 8 * SEQ_LEN * 4 / 2**20
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_source(path: str, key: int, file_tokens: list[int]) -> None:
    os.makedirs(path)
    files, pos = [], 0
    for i, n in enumerate(file_tokens):
        name = f"{i:06d}.bin"
        (key * 10**8 + np.arange(pos, pos + n, dtype=np.uint32)).tofile(os.path.join(path, name))
        files.append({"path": name, "num_tokens": n})
        pos += n
    write_index(path, dtype="uint32", tokenizer=TOKENIZER, eos_id=EOS, files=files)


def write_mix(path: str, sources: dict[str, str], body: str) -> str:
    with open(path, "w") as f:
        f.write(f"SOURCES = {sources!r}\n\n{body}")
    return path


SCHEDULE = """
def weights(step, total_steps):
    if step < total_steps // 2:
        return {"web": 0.7, "code": 0.25, "tiny": 0.05}
    return {"web": 0.2, "code": 0.5, "tiny": 0.3}
"""


def data_cfg(mix: str, **kw) -> dict:
    cfg = {"mix": mix, "seq_len": SEQ_LEN, "global_batch_size": 16, "micro_batch_size": 2,
           "seed": 3, "block_mb": BLOCK_MB, "window_blocks": 2, "fetch_threads": 4}
    return cfg | kw


def make_sources(tmp: str) -> dict[str, str]:
    # web: two files with ragged tails; tiny: 40 rows, wraps epochs fast
    make_source(f"{tmp}/web", 1, [SEQ_LEN * 300 + 5, SEQ_LEN * 213 + 11])
    make_source(f"{tmp}/code", 2, [SEQ_LEN * 257])
    make_source(f"{tmp}/tiny", 3, [SEQ_LEN * 40 + 3])
    return {"web": f"{tmp}/web", "code": f"{tmp}/code", "tiny": f"{tmp}/tiny"}


def run(loader: MixtureLoader, n: int) -> list[torch.Tensor]:
    return [next(loader)["input_ids"] for _ in range(n)]


def roundtrip(state: dict) -> dict:
    """through torch.save, as the checkpointer stores it."""
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=False)


def test_allocate() -> None:
    rng = np.random.default_rng(0)
    credit = np.zeros(4)
    given = np.zeros(4)
    target = np.zeros(4)
    for _ in range(2000):
        w = rng.dirichlet(np.ones(4) * 0.3)
        w[3] = 1e-3  # a sliver, far below one row per step
        w /= w.sum()
        counts, credit = allocate(credit, w, 64)
        assert counts.sum() == 64 and (counts >= 0).all()
        given += counts
        target += w * 64
        assert np.all(np.abs(given - target) < 1 + 1e-9), (given, target)
    assert given[3] > 0, "a small weight must still get rows over time"
    try:
        normalize({"nope": 1.0}, ["a"], 0)
    except AssertionError:
        pass
    else:
        raise AssertionError("unknown source name must be rejected")
    print("ok  allocate: exact batch, carried rounding, no starvation")


def test_streams_disjoint(tmp: str) -> None:
    src = TokenSource(f"{tmp}/web", SEQ_LEN, BLOCK_MB, min_blocks=4)
    assert src.num_rows == 300 + 213
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(2)
    seen = []
    for rank in range(4):
        stream = SourceStream(src, "web", 3, rank, 4, 2, pool)
        while stream.epoch == 0:
            seen.append(int(stream.take()[0]))
    pool.shutdown()
    assert len(seen) == len(set(seen)) == src.num_rows, "epoch 0 must cover each row exactly once"
    # rows are whole seq_len slices of the right file (no straddling the tail)
    starts = {1 * 10**8 + r * SEQ_LEN for r in range(300)}
    starts |= {1 * 10**8 + (300 * SEQ_LEN + 5) + r * SEQ_LEN for r in range(213)}
    assert set(seen) == starts
    print("ok  streams: 4 ranks cover one epoch exactly once, disjointly")


def test_plan_and_ranks(mix: str, dp: int = 4, steps: int = 12) -> None:
    cfg = data_cfg(mix)
    loaders = [MixtureLoader(cfg, r, dp, total_steps=steps) for r in range(dp)]
    per_step = cfg["global_batch_size"] // (cfg["micro_batch_size"] * dp)
    credit = np.zeros(3)
    names = loaders[0].names
    for step in range(steps):
        rows = torch.cat([b for ld in loaders for b in run(ld, per_step)])
        ids = rows[:, 0].numpy() // 10**8
        got = np.array([(ids == k).sum() for k in (1, 2, 3)])
        w = normalize(loaders[0].mix.weights(step, steps), names, step)
        want, credit = allocate(credit, w, cfg["global_batch_size"])
        assert (got == want).all(), (step, got, want)
        assert len(set(rows[:, 0].tolist())) == len(rows), "no row twice within a step"
    for ld in loaders:
        ld.close()
    print("ok  plan: every step's global batch matches the schedule exactly across 4 ranks")


def test_resume(mix: str, total: int = 90) -> None:
    cfg = data_cfg(mix)
    ref_loader = MixtureLoader(cfg, 1, 2, total_steps=20)
    ref = run(ref_loader, total)
    ref_loader.close()
    per_step = cfg["global_batch_size"] // (cfg["micro_batch_size"] * 2)
    # mid-step and on a step boundary; past tiny's epoch wrap
    for cut in (1, per_step, 3 * per_step + 1, 61):
        a = MixtureLoader(cfg, 1, 2, total_steps=20)
        head = run(a, cut)
        state = roundtrip(a.state_dict())
        a.close()
        b = MixtureLoader(cfg, 1, 2, total_steps=20)
        b.load_state_dict(state)
        tail = run(b, total - cut)
        b.close()
        for i, (x, y) in enumerate(zip(head + tail, ref, strict=True)):
            assert torch.equal(x, y), f"cut {cut}: batch {i} differs after resume"
    print("ok  resume: identical batches resuming mid-step, at a boundary, after an epoch wrap")


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """serves a directory with Range support; fails each path's first GET with 503."""
    failed: set = set()

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path not in RangeHandler.failed:
            RangeHandler.failed.add(self.path)
            self.send_error(503)
            return
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, "rb") as f:
            data = f.read()
        rng = self.headers.get("Range")
        if rng:
            lo, hi = rng.removeprefix("bytes=").split("-")
            lo, hi = int(lo), (int(hi) if hi else len(data) - 1)
            body = data[lo:hi + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_http(tmp: str, local_mix: str) -> None:
    import torchure.dataloader.shards as shards
    shards.time.sleep = lambda s: None  # no backoff waits in the test
    handler = functools.partial(RangeHandler, directory=tmp)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        http_mix = write_mix(
            f"{tmp}/http_mix.py",
            {"web": f"{base}/web", "code": f"{base}/code", "tiny": f"{base}/tiny"},
            SCHEDULE,
        )
        a = MixtureLoader(data_cfg(local_mix), 0, 2, total_steps=20)
        b = MixtureLoader(data_cfg(http_mix), 0, 2, total_steps=20)
        for x, y in zip(run(a, 40), run(b, 40), strict=True):
            assert torch.equal(x, y)
        a.close()
        b.close()
    finally:
        server.shutdown()
    assert RangeHandler.failed, "the server never saw a request"
    print("ok  http: range reads (retrying a 503 on every file) match local reads")


def test_config(mix: str) -> None:
    base = os.path.join(ROOT, "configs/qwen3_dense_climbmix.json")
    with tempfile.TemporaryDirectory() as tmp:
        both = os.path.join(tmp, "both.json")
        with open(both, "w") as f:
            json.dump({"base": base, "data": {"mix": mix}}, f)
        try:
            load_config(both)
        except ConfigError:
            pass
        else:
            raise AssertionError("name + mix together must be rejected")
        only = os.path.join(tmp, "only.json")
        with open(only, "w") as f:
            json.dump({"base": base, "data": {"mix": mix, "name": None}}, f)
        cfg = load_config(only)
        assert cfg["data"]["mix"] == mix and cfg["data"]["block_mb"] == 16
    print("ok  config: exactly one of data.name / data.mix")


def test_tokenizer_mismatch(mix: str) -> None:
    try:
        MixtureLoader(data_cfg(mix), 0, 1, total_steps=10, tokenizer="other/tokenizer")
    except AssertionError as e:
        assert "tokenized with" in str(e)
    else:
        raise AssertionError("a source tokenized with another tokenizer must be rejected")
    print("ok  sources: tokenizer mismatch rejected")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sources = make_sources(tmp)
        mix = write_mix(f"{tmp}/mix.py", sources, SCHEDULE)
        test_allocate()
        test_streams_disjoint(tmp)
        test_plan_and_ranks(mix)
        test_resume(mix)
        test_tokenizer_mismatch(mix)
        test_config(mix)
        test_http(tmp, mix)
    print("all mixture loader tests passed")


if __name__ == "__main__":
    main()
