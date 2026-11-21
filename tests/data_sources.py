"""
raw sources (torchure/dataloader/sources.py) on small synthetic files. cpu
only, no network: a word-level tokenizer is built in memory and the http
path runs against a local range-capable server.

every doc is unique ("f<file> d<index>" plus filler), so a doc's first two
tokens say exactly which doc it is.

    uv run tests/data_sources.py
"""

import functools
import gzip
import http.server
import io
import json
import os
import random
import tempfile
import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from compression import zstd
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers

from torchure.dataloader.mixture import MixtureLoader
from torchure.dataloader.sources import RowConfig, Vision, open_source

SPECIALS = ["<|endoftext|>", "<|fim_pad|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
BLOCK_MB = 4096 / 2**20  # the jsonl minimum


def make_tokenizer() -> Tokenizer:
    words = ["[UNK]", "x", "é"] + [f"f{i}" for i in range(10)] + [f"d{i}" for i in range(5000)]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens(SPECIALS)
    return tok


TOK = make_tokenizer()
EOS = TOK.token_to_id("<|endoftext|>")
PAD = TOK.token_to_id("<|fim_pad|>")


def docs_for(file: int, n: int, rng: random.Random) -> list[str]:
    """doc lengths from 1 to ~3000 words (the long ones span several 4KB blocks)."""
    out = []
    for i in range(n):
        filler = rng.choice([0, 1, 5, 40, 300, 3000 if rng.random() < 0.05 else 2])
        out.append(" ".join([f"f{file}", f"d{i}"] + ["x"] * filler + (["é"] if i % 7 == 0 else [])))
    return out


def write_jsonl(path: str, docs: list[str], trailing_newline: bool = True) -> bytes:
    body = "\n".join(json.dumps({"text": d, "id": i}) for i, d in enumerate(docs))
    body = (body + ("\n" if trailing_newline else "")).encode()
    with open(path, "wb") as f:
        f.write(body)
    return body


def make_text_dir(root: str) -> dict[str, list[str]]:
    """a jsonl dir, a compressed dir and a parquet dir, each with its docs in file order."""
    rng = random.Random(0)
    out = {}
    os.makedirs(f"{root}/jsonl/sub")
    docs = []
    for f, (n, nl) in enumerate([(400, True), (250, False), (3, True)]):
        d = docs_for(f, n, rng)
        write_jsonl(f"{root}/jsonl/{'sub/' if f == 1 else ''}{f}.jsonl", d, nl)
        docs.append(d)
    # list order is sorted paths: 0.jsonl, 2.jsonl, sub/1.jsonl
    out["jsonl"] = docs[0] + docs[2] + docs[1]

    os.makedirs(f"{root}/zst")
    d0, d1 = docs_for(3, 120, rng), docs_for(4, 80, rng)
    write_jsonl(f"{root}/zst/a.jsonl", d0)
    write_jsonl(f"{root}/zst/b.jsonl", d1)
    for name, fn, ext in (("a", zstd.compress, ".zst"), ("b", gzip.compress, ".gz")):
        with open(f"{root}/zst/{name}.jsonl", "rb") as f:
            data = fn(f.read())
        os.remove(f"{root}/zst/{name}.jsonl")
        with open(f"{root}/zst/{name}.jsonl{ext}", "wb") as f:
            f.write(data)
    out["zst_mixed"] = d0 + d1

    os.makedirs(f"{root}/parquet")
    pdocs = []
    for f in range(3):
        d = docs_for(5 + f, 61 + 17 * f, rng)
        table = pa.table({"id": list(range(len(d))), "text": d})
        pq.write_table(table, f"{root}/parquet/part{f}.parquet", row_group_size=7)
        pdocs += d
    out["parquet"] = pdocs
    return out


def rows_cfg(seq_len: int = 32, vision: Vision | None = None) -> RowConfig:
    return RowConfig(seq_len, TOK, EOS, PAD, -100, vision)


def all_texts(src) -> list[str]:
    return [t for b in src.blocks for t in src.records(b)["text"]]


def test_jsonl_blocks(root: str, docs: dict) -> None:
    for block_mb in (BLOCK_MB, 3 * BLOCK_MB, 16):
        src = open_source(f"{root}/jsonl", rows_cfg(), block_mb, 1)
        assert all_texts(src) == docs["jsonl"], f"block_mb {block_mb}: lines lost or doubled"
    src = open_source(f"{root}/jsonl", rows_cfg(), BLOCK_MB, 1)
    assert len(src.blocks) > 20, len(src.blocks)
    # blocks inside one long line own nothing
    assert any(not src.records(b)["text"] for b in src.blocks)
    # min_blocks shrinks blocks
    small = open_source(f"{root}/jsonl", rows_cfg(), 16, 30)
    assert len(small.blocks) >= 30 and all_texts(small) == docs["jsonl"]
    # files glob
    sub = open_source({"path": f"{root}/jsonl", "files": "sub/*"}, rows_cfg(), BLOCK_MB, 1)
    assert all_texts(sub) == docs["jsonl"][403:]
    # a single file location
    one = open_source(f"{root}/jsonl/2.jsonl", rows_cfg(), BLOCK_MB, 1)
    assert all_texts(one) == docs["jsonl"][400:403]
    print("ok  jsonl: byte-range blocks own each line exactly once, any block size")


def test_compressed(root: str, docs: dict) -> None:
    try:
        open_source(f"{root}/zst", rows_cfg(), BLOCK_MB, 1)
    except AssertionError as e:
        assert "mixed file kinds" in str(e)
    else:
        raise AssertionError("a .zst + .gz directory must be rejected")
    a = open_source({"path": f"{root}/zst", "files": "*.zst"}, rows_cfg(), BLOCK_MB, 1)
    b = open_source({"path": f"{root}/zst", "files": "*.gz"}, rows_cfg(), BLOCK_MB, 1)
    assert len(a.blocks) == 1 and a.window_blocks == 1
    assert all_texts(a) + all_texts(b) == docs["zst_mixed"]
    print("ok  compressed: zst and gz files are whole-file blocks")


def test_parquet(root: str, docs: dict) -> None:
    src = open_source(f"{root}/parquet", rows_cfg(), 16, 1)
    assert len(src.blocks) == sum(-(-n // 7) for n in (61, 78, 95))
    reads = []
    real = src.reader.read
    src.reader.read = lambda *a, **k: reads.append(a) or real(*a, **k)
    assert all_texts(src) == docs["parquet"]
    assert len(reads) == len(src.blocks), "one range read per row group"
    print("ok  parquet: row-group blocks, one range read each")


def test_pack(root: str) -> None:
    """rows are the block's docs + eos, concatenated, cut at seq_len, tail dropped."""
    src = open_source(f"{root}/jsonl", rows_cfg(seq_len=32), BLOCK_MB, 1)
    for b in src.blocks[:30]:
        texts = src.records(b)["text"]
        flat = [t for enc in TOK.encode_batch(texts) for t in enc.ids + [EOS]]
        n = len(flat) // 32
        got = src.fetch(b)["input_ids"]
        assert got.shape == (n, 32) and got.dtype == np.uint16
        assert got.ravel().tolist() == flat[:n * 32]
    print("ok  pack: rows match tokenize + eos + cut")


def make_caption_parquet(root: str, n: int = 23) -> list[tuple[str, np.ndarray]]:
    os.makedirs(f"{root}/coco")
    rng = np.random.default_rng(0)
    caps, imgs, ref = [], [], []
    for i in range(n):
        h, w = (48, 64) if i % 2 else (40, 33)
        arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG" if i % 3 else "PNG")
        caption = " ".join(["d" + str(i)] + ["x"] * (i % 9))
        caps.append(caption)
        imgs.append({"bytes": buf.getvalue(), "path": None})
        ref.append(buf.getvalue())
    table = pa.table({"image": imgs, "caption": caps})
    pq.write_table(table, f"{root}/coco/train.parquet", row_group_size=5)
    return list(zip(caps, ref, strict=True))


def test_captions(root: str) -> None:
    data = make_caption_parquet(root)
    vision = Vision(image_size=16, num_image_tokens=4)
    seq_len = 12
    spec = {"path": f"{root}/coco", "text_field": "caption", "image_field": "image"}
    src = open_source(spec, rows_cfg(seq_len, vision), 16, 1)
    assert src.keys == ("input_ids", "labels", "pixel_values")
    out = [src.fetch(b) for b in src.blocks]
    ids = np.concatenate([o["input_ids"] for o in out])
    labels = np.concatenate([o["labels"] for o in out])
    pixels = np.concatenate([o["pixel_values"] for o in out])
    prefix = [TOK.token_to_id("<|vision_start|>")] + [TOK.token_to_id("<|image_pad|>")] * 4 \
        + [TOK.token_to_id("<|vision_end|>")]
    for i, (caption, raw) in enumerate(data):
        seq = (prefix + TOK.encode(caption, add_special_tokens=False).ids + [EOS])[:seq_len]
        want = seq + [PAD] * (seq_len - len(seq))
        assert ids[i].tolist() == want, (i, ids[i], want)
        want_labels = [-100] * 6 + seq[6:] + [-100] * (seq_len - len(seq))
        assert labels[i].tolist() == want_labels
        image = Image.open(io.BytesIO(raw))
        image.draft("RGB", (16, 16))
        image = image.convert("RGB").resize((16, 16), Image.Resampling.BICUBIC)
        assert np.array_equal(pixels[i], np.asarray(image).transpose(2, 0, 1))
    try:
        open_source(spec, rows_cfg(seq_len, None), 16, 1)
    except AssertionError as e:
        assert "vision tower" in str(e)
    else:
        raise AssertionError("image_field without a vision model must be rejected")
    print("ok  captions: prefix + caption + eos + pad, labels masked, pixels decoded")

    cfg = loader_cfg(spec, seq_len=seq_len, global_batch_size=4, micro_batch_size=2)
    loader = MixtureLoader(cfg, 0, 1, 10, tokenizer=TOK, vision=vision, pin_memory=False)
    batch = next(loader)
    loader.close()
    assert batch["pixel_values"].dtype == torch.uint8
    assert batch["pixel_values"].shape == (2, 3, 16, 16)
    assert batch["labels"].dtype == torch.int64 and (batch["labels"][:, :6] == -100).all()
    print("ok  captions: loader batches carry uint8 pixel_values and masked labels")


def loader_cfg(source=None, mix=None, **kw) -> dict:
    cfg = {"source": source, "mix": mix, "seq_len": 32, "global_batch_size": 8,
           "micro_batch_size": 2, "seed": 3, "block_mb": BLOCK_MB, "window_blocks": 2,
           "prefetch_windows": 2, "fetch_threads": 4, "eos_token": "<|endoftext|>",
           "pad_token": "<|fim_pad|>"}
    return cfg | kw


def run(loader, n: int) -> list[dict]:
    return [next(loader) for _ in range(n)]


def test_mix_resume(root: str) -> None:
    mix = f"{root}/mix.py"
    with open(mix, "w") as f:
        f.write(f"SOURCES = {{'web': {f'{root}/jsonl'!r}, 'pq': {f'{root}/parquet'!r}}}\n\n"
                "def weights(step, total_steps):\n"
                "    return {'web': 0.75, 'pq': 0.25} if step < 5 else {'web': 0.25, 'pq': 0.75}\n")
    cfg = loader_cfg(mix=mix)
    ref_loader = MixtureLoader(cfg, 1, 2, 20, tokenizer=TOK, pin_memory=False)
    ref = run(ref_loader, 120)
    ref_loader.close()
    for cut in (1, 2, 37, 101):
        a = MixtureLoader(cfg, 1, 2, 20, tokenizer=TOK, pin_memory=False)
        head = run(a, cut)
        state = a.state_dict()
        a.close()
        b = MixtureLoader(cfg, 1, 2, 20, tokenizer=TOK, pin_memory=False)
        b.load_state_dict(state)
        tail = run(b, 120 - cut)
        b.close()
        for i, (x, y) in enumerate(zip(head + tail, ref, strict=True)):
            assert torch.equal(x["input_ids"], y["input_ids"]), f"cut {cut}: batch {i}"
    # the parquet source wrapped epochs within those 120 micro batches
    print("ok  mix: raw jsonl + parquet sources resume to identical batches")


def test_workers(root: str) -> None:
    """worker processes yield exactly the thread path's batches, across a resume too."""
    spec = {"path": f"{root}/coco", "text_field": "caption", "image_field": "image"}
    vision = Vision(image_size=16, num_image_tokens=4)
    for cfg, kw in ((loader_cfg(source=f"{root}/jsonl"), {}),
                    (loader_cfg(source=spec, seq_len=12, global_batch_size=4,
                                micro_batch_size=2), {"vision": vision})):
        threads = MixtureLoader(cfg, 0, 2, 10, tokenizer=TOK, pin_memory=False, **kw)
        ref = run(threads, 30)
        threads.close()
        procs = MixtureLoader(cfg | {"workers": 2}, 0, 2, 10, tokenizer=TOK, pin_memory=False, **kw)
        head = run(procs, 11)
        state = procs.state_dict()
        workers = list(procs.workers._processes.values())
        procs.close()
        # killed; is_alive() can lag while the executor's own thread reaps them
        deadline = time.monotonic() + 10
        while any(p.is_alive() for p in workers) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not any(p.is_alive() for p in workers), "close() must stop the workers"
        resumed = MixtureLoader(cfg | {"workers": 2}, 0, 2, 10, tokenizer=TOK, pin_memory=False,
                                **kw)
        resumed.load_state_dict(state)
        tail = run(resumed, 19)
        resumed.close()
        for i, (x, y) in enumerate(zip(head + tail, ref, strict=True)):
            assert x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x), i
    print("ok  workers: process workers match threads batch for batch, through a resume")


def test_ranks_disjoint(root: str) -> None:
    """one epoch over 4 ranks sees every doc start (f, d) once, from rows."""
    cfg = loader_cfg(source=f"{root}/parquet", seq_len=8, window_blocks=3)
    seen = []
    for rank in range(4):
        loader = MixtureLoader(cfg, rank, 4, 10, tokenizer=TOK, pin_memory=False)
        stream = loader.streams["data"]
        while stream.epoch == 0:
            seen.append(tuple(stream.take()["input_ids"].tolist()))
        loader.close()
    src = open_source(f"{root}/parquet", rows_cfg(8), 16, 4)
    want = [tuple(r) for b in src.blocks for r in src.fetch(b)["input_ids"].tolist()]
    assert sorted(seen) == sorted(want), "an epoch over 4 ranks must cover each row once"
    print("ok  ranks: 4 ranks split a raw source's rows exactly once per epoch")


def test_empty_windows(root: str) -> None:
    """seq_len longer than most blocks' text: empty windows are skipped, not fatal."""
    cfg = loader_cfg(source=f"{root}/jsonl", seq_len=600, window_blocks=1,
                     global_batch_size=2, micro_batch_size=2)
    loader = MixtureLoader(cfg, 0, 1, 10, tokenizer=TOK, pin_memory=False)
    batches = run(loader, 20)
    loader.close()
    assert all(b["input_ids"].shape == (2, 600) for b in batches)
    print("ok  empty windows: skipped")


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """
    a range-capable static server. /hub/<path> mimics the hub: a 302 to
    /<path>?sig=<n>, and a signature serves 3 requests before it 403s
    (expired), so the reader must resolve the redirect again.
    """
    sig = 0
    uses: dict = {}
    redirects = 0
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_GET(self):
        cls = RangeHandler
        if self.path.startswith("/hub/"):
            with cls.lock:
                cls.sig += 1
                cls.redirects += 1
                sig = cls.sig
            self.send_response(302)
            self.send_header("Location", f"/{self.path[5:]}?sig={sig}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if "?sig=" in self.path:
            self.path, sig = self.path.split("?sig=")
            with cls.lock:
                cls.uses[sig] = cls.uses.get(sig, 0) + 1
                expired = cls.uses[sig] > 3
            if expired:
                self.send_error(403)
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
            lo, hi = int(lo), min(int(hi) if hi else len(data) - 1, len(data) - 1)
            body = data[lo:hi + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_http(root: str) -> None:
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(RangeHandler, directory=root)
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for rel, block_mb in (("jsonl/0.jsonl", BLOCK_MB), ("parquet/part1.parquet", 16),
                              ("zst/a.jsonl.zst", 16)):
            local = open_source(f"{root}/{rel}", rows_cfg(), block_mb, 1)
            remote = open_source(f"{base}/{rel}", rows_cfg(), block_mb, 1)
            assert local.blocks == remote.blocks
            for b in local.blocks:
                assert np.array_equal(local.fetch(b)["input_ids"], remote.fetch(b)["input_ids"])
        # through redirects whose targets expire: same bytes, re-resolved
        local = open_source(f"{root}/jsonl/0.jsonl", rows_cfg(), BLOCK_MB, 1)
        hub = open_source(f"{base}/hub/jsonl/0.jsonl", rows_cfg(), BLOCK_MB, 1)
        for b in local.blocks:
            assert np.array_equal(local.fetch(b)["input_ids"], hub.fetch(b)["input_ids"])
        assert RangeHandler.redirects >= len(local.blocks) // 3, RangeHandler.redirects
        assert RangeHandler.redirects < len(local.blocks), "redirects must be cached"
        try:
            open_source(f"{base}/jsonl", rows_cfg(), 16, 1)
        except AssertionError as e:
            assert "can't list files" in str(e)
        else:
            raise AssertionError("a plain-http directory must be rejected")
    finally:
        server.shutdown()
    print("ok  http: jsonl / parquet / zst over range requests match local reads, "
          "through cached redirects that expire")


def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        docs = make_text_dir(root)
        test_jsonl_blocks(root, docs)
        test_compressed(root, docs)
        test_parquet(root, docs)
        test_pack(root)
        test_captions(root)
        test_mix_resume(root)
        test_workers(root)
        test_ranks_disjoint(root)
        test_empty_windows(root)
        test_http(root)
    print("all data source tests passed")


if __name__ == "__main__":
    main()
