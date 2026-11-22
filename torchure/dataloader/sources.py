"""
raw-data sources for the mixture loader: jsonl and parquet files read by
byte range from local disk or over http, and tokenized as they arrive.
nothing is downloaded ahead of time or written to disk. pre-tokenized token
shards (shards.py) are the other kind, for data you tokenize once and reuse.

every source looks the same to mixture.py:

    source.blocks         block descriptors; a pure function of the files, so
                          identical on every rank
    source.fetch(block)   {key: array}, one row per leading index. runs on a
                          fetch thread, and must be deterministic (resume
                          refetches a block and expects the same rows)
    source.keys           fetch()'s keys
    source.eos_id
    source.window_blocks  a default blocks-per-window, or None for the loader's

kinds, picked from the files at the location:

    *.jsonl               a block is a ~block_mb byte range and owns the lines
                          that START in it: it reads from one byte before its
                          start (to see whether it starts on a line) and past
                          its end until its last line's newline. no index needed
    *.jsonl.zst / .gz     compressed streams don't seek: a block is a whole file
    *.parquet             a block is one row group, fetched with one range read
    index.json + *.bin    pre-tokenized token shards (shards.TokenSource)

text rows: a block's docs are tokenized (the same encode_batch call as
tokenize_shards.py), each followed by eos, concatenated and cut into seq_len
rows. the block's tail (< seq_len tokens) is dropped: ~0.1% of a 16MB block.

caption rows (the source sets image_field; parquet only): one image and one
caption per row, not packed with other rows:

    <|vision_start|> <|image_pad|> x num_image_tokens <|vision_end|> caption eos pad...

labels are ignore_index over the image span and the padding. images are
decoded (PIL) and resized (torch, antialiased bicubic) here, on the fetch
threads. jpegs use PIL's draft mode (downscale by 2/4/8 inside the decoder,
before the resize), with pixels slightly different from a full-size decode.
pixel_values stay uint8 (3, S, S); the model normalizes them on the gpu.
"""

import fnmatch
import gzip
import io
import json
import os
import threading
import urllib.parse
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from torchure.dataloader.shards import Reader, TokenSource

TEXT_EXTS = (".jsonl", ".jsonl.zst", ".jsonl.gz", ".parquet")
# docs per encode_batch call
TOKENIZE_DOCS = 256
SPEC_KEYS = {"path", "files", "text_field", "image_field", "window_blocks"}


@dataclass(frozen=True)
class SourceSpec:
    path: str
    # glob over file paths relative to `path`, e.g. "data/train-*.parquet"
    files: str | None = None
    text_field: str = "text"
    image_field: str | None = None
    # blocks per shuffle window for this source; None = the source's default
    window_blocks: int | None = None


def parse_spec(spec: str | dict) -> SourceSpec:
    """a SOURCES value: a location string, or a dict with `path` and options."""
    if isinstance(spec, str):
        return SourceSpec(path=spec)
    assert isinstance(spec, dict) and "path" in spec, f"source {spec!r}: needs a `path`"
    unknown = set(spec) - SPEC_KEYS
    assert not unknown, f"source {spec['path']!r}: unknown keys {sorted(unknown)}"
    return SourceSpec(**spec)


@dataclass(frozen=True)
class Vision:
    """what a vlm needs from caption rows."""
    image_size: int
    num_image_tokens: int


@dataclass(frozen=True)
class RowConfig:
    """how a raw source turns docs into rows; shared by every source of a loader."""
    seq_len: int
    tokenizer: object  # tokenizers.Tokenizer
    eos_id: int
    pad_id: int | None = None
    ignore_index: int = -100
    vision: Vision | None = None


# fetches that asked for a parallel tokenizer and haven't finished
_bursting = 0
_burst_lock = threading.Lock()


def fetch_block(source, block, burst: bool = False) -> dict[str, np.ndarray]:
    """
    source.fetch(block) on a fetch thread.

    the loader keeps tokenizers' own thread pool (rayon, every core) off:
    each fetch tokenizes its block on one core, with the gil released, so
    the parallelism is the fetch threads and a token costs the same cpu as
    in a one-process-per-core loader. with the pool on, 8 threads each
    fanning a block out over every core cost 2.3x the cpu per token on a
    14-core laptop, and a 130-core box measured ~10x (notes/DATA_LOADING.md).

    burst=True turns the pool on until this fetch returns: for the window a
    cold loader (startup, resume) is waiting on, where the wall time to the
    first batch matters and the cpu doesn't. the switch is tokenizers' env
    var, read on every encode_batch call, so it is process-wide: other
    fetches running meanwhile use the pool too.
    """
    global _bursting
    if not burst:
        return source.fetch(block)
    with _burst_lock:
        _bursting += 1
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    try:
        return source.fetch(block)
    finally:
        with _burst_lock:
            _bursting -= 1
            if _bursting == 0:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _ext(path: str) -> str | None:
    return next((e for e in TEXT_EXTS if path.endswith(e)), None)


def list_files(reader: Reader) -> list[tuple[str, int]]:
    """(path relative to the location, size in bytes) of every file, sorted."""
    if not reader.http:
        out = []
        for root, dirs, names in os.walk(reader.base):
            dirs.sort()
            for name in names:
                full = os.path.join(root, name)
                out.append((os.path.relpath(full, reader.base), os.path.getsize(full)))
        return sorted(out)
    prefix = "https://huggingface.co/datasets/"
    assert reader.base.startswith(prefix), (
        f"{reader.base}: can't list files over plain http. point at one file, a "
        "token-shard directory (index.json), or an hf://datasets/ location"
    )
    # .../datasets/<org>/<name>/resolve/<rev>[/<subdir>]
    org, name, _, rev, *sub = reader.base[len(prefix):].split("/")
    subdir = "/".join(sub)
    url = (f"https://huggingface.co/api/datasets/{org}/{name}/tree/{rev}/{subdir}"
           "?recursive=true").replace("//?", "/?")
    out = []
    while url:
        body, headers = reader.get(url)
        for entry in json.loads(body):
            if entry["type"] == "file":
                path = entry["path"]
                out.append((path[len(subdir) + 1:] if subdir else path, entry["size"]))
        # the hub pages listings: Link: <url>; rel="next"
        url = None
        for part in headers.get("Link", "").split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
    return sorted(out)


def open_source(spec: str | dict, rows: RowConfig, block_mb: float, min_blocks: int,
                tokenizer_name: str | None = None):
    """the source at spec's location: token shards if it holds an index.json, else raw files."""
    spec = parse_spec(spec)
    location = spec.path
    single = _ext(location)
    if single is not None:
        # one file: read it relative to its directory
        if location.startswith("hf://"):
            location, _, name = location.rpartition("/")
        else:
            parsed = urllib.parse.urlsplit(location)
            head, _, name = parsed.path.rpartition("/")
            location = urllib.parse.urlunsplit(parsed._replace(path=head)) if parsed.scheme \
                else (os.path.dirname(location) or ".")
        files = None
    reader = Reader(location)

    if single is None:
        try:
            has_index = (reader.read("index.json") is not None if reader.http
                         else os.path.isfile(os.path.join(reader.base, "index.json")))
        except FileNotFoundError:
            has_index = False
        if has_index:
            assert spec.image_field is None, f"{spec.path}: token shards have no images"
            src = TokenSource(location, rows.seq_len, block_mb, min_blocks=min_blocks)
            assert tokenizer_name is None or src.tokenizer == tokenizer_name, (
                f"source {spec.path!r} was tokenized with {src.tokenizer!r}, "
                f"the config's tokenizer is {tokenizer_name!r}"
            )
            src.window_blocks = spec.window_blocks
            return src
        files = [(p, n) for p, n in list_files(reader) if _ext(p) is not None]
        if spec.files is not None:
            files = [(p, n) for p, n in files if fnmatch.fnmatch(p, spec.files)]
    else:
        files = [(name, reader.size(name))]
    assert files, f"{spec.path}: no {'/'.join(TEXT_EXTS)} files" + (
        f" matching {spec.files!r}" if spec.files else "")
    kinds = {_ext(p) for p, _ in files}
    assert len(kinds) == 1, f"{spec.path}: mixed file kinds {sorted(kinds)}; use `files` to pick"
    kind = kinds.pop()
    if spec.image_field is not None:
        assert kind == ".parquet", f"{spec.path}: image sources must be parquet"
        assert rows.vision is not None, (
            f"{spec.path} sets image_field, but the model has no vision tower"
        )
    cls = {".jsonl": JsonlSource, ".parquet": ParquetSource}.get(kind, CompressedJsonlSource)
    return cls(reader, files, spec, rows, block_mb, min_blocks)


# raw sources


@dataclass(frozen=True)
class ByteRange:
    file: int
    start: int
    end: int


@dataclass(frozen=True)
class WholeFile:
    file: int


@dataclass(frozen=True)
class RowGroup:
    file: int
    group: int


class RawSource:
    """
    base of the raw kinds. a subclass sets self.blocks and implements
    records(block) -> {field: list}; this class turns the records into text
    rows or caption rows.
    """
    window_blocks: int | None = None
    # set by the loader's close(): fetches still running give up
    closed = False

    def __init__(self, reader: Reader, files: list[tuple[str, int]], spec: SourceSpec,
                 rows: RowConfig, block_mb: float, min_blocks: int):
        self.reader = reader
        self.location = spec.path
        self.files = [p for p, _ in files]
        self.sizes = [n for _, n in files]
        self.spec = spec
        self.rows = rows
        self.seq_len = rows.seq_len
        self.eos_id = rows.eos_id
        if spec.window_blocks is not None:
            self.window_blocks = spec.window_blocks
        self.images = spec.image_field is not None
        if self.images:
            self.keys = ("input_ids", "labels", "pixel_values")
            tok = rows.tokenizer

            def token(text: str) -> int:
                id_ = tok.token_to_id(text)
                assert id_ is not None, f"{text!r} not in the tokenizer vocab"
                return id_

            assert rows.pad_id is not None, "caption rows need a pad token"
            self.prefix = [token("<|vision_start|>"),
                           *[token("<|image_pad|>")] * rows.vision.num_image_tokens,
                           token("<|vision_end|>")]
            assert rows.seq_len > len(self.prefix), (
                f"seq_len {rows.seq_len} leaves no room for a caption after "
                f"{len(self.prefix)} image tokens"
            )
        else:
            self.keys = ("input_ids",)
        vocab = rows.tokenizer.get_vocab_size()
        self.dtype = np.uint16 if vocab <= 2**16 else np.uint32
        self.blocks = self._make_blocks(block_mb, min_blocks)
        assert len(self.blocks) >= min_blocks, (
            f"{self.location}: {len(self.blocks)} blocks, fewer than the {min_blocks} "
            "dp ranks that each need one"
        )

    def _make_blocks(self, block_mb: float, min_blocks: int) -> list:
        raise NotImplementedError

    def records(self, block) -> dict[str, list]:
        """the block's fields, in file order."""
        raise NotImplementedError

    def fetch(self, block) -> dict[str, np.ndarray]:
        recs = self.records(block)
        if self.images:
            return self._caption_rows(recs[self.spec.text_field], recs[self.spec.image_field])
        return {"input_ids": self.pack(recs[self.spec.text_field])}

    def tokenize(self, texts: list[str]) -> np.ndarray:
        """docs -> one flat token array, each doc followed by eos."""
        # encode_batch_fast: no offsets. tracking them is ~25% of the cpu, and
        # with several threads tokenizing at once it more than doubles the
        # cost of a token (notes/DATA_LOADING.md). in chunks, so a block's
        # Encodings (far bigger than its tokens) never all exist at once, and
        # a closed loader's fetch threads stop within a chunk
        parts = []
        for start in range(0, len(texts), TOKENIZE_DOCS):
            if self.closed:
                raise RuntimeError(f"{self.location}: the loader was closed")
            ids = [e.ids for e in
                   self.rows.tokenizer.encode_batch_fast(texts[start:start + TOKENIZE_DOCS])]
            ends = np.cumsum(np.fromiter(map(len, ids), np.int64, len(ids)) + 1)
            flat = np.empty(int(ends[-1]), dtype=self.dtype)
            for doc, end in zip(ids, ends, strict=True):
                flat[end - 1 - len(doc):end - 1] = doc
            flat[ends - 1] = self.eos_id
            parts.append(flat)
        return np.concatenate(parts) if parts else np.empty(0, dtype=self.dtype)

    def pack(self, texts: list[str]) -> np.ndarray:
        flat = self.tokenize(texts)
        n = len(flat) // self.seq_len
        return flat[:n * self.seq_len].reshape(n, self.seq_len)

    def _caption_rows(self, captions: list[str], images: list) -> dict[str, np.ndarray]:
        import torch
        from PIL import Image
        from torch.nn.functional import interpolate

        rows = self.rows
        n, L, P = len(captions), self.seq_len, len(self.prefix)
        ids = np.full((n, L), rows.pad_id, dtype=np.int32)
        labels = np.full((n, L), rows.ignore_index, dtype=np.int32)
        encodings = rows.tokenizer.encode_batch(captions, add_special_tokens=False)
        for i, enc in enumerate(encodings):
            seq = (self.prefix + enc.ids + [self.eos_id])[:L]
            ids[i, :len(seq)] = seq
            labels[i, P:len(seq)] = seq[P:]
        S = rows.vision.image_size
        pixels = np.empty((n, 3, S, S), dtype=np.uint8)
        # _rgb's zero-copy view is read-only, which torch warns about (once)
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        for i, data in enumerate(images):
            image = Image.open(io.BytesIO(data))
            image.draft("RGB", (S, S))  # jpeg only; a no-op otherwise
            if image.mode != "RGB":
                image = image.convert("RGB")
            # torch's antialiased bicubic on uint8 is PIL's resize (same
            # kernel, within a few levels on ~5% of values) vectorized: 0.6 ms
            # for what takes PIL 2.1, next to a 2.7 ms decode
            hwc = torch.from_numpy(_rgb(image))
            chw = interpolate(hwc.permute(2, 0, 1)[None], size=(S, S), mode="bicubic",
                              antialias=True)[0, :3]
            # copied by numpy: a torch copy this size starts an openmp team,
            # one per fetch thread
            pixels[i] = chw.numpy()
        return {"input_ids": ids, "labels": labels, "pixel_values": pixels}


def _rgb(image) -> np.ndarray:
    """
    an RGB PIL image's pixels, (H, W, 3 or 4) uint8. PIL stores RGB as 4
    bytes a pixel and can lend that memory out through arrow: no copy, and
    none of the gil-held packing np.array(image) does (~15% of an image's
    cost, most of what the fetch threads serialize on). the 4th channel is
    padding. falls back to the copy when PIL can't (an old PIL, or an image
    too big for one of its memory blocks).
    """
    image.load()
    if hasattr(image, "__arrow_c_array__"):
        import pyarrow as pa

        try:
            flat = pa.array(image)
            return flat.values.to_numpy().reshape(image.height, image.width, flat.type.list_size)
        except (ValueError, TypeError, NotImplementedError):
            pass
    return np.array(image)


class JsonlSource(RawSource):
    """byte-range blocks over uncompressed jsonl."""

    def _make_blocks(self, block_mb: float, min_blocks: int) -> list[ByteRange]:
        size = max(4096, int(block_mb * 2**20))
        while True:
            blocks = [ByteRange(f, start, min(start + size, n))
                      for f, n in enumerate(self.sizes) for start in range(0, n, size)]
            if len(blocks) >= min_blocks or size == 4096:
                return blocks
            size = max(4096, size // 2)

    def lines(self, block: ByteRange) -> bytes:
        """the lines that start in [start, end), newline-terminated (bar a file's last)."""
        path, size = self.files[block.file], self.sizes[block.file]
        lo = max(0, block.start - 1)
        data = self.reader.read(path, lo, block.end - lo)
        if block.start > 0:
            # data[0] is the byte before the block: a line starts after the first newline
            cut = data.find(b"\n")
            if cut < 0:
                return b""  # inside one long line that an earlier block owns
            data = data[cut + 1:]
        if not data:
            return b""
        # finish the last line, which may run past the block
        pos, step, tail = block.end, 1 << 16, []
        while not data.endswith(b"\n") and not (tail and tail[-1].endswith(b"\n")) and pos < size:
            chunk = self.reader.read(path, pos, min(step, size - pos))
            cut = chunk.find(b"\n")
            tail.append(chunk if cut < 0 else chunk[:cut + 1])
            pos += len(chunk)
            step = min(step * 2, 1 << 24)
        return data + b"".join(tail)

    def records(self, block: ByteRange) -> dict[str, list]:
        field = self.spec.text_field
        return {field: [json.loads(line)[field] for line in self.lines(block).splitlines()
                        if line.strip()]}


class CompressedJsonlSource(RawSource):
    """whole-file blocks over .jsonl.zst / .jsonl.gz."""
    # a whole compressed file is one block (tens of MB), so one per window
    window_blocks = 1

    def _make_blocks(self, block_mb: float, min_blocks: int) -> list[WholeFile]:
        return [WholeFile(f) for f in range(len(self.files))]

    def records(self, block: WholeFile) -> dict[str, list]:
        path = self.files[block.file]
        data = self.reader.read(path)
        if path.endswith(".zst"):
            from compression import zstd  # stdlib, python 3.14
            data = zstd.decompress(data)
        else:
            data = gzip.decompress(data)
        field = self.spec.text_field
        return {field: [json.loads(line)[field] for line in data.splitlines() if line.strip()]}


class _SpanFile(io.RawIOBase):
    """
    a read-only file of known size for pyarrow: reads inside a preloaded span
    come from memory, anything else is a range read.
    """

    def __init__(self, reader: Reader, path: str, size: int, spans: list[tuple[int, bytes]]):
        self.reader, self.path, self._size, self.spans = reader, path, size, spans
        self.pos = 0
        self.misses = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: 0, 1: self.pos, 2: self._size}[whence] + offset
        return self.pos

    def readinto(self, buf) -> int:
        n = min(len(buf), self._size - self.pos)
        if n <= 0:
            return 0
        for start, data in self.spans:
            if start <= self.pos and self.pos + n <= start + len(data):
                buf[:n] = data[self.pos - start:self.pos - start + n]
                break
        else:
            self.misses += 1
            buf[:n] = self.reader.read(self.path, self.pos, n)
        self.pos += n
        return n


class ParquetSource(RawSource):
    """row-group blocks over parquet; each file's footer is read once, at startup."""

    def _make_blocks(self, block_mb: float, min_blocks: int) -> list[RowGroup]:
        import pyarrow.parquet as pq

        def footer(f: int):
            path, size = self.files[f], self.sizes[f]
            lo = max(0, size - (1 << 16))
            tail = _SpanFile(self.reader, path, size, [(lo, self.reader.read(path, lo, size - lo))])
            return pq.ParquetFile(tail).metadata

        with ThreadPoolExecutor(32) as pool:
            self.meta = list(pool.map(footer, range(len(self.files))))
        self.fields = [self.spec.text_field] + ([self.spec.image_field] if self.images else [])
        for path, meta in zip(self.files, self.meta, strict=True):
            names = set(meta.schema.to_arrow_schema().names)
            missing = [f for f in self.fields if f not in names]
            assert not missing, f"{self.location}/{path}: no column(s) {missing}; has {sorted(names)}"
        return [RowGroup(f, g) for f, meta in enumerate(self.meta)
                for g in range(meta.num_row_groups) if meta.row_group(g).num_rows > 0]

    def _span(self, block: RowGroup) -> tuple[int, int]:
        """byte range covering the row group's chunks of the fields we read."""
        rg = self.meta[block.file].row_group(block.group)
        lo, hi = None, 0
        for j in range(rg.num_columns):
            col = rg.column(j)
            if col.path_in_schema.split(".")[0] not in self.fields:
                continue
            start = col.data_page_offset
            if col.has_dictionary_page and col.dictionary_page_offset:
                start = min(start, col.dictionary_page_offset)
            lo = start if lo is None else min(lo, start)
            hi = max(hi, start + col.total_compressed_size)
        return lo, hi

    def records(self, block: RowGroup) -> dict[str, list]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path, size = self.files[block.file], self.sizes[block.file]
        lo, hi = self._span(block)
        f = _SpanFile(self.reader, path, size, [(lo, self.reader.read(path, lo, hi - lo))])
        table = pq.ParquetFile(f, metadata=self.meta[block.file]).read_row_group(
            block.group, columns=self.fields
        )
        out = {self.spec.text_field: table.column(self.spec.text_field).to_pylist()}
        if self.images:
            col = table.column(self.spec.image_field).combine_chunks()
            # hf's Image feature is a struct {bytes, path}; a plain binary column also works
            if pa.types.is_struct(col.type):
                col = col.field("bytes")
            out[self.spec.image_field] = col.to_pylist()
        return out
