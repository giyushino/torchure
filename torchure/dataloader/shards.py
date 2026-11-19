"""
pre-tokenized token shards, read by byte range from a local path or over http.

a source is a directory (or url prefix) holding

    index.json        {"format": "torchure-tokens", "version": 1, "dtype": "uint32",
                       "tokenizer": ..., "eos_id": ..., "files": [{"path", "num_tokens"}]}
    000000.bin ...    raw little-endian tokens, every doc followed by eos

written by torchure/dataloader/tokenize_shards.py. a source is cut into
seq_len rows (per file; each file's tail < seq_len is dropped) and the rows
into blocks: one block is one range request. see notes/DATA_MIXING.md.

locations:
    /abs/or/relative/dir          local disk (pread)
    https://host/prefix           any server that honors Range (r2, s3, a cdn)
    hf://datasets/<org>/<name>[/<subdir>]
                                  huggingface hub; HF_TOKEN is sent if set
"""

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np

FORMAT = "torchure-tokens"

# 408/425/429 and 5xx are worth retrying; any other 4xx is our fault
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


def resolve(location: str) -> str:
    """hf:// -> its https resolve url; anything else unchanged."""
    if location.startswith("hf://"):
        parts = location[len("hf://"):].split("/")
        assert len(parts) >= 3 and parts[0] == "datasets", (
            f"{location!r}: expected hf://datasets/<org>/<name>[/<subdir>]"
        )
        repo, subdir = "/".join(parts[1:3]), "/".join(parts[3:])
        url = f"https://huggingface.co/datasets/{repo}/resolve/main"
        return f"{url}/{subdir}".rstrip("/")
    return location.rstrip("/")


def _is_http(location: str) -> bool:
    return location.startswith(("http://", "https://"))


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """
    urllib forwards every header through a redirect. the hub redirects to a
    presigned cdn url, which rejects a second auth mechanism, and the token
    has no business leaving huggingface.co anyway. Range is kept.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != req.host:
            new.remove_header("Authorization")
        return new


_opener = urllib.request.build_opener(_StripAuthOnRedirect)


class Reader:
    """
    byte-range reads of one source's files. stateless apart from config, so
    one reader serves every fetch thread (local reads open the file per call;
    urllib makes a connection per call, which is noise next to a multi-MB
    block).
    """

    def __init__(self, location: str, retries: int = 8, timeout: float = 60.0):
        self.base = resolve(location)
        self.http = _is_http(self.base)
        self.retries = retries
        self.timeout = timeout
        self.headers = {}
        token = os.environ.get("HF_TOKEN")
        if token and self.base.startswith("https://huggingface.co/"):
            self.headers["Authorization"] = f"Bearer {token}"

    def read(self, path: str, offset: int = 0, length: int | None = None) -> bytes:
        """bytes [offset, offset + length) of `path`; length None = to the end."""
        if not self.http:
            with open(os.path.join(self.base, path), "rb") as f:
                if length is None:
                    f.seek(offset)
                    return f.read()
                data = os.pread(f.fileno(), length, offset)
            if len(data) != length:
                raise OSError(f"{path}: short read at {offset} ({len(data)} of {length} bytes)")
            return data
        return self._http_read(f"{self.base}/{path}", offset, length)

    def _http_read(self, url: str, offset: int, length: int | None) -> bytes:
        headers = dict(self.headers)
        if length is not None:
            headers["Range"] = f"bytes={offset}-{offset + length - 1}"
        elif offset:
            headers["Range"] = f"bytes={offset}-"
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(url, headers=headers)
                with _opener.open(request, timeout=self.timeout) as response:
                    # a server that ignores Range answers 200 with the whole file
                    if "Range" in headers and response.status != 206:
                        raise OSError(f"{url}: server ignored the Range header "
                                      f"(status {response.status})")
                    data = response.read()
                if length is not None and len(data) != length:
                    raise urllib.error.URLError(f"short body: {len(data)} of {length} bytes")
                return data
            except urllib.error.HTTPError as e:
                if e.code not in _RETRY_STATUS or attempt == self.retries:
                    raise OSError(f"{url}: http {e.code}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.retries:
                    raise OSError(f"{url}: {e} (after {self.retries} retries)") from e
            # exponential backoff with jitter, capped at a minute
            time.sleep(min(60.0, 0.5 * 2**attempt) * (0.5 + random.random()))
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class Block:
    file: int        # index into the source's files
    row: int         # first row within that file
    num_rows: int


class TokenSource:
    """
    one source's index plus its seq_len row / block layout. blocks hold about
    block_mb each, shrunk if needed so there are at least `min_blocks` of
    them (every dp rank must own at least one).
    """

    def __init__(self, location: str, seq_len: int, block_mb: float = 16, min_blocks: int = 1):
        self.location = location
        self.reader = Reader(location)
        index = json.loads(self.reader.read("index.json"))
        assert index.get("format") == FORMAT, f"{location}: not a {FORMAT} source"
        assert index.get("version") == 1, f"{location}: unknown version {index.get('version')}"
        self.index = index
        self.dtype = np.dtype(index["dtype"]).newbyteorder("<")
        self.tokenizer = index["tokenizer"]
        self.eos_id = index["eos_id"]
        self.seq_len = seq_len
        self.files = [f["path"] for f in index["files"]]
        file_rows = [f["num_tokens"] // seq_len for f in index["files"]]
        self.num_rows = sum(file_rows)
        assert self.num_rows >= min_blocks, (
            f"{location}: {self.num_rows} rows of {seq_len} tokens, fewer than the "
            f"{min_blocks} dp ranks that each need one"
        )

        row_bytes = seq_len * self.dtype.itemsize
        block_rows = max(1, int(block_mb * 2**20) // row_bytes)
        while True:
            self.blocks = [
                Block(f, start, min(block_rows, rows - start))
                for f, rows in enumerate(file_rows)
                for start in range(0, rows, block_rows)
            ]
            if len(self.blocks) >= min_blocks or block_rows == 1:
                break
            block_rows = max(1, block_rows // 2)
        self.block_rows = block_rows

    def fetch(self, block: Block) -> np.ndarray:
        """the block's rows, (num_rows, seq_len), in the shard dtype."""
        row_bytes = self.seq_len * self.dtype.itemsize
        data = self.reader.read(
            self.files[block.file], block.row * row_bytes, block.num_rows * row_bytes
        )
        return np.frombuffer(data, dtype=self.dtype).reshape(block.num_rows, self.seq_len)


def write_index(out_dir: str, *, dtype: str, tokenizer: str, eos_id: int,
                files: list[dict], extra: dict | None = None) -> None:
    """the index goes last, so a source is readable only once complete."""
    index = {
        "format": FORMAT, "version": 1, "dtype": dtype, "tokenizer": tokenizer,
        "eos_id": eos_id, "files": files, **(extra or {}),
    }
    tmp = os.path.join(out_dir, "index.json.tmp")
    with open(tmp, "w") as f:
        json.dump(index, f, indent=1)
    os.replace(tmp, os.path.join(out_dir, "index.json"))
