"""
pre-tokenized token shards, read by byte range from a local path or over http.

a source is a directory (or url prefix) holding

    index.json        {"format": "torchure-tokens", "version": 1, "dtype": "uint32",
                       "tokenizer": ..., "eos_id": ..., "files": [{"path", "num_tokens"}]}
    000000.bin ...    raw little-endian tokens, every doc followed by eos

written by torchure/dataloader/tokenize_shards.py; the other source kinds
(raw jsonl / parquet, tokenized while training) are in sources.py. a source is cut into
seq_len rows (per file; each file's tail < seq_len is dropped) and the rows
into blocks: one block is one range request. see notes/DATA_MIXING.md.

locations:
    /abs/or/relative/dir          local disk (pread)
    https://host/prefix           any server that honors Range (r2, s3, a cdn)
    hf://datasets/<org>/<name>[/<subdir>]
                                  huggingface hub; HF_TOKEN is sent if set
"""

import http.client
import json
import os
import random
import threading
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
    byte-range reads of one source's files, shared by every fetch thread.

    over http, range reads keep one connection per host per thread, and
    remember where each file redirected to: the hub answers every resolve url
    with a redirect to a presigned cdn url, and a fresh connection plus that
    redirect is most of a 3.5MB read's time (0.38s vs 0.08s measured). a
    cached redirect that stops working (the signature expired: 401/403/404
    from the cdn) is dropped and resolved again.
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
        self._redirects: dict[str, str] = {}
        self._local = threading.local()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_local"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()

    def url(self, path: str) -> str:
        return f"{self.base}/{path}" if path else self.base

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
        return self._http_read(self.url(path), offset, length)

    def size(self, path: str) -> int:
        """bytes in `path`. over http: a one-byte range read's Content-Range."""
        if not self.http:
            return os.path.getsize(os.path.join(self.base, path))
        _, headers = self.get(self.url(path), {"Range": "bytes=0-0"})
        total = headers.get("Content-Range", "").rpartition("/")[2]
        assert total.isdigit(), f"{self.url(path)}: no size in Content-Range"
        return int(total)

    def _conn(self, scheme: str, netloc: str) -> http.client.HTTPConnection:
        conns = getattr(self._local, "conns", None)
        if conns is None:
            conns = self._local.conns = {}
        key = (scheme, netloc)
        if key not in conns:
            cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
            conns[key] = cls(netloc, timeout=self.timeout)
        return conns[key]

    def _drop_conn(self, scheme: str, netloc: str) -> None:
        conn = getattr(self._local, "conns", {}).pop((scheme, netloc), None)
        if conn is not None:
            conn.close()

    def _http_read(self, url: str, offset: int, length: int | None) -> bytes:
        rng = {}
        if length is not None:
            rng["Range"] = f"bytes={offset}-{offset + length - 1}"
        elif offset:
            rng["Range"] = f"bytes={offset}-"
        attempt = hops = 0
        while True:
            target = self._redirects.get(url, url)
            parts = urllib.parse.urlsplit(target)
            headers = dict(rng)
            # the token only ever goes to the host it was configured for
            if parts.netloc == urllib.parse.urlsplit(url).netloc:
                headers |= self.headers
            path = parts.path + (f"?{parts.query}" if parts.query else "")
            error = None
            try:
                conn = self._conn(parts.scheme, parts.netloc)
                conn.request("GET", path, headers=headers)
                response = conn.getresponse()
                body = response.read()
                status = response.status
            except (OSError, http.client.HTTPException) as e:
                self._drop_conn(parts.scheme, parts.netloc)
                status, error = None, e
            if status in (301, 302, 303, 307, 308):
                hops += 1
                if hops > 5:
                    raise OSError(f"{url}: too many redirects")
                self._redirects[url] = urllib.parse.urljoin(target, response.headers["Location"])
                continue
            if status in (401, 403, 404) and target != url:
                # a cached presigned url went stale
                self._redirects.pop(url, None)
                continue
            if status == 404:
                raise FileNotFoundError(f"{url}: http 404")
            if status is not None and status not in _RETRY_STATUS:
                if status >= 400:
                    raise OSError(f"{url}: http {status}")
                # a server that ignores Range answers 200 with the whole file
                if rng and status != 206:
                    raise OSError(f"{url}: server ignored the Range header (status {status})")
                if length is not None and len(body) != length:
                    error = OSError(f"short body: {len(body)} of {length} bytes")
                else:
                    return body
            if attempt == self.retries:
                raise OSError(f"{url}: {error or f'http {status}'} "
                              f"(after {self.retries} retries)") from error
            attempt += 1
            # exponential backoff with jitter, capped at a minute
            time.sleep(min(60.0, 0.5 * 2**(attempt - 1)) * (0.5 + random.random()))

    def get(self, url: str, headers: dict | None = None) -> tuple[bytes, dict]:
        """(body, response headers) of a GET, with retries. 404 -> FileNotFoundError."""
        headers = self.headers | (headers or {})
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(url, headers=headers)
                with _opener.open(request, timeout=self.timeout) as response:
                    # a server that ignores Range answers 200 with the whole file
                    if "Range" in headers and response.status != 206:
                        raise OSError(f"{url}: server ignored the Range header "
                                      f"(status {response.status})")
                    data = response.read()
                    length = response.headers.get("Content-Length")
                    if length is not None and len(data) != int(length):
                        raise urllib.error.URLError(f"short body: {len(data)} of {length} bytes")
                    return data, dict(response.headers)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise FileNotFoundError(f"{url}: http 404") from e
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

    keys = ("input_ids",)
    kind = "tokens"

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

    def fetch(self, block: Block) -> dict[str, np.ndarray]:
        """the block's rows, {"input_ids": (num_rows, seq_len)} in the shard dtype."""
        row_bytes = self.seq_len * self.dtype.itemsize
        data = self.reader.read(
            self.files[block.file], block.row * row_bytes, block.num_rows * row_bytes
        )
        rows = np.frombuffer(data, dtype=self.dtype).reshape(block.num_rows, self.seq_len)
        return {"input_ids": rows}


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
