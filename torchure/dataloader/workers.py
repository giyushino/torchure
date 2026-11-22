"""
fetch worker processes for the mixture loader (`data.workers`): each runs
source.fetch(block) for the trainer, so tokenizing and image decoding stay
off the trainer's gil.

not multiprocessing / ProcessPoolExecutor. a worker is a plain subprocess
that imports this module and nothing of the trainer's:

  startup   a spawn child re-imports the parent's __main__, i.e. the whole
            training script, once per worker. these import the source code
            only (plus torch, for image sources: it resizes), and do it
            while the parent is still listing files.
  results   rows come back as raw array bytes on a pipe, read straight into
            the destination array with the gil released: one copy, no pickle.
  exit      a worker leaves when its task pipe closes, which the parent's
            death does for it. close() kills them; there are no manager
            threads or queues left to join at interpreter exit.

protocol, per worker, one request at a time. a message is an 8-byte length
and a pickle:

    parent -> worker   ("init", pickled {name: source})    once
                       ("fetch", name, block)
    worker -> parent   ("ok", [(key, dtype, shape), ...]), then each array's bytes
                       ("err", exception or None, traceback text)
"""

import fcntl
import os
import pickle
import queue
import signal
import struct
import subprocess
import sys
import traceback
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

_LENGTH = struct.Struct("<Q")
_PIPE_BYTES = 1 << 20


def _write(fd: int, data) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _read_into(fd: int, view: memoryview) -> None:
    while view:
        n = os.readinto(fd, view)
        if n == 0:
            raise EOFError
        view = view[n:]


def _send(fd: int, message) -> None:
    body = pickle.dumps(message, protocol=5)
    _write(fd, _LENGTH.pack(len(body)))
    _write(fd, body)


def _recv(fd: int):
    head = bytearray(_LENGTH.size)
    _read_into(fd, memoryview(head))
    body = bytearray(_LENGTH.unpack(head)[0])
    _read_into(fd, memoryview(body))
    return pickle.loads(body)


def _bytes_of(array: np.ndarray) -> memoryview:
    """a contiguous array's memory as flat bytes."""
    return memoryview(array.reshape(-1).view(np.uint8))


def main(task_fd: int, result_fd: int, images: bool) -> None:
    """a worker's whole life: serve fetches until the task pipe closes."""
    # the trainer answers a job-wide SIGTERM (or a ctrl-c) by checkpointing
    # and needs batches until then; it ends us by closing the pipe or kill()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # one core per worker, not a tokenizers thread pool in each
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # what the first fetch would import, while the parent opens the sources
    import tokenizers  # noqa: F401
    if images:
        import PIL.Image  # noqa: F401
        import pyarrow.parquet  # noqa: F401
        import torch

        torch.set_num_threads(1)

    sources = None
    try:
        while True:
            message = _recv(task_fd)
            if message[0] == "init":
                sources = pickle.loads(message[1])
                continue
            _, name, block = message
            try:
                rows = sources[name].fetch(block)
                arrays = [(key, np.ascontiguousarray(value)) for key, value in rows.items()]
            except Exception as e:
                text = traceback.format_exc()
                try:
                    pickle.loads(pickle.dumps(e))
                except Exception:
                    e = None
                _send(result_fd, ("err", e, text))
                continue
            _send(result_fd, ("ok", [(key, a.dtype.str, a.shape) for key, a in arrays]))
            for _, a in arrays:
                _write(result_fd, _bytes_of(a))
    except (EOFError, BrokenPipeError):
        pass  # the parent closed the pipes or died


class _Worker:
    """the parent's end of one worker process."""

    def __init__(self, images: bool):
        task_r, self.task_w = os.pipe()
        self.result_r, result_w = os.pipe()
        for fd in (self.task_w, self.result_r):
            try:
                fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, _PIPE_BYTES)
            except (AttributeError, OSError):
                pass  # not linux, or over the pipe size limit: the default works
        # the parent's import path, so `torchure` resolves the same way there
        env = os.environ | {"PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        self.process = subprocess.Popen(
            [sys.executable, "-c",
             "from torchure.dataloader.workers import main; "
             f"main({task_r}, {result_w}, {images})"],
            pass_fds=(task_r, result_w), stdin=subprocess.DEVNULL, env=env,
        )
        os.close(task_r)
        os.close(result_w)
        self.started = False

    def fetch(self, payload: bytes, name: str, block) -> dict[str, np.ndarray]:
        try:
            if not self.started:
                _send(self.task_w, ("init", payload))
                self.started = True
            _send(self.task_w, ("fetch", name, block))
            status, *rest = _recv(self.result_r)
            if status == "ok":
                rows = {}
                for key, dtype, shape in rest[0]:
                    rows[key] = np.empty(shape, dtype=dtype)
                    _read_into(self.result_r, _bytes_of(rows[key]))
                return rows
        except (EOFError, OSError) as e:
            raise RuntimeError(
                f"loader worker {self.process.pid} died (exit code {self.process.poll()})"
            ) from e
        error, text = rest
        if error is None:
            error = RuntimeError("loader worker failed")
        error.add_note(f"in loader worker {self.process.pid}:\n{text}")
        raise error

    def close(self) -> None:
        self.process.kill()
        self.process.wait()
        os.close(self.task_w)
        os.close(self.result_r)


class WorkerPool:
    """
    n worker processes, started on construction (before the sources exist,
    so their imports overlap the parent opening them). start(sources) then
    submit(name, block) -> Future of that source's fetch(block).
    images: the sources will decode images.
    """

    def __init__(self, n: int, images: bool = False):
        self.workers = [_Worker(images) for _ in range(n)]
        self._idle: queue.SimpleQueue[_Worker] = queue.SimpleQueue()
        for worker in self.workers:
            self._idle.put(worker)
        # one thread per worker: it writes the request and blocks, gil
        # released, on the result
        self._threads = ThreadPoolExecutor(n, thread_name_prefix="mix-worker")
        self._payload: bytes | None = None

    @property
    def processes(self) -> list[subprocess.Popen]:
        return [w.process for w in self.workers]

    def start(self, sources: dict) -> None:
        self._payload = pickle.dumps(sources, protocol=5)

    def _fetch(self, name: str, block) -> dict[str, np.ndarray]:
        worker = self._idle.get()
        try:
            return worker.fetch(self._payload, name, block)
        finally:
            self._idle.put(worker)

    def submit(self, name: str, block) -> Future:
        assert self._payload is not None, "start(sources) first"
        return self._threads.submit(self._fetch, name, block)

    def close(self) -> None:
        self._threads.shutdown(wait=False, cancel_futures=True)
        # killing a worker ends the read its thread is blocked in
        for worker in self.workers:
            worker.process.kill()
        self._threads.shutdown(wait=True)
        for worker in self.workers:
            worker.close()
        self.workers = []
