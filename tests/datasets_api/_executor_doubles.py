"""Deterministic executor failure doubles for datasets API door tests.
They model work that never starts and a broken pool submission.
"""

from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures.thread import BrokenThreadPool


class NeverStartingExecutor:
    """Return pending futures that never run, without relying on thread timing.

    Cancelling them tests that shutdown releases each submission's backpressure permit.
    """

    def __init__(self, *, max_workers: int, thread_name_prefix: str = "") -> None:
        self.pending: list[Future] = []

    def submit(self, fn, /, *args, **kwargs) -> Future:
        future: Future = Future()
        self.pending.append(future)
        return future

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        if cancel_futures:
            for future in self.pending:
                future.cancel()


class BrokenSubmitExecutor:
    """Raise `BrokenThreadPool` on submit to model a dead worker.

    This distinguishes a broken pool from a post-shutdown `RuntimeError`.
    """

    def __init__(self, *, max_workers: int, thread_name_prefix: str = "") -> None:
        pass

    def submit(self, fn, /, *args, **kwargs) -> Future:
        raise BrokenThreadPool("pool broken for testing: a worker thread died")

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        pass
