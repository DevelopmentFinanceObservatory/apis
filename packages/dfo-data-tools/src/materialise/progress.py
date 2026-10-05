"""A thread-safe row counter for materialisation progress and cancellation.
Callers increment it when rows enter the result frame; a cancelled run raises
`MaterialisationCancelled` on its next increment. This package must not import
`datasets_api`, keeping materialisation independent of the operation store.
"""

from __future__ import annotations

import threading


class MaterialisationCancelled(Exception):
    """Raised by `RowProgress.add` once `RowProgress.cancel` has been
    called: the caller that owns the counter no longer wants the run."""


class RowProgress:
    """A lock-guarded total shared by fetch workers and the heartbeat thread.
    Increment once per completed chunk that survives into the result frame.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0
        self._cancelled = threading.Event()

    def add(self, rows: int) -> None:
        self.raise_if_cancelled()
        with self._lock:
            self._value += rows

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        if self._cancelled.is_set():
            raise MaterialisationCancelled("the run was cancelled by its caller")

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


__all__ = ["MaterialisationCancelled", "RowProgress"]
