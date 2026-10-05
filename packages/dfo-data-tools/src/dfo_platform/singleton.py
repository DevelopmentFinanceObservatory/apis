"""Lock-guarded process singleton with reference-counted teardown.

`build` runs once on the 0-to-1 transition with the winning caller's arguments. A failed build
leaves the singleton empty. `begin_release` returns the last instance for teardown outside the lock.
`close` is optional.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager


class RefCountedSingleton[T]:
    """Build once for concurrent acquirers and close after the last release. `force_reset()` is a
    test-only escape hatch for instances that remain acquired for process lifetime.
    """

    def __init__(
        self,
        *,
        build: Callable[..., T],
        close: Callable[[T], None] | None = None,
        label: str = "instance",
    ) -> None:
        self._build = build
        self._close = close
        self._label = label
        self._lock = threading.Lock()
        self._instance: T | None = None
        self._refcount = 0

    def acquire(self, *args: object, **kwargs: object) -> T:
        """Return the shared instance, building it under the lock on the first acquisition."""
        with self._lock:
            if self._instance is None:
                self._instance = self._build(*args, **kwargs)
            self._refcount += 1
            return self._instance

    def release(self, instance: T) -> None:
        """Release one holder and close the instance at zero. Raises `RuntimeError` for an
        unacquired instance or an extra release.
        """
        with self._lock:
            self._check_holder(instance)
            self._refcount -= 1
            if self._refcount == 0:
                self._instance = None
                if self._close is not None:
                    self._close(instance)

    def begin_release(self, instance: T) -> T | None:
        """Release one holder and return the instance for out-of-lock teardown at zero. Return
        `None` while other holders remain. Raises `RuntimeError` on misuse.
        """
        with self._lock:
            self._check_holder(instance)
            self._refcount -= 1
            if self._refcount == 0:
                closing, self._instance = self._instance, None
                return closing
            return None

    def force_reset(self) -> None:
        """Test-only reset. Drops the instance and refcount. Call `release()` first when teardown
        must run.
        """
        with self._lock:
            self._instance = None
            self._refcount = 0

    @contextmanager
    def isolated(self) -> Iterator[None]:
        """Test-only isolation scope. Swaps out the instance and refcount inside the block, then
        restores both on exit without running teardown.
        """
        with self._lock:
            snapshot = self._instance, self._refcount
            self._instance, self._refcount = None, 0
        try:
            yield
        finally:
            with self._lock:
                self._instance, self._refcount = snapshot

    @contextmanager
    def poisoned_for_test(self, instance: T) -> Iterator[None]:
        """Test-only: substitute an instance inside the block and restore the prior value on exit.
        The refcount is unchanged.
        """
        with self._lock:
            previous = self._instance
            self._instance = instance
        try:
            yield
        finally:
            with self._lock:
                self._instance = previous

    def _check_holder(self, instance: T) -> None:
        if self._instance is not instance or self._refcount <= 0:
            raise RuntimeError(
                f"release() called on a {self._label} that acquire() never returned, "
                "or called more times than acquire() was"
            )
