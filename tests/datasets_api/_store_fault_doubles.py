"""DatasetStore doubles for deterministic driver failures, pool timeouts, and concurrency tests.
All run without Postgres connections or scheduler races.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager

from dataset_store.errors import PoolTimeoutError
from tests.spec_registry.conftest import FakeRegistryState, FakeTransaction


class PoolBoundedFakeDatasetStore:
    """Bound open transactions with a non-blocking semaphore of size `pool_size`.

    Concurrent overuse raises `RuntimeError`. Sequential registry writes still accept collections
    larger than the cap because they hold one transaction at a time.
    """

    def __init__(self, *, pool_size: int) -> None:
        self.state = FakeRegistryState()
        self._semaphore = threading.Semaphore(pool_size)

    @contextmanager
    def transaction(self) -> Generator[FakeTransaction, None, None]:
        if not self._semaphore.acquire(blocking=False):
            raise RuntimeError(
                "PoolBoundedFakeDatasetStore: no free connection -- more transactions "
                "were opened concurrently than the configured pool size allows"
            )
        try:
            with self.state.lock:
                yield FakeTransaction(self.state)
        finally:
            self._semaphore.release()


class _ExplodingTransaction:
    """Raise a bare `RuntimeError` for each query, outside `DatasetStoreError`."""

    def __init__(self, *, message: str) -> None:
        self._message = message

    def execute(self, sql: str, params: object = ()) -> None:
        raise RuntimeError(self._message)

    def fetch_one(self, sql: str, params: object = ()) -> object:
        raise RuntimeError(self._message)

    def fetch_all(self, sql: str, params: object = ()) -> object:
        raise RuntimeError(self._message)


class DriverExplodingFakeDatasetStore:
    """Raise bare `RuntimeError` query failures so tests cover errors outside `DatasetStoreError`."""

    def __init__(self, *, message: str) -> None:
        self._message = message

    @contextmanager
    def transaction(self) -> Generator[_ExplodingTransaction, None, None]:
        yield _ExplodingTransaction(message=self._message)


class _SqlstateExplodingTransaction:
    """Raise `RuntimeError` with pg8000's `{"C": sqlstate}` payload on every query."""

    def __init__(self, *, sqlstate: str) -> None:
        self._sqlstate = sqlstate

    def _raise(self) -> None:
        raise RuntimeError({"C": self._sqlstate})

    def execute(self, sql: str, params: object = ()) -> None:
        self._raise()

    def fetch_one(self, sql: str, params: object = ()) -> object:
        self._raise()

    def fetch_all(self, sql: str, params: object = ()) -> object:
        self._raise()


class SqlstateExplodingFakeDatasetStore:
    """Inject a pg8000-shaped SQLSTATE failure to test driver-error mapping.

    `DriverExplodingFakeDatasetStore` supplies a bare error for comparison.
    """

    def __init__(self, *, sqlstate: str) -> None:
        self._sqlstate = sqlstate

    @contextmanager
    def transaction(self) -> Generator[_SqlstateExplodingTransaction, None, None]:
        yield _SqlstateExplodingTransaction(sqlstate=self._sqlstate)


class PoolTimeoutFakeDatasetStore:
    """Raise `PoolTimeoutError` before yielding, as a timed-out pool borrow does."""

    @contextmanager
    def transaction(self) -> Generator[FakeTransaction, None, None]:
        raise PoolTimeoutError("no connection freed up within 30.0s (pool size 1)")
        yield  # pragma: no cover -- unreachable, satisfies the generator shape


class BlockingFakeDatasetStore:
    """Block `transaction` on `release` so backpressure tests can occupy worker slots."""

    def __init__(self) -> None:
        self.state = FakeRegistryState()
        self.entered = threading.Event()
        self.release = threading.Event()

    @contextmanager
    def transaction(self) -> Generator[FakeTransaction, None, None]:
        self.entered.set()
        self.release.wait(timeout=5.0)
        with self.state.lock:
            yield FakeTransaction(self.state)
