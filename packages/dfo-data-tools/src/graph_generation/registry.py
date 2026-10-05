"""Thread-safe registry for cache callbacks triggered by generation changes."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from typing import Protocol

logger = logging.getLogger(__name__)


class RegistrationHandle(Protocol):
    def close(self) -> None:
        """Unregister the cache. Repeated calls have no effect."""
        ...


class Registration:
    """The handle `CacheRegistry.register` returns."""

    def __init__(self, *, registry: CacheRegistry, name: str) -> None:
        self._registry = registry
        self._name = name
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._registry._remove(self._name)


class CacheRegistry:
    """One provider's cache table. Duplicate registration names raise `ValueError`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[Callable[[], Mapping[str, int]], Callable[[], None]]] = {}

    def register(
        self,
        name: str,
        *,
        stats: Callable[[], Mapping[str, int]],
        clear: Callable[[], None],
    ) -> Registration:
        with self._lock:
            if name in self._entries:
                raise ValueError(f"a cache named {name!r} is already registered")
            self._entries[name] = (stats, clear)
        return Registration(registry=self, name=name)

    def registered_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._entries))

    def clear_all(self) -> None:
        """Clear a snapshot of registrations outside the lock.

        Log callback errors and continue with the remaining caches."""
        with self._lock:
            entries = [(name, clear) for name, (_stats, clear) in self._entries.items()]
        for name, clear in entries:
            try:
                clear()
            except Exception:
                logger.exception("graph generation: clearing cache %r failed", name)

    def stats_summary(self) -> dict[str, Mapping[str, int]]:
        with self._lock:
            items = list(self._entries.items())
        return {name: stats() for name, (stats, _clear) in items}

    def _remove(self, name: str) -> None:
        with self._lock:
            self._entries.pop(name, None)
