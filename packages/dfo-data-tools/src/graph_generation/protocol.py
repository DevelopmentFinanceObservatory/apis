"""Structural contract for graph-generation providers used by caches."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from graph_generation.registry import RegistrationHandle
from graph_generation.stamp import GenerationStamp


class CatalogueBuilds(Protocol):
    """The origin catalogue a process serves, as a generation provider reads it."""

    def request_build_id(self) -> str:
        """The build id of the catalogue bound to the calling context, or of the loaded one
        where none is bound. Does no I/O."""
        ...

    def on_swap(self, callback: Callable[[], None]) -> None:
        """Run `callback` after each change of the loaded build."""
        ...


class GenerationProvider(Protocol):
    def stamp(self) -> GenerationStamp:
        """Return the cache key and lifetime snapshot without I/O.

        Cache callers invoke this outside their locks."""
        ...

    def register(
        self,
        name: str,
        *,
        stats: Callable[[], Mapping[str, int]],
        clear: Callable[[], None],
    ) -> RegistrationHandle:
        """Register a cache. `clear` runs on token changes, and `stats` feeds summary logs.

        The returned handle unregisters it on `close()`."""
        ...

    async def refresh_if_stale(self, *, timeout_s: float = 1.0) -> None:
        """Refresh a stale stamp before serving a cache entry.

        Providers without live sources return immediately. Live implementations bound the wait by
        `timeout_s`."""
        ...
