"""Settable `GenerationProvider` test double. Defaults to unknown. Pass `token` to start known."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from graph_generation.registry import CacheRegistry, RegistrationHandle
from graph_generation.stamp import GenerationStamp


class FakeGeneration:
    """Settable generation provider with cache invalidation on token changes.

    `advance()` clears caches on changes to a known token. First assignment and same-token
    recovery leave caches intact. `set_unknown()` retains the last known token, and
    `refresh_if_stale()` is a no-op."""

    def __init__(self, *, token: str | None = None) -> None:
        self._token = token
        self._last_known_token = token
        self._registry = CacheRegistry()

    def stamp(self) -> GenerationStamp:
        return GenerationStamp(token=self._token)

    def register(
        self,
        name: str,
        *,
        stats: Callable[[], Mapping[str, int]],
        clear: Callable[[], None],
    ) -> RegistrationHandle:
        return self._registry.register(name, stats=stats, clear=clear)

    def registered_names(self) -> tuple[str, ...]:
        return self._registry.registered_names()

    async def refresh_if_stale(self, *, timeout_s: float = 1.0) -> None:
        return None

    def advance(self, new_token: str) -> None:
        """Set the token and clear caches when it differs from the last known token."""
        changed = self._last_known_token is not None and self._last_known_token != new_token
        self._token = new_token
        self._last_known_token = new_token
        if changed:
            self._registry.clear_all()

    def set_unknown(self) -> None:
        """Set the current token to unknown without clearing caches.

        Retain the last known token for same-token recovery."""
        self._token = None
