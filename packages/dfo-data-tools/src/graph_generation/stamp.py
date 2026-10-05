"""Immutable graph-generation snapshot used for cache keys and lifetimes.
Caches read it once per computation, so stale results remain under stale keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graph_generation.config import (
    GRAPH_CACHE_MAX_AGE_SECONDS,
    GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS,
)

# Reserved cache-key token for an unknown generation.
UNKNOWN_KEY = "unknown"


@dataclass(frozen=True, slots=True)
class GenerationStamp:
    """`token` is the last successfully read Cloud Run revisions, joined with `+`, or `None` if
    unknown or stale past the provider's `stale_after`. `catalogue` is the build id of the
    origin catalogue the stamping request is bound to, or `None` for a provider that serves no
    catalogue."""

    token: str | None
    catalogue: str | None = None

    @property
    def key(self) -> str:
        """`token`, or `UNKNOWN_KEY` when unknown, then `@` and `catalogue` when set. The
        catalogue joins an unknown token too, so requests bound to different builds never share
        an entry."""
        revision = self.token if self.token is not None else UNKNOWN_KEY
        return revision if self.catalogue is None else f"{revision}@{self.catalogue}"

    @property
    def max_age_s(self) -> float:
        """Uses `GRAPH_CACHE_MAX_AGE_SECONDS` when the token is known and
        `GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS` when it is unknown. An unknown token
        gives no invalidation signal, so its lifetime is shorter."""
        return (
            float(GRAPH_CACHE_MAX_AGE_SECONDS)
            if self.token is not None
            else float(GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS)
        )


# Stamp returned by unconfigured providers and providers without a successful read.
UNKNOWN_STAMP = GenerationStamp(token=None)


def ttu_from_stamp_key(key: Any, value: Any, now: float) -> float:
    """`cachetools.TLRUCache` callback for keys `(stamp.key, *original_key)`. Returns the expiry
    of the stamp `key[0]` came from, keeping known/unknown lifetimes consistent across caches: a
    key that is `UNKNOWN_KEY`, or starts with it and `@`, had an unknown token. `Any` supports
    each cache's key and value types."""
    stamp_key = key[0]
    unknown = stamp_key == UNKNOWN_KEY or stamp_key.startswith(f"{UNKNOWN_KEY}@")
    return now + GenerationStamp(token=None if unknown else stamp_key).max_age_s
