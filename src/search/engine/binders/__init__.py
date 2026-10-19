"""The binders the engine reads a part's quantity and elects its shape through
(`base.Binder`), and the one `Settings.binder` routes a request to."""

from __future__ import annotations

import hashlib

from search.engine.binders.base import (
    ApplyPolicies,
    Binder,
    BindRequest,
    BindResult,
    QuantityRequest,
)
from search.engine.binders.fallback import FallbackBinder, Fallbacks, recording_fallbacks
from search.engine.binders.gemini import GeminiBinder
from search.engine.binders.jev import JevBinder
from search.engine.binders.part_log import DirectBinder
from search.engine.binders.shadow import ShadowBinder
from search.engine.config import settings
from search.engine.extraction_store import normalise_query


def configured_binder(query: str | None = None) -> Binder:
    """The binder `Settings.binder` (``SEARCH_BINDER``) routes the request asking `query`
    to: Gemini for ``gemini``; for ``jev``, Jev with Gemini as its fallback when
    `routes_to_jev` picks the query, else Gemini; for ``shadow``, Gemini with Jev beside
    it on `Settings.jev_shadow_sample` of calls, or Gemini again on
    `Settings.jev_shadow_control` of those."""
    config = settings()
    match config.binder:
        case "gemini":
            return DirectBinder(GeminiBinder())
        case "jev" if routes_to_jev(query, config.jev_traffic_pct):
            return FallbackBinder(primary=JevBinder(), fallback=GeminiBinder())
        case "jev":
            return DirectBinder(GeminiBinder())
        case "shadow":
            return ShadowBinder(
                served=GeminiBinder(),
                shadow=JevBinder(),
                sample=config.jev_shadow_sample,
                control=config.jev_shadow_control,
            )


def routes_to_jev(query: str | None, percent: int) -> bool:
    """Whether `query` falls in the `percent` share of queries Jev serves: the bucket a hash
    of its normalised form (`extraction_store.normalise_query`) falls in, of 100, is below
    `percent`, so a repeated query always takes the same binder. A request with no query
    text goes to Jev only when every query does."""
    if percent >= 100:
        return True
    if query is None or percent <= 0:
        return False
    digest = hashlib.sha256(normalise_query(query).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100 < percent


__all__ = [
    "ApplyPolicies",
    "BindRequest",
    "BindResult",
    "Binder",
    "DirectBinder",
    "FallbackBinder",
    "Fallbacks",
    "GeminiBinder",
    "JevBinder",
    "QuantityRequest",
    "ShadowBinder",
    "configured_binder",
    "recording_fallbacks",
    "routes_to_jev",
]
