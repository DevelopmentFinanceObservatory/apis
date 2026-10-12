"""Mint, definition, snapshot and download responses carry `no-store`, so a cache never
serves a listing or a signed redirect for an artifact revoked or expired since. Meta responses
use `public, max-age=300`. Pin responses carry only `PinnedCoordinates` and bypass this helper.
"""

from __future__ import annotations

NO_STORE_HEADERS = {"cache-control": "no-store"}


def no_store_headers(*, server_timing: str) -> dict[str, str]:
    """A fresh dict holding `NO_STORE_HEADERS` plus a `Server-Timing` value."""
    return {**NO_STORE_HEADERS, "Server-Timing": server_timing}
