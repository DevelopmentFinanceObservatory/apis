"""Signed artifact URLs appear on mint, definition, and snapshot responses.
`no-store` prevents caches from serving expired URLs or links to revoked
artifacts. Meta responses use `public, max-age=300`. Pin responses carry only
`PinnedCoordinates` and bypass this helper.
"""

from __future__ import annotations

NO_STORE_HEADERS = {"cache-control": "no-store"}


def no_store_headers(*, server_timing: str) -> dict[str, str]:
    """A fresh dict holding `NO_STORE_HEADERS` plus a `Server-Timing` value."""
    return {**NO_STORE_HEADERS, "Server-Timing": server_timing}
