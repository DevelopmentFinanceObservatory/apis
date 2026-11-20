"""Mint, definition, snapshot and download responses carry `no-store`, so a cache never
serves a listing or a signed redirect for an artifact revoked or expired since. Meta responses
use `public, max-age=300`. Pin responses carry only `PinnedCoordinates` and bypass this helper.

`GET /api/datasets/{id}` answers 200 and 304 with `no-cache` instead (`head_answer_headers`): a
cache may store the answer and must revalidate it, by the strong `ETag` the route sets over the
body and its `x-dfo-freshness`. Its 202 and every refusal keep `no-store`.
"""

from __future__ import annotations

import hashlib

NO_STORE_HEADERS = {"cache-control": "no-store"}

# `GET /api/datasets/{id}`'s 200 and 304.
NO_CACHE = "no-cache"

FRESHNESS_HEADER = "x-dfo-freshness"
"""Set on `GET /api/datasets/{id}`'s 200 and 304: `current` when the served head was confirmed
inside the recheck window, `pending` past it while its check is pending or running, and `stale`
past it while no check runs (`datasets_api.operations.head_check`)."""

CONTENT_VERSION_HEADER = "x-dfo-content-version"
"""Set on `GET /api/datasets/{id}`'s 200 and 304: the served content version. The website
gateway reads it for its cache tag and delivery line, since a 304 has no body, and does not
relay it."""


def no_store_headers(*, server_timing: str) -> dict[str, str]:
    """A fresh dict holding `NO_STORE_HEADERS` plus a `Server-Timing` value."""
    return {**NO_STORE_HEADERS, "Server-Timing": server_timing}


def head_answer_etag(body: bytes, *, freshness: str) -> str:
    """The strong ETag of a `GET /api/datasets/{id}` answer: the first 32 hex characters of the
    SHA-256 of the serialised `body` and its `freshness`, quoted. The freshness changes while
    the body stays, so a revalidation of a `pending` answer never keeps `pending` once its check
    has run."""
    digest = hashlib.sha256(body + b"\n" + freshness.encode()).hexdigest()[:32]
    return f'"{digest}"'


def etag_matches(if_none_match: str | None, etag: str) -> bool:
    """Whether `If-None-Match` names `etag` or `*`, compared weakly as RFC 9110 says for it,
    since the gzip middleware sends a large body's strong ETag as weak."""
    if if_none_match is None:
        return False
    tags = {tag.strip().removeprefix("W/") for tag in if_none_match.split(",")}
    return "*" in tags or etag in tags


def head_answer_headers(
    *, server_timing: str, etag: str, freshness: str, content_version: str
) -> dict[str, str]:
    """The headers of `GET /api/datasets/{id}`'s 200 and 304."""
    return {
        "cache-control": NO_CACHE,
        "ETag": etag,
        FRESHNESS_HEADER: freshness,
        CONTENT_VERSION_HEADER: content_version,
        "Server-Timing": server_timing,
    }


ARTIFACT_TIER_HEADER = "x-dfo-artifact-tier"
"""Set on the digest-file redirect: `pinned` when the stored object sits in the pinned tier,
`working` otherwise. Pinned bytes stay at their digest while the version is not revoked, and a
revocation must purge the website gateway's cache by the content-version tag it sets
(`Cache-Tag: <content_version>`)."""
