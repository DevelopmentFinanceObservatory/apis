"""Mints deterministic series IDs from coordinates, stable across frame refreshes.

An id is `s_` followed by a prefix of the SHA-256 digest of every coordinate. The
prefix is 16 hex characters and grows by two for each extra series that shares it,
up to 62, so no id exceeds the 64 characters the spec-kit schemas allow."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

__all__ = ["Coordinates", "mint_series_id", "mint_series_ids"]

Coordinates = tuple[tuple[str, str | None], ...]

_PREFIX = "s_"
_BASE_TAIL_LENGTH = 16
_MAX_TAIL_LENGTH = 62


def _digest(coordinates: Coordinates) -> str:
    """SHA-256 of dimension-sorted JSON pairs, distinguishing `None` from the string `"null"`."""
    ordered = sorted(coordinates, key=lambda pair: pair[0])
    preimage = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def mint_series_id(*, coordinates: Coordinates) -> str:
    """Builds an id from the digest of every coordinate.

    Identical coordinates yield the same id across frames and content versions."""
    return f"{_PREFIX}{_digest(coordinates)[:_BASE_TAIL_LENGTH]}"


def mint_series_ids(*, coordinates_list: Sequence[Coordinates]) -> tuple[str, ...]:
    """Mints IDs for one response, extending every colliding tail in full-digest order.

    Collision handling is invariant to input order. A group of colliding series
    extends its tails by two hex characters per rank, capped at 62."""
    digests = [_digest(coordinates) for coordinates in coordinates_list]
    groups: dict[str, list[int]] = {}
    for index, digest in enumerate(digests):
        groups.setdefault(digest[:_BASE_TAIL_LENGTH], []).append(index)

    resolved = [""] * len(digests)
    for indices in groups.values():
        ranked = sorted(indices, key=lambda i: digests[i])
        for rank, index in enumerate(ranked):
            tail_length = min(_BASE_TAIL_LENGTH + rank * 2, _MAX_TAIL_LENGTH)
            resolved[index] = f"{_PREFIX}{digests[index][:tail_length]}"
    return tuple(resolved)
