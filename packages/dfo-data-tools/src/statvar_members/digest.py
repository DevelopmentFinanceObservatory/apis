"""Shared StatVar family digest used by search and datasets APIs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import rfc8785


def member_digest(dcids: Iterable[str]) -> str:
    """Return lowercase SHA-256 hex of RFC 8785 canonical JSON for sorted bare dcids.

    Callers strip the `dcid:` prefix. This function hashes every supplied list.
    Callers handle `None` for absent or inexact populations."""
    canonical = rfc8785.dumps(sorted(dcids))
    return hashlib.sha256(canonical).hexdigest()
