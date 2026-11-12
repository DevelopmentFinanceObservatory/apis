"""Shared StatVar family digest used by search and datasets APIs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

import rfc8785


def member_digest(dcids: Iterable[str]) -> str:
    """Return lowercase SHA-256 hex of RFC 8785 canonical JSON for sorted bare dcids.

    Callers strip the `dcid:` prefix. This function hashes every supplied list.
    Callers handle `None` for absent or inexact populations."""
    canonical = rfc8785.dumps(sorted(dcids))
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class MemberFamily:
    """The whole population an identity admits, before antichain reduction.

    `digest` is `member_digest` over the bare dcids of every admitted member and `count`
    their number. `resolved_stat_vars` lists the members served, which leave out each
    admitted member a broader one dominates.
    """

    digest: str
    count: int


def member_family_of(dcids: Iterable[str]) -> MemberFamily:
    """State the whole admitted population `dcids` (bare dcids, each once) by its digest
    and count."""
    members = frozenset(dcids)
    return MemberFamily(digest=member_digest(members), count=len(members))
