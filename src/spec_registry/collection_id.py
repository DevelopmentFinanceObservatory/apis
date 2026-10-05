"""Mint `col_` ids, always server-side and never supplied on the register door, from JCS bytes for `{"identity_version": "col1", "members": [...]}`.
Members are NFC-normalised, sorted, and deduplicated before hashing. Ids use 32 SHA-256 hex
characters. The registry owns this projection because the spec-kit helper mints only `spec_` ids.
"""

from __future__ import annotations

import hashlib
import unicodedata

from dfo_spec_kit import identity

from spec_registry.errors import EmptyCollectionMembers

_IDENTITY_VERSION = "col1"
_PREFIX = "col_"
_DIGEST_HEX_LENGTH = 32


def sorted_unique_member_ids(member_spec_ids: list[str]) -> list[str]:
    """NFC-normalise, deduplicate, and sort member ids by Unicode code point."""
    return sorted({unicodedata.normalize("NFC", spec_id) for spec_id in member_spec_ids})


def collection_id_projection(member_spec_ids: list[str]) -> dict:
    """Build the col1 projection from sorted, unique members.

    Raise `EmptyCollectionMembers` when the input is empty.
    """
    members = sorted_unique_member_ids(member_spec_ids)
    if not members:
        raise EmptyCollectionMembers("a collection must name at least one member spec_id")
    return {"identity_version": _IDENTITY_VERSION, "members": members}


def canonical_collection_bytes(member_spec_ids: list[str]) -> bytes:
    """Return canonical JCS bytes for the col1 projection."""
    return identity.canonicalize(collection_id_projection(member_spec_ids))


def mint_collection_id(member_spec_ids: list[str]) -> str:
    """Mint a `col_` id from the canonical col1 projection."""
    canonical = canonical_collection_bytes(member_spec_ids)
    digest = hashlib.sha256(canonical).hexdigest()
    return f"{_PREFIX}{digest[:_DIGEST_HEX_LENGTH]}"
