"""Validate and canonicalise registry identity documents through `dfo_spec_kit.identity`.
The adapter also stores the full SHA-256 digest beside the kit-minted `spec_id`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

from dfo_spec_kit import identity

from dfo_platform.roles import role_key_axis

# The adapter receives parsed values, so it enforces the kit's 65,536-byte cap on compact JSON.
_MAX_IDENTITY_DOCUMENT_BYTES = 65_536

# Search's `BaseIdentity.entity_groups` records how a request named a role's entities. It
# stays outside the identity core and the hash (`BaseIdentity.core_document()`), so a Search
# definition posted here verifies against the spec_id Search minted.
_NON_CORE_MEMBERS = frozenset({"entity_groups"})


def _refuse_bound_open_axes(projection: dict[str, object]) -> None:
    """Refuse a projection that binds a role in `entities` on the axis of one of its
    `open_roles`. The kit compares the two by key, and `entities` keys keep the caller's
    spelling, so `{"creditor": [...]}` beside an open `origin` passes it. A role is bound
    or open, and never both."""
    entities = cast("dict[str, object]", projection.get("entities", {}))
    open_roles = cast("list[str]", projection.get("open_roles", []))
    bound_axes = {role_key_axis(role) for role in entities}
    both = [axis for axis in open_roles if axis in bound_axes]
    if both:
        raise identity.IdentityError(
            "schema_violation",
            f"open_roles names {both[0]!r}, an axis entities binds a role on",
        )


@dataclass(frozen=True)
class RecomputedIdentity:
    """Server-recomputed id, canonical identity bytes, and their full SHA-256 digest."""

    spec_id: str
    identity_bytes: bytes
    identity_digest: str

    @property
    def identity(self) -> dict[str, object]:
        """The canonical identity document, parsed from `identity_bytes`.

        The registry stores this document and a reader returns it as
        `StoredDefinition.identity`, so a caller holding an unregistered identity reads the
        same document a registration of it would serve.
        """
        return json.loads(self.identity_bytes)


def recompute_identity(identity_definition: object) -> RecomputedIdentity:
    """Recompute identity from the parsed identity-core document, excluding the wire envelope.

    Members outside the identity core (`_NON_CORE_MEMBERS`) are dropped first, so neither the
    hash nor the returned canonical document carries them.

    Raises the kit's `IdentityError` on schema failure, when compact JSON exceeds 65,536
    bytes, or when `entities` binds a role on the axis of one of `open_roles`
    (`_refuse_bound_open_axes`).
    """
    compact = json.dumps(identity_definition, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    if len(compact) > _MAX_IDENTITY_DOCUMENT_BYTES:
        raise identity.IdentityError(
            "input_limit_exceeded",
            f"identity document is {len(compact)} bytes, exceeding the maximum of "
            f"{_MAX_IDENTITY_DOCUMENT_BYTES}",
        )

    if isinstance(identity_definition, dict):
        identity_definition = {
            key: value for key, value in identity_definition.items() if key not in _NON_CORE_MEMBERS
        }
    projection = identity.build_projection(identity_definition)
    _refuse_bound_open_axes(projection)
    canonical = identity.canonicalize(projection)
    spec_id = identity.mint_spec_id(canonical)
    identity_digest = hashlib.sha256(canonical).hexdigest()
    return RecomputedIdentity(
        spec_id=spec_id, identity_bytes=canonical, identity_digest=identity_digest
    )
