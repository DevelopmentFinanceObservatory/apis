"""Synchronous, HTTP-free registry for canonical spec and collection definitions.
It recomputes identities, stores definitions transactionally, and quarantines collisions.
Callers crossing an async boundary run registry calls in a worker thread.
"""

from __future__ import annotations

from spec_registry.collection_id import (
    canonical_collection_bytes,
    collection_id_projection,
    mint_collection_id,
    sorted_unique_member_ids,
)
from spec_registry.errors import (
    EmptyCollectionMembers,
    IdentityCollision,
    IdentityQuarantined,
    SpecIdMismatch,
    SpecRegistryError,
    SpecRegistryStoreError,
    SpecRegistryUnavailable,
)
from spec_registry.identity import RecomputedIdentity, recompute_identity
from spec_registry.reader import LiveSpecReader, SpecReader, StoredCollection, StoredDefinition
from spec_registry.registry import (
    IDENTITY_COLLISION_QUARANTINED_MARKER,
    LiveSpecRegistry,
    MemberDefinition,
    RegisteredCollection,
    RegisteredDefinition,
    SpecRegistry,
)

__all__ = [
    "IDENTITY_COLLISION_QUARANTINED_MARKER",
    "EmptyCollectionMembers",
    "IdentityCollision",
    "IdentityQuarantined",
    "LiveSpecReader",
    "LiveSpecRegistry",
    "MemberDefinition",
    "RecomputedIdentity",
    "RegisteredCollection",
    "RegisteredDefinition",
    "SpecIdMismatch",
    "SpecReader",
    "SpecRegistry",
    "SpecRegistryError",
    "SpecRegistryStoreError",
    "SpecRegistryUnavailable",
    "StoredCollection",
    "StoredDefinition",
    "canonical_collection_bytes",
    "collection_id_projection",
    "mint_collection_id",
    "recompute_identity",
    "sorted_unique_member_ids",
]
