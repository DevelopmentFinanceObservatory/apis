"""Synchronous, HTTP-free registry for canonical spec definitions.
It recomputes identities, stores definitions transactionally, and quarantines collisions.
Callers crossing an async boundary run registry calls in a worker thread.
"""

from __future__ import annotations

from spec_registry.errors import (
    IdentityCollision,
    IdentityQuarantined,
    SpecIdMismatch,
    SpecRegistryError,
    SpecRegistryStoreError,
    SpecRegistryUnavailable,
)
from spec_registry.identity import RecomputedIdentity, recompute_identity
from spec_registry.reader import LiveSpecReader, SpecReader, StoredDefinition
from spec_registry.registry import (
    IDENTITY_COLLISION_QUARANTINED_MARKER,
    LiveSpecRegistry,
    RegisteredDefinition,
    SpecRegistry,
)

__all__ = [
    "IDENTITY_COLLISION_QUARANTINED_MARKER",
    "IdentityCollision",
    "IdentityQuarantined",
    "LiveSpecReader",
    "LiveSpecRegistry",
    "RecomputedIdentity",
    "RegisteredDefinition",
    "SpecIdMismatch",
    "SpecReader",
    "SpecRegistry",
    "SpecRegistryError",
    "SpecRegistryStoreError",
    "SpecRegistryUnavailable",
    "StoredDefinition",
    "recompute_identity",
]
