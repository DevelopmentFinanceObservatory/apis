"""Resolve text to graph-confirmed entities for Search and the graph API.

Resolvekit candidates pass a listed-name gate and graph confirmation. Misses use description
arcs. Group results carry confirmed members. Results align with input texts, and callers own
role-admission policy. Surfaces share a reference-counted resolvekit resolver."""

from __future__ import annotations

from entity_resolution.admission import Admission
from entity_resolution.resolver import EntityResolutionError, EntityResolver
from entity_resolution.shared import MODULE_IDS, SharedResolver, acquire_shared, release_shared
from entity_resolution.types import (
    ADMITTED_TYPES,
    Ambiguous,
    Entity,
    EntityKind,
    GraphLookup,
    Group,
    NotFound,
    NotHeld,
    Resolution,
    Resolved,
)

__all__ = [
    "ADMITTED_TYPES",
    "MODULE_IDS",
    "Admission",
    "Ambiguous",
    "Entity",
    "EntityKind",
    "EntityResolutionError",
    "EntityResolver",
    "GraphLookup",
    "Group",
    "NotFound",
    "NotHeld",
    "Resolution",
    "Resolved",
    "SharedResolver",
    "acquire_shared",
    "release_shared",
]
