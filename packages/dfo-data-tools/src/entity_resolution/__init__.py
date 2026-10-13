"""Resolve text to graph-confirmed entities for Search and the graph API.

Resolvekit candidates pass a listed-name gate and graph confirmation. Misses use description
arcs. Groups come from the membership registry, which confirms them and their members. A
`kinds` filter keeps the requested kinds. Results align with input texts, and callers own
role-admission policy. Surfaces share a reference-counted resolvekit resolver; the group
registry and membership policy live for the whole process."""

from __future__ import annotations

from entity_resolution.admission import Admission
from entity_resolution.resolver import EntityResolutionError, EntityResolver
from entity_resolution.shared import (
    MODULE_IDS,
    SharedResolver,
    acquire_shared,
    release_shared,
    shared_membership_policy,
)
from entity_resolution.types import (
    ADMITTED_TYPES,
    DEFAULT_KINDS,
    Ambiguous,
    CodeHit,
    CodeLookup,
    Entity,
    EntityKind,
    GraphLookup,
    Group,
    Matched,
    MatchedAs,
    NotFound,
    NotHeld,
    Reading,
    Resolution,
    Resolved,
)

__all__ = [
    "ADMITTED_TYPES",
    "DEFAULT_KINDS",
    "MODULE_IDS",
    "Admission",
    "Ambiguous",
    "CodeHit",
    "CodeLookup",
    "Entity",
    "EntityKind",
    "EntityResolutionError",
    "EntityResolver",
    "GraphLookup",
    "Group",
    "Matched",
    "MatchedAs",
    "NotFound",
    "NotHeld",
    "Reading",
    "Resolution",
    "Resolved",
    "SharedResolver",
    "acquire_shared",
    "release_shared",
    "shared_membership_policy",
]
