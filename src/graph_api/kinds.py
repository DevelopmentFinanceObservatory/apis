"""The closed kind list the Graph API serves, and the rule that gives a node its kind.

Resolve and the node route share both: the list is `entity_resolution.types.Kind`, and the rule
`entity_resolution.kinds.classify`, fed the code index and the membership policy on both
routes, so a node has one kind whichever route serves it.
"""

from __future__ import annotations

from code_index.query import CodeIndex
from code_index.types import CodeSystemDeclaration
from entity_resolution.kinds import classify
from entity_resolution.types import KINDS, PLACE_KINDS, Kind, PlaceKind

__all__ = ["KINDS", "PLACE_KINDS", "Kind", "PlaceKind", "classify", "code_system_of"]


def code_system_of(index: CodeIndex, dcid: str) -> CodeSystemDeclaration | None:
    """The code system whose graph id is `dcid`, or `None`."""
    return next((d for d in index.declarations.values() if d.enum_dcid == dcid), None)
