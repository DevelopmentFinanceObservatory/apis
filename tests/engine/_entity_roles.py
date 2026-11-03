"""Builders for the tests of the entity roles the raw-text door derives."""

from __future__ import annotations

from search.engine.entry.raw_text import _prepare_extraction
from search.engine.extract import (
    ExtractedEntity,
    Extraction,
    ExtractionResult,
    _ground_roles,
)
from search.engine.role_text import DoorRole, Party


def linked(
    text: str,
    role: Party | None = None,
    cue: str | None = None,
    *,
    parts: tuple[int, ...] = (),
    span: str | None = None,
    members: bool = False,
) -> ExtractedEntity:
    """An extracted place linked to the measures at `parts` (`ExtractedEntity.variables`)."""
    return ExtractedEntity(
        text=text,
        span=span or text,
        variables=list(parts),
        cue=cue,
        role=role,
        members=members,
    )


def part_roles(
    entities: list[ExtractedEntity], *, query: str, measures: list[str]
) -> list[dict[str, list[DoorRole]]]:
    """The raw-text door's roles of each part for scripted `entities`, grounded as `extract`
    grounds them (`_ground_roles`) and read as `_prepare_extraction` reads an extraction of
    `measures`."""
    grounded = _ground_roles(query, entities, measures)
    extraction = Extraction(kind="resolution", variables=measures, entities=grounded)
    result = ExtractionResult(extraction=extraction, qualifiers={})
    return _prepare_extraction(extraction, result, query=query).per_variable_entity_roles
