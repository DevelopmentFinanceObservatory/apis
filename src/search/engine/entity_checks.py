"""The parsed door's synchronous checks of one part's entities, and what its leg reads of
them. The parsed door runs them on every posted part, and an entity choice
(`conjoin._option`) runs them on every option it offers, so an option the door would
refuse is not offered."""

from __future__ import annotations

import dataclasses
from typing import NoReturn

from search.engine.errors import EngineInputError
from search.engine.qualifier_match import _normalize
from search.engine.role_text import role_axis_for_text
from search.engine.surface_resolution import is_surface
from search.models import ParsedPart, RoleAxis


def _member_entities_for_part(part: ParsedPart, *, pointer_prefix: str) -> frozenset[str]:
    """The identities of the entities any row of the part asks for as their members
    (`ParsedEntity.members`).

    One part names a set once on each flow end its rows' `role_text` gives, so an entity
    posted with both values on one end, or with both values and a row on no end, raises
    `request_invalid` at the part's entities. An entity can be asked for as its members on
    one end and as a whole on the other.
    """
    asked: dict[str, dict[RoleAxis | None, bool]] = {}
    for entity in part.entities or ():
        ends = asked.setdefault(entity.identity, {})
        members = bool(entity.members)
        if ends.setdefault(role_axis_for_text(entity.role_text), members) != members:
            _refuse_both_readings(entity.identity, pointer_prefix=pointer_prefix)
    for identity, ends in asked.items():
        if None in ends and len(set(ends.values())) > 1:
            _refuse_both_readings(identity, pointer_prefix=pointer_prefix)
    return frozenset(identity for identity, ends in asked.items() if any(ends.values()))


def _refuse_both_readings(identity: str, *, pointer_prefix: str) -> NoReturn:
    """Raise `request_invalid` at the part's entities for `identity` posted with both
    `members` values on one end."""
    detail = (
        f"part posts entity {identity!r} with members both true and false on one end; one "
        "entity names one set on each end of a single part, as its members or as a whole"
    )
    raise EngineInputError(detail, code="request_invalid", pointer=f"{pointer_prefix}/entities")


def _check_no_name_is_a_pin(part: ParsedPart, *, pointer_prefix: str) -> None:
    """Refuse an unpinned row whose identity, its name or its text, is a dcid another row of
    the part pins, since the two would bind as one entity: `request_invalid` at the
    unpinned row."""
    entities = part.entities or ()
    pinned = {entity.dcid for entity in entities if entity.dcid is not None}
    for index, entity in enumerate(entities):
        if entity.dcid is None and entity.identity in pinned:
            detail = (
                f"part posts entity {entity.identity!r} unpinned beside a row pinning the "
                "dcid it names; a name or text that is a pinned dcid of the part is invalid"
            )
            raise EngineInputError(
                detail,
                code="request_invalid",
                pointer=f"{pointer_prefix}/entities/{index}",
            )


def _entity_surfaces_for_part(part: ParsedPart, *, pointer_prefix: str) -> dict[str, str]:
    """Each unpinned entity's identity mapped to its text, where the text names it in other
    words (`surface_resolution.is_surface`): the words its resolution reads beside the
    name.

    A pinned row resolves nothing, so its text is display only and stays out. An unpinned
    entity resolves through one text, so one posted under two texts that are not the same
    words once normalised (`qualifier_match._normalize`) raises `request_invalid` at the
    part's entities.
    """
    texts: dict[str, str] = {}
    for entity in part.entities or ():
        if entity.dcid is not None:
            continue
        text = texts.setdefault(entity.identity, entity.text)
        if _normalize(text) != _normalize(entity.text):
            detail = (
                f"part posts entity {entity.identity!r} under two texts; one entity resolves "
                "through one text within a single part"
            )
            raise EngineInputError(
                detail,
                code="request_invalid",
                pointer=f"{pointer_prefix}/entities",
            )
    return {identity: text for identity, text in texts.items() if is_surface(text, identity)}


def _entity_words_for_part(part: ParsedPart) -> dict[str, str]:
    """Each entity's identity mapped to the request's own words for it, where they differ:
    the text of the first row holding that identity. Bind masks them in the variable phrase
    (`ResolveState.entity_texts`) and refusals quote them (`RegionResult.entity_words`)."""
    words: dict[str, str] = {}
    for entity in part.entities or ():
        words.setdefault(entity.identity, entity.text)
    return {identity: text for identity, text in words.items() if identity != text}


@dataclasses.dataclass(frozen=True, slots=True)
class PartEntities:
    """What one part's entities hand its fan-out leg beside their identities."""

    member_entities: frozenset[str]
    surfaces: dict[str, str]
    words: dict[str, str]


def part_entities(part: ParsedPart, *, pointer_prefix: str) -> PartEntities:
    """Check one part's entities as the parsed door does and read what its leg needs of
    them: the identities asked for as members (`_member_entities_for_part`), the texts
    resolution reads (`_entity_surfaces_for_part`), and the request's words for each
    (`_entity_words_for_part`).

    Raises `EngineInputError` with `request_invalid` at `pointer_prefix` for entities the
    door refuses, an unpinned row whose identity another row pins among them
    (`_check_no_name_is_a_pin`), so a caller building a part checks it as the door would.
    """
    _check_no_name_is_a_pin(part, pointer_prefix=pointer_prefix)
    return PartEntities(
        member_entities=_member_entities_for_part(part, pointer_prefix=pointer_prefix),
        surfaces=_entity_surfaces_for_part(part, pointer_prefix=pointer_prefix),
        words=_entity_words_for_part(part),
    )
