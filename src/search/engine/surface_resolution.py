"""Entity resolution through the query words that name each entity.

The raw-text door keys every entity by its canonical English `text` and may know the
query words that name it (`ExtractedEntity.span`). A surface is such a span, recorded
only when it occurs in the query. `resolve_names` resolves each name through its surface
and settles between the surface's answer and the name's own:

- a qualified surface ("Asia (DAC)" for the name "Asia") is the only text resolved, so a
  named aggregate never widens to the bare place;
- otherwise the surface and the name are resolved in one batch, and one leads. A surface
  that adds words to the name ("Nairobi county" for "Nairobi") leads, since the extractor
  dropped words the user wrote. So does a surface the name adds words to ("Congo" for
  "Democratic Republic of the Congo", "Korea" for "South Korea"), since the added words are
  the extractor's and no word of the query. A surface the name rewrites in other words
  ("Canadian" for "Canada", "UK" for "United Kingdom", "Kenia" for "Kenya") follows, since
  the name is the extractor's reading of it; "Canadian" alone matches two US towns. The
  leading answer stands, except that a `NotFound` lead takes the other answer, a `NotHeld`
  lead takes it unless that is `NotFound`, and an `Ambiguous` lead takes it when it is one
  of the lead's candidates. A name that only adds words to its surface settles the
  surface's ambiguity only when it is the one candidate of its kind: "Trinidad and Tobago"
  settles "Trinidad" against a town of that name, and "Federated States of Micronesia"
  settles "Micronesia" against the region, while "Congo" names two countries and the
  query does not choose between them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from entity_resolution import (
    Ambiguous,
    Entity,
    EntityResolver,
    NotFound,
    NotHeld,
    Resolution,
    Resolved,
)


def _is_qualified(surface: str, name: str) -> bool:
    """Whether `surface` carries a parenthetical qualifier `name` lacks."""
    return "(" in surface and "(" not in name


def _adds_words(surface: str, name: str) -> bool:
    """Whether `surface` holds `name` as whole words and more ("Nairobi county", "Nairobi")."""
    folded = surface.casefold()
    return folded != name.casefold() and bool(
        re.search(rf"(?<!\w){re.escape(name.casefold())}(?!\w)", folded)
    )


def _texts(name: str, surface: str | None) -> list[str]:
    """The texts resolved for `name`, its surface first."""
    if surface is None:
        return [name]
    if _is_qualified(surface, name):
        return [surface]
    return [surface, name]


def _settle(name: str, surface: str | None, by_text: Mapping[str, Resolution]) -> Resolution:
    """The resolution `name` takes from its surface's answer and its own."""
    if surface is None:
        return by_text[name]
    through_surface = by_text[surface]
    if _is_qualified(surface, name):
        return through_surface
    own = by_text[name]
    expands = _adds_words(name, surface)
    surface_leads = expands or _adds_words(surface, name)
    lead, other = (through_surface, own) if surface_leads else (own, through_surface)
    if isinstance(lead, NotFound):
        return other
    if isinstance(lead, NotHeld):
        return lead if isinstance(other, NotFound) else other
    if (
        isinstance(lead, Ambiguous)
        and isinstance(other, Resolved)
        and other.entity.dcid in {candidate.dcid for candidate in lead.candidates}
        and (not expands or _sole_of_its_kind(other.entity, lead.candidates))
    ):
        return other
    return lead


def _sole_of_its_kind(entity: Entity, candidates: Sequence[Entity]) -> bool:
    """Whether `entity` is the only one of `candidates` of its kind."""
    return all(c.kind != entity.kind for c in candidates if c.dcid != entity.dcid)


async def resolve_names(
    names: Sequence[str], *, resolver: EntityResolver, surfaces: Mapping[str, str]
) -> list[Resolution]:
    """Resolve `names` in one resolver call, in order, each through its surface.

    `surfaces` maps a name to the query words that name it; a name it omits resolves as
    itself. With no surface among `names`, the resolver receives `names` unchanged.
    """
    if not any(name in surfaces for name in names):
        return await resolver.resolve(list(names))
    texts = list(dict.fromkeys(text for name in names for text in _texts(name, surfaces.get(name))))
    by_text = dict(zip(texts, await resolver.resolve(texts), strict=True))
    return [_settle(name, surfaces.get(name), by_text) for name in names]
