"""Entity resolution through the query words that name each entity.

The raw-text door keys every entity by its canonical English `text` and may know the
query words that name it (`ExtractedEntity.span`). A surface is such a span, recorded
only when it occurs in the query. `resolve_names` resolves each name through its surface
and settles between the surface's answer and the name's own:

- a qualified surface ("Asia (DAC)" for the name "Asia") is the only text resolved, so a
  named aggregate never widens to the bare place;
- otherwise the surface and the name are resolved in one batch. The surface's answer
  stands, except that a `NotFound` or `NotHeld` surface takes the name's answer, and an
  `Ambiguous` surface takes the name's answer when that is one of its candidates
  ("Congo" read as "Democratic Republic of the Congo").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from entity_resolution import Ambiguous, EntityResolver, NotFound, NotHeld, Resolution, Resolved


def _is_qualified(surface: str, name: str) -> bool:
    """Whether `surface` carries a parenthetical qualifier `name` lacks."""
    return "(" in surface and "(" not in name


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
    if isinstance(through_surface, NotFound):
        return own
    if isinstance(through_surface, NotHeld):
        return through_surface if isinstance(own, NotFound) else own
    if (
        isinstance(through_surface, Ambiguous)
        and isinstance(own, Resolved)
        and own.entity.dcid in {candidate.dcid for candidate in through_surface.candidates}
    ):
        return own
    return through_surface


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
