"""Description-arc lookups: the fallback for misses from resolvekit or dcid mapping, and the
places that share a resolvekit hit's name.

The fallback tries Country, Continent, UNGeoRegion, then unrestricted description arcs. Accept
candidates only when their normalized `name` or `alternateName` lists the query. Keep every
match for ambiguity handling. Description arcs avoid unrelated `name` matches from
`SDG_CounterpartEnum`, `Source`, `DebtFinancerEnum`, and EIA facilities.
Each rung batches remaining queries into one `resolve_by_arc` call, and one `node_names_many`
call when the arc answers any node."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping

from resolvekit.core.util import NormalizationError

from entity_resolution.types import GraphLookup

_DESCRIPTION_ARC = "<-description->dcid"

_LADDER_ARCS: tuple[str, ...] = (
    "<-description{typeOf:Country}->dcid",
    "<-description{typeOf:Continent}->dcid",
    "<-description{typeOf:UNGeoRegion}->dcid",
    _DESCRIPTION_ARC,
)


def resolve_fallback(
    graph: GraphLookup,
    texts: Mapping[int, str],
    *,
    normalize: Callable[[str], str],
    deadline: float | None = None,
) -> dict[int, tuple[str, ...]]:
    """Resolve indexed texts through the fallback ladder and return each match's dcids.

    A rung accepts a text only when an answering node lists it under the supplied normalizer.
    Treat `NormalizationError` as a miss. Normalize each distinct text and graph name once per
    rung. Results preserve each arc's candidate order.

    Pass the absolute monotonic `deadline` unchanged to graph reads. Propagate
    `GraphDeadlineExceededError` so partial rung results cannot look complete."""
    remaining: dict[int, str] = dict(texts)
    found: dict[int, tuple[str, ...]] = {}
    for arc in _LADDER_ARCS:
        if not remaining:
            break
        listed = _listing_nodes(
            graph, arc, remaining, known={}, normalize=normalize, deadline=deadline
        )
        found |= listed
        remaining = {index: text for index, text in remaining.items() if index not in listed}
    return found


def resolve_sharing(
    graph: GraphLookup,
    texts: Mapping[int, str],
    *,
    known: Mapping[int, Collection[str]],
    normalize: Callable[[str], str],
    deadline: float | None = None,
) -> dict[int, tuple[str, ...]]:
    """The other nodes that list each indexed text, beside the dcids `known` holds for it.

    Reads the fallback's last, unrestricted rung, which reaches the places resolvekit's packs
    leave out (the US state `geoId/13` beside the country Georgia). One `resolve_by_arc` call
    covers every text, and one `node_names_many` call follows only when the arc answers a node
    outside `known`. A text with no such node is left out. `deadline` follows
    `resolve_fallback`."""
    return _listing_nodes(
        graph, _DESCRIPTION_ARC, texts, known=known, normalize=normalize, deadline=deadline
    )


def _listing_nodes(
    graph: GraphLookup,
    arc: str,
    texts: Mapping[int, str],
    *,
    known: Mapping[int, Collection[str]],
    normalize: Callable[[str], str],
    deadline: float | None,
) -> dict[int, tuple[str, ...]]:
    """The nodes `arc` answers for each text, outside `known`, whose normalized names list the
    text, in the arc's order. A text with none is left out."""
    names = list(dict.fromkeys(texts.values()))
    answers = graph.resolve_by_arc(names, arc=arc, deadline=deadline)
    answered: dict[int, list[str]] = {
        index: [dcid for dcid in answers.get(text) or [] if dcid not in known.get(index, ())]
        for index, text in texts.items()
    }
    answered_dcids = dict.fromkeys(dcid for dcids in answered.values() for dcid in dcids)
    if not answered_dcids:
        return {}
    node_names = graph.node_names_many(list(answered_dcids), deadline=deadline)
    distinct_strings = set(names) | {
        name for names_for_dcid in node_names.values() for name in names_for_dcid
    }
    normalized = {value: normalize_or_none(normalize, value) for value in distinct_strings}
    normalized_names_by_dcid = {
        dcid: {normalized[name] for name in names_for_dcid if normalized[name] is not None}
        for dcid, names_for_dcid in node_names.items()
    }
    found: dict[int, tuple[str, ...]] = {}
    for index, text in texts.items():
        normalized_text = normalized[text]
        if normalized_text is None:
            continue
        listed = tuple(
            dcid
            for dcid in answered[index]
            if normalized_text in normalized_names_by_dcid.get(dcid, set())
        )
        if listed:
            found[index] = listed
    return found


def normalize_or_none(normalize: Callable[[str], str], text: str) -> str | None:
    """Return the normalized text, or `None` when normalization raises `NormalizationError`."""
    try:
        return normalize(text)
    except NormalizationError:
        return None
