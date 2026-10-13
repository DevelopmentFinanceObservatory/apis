"""Description-arc fallback for misses from resolvekit or dcid mapping.

Try Country, Continent, UNGeoRegion, then unrestricted description arcs. Accept candidates only
when their normalized `name` or `alternateName` lists the query. Keep every match for
ambiguity handling. Description arcs avoid unrelated `name` matches from
`SDG_CounterpartEnum`, `Source`, `DebtFinancerEnum`, and EIA facilities.
Each rung batches remaining queries into one `resolve_by_arc` and one `node_names_many` call."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from resolvekit.core.util import NormalizationError

from entity_resolution.types import GraphLookup

_LADDER_ARCS: tuple[str, ...] = (
    "<-description{typeOf:Country}->dcid",
    "<-description{typeOf:Continent}->dcid",
    "<-description{typeOf:UNGeoRegion}->dcid",
    "<-description->dcid",
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
        names = list(dict.fromkeys(remaining.values()))
        answers = graph.resolve_by_arc(names, arc=arc, deadline=deadline)
        answered_dcids = dict.fromkeys(
            dcid for text in remaining.values() for dcid in answers.get(text) or []
        )
        if not answered_dcids:
            continue
        node_names = graph.node_names_many(list(answered_dcids), deadline=deadline)
        distinct_strings = set(names) | {
            name for names_for_dcid in node_names.values() for name in names_for_dcid
        }
        normalized = {value: normalize_or_none(normalize, value) for value in distinct_strings}
        normalized_names_by_dcid = {
            dcid: {normalized[name] for name in names_for_dcid if normalized[name] is not None}
            for dcid, names_for_dcid in node_names.items()
        }
        still_remaining: dict[int, str] = {}
        for index, text in remaining.items():
            normalized_text = normalized[text]
            listed = tuple(
                dcid
                for dcid in answers.get(text) or []
                if normalized_text is not None
                and normalized_text in normalized_names_by_dcid.get(dcid, set())
            )
            if listed:
                found[index] = listed
            else:
                still_remaining[index] = text
        remaining = still_remaining
    return found


def normalize_or_none(normalize: Callable[[str], str], text: str) -> str | None:
    """Return the normalized text, or `None` when normalization raises `NormalizationError`."""
    try:
        return normalize(text)
    except NormalizationError:
        return None
