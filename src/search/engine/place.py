"""Expand `Continent` and `UNGeoRegion` place sets to member countries.

Graph reads are synchronous. Async callers use `asyncio.to_thread`. The engine expands top-level
member countries, while interactive pickers can add deeper drill-downs. A query that names no
place takes the default place scope the datasets mint takes (`materialise.place_scope`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from materialise import place_scope
from materialise.place_scope import (
    DefaultPlaceScope,
    DefaultPresence,
    DefaultScopeUnknownError,
    PlaceScopeReadError,
    is_worldwide,
)
from search.engine.errors import GraphReadFailure

if TYPE_CHECKING:
    from search.engine.clients.graph import EngineGraphClient
    from search.engine.clients.sdmx import EngineSdmxClient

# Map every resolved place-set graph type here to support expansion.
_CHILD_PLACE_TYPES: dict[str, str] = {
    "Continent": "Country",
    "UNGeoRegion": "Country",
}


@dataclass
class PlaceExpansion:
    """Every member dcid of a place set, in graph order, and the graph type they share. A
    covered set with no members returns an empty expansion. An unsupported type returns `None`.
    """

    children: list[str]
    child_type: str


def expand_places(
    resolved: list[tuple[str, str]],
    *,
    graph: EngineGraphClient,
) -> dict[str, PlaceExpansion | None]:
    """Expand resolved `(place_dcid, type_of)` pairs, grouping them by child type for one graph call
    per type. Return one entry per input dcid. Unsupported types map to `None`. Supported sets with
    no recorded members return an empty `PlaceExpansion`.
    """
    by_child_type: dict[str, list[str]] = {}
    for dcid, type_of in resolved:
        child_type = _CHILD_PLACE_TYPES.get(type_of)
        if child_type is not None:
            by_child_type.setdefault(child_type, []).append(dcid)

    results: dict[str, PlaceExpansion | None] = {dcid: None for dcid, _ in resolved}
    for child_type, dcids in by_child_type.items():
        children_by_dcid = graph.child_places_batch(dcids, child_type)
        for dcid in dcids:
            results[dcid] = PlaceExpansion(
                children=children_by_dcid.get(dcid, []), child_type=child_type
            )
    return results


def expand_place(
    dcid: str,
    type_of: str,
    *,
    graph: EngineGraphClient,
) -> PlaceExpansion | None:
    """Expand one place-set through one `child_places_batch` call. Unsupported types return
    `None`. Empty supported sets return an empty expansion.
    """
    return expand_places([(dcid, type_of)], graph=graph).get(dcid)


def is_worldwide_scope(place_dcids: Sequence[str] | None) -> bool:
    """Check whether the place scope is Earth alone, a worldwide total. A single named country has
    a different dcid, so callers can distinguish worldwide coverage from a bound place.
    """
    return place_dcids is not None and is_worldwide(place_dcids)


def default_place_scope_message(scope: DefaultPlaceScope) -> str:
    """Return the shared `PLACE_SCOPE_DEFAULTED` message: a worldwide total for Earth alone,
    every place with data, or the countries with data when any StatVar's full list was too
    large or unavailable."""
    if is_worldwide(scope.places):
        return "no place was named; the answer was resolved as a worldwide total."
    if scope.every_place:
        return "no place was named; the answer covers every place with data, at every level."
    return (
        "no place was named; the answer covers the countries with data, because the full list "
        "of places was too large or unavailable."
    )


def read_default_presence(
    *,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    stat_vars: Iterable[str],
    known: DefaultPresence | None = None,
    deadline: float | None = None,
) -> DefaultPresence:
    """The default place scopes (`materialise.place_scope.read_default_presence`) of
    `stat_vars` over Search's graph client, under the request's `deadline`. `sdmx` reads our
    variables' availability; `graph` reads the UN deployment's.

    An availability error the countries fallback does not cover
    (`place_scope.PlaceScopeReadError`), and a listing of Earth that names no country, raise
    `GraphReadFailure` with the upstream status, so Search refuses either as a retryable
    infrastructure failure. The read may be one the datasets door's
    reader started, so the error carries no SDMX client's type. A read the deadline
    ends raises `place_scope.PlaceScopeDeadlineError`, a `TimeoutError`, which Search refuses
    as a timeout."""
    try:
        return place_scope.read_default_presence(
            graph=graph, availability=sdmx, stat_vars=stat_vars, known=known, deadline=deadline
        )
    except (DefaultScopeUnknownError, PlaceScopeReadError) as exc:
        raise GraphReadFailure(str(exc), upstream_status=exc.upstream_status) from exc


def earth_countries(*, graph: EngineGraphClient, deadline: float | None = None) -> DefaultPresence:
    """Earth's countries (`materialise.place_scope.earth_countries`) as a presence that covers
    no StatVar yet, which a later `read_default_presence` takes as `known`. A listing that
    names no country raises `GraphReadFailure`."""
    try:
        countries = place_scope.earth_countries(graph, deadline=deadline)
    except DefaultScopeUnknownError as exc:
        raise GraphReadFailure(str(exc)) from exc
    return DefaultPresence(countries=countries, scopes={})
