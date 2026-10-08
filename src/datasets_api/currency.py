"""Derive the materialisation currency signals used by mint and definition retrieval.

`derive_signals` reads the facet probe used for the fetch. A probe with no candidate on any
member gives the zero signal, as materialisation gives the empty result. For a multi-member
combined group, it matches each member against the shared group selection and combines every
dispatched facet.
Choosing only one facet could miss a later update or trigger a rebuild for an unchanged facet.
A single-member spec uses its request's own selected facets. Mint and later GETs
therefore compare signals derived by the same rule. A request whose window leaves its end unstated
reads under the current year, so its signal also carries the window it read (`clock_window`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from datasets_api.versions import MemberSignal
from dc_graph import earliest_date as _earliest
from dc_graph import latest_date as _latest
from materialise import CollectionFacetProbe, MaterialisationRequest
from materialise.content_version import recipe_dependency_fingerprint
from materialise.errors import DefaultFacetUndefinedError, FacetNotAvailableError
from materialise.facets import MemberProbe, joint_dispatch_facet_ids
from materialise.generation import _select_member_facets
from materialise.graph import (
    Facet,
    GraphReader,
    ProvenanceRefresh,
)
from materialise.group_facets import group_breadth, select_group_facets
from materialise.recipes import NonCurrencyUnitError
from materialise.window import RequestedWindow, election_window

# Missing or retired recipe fingerprints force a signal mismatch. One rebuild writes a verifiable
# fingerprint, so later currency checks can compare normally.
_UNVERIFIABLE_RECIPE_FINGERPRINT: tuple[str, str] = ("__unverifiable__", "__unverifiable__")

# The signal of a spec that dispatches no facet, because its probe is empty or no candidate
# matches its selection.
_NO_DISPATCH_SIGNAL = MemberSignal(
    dispatch_facet_id=None, obs_count=0, earliest_date=None, latest_date=None
)


def _refresh_dates(
    provenance_id: str | None, provenance_refresh: Mapping[str, ProvenanceRefresh]
) -> tuple[str | None, str | None]:
    """Return the provenance refresh dates from the probe map, or `(None, None)` if absent."""
    if provenance_id is None:
        return None, None
    refresh = provenance_refresh.get(provenance_id)
    if refresh is None:
        return None, None
    return refresh.last_data_refresh_date, refresh.next_data_refresh_date


def _member_signal(
    request: MaterialisationRequest,
    probe: MemberProbe,
    *,
    graph: GraphReader,
    provenance_refresh: Mapping[str, ProvenanceRefresh],
) -> MemberSignal:
    """Build the signal from the facets selected for this member's fetch.

    A probe with no candidate, or no dispatched facet, yields zero counts and no facet id
    (`_NO_DISPATCH_SIGNAL`). Multiple matches sum `obs_count`
    and widen the date range. The lowest `facet_id` represents
    the signal. Its observation period and provenance refresh dates supply the identity fields.
    """
    candidates = probe.candidates
    if not candidates:
        return _NO_DISPATCH_SIGNAL
    selections = _select_member_facets(
        request, candidates, graph=graph, breadth=probe.breadth_in(request.window)
    )
    dispatch_ids = joint_dispatch_facet_ids(candidates, selections)
    if not dispatch_ids:
        return _NO_DISPATCH_SIGNAL
    matched = [candidate for candidate in candidates if candidate.facet_id in dispatch_ids]
    obs_count = 0
    earliest_date: str | None = None
    latest_date: str | None = None
    for candidate in matched:
        obs_count += candidate.obs_count
        earliest_date = _earliest(earliest_date, candidate.earliest_date)
        latest_date = _latest(latest_date, candidate.latest_date)
    representative = next(
        candidate for candidate in matched if candidate.facet_id == dispatch_ids[0]
    )
    last_refresh, next_refresh = _refresh_dates(representative.provenance_id, provenance_refresh)
    return MemberSignal(
        dispatch_facet_id=dispatch_ids[0],
        obs_count=obs_count,
        earliest_date=earliest_date,
        latest_date=latest_date,
        observation_period=representative.observation_period,
        provenance_last_refresh=last_refresh,
        provenance_next_refresh=next_refresh,
    )


def _group_signal(
    requests: Sequence[MaterialisationRequest], probe: CollectionFacetProbe, *, graph: GraphReader
) -> MemberSignal | None:
    """Build the head signal from every facet each member dispatches under the shared group
    selection.

    Selection is shared, but dispatch ids are checked against each member's candidates. A member
    can therefore dispatch from facets outside the group's narrowed candidate set, including
    distinct facet ids with matching axes. Omitting one could miss an update.

    Return `None` when group selection raises `DefaultFacetUndefinedError`,
    `FacetNotAvailableError`, or `NonCurrencyUnitError`. The caller then re-materialises. Otherwise
    sum counts across all dispatched member/facet pairs, widen the date range, and use the lowest
    `facet_id` as representative. Its observation period and provenance dates supply identity
    fields. With no candidate on any member, or no dispatched facets, return a zero signal
    without a facet id (`_NO_DISPATCH_SIGNAL`).
    """
    if not any(probe.candidates):
        return _NO_DISPATCH_SIGNAL
    try:
        selections = select_group_facets(
            requests,
            probe.candidates,
            graph=graph,
            breadth=group_breadth(probe.members, window=requests[0].window),
        )
    except (DefaultFacetUndefinedError, FacetNotAvailableError, NonCurrencyUnitError):
        return None
    matched: list[Facet] = []
    for candidates in probe.candidates:
        dispatch_ids = joint_dispatch_facet_ids(candidates, selections)
        matched.extend(candidate for candidate in candidates if candidate.facet_id in dispatch_ids)
    if not matched:
        return _NO_DISPATCH_SIGNAL
    obs_count = 0
    earliest_date: str | None = None
    latest_date: str | None = None
    for candidate in matched:
        obs_count += candidate.obs_count
        earliest_date = _earliest(earliest_date, candidate.earliest_date)
        latest_date = _latest(latest_date, candidate.latest_date)
    # `joint_dispatch_facet_ids` returns string facet ids, so every matched candidate has one.
    # The filter narrows the annotation for the type checker.
    dispatch_facet_id = min(
        candidate.facet_id for candidate in matched if candidate.facet_id is not None
    )
    representative = next(
        candidate for candidate in matched if candidate.facet_id == dispatch_facet_id
    )
    last_refresh, next_refresh = _refresh_dates(
        representative.provenance_id, probe.provenance_refresh
    )
    return MemberSignal(
        dispatch_facet_id=dispatch_facet_id,
        obs_count=obs_count,
        earliest_date=earliest_date,
        latest_date=latest_date,
        observation_period=representative.observation_period,
        provenance_last_refresh=last_refresh,
        provenance_next_refresh=next_refresh,
    )


def _clock_window(window: RequestedWindow) -> tuple[int | None, int | None] | None:
    """The start and end year `window` reads when its definition leaves the end unstated, the
    end `election_window` gave it, or None for a window that reads no clock."""
    if window.kind != "range" or not window.end_unstated:
        return None
    return window.start_year, window.end_year


def clock_moved(signal: Sequence[MemberSignal]) -> bool:
    """Whether a member of a stored head's `signal` read a window its definition left open
    that a read now ends elsewhere (`MemberSignal.clock_window`): its start, ended as the
    adapter ends an identity's open window (`adapter.to_materialisation_request`), gives
    another end year. The check reads the clock alone, so a head served without a probe is
    held to the year too."""
    return any(
        member.clock_window is not None
        and election_window(
            RequestedWindow(kind="range", start_year=member.clock_window[0])
        ).end_year
        != member.clock_window[1]
        for member in signal
    )


def derive_signals(
    requests: Sequence[MaterialisationRequest],
    probe: CollectionFacetProbe,
    *,
    graph: GraphReader,
) -> tuple[MemberSignal, ...] | None:
    """Return the spec's one signal, or `None` when selection or conversion fails.

    A multi-member group uses `_group_signal` for all dispatched facets. A single-member spec
    uses `_member_signal`. The signal carries the window the requests read, which they share,
    when that window's end is unstated (`_clock_window`).
    """
    if len(requests) > 1:
        signal = _group_signal(requests, probe, graph=graph)
        if signal is None:
            return None
    else:
        (request,) = requests
        (member,) = probe.members
        try:
            signal = _member_signal(
                request, member, graph=graph, provenance_refresh=probe.provenance_refresh
            )
        except (DefaultFacetUndefinedError, FacetNotAvailableError, NonCurrencyUnitError):
            return None
    return (replace(signal, clock_window=_clock_window(requests[0].window)),)


def recheck_rate_provenance(
    signals: Sequence[MemberSignal],
    stale_signal: Sequence[MemberSignal],
    *,
    graph: GraphReader,
    provenance_refresh: Mapping[str, ProvenanceRefresh],
) -> tuple[MemberSignal, ...]:
    """Refresh stored rate-provenance dates on the matching fresh signals.

    The stale head supplies provenance ids because only materialisation runs conversion recipes.
    Pair entries by position. A length mismatch returns `signals` unchanged. Probe results cover
    known ids, and one batched graph call fetches all other distinct ids. This costs at most one
    extra round trip. Members with an empty provenance list pass through unchanged.
    """
    if len(signals) != len(stale_signal):
        return tuple(signals)
    all_ids = sorted(
        {
            provenance_id
            for stale in stale_signal
            for provenance_id, _ in stale.rate_provenance_refresh
        }
    )
    if not all_ids:
        return tuple(signals)
    missing = [
        provenance_id for provenance_id in all_ids if provenance_id not in provenance_refresh
    ]
    fetched: dict[str, str | None] = {}
    if missing:
        arcs_by_id = graph.node_arcs_many(missing, properties=("lastDataRefreshDate",))
        for provenance_id in missing:
            arcs = arcs_by_id.get(provenance_id)
            values = arcs.get("lastDataRefreshDate", ()) if arcs is not None else ()
            fetched[provenance_id] = values[0] if values else None

    def _current_refresh(provenance_id: str) -> str | None:
        refresh = provenance_refresh.get(provenance_id)
        if refresh is not None:
            return refresh.last_data_refresh_date
        return fetched.get(provenance_id)

    rechecked: list[MemberSignal] = []
    for fresh, stale in zip(signals, stale_signal, strict=True):
        if not stale.rate_provenance_refresh:
            rechecked.append(fresh)
            continue
        rechecked.append(
            replace(
                fresh,
                rate_provenance_refresh=tuple(
                    (provenance_id, _current_refresh(provenance_id))
                    for provenance_id, _ in stale.rate_provenance_refresh
                ),
            )
        )
    return tuple(rechecked)


def recheck_recipe_fingerprint(
    signals: Sequence[MemberSignal], stale_signal: Sequence[MemberSignal]
) -> tuple[MemberSignal, ...]:
    """Refresh converted members' recipe fingerprints from the deployed `RECIPE_BODIES`.

    The fingerprint uses deployed `RECIPE_BODIES` and covers recipe content and `table_provenance`.
    The currency comparison catches edits under unchanged recipe ids. Only materialisation runs
    recipes and records fingerprints. Unconverted members pass through. A converted
    member with a missing or retired fingerprint gets a forced mismatch, so one rebuild writes a
    verifiable value. Reads no graph. Pair entries by position. On a length mismatch, return
    `signals` unchanged.
    """
    if len(signals) != len(stale_signal):
        return tuple(signals)
    rechecked: list[MemberSignal] = []
    for fresh, stale in zip(signals, stale_signal, strict=True):
        if not stale.rate_provenance_refresh:
            rechecked.append(fresh)
            continue
        if stale.recipe_dependency_fingerprint is None:
            rechecked.append(
                replace(fresh, recipe_dependency_fingerprint=_UNVERIFIABLE_RECIPE_FINGERPRINT)
            )
            continue
        recipe_id, _stale_digest = stale.recipe_dependency_fingerprint
        current = recipe_dependency_fingerprint(recipe_id)
        rechecked.append(
            replace(
                fresh,
                recipe_dependency_fingerprint=(
                    current if current is not None else _UNVERIFIABLE_RECIPE_FINGERPRINT
                ),
            )
        )
    return tuple(rechecked)
