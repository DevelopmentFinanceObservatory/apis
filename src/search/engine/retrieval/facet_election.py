"""Elect the facet a bulk observation preview serves, before and during the read.

A bulk preview serves one facet across many series: a scope-defaulted query across every
place in its scope, or a role-bearing query across more role pairs than the per-pair path
takes. Both merge the series' candidates by `facet_id` and rank them with
`materialise.facets.rank_pool`, the ordering a single member uses, led by the number of
series each facet covers. Coverage calls the same elections to date a `latest` answer, so
the year it names is the year the preview reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from materialise.facets import (
    SeriesCounts,
    at_cadence,
    cadences_by_member,
    continuation,
    merge_facets,
    overlaps_window,
    rank_pool,
)
from materialise.window import RequestedWindow
from search.engine.clients.graph import Facet
from search.engine.clients.sdmx import RoleFacetCandidate, RoleFacetIndex, RoleTuple

if TYPE_CHECKING:
    from series_coverage.coverage_types import FacetSpan


@dataclass
class ScopeFacetCandidate:
    """A facet's coverage across a defaulted place scope, merged across StatVars.

    `by_entity` holds the facet at each place, merged across StatVars
    (`materialise.facets.merge_facets`), and `facet` the facet merged across every place.
    """

    facet_id: str
    by_entity: dict[str, Facet] = field(default_factory=dict)

    @property
    def facet(self) -> Facet:
        """The facet across every place: counts summed, dated points concatenated, date
        bounds widened, and metadata from the first place."""
        return merge_facets(list(self.by_entity.values()))

    def breadth(self, window: RequestedWindow | None) -> int:
        """The places this facet holds a point at inside `window` (`overlaps_window`)."""
        return len(self.places_held(window))

    def places_held(self, window: RequestedWindow | None) -> frozenset[str]:
        """The places at which this facet holds a point inside `window`."""
        return frozenset(
            entity
            for entity, f in self.by_entity.items()
            if f.obs_count > 0 and overlaps_window(f, window=window)
        )


def merge_scope_candidates(
    facets_by_entity: Mapping[str, Sequence[Facet]], entity_dcids: Sequence[str]
) -> dict[str, ScopeFacetCandidate]:
    """Merge per-place probe facets by `facet_id` across the requested scope.

    Skip missing ids because an empty SDMX facet filter means unfiltered data. Visit places
    in `entity_dcids` order for deterministic ties. The first candidate supplies metadata.
    """
    by_place: dict[str, dict[str, list[Facet]]] = {}
    for entity in entity_dcids:
        for f in facets_by_entity.get(entity, ()):
            if f.facet_id is None:
                continue
            by_place.setdefault(f.facet_id, {}).setdefault(entity, []).append(f)
    return {
        facet_id: ScopeFacetCandidate(
            facet_id=facet_id,
            by_entity={entity: merge_facets(facets) for entity, facets in places.items()},
        )
        for facet_id, places in by_place.items()
    }


def rank_scope_candidates(
    candidates: Mapping[str, ScopeFacetCandidate],
    *,
    stat_vars: Sequence[str],
    window: RequestedWindow | None = None,
) -> list[ScopeFacetCandidate]:
    """Narrow to candidates holding a point in `window` (`overlaps_window`, over the places'
    dates when every place's facet carries them), then rank by `rank_pool`, led by the
    number of places each facet holds a point at inside `window`
    (`ScopeFacetCandidate.breadth`), then by the window years its span covers across every
    place.

    A scope read serves one latest point per place, so coverage wins over depth: a probe over
    252 countries ranked 251/252 places (10,586 observations) above 216/252 (14,010). A
    window no candidate reaches leaves the pool whole.
    """
    ranked = rank_pool(
        [c.facet for c in candidates.values()],
        stat_vars=stat_vars,
        window=window,
        breadth=SeriesCounts(
            counts={c.facet_id: c.breadth(window) for c in candidates.values()},
            union_of=lambda ids: len(
                frozenset().union(*(candidates[i].places_held(window) for i in ids))
            ),
        ),
    ).ranked
    return [candidates[f.facet_id] for f in ranked if f.facet_id is not None]


def scope_served(
    lead: ScopeFacetCandidate,
    candidates: Mapping[str, ScopeFacetCandidate],
    *,
    window: RequestedWindow | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The facet ids a mint of the answer's Spec serves when it elects `lead`, each facet
    merged across every place: its selection, the facets of `candidates` sharing `lead`'s
    provenance, method and unit at its cadence (`materialise.facets.cadences_by_member`),
    `lead` first, as `materialise.facets.joint_matches` admits them; then the facets
    continuing the selection's series in `window`, each past its own place's last year
    (`materialise.facets.continuation`). Several qualifying methods rank by the places each
    facet holds a point at. The continuation is empty where nothing continues the
    selection."""
    by_id = {facet_id: c.facet for facet_id, c in candidates.items()}
    facets = list(by_id.values())
    (cadence_of,) = cadences_by_member([facets])
    cadence = cadence_of.get(lead.facet_id)

    def serves(facet_id: str) -> bool:
        return at_cadence(facet_id, cadence=cadence, cadence_of=cadence_of)

    head = by_id[lead.facet_id]
    axes = (head.provenance_id, head.measurement_method, head.unit)
    selection = [
        head,
        *(
            f
            for facet_id, f in by_id.items()
            if facet_id != lead.facet_id
            and serves(facet_id)
            and (f.provenance_id, f.measurement_method, f.unit) == axes
        ),
    ]
    continuing = continuation(
        selection,
        facets,
        window=window,
        serves=serves,
        breadth={c.facet_id: c.breadth(window) for c in candidates.values()},
    )
    return (
        tuple(f.facet_id for f in selection if f.facet_id is not None),
        tuple(f.facet_id for f in continuing if f.facet_id is not None),
    )


@dataclass
class RoleBulkCandidate:
    """A facet's coverage summary merged across StatVars.

    `by_sv` keeps each StatVar's role pairs separate. `pair_count` sums them, and
    `latest_date` is the latest of each pair's latest point.
    Metadata is shared by facet id in the coverage index. `lifetime` holds each StatVar's
    observation count, span and first series' end over every year under the same bound
    roles (`CoverageIndex.facet_spans`), the values a mint ranks and continues the facet by;
    it is empty for candidates built outside coverage.
    """

    facet_id: str
    by_sv: dict[str, list[RoleFacetCandidate]] = field(default_factory=dict)
    pair_count: int = 0
    latest_date: str | None = None
    provenance_id: str | None = None
    measurement_method: str | None = None
    unit: str | None = None
    observation_period: str | None = None
    lifetime: dict[str, FacetSpan] = field(default_factory=dict)


def merge_role_candidates(
    index: RoleFacetIndex,
    sv_dcids: Sequence[str],
    *,
    lifetime: Mapping[tuple[str, str], FacetSpan] | None = None,
) -> dict[str, RoleBulkCandidate]:
    """Merge candidates by facet id across the requested StatVars, each keeping its
    StatVars' spans from `lifetime`, keyed by `(stat_var, facet_id)`.

    Coverage rows require a facet id. The assertion checks the index schema invariant.
    """
    merged: dict[str, RoleBulkCandidate] = {}
    for sv in dict.fromkeys(sv_dcids):
        for c in index.get(sv, ()):
            assert c.facet_id is not None, "role index candidate carries no facet_id"
            candidate = merged.get(c.facet_id)
            if candidate is None:
                candidate = RoleBulkCandidate(
                    facet_id=c.facet_id,
                    provenance_id=c.provenance_id,
                    measurement_method=c.measurement_method,
                    unit=c.unit,
                    observation_period=c.observation_period,
                )
                merged[c.facet_id] = candidate
            candidate.by_sv.setdefault(sv, []).append(c)
            if lifetime is not None and (span := lifetime.get((sv, c.facet_id))) is not None:
                candidate.lifetime[sv] = span
            candidate.pair_count += c.pair_count
            if c.latest_date is not None and (
                candidate.latest_date is None or c.latest_date > candidate.latest_date
            ):
                candidate.latest_date = c.latest_date
    return merged


def role_n_obs(
    candidate: RoleBulkCandidate, *, first_year: int | None, last_year: int | None
) -> int:
    """Summed coverage observation count across every pair of `candidate`, inside
    `[first_year, last_year]`."""
    return sum(
        c.pairs.n_obs(first_year=first_year, last_year=last_year)
        for candidates in candidate.by_sv.values()
        for c in candidates
    )


def window_of(time_period: Sequence[str] | None) -> RequestedWindow | None:
    """The year range `time_period` names, or None when the read is unwindowed."""
    if time_period is None:
        return None
    return RequestedWindow(
        kind="range", start_year=int(time_period[0]), end_year=int(time_period[-1])
    )


def rank_role_candidates(
    candidates: Mapping[str, RoleBulkCandidate],
    *,
    stat_vars: Sequence[str],
    window: RequestedWindow | None,
) -> list[RoleBulkCandidate]:
    """Narrow to candidates reaching `window`, then rank by `rank_pool`, led by the number
    of role pairs each facet covers, then the window years its span covers, then its
    observation count. Every read plan builds `candidates` from a role index over the
    window's years, so each holds a pair with data in them and the pair count is those
    years' count.

    The span, the observation count and the first series' end are the facet's over every
    year (`lifetime`, summed and widened across the StatVars the candidate keeps), as a mint
    of the answer's Spec reads them from coverage (`materialise.facets.coverage_probe`), so
    a bulk answer and its mint rank and continue each key on the same value: a series that
    ends before the window opens still ends the facet's first series. A candidate built
    outside coverage carries no lifetime: it ranks on its count inside the window and its
    latest date, with its covered years running from the window's start.

    A joined candidate (`materialise.facets.union_count`) counts the role tuples any of its
    facets holds, read from their pairs only when the ranking forms it (`_pair_union`).
    """
    ranked = rank_pool(
        [_ranked_facet(c, window=window) for c in candidates.values()],
        stat_vars=stat_vars,
        window=window,
        breadth=SeriesCounts(
            counts={c.facet_id: c.pair_count for c in candidates.values()},
            union_of=lambda ids: _pair_union([candidates[i] for i in ids], window=window),
        ),
    ).ranked
    return [candidates[f.facet_id] for f in ranked if f.facet_id is not None]


def _pair_union(candidates: Sequence[RoleBulkCandidate], *, window: RequestedWindow | None) -> int:
    """The role tuples at least one of `candidates` holds with data inside `window`, counted
    per StatVar and summed as `pair_count` is. Each facet's pairs are read once."""
    first, last = (window.start_year, window.end_year) if window is not None else (None, None)
    by_sv: dict[str, set[RoleTuple]] = {}
    for candidate in candidates:
        for sv, role_candidates in candidate.by_sv.items():
            held = by_sv.setdefault(sv, set())
            for c in role_candidates:
                held.update(
                    role_tuple
                    for role_tuple, _n_obs in c.pairs.head(
                        c.pair_count, first_year=first, last_year=last, with_data=True
                    )
                )
    return sum(len(held) for held in by_sv.values())


def _ranked_facet(candidate: RoleBulkCandidate, *, window: RequestedWindow | None) -> Facet:
    """`candidate` as the facet `rank_pool` ranks (`rank_role_candidates`)."""
    spans = [candidate.lifetime[sv] for sv in candidate.by_sv if sv in candidate.lifetime]
    if spans:
        obs_count = sum(span.n_obs for span in spans)
        earliest_date = str(min(span.first_year for span in spans))
        latest_date: str | None = str(max(span.last_year for span in spans))
        first_series_end: str | None = str(min(span.first_series_end for span in spans))
    else:
        first, last = (window.start_year, window.end_year) if window is not None else (None, None)
        obs_count = role_n_obs(candidate, first_year=first, last_year=last)
        earliest_date, latest_date, first_series_end = None, candidate.latest_date, None
    return Facet(
        facet_id=candidate.facet_id,
        provenance_id=candidate.provenance_id,
        measurement_method=candidate.measurement_method,
        unit=candidate.unit,
        obs_count=obs_count,
        earliest_date=earliest_date,
        latest_date=latest_date,
        first_series_end=first_series_end,
        observation_period=candidate.observation_period,
        # Coverage's years, whatever the facet's cadence, so no evidence that it is annual.
        dates_coarsened=True,
    )


def role_continuation(
    best: RoleBulkCandidate,
    candidates: Mapping[str, RoleBulkCandidate],
    *,
    window: RequestedWindow | None,
) -> tuple[str, ...]:
    """The facet ids of `candidates` continuing `best`'s series in `window`, each past its
    own role tuple's last year (`materialise.facets.continuation`), each read as
    `rank_role_candidates` ranks it, as a mint of the answer's Spec continues the facet it elects. Several qualifying
    methods rank by the role pairs each facet covers. A facet continues only at `best`'s
    cadence (`materialise.facets.cadences_by_member`), read from the coverage's stated
    `observationPeriod`. Empty where nothing continues `best`."""
    facets = [_ranked_facet(c, window=window) for c in candidates.values()]
    (cadence_of,) = cadences_by_member([facets])
    cadence = cadence_of.get(best.facet_id)
    continuing = continuation(
        [_ranked_facet(best, window=window)],
        facets,
        window=window,
        serves=lambda facet_id: at_cadence(facet_id, cadence=cadence, cadence_of=cadence_of),
        breadth={c.facet_id: c.pair_count for c in candidates.values()},
    )
    return tuple(f.facet_id for f in continuing if f.facet_id is not None)


def select_role_candidate(
    candidates: Mapping[str, RoleBulkCandidate],
    *,
    stat_vars: Sequence[str],
    time_period: Sequence[str] | None,
) -> RoleBulkCandidate | None:
    """The facet a bulk role read serves from `candidates`: the top of
    `rank_role_candidates` over `time_period`'s window. `read_plan.BulkPlan` elects through
    here, at build and again after a filter narrows it. None when `candidates` is empty."""
    if not candidates:
        return None
    return rank_role_candidates(candidates, stat_vars=stat_vars, window=window_of(time_period))[0]
