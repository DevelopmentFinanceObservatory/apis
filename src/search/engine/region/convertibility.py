"""Decide unit conversion for each resolved StatVar before fetching observations.

The text, parsed, and promote paths share request construction. Each caller maps
decision errors to its own wire response. Request fields match the preview
builder, including the window, which narrows the facets the decision reads.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from itertools import product
from typing import TYPE_CHECKING, Protocol

from dc_sdmx import EntityFilter
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise import FacetBinding, GraphReader, MaterialisationRequest, decide_group_unit
from materialise.errors import DefaultFacetUndefinedError, FacetNotAvailableError
from materialise.facets import FacetSelection, MemberProbe, UnitRequest, native_currency_iso
from materialise.finish import (
    currency_unmapped_omitted_warning,
    deflator_missing_omitted_warning,
)
from materialise.member_conversion import (
    MemberConversionDecision,
    ServedBlock,
    conversion_graph_reader,
    coverage_decision,
    require_decision_coverage,
    served_block,
)
from materialise.rate_tables import RateTableCache
from materialise.recipes import (
    NonCurrencyUnitError,
    RateCoverageGapError,
    RecipeNotDeclaredError,
    currency_unmapped_error,
    deflator_missing_error,
    graph_role_for_spec_role,
    resolve_deflator_role,
    resolve_entity_currencies,
)
from search.engine.clients.graph import Facet
from search.engine.degraded_reads import note_degraded_read
from search.engine.departures import (
    conversion_period_fact,
    deflator_omission_fact,
    deflator_omission_undecided_fact,
)
from search.engine.errors import GraphInfraError
from search.engine.region.bound_unit_gate import bound_unit_gate_reason, unit_served
from search.engine.region.codes import (
    CURRENCY_UNMAPPED_OMITTED,
    DATE_COVERAGE_PARTIAL,
    DEFLATOR_MISSING_OMITTED,
)
from search.engine.region.facet_probe import FacetPins, facet_pins
from search.engine.region.facets import FilteredAnswer, plan_answer
from search.engine.retrieval.download import deflator_omissions
from search.engine.retrieval.elected_dataset import NO_DATASETS, ElectedDatasets
from search.engine.retrieval.read_plan import (
    BulkPlan,
    Omissions,
    PairPlan,
    PlanSeries,
    PlanYears,
    ReadPlan,
    YearBounds,
    dated_without_complete_dates,
    facet_dates,
    plan_window,
    served_dates,
    served_role_entities,
    serves_nothing,
    year_of,
)
from search.engine.unit.conversion import (
    UNDECIDED,
    ConversionAvailability,
    ConversionPeriod,
    CountedConversion,
    conversion_period,
    counted_conversion,
)
from search.engine.unit.conversion_catalogue import ConversionCatalogue
from search.models import (
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityFacetBinding,
    IdentityFacetValueAbsent,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    TimeWindow,
    Warning,
    in_window,
)
from series_coverage.index import CoverageIndex

if TYPE_CHECKING:
    from search.engine.clients.sdmx import RoleFacetCandidate
    from search.engine.extract import DateRequest
    from search.engine.retrieval.retrieve import Materialised
    from search.heading.departure_fact import DepartureFact

logger = logging.getLogger(__name__)

_STAT_VAR_REF_PREFIX = "dcid:"
"""Matches `MaterialisationRequest.stat_var`'s precondition in
`materialise.generation._STAT_VAR_REF_PREFIX` and `region/observations/fetch.py`."""


def survivor_declared_roles(mat_result: Materialised) -> tuple[str, ...]:
    """Return the roles the resolved StatVars declare, for a unit decision.

    A shape's own `declared_roles` is the union over its group, so it can name a
    role the survivors lack (the by-creditor role beside a total). SDMX gets the
    survivors' shared signature, in the first survivor's order, which the role-class
    election guarantees exists. A result without `member_facts`, or whose survivors
    disagree, raises `ValueError`; every resolver populates them and elects one class.
    """
    if mat_result.member_facts is None:
        raise ValueError("unit decision needs the materialised result's member_facts")
    roles_by_sv = {
        sv: mat_result.member_facts[sv].read_declared_roles() for sv in mat_result.sv_dcids
    }
    if len({frozenset(roles) for roles in roles_by_sv.values()}) > 1:
        raise ValueError(f"survivors disagree on declared roles: {roles_by_sv!r}")
    return next(iter(roles_by_sv.values()), ("observationAbout",))


def unit_request_for_materialise(request: IdentityUnitRequest) -> UnitRequest:
    """Convert the identity request to a materialise unit request.

    A string currency passes through; a role-object currency becomes its bare
    role key. ``decide_group_unit`` checks whether that role is declared.
    """
    currency = request.currency if isinstance(request.currency, str) else request.currency.role
    return UnitRequest(currency=currency, prices=request.prices, base_year=request.base_year)


_BOUND_FACET_COMBINATIONS_MAX = 60
"""Ceiling on the Cartesian product `_bound_facet_combinations` enumerates
across every Set-bound axis in one `bound_facets` mapping (`provenance`/
`measurementMethod`, the only two Set-capable bound_facets axes -- `unit`
is always `state="request"`, never Set-bound). Checked at the point of
enumeration, before a single combination is yielded: each yielded
combination drives one real, unbatched network probe per StatVar
in a unit decision's own per-combination loop, so counting or filtering
afterwards would leave the peak probe count untouched. The ceiling is on
the product across every Set-bound axis together, not on any one axis
alone.

Derived from observed usage, not measured directly: at the time this
guard was added, the largest bound-facet Set found anywhere in this
repository's tests, fixtures, and docs (`provenance`, `measurementMethod`,
or the analogous `unit`/`activitySource` Set-bound axes) was 3 refs
(`tests/engine/test_facet_refusals.py`'s three-member `measurementMethod`
Set); most were 2. Separately, `docs/dcp-api-usage.md`'s own live
measurement of a broad real query ("ODA grants to Kenya") confirmed
around 26 distinct facets in total, across every axis and StatVar
combined -- not a bound-Set width, but the closest real facet-cardinality
figure this deployment has recorded. 60 sits comfortably above both: a
wide margin against everything observed, not a tight fit against a named
real case. There is no telemetry on real Set widths on these two axes
specifically, so this number should move only when such evidence appears
-- widen it from a measured real binding that needed more, not from
analogy to a cap on a different quantity.

Over the ceiling: turns a pathological 32x32 (1024 combinations) or
1024x1024 (the per-axis wire-schema max -- see `IdentityBindingSet.refs`)
binding into a hard refusal instead of up to 1024 (or 1,048,576) real
graph probes."""


_REQUEST_PROBE_BUDGET_MAX = 10_000
"""Ceiling on `len(sv_dcids) * _combination_count(bound_facets)`: the total
probe count one `decide_spec_unit` or `decide_option_units` call would pay
across every StatVar it decides, checked once before the combination loop
starts. `_BOUND_FACET_COMBINATIONS_MAX` bounds the product for a single
`sv_dcids` entry; since `bound_facets` does not vary by `sv`, that bound is
already sound on its own. What it leaves open is the UNDER-ceiling case:
a legitimate combination width multiplied across every StatVar in a large
population, which nothing bounded before this guard. `options/surface.py`
calls `decide_option_units` with its full server-derived population, which
carries no StatVar-count precheck of its own (see that module's own
docstring), and the code index's own population is deliberately
never capped at all (see `options/population.py`'s own docstring) -- both
make a large `sv_dcids` reachable here.

Derived from observed usage, not measured directly. The largest real
population this deployment has documented reaching this code path is the
ODA-to-Kenya CRS family at 3,812 StatVars (`options/population.py`'s own
worked example of an intentionally uncapped index-branch population). The
most common bound-facet combination width, per `_BOUND_FACET_COMBINATIONS_
MAX`'s own derivation above, is 1 or 2; the single largest ever found in
this repository's tests is 3. A 3,812-member population under a typical
2-wide binding pays 7,624 probes -- ordinary, legitimate traffic that must
not refuse. 10,000 sits above that with room (about 1.3x), the same "wide
margin, not a tight fit" spirit `_BOUND_FACET_COMBINATIONS_MAX`'s own
derivation uses, while sitting well below the 42,000 probes a 700-member
population would pay under a binding at the per-entry ceiling (700 * 60) --
the exact pathological combination this guard exists to refuse before it
starts; nothing upstream caps `sv_dcids` at 700 any more (see this
docstring's own opening paragraph), so a wider population than that pays
proportionally more probes, still refused by this same ceiling once the
product crosses it. As with
`_BOUND_FACET_COMBINATIONS_MAX`, widen this only from a measured real
request that legitimately needed more, not by analogy to a cap on a
different quantity."""


class BoundFacetCombinationsCeilingExceededError(Exception):
    """Raised before any unit probe when either probe budget is exceeded.

    ``count`` and ``cap`` report the product and limit that triggered the
    error. Callers map both limits to the same ``facet_probe_ceiling`` 422.
    """

    def __init__(self, count: int, cap: int) -> None:
        self.count = count
        self.cap = cap
        super().__init__(
            f"{count} bound facet combinations exceed the {cap}-combination "
            "limit; bind fewer refs on the provenance/measurementMethod Set "
            "axis, or bind a single ref instead of a Set on one axis"
        )


BoundFacetRefs = str | None | Sequence[str]
"""One bound non-unit axis: a ref, `None` for the facets that declare no value on the
axis, or a sequence of refs."""


def bound_facet_refs(
    facets: Mapping[str, IdentityFacetBinding] | None, source_dcids: Sequence[str]
) -> dict[str, BoundFacetRefs]:
    """The non-unit pins a unit decision runs under (`decide_spec_unit`'s `bound_facets`).

    Provenance comes from `source_dcids`: one ref, or a sequence for a set. The method
    comes from `facets`: a ref, a sequence for a set, or `None` for a null ref, which pins
    the facets that declare no method. An unpinned axis is absent, so the decision reads
    it open. The options door and the resubmit door both build their pins here.
    """
    refs: dict[str, BoundFacetRefs] = {}
    if len(source_dcids) == 1:
        refs["provenance"] = source_dcids[0]
    elif source_dcids:
        refs["provenance"] = list(source_dcids)
    method = (facets or {}).get("measurementMethod")
    if isinstance(method, IdentityBindingValue | IdentityFacetValueAbsent):
        refs["measurementMethod"] = method.ref
    elif isinstance(method, IdentityBindingSet):
        refs["measurementMethod"] = method.refs
    return refs


def _combination_count(bound_facets: Mapping[str, BoundFacetRefs]) -> int:
    """Count combinations without enumeration, treating each string or `None` as one choice."""
    return math.prod(
        1 if value is None or isinstance(value, str) else len(value)
        for value in bound_facets.values()
    )


def _bound_facet_combinations(
    bound_facets: Mapping[str, BoundFacetRefs],
) -> Iterator[dict[str, str | None]]:
    """Yield the Cartesian product of bound-facet choices in axis order.

    A string or `None` contributes one value. A sequence contributes each member in
    order. Raise before yielding if the product exceeds the ceiling because
    every combination triggers a network probe.
    """
    axis_choices = [
        [(axis, value)]
        if value is None or isinstance(value, str)
        else [(axis, member) for member in value]
        for axis, value in bound_facets.items()
    ]
    combination_count = _combination_count(bound_facets)
    if combination_count > _BOUND_FACET_COMBINATIONS_MAX:
        raise BoundFacetCombinationsCeilingExceededError(
            combination_count, _BOUND_FACET_COMBINATIONS_MAX
        )
    for combo in product(*axis_choices):
        yield dict(combo)


def decide_spec_unit(
    members: Sequence[str],
    *,
    unit_request: IdentityUnitRequest,
    role_bindings: Mapping[str, EntityFilter],
    declared_roles: tuple[str, ...],
    bound_facets: Mapping[str, BoundFacetRefs] = {},
    years: PlanYears,
    graph: GraphReader,
    sdmx: MaterialiseSdmxClient,
    coverage: CoverageIndex,
    probes: Mapping[str, MemberProbe] | None = None,
    datasets: ElectedDatasets = NO_DATASETS,
    deadline: float | None = None,
) -> Mapping[str, tuple[Mapping[str, FacetSelection], ...]]:
    """Decide unit conversion for the members of one Spec under every bound-facet
    combination, as a mint of the Spec decides it: one election per combination across
    every member (`materialise.pipeline.decide_group_unit`, the election
    `generation._plan_group_members` runs), then each served member's conversion under it.

    `datasets` holds the elected dataset of each member of an index-resolved family. Each
    decision's request carries it (`MaterialisationRequest.dataset`), so the decision ranks
    that dataset's facets alone, over `probes` or over its own probe, as a mint does. When
    every member has one the family is index-resolved, and the election agrees on
    provenance and unit (`MaterialisationRequest.index_sourced`).

    `probes` holds each member's election input from the answer's read plan
    (`read_plan.plan_probes`): its candidates merged across the places and role tuples the
    answer names, and each facet's count of those it holds a point at. A member with a
    dispatchable candidate there decides over it and makes no graph read, so the decision
    ranks what the plan ranks. Any other member probes, once per combination, and joins
    the same election.

    Role bindings and declared roles describe the resolved shape. Non-unit
    axes such as provenance and measurementMethod hold a ref, `None` for the facets
    without a value, or a sequence of refs. The unit axis is always a request. ``coverage``
    is forwarded unchanged.

    Return each served member's selection mappings in combination order, the election's,
    every served member holding the same ones. A member the group of a combination leaves
    out holds none of that combination's, and one no combination serves is absent, so
    `elected_unit_refs` admits none of its facets: the requested currency decides which
    members serve, native data first among the groups serving the most members
    (`materialise.group_facets.elect_group`). Skip ``FacetNotAvailableError`` and
    ``DefaultFacetUndefinedError`` combinations, but try all combinations. Raise the first
    other error in encounter order, even after a prior success. If all combinations are
    skipped, re-raise the last skip. Empty ``members`` is a no-op.

    Raise ``BoundFacetCombinationsCeilingExceededError`` before any probe when
    either the per-member product exceeds 60 or
    ``len(members) * combination_count`` exceeds 10,000.

    Decisions read the graph through one ``conversion_graph_reader``: the first member
    read fetches every member's conversion arcs in one batch, then each provenance,
    dataset and currency node is read once. A decision that needs no graph read makes none.
    """
    _check_probe_budget(len(members), bound_facets)
    if not members:
        return {}
    return _decide_group(
        members,
        unit_request=unit_request,
        role_bindings=role_bindings,
        declared_roles=declared_roles,
        bound_facets=bound_facets,
        years=years,
        graph=conversion_graph_reader(graph, stat_vars=members),
        sdmx=sdmx,
        coverage=coverage,
        probes=probes,
        datasets=datasets,
        deadline=deadline,
    )


def decide_option_units(
    sv_dcids: Sequence[str],
    *,
    unit_request: IdentityUnitRequest,
    role_bindings: Mapping[str, EntityFilter],
    declared_roles: tuple[str, ...],
    bound_facets: Mapping[str, BoundFacetRefs] = {},
    years: PlanYears,
    graph: GraphReader,
    sdmx: MaterialiseSdmxClient,
    coverage: CoverageIndex,
    datasets: ElectedDatasets = NO_DATASETS,
    deadline: float | None = None,
) -> Mapping[str, tuple[Mapping[str, FacetSelection], ...]]:
    """Decide unit conversion for each of `sv_dcids` alone, as `decide_spec_unit` decides a
    Spec of that one StatVar: the options door's StatVars are alternatives to one another,
    not members of one Spec, so each elects its own unit. Every StatVar is returned or the
    first refusal raises, in `sv_dcids` order.

    Raise ``BoundFacetCombinationsCeilingExceededError`` before any probe when the
    per-StatVar product exceeds 60 or ``len(sv_dcids) * combination_count`` exceeds 10,000,
    the probes the call pays across every StatVar. One ``conversion_graph_reader`` serves
    every decision, as `decide_spec_unit` describes."""
    _check_probe_budget(len(sv_dcids), bound_facets)
    graph = conversion_graph_reader(graph, stat_vars=sv_dcids)
    decided: dict[str, tuple[Mapping[str, FacetSelection], ...]] = {}
    for sv in sv_dcids:
        decided.update(
            _decide_group(
                [sv],
                unit_request=unit_request,
                role_bindings=role_bindings,
                declared_roles=declared_roles,
                bound_facets=bound_facets,
                years=years,
                graph=graph,
                sdmx=sdmx,
                coverage=coverage,
                probes=None,
                datasets=datasets,
                deadline=deadline,
            )
        )
    return decided


def _check_probe_budget(stat_vars: int, bound_facets: Mapping[str, BoundFacetRefs]) -> None:
    """Refuse a decision whose probes, `stat_vars` times the bound-facet combinations,
    pass `_REQUEST_PROBE_BUDGET_MAX`, before any probe."""
    probe_budget = stat_vars * _combination_count(bound_facets)
    if probe_budget > _REQUEST_PROBE_BUDGET_MAX:
        raise BoundFacetCombinationsCeilingExceededError(probe_budget, _REQUEST_PROBE_BUDGET_MAX)


def _decide_group(
    members: Sequence[str],
    *,
    unit_request: IdentityUnitRequest,
    role_bindings: Mapping[str, EntityFilter],
    declared_roles: tuple[str, ...],
    bound_facets: Mapping[str, BoundFacetRefs],
    years: PlanYears,
    graph: GraphReader,
    sdmx: MaterialiseSdmxClient,
    coverage: CoverageIndex,
    probes: Mapping[str, MemberProbe] | None,
    datasets: ElectedDatasets,
    deadline: float | None,
) -> dict[str, tuple[Mapping[str, FacetSelection], ...]]:
    """`decide_spec_unit` over non-empty `members`, reading through `graph`, a
    `conversion_graph_reader` the caller built."""
    unit_binding = FacetBinding(state="request", request=unit_request_for_materialise(unit_request))
    requested_window = plan_window(years)
    index_sourced = all(sv in datasets for sv in members)
    member_probes = [
        _dispatchable(probes.get(sv) if probes is not None else None) for sv in members
    ]
    last_unavailable: FacetNotAvailableError | DefaultFacetUndefinedError | None = None
    decided = False
    by_member: dict[str, list[Mapping[str, FacetSelection]]] = {}
    for combo in _bound_facet_combinations(bound_facets):
        facet_bindings: dict[str, FacetBinding] = {
            axis: FacetBinding(state="bound", ref=ref) for axis, ref in combo.items()
        }
        facet_bindings["unit"] = unit_binding
        requests = [
            MaterialisationRequest(
                stat_var=f"{_STAT_VAR_REF_PREFIX}{sv}",
                role_bindings=role_bindings,
                declared_roles=declared_roles,
                facet_bindings=facet_bindings,
                window=requested_window,
                index_sourced=index_sourced,
                dataset=datasets.get(sv),
                # The decision selects as the answer's reads do (`observations.fetch`).
                requires_provenance=False,
            )
            for sv in members
        ]
        try:
            served = decide_group_unit(
                requests,
                member_probes,
                graph=graph,
                sdmx=sdmx,
                coverage=coverage,
                settle=False,
                deadline=deadline,
            )
        except (FacetNotAvailableError, DefaultFacetUndefinedError) as exc:
            last_unavailable = exc
            continue
        decided = True
        for sv, selections in zip(members, served, strict=True):
            if selections is not None:
                by_member.setdefault(sv, []).append(selections)
    if not decided:
        assert last_unavailable is not None
        raise last_unavailable
    return {sv: tuple(by_member[sv]) for sv in members if sv in by_member}


def _dispatchable(probe: MemberProbe | None) -> MemberProbe | None:
    """`probe` when a candidate of it carries a facet id, which a decision can dispatch, and
    None otherwise, so the decision probes the member itself."""
    if probe is None or not any(c.facet_id is not None for c in probe.candidates):
        return None
    return probe


def option_pins(
    facets: Mapping[str, IdentityFacetBinding] | None,
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None,
) -> FacetPins:
    """The pins a spec's facet options are judged against: its identity's facets, with a
    request-form unit admitted per StatVar by the units `decide_spec_unit` elected, the same
    units its pin refusal admits. With no election the request is tested literally. A first
    resolve, a candidate and a resubmit all build their option pins here."""
    return facet_pins(
        facets, elected_units=elected_unit_refs(elected) if elected is not None else None
    )


def elected_unit_refs(
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]],
) -> Mapping[str, frozenset[str]]:
    """Return each StatVar's elected unit refs. Unset selections are omitted."""
    return {
        sv: frozenset(
            selections["unit"].ref
            for selections in combinations
            if selections["unit"].ref is not None
        )
        for sv, combinations in elected.items()
    }


type UnitCoverageRefusal = RateCoverageGapError | RecipeNotDeclaredError | NonCurrencyUnitError

type _FacetKey = tuple[str | None, str | None]
"""A served facet's `(unit, provenance_id)`, the two axes a conversion decision reads."""

type _ServedRows = set[ServedBlock]
"""The rows a plan serves under one facet key, as conversion sees them
(`member_conversion.ServedBlock`)."""


def _facet_years(facet: Facet, years: PlanYears) -> set[int]:
    """The years `facet` serves under `years`: its dated points when it carries them all,
    else every year of its span there (`read_plan.facet_dates`)."""
    if facet.dates and facet.dates_complete:
        window = years if isinstance(years, TimeWindow) else None
        return {
            year
            for d in facet.dates
            if (window is None or in_window(d, window)) and (year := year_of(d)) is not None
        }
    span = facet_dates(facet, years)
    first, last = (year_of(d) for d in span) if span is not None else (None, None)
    if first is None or last is None:
        return set()
    return set(range(first, last + 1))


def _pair_rows(served: Sequence[tuple[PlanSeries, Facet, PlanYears]]) -> _ServedRows:
    """The rows each `(series, facet, years)` serves: one per year the facet serves under the
    series' read years, binding the series' own role tuple, so each block is one row and an
    entity is paired only with the entities its series names."""
    rows: _ServedRows = set()
    for series, facet, years in served:
        entities = {role: (dcid,) for role, dcid in series.role_tuple}
        rows.update(served_block(year, entities) for year in _facet_years(facet, years))
    return rows


def _bulk_member_rows(candidates: Sequence[RoleFacetCandidate], years: YearBounds) -> _ServedRows:
    """A member's rows in a bulk read of `candidates`, its candidates under the elected facet
    and the facets continuing it (`BulkPlan.served_candidates`): for each candidate and each
    year of `years`, the years the read serves the member in (`BulkPlan.member_years`), that
    its pairs hold data in, one block of the members of each role with a tuple holding data
    in that year (`RolePairs.members_with_data`), so a member is checked in exactly the years
    its own tuples reach. A continuing facet's tuples are checked in every such year, which
    includes the years the read leaves to the elected facet (`BulkPlan.continued_pairs`),
    since telling them apart decodes the tuples.

    A block pairs its roles' members without stating which tuples exist, since listing
    tuples decodes the population (`RolePairs.head`), which a bulk read exists to avoid.
    The check stays exact because every conversion term is keyed on one entity or currency
    and the year (`convert._gather_terms`), so the block needs no term a served tuple does
    not. `candidates` must already leave out the tuples a conversion omits (`BulkPlan.omitting`,
    which `unit_coverage` applies first). A member whose tuples in a year all bound an
    omitted entity then holds no data there and is not checked, as the read never converts
    its rows."""
    first, last = years
    rows: _ServedRows = set()
    for candidate in candidates:
        span = candidate.pairs.year_span(first_year=first, last_year=last)
        if span is None:
            continue
        for year in range(span[0], span[1] + 1):
            members = {
                role: candidate.pairs.members_with_data(first_year=year, last_year=year, role=role)
                for role in candidate.members_by_role
            }
            if any(members.values()):
                rows.add(served_block(year, members))
    return rows


def _served_keys(plan: ReadPlan) -> dict[str, dict[_FacetKey, Callable[[], _ServedRows]]]:
    """Each StatVar's facet keys `plan` serves rows under, in plan order, each with the
    expansion of those rows (`_ServedRows`), which runs only for a key whose rows convert.

    A per-series read keys each served series' group facets, and under `latest` reads each
    series in the year of its StatVar's one period alone (`PairPlan.series_dates`), the
    period a latest read serves and every served series holds (`PairPlan.served`). A bulk
    read keys its served members under the elected facet, which the facets continuing it
    share."""
    if isinstance(plan, BulkPlan):
        best = plan.elected
        if best is None:
            return {}
        key: _FacetKey = (best.unit, best.provenance_id)
        return {
            sv: {key: partial(_bulk_member_rows, plan.served_candidates(sv), plan.member_years(sv))}
            for sv in plan.served_members
        }
    by_sv: dict[str, dict[_FacetKey, list[tuple[PlanSeries, Facet, PlanYears]]]] = {}
    for series in plan.served:
        years = plan.years
        if years == "latest":
            dates = plan.series_dates(series)
            latest = year_of(dates[1]) if dates is not None else None
            if latest is None:
                continue
            years = TimeWindow(start_year=latest, end_year=latest)
        for f in series.group:
            keys = by_sv.setdefault(series.stat_var, {})
            keys.setdefault((f.unit, f.provenance_id), []).append((series, f, years))
    return {
        sv: {key: partial(_pair_rows, served) for key, served in keys.items()}
        for sv, keys in by_sv.items()
    }


def served_units(plan: ReadPlan) -> frozenset[str | None]:
    """The facet units `plan` serves rows under, `None` for a facet stating none."""
    return frozenset(unit for by_key in _served_keys(plan).values() for unit, _ in by_key)


def facet_selections(
    combos: Sequence[Mapping[str, FacetSelection]], key: _FacetKey
) -> Mapping[str, FacetSelection] | None:
    """The conversion input for rows a facet keyed `key` (its unit and provenance) serves:
    the elected selections whose native unit is the facet's, with the facet's own
    provenance, the provenance the read converts them under. None when no elected unit is
    the facet's. A ref-form unit converts nothing, so its selections come as elected and
    their decision (`coverage_decision`) is None. The answer's gate (`unit_coverage`) and a
    bulk read's conversion (`bulk_conversion.convert_pairs`) both read this, so rows the gate
    passes convert."""
    unit, provenance = key
    for selections in combos:
        if selections["unit"].ref != unit:
            continue
        if selections["unit"].request is None:
            return selections
        return {**selections, "provenance": replace(selections["provenance"], ref=provenance)}
    return None


@dataclass(frozen=True)
class UnitCoverage:
    """What the elected conversion does to the rows a plan serves: `refusal` is the first
    StatVar whose rows it cannot convert, with its refusal, or None when every row
    converts; `omitted` are the rows a conversion into each entity's own currency leaves
    out because no single currency names their entity, by StatVar, graph role and entity
    (`read_plan.Omission`)."""

    refusal: tuple[str, UnitCoverageRefusal] | None = None
    omitted: Omissions = frozenset()

    @property
    def omitted_entities(self) -> tuple[str, ...]:
        """The entities, sorted, whose rows `omitted` names."""
        return tuple(sorted({entity for _sv, _role, entity in self.omitted}))

    def omitted_warning(
        self, *, catalogue: ConversionCatalogue | None, spec_id: str | None = None
    ) -> Warning | None:
        """The `currency_unmapped_omitted` warning counting `omitted_entities`, the one a read
        of the same rows carries (`materialise.finish.currency_unmapped_omitted_warning`),
        with the entities it carries as `params` (`_omitted_params`) named from `catalogue`,
        or None when no entity was omitted."""
        if not self.omitted_entities:
            return None
        built = currency_unmapped_omitted_warning(self.omitted_entities)
        counted = built["omitted_entities"]
        return Warning(
            code=CURRENCY_UNMAPPED_OMITTED,
            severity=built["severity"],
            message=built["message"],
            spec_id=spec_id,
            params=_omitted_params(counted["count"], counted["refs"], catalogue=catalogue),
        )


class DatedFacetReader(Protocol):
    """The Search graph's facet read (`EngineGraphClient.observation_facets_batch`), through
    which the gate reads the dates of a series whose facets state only their span."""

    def observation_facets_batch(
        self,
        stat_vars: list[str],
        entities: list[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, list[Facet]]]: ...


def _dated_plan(
    plan: PairPlan, reader: DatedFacetReader, *, deadline: float | None
) -> PairPlan | None:
    """`plan` with the group facets of each served place series that lacks its dates
    (`read_plan.dated_without_complete_dates`) replaced by the same facets read with every
    date, where that read returns them all. The read runs under `deadline`. None when no
    such series is served, the deadline has passed, or the read fails, and the gate then
    keeps the span's result. The last two log why and note a degraded read
    (`degraded_reads.note_degraded_read`), so no cache stores an answer a later request
    with time to read the dates could decide otherwise."""
    sparse = [
        s for s in plan.served if len(s.role_tuple) == 1 and dated_without_complete_dates(s.group)
    ]
    if not sparse:
        return None
    stat_vars = sorted({s.stat_var for s in sparse})
    entities = sorted({s.role_tuple[0][1] for s in sparse})
    if deadline is not None and time.monotonic() >= deadline:
        logger.warning(
            "unit coverage dated read skipped past the deadline for stat_vars=%r "
            "entities=%r; keeping the span's result",
            stat_vars,
            entities,
        )
        note_degraded_read("unit coverage dates")
        return None
    try:
        dated = reader.observation_facets_batch(
            stat_vars, entities, needs_dates=True, deadline=deadline
        )
    except GraphInfraError:
        logger.warning(
            "unit coverage dated read failed for stat_vars=%r entities=%r; keeping the span's result",
            stat_vars,
            entities,
            exc_info=True,
        )
        note_degraded_read("unit coverage dates")
        return None

    def with_dates(series: PlanSeries) -> PlanSeries:
        ((_role, entity),) = series.role_tuple
        complete = {
            f.facet_id: f
            for f in dated.get(series.stat_var, {}).get(entity, ())
            if f.facet_id is not None and f.dates and f.dates_complete
        }
        return replace(series, group=tuple(complete.get(f.facet_id, f) for f in series.group))

    return replace(
        plan, series=tuple(with_dates(s) if len(s.role_tuple) == 1 else s for s in plan.series)
    )


def _coverage(
    plan: ReadPlan,
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]],
    *,
    declared_roles: tuple[str, ...],
    graph: GraphReader,
    sdmx: MaterialiseSdmxClient,
    rate_cache: RateTableCache | None,
    deadline: float | None,
) -> UnitCoverage:
    """`unit_coverage` over the years `plan`'s facets state."""
    groups: dict[MemberConversionDecision, tuple[list[str], _ServedRows]] = {}
    decision_refusal: tuple[str, UnitCoverageRefusal] | None = None
    # Rows no conversion reads survive as served; their keys expand only when an
    # omission makes survival decide the outcome.
    native: list[Callable[[], _ServedRows]] = []
    for sv, by_key in _served_keys(plan).items():
        for key, role_years in by_key.items():
            selections = facet_selections(elected.get(sv, ()), key)
            if selections is None:
                native.append(role_years)
                continue
            try:
                decision = coverage_decision(
                    stat_var=f"{_STAT_VAR_REF_PREFIX}{sv}",
                    declared_roles=declared_roles,
                    selections=selections,
                    graph=graph,
                    deadline=deadline,
                )
            except (RecipeNotDeclaredError, NonCurrencyUnitError) as exc:
                # A key serving no row needs no decision, so its refusal stands only
                # when the key serves one.
                if role_years():
                    decision_refusal = (sv, exc)
                    break
                continue
            if decision is None:
                native.append(role_years)
                continue
            served = role_years()
            if not served:
                continue
            svs, pooled = groups.setdefault(decision, ([], set()))
            if sv not in svs:
                svs.append(sv)
            pooled.update(served)
        if decision_refusal is not None:
            break
    survives = False
    omitted: set[tuple[str, str, str]] = set()
    deflated: set[tuple[str, int]] = set()
    first_omitting: str | None = None
    for decision, (svs, pooled) in groups.items():
        first_sv = svs[0]
        try:
            coverage = require_decision_coverage(
                decision,
                served=pooled,
                graph=graph,
                sdmx=sdmx,
                rate_cache=rate_cache,
                deadline=deadline,
            )
        except (RateCoverageGapError, NonCurrencyUnitError) as exc:
            return UnitCoverage(refusal=(first_sv, exc))
        survives = survives or coverage.survives
        deflated.update(coverage.deflator_omitted)
        if (coverage.omitted_entities or coverage.deflator_omitted) and first_omitting is None:
            first_omitting = first_sv
        role = decision.graph_role
        if role is not None:
            omitted.update((sv, role, entity) for sv in svs for entity in coverage.omitted_entities)
    if decision_refusal is not None:
        return UnitCoverage(refusal=decision_refusal)
    if first_omitting is not None and not (survives or any(rows() for rows in native)):
        entities = {entity for _sv, _role, entity in omitted}
        refusal = (
            currency_unmapped_error(entities) if entities else deflator_missing_error(deflated)
        )
        return UnitCoverage(refusal=(first_omitting, refusal))
    return UnitCoverage(omitted=frozenset(omitted))


def unit_coverage(
    plan: ReadPlan | None,
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None,
    *,
    declared_roles: tuple[str, ...],
    graph: GraphReader,
    sdmx: MaterialiseSdmxClient,
    rate_cache: RateTableCache | None,
    dated_facets: DatedFacetReader | None = None,
    deadline: float | None = None,
) -> UnitCoverage:
    """The first StatVar, in plan order, whose served rows the elected conversion cannot
    convert, with its refusal, and the rows the conversion omits.

    The rows are the years and entities `plan`, narrowed to the elected units, serves
    (`_served_keys`). Each StatVar's facet keys are decided before any row is expanded, so
    a natively satisfied or ref-form unit expands none unless survival decides the
    outcome. Members whose conversion decisions are equal
    (`materialise.member_conversion.coverage_decision`) read the same tables through the
    same recipe, so their distinct terms are pooled and checked once through
    `require_decision_coverage`, the lookup a read's conversion runs, and a refusal is
    attributed to the group's first StatVar. A role-shaped target omits the rows of
    entities no single currency names, as a read does, from every StatVar of the group that
    serves them; when no served row survives, the plan refuses `currency_unmapped` naming
    them, attributed to the first group that omitted any, as a read refuses
    (`refuse_if_every_row_omitted`). A decision refusal
    stops the collection, and the groups collected before it are checked first. A natively
    satisfied unit loads no table.

    A facet carrying all its dates serves those years; one stating only its span is first
    checked over every year of the span, which reads no observation. When that check finds
    a rate gap or omits an entity in a windowed or dateless per-series plan, the served place
    series lacking their dates are read with every date through `dated_facets` and the whole
    check runs again over the years they hold, which decides the refusal, the omitted
    entities and whether any row survives. A series whose span covers a year it holds no
    point in serves no row there, so that year needs no rate, and a series with no point in
    the window keeps no row to survive or omit. Without `dated_facets`, past `deadline`, or when that read
    fails, the span's result stands. An answer therefore refuses, and names the entities it
    omits, exactly as its preview or mint would, short of a dated read it could not make.

    A bulk plan, and a per-series plan under `latest`, omit first, as a read does. The read
    leaves out the rows of every entity no currency names among the rows it fetched, and
    only then converts them and, under `latest`, takes each StatVar's latest period
    (`materialise.generation._servable_rows`), so an omitted series cannot set it. Their
    omissions are therefore found over the rows the read fetches (`_plan_omissions`), the
    plan drops them (`ReadPlan.omitting`, which moves a `latest` period to the newest one a
    remaining series or pair holds), and the check runs over that plan (`_omit_then_check`),
    so a member whose rows all bound an omitted entity is never checked. One pass suffices:
    the remaining rows bind no entity left to omit.
    """
    if plan is None or not elected:
        return UnitCoverage()
    graph = conversion_graph_reader(graph, stat_vars=list(elected))
    check = partial(
        _coverage,
        elected=elected,
        declared_roles=declared_roles,
        graph=graph,
        sdmx=sdmx,
        rate_cache=rate_cache,
        deadline=deadline,
    )
    if isinstance(plan, BulkPlan) or plan.years == "latest":
        omissions = _plan_omissions(
            plan, elected, declared_roles=declared_roles, graph=graph, deadline=deadline
        )
        return _omit_then_check(plan, omissions, check=check)
    coverage = check(plan)
    rate_gap = coverage.refusal is not None and isinstance(
        coverage.refusal[1], RateCoverageGapError
    )
    if (
        not (rate_gap or coverage.omitted_entities)
        or not isinstance(plan, PairPlan)
        or dated_facets is None
    ):
        return coverage
    dated = _dated_plan(plan, dated_facets, deadline=deadline)
    return coverage if dated is None else check(dated)


def _omit_then_check(
    plan: ReadPlan,
    omissions: tuple[str, Omissions | NonCurrencyUnitError],
    *,
    check: Callable[[ReadPlan], UnitCoverage],
) -> UnitCoverage:
    """`unit_coverage` of `plan` given the rows its read omits before converting
    (`omissions`, with the first StatVar omitting any). It runs the check over the plan less
    those rows, with their omissions added. The currency read's refusal is the first
    StatVar's. A plan left serving nothing refuses `currency_unmapped` naming the omitted
    entities."""
    first_sv, omitted = omissions
    if isinstance(omitted, NonCurrencyUnitError):
        return UnitCoverage(refusal=(first_sv, omitted))
    if not omitted:
        return check(plan)
    narrowed = plan.omitting(omitted)
    if serves_nothing(narrowed):
        entities = {entity for _sv, _role, entity in omitted}
        return UnitCoverage(refusal=(first_sv, currency_unmapped_error(entities)))
    coverage = check(narrowed)
    if coverage.refusal is not None:
        return coverage
    return replace(coverage, omitted=coverage.omitted | omitted)


def _fetched_entities(
    plan: ReadPlan,
) -> dict[str, dict[_FacetKey, Callable[[str], frozenset[str]]]]:
    """Each StatVar's facet keys among the rows `plan`'s read fetches before it omits any,
    each with the entities those rows bind on a graph role. A bulk read fetches its served
    members' pairs under the elected facet and the facets continuing it
    (`BulkPlan.served_candidates`), over every year under `latest`, the pairs each
    member's year is elected or dated from before the read reduces it to that year, and
    the member's years (`BulkPlan.member_years`) otherwise. A per-series read fetches its
    held series (`PairPlan.held`), every series with data under `latest`, under their group
    facets."""
    if isinstance(plan, BulkPlan):
        best = plan.elected
        if best is None:
            return {}
        key: _FacetKey = (best.unit, best.provenance_id)
        fetched: dict[str, dict[_FacetKey, Callable[[str], frozenset[str]]]] = {}
        for sv in plan.served_members:
            first, last = (None, None) if plan.years == "latest" else plan.member_years(sv)
            fetched[sv] = {key: partial(_pair_members, plan.served_candidates(sv), first, last)}
        return fetched
    by_sv: dict[str, dict[_FacetKey, list[PlanSeries]]] = {}
    for series in plan.held:
        for f in series.group:
            keys = by_sv.setdefault(series.stat_var, {})
            keys.setdefault((f.unit, f.provenance_id), []).append(series)
    return {
        sv: {key: partial(_series_members, held) for key, held in keys.items()}
        for sv, keys in by_sv.items()
    }


def _pair_members(
    candidates: Sequence[RoleFacetCandidate], first: int | None, last: int | None, role: str
) -> frozenset[str]:
    """The entities on `role` the pairs of `candidates` hold data for in `first`..`last`."""
    return frozenset().union(
        *(
            c.pairs.members_with_data(first_year=first, last_year=last, role=role)
            for c in candidates
        )
    )


def _series_members(series: Sequence[PlanSeries], role: str) -> frozenset[str]:
    """The entities `series` bind on `role`."""
    return frozenset(dcid for s in series for r, dcid in s.role_tuple if r == role)


def _plan_omissions(
    plan: ReadPlan,
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]],
    *,
    declared_roles: tuple[str, ...],
    graph: GraphReader,
    deadline: float | None,
) -> tuple[str, Omissions | NonCurrencyUnitError]:
    """The rows `plan`'s read leaves out before it converts: for each facet key of a StatVar
    whose elected conversion targets each entity's own currency, the entities on that role
    among the rows the read fetches (`_fetched_entities`) that no single currency names,
    resolved in one read (`recipes.resolve_entity_currencies`). Returned with the first
    StatVar omitting any, or with the `NonCurrencyUnitError` that read raises. A key whose
    decision refuses is left to the check, which refuses it."""
    served: list[tuple[str, str, frozenset[str]]] = []
    for sv, by_key in _fetched_entities(plan).items():
        for key, entities_on in by_key.items():
            selections = facet_selections(elected.get(sv, ()), key)
            if selections is None:
                continue
            try:
                decision = coverage_decision(
                    stat_var=f"{_STAT_VAR_REF_PREFIX}{sv}",
                    declared_roles=declared_roles,
                    selections=selections,
                    graph=graph,
                    deadline=deadline,
                )
            except (RecipeNotDeclaredError, NonCurrencyUnitError):
                continue
            role = decision.graph_role if decision is not None else None
            if decision is None or decision.uniform_target is not None or role is None:
                continue
            if entities := entities_on(role):
                served.append((sv, role, entities))
    if not served:
        return "", frozenset()
    first_sv = served[0][0]
    try:
        currencies = resolve_entity_currencies(
            graph,
            entities=sorted(frozenset().union(*(entities for _sv, _role, entities in served))),
            deadline=deadline,
        )
    except NonCurrencyUnitError as exc:
        return first_sv, exc
    unmapped = frozenset(currencies.unmapped)
    omitted = frozenset(
        (sv, role, entity) for sv, role, entities in served for entity in entities & unmapped
    )
    first = next((sv for sv, _role, entities in served if entities & unmapped), first_sv)
    return first, omitted


@dataclass(frozen=True)
class UnitGate:
    """The bound-unit gate's verdict before an answer's conversion runs: `reason` is the
    no-data reason it answers, or None when the conversion runs, read on `plan`."""

    plan: ReadPlan | None
    reason: str | None


def unit_gate(
    mat: Materialised,
    plan: ReadPlan | None,
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None,
    obs_entities: Mapping[str, Sequence[str]],
    *,
    graph: GraphReader | None,
    deadline: float | None,
) -> UnitGate:
    """The bound-unit gate (`bound_unit_gate_reason`) on the plan an answer's read serves
    from, which a fresh request, a resubmit and each candidate read before their conversion.

    That plan is `plan`, except under `latest`: its read omits the rows of entities no
    currency names before it takes each StatVar's latest period, so the gate reads the plan
    less those rows (`_plan_omissions`, one currency read and no rate table), whose latest
    periods are the ones the read serves. A windowed or dateless plan keeps its years
    whatever it omits, so the gate reads it as it is. A currency read that refuses
    leaves the plan as it is, for the conversion to refuse. The final gate reads the
    returned plan until the conversion narrows the answer's own."""
    gate_plan = plan
    if plan is not None and plan.years == "latest" and elected and graph is not None:
        _first, omitted = _plan_omissions(
            plan,
            elected,
            declared_roles=survivor_declared_roles(mat),
            graph=conversion_graph_reader(graph, stat_vars=list(elected)),
            deadline=deadline,
        )
        if omitted and not isinstance(omitted, NonCurrencyUnitError):
            gate_plan = plan.omitting(omitted)
    return UnitGate(
        plan=gate_plan,
        reason=bound_unit_gate_reason(gate_plan, obs_entities, unit_served(elected, mat.sv_dcids)),
    )


def convertible_answer(
    mat: Materialised,
    filtered: FilteredAnswer,
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None,
    *,
    graph: GraphReader | None,
    sdmx: MaterialiseSdmxClient | None,
    rate_cache: RateTableCache | None,
    dated_facets: DatedFacetReader | None,
    deadline: float | None,
    date_request: DateRequest | None,
    coverage_index: CoverageIndex,
) -> tuple[FilteredAnswer, UnitCoverage]:
    """The answer `filtered` serves once its elected conversion has run, with what that
    conversion does to its rows (`unit_coverage`).

    When the conversion leaves out the rows of entities no currency names and the rest
    convert, the plan drops them (`ReadPlan.omitting`) and the coverage and headline are
    built again from it (`facets.plan_answer`). The grounding, sources, dates, scope and
    preview each answer reads from that plan then state only the rows it serves, as its
    preview or mint does. A refusal, or a conversion omitting nothing, leaves the answer as
    it is; so does an answer with no plan, no election or no materialise client, whose
    coverage is empty. The bound-unit gate reports the refusal or the omission warning
    from the returned coverage."""
    plan = filtered.plan
    if plan is None or not elected or graph is None or sdmx is None:
        return filtered, UnitCoverage()
    coverage = unit_coverage(
        plan,
        elected,
        declared_roles=survivor_declared_roles(mat),
        graph=graph,
        sdmx=sdmx,
        rate_cache=rate_cache,
        dated_facets=dated_facets,
        deadline=deadline,
    )
    if coverage.refusal is not None or not coverage.omitted:
        return filtered, coverage
    surviving = plan_answer(
        mat,
        plan.omitting(coverage.omitted),
        date_request=date_request,
        coverage_index=coverage_index,
    )
    return surviving, coverage


def unit_request_period(
    plan: ReadPlan | None,
    unit_request: IdentityUnitRequest,
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]],
    *,
    declared_roles: tuple[str, ...],
    catalogue: ConversionCatalogue | None,
) -> ConversionPeriod | None:
    """The period `unit_request` shows of the answer `plan` serves under its elected units,
    when the conversion tables cover only part of the plan's served years
    (`unit.conversion.conversion_period`). Every door reads it after its unit election, and
    reads the answer again over the period's window.

    The served years run from the first to the last year the plan serves
    (`read_plan.served_dates`). The years that convert are those
    `unit.conversion.ConversionAvailability` gives for the rows the elected units convert
    (`_reads_tables`): from their facets' currencies into the target currency, or into each
    currency the role's served entities are named by, and at constant prices through each
    entity the plan serves on the recipe's deflator role, in the request's base year. The
    offers read the same availability, so a period never starts before a year they state.

    None when the plan serves `latest` or no dated year, its facets name no single recipe or
    a unit outside a currency, no catalogue is loaded, the request's base-year terms are
    missing, or the tables cover every served year or none. In each of those cases the
    conversion runs, or refuses, without a narrowed period."""
    if catalogue is None or plan is None or plan.years == "latest":
        return None
    served = _served_span(plan)
    keys = {
        key
        for sv, by_key in _served_keys(plan).items()
        for key in by_key
        if _reads_tables(facet_selections(elected.get(sv, ()), key))
    }
    if not keys:
        return None
    recipes = {catalogue.recipe(provenance) if provenance else None for _unit, provenance in keys}
    natives = {native_currency_iso(unit) for unit, _provenance in keys}
    recipe = next(iter(recipes)) if len(recipes) == 1 else None
    if served is None or recipe is None or None in natives:
        return None
    tables = catalogue.tables[recipe.recipe_id]
    deflator_entities: frozenset[str] = frozenset()
    try:
        targets = _target_currencies(plan, unit_request, declared_roles, catalogue)
        if unit_request.prices == "constant":
            role = resolve_deflator_role(recipe, declared_roles=declared_roles)
            deflator_entities = served_role_entities(plan, role)
    except RecipeNotDeclaredError:
        return None
    if not targets:
        return None
    availability = ConversionAvailability.of(
        tables,
        recipe,
        natives=[code for code in natives if code is not None],
        deflator_entities=deflator_entities,
    )
    converted = availability.converted_years(
        years=frozenset(range(served[0], served[1] + 1)),
        targets=targets,
        prices=unit_request.prices,
        base_year=unit_request.base_year,
    )
    return conversion_period(served=served, converted=converted)


def over_conversion_period(mat: Materialised, period: ConversionPeriod) -> Materialised:
    """`mat` with its read plan read over `period`'s window (`ReadPlan.windowed`), the
    answer the same request builds under that window. The door then filters the plan again
    and states the answer from it."""
    plan = mat.read_plan
    return mat if plan is None else replace(mat, read_plan=plan.windowed(period.window))


def conversion_period_departure(
    period: ConversionPeriod, *, requested: TimeWindow | None, plan: ReadPlan | None
) -> tuple[Warning, DepartureFact]:
    """The `date_coverage_partial` warning and the `year_unavailable` fact it discloses for an
    answer narrowed to `period`: the table that leaves the data's other years out, the
    `requested` range, and the years the narrowed `plan` serves."""
    first, last = period.years
    shown = (_served_span(plan) if plan is not None else None) or period.years
    table = "exchange rates" if period.limited_by == "exchange_rates" else "deflators"
    warning = Warning(
        code=DATE_COVERAGE_PARTIAL,
        severity="warn",
        message=(
            f"the {table} this conversion reads cover {first} to {last} of the years the "
            f"data holds ({period.served[0]} to {period.served[1]}), so the answer shows "
            f"{shown[0]} to {shown[1]}."
        ),
    )
    fact = conversion_period_fact(
        served=period.served,
        converted=period.years,
        limited_by=period.limited_by,
        requested=(requested.start_year, requested.end_year) if requested is not None else None,
        shown=shown,
    )
    return warning, fact


def deflator_omission_departure(
    plan: ReadPlan | None,
    unit_request: IdentityUnitRequest,
    *,
    coverage_index: CoverageIndex | None,
    catalogue: ConversionCatalogue | None,
) -> tuple[Warning, DepartureFact] | None:
    """The `deflator_missing_omitted` warning and the `year_unavailable` fact it discloses
    for an answer whose constant-price conversion leaves out the rows of entities whose
    deflator the tables lack in some served years, or in the base year
    (`download.deflator_omissions`, over the rows the answer's counts read). The fact names
    the entities that lose the most years first, by the name the catalogue holds for each
    (`ConversionCatalogue.entity_names`), and counts those it holds no name for. The
    warning's `params` carry the entities it counts, named from the same catalogue, with
    their years (`_omitted_params`). When the conversion cannot be read in memory
    (`unit.conversion.counted_conversion` is `UNDECIDED`), a constant-price answer warns,
    with no `params`, that rows may be left out and departs naming no entity, since the mint still leaves out the rows its deflators lack. None
    when no row is left out or at current prices. Every door reads it after the period a
    unit request shows (`unit_request_period`)."""
    if unit_request.prices != "constant":
        return None
    conversion = counted_conversion(
        plan, {"unit": IdentityFacetValueRequest(request=unit_request)}, catalogue=catalogue
    )
    if conversion == UNDECIDED:
        warning = Warning(
            code=DEFLATOR_MISSING_OMITTED,
            severity="warn",
            message=(
                "a constant-price conversion leaves out each entity's rows in the years the "
                "deflator table holds no value for it; the loaded conversion tables cannot "
                "say which entities, so some rows may be left out."
            ),
        )
        return warning, deflator_omission_undecided_fact()
    if plan is None or catalogue is None or not isinstance(conversion, CountedConversion):
        return None
    omitted = deflator_omissions(plan, coverage_index, conversion)
    if not omitted:
        return None
    built = deflator_missing_omitted_warning(
        (entity, year) for entity, years in omitted.items() for year in years.lost
    )
    counted = built["omitted_entity_years"]
    ranked = sorted(omitted.items(), key=lambda item: (-len(item[1].lost), item[0]))
    names = catalogue.entity_names
    fact = deflator_omission_fact(
        [
            (names[entity], tuple(sorted(years.lost)), tuple(sorted(years.kept)))
            for entity, years in ranked
            if entity in names
        ],
        unnamed=sum(1 for entity in omitted if entity not in names),
    )
    warning = Warning(
        code=DEFLATOR_MISSING_OMITTED,
        severity=built["severity"],
        message=built["message"],
        params={
            **_omitted_params(
                counted["count"],
                [entity["ref"] for entity in counted["entities"]],
                catalogue=catalogue,
            ),
            "entity_years": [", ".join(entity["years"]) for entity in counted["entities"]],
        },
    )
    return warning, fact


def _omitted_params(
    count: int, refs: Sequence[str], *, catalogue: ConversionCatalogue | None
) -> dict[str, int | str | list[str]]:
    """The `params` of a conversion's omission warning: `count`, the entities it counts;
    `entities`, the bare dcids of `refs`, the `dcid:` refs of the first ten it carries; and
    `entity_labels`, each one's name in `catalogue` (`ConversionCatalogue.entity_names`),
    `""` where it holds none, in `entities` order. The catalogue names deflator-table
    entities and places with a national currency, so an entity no currency names, such as
    a regional aggregate, usually has `""`; its dcid stays in `entities`."""
    names = catalogue.entity_names if catalogue is not None else {}
    entities = [ref.removeprefix(_STAT_VAR_REF_PREFIX) for ref in refs]
    return {
        "count": count,
        "entities": entities,
        "entity_labels": [names.get(entity, "") for entity in entities],
    }


def _reads_tables(selections: Mapping[str, FacetSelection] | None) -> bool:
    """Whether rows served under `selections` (`facet_selections`) read conversion tables:
    a request-form unit other than a current-price request for the facet's own currency,
    which is natively satisfied, as `member_conversion.coverage_decision` reads it."""
    if selections is None or (request := selections["unit"].request) is None:
        return False
    return not (
        request.prices == "current"
        and request.currency == native_currency_iso(selections["unit"].ref)
    )


def _served_span(plan: ReadPlan) -> tuple[int, int] | None:
    """The first and last years `plan` serves across its StatVars (`served_dates`), or None
    when a StatVar's bound is unknown or none is served."""
    years = [year_of(date) for span in served_dates(plan).values() for date in span]
    if not years or None in years:
        return None
    known = [year for year in years if year is not None]
    return min(known), max(known)


def _target_currencies(
    plan: ReadPlan,
    unit_request: IdentityUnitRequest,
    declared_roles: tuple[str, ...],
    catalogue: ConversionCatalogue,
) -> frozenset[str]:
    """The currencies `unit_request` converts the plan's rows into: its ISO code, or each
    national currency of the entities the plan serves on the role it names
    (`graph_role_for_spec_role`), leaving out an entity no single currency names, whose
    rows the conversion omits. Raises `RecipeNotDeclaredError` for a role the members do
    not declare."""
    currency = unit_request.currency
    if isinstance(currency, str):
        return frozenset({currency})
    role = graph_role_for_spec_role(spec_role=currency.role, declared_roles=declared_roles)
    return frozenset(
        code
        for entity in served_role_entities(plan, role)
        if (code := catalogue.national_currencies.get(entity)) is not None
    )
