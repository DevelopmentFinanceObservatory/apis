"""Generates member, group, and preview frames.

`materialise.finish` builds content versions, warnings, statistics, and traces
once generation returns.
"""

from __future__ import annotations

import io
import logging
import math
import threading
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import cache, partial
from typing import TYPE_CHECKING, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dc_sdmx import (
    DEFAULT_PLANNING_LANES,
    ContainedIn,
    EntityFilter,
    SdmxCapExceededError,
    SdmxClient,
    SdmxIncompatibleVariablesError,
    observation_columns,
    observation_schema,
)
from dfo_platform.executor import submit_with_context
from materialise import columnar
from materialise.ceiling import check_ceiling
from materialise.content_version import stream_frame_sha256
from materialise.continued_series import ContinuedSeries
from materialise.convert import ConversionRecord, merged_record
from materialise.errors import (
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    MaterialisationError,
    MaterialisationIssue,
    MaterialisationUsageError,
    MembersNotHeldError,
    MixedGrainError,
    OverCeilingError,
    UpstreamResponseError,
)
from materialise.facets import (
    _SUBJECT_ROLE_ONLY,
    FACET_AXES,
    CollectionFacetProbe,
    CoverageAdmission,
    FacetBinding,
    FacetElection,
    FacetSelection,
    MemberProbe,
    ProvenanceRefresh,
    RowAdmission,
    VariableBreadth,
    _alternatives,
    _bound_roles_for_coverage,
    _probe_provenance_refresh,
    admitted_candidates,
    cadences_by_member,
    continued_election,
    dataset_candidates,
    declared_default_policies,
    default_scope_of,
    facet_election,
    fetch_role_bindings,
    joint_dispatch_facet_ids,
    joint_matches,
    overlaps_window,
    probe_collection_candidates,
    probe_facets,
    probe_from_candidates,
    request_scope_key,
    role_bearing_facet_candidates,
    select_facet,
    selected_continued_series,
)
from materialise.frame_build import (
    Frame,
    ParquetFrameChunks,
    build_frame_from_member_tables,
    build_frame_from_table,
    fold_method_periods,
    period_grains,
)
from materialise.graph import Facet, GraphReader, GraphReadError
from materialise.group_facets import (
    _as_issue,
    group_breadth,
    group_variable_breadth,
    select_group_facets,
)
from materialise.member_conversion import (
    MemberConversionPlan,
    apply_member_conversion,
    apply_member_conversion_by_series,
    conversion_graph_reader,
    omit_unmapped_rows,
    plan_member_conversion,
    without_unmapped,
)
from materialise.national_currency import CoverageGap, narrow_to_national_currency
from materialise.observation_rows import stream_observation_batches
from materialise.place_scope import NO_DEFAULT_SCOPE, AvailabilityReader, DefaultPlaceScope
from materialise.progress import RowProgress
from materialise.rate_tables import RateTableCache
from materialise.recipes import EntityCurrencies, NonCurrencyUnitError
from materialise.series_conversion import SeriesConversion, SeriesKey, series_key
from materialise.statistics import (
    SeriesStatisticsFile,
    SeriesStatisticsSink,
    compute_series_statistics_table,
)
from materialise.window import (
    AppliedWindow,
    Coverage,
    LatestYear,
    ReadSeries,
    RequestedWindow,
    compute_applied_window,
    latest_period,
)
from origin_catalogue.graph import is_un_dcid
from series_coverage.schema import ABSENT_ENTITY, RoleNames

if TYPE_CHECKING:
    from materialise.pipeline import MaterialisationRequest

    # Annotation-only import. `series_coverage.index` requires NumPy, absent
    # from `materialise`'s pure install.
    from series_coverage.coverage_types import BoundRoles, Role
    from series_coverage.index import CoverageIndex

logger = logging.getLogger(__name__)

# Mirrors `frame_build._STAT_VAR_REF_PREFIX`. Graph and SDMX filters use bare
# dcids, so this module strips the prefix only at wire call sites.
_STAT_VAR_REF_PREFIX = "dcid:"


def bare_stat_var_ref(stat_var: str) -> str:
    """Returns the bare dcid used by graph probes and SDMX filters.

    Requests keep the `dcid:` prefix required by `build_frame_from_table`.
    """
    return stat_var.removeprefix(_STAT_VAR_REF_PREFIX)


# Raw SDMX rows are capped at 20 x `ceiling`. All/range reads fetch full
# histories, so survivor limits cannot bound responses whose filters retain
# few rows. Latest reads fetch one row per matched series, then reduce again
# to the member's one period. A member read again at the year it elected
# (`_with_elected_year`) returns at most one row per series the first read held.
#
# A single-member fetch gets its own 20 x allowance. Group chunks share one
# 20 x `ceiling` total across workers. The total bounds
# cumulative rows, not peak memory. Each stream also has `max_rows=total`.
# Checks run after each batch, so the total can overshoot by one batch per
# active chunk.
_RAW_FETCH_ROWS_GUARD_MULTIPLIER = 20


class _RawRowBudget:
    """Thread-safe `20 * ceiling` row budget shared by all group chunks.

    It bounds cumulative rows, while peak memory depends on in-flight chunks.
    Single-member fetches use a separate local `20 * ceiling` guard.
    """

    def __init__(self, ceiling: int) -> None:
        self.total = _RAW_FETCH_ROWS_GUARD_MULTIPLIER * ceiling
        self._used = 0
        self._lock = threading.Lock()

    def consume(self, rows: int = 1) -> int:
        """Adds `rows` under a lock and returns the shared running total.

        The caller raises `OverCeilingError` with chunk-specific context.
        """
        with self._lock:
            self._used += rows
            return self._used


def _time_period_for_window(
    window: RequestedWindow, *, year: int | None = None
) -> Literal["LATEST"] | list[str] | None:
    """Uses server-side `LATEST` for latest windows and no time filter otherwise. With
    `year`, the year a latest read's member elected, the exact period of that year.

    The client still selects the member's one period because SDMX returns one latest row
    per matched series. SDMX takes exact period strings alone, so a year is asked for by
    its four digits, the period a yearly facet dates it by. A member elects a year only
    when no facet of its states a period finer than a year.
    """
    if year is not None:
        return [str(year)]
    return "LATEST" if window.kind == "latest" else None


type _Election = Callable[[Sequence[tuple[str, str]]], int | None]
"""The year a `latest` read of one member elected, None when it elects none. It is given
the `(graph role, entity)` rows a conversion leaves out of the member's rows, since the
year is counted over the series the read keeps."""


@dataclass(frozen=True)
class _ElectedRead:
    """How a `latest` read of one member learns the year it elected (`elect`) and reads the
    member's rows at a year (`read`), which hands them over as the `LATEST` rows arrived:
    under the member's dispatched facets and the read's own row filters. `counted` gives
    the entities on a graph role whose series the election counts (`elected_entities`), so
    a conversion's omissions among them are known before it elects; None for an election
    that leaves its own omissions out, as Search's plan does."""

    elect: _Election
    read: Callable[[int], pa.Table]
    counted: Callable[[str], Sequence[str]] | None = None


def elected_latest_year(
    request: MaterialisationRequest,
    omitted: Sequence[tuple[str, str]],
    *,
    coverage: CoverageIndex | None,
    facet_ids: Sequence[str],
) -> int | None:
    """The year a `latest` read of role-bearing `request` serves, elected by the coverage
    index (`CoverageIndex.elected_years`): over the series of `facet_ids`, the facets the
    read dispatches, under the request's bound roles and the source scope in context, less
    the `omitted` `(graph role, entity)` rows a conversion leaves out. None for a window
    other than `latest`, a subject-only request, one read without coverage or dispatching
    no facet, and a StatVar the index cannot elect for, which all serve the newest period
    any series holds.

    A mint calls it once per member and records the year with `facet_ids` and `omitted`
    (`window.LatestYear`), so a later call with those arguments tells whether the index
    still elects that year."""
    scope = _election_scope(request, coverage=coverage, facet_ids=facet_ids)
    if scope is None:
        return None
    stat_var, names, bound = scope
    assert coverage is not None  # `_election_scope` holds none without it
    coverage_role: dict[str | None, Role] = {names.donor: "donor", names.recipient: "recipient"}
    return coverage.elected_years(
        variables=[stat_var],
        bound=bound,
        served=[(stat_var, facet_id, None, None) for facet_id in facet_ids],
        excluded=[
            (stat_var, coverage_role[role], entity)
            for role, entity in omitted
            if role in coverage_role
        ],
    ).get(stat_var)


def elected_entities(
    request: MaterialisationRequest,
    graph_role: str,
    *,
    coverage: CoverageIndex | None,
    facet_ids: Sequence[str],
) -> tuple[str, ...]:
    """The entities on `graph_role` of the series `elected_latest_year` counts for
    `request` over `facet_ids`, in dcid order, whether or not a row of theirs holds a
    value. Empty where it elects nothing, and for a role the index does not key."""
    scope = _election_scope(request, coverage=coverage, facet_ids=facet_ids)
    if scope is None:
        return ()
    stat_var, names, bound = scope
    assert coverage is not None  # `_election_scope` holds none without it
    position = {names.recipient: 1, names.donor: 2}.get(graph_role)
    found = coverage.series_years(
        variables=[stat_var],
        bound=bound,
        served=[(stat_var, facet_id, None, None) for facet_id in facet_ids],
    )
    if position is None or found is None:
        return ()
    series = found.keys()
    return tuple(sorted({key[position] for key in series} - {ABSENT_ENTITY}))


def _election_scope(
    request: MaterialisationRequest, *, coverage: CoverageIndex | None, facet_ids: Sequence[str]
) -> tuple[str, RoleNames, BoundRoles] | None:
    """`request`'s bare StatVar, the index's role names for it and its bound roles, for a
    `latest` read of a role-bearing request over some dispatched facet of a StatVar the
    index holds role names for; None otherwise."""
    stat_var = bare_stat_var_ref(request.stat_var)
    if (
        request.window.kind != "latest"
        or coverage is None
        or not facet_ids
        or request.declared_roles == _SUBJECT_ROLE_ONLY
    ):
        return None
    names = coverage.role_names.get(stat_var)
    bound = _bound_roles_for_coverage(request.role_bindings, coverage=coverage, stat_var=stat_var)
    if names is None or bound is None:
        return None
    return stat_var, names, bound


def _omitted_rows(
    conversion_plan: MemberConversionPlan | None,
    currencies: EntityCurrencies | None,
    *,
    also: Collection[str] = (),
) -> tuple[tuple[str, str], ...]:
    """The `(graph role, entity)` rows a role-shaped conversion leaves out, in dcid order:
    the entities `currencies` holds as unmapped (`omit_unmapped_rows`) and `also`, on the
    plan's role."""
    if conversion_plan is None or conversion_plan.graph_role is None or currencies is None:
        return ()
    role = conversion_plan.graph_role
    return tuple((role, entity) for entity in sorted({*currencies.unmapped, *also}))


def _continuing_facet_ids(
    candidates: Sequence[Facet], election: FacetElection, *, member: int
) -> tuple[str, ...]:
    """The dispatch IDs (`joint_dispatch_facet_ids`) of the method that continues the
    selected series past its last year (`FacetSelection.continued_by`), none when nothing
    continues it. A range window alone continues a series, and its rows serve only the
    years after that series' last one (`columnar.continuation_mask`)."""
    method = election.selections.get("measurementMethod")
    if method is None or method.continued_by is None:
        return ()
    by_id = {c.facet_id: c for c in candidates}
    return tuple(
        facet_id
        for facet_id in joint_dispatch_facet_ids(candidates, election, member=member)
        if by_id[facet_id].measurement_method == method.continued_by
    )


def _read_series(
    dispatch_ids: Sequence[str], continuing_ids: Sequence[str], omitted: Sequence[tuple[str, str]]
) -> ReadSeries:
    """The series a member's read holds of its selection: those of `dispatch_ids` less the
    facets of a method continuing them (`continuing_ids`), less `omitted`. They are the
    series a `latest` read of the request holds, which no method continues, so a named
    year's counts are those a `latest` frame states."""
    continuing = set(continuing_ids)
    return ReadSeries(
        facet_ids=tuple(facet_id for facet_id in dispatch_ids if facet_id not in continuing),
        omitted=tuple(omitted),
    )


def _latest_years(
    request: MaterialisationRequest,
    *,
    elected: int | None,
    applied_window: AppliedWindow | None,
    read: ReadSeries,
) -> dict[str, LatestYear]:
    """`request`'s entry of a result's `latest_years`: the year it elected over the series
    it `read`, and the year its rows hold (`applied_window`). Empty for a window other than
    `latest`."""
    if request.window.kind != "latest":
        return {}
    return {
        request.stat_var: LatestYear(
            elected=elected,
            served=applied_window.end_year if applied_window is not None else None,
            facet_ids=read.facet_ids,
            omitted=read.omitted,
        )
    }


@dataclass(frozen=True)
class _MemberOutcome:
    """One member's frame and summary, consumed by `materialise.finish` after revision reads.

    `coverage` and `served_entities_by_role` replace the survivor table to
    avoid retaining rows across a group generation. `conversion_record`
    is `None` when no arithmetic ran. `conversion_recipe_id` is `None` for a
    native or non-request unit; a conversion plan resolves it before fetch.
    `default_scope` is the member's default place scope (`MemberProbe.scope`), for a
    request that binds no entity. `series_conversions` holds each converted series' record
    (`_convert_by_series`). `probe_empty` is true when the member's probe returned no
    candidate, and the outcome then holds no row and no selection. `provenance_refresh` holds
    the dispatched provenance's refresh dates, read before the fetch (`_dispatch_refresh`).
    `coverage_gaps` holds the places a national-currency unit binding serves no row at
    (`_national_currency_probe`). `latest_years` holds the member's year under a `latest`
    window (`_latest_years`), and `read_series` the series its read holds under any window.
    """

    frame: Frame
    served_entities_by_role: dict[str, set[str]]
    selections: dict[str, FacetSelection]
    coverage: Coverage
    applied_window: AppliedWindow | None
    conversion_record: ConversionRecord | None
    conversion_recipe_id: str | None
    default_scope: DefaultPlaceScope = NO_DEFAULT_SCOPE
    series_conversions: dict[SeriesKey, SeriesConversion] = field(default_factory=dict)
    probe_empty: bool = False
    provenance_refresh: Mapping[str, ProvenanceRefresh] = field(default_factory=dict)
    coverage_gaps: tuple[CoverageGap, ...] = ()
    latest_years: Mapping[str, LatestYear] = field(default_factory=dict)
    read_series: Mapping[str, ReadSeries] = field(default_factory=dict)


@dataclass(frozen=True)
class _GroupOutcome:
    """One aggregate group result, consumed by `materialise.finish` after revision reads.

    The frame is rebuilt on demand from in-memory canonical Parquet, one row
    group at a time. `frame_sha256` and `series_statistics` are computed during
    the fetch pass, avoiding a second read. Pooled coverage and served entities
    summarise all rows; per-member counts and conversion details remain keyed
    by `stat_var` for trace and preview consumers. A conversion record is
    `None` when no arithmetic ran; recipe IDs are `None` for ref/unbound or
    native-satisfied units. `default_scope` is the group's default place
    scope, for members that bind no entity. `series_conversions` holds every
    member's converted series' records (`_convert_by_series`). `off_cadence_members` and
    `out_of_window_members` are `_GroupPlan`'s. `probe_empty` is true when no member's probe returned a
    candidate, and the outcome then holds no row and no selection. `provenance_refresh` is
    `_MemberOutcome.provenance_refresh` for the group's shared selection. `method_periods`
    maps each `measurementMethod` the rows name to the earliest and latest period its rows
    hold (`frame_build.fold_method_periods`). `latest_years` holds each member's year under
    a `latest` window (`_latest_years`), and `read_series` the series each member's read
    holds under any window.
    """

    frame_chunks: Callable[[], Iterator[Frame]]
    frame_sha256: str
    series_statistics: SeriesStatisticsFile
    row_count: int
    served_entities_by_role: dict[str, set[str]]
    selections: dict[str, FacetSelection]
    coverage: Coverage
    applied_window: AppliedWindow | None
    member_observation_counts: dict[str, int]
    member_conversion_records: dict[str, ConversionRecord | None]
    member_conversion_recipe_ids: dict[str, str | None]
    default_scope: DefaultPlaceScope = NO_DEFAULT_SCOPE
    series_conversions: dict[SeriesKey, SeriesConversion] = field(default_factory=dict)
    off_cadence_members: tuple[str, ...] = ()
    out_of_window_members: tuple[str, ...] = ()
    probe_empty: bool = False
    provenance_refresh: Mapping[str, ProvenanceRefresh] = field(default_factory=dict)
    method_periods: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    latest_years: Mapping[str, LatestYear] = field(default_factory=dict)
    read_series: Mapping[str, ReadSeries] = field(default_factory=dict)


@dataclass(frozen=True)
class _PreviewOutcome:
    """Preview result with a limited frame and full-survivor summary.

    `total_rows`, coverage, and served entities describe every survivor.
    `frame` carries at most `limit` rows, or for a read with `series_roles`
    at most `series_limit` rows of each series the read returned.
    `series_total_rows` holds each series' survivors for a read with
    `series_roles`, keyed by the series' dcids on those roles, and is empty
    otherwise. `conversion_record` is `None` when
    no arithmetic ran. `conversion_recipe_id` is `None` for ref/unbound or
    native-satisfied units.

    `full_frame` contains all survivors only when `mint_full_frame=True`, for
    hashing a preview's content version. Otherwise it is `None` to avoid a
    second frame build. `default_scope` is `_MemberOutcome.default_scope`.

    `series_conversion_records` holds one record per series of a preview read
    with `series_roles`, keyed by the series' dcids on those roles, when
    conversion ran arithmetic. It is empty otherwise.
    """

    frame: Frame
    full_frame: Frame | None
    served_entities_by_role: dict[str, set[str]]
    selections: dict[str, FacetSelection]
    coverage: Coverage
    applied_window: AppliedWindow | None
    total_rows: int
    truncated: bool
    conversion_record: ConversionRecord | None
    conversion_recipe_id: str | None
    default_scope: DefaultPlaceScope = NO_DEFAULT_SCOPE
    series_conversion_records: dict[tuple[str, ...], ConversionRecord] = field(default_factory=dict)
    series_total_rows: dict[tuple[str, ...], int] = field(default_factory=dict)


@dataclass(frozen=True)
class _PreparedPreview:
    """A preview after selection, admission and latest reduction, before arithmetic.

    Search uses this short-lived state to choose one actual latest period across
    several reads.  It is deliberately kept internal: ``pipeline`` exposes only
    the periods needed for that coordination and finalises it into a PreviewResult.
    `year` is the year a `latest` read elected, None when it elected none. `table` then
    holds the rows in that year beside each series' latest (`columnar.latest_rows`).
    """

    request: MaterialisationRequest
    probe: MemberProbe
    selections: Mapping[str, FacetSelection]
    conversion_plan: MemberConversionPlan | None
    table: pa.Table
    currencies: EntityCurrencies | None
    series_currencies: Mapping[tuple[str, ...], EntityCurrencies | None]
    limit: int
    ceiling: int
    mint_full_frame: bool
    series_roles: tuple[str, ...]
    series_limit: int | None
    year: int | None = None

    @property
    def latest_period(self) -> str | None:
        """The greatest retained period after admission and currency omissions."""
        if not self.table.num_rows:
            return None
        return latest_period(pc.unique(self.table["TIME_PERIOD"]).to_pylist())

    @property
    def elected_period(self) -> str | None:
        """The newest retained period in the year the read elected. None when it elected
        none or retains no row in that year."""
        if self.year is None or not self.table.num_rows:
            return None
        return latest_period(pc.unique(self.table["TIME_PERIOD"]).to_pylist(), year=self.year)


def _admitted(probe: MemberProbe, candidate_filter: Callable[[Facet], bool] | None) -> MemberProbe:
    """`probe` with only the candidates the request's facet filter admits
    (`admitted_candidates`), which every election then ranks. Its row admission is kept."""
    return replace(probe, candidates=tuple(admitted_candidates(probe.candidates, candidate_filter)))


def _national_currency_probe(
    request: MaterialisationRequest,
    probe: MemberProbe,
    *,
    graph: GraphReader,
    deadline: float | None,
) -> tuple[MemberProbe, tuple[CoverageGap, ...]]:
    """`probe` narrowed to each place's own-currency facets when the request's unit binds
    each place's national currency (`national_currency.narrow_to_national_currency`), with
    the places it leaves out; otherwise `probe` and no gap."""
    binding = request.facet_bindings.get("unit")
    if binding is None or binding.national_currency is None:
        return probe, ()
    return narrow_to_national_currency(
        probe, binding.national_currency, reader=graph, deadline=deadline
    )


def _select_member_facets(
    request: MaterialisationRequest,
    candidates: Sequence[Facet],
    *,
    graph: GraphReader,
    breadth: Mapping[str, int] | None = None,
    variable_breadth: VariableBreadth = None,
) -> FacetElection:
    """Resolves all facet axes for one member from the probed candidates, ranked with
    `breadth`, each facet's series count under `request.window` (`MemberProbe.breadth_in`),
    with `variable_breadth` (`MemberProbe.variable_breadth`) breaking a tie before the unit,
    and the cadence its dispatch serves: the cadence of the facet the same ranking puts first
    among the candidates matching every selected ref (`facets.dispatch_cadence`).

    The helper performs no graph reads, so a group can plan each member
    before fetching observations. Declared defaults resolve first, and
    the deployment-fallback pool keeps the facets that carry them before
    `request.window` narrows it to facets overlapping the requested years
    (`facets.fallback_pool`). It evaluates all axes and aggregates
    failures. `DefaultFacetUndefinedError` takes priority over
    `FacetNotAvailableError`; a lone `NonCurrencyUnitError` passes through,
    while mixed failures include it in the aggregate.

    With fewer than two bound axes, the selected refs must match one candidate
    jointly. Otherwise the helper raises `FacetNotAvailableError` with one
    issue per axis. An unbound method axis whose served series ends before a range window
    does names the estimated method that continues it (`facets.continued_election`). Two or more explicit bound axes skip this system-coherence
    check, so a valid caller filter that matches no candidate yields an empty
    result.

    A member of an index-resolved family selects among the candidates its elected
    dataset publishes (`facets.dataset_candidates`), so the selected provenance is that
    dataset's.
    """
    # The candidates its dispatch reads, each merged across places with the AND of its
    # places' annual evidence (`Facet.reads_annual`).
    (cadence_of,) = cadences_by_member([candidates])
    candidates = dataset_candidates(request, candidates)
    selections: dict[str, FacetSelection] = {}
    issues: list[MaterialisationIssue] = []
    codes_seen: set[str] = set()
    unit_error: NonCurrencyUnitError | None = None
    # One policy read per axis, shared so each fallback axis ranks only facets that carry
    # the other axes' declared defaults.
    policies = declared_default_policies(
        request.facet_bindings, graph=graph, stat_var=request.stat_var
    )
    for axis in FACET_AXES:
        binding = request.facet_bindings.get(axis, FacetBinding(state="unbound"))
        try:
            selections[axis] = select_facet(
                axis,
                binding,
                candidates=candidates,
                graph=graph,
                # `select_facet` calls `graph.default_facet_policy`, a
                # separate wire call from `graph.observation_facets_batch` --
                # only the latter keys on the bare dcid, so this keeps the
                # request's canonical `dcid:`-prefixed form.
                stat_var=request.stat_var,
                window=request.window,
                all_bindings=request.facet_bindings,
                policies=policies,
                breadth=breadth,
                variable_breadth=variable_breadth,
            )
        except NonCurrencyUnitError as exc:
            unit_error = exc
            codes_seen.add(exc.code)
            issues.append(_as_issue(axis, exc))
        except MaterialisationError as exc:
            codes_seen.add(exc.code)
            issues.append(_as_issue(axis, exc))
    if unit_error is not None and len(issues) == 1:
        raise unit_error
    if issues:
        if "default_facet_undefined" in codes_seen:
            raise DefaultFacetUndefinedError(
                "one or more facet axes could not be resolved",
                issues=tuple(issues),
                request=request,
            )
        raise FacetNotAvailableError(
            "one or more facet axes could not be resolved",
            issues=tuple(issues),
        )
    bound_axis_count = sum(1 for selection in selections.values() if selection.basis == "bound")
    matching = joint_matches(candidates, selections)
    if bound_axis_count < 2 and not matching:
        dispatchable = [c for c in candidates if c.facet_id is not None]
        coherence_issues: list[MaterialisationIssue] = []
        for axis in FACET_AXES:
            alternatives, truncated = _alternatives(dispatchable, axis, graph)
            coherence_issues.append(
                _as_issue(
                    axis,
                    FacetNotAvailableError(
                        facet_pointer=f"/facets/{axis}",
                        alternatives=alternatives,
                        alternatives_truncated=truncated,
                    ),
                )
            )
        raise FacetNotAvailableError(
            "each facet axis resolved on its own, but no probed candidate carries "
            "all three selected refs at once",
            issues=tuple(coherence_issues),
        )
    election = facet_election(
        selections,
        matching=matching,
        member_matches=[matching],
        cadence_of=[cadence_of],
        stat_vars=[request.stat_var],
        window=request.window,
        breadth=breadth,
        variable_breadth=variable_breadth,
    )
    return continued_election(election, [candidates], window=request.window, breadth=breadth)


def _validate_empty_member_probe(
    request: MaterialisationRequest,
    *,
    coverage: CoverageIndex | None,
    graph: GraphReader,
) -> None:
    """Validate explicit facet pins before an empty served role probe returns.

    The served probe respects the active source scope and per-series DFO
    precedence.  When that leaves it empty, a bound facet still needs the
    globally-admitted candidate view so an unavailable pin is reported as a
    client-correctable facet refusal.  No bound pin, no coverage index, or no
    global candidate stays the ordinary empty/no-data outcome.
    """
    if coverage is None or not any(
        binding.state == "bound" for binding in request.facet_bindings.values()
    ):
        return
    candidates = role_bearing_facet_candidates(
        stat_var=bare_stat_var_ref(request.stat_var),
        role_bindings=request.role_bindings,
        coverage=coverage,
    )
    if candidates:
        _select_member_facets(request, candidates, graph=graph)


def _log_no_joint_facet_match(stat_var: str, selections: Mapping[str, FacetSelection]) -> None:
    """Logs a skipped fetch when no candidate matches every selected ref.

    This DEBUG message has no wire or trace effect. `.get` handles group
    selections that omit unbound `measurementMethod` for index-sourced groups.
    """
    provenance = selections.get("provenance")
    measurement_method = selections.get("measurementMethod")
    unit = selections.get("unit")
    logger.debug(
        "no probed candidate jointly matched every selected facet ref for "
        "stat_var=%r; skipping the SDMX fetch (provenance=%r, measurementMethod=%r, "
        "unit=%r)",
        stat_var,
        provenance.ref if provenance is not None else None,
        measurement_method.ref if measurement_method is not None else None,
        unit.ref if unit is not None else None,
    )


def _raw_fetch_guard_error(
    request: MaterialisationRequest, exc: SdmxCapExceededError, *, raw_guard: int, ceiling: int
) -> OverCeilingError:
    """Builds the raw-fetch `OverCeilingError` for a single member.

    The cap applies before window and facet filtering, so `exc.observed`
    describes raw rows or bytes, not surviving coverage. Latest reads return
    one row per matched series. The exception reports the cap unit and count.
    """
    observed_unit = "bytes" if exc.cap_kind == "bytes" else "rows"
    return OverCeilingError(
        f"the raw SDMX upstream response for stat_var {request.stat_var!r} was "
        f"too large to filter synchronously (observed {exc.observed} "
        f"{observed_unit}, past the raw fetch guard of {raw_guard} rows)",
        detail=(
            f"the SDMX upstream read for stat_var {request.stat_var!r} was "
            f"aborted after {exc.observed} {observed_unit}, past the raw "
            f"fetch guard of {raw_guard} rows. This guard bounds "
            f"memory and transport on the unfiltered upstream response only. "
            f"Whether this request's actual (post-window/facet-filter) "
            f"coverage would have exceeded the sync materialisation ceiling "
            f"of {ceiling} rows was not measured"
        ),
        ceiling_kind=observed_unit,
        count=exc.observed,
        cap=exc.cap,
    )


def _split_by_roles(
    table: pa.Table, roles: Sequence[str]
) -> list[tuple[tuple[str, ...], pa.Table]]:
    """`table`'s rows grouped by their dcids on `roles`, in first-seen order: one group per
    series of a read that leaves those roles open."""
    columns = [table[role].to_pylist() for role in roles]
    rows_by_key: dict[tuple[str, ...], list[int]] = {}
    for row, key in enumerate(zip(*columns, strict=True)):
        rows_by_key.setdefault(key, []).append(row)
    return [(key, table.take(pa.array(rows, type=pa.int64()))) for key, rows in rows_by_key.items()]


def _convert_by_series(
    plan: MemberConversionPlan | None,
    table: pa.Table,
    *,
    request: MaterialisationRequest,
    currencies: EntityCurrencies | None,
) -> tuple[pa.Table, ConversionRecord | None, dict[SeriesKey, SeriesConversion]]:
    """Converts a mint's servable rows in one call that keeps each series apart, as a preview
    read with `series_roles` converts each series on its own, and returns the converted rows,
    the member's record, and each series' record keyed by `series_key`.

    A series is one dcid on every declared role. `apply_member_conversion_by_series` ranks
    facets within each series and splits the conversion's fallback and projection markers by
    series, so each series' record counts that series' rows alone and the member's record is
    `convert.merged_record` of them. A plan with no arithmetic, or a table with no rows,
    converts whole and records no series, so an empty converted member keeps its zero-count
    record."""
    roles = request.declared_roles
    if plan is None or plan.natively_satisfied or not table.num_rows or not roles:
        converted, record = apply_member_conversion(
            plan, table, declared_roles=roles, currencies=currencies
        )
        return converted, record, {}
    converted, record, records = apply_member_conversion_by_series(
        plan, table, declared_roles=roles, currencies=currencies
    )
    bare_stat_var = bare_stat_var_ref(request.stat_var)
    return (
        converted,
        record,
        {
            series_key(bare_stat_var, roles, dcids): SeriesConversion.from_record(series_record)
            for dcids, series_record in records.items()
        },
    )


def _admitted_series_rows(
    table: pa.Table, roles: Sequence[str], allowed: frozenset[tuple[str, ...]] | None
) -> pa.Table:
    """Keep only planned open-role tuples before omissions and latest reduction."""
    if not roles or allowed is None:
        return table
    keys = list(zip(*(table[role].to_pylist() for role in roles), strict=True))
    return table.filter(pa.array([key in allowed for key in keys], type=pa.bool_()))


def _most_recent_rows(table: pa.Table, limit: int) -> pa.Table:
    """`table`'s `limit` rows with the greatest `TIME_PERIOD` string, in period order, or
    `table` itself when it holds no more. Null periods sort first, so they are cut first, and
    a tie at the boundary period keeps the later rows in `table`'s order. A row-index column
    and `pc.sort_indices` (`null_placement="at_start"`, ascending on `TIME_PERIOD` then the
    row index) apply that stable-sort-then-tail rule columnwise."""
    if table.num_rows <= limit:
        return table
    indexed = table.append_column("_row_index", pa.array(range(table.num_rows), type=pa.int64()))
    order = pc.sort_indices(
        indexed,
        sort_keys=[("TIME_PERIOD", "ascending"), ("_row_index", "ascending")],
        null_placement="at_start",
    )
    return indexed.take(order[-limit:]).drop_columns(["_row_index"])


def _with_elected_year(
    table: pa.Table, *, year: int, admission: RowAdmission, read: Callable[[int], pa.Table]
) -> pa.Table:
    """`table`, a `LATEST` read's rows of one member, holding every row of the member in
    `year`, the year it elected.

    A `LATEST` read holds each series at its newest period, so a series ending after
    `year` shows no row in it. When an admitted row is newer than `year`, `read(year)`
    reads the member at that year, and its rows replace those `table` holds in `year`,
    which the second read returns again. When no series ends after `year` the first read
    holds every row in it and no second read is made. A series ending before `year` holds
    no row in it on either read.
    """
    bounds = columnar.served_year_bounds(
        table.filter(columnar.admission_mask(table, admission=admission))
    )
    if bounds is None or bounds[1] <= year:
        return table

    again = read(year)
    return pa.concat_tables(
        [
            table.filter(pc.invert(columnar.year_mask(table, year=year))),
            again.filter(columnar.year_mask(again, year=year)),
        ]
    )


def _apply_latest_reduction(
    table: pa.Table,
    *,
    window: RequestedWindow,
    selections: Mapping[str, FacetSelection],
    admission: RowAdmission,
    year: int | None = None,
) -> pa.Table:
    """Reduces latest-window rows to the member's one latest period (`columnar.latest_rows`,
    by `window.latest_period`), then filters them. With `year`, the year the member
    elected, the rows at the newest period in `year` stay beside them, and
    `columnar.elected_rows` chooses between the two once the rows are filtered.

    SDMX's `LATEST` filter returns one row per series. Admission filters
    before the reduction, so a withheld row cannot set the member's latest
    period. Other window kinds arrive pre-filtered and pass through unchanged.
    """
    if window.kind != "latest":
        return table
    table = table.filter(columnar.admission_mask(table, admission=admission))
    table = columnar.latest_rows(table, year=year)
    return table.filter(
        columnar.survivor_mask(table, selections=selections, window=window, admission=admission)
    )


def _servable_rows(
    table: pa.Table,
    *,
    window: RequestedWindow,
    selections: Mapping[str, FacetSelection],
    admission: RowAdmission,
    conversion_plan: MemberConversionPlan | None,
    ceiling_check: Callable[[pa.Table], None] | None = None,
    elected: _ElectedRead | None = None,
) -> tuple[pa.Table, EntityCurrencies | None, int | None, tuple[tuple[str, str], ...]]:
    """Omits the rows of entities no single currency names (`omit_unmapped_rows`), then
    reduces latest-window rows (`_apply_latest_reduction`), so a member's latest period is
    one a converted row holds, then keeps a continuing method's rows only past each
    series' last served period (`columnar.continuation_mask`). Returns the rows, the
    currencies `apply_member_conversion` converts them into, the year a `latest` read
    elected, and the `(graph role, entity)` rows the election counted without
    (`_omitted_rows`).

    A `latest` read given `elected` asks it for the member's year, counted without the
    omitted entities, and reads the rows in that year a `LATEST` read lacks
    (`_with_elected_year`). The returned rows then hold the newest period in that year
    beside the newest period any series holds, and the caller serves the first when a row
    holds it (`columnar.elected_rows`). A member that elects no year, and a read given no
    `elected`, keep the newest period any series holds alone.

    A latest window's rows arrive unfiltered, so currencies resolve for the rows that pass
    the facet and admission filters, whatever their value. These are each series' newest
    row, and so every entity the read of the elected year can add a row of. They resolve in
    the same read for the entities the election counts (`_ElectedRead.counted`), and the
    election runs once, before any row's value is read, without the ones no currency names.
    The returned currencies name as unmapped only the entities a row of either read holds a
    value for (`_holding_a_value`), which warnings and the empty result report, and one of
    those the election did not count is logged. Other window kinds arrive filtered. When the currency read fails on a filtered window
    (a `NonCurrencyUnitError`, a graph read or the deadline), `ceiling_check` runs on the
    rows before omission, a continuing method's earlier rows left out, and its
    `OverCeilingError` wins, then the failure re-raises. Those are the rows that would serve
    had no entity been omitted. A latest window converts before its ceiling check, so its
    failure raises as it is.
    """
    served = (
        table.filter(
            pc.and_(
                columnar.selection_mask(table, selections=selections),
                columnar.admission_mask(table, admission=admission),
            )
        )
        if window.kind == "latest"
        else table
    )
    role = conversion_plan.graph_role if conversion_plan is not None else None
    counted = (
        elected.counted(role)
        if elected is not None
        and elected.counted is not None
        and role is not None
        and window.kind == "latest"
        else ()
    )
    try:
        _, resolved = omit_unmapped_rows(conversion_plan, table, served=served, entities=counted)
    except Exception:
        if ceiling_check is not None and window.kind != "latest":
            ceiling_check(table.filter(columnar.continuation_mask(table, selections=selections)))
        raise
    currencies = _holding_a_value(resolved, conversion_plan, served)
    table = without_unmapped(conversion_plan, table, currencies)
    left_out = set(counted).intersection(resolved.unmapped) if resolved is not None else set()
    year = None
    if elected is not None and window.kind == "latest":
        year = elected.elect(tuple((role, entity) for entity in sorted(left_out) if role))
        if year is not None:
            read = _with_elected_year(table, year=year, admission=admission, read=elected.read)
            currencies = _holding_a_value(resolved, conversion_plan, served, read)
            table = without_unmapped(conversion_plan, read, currencies)
        if elected.counted is not None and currencies is not None:
            uncounted = sorted(set(currencies.unmapped) - left_out)
            if uncounted:
                # The election counts the index's series, so an entity the index holds no
                # series of changes no count. Its rows are left out as any unmapped entity's.
                logger.warning(
                    "latest read omits %d entities no currency names that the coverage "
                    "index holds no series of, so the election did not leave them out: %s",
                    len(uncounted),
                    uncounted[:10],
                )
    table = _apply_latest_reduction(
        table, window=window, selections=selections, admission=admission, year=year
    )
    rows = table.filter(columnar.continuation_mask(table, selections=selections))
    return rows, currencies, year, _omitted_rows(conversion_plan, currencies, also=left_out)


def _holding_a_value(
    currencies: EntityCurrencies | None,
    conversion_plan: MemberConversionPlan | None,
    *tables: pa.Table,
) -> EntityCurrencies | None:
    """`currencies` with only the unmapped entities that some row of `tables` binds, on the
    plan's role, with a value (`columnar.observed_mask`). An entity with no such row serves
    nothing, and a conversion leaves nothing of it out."""
    if currencies is None or conversion_plan is None or conversion_plan.graph_role is None:
        return currencies
    role = conversion_plan.graph_role
    holding = {
        entity
        for table in tables
        for entity in table.filter(columnar.observed_mask(table))[role].to_pylist()
    }
    return replace(
        currencies, unmapped=tuple(entity for entity in currencies.unmapped if entity in holding)
    )


def _counts_after_conversion(conversion_plan: MemberConversionPlan | None) -> bool:
    """Whether a member's served rows are counted only once converted (`_finish_member_table`).
    A constant-price conversion leaves out the rows a deflator lacks
    (`ConversionRecord.deflator_omitted`), which are known only when its terms are gathered,
    so a count of its rows before conversion would refuse a file whose served rows fit."""
    return (
        conversion_plan is not None
        and not conversion_plan.natively_satisfied
        and conversion_plan.unit_request.prices == "constant"
    )


def _streaming_ceiling(
    conversion_plan: MemberConversionPlan | None,
    selections: Mapping[str, FacetSelection],
    ceiling: int,
) -> int | None:
    """The survivor ceiling a member's rows are checked against as they stream: `ceiling`,
    or None for a selection a method continues or a conversion that omits rows. A continuing
    method's rows serve only past each series' last served period
    (`columnar.continuation_mask`), known only once every row has arrived, so its rows are
    counted once filtered and before conversion (`_check_servable_ceiling`). A role-shaped
    conversion omits the rows of entities no single currency names (`_servable_rows`), known
    only once every row has arrived, so its survivors are counted after omission and before
    conversion (`_check_servable_ceiling`). A constant-price conversion omits rows a deflator
    lacks, so its rows are counted after conversion (`_counts_after_conversion`).
    `over_ceiling` therefore precedes every conversion refusal on a filtered window of a
    current-price member: a rate or recipe gap, which conversion raises, and a failed
    currency read, before which the rows are counted as they arrived. A constant-price
    member, and a `latest` window, convert before they count, so their conversion refusals
    precede `over_ceiling`. The raw-fetch guard still bounds what the read holds."""
    if any(selection.continued_by is not None for selection in selections.values()):
        return None
    if conversion_plan is not None and (
        conversion_plan.spec_role is not None or _counts_after_conversion(conversion_plan)
    ):
        return None
    return ceiling


def _check_servable_ceiling(
    request: MaterialisationRequest,
    table: pa.Table,
    *,
    selections: Mapping[str, FacetSelection],
    conversion_plan: MemberConversionPlan | None,
    ceiling: int,
) -> None:
    """Raises `OverCeilingError` when the servable rows (`_servable_rows`) of a member whose
    rows skipped the check as they streamed (`_streaming_ceiling`) pass `ceiling`, before
    `apply_member_conversion` runs: a selection a method continues, or a role-shaped
    current-price conversion. Every other plan was checked as its rows arrived, and a
    constant-price plan is checked on its converted rows (`_counts_after_conversion`)."""
    if _streaming_ceiling(
        conversion_plan, selections, ceiling
    ) is not None or _counts_after_conversion(conversion_plan):
        return
    applied_window = compute_applied_window(request.window, columnar.served_year_bounds(table))
    check_ceiling(
        columnar.coverage(
            table, declared_roles=request.declared_roles, applied_window=applied_window
        ),
        ceiling=ceiling,
    )


def _dispatch_entities_and_roles(
    *,
    role_bindings: Mapping[str, EntityFilter],
    declared_roles: tuple[str, ...],
    bare_stat_vars: Sequence[str],
    coverage: CoverageIndex | None,
) -> tuple[Mapping[str, EntityFilter], tuple[str, ...] | None]:
    """Builds the `entities` and `roles` arguments for a shared-shape fetch.

    Role-bearing requests send `declared_roles`; subject-only requests send
    `None`. Explicit roles skip SDMX's header pre-scan, reducing a Kenya-scale
    fetch from about 70 requests to 35. A role mismatch raises in the stream.

    Empty `role_bindings` use `CoverageIndex.exhaustive_single_axis_filter`
    when role names are available. Shared stat vars have the same role names.
    That filter covers every year, as the facet probe's coverage candidates
    do, so a fetch that has dispatch facets has entities, and
    `datasets_api.admission.sync_estimate` prices the same list. Rows outside
    a member's window are dropped as batches arrive. Without coverage or role
    names, SDMX raises its required-entity-filter error.
    """
    if declared_roles == _SUBJECT_ROLE_ONLY:
        return role_bindings, None
    if role_bindings or coverage is None or not bare_stat_vars:
        return role_bindings, declared_roles
    names = coverage.role_names.get(bare_stat_vars[0])
    if names is None:
        return role_bindings, declared_roles
    exhaustive = coverage.exhaustive_single_axis_filter(
        variables=bare_stat_vars, donor_role=names.donor, recipient_role=names.recipient
    )
    return exhaustive, declared_roles


def _fetch_member_batches(
    request: MaterialisationRequest,
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    bare_stat_var: str,
    dispatch_ids: tuple[str, ...],
    selections: Mapping[str, FacetSelection],
    probe: MemberProbe,
    ceiling: int,
    coverage: CoverageIndex | None = None,
    deadline: float | None,
    progress: RowProgress | None = None,
    streaming_ceiling: int | None,
    year: int | None = None,
) -> tuple[pa.Schema, list[pa.RecordBatch]]:
    """Fetches one member under a raw-row cap of `20 * ceiling`. With `year`, the year a
    `latest` read's member elected, it fetches the member's rows at that year alone.

    Checks header roles before consuming batches. Coverage supplies roles and
    an exhaustive entity filter for unbound role requests. A subject-only
    request that binds no entity fetches `probe.places`, and rows pass
    `probe.admission`. Non-latest batches
    are filtered as they arrive; exceeding `streaming_ceiling` (`_streaming_ceiling`) raises
    `OverCeilingError` with up to one batch of overshoot. Latest reads buffer raw rows for a
    global period reduction, then filter. The raw cap also raises
    `OverCeilingError`. Progress counts only non-latest survivors; heartbeats
    renew the worker lease independently.
    """
    raw_guard = _RAW_FETCH_ROWS_GUARD_MULTIPLIER * ceiling
    entities, dispatch_roles = _dispatch_entities_and_roles(
        role_bindings=fetch_role_bindings(
            request.role_bindings, request.declared_roles, places=probe.places
        ),
        declared_roles=request.declared_roles,
        bare_stat_vars=(bare_stat_var,),
        coverage=coverage,
    )
    try:
        with stream_observation_batches(
            sdmx=sdmx,
            graph=graph,
            variables=[bare_stat_var],
            entities=entities,
            facet_ids=dispatch_ids,
            time_period=_time_period_for_window(request.window, year=year),
            roles=dispatch_roles,
            max_rows=raw_guard,
            deadline=deadline,
        ) as (roles, sdmx_batches):
            _check_role_divergence(roles, request)
            schema = observation_schema(roles)
            if request.window.kind == "latest":
                batches = list(sdmx_batches)
            else:
                batches = _collect_surviving_batches(
                    sdmx_batches,
                    schema=schema,
                    window=request.window,
                    selections=selections,
                    admission=probe.admission,
                    ceiling=streaming_ceiling,
                    stat_var=request.stat_var,
                    progress=progress,
                )
    except SdmxCapExceededError as exc:
        raise _raw_fetch_guard_error(request, exc, raw_guard=raw_guard, ceiling=ceiling) from exc
    return schema, batches


def _planning_graph(
    requests: Sequence[MaterialisationRequest], *, graph: GraphReader
) -> GraphReader:
    """`graph` itself unless some member binds a unit request. Then a
    `conversion_graph_reader` over every member's StatVar, so planning reads
    each StatVar's conversion arcs in one batch and each provenance and
    currency node once, however many members share them.
    """
    if not any(
        (binding := request.facet_bindings.get("unit")) is not None and binding.state == "request"
        for request in requests
    ):
        return graph
    return conversion_graph_reader(graph, stat_vars=[request.stat_var for request in requests])


def _probe_member(
    request: MaterialisationRequest,
    *,
    graph: GraphReader,
    availability: AvailabilityReader,
    coverage: CoverageIndex | None,
    precomputed_probe: CollectionFacetProbe | None = None,
    deadline: float | None,
) -> MemberProbe:
    """Returns one member's probe (`facets.probe_facets`).

    `precomputed_probe` is a one-request `CollectionFacetProbe` and skips the graph read.
    """
    if precomputed_probe is not None:
        return precomputed_probe.members[0]
    return probe_facets(
        stat_var=bare_stat_var_ref(request.stat_var),
        bound_entities=_flatten_bound_entities(request.role_bindings),
        graph=graph,
        availability=availability,
        settle=True,
        declared_roles=request.declared_roles,
        role_bindings=request.role_bindings,
        coverage=coverage,
        deadline=deadline,
    )


def dispatched_provenances(selections: Mapping[str, FacetSelection]) -> list[str]:
    """The sorted provenance refs `selections` dispatch."""
    return sorted(
        {
            selection.ref
            for selection in selections.values()
            if selection.axis == "provenance" and selection.ref is not None
        }
    )


def _dispatch_refresh(
    selections: Mapping[str, FacetSelection],
    *,
    precomputed_probe: CollectionFacetProbe | None,
    graph: GraphReader,
) -> dict[str, ProvenanceRefresh]:
    """The refresh dates of the provenances `selections` dispatch, called before the fetch so
    a run's source tokens are never newer than its rows. Dates `precomputed_probe` holds
    serve as read; the rest come from one `node_arcs_many`."""
    refs = dispatched_provenances(selections)
    known = precomputed_probe.provenance_refresh if precomputed_probe is not None else {}
    missing = [ref for ref in refs if ref not in known]
    read = _probe_provenance_refresh(missing, graph=graph)
    return {ref: known[ref] if ref in known else read[ref] for ref in refs}


def _plan_conversion(
    *,
    request: MaterialisationRequest,
    selections: Mapping[str, FacetSelection],
    graph: GraphReader,
    sdmx: SdmxClient,
    rate_cache: RateTableCache | None,
    deadline: float | None,
) -> MemberConversionPlan | None:
    """Plans unit conversion before fetch when the selection carries a request.

    Ref or unbound units return `None` without touching `rate_cache`. A
    Request-form units need the process-wide cache, which retains tables
    across calls; a missing cache raises `MaterialisationUsageError`.
    """
    unit = selections.get("unit")
    if unit is None or unit.request is None:
        return None
    if rate_cache is None:
        raise MaterialisationUsageError(
            "a request-form unit binding needs the process-wide rate_tables."
            "RateTableCache to load its conversion tables through; none was supplied"
        )
    return plan_member_conversion(
        request=request,
        selections=selections,
        graph=graph,
        sdmx=sdmx,
        rate_cache=rate_cache,
        deadline=deadline,
    )


def _generate_member(
    request: MaterialisationRequest,
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    ceiling: int,
    rate_cache: RateTableCache | None,
    coverage: CoverageIndex | None = None,
    precomputed_probe: CollectionFacetProbe | None = None,
    deadline: float | None = None,
    progress: RowProgress | None = None,
) -> _MemberOutcome:
    """Probes, selects facets, fetches, and builds one member inside a revision read.

    It checks response roles before consuming batches. Rows must match all
    selected refs together; SDMX facet IDs only narrow the OR-set of joint
    matches. With no joint candidate, it skips the fetch and returns empty.
    Non-latest rows filter batch by batch. Latest rows buffer for a global
    period reduction. `_finish_member_table` applies conversion, collapse,
    coverage, ceiling checks, and frame construction.

    A probe with no candidate returns the empty outcome before any selection
    or conversion plan (`_MemberOutcome.probe_empty`), since the request's places hold
    no observation of its variable.

    Conversion plans resolve before fetch. A supplied one-request
    `precomputed_probe` skips the graph probe; otherwise the function probes
    (`_probe_member`). Rows pass the probe's admission as well as the
    selection. `coverage` supplies
    probe counts and fetch filters. `deadline` is an absolute monotonic time
    passed to SDMX. `progress` tracks non-latest survivors.
    """
    bare_stat_var = bare_stat_var_ref(request.stat_var)
    probe, coverage_gaps = _national_currency_probe(
        request,
        _probe_member(
            request,
            graph=graph,
            availability=sdmx,
            coverage=coverage,
            precomputed_probe=precomputed_probe,
            deadline=deadline,
        ),
        graph=graph,
        deadline=deadline,
    )
    probe = _admitted(probe, request.candidate_filter)
    if not probe.candidates:
        _validate_empty_member_probe(request, coverage=coverage, graph=graph)
        empty = _finish_member_table(
            request,
            {},
            pa.Table.from_batches([], schema=observation_schema(request.declared_roles)),
            ceiling=ceiling,
            conversion_plan=None,
            currencies=None,
            default_scope=probe.scope,
        )
        return replace(
            empty,
            probe_empty=True,
            coverage_gaps=coverage_gaps,
            latest_years=_latest_years(
                request, elected=None, applied_window=None, read=ReadSeries()
            ),
            read_series={request.stat_var: ReadSeries()},
        )

    election = _select_member_facets(
        request,
        probe.candidates,
        graph=graph,
        breadth=probe.breadth_in(request.window),
        variable_breadth=probe.variable_breadth,
    )
    selections = election.selections
    conversion_plan = _plan_conversion(
        request=request,
        selections=selections,
        graph=graph,
        sdmx=sdmx,
        rate_cache=rate_cache,
        deadline=deadline,
    )
    provenance_refresh = _dispatch_refresh(
        selections, precomputed_probe=precomputed_probe, graph=graph
    )
    dispatch_ids = joint_dispatch_facet_ids(probe.candidates, election, member=0)
    elected: _ElectedRead | None = None
    if dispatch_ids:
        fetch = partial(
            _fetch_member_batches,
            request,
            sdmx=sdmx,
            graph=graph,
            bare_stat_var=bare_stat_var,
            dispatch_ids=dispatch_ids,
            selections=selections,
            probe=probe,
            ceiling=ceiling,
            coverage=coverage,
            deadline=deadline,
            progress=progress,
            streaming_ceiling=_streaming_ceiling(conversion_plan, selections, ceiling),
        )
        schema, batches = fetch()
        elected = _ElectedRead(
            elect=partial(elected_latest_year, request, coverage=coverage, facet_ids=dispatch_ids),
            read=lambda year: pa.Table.from_batches(fetch(year=year)[1], schema=schema),
            counted=partial(elected_entities, request, coverage=coverage, facet_ids=dispatch_ids),
        )
    else:
        # No probed candidate's (provenance, measurementMethod, unit) jointly
        # matches all three selected refs at once, so the post-fetch per-row
        # AND-filter would keep zero rows from any fetch. This degrades to
        # empty_result (handled below/in `_finish_member`) without spending a
        # network call. Passing `dispatch_ids` through unconditionally would
        # be wrong here: `stream_observation_batches` treats a falsy
        # `facet_ids` as "no facetId filter" (fetch every facet), the
        # opposite of what an empty joint dispatch set means.
        _log_no_joint_facet_match(request.stat_var, selections)
        schema = observation_schema(request.declared_roles)
        batches = []

    fetched = pa.Table.from_batches(batches, schema=schema)
    table, currencies, year, omitted = _servable_rows(
        fetched,
        window=request.window,
        selections=selections,
        admission=probe.admission,
        conversion_plan=conversion_plan,
        ceiling_check=partial(
            _check_servable_ceiling,
            request,
            selections=selections,
            conversion_plan=conversion_plan,
            ceiling=ceiling,
        ),
        elected=elected,
    )
    outcome = _finish_member_table(
        request,
        selections,
        columnar.elected_rows(table, year=year),
        ceiling=ceiling,
        conversion_plan=conversion_plan,
        currencies=currencies,
        default_scope=probe.scope,
    )
    read = _read_series(
        dispatch_ids, _continuing_facet_ids(probe.candidates, election, member=0), omitted
    )
    return replace(
        outcome,
        provenance_refresh=provenance_refresh,
        coverage_gaps=coverage_gaps,
        latest_years=_latest_years(
            request, elected=year, applied_window=outcome.applied_window, read=read
        ),
        read_series={request.stat_var: read},
    )


def _surviving_tables(
    batches: Iterator[pa.RecordBatch],
    *,
    schema: pa.Schema,
    window: RequestedWindow,
    selections: Mapping[str, FacetSelection],
    admission: RowAdmission,
) -> Iterator[pa.Table]:
    """Each batch filtered by window, facet, admission, and observed-value rules, as the read
    yields it."""
    for batch in batches:
        table = pa.Table.from_batches([batch], schema=schema)
        yield table.filter(
            columnar.survivor_mask(table, selections=selections, window=window, admission=admission)
        )


def _collect_surviving_batches(
    batches: Iterator[pa.RecordBatch],
    *,
    schema: pa.Schema,
    window: RequestedWindow,
    selections: Mapping[str, FacetSelection],
    admission: RowAdmission,
    ceiling: int | None,
    stat_var: str,
    progress: RowProgress | None = None,
) -> list[pa.RecordBatch]:
    """Keeps each batch's survivors (`_surviving_tables`).

    Raises `OverCeilingError` as soon as survivors exceed `ceiling`, with up
    to one batch of overshoot, and so stops pulling batches from the read. A `None`
    ceiling keeps every survivor (`_streaming_ceiling`).
    Progress, when supplied, counts each batch's surviving rows. Latest windows
    are reduced after buffering the response.
    """
    survivors: list[pa.RecordBatch] = []
    survivor_count = 0
    for survivor_table in _surviving_tables(
        batches, schema=schema, window=window, selections=selections, admission=admission
    ):
        survivor_count += survivor_table.num_rows
        survivors.extend(survivor_table.to_batches())
        if progress is not None:
            progress.add(survivor_table.num_rows)
        if ceiling is not None and survivor_count > ceiling:
            message, detail = _mid_stream_ceiling_message(stat_var, chunk_size=1, ceiling=ceiling)
            raise OverCeilingError(
                message, detail=detail, ceiling_kind="rows", count=survivor_count, cap=ceiling
            )
    return survivors


def _finish_member_table(
    request: MaterialisationRequest,
    selections: dict[str, FacetSelection],
    table: pa.Table,
    *,
    ceiling: int,
    conversion_plan: MemberConversionPlan | None,
    currencies: EntityCurrencies | None,
    default_scope: DefaultPlaceScope = NO_DEFAULT_SCOPE,
) -> _MemberOutcome:
    """Converts an already-filtered member table (`_servable_rows`) in one batched call that
    also records each series' conversion (`_convert_by_series`), summarises it, then builds
    its `Frame`.

    Duplicate identity facets collapse before conversion because collapse
    reads `unit` as a string. A missing plan skips arithmetic. Coverage and
    the ceiling check use the converted rows; the grammar receives `unit`
    unchanged, whether it is a native string or converted struct. The servable rows of a
    selection a method continues, and of a role-shaped current-price member, are checked
    against the ceiling before conversion (`_check_servable_ceiling`). A frame whose periods
    hold more than one grain raises `MixedGrainError` (`frame_build.period_grains`).
    """
    _check_servable_ceiling(
        request, table, selections=selections, conversion_plan=conversion_plan, ceiling=ceiling
    )
    table, conversion_record, series_conversions = _convert_by_series(
        conversion_plan, table, request=request, currencies=currencies
    )
    conversion_recipe_id = (
        conversion_plan.recipe.recipe_id
        if conversion_plan is not None and conversion_plan.recipe is not None
        else None
    )

    applied_window = compute_applied_window(request.window, columnar.served_year_bounds(table))
    coverage = columnar.coverage(
        table, declared_roles=request.declared_roles, applied_window=applied_window
    )
    check_ceiling(coverage, ceiling=ceiling)  # over_ceiling raises before Frame.build
    served_entities_by_role = columnar.served_entities_by_role(table, roles=request.declared_roles)

    frame = build_frame_from_table(
        table, role_keys=request.declared_roles, stat_var=request.stat_var
    )
    grains = period_grains(frame)
    if len(grains) > 1:
        raise MixedGrainError(grains, stat_vars=[request.stat_var])
    return _MemberOutcome(
        frame=frame,
        served_entities_by_role=served_entities_by_role,
        selections=selections,
        coverage=coverage,
        applied_window=applied_window,
        conversion_record=conversion_record,
        conversion_recipe_id=conversion_recipe_id,
        default_scope=default_scope,
        series_conversions=series_conversions,
    )


@dataclass(frozen=True)
class _MemberFetchPlan:
    """Holds one member's selections, dispatch IDs, admission, and conversion plan.

    Group plans carry the group's shared selections and each member's own admission.
    `continuing_ids` holds the dispatch IDs of the method continuing the selected series
    (`_continuing_facet_ids`).
    """

    selections: dict[str, FacetSelection]
    dispatch_ids: tuple[str, ...]
    admission: RowAdmission
    conversion_plan: MemberConversionPlan | None
    default_scope: DefaultPlaceScope = NO_DEFAULT_SCOPE
    continuing_ids: tuple[str, ...] = ()


def request_batch_key(request: MaterialisationRequest) -> tuple[object, ...]:
    """Keys a shared fetch by origin, declared roles, entity scope, and latest-window status.

    UN System Data Commons variables are read from the UN deployment and the
    rest from our SDMX service, so the two never share a fetch. A request
    carries one `TIME_PERIOD` filter, so latest and other windows cannot
    share. Admission estimates use the same key and `dispatch_lanes`. The roles
    and scope are `facets.request_scope_key`, which also groups the facet probe.
    """
    return (
        is_un_dcid(bare_stat_var_ref(request.stat_var)),
        *request_scope_key(request),
        request.window.kind == "latest",
    )


@dataclass(frozen=True)
class _ShapeChunk:
    """Groups members with matching roles, entity scope, and latest-window status.

    Each chunk has distinct bare StatVars, so `index_by_stat_var` routes rows
    unambiguously. Member indices retain request order and respect the lane
    target. The shared filters apply to every member in the chunk.
    """

    member_indices: tuple[int, ...]
    bare_stat_vars: tuple[str, ...]
    role_bindings: Mapping[str, EntityFilter]
    index_by_stat_var: dict[str, int]


def _plan_collection_chunks(
    requests: Sequence[tuple[str, MaterialisationRequest]],
    *,
    places: Sequence[Sequence[str]],
    lanes: int = DEFAULT_PLANNING_LANES,
) -> list[_ShapeChunk]:
    """Groups requests by `request_batch_key`, then splits each group into chunks.

    Repeated bare StatVars force a chunk boundary for unambiguous row routing.
    `lanes` defaults to 1; dispatch callers pass their concurrency. Chunks
    retain input order so singleton fetch failures follow request order.
    `places` holds, for each request in order, the places its probe resolved
    when it is subject-only and binds no entity (`MemberProbe.places`), and is
    empty for the rest. Such requests share a chunk only when their places
    match, and the chunk carries those places as its entity filter
    (`facets.fetch_role_bindings`).
    """
    groups: dict[tuple[object, ...], list[int]] = {}
    group_order: list[tuple[object, ...]] = []
    for index, (_spec_id, request) in enumerate(requests):
        key = (request_batch_key(request), tuple(places[index]))
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(index)

    chunks: list[_ShapeChunk] = []
    for key in group_order:
        indices = groups[key]
        first = requests[indices[0]][1]
        role_bindings = fetch_role_bindings(
            first.role_bindings, first.declared_roles, places=places[indices[0]]
        )
        bare_stat_vars = [bare_stat_var_ref(requests[i][1].stat_var) for i in indices]
        for position_group in stat_var_chunk_positions(bare_stat_vars, lanes=lanes):
            chunk_indices = [indices[position] for position in position_group]
            chunks.append(_build_shape_chunk(chunk_indices, requests, role_bindings=role_bindings))

    chunks.sort(key=lambda chunk: chunk.member_indices[0])
    return chunks


def stat_var_chunk_positions(bare_stat_vars: Sequence[str], *, lanes: int) -> list[tuple[int, ...]]:
    """Splits ordered positions at the lane target or a repeated bare StatVar.

    Each group has at most `ceil(len(bare_stat_vars) / lanes)` positions and
    one occurrence of each StatVar. Admission estimates use the same rule.
    """
    target = max(1, math.ceil(len(bare_stat_vars) / lanes))
    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    current_stat_vars: set[str] = set()
    for position, bare_stat_var in enumerate(bare_stat_vars):
        if len(current) >= target or bare_stat_var in current_stat_vars:
            groups.append(tuple(current))
            current = []
            current_stat_vars = set()
        current.append(position)
        current_stat_vars.add(bare_stat_var)
    if current:
        groups.append(tuple(current))
    return groups


def _build_shape_chunk(
    indices: Sequence[int],
    requests: Sequence[tuple[str, MaterialisationRequest]],
    *,
    role_bindings: Mapping[str, EntityFilter],
) -> _ShapeChunk:
    member_indices = tuple(indices)
    bare_stat_vars = tuple(bare_stat_var_ref(requests[i][1].stat_var) for i in member_indices)
    index_by_stat_var = dict(zip(bare_stat_vars, member_indices, strict=True))
    return _ShapeChunk(
        member_indices=member_indices,
        bare_stat_vars=bare_stat_vars,
        role_bindings=role_bindings,
        index_by_stat_var=index_by_stat_var,
    )


@dataclass(frozen=True)
class _MemberFetchResult:
    """One routed member table, using the chunk header's schema.

    Non-latest tables hold survivors. Latest tables hold raw rows for later
    reduction. Members with no rows still receive an empty, typed table. `read_year` reads
    a fetched `latest` member's rows at one year (`_ElectedRead.read`), and is None for
    every other member.
    """

    table: pa.Table
    read_year: Callable[[int], pa.Table] | None = None


def _generate_chunk(
    chunk: _ShapeChunk,
    requests: Sequence[tuple[str, MaterialisationRequest]],
    plans: Sequence[_MemberFetchPlan],
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    ceiling: int,
    raw_budget: _RawRowBudget,
    coverage: CoverageIndex | None = None,
    deadline: float | None = None,
) -> dict[int, _MemberFetchResult]:
    """Fetches a chunk's members in one SDMX request and routes rows by StatVar.

    Header roles are checked before batches are consumed. The shared
    `facetId` filter is the union of member dispatch IDs and only narrows
    the read; each member's row mask enforces its own selection and
    admission. An empty union skips the fetch. Every member keeps only rows
    under its own dispatch IDs (`columnar.dispatched_rows`), so a member with
    none, whether its probe found no facet or none at the group's cadence,
    serves nothing. Under place admission a row naming no facet or no place
    still reaches the member's admission mask, which refuses it as malformed;
    a role-bearing member with no dispatch IDs is skipped outright, since its
    mask keeps every row.

    Non-latest batches filter per member. Exceeding the member's `_streaming_ceiling`
    raises `OverCeilingError`, with up to one batch of overshoot; a selection a method
    continues counts its rows once the continuation filters them, and a conversion that omits
    rows after omission (`_streaming_ceiling`). Latest chunks are
    homogeneous. SDMX returns one row per series, then the caller's `_servable_rows`
    reduces to the member's global latest period and checks its coverage.

    `raw_budget` is shared by every group chunk. Each stream
    uses `raw_budget.total` as `max_rows`, while each arriving batch draws
    from the shared total. A batch can overshoot that total before the check
    raises `OverCeilingError`. An SDMX cap error raises the same error. A `latest`
    member read again at the year it elected (`_MemberFetchResult.read_year`) is one more
    request naming that member alone, and its rows draw on the same total.

    SDMX rejects a chunk whose StatVars disagree on observation properties. A
    StatVar with no graph node has none, so one absent member breaks the whole
    chunk. On that rejection `_absent_stat_vars` probes the chunk in `graph`:
    absent StatVars raise `MembersNotHeldError`. A fully held chunk, or a probe
    that fails, re-raises the SDMX error.

    `deadline` is an absolute monotonic time passed to SDMX. Its transport
    timeout or this deadline can expire first.
    """
    member_requests = {i: requests[i][1] for i in chunk.member_indices}
    chunk_facet_ids = tuple(
        sorted({fid for i in chunk.member_indices for fid in plans[i].dispatch_ids})
    )
    if not chunk_facet_ids:
        for i in chunk.member_indices:
            _log_no_joint_facet_match(member_requests[i].stat_var, plans[i].selections)
        return {
            i: _MemberFetchResult(
                table=pa.Table.from_batches(
                    [], schema=observation_schema(member_requests[i].declared_roles)
                )
            )
            for i in chunk.member_indices
        }

    latest_indices = {i for i in chunk.member_indices if member_requests[i].window.kind == "latest"}
    # `_plan_collection_chunks` never mixes a `"latest"` member with a
    # non-`"latest"` one in the same chunk, so `latest_indices` here is
    # always either every member in `chunk` or none of them.
    time_period = "LATEST" if latest_indices else None
    parts: dict[int, list[pa.RecordBatch]] = {i: [] for i in chunk.member_indices}
    survivor_counts: dict[int, int] = dict.fromkeys(chunk.member_indices, 0)
    streaming_ceilings = {
        i: _streaming_ceiling(plans[i].conversion_plan, plans[i].selections, ceiling)
        for i in chunk.member_indices
    }
    # Every member in `chunk` shares one `declared_roles` (`_ShapeChunk`'s
    # own docstring), so `_dispatch_entities_and_roles` needs only the
    # first member's own copy of it to speak for the whole chunk. See that
    # function's own docstring for `entities`/`roles`.
    chunk_declared_roles = member_requests[chunk.member_indices[0]].declared_roles
    entities, dispatch_roles = _dispatch_entities_and_roles(
        role_bindings=chunk.role_bindings,
        declared_roles=chunk_declared_roles,
        bare_stat_vars=chunk.bare_stat_vars,
        coverage=coverage,
    )

    def draw(batch: pa.RecordBatch) -> None:
        """Counts `batch` against the shared raw-row budget."""
        used = raw_budget.consume(batch.num_rows)
        if used > raw_budget.total:
            message, detail = _raw_guard_message(
                chunk,
                member_requests,
                observed=used,
                cap_kind="rows",
                raw_budget=raw_budget,
                ceiling=ceiling,
            )
            raise OverCeilingError(
                message, detail=detail, ceiling_kind="rows", count=used, cap=raw_budget.total
            )

    def over_cap(exc: SdmxCapExceededError) -> OverCeilingError:
        message, detail = _raw_guard_message(
            chunk,
            member_requests,
            observed=exc.observed,
            cap_kind=exc.cap_kind,
            raw_budget=raw_budget,
            ceiling=ceiling,
        )
        return OverCeilingError(
            message,
            detail=detail,
            ceiling_kind="bytes" if exc.cap_kind == "bytes" else "rows",
            count=exc.observed,
            cap=exc.cap,
        )

    stream = partial(
        stream_observation_batches,
        sdmx=sdmx,
        graph=graph,
        entities=entities,
        roles=dispatch_roles,
        max_rows=raw_budget.total,
        deadline=deadline,
    )

    def read_year(index: int, schema: pa.Schema, year: int) -> pa.Table:
        """Member `index`'s rows at `year`, under its own dispatch ids."""
        plan = plans[index]
        try:
            with stream(
                variables=[bare_stat_var_ref(member_requests[index].stat_var)],
                facet_ids=plan.dispatch_ids,
                time_period=_time_period_for_window(member_requests[index].window, year=year),
            ) as (_roles, sdmx_batches):
                batches: list[pa.RecordBatch] = []
                for batch in sdmx_batches:
                    draw(batch)
                    batches.append(
                        columnar.dispatched_rows(
                            batch, facet_ids=plan.dispatch_ids, admission=plan.admission
                        )
                    )
        except SdmxCapExceededError as exc:
            raise over_cap(exc) from exc
        return pa.Table.from_batches(batches, schema=schema)

    try:
        with stream(
            variables=list(chunk.bare_stat_vars),
            facet_ids=chunk_facet_ids,
            time_period=time_period,
        ) as (roles, sdmx_batches):
            for i in chunk.member_indices:
                _check_role_divergence(roles, member_requests[i])
            schema = observation_schema(roles)
            for batch in sdmx_batches:
                draw(batch)
                routed = columnar.split_by_variable(
                    batch, index_by_bare_stat_var=chunk.index_by_stat_var
                )
                for index, member_batch in routed.items():
                    plan = plans[index]
                    if not plan.dispatch_ids and isinstance(plan.admission, CoverageAdmission):
                        # A role-bearing member dispatching no facet serves nothing. Its
                        # admission mask keeps every row, so a row naming no facet would
                        # otherwise pass on the group's shared `selections` alone.
                        continue
                    # The fetch filters on every member's ids, so a member's variable can
                    # come back under an id only another member dispatched.
                    member_batch = columnar.dispatched_rows(
                        member_batch, facet_ids=plan.dispatch_ids, admission=plan.admission
                    )
                    if index in latest_indices:
                        # Raw, pre-reduction rows for a `"latest"` member:
                        # bounded by the shared `raw_budget` draw above
                        # only. The caller's `_servable_rows` reduces this buffer
                        # to its single latest period and checks the
                        # resulting coverage against `ceiling` afterward.
                        parts[index].append(member_batch)
                        continue
                    request = member_requests[index]
                    member_table = pa.Table.from_batches([member_batch], schema=schema)
                    mask = columnar.survivor_mask(
                        member_table,
                        selections=plans[index].selections,
                        window=request.window,
                        admission=plans[index].admission,
                    )
                    survivor_table = member_table.filter(mask)
                    survivor_counts[index] += survivor_table.num_rows
                    parts[index].extend(survivor_table.to_batches())
                    member_ceiling = streaming_ceilings[index]
                    if member_ceiling is not None and survivor_counts[index] > member_ceiling:
                        message, detail = _mid_stream_ceiling_message(
                            member_requests[index].stat_var,
                            chunk_size=len(chunk.member_indices),
                            ceiling=ceiling,
                        )
                        raise OverCeilingError(
                            message,
                            detail=detail,
                            ceiling_kind="rows",
                            count=survivor_counts[index],
                            cap=ceiling,
                        )
            result = {
                i: _MemberFetchResult(
                    table=pa.Table.from_batches(parts[i], schema=schema),
                    read_year=partial(read_year, i, schema) if i in latest_indices else None,
                )
                for i in chunk.member_indices
            }
    except SdmxCapExceededError as exc:
        raise over_cap(exc) from exc
    except SdmxIncompatibleVariablesError as exc:
        try:
            absent = _absent_stat_vars(chunk.bare_stat_vars, graph=graph, deadline=deadline)
        except GraphReadError:
            # The SDMX rejection is deterministic. A retryable probe failure must not replace it.
            logger.warning(
                "could not probe the StatVars of a chunk SDMX rejected as incompatible: %s",
                list(chunk.bare_stat_vars),
                exc_info=True,
            )
            absent = []
        if absent:
            raise MembersNotHeldError(stat_vars=absent) from exc
        raise

    return result


def _absent_stat_vars(
    bare_stat_vars: Sequence[str], *, graph: GraphReader, deadline: float | None
) -> list[str]:
    """Returns the StatVars with no graph node.

    Probes `typeOf`, which every held StatVar carries. A held StatVar may lack
    `observationProperties`. The budget is unbounded because an unprobed dcid
    reads as absent.
    """
    arcs = graph.node_arcs_many(
        bare_stat_vars, properties=("typeOf",), individual_probe_budget=None, deadline=deadline
    )
    return [stat_var for stat_var in bare_stat_vars if arcs[stat_var] is None]


def _mid_stream_ceiling_message(stat_var: str, *, chunk_size: int, ceiling: int) -> tuple[str, str]:
    """Formats a member's survivor-count refusal inside a shared chunk.

    Singleton chunks match the single-member message. Larger chunks name
    their other member count. Latest-window rows are checked after reduction.
    """
    message = (
        f"stat_var {stat_var!r} was aborted mid-stream after observing more "
        f"than {ceiling} rows, past the sync materialisation ceiling"
    )
    detail = (
        f"the SDMX upstream read for stat_var {stat_var!r} was aborted "
        f"mid-stream after observing more than {ceiling} rows. "
        f"The exact coverage was not measured"
    )
    if chunk_size > 1:
        clause = (
            f" (fetched together with {chunk_size - 1} other member(s) "
            "sharing this member's own entity scope, in one combined SDMX request)"
        )
        message += clause
        detail += clause
    return message, detail


def _raw_guard_message(
    chunk: _ShapeChunk,
    member_requests: Mapping[int, MaterialisationRequest],
    *,
    observed: int,
    # Availability caps also use `"values"`; this /data path receives only
    # `"rows"` or `"bytes"`. Keep the full union to pass `exc.cap_kind` through.
    cap_kind: Literal["rows", "bytes", "values"],
    raw_budget: _RawRowBudget,
    ceiling: int,
) -> tuple[str, str]:
    """Formats a shared raw-fetch refusal for group chunks.

    `observed` and `cap_kind` describe either the stream cap or the shared
    row-budget breach. Filtering has not run, so surviving coverage remains
    unknown. Multi-member errors name every member's StatVar because rows
    have not yet been attributed to members.
    """
    observed_unit = "bytes" if cap_kind == "bytes" else "rows"
    raw_guard = raw_budget.total
    if len(chunk.member_indices) == 1:
        stat_var = member_requests[chunk.member_indices[0]].stat_var
        return (
            f"the raw SDMX upstream response for stat_var {stat_var!r} was "
            f"too large to filter synchronously (observed {observed} "
            f"{observed_unit}, past this call's shared raw fetch guard "
            f"of {raw_guard} rows)",
            (
                f"the SDMX upstream read for stat_var {stat_var!r} was "
                f"aborted after {observed} {observed_unit}, past this "
                f"call's shared raw fetch guard of {raw_guard} rows. "
                f"This guard bounds memory and transport, across every member of "
                f"this call, on the unfiltered "
                f"upstream response only. Whether this request's actual "
                f"(post-window/facet-filter) coverage would have exceeded "
                f"the sync materialisation ceiling of {ceiling} rows was "
                f"not measured"
            ),
        )
    stat_vars = tuple(member_requests[i].stat_var for i in chunk.member_indices)
    return (
        f"the raw SDMX upstream response for {len(stat_vars)} stat_vars {stat_vars!r} "
        f"fetched together in one combined SDMX request was too large to filter "
        f"synchronously (observed {observed} {observed_unit}, past this "
        f"call's shared raw fetch guard of {raw_guard} rows)",
        (
            f"the SDMX upstream read for stat_vars {stat_vars!r}, fetched together "
            f"in one combined SDMX request sharing an identical entity scope, was "
            f"aborted after {observed} {observed_unit}, past this call's shared "
            f"raw fetch guard of {raw_guard} rows. This guard bounds "
            f"memory and transport, across every member of this call, "
            f"on the unfiltered upstream response only. Whether any one of "
            f"these stat_vars' actual (post-window/facet-filter) coverage would have "
            f"exceeded the sync materialisation ceiling of {ceiling} rows was not "
            f"measured"
        ),
    )


_FRAME_CHUNK_TARGET_ROWS = 100_000
"""Group rows are coalesced into chunks of about 100,000 before frame and Parquet writes.

Members stay whole and sort by dcid. This keeps each chunk's row-sort keys
above the previous chunk's keys, as required by streaming frame hashing."""

_DISPATCH_CHUNK_TARGET_MEMBERS = 8
"""Group fetches target eight members per combined request, regardless of concurrency.

With one shared entity scope, using only `member_concurrency` lanes would
place the whole group in its first dispatch wave and defeat row coalescing.
`lanes = max(member_concurrency, ceil(request_count / 8))` bounds each
chunk to about eight members and resident member tables to roughly
`member_concurrency * 8`. Kenya-scale fetches use about 35 requests instead
of 110."""


def dispatch_lanes(member_concurrency: int, request_count: int) -> int:
    """Returns the lane count used by group fetches.

    Admission estimates use the same formula so predicted and dispatched
    request counts agree.
    """
    return max(member_concurrency, math.ceil(request_count / _DISPATCH_CHUNK_TARGET_MEMBERS))


class _PooledCoverageTotals:
    """Accumulates pooled coverage and served-entity totals one row chunk at a time."""

    def __init__(self, *, shared_roles: tuple[str, ...]) -> None:
        self.rows = 0
        self.entity_refs: dict[str, frozenset[str | int]] = {
            label: frozenset() for label in (*shared_roles, "years")
        }
        self.served: dict[str, set[str]] = {}
        self.year_min: int | None = None
        self.year_max: int | None = None

    def fold(self, chunk_table: pa.Table, *, shared_roles: tuple[str, ...]) -> None:
        """Folds a concatenated chunk narrowed to `observation_columns(shared_roles)`.

        Called before the member tables are released.
        """
        self.rows += chunk_table.num_rows
        chunk_coverage = columnar.coverage(
            chunk_table, declared_roles=shared_roles, applied_window=None
        )
        for label, refs in chunk_coverage.entity_refs.items():
            self.entity_refs[label] = self.entity_refs[label] | refs
        for role, dcids in columnar.served_entities_by_role(
            chunk_table, roles=shared_roles
        ).items():
            self.served.setdefault(role, set()).update(dcids)
        bounds = columnar.served_year_bounds(chunk_table)
        if bounds is not None:
            chunk_min, chunk_max = bounds
            self.year_min = chunk_min if self.year_min is None else min(self.year_min, chunk_min)
            self.year_max = chunk_max if self.year_max is None else max(self.year_max, chunk_max)

    def year_bounds(self) -> tuple[int, int] | None:
        if self.year_min is None or self.year_max is None:
            return None
        return self.year_min, self.year_max


class _CanonicalParquetSink:
    """Writes each closed group chunk as a row group in an in-memory Parquet file.

    The first chunk sets the shared schema. `_generate_group` always yields a
    chunk, even when every member has zero rows. `finish()` returns a zero-copy
    view; durable storage happens only after a successful materialisation.
    """

    def __init__(self) -> None:
        self._sink = io.BytesIO()
        self._writer: pq.ParquetWriter | None = None

    def write(self, table: pa.Table) -> None:
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._sink, table.schema)
        self._writer.write_table(table)

    def finish(self) -> memoryview:
        assert self._writer is not None, (
            "_CanonicalParquetSink.finish() called before any chunk was written"
        )
        self._writer.close()
        # `getbuffer()` returns a zero-copy view and keeps the sink alive.
        # The view blocks later writes, and `finish()` is the final call.
        return self._sink.getbuffer()


def _probe_group_members(
    requests: Sequence[MaterialisationRequest],
    *,
    graph: GraphReader,
    availability: AvailabilityReader,
    coverage_index: CoverageIndex | None,
    precomputed_probe: CollectionFacetProbe | None,
    deadline: float | None,
) -> tuple[MemberProbe, ...]:
    """Returns every member's probe in request order, each carrying the places a member that
    binds no entity is fetched at.

    A precomputed probe needs one member probe per request in that order. Otherwise the group
    is probed in batches by entity scope.
    """
    if precomputed_probe is not None:
        probed_members = precomputed_probe.members
    else:
        probed_members = probe_collection_candidates(
            requests,
            graph=graph,
            availability=availability,
            settle=True,
            coverage=coverage_index,
            deadline=deadline,
        )
    return tuple(
        _admitted(probe, request.candidate_filter)
        for probe, request in zip(probed_members, requests, strict=True)
    )


@dataclass(frozen=True)
class _GroupPlan:
    """Shared group selections and per-member fetch plans in request order.
    Each member whose probe jointly matches the selections and whom the election did not
    count (`FacetElection.members`) serves nothing, and is named, in request order, in one of
    two tuples. `off_cadence_members` holds those with a jointly matching facet holding a point
    in the requested window (`facets.overlaps_window`, every facet under a window that is not
    a range): their observations in those years are at another observation period than the
    group's, so the frame leaves them out. `out_of_window_members` holds those with no such
    facet: they hold observations only outside the requested years."""

    selections: dict[str, FacetSelection]
    plans: list[_MemberFetchPlan]
    off_cadence_members: tuple[str, ...] = ()
    out_of_window_members: tuple[str, ...] = ()


def _plan_group_members(
    requests: Sequence[MaterialisationRequest],
    probes: Sequence[MemberProbe],
    *,
    graph: GraphReader,
    sdmx: SdmxClient,
    rate_cache: RateTableCache | None,
    deadline: float | None,
) -> _GroupPlan:
    """Resolves one shared facet selection and each member's conversion plan before fetch.

    Each conversion uses the member's StatVar and roles with the shared
    selections. A request-form unit with no rate cache raises
    `MaterialisationUsageError` in request order. Ref and unbound units need
    no conversion plan.

    Each member dispatches only its facets at the group's one cadence
    (`FacetElection.cadence`, the one serving the most members). A member the election did
    not count serves nothing and is named in `off_cadence_members` or
    `out_of_window_members` (`_GroupPlan`), for `finish._build_warnings` to report.
    """
    graph = _planning_graph(requests, graph=graph)
    election = select_group_facets(
        requests,
        [probe.candidates for probe in probes],
        graph=graph,
        breadth=group_breadth(probes, window=requests[0].window),
        variable_breadth=group_variable_breadth(probes),
    )
    selections = election.selections
    conversion_plans = [
        _plan_conversion(
            request=request,
            selections=selections,
            graph=graph,
            sdmx=sdmx,
            rate_cache=rate_cache,
            deadline=deadline,
        )
        for request in requests
    ]
    plans = [
        _MemberFetchPlan(
            selections=selections,
            dispatch_ids=joint_dispatch_facet_ids(probes[index].candidates, election, member=index),
            admission=probes[index].admission,
            conversion_plan=conversion_plans[index],
            continuing_ids=_continuing_facet_ids(probes[index].candidates, election, member=index),
        )
        for index in range(len(requests))
    ]
    window = requests[0].window
    # Each uncounted member's dcid, and whether a facet it matches holds a point in the window.
    uncounted = [
        (request.stat_var, any(overlaps_window(facet, window=window) for facet in matches))
        for index, (request, probe) in enumerate(zip(requests, probes, strict=True))
        if index not in election.members
        and (matches := joint_matches(probe.candidates, selections))
    ]
    return _GroupPlan(
        selections=selections,
        plans=plans,
        off_cadence_members=tuple(ref for ref, in_window in uncounted if in_window),
        out_of_window_members=tuple(ref for ref, in_window in uncounted if not in_window),
    )


def _empty_group_plan(probes: Sequence[MemberProbe]) -> _GroupPlan:
    """The plan of a group none of whose probes returned a candidate: no selection, and for
    each member no dispatch id and no conversion, so `_generate_chunk` fetches nothing and
    every member contributes its empty table."""
    return _GroupPlan(
        selections={},
        plans=[
            _MemberFetchPlan(
                selections={}, dispatch_ids=(), admission=probe.admission, conversion_plan=None
            )
            for probe in probes
        ],
    )


@dataclass(frozen=True)
class _DcidOrderedGroup:
    """Dcid-sorted requests and plans, with chunks planned over that order."""

    dcid_requests: list[tuple[str, MaterialisationRequest]]
    dcid_plans: list[_MemberFetchPlan]
    chunks: list[_ShapeChunk]


def _order_group_by_dcid(
    requests: Sequence[MaterialisationRequest],
    plans: Sequence[_MemberFetchPlan],
    *,
    member_concurrency: int,
    places: Sequence[Sequence[str]],
) -> _DcidOrderedGroup:
    """Orders fetches by dcid for row coalescing; planning errors retain request order.

    Chunks group matching roles, entity scope, and latest-window status, with
    bounded width and one request per chunk. Scope or window differences
    produce singleton chunks. Group roles must agree, while entity bindings
    may vary by member.
    """
    dcid_order = sorted(range(len(requests)), key=lambda i: bare_stat_var_ref(requests[i].stat_var))
    dcid_requests = [
        (str(original_index), requests[original_index]) for original_index in dcid_order
    ]
    dcid_plans = [plans[original_index] for original_index in dcid_order]
    lanes = dispatch_lanes(member_concurrency, len(requests))
    chunks = _plan_collection_chunks(
        dcid_requests,
        places=[places[original_index] for original_index in dcid_order],
        lanes=lanes,
    )
    return _DcidOrderedGroup(dcid_requests=dcid_requests, dcid_plans=dcid_plans, chunks=chunks)


@dataclass(frozen=True)
class _ConvertedMember:
    """One group member's converted rows and records, and the year its `latest` read
    elected less the `omitted` rows (`_omitted_rows`), from its chunk's pool worker."""

    table: pa.Table
    conversion_record: ConversionRecord | None
    recipe_id: str | None
    series_conversions: dict[SeriesKey, SeriesConversion]
    elected_year: int | None = None
    omitted: tuple[tuple[str, str], ...] = ()


class _GroupRowStreamer:
    """Fetches group members and yields dcid-sorted frames of about 100,000 rows.

    Call `stream_rows()` once to exhaustion. It fills per-member results,
    pooled totals, and both Parquet sinks. Every group yields at least one
    chunk, including groups whose members have zero rows.

    `member_concurrency` bounds in-flight fetches. The drain waits in dcid
    order, so that order selects the visible fetch error; up to
    `member_concurrency - 1` later chunks can issue requests before it surfaces.
    Successful rows and pooled coverage stay independent of concurrency.
    """

    def __init__(
        self,
        *,
        dcid_requests: Sequence[tuple[str, MaterialisationRequest]],
        dcid_plans: Sequence[_MemberFetchPlan],
        chunks: Sequence[_ShapeChunk],
        sdmx: SdmxClient,
        graph: GraphReader,
        ceiling: int,
        raw_budget: _RawRowBudget,
        coverage_index: CoverageIndex | None,
        deadline: float | None,
        shared_roles: tuple[str, ...],
        member_concurrency: int,
        progress: RowProgress | None,
        continuing: Sequence[ContinuedSeries],
    ) -> None:
        self._dcid_requests = dcid_requests
        self._dcid_plans = dcid_plans
        self._chunks = chunks
        self._sdmx = sdmx
        self._graph = graph
        self._ceiling = ceiling
        self._raw_budget = raw_budget
        self._coverage_index = coverage_index
        self._deadline = deadline
        self._shared_roles = shared_roles
        self._shared_columns = observation_columns(shared_roles)
        self._member_concurrency = member_concurrency
        self._progress = progress
        self.observation_counts: dict[str, int] = {}
        self.conversion_records: dict[str, ConversionRecord | None] = {}
        self.conversion_recipe_ids: dict[str, str | None] = {}
        self.series_conversions: dict[SeriesKey, SeriesConversion] = {}
        self.latest_windows: list[AppliedWindow | None] = []
        self.latest_years: dict[str, LatestYear] = {}
        self.read_series: dict[str, ReadSeries] = {}
        self.frame_columns: list[tuple[str, ...]] = []
        self.grains: dict[str, str] = {}
        self.method_periods: dict[str, tuple[str, str]] = {}
        self.totals = _PooledCoverageTotals(shared_roles=shared_roles)
        self.parquet_sink = _CanonicalParquetSink()
        self.statistics_sink = SeriesStatisticsSink(continuing=continuing)

    def _generate_one_chunk(self, chunk: _ShapeChunk) -> dict[int, _ConvertedMember]:
        """Runs entirely inside one pool worker: `_generate_chunk` is this
        chunk's own shared fetch (one `stream_observation_batches` call
        naming every member in `chunk` at once), then each member's own
        servable rows (`_servable_rows`) and per-series unit
        conversion (`_convert_by_series`), without a per-member
        `check_ceiling` call (`stream_rows` runs one pooled ceiling check
        after every row chunk has closed). Keys its own returned dict, and
        reads `chunk.member_indices` in `dcid_requests`/`dcid_plans`'s own
        dcid-sorted position space, not `requests`' own.
        """
        fetch_results = _generate_chunk(
            chunk,
            self._dcid_requests,
            self._dcid_plans,
            sdmx=self._sdmx,
            graph=self._graph,
            ceiling=self._ceiling,
            raw_budget=self._raw_budget,
            coverage=self._coverage_index,
            deadline=self._deadline,
        )
        outcomes: dict[int, _ConvertedMember] = {}
        for index, fetch_result in fetch_results.items():
            request = self._dcid_requests[index][1]
            plan = self._dcid_plans[index]
            table, currencies, year, omitted = _servable_rows(
                fetch_result.table,
                window=request.window,
                selections=plan.selections,
                admission=plan.admission,
                conversion_plan=plan.conversion_plan,
                ceiling_check=partial(
                    _check_servable_ceiling,
                    request,
                    selections=plan.selections,
                    conversion_plan=plan.conversion_plan,
                    ceiling=self._ceiling,
                ),
                elected=(
                    _ElectedRead(
                        elect=partial(
                            elected_latest_year,
                            request,
                            coverage=self._coverage_index,
                            facet_ids=plan.dispatch_ids,
                        ),
                        read=fetch_result.read_year,
                        counted=partial(
                            elected_entities,
                            request,
                            coverage=self._coverage_index,
                            facet_ids=plan.dispatch_ids,
                        ),
                    )
                    if fetch_result.read_year is not None
                    else None
                ),
            )
            table = columnar.elected_rows(table, year=year)
            _check_servable_ceiling(
                request,
                table,
                selections=plan.selections,
                conversion_plan=plan.conversion_plan,
                ceiling=self._ceiling,
            )
            table, conversion_record, series_conversions = _convert_by_series(
                plan.conversion_plan, table, request=request, currencies=currencies
            )
            recipe_id = (
                plan.conversion_plan.recipe.recipe_id
                if plan.conversion_plan is not None and plan.conversion_plan.recipe is not None
                else None
            )
            outcomes[index] = _ConvertedMember(
                table=table,
                conversion_record=conversion_record,
                recipe_id=recipe_id,
                series_conversions=series_conversions,
                elected_year=year,
                omitted=omitted,
            )
        return outcomes

    def _close_row_chunk(self, members: list[tuple[MaterialisationRequest, pa.Table]]) -> Frame:
        """One row chunk's own tail, run once as it closes: builds its
        `Frame` (`Frame.build`'s own grain-sort and duplicate-grain refusal
        run here, per chunk; every row still carries exactly one
        provenance, since `_generate_group`'s callers dispatch every member
        from the one selection the group shares, so
        `datasets_api.groupby.refuse_provenance_disagreement` never sees
        two provenances in one chunk this builds; a group both over
        ceiling and carrying a duplicate grain raises this refusal first,
        since a row chunk always closes before `_assemble_group_outcome`'s
        pooled ceiling check runs), refuses with `MixedGrainError` once the
        chunks closed so far hold periods of two grains (`self.grains`, each
        grain's earliest period), widens each method's period bounds
        (`self.method_periods`), writes that `Frame`'s
        own canonical Arrow table as one Parquet row group
        (`self.parquet_sink`), folds it into the group's pooled coverage
        (`self.totals`), and writes its own statistics table as one row
        group of the group's statistics Parquet build
        (`self.statistics_sink`, fed
        `statistics.compute_series_statistics_table(frame)`, correct per
        chunk since a series' key includes `stat_var` and this class's own
        caller never splits one member across two chunks, so no series'
        rows ever span a chunk boundary). Returns the `Frame` for the
        caller to yield onward into the streaming content-version hash.
        `members`' own tables are never referenced again once this
        returns. Increments `self._progress`, when given, by this chunk's
        own row count, once the chunk's `Frame` is genuinely built.
        """
        frame = build_frame_from_member_tables(
            [(request.stat_var, table) for request, table in members], role_keys=self._shared_roles
        )
        if not self.frame_columns:
            self.frame_columns.append(frame.columns)
        for grain, period in period_grains(frame).items():
            self.grains[grain] = min(period, self.grains.get(grain, period))
        if len(self.grains) > 1:
            raise MixedGrainError(
                self.grains, stat_vars=[request.stat_var for _, request in self._dcid_requests]
            )
        fold_method_periods(self.method_periods, frame)
        self.parquet_sink.write(frame.table)
        chunk_table = pa.concat_tables([table.select(self._shared_columns) for _, table in members])
        self.totals.fold(chunk_table, shared_roles=self._shared_roles)
        self.statistics_sink.write(compute_series_statistics_table(frame))
        if self._progress is not None:
            self._progress.add(frame.table.num_rows)
        return frame

    def stream_rows(self) -> Iterator[Frame]:
        """Dispatches `self._chunks` (already `dcid:`-sorted) on a bounded
        sliding-window pool: at most `member_concurrency` chunks' own
        fetches in flight at once, primed with the first `max_workers`
        submissions and refilled one at a time as each ahead of it is
        consumed. The drain loop pops the front pending future and blocks
        on `.result()` in chunk submission order, so the first
        still-pending future is what raises, regardless of which chunk's
        fetch actually finishes first. An out-of-order completion is
        simply not observed until the drain reaches it, which is what
        bounds how far ahead dispatch can get past the coalescing cursor
        below.

        Each drained chunk's own member outcomes are recorded into
        `pending_members`, keyed by dcid position, and `cursor` then walks
        forward through however much of that keyspace has become
        contiguous: while `cursor` names a position `pending_members`
        already carries, that member is folded into the row chunk
        currently being assembled (closing and yielding it first, via
        `_close_row_chunk`, when the next member would put it over
        `_FRAME_CHUNK_TARGET_ROWS`), and its own table is dropped from
        `pending_members` in the same step. Because `chunks` is
        dcid-sorted and drained in that same order, `cursor` almost always
        advances to the end of `pending_members` in one pass per drained
        chunk; it can fall behind only when the group spans more than one
        entity-scope/`"latest"`-ness group, which interleaves dcid
        positions across chunks. `pending_members` is exactly the
        bounded buffer that gap waits in, never larger than the members
        `member_concurrency` chunks ahead of `cursor` can carry. Peak
        memory during this whole pass is bounded by that in-flight window
        plus the one row chunk currently being assembled, never by the
        group's own total row count.

        `finally: pool.shutdown(wait=True, cancel_futures=True)` drops any
        chunk never submitted and blocks for every already-submitted
        chunk's own fetch to finish, past whichever future raised, before
        this generator re-raises.
        """
        chunks = self._chunks
        pending_members: dict[int, tuple[MaterialisationRequest, pa.Table]] = {}
        accumulator: list[tuple[MaterialisationRequest, pa.Table]] = []
        accumulator_rows = 0
        cursor = 0

        def _submit(
            pool: ThreadPoolExecutor, chunk_index: int
        ) -> Future[dict[int, _ConvertedMember]]:
            return submit_with_context(pool, self._generate_one_chunk, chunks[chunk_index])

        max_workers = max(1, min(self._member_concurrency, len(chunks)))
        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            pending = [_submit(pool, i) for i in range(min(max_workers, len(chunks)))]
            next_chunk = len(pending)

            for _shape_chunk in chunks:
                result = pending.pop(0).result()
                if next_chunk < len(chunks):
                    pending.append(_submit(pool, next_chunk))
                    next_chunk += 1
                for index, member in result.items():
                    request = self._dcid_requests[index][1]
                    table = member.table
                    self.observation_counts[request.stat_var] = table.num_rows
                    self.conversion_records[request.stat_var] = member.conversion_record
                    self.conversion_recipe_ids[request.stat_var] = member.recipe_id
                    self.series_conversions.update(member.series_conversions)
                    plan = self._dcid_plans[index]
                    read = _read_series(plan.dispatch_ids, plan.continuing_ids, member.omitted)
                    self.read_series[request.stat_var] = read
                    if request.window.kind == "latest":
                        applied_window = compute_applied_window(
                            request.window, columnar.served_year_bounds(table)
                        )
                        self.latest_windows.append(applied_window)
                        self.latest_years.update(
                            _latest_years(
                                request,
                                elected=member.elected_year,
                                applied_window=applied_window,
                                read=read,
                            )
                        )
                    pending_members[index] = (request, table)
                while cursor < len(self._dcid_requests) and cursor in pending_members:
                    request, table = pending_members.pop(cursor)
                    if accumulator and accumulator_rows + table.num_rows > _FRAME_CHUNK_TARGET_ROWS:
                        yield self._close_row_chunk(accumulator)
                        accumulator = []
                        accumulator_rows = 0
                    accumulator.append((request, table))
                    accumulator_rows += table.num_rows
                    cursor += 1
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if accumulator:
            yield self._close_row_chunk(accumulator)


def _assemble_group_outcome(
    streamer: _GroupRowStreamer,
    requests: Sequence[MaterialisationRequest],
    *,
    selections: dict[str, FacetSelection],
    ceiling: int,
    default_scope: DefaultPlaceScope = NO_DEFAULT_SCOPE,
    off_cadence_members: tuple[str, ...] = (),
    out_of_window_members: tuple[str, ...] = (),
    probe_empty: bool = False,
) -> _GroupOutcome:
    """Drives `streamer`'s streaming fetch to completion, then assembles
    the group's own `_GroupOutcome` from what it accumulated.

    `stream_frame_sha256` pulls one `Frame` at a time from
    `streamer.stream_rows()` (`dfo_frame_grammar
    .stream_canonical_frame_bytes`'s own docstring: never more than one
    chunk's rows, or its serialised bytes, in memory at once), which is
    what actually drives the fetch/coalesce pass: every chunk's own
    Parquet write, coverage fold, and series-statistics append
    (`streamer._close_row_chunk`) runs as a side effect of this one pull,
    before that chunk's `Frame` is handed to the hasher. This computes
    only the digest; `materialise.finish._finish_group` mints the token
    from it afterward (`content_version.mint_version_from_digest`),
    without a second read of these rows. A `materialise_group` call mints
    exactly one `content_version` this way.

    The ceiling check runs once, here, after every row chunk has already
    closed and every member's own conversion has already run, so a
    member's own `rate_coverage_gap` (or any other conversion refusal)
    always preempts this pooled `OverCeilingError`. Since every row chunk's
    own bytes are already streamed into the canonical Parquet build by the
    time this check runs, a breach here discards that in-memory build
    rather than ever exposing it: no artifact is stored anywhere until a
    caller downstream of a successful `materialise_group` return writes
    one. `_generate_chunk`'s own mid-stream abort, inside the fetch itself,
    fires when one member's own survivors, within its own chunk, exceed
    `ceiling`; this aggregate check instead catches several members each
    individually under `ceiling` but summing past it.
    """
    frame_sha256 = stream_frame_sha256(streamer.stream_rows())

    year_bounds = streamer.totals.year_bounds()
    if requests[0].window.kind == "latest":
        applied_window = _merge_latest_windows(streamer.latest_windows)
    else:
        applied_window = compute_applied_window(requests[0].window, year_bounds)
    # Every member shares one role set (`_check_group_role_agreement`) and one
    # requested window (the same identity's own scope), so the first
    # member's copy of each stands in for the whole group here.
    shared_roles = requests[0].declared_roles
    coverage = Coverage(
        rows=streamer.totals.rows,
        dimensions=tuple(
            {"label": label, "count": len(streamer.totals.entity_refs[label])}
            for label in (*shared_roles, "years")
        ),
        window=applied_window.as_trace_entry() if applied_window is not None else None,
        entity_refs=streamer.totals.entity_refs,
    )
    check_ceiling(coverage, ceiling=ceiling)
    parquet_bytes = streamer.parquet_sink.finish()

    return _GroupOutcome(
        frame_chunks=ParquetFrameChunks(parquet_bytes, columns=streamer.frame_columns[0]),
        frame_sha256=frame_sha256,
        series_statistics=streamer.statistics_sink.finish(),
        row_count=coverage.rows,
        served_entities_by_role=streamer.totals.served,
        selections=selections,
        coverage=coverage,
        applied_window=applied_window,
        member_observation_counts=streamer.observation_counts,
        member_conversion_records=streamer.conversion_records,
        member_conversion_recipe_ids=streamer.conversion_recipe_ids,
        default_scope=default_scope,
        series_conversions=streamer.series_conversions,
        off_cadence_members=off_cadence_members,
        out_of_window_members=out_of_window_members,
        probe_empty=probe_empty,
        method_periods=streamer.method_periods,
        latest_years=streamer.latest_years,
        read_series=streamer.read_series,
    )


def _generate_group(
    requests: Sequence[MaterialisationRequest],
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    ceiling: int,
    member_concurrency: int,
    rate_cache: RateTableCache | None = None,
    # Named `coverage_index`, not `coverage`, only in this one function:
    # `_assemble_group_outcome` already binds the local name `coverage` to
    # a `materialise.window.Coverage` (the row/dimension/window report,
    # `_GroupOutcome.coverage`'s own field), a different type sharing the
    # more natural name. Every other function in this module keeps the
    # shorter `coverage: CoverageIndex | None = None` parameter name; this
    # is the only one with a same-scope collision to avoid.
    coverage_index: CoverageIndex | None = None,
    precomputed_probe: CollectionFacetProbe | None = None,
    deadline: float | None = None,
    progress: RowProgress | None = None,
) -> _GroupOutcome:
    """Runs once for `len(requests) >= 2` and returns one `_GroupOutcome` for the group, with one
    facet selection shared by every member, pooled coverage folded from every member's own surviving
    rows, and exactly one `content_version`-worthy digest bounded by exactly one ceiling check over
    the pooled total.

    Runs in four phases, each of which raises before any later phase's own
    work is attempted: `_probe_group_members` probes every member's own
    facet candidates; `_plan_group_members` resolves the group's one shared facet
    selection and each member's own conversion plan, raising
    `DefaultFacetUndefinedError`/`FacetNotAvailableError`/
    `MaterialisationUsageError`/`NonCurrencyUnitError` as appropriate, and is
    skipped for `_empty_group_plan` when no member's probe returned a candidate
    (`_GroupOutcome.probe_empty`);
    `_order_group_by_dcid` reorders the members by bare `stat_var` and
    plans the shared SDMX chunks; and `_assemble_group_outcome` drives
    `_GroupRowStreamer`'s chunked fetch to completion and raises
    `OverCeilingError` on either a raw-fetch guard breach, a mid-stream
    per-chunk breach, or the final pooled-coverage breach. See each
    function's own docstring for its phase's own rationale.

    `precomputed_probe`/`coverage_index` are forwarded to
    `_probe_group_members`; `progress` is forwarded to
    `_GroupRowStreamer`, incremented once per closed row chunk by that
    chunk's own row count. `deadline`, when supplied, is one absolute
    `time.monotonic()` timestamp forwarded to every phase's own SDMX/graph
    calls, each of which honours whichever of its own transport timeout
    and this deadline expires first.
    """
    probed = _probe_group_members(
        requests,
        graph=graph,
        availability=sdmx,
        coverage_index=coverage_index,
        precomputed_probe=precomputed_probe,
        deadline=deadline,
    )
    probe_empty = not any(member.candidates for member in probed)
    group_plan = (
        _empty_group_plan(probed)
        if probe_empty
        else _plan_group_members(
            requests,
            probed,
            graph=graph,
            sdmx=sdmx,
            rate_cache=rate_cache,
            deadline=deadline,
        )
    )
    provenance_refresh = _dispatch_refresh(
        group_plan.selections, precomputed_probe=precomputed_probe, graph=graph
    )
    ordered = _order_group_by_dcid(
        requests,
        group_plan.plans,
        member_concurrency=member_concurrency,
        places=[member.places for member in probed],
    )
    streamer = _GroupRowStreamer(
        dcid_requests=ordered.dcid_requests,
        dcid_plans=ordered.dcid_plans,
        chunks=ordered.chunks,
        sdmx=sdmx,
        graph=graph,
        ceiling=ceiling,
        raw_budget=_RawRowBudget(ceiling),
        coverage_index=coverage_index,
        deadline=deadline,
        shared_roles=requests[0].declared_roles,
        member_concurrency=member_concurrency,
        progress=progress,
        continuing=selected_continued_series(group_plan.selections),
    )
    outcome = _assemble_group_outcome(
        streamer,
        requests,
        selections=group_plan.selections,
        ceiling=ceiling,
        default_scope=default_scope_of(probed),
        off_cadence_members=group_plan.off_cadence_members,
        out_of_window_members=group_plan.out_of_window_members,
        probe_empty=probe_empty,
    )
    return replace(outcome, provenance_refresh=provenance_refresh)


def _merge_latest_windows(windows: Sequence[AppliedWindow | None]) -> AppliedWindow | None:
    """Group-level applied window for `window.kind == "latest"`, spanning the
    earliest through the latest period actually served across every member.

    Each entry of `windows` is one member's own applied window, computed by
    `_generate_group`'s own `_GroupRowStreamer.stream_rows` from that member's own
    table (already narrowed to its own latest period -- `_generate_group`'s
    per-member carve-out, pinned by
    `test_group_latest_window_over_a_huge_raw_response_does_not_early_refuse_and_succeeds`),
    the moment that member's own fetch resolves, `None` when
    `compute_applied_window` found nothing to report for it. Members can
    carry different latest years. Pooling every member's surviving rows
    into one `compute_applied_window` call would report a single latest
    year (that function's "latest" branch always does) while the frame
    it's paired with holds rows from more than one year, so each member's
    own window is computed and merged instead: this function only merges
    the already-computed per-member windows into one spanning their
    earliest start through their latest end.
    """
    start_years: list[int] = []
    end_years: list[int] = []
    for member_window in windows:
        if member_window is None:
            continue
        assert member_window.start_year is not None
        assert member_window.end_year is not None
        start_years.append(member_window.start_year)
        end_years.append(member_window.end_year)
    if not start_years:
        return None
    return AppliedWindow(
        kind="latest", start_year=min(start_years), end_year=max(end_years), origin="query"
    )


def _prepare_preview_member(
    request: MaterialisationRequest,
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    limit: int,
    ceiling: int,
    rate_cache: RateTableCache | None = None,
    # Named `coverage_index`, not `coverage`, for the same reason
    # `_generate_group`'s own parameter is: this function's own tail binds
    # the local name `coverage` to a `materialise.window.Coverage`
    # (`_PreviewOutcome.coverage`'s own field), a different type sharing
    # the more natural name.
    coverage_index: CoverageIndex | None = None,
    deadline: float | None = None,
    mint_full_frame: bool = False,
    precomputed: Sequence[Facet] | None = None,
    series_roles: Sequence[str] = (),
    series_limit: int | None = None,
    allowed_series_keys: frozenset[tuple[str, ...]] | None = None,
    elect: _Election | None = None,
) -> _PreparedPreview:
    """Runs the exact facet-selection/dispatch machinery `_generate_member`
    runs for the same `MaterialisationRequest`: `_probe_member`,
    `_select_member_facets`, `joint_dispatch_facet_ids`,
    `stream_observation_batches` under the same raw-fetch guard, 20 times
    `ceiling`. A preview's facet selection therefore cannot diverge
    from what a real `materialise_spec` call would choose for the same
    identity. A raw read past the guard raises `SdmxCapExceededError`.

    Once `selections` is resolved, a request-form unit binding is
    planned right away (`_plan_conversion`, before any fetch), the same
    splice `_generate_member` runs. `rate_cache=None` with a request-form
    binding raises `MaterialisationUsageError` before any fetch, mirroring
    `_generate_member`'s own rule. A ref or unbound unit binding plans
    nothing and passes through unconverted. The plan is applied
    (`member_conversion.apply_member_conversion`, which folds in
    `facet_collapse.collapse_duplicate_identity_facets_table` first) to the
    full survivor table before truncation, so `total_rows`/`coverage` and
    the delivered `frame` alike see converted values, and a preview shows
    only rows that clear all refusal checks. The member's own
    `convert.ConversionRecord`, when conversion actually ran arithmetic, is
    carried on `_PreviewOutcome.conversion_record`.

    `limit` only ever bounds what's delivered (`frame`), never what's read
    or counted (`total_rows`). Every survivor is kept, and `ceiling` sizes
    only the raw-fetch guard. A `"latest"` window fetches with
    `time_period="LATEST"` (`_time_period_for_window`, as `_generate_member`
    does), which narrows the read server-side to one row per matched series.
    `_servable_rows` then buffers every batch under the raw-fetch guard and
    filters afterward, since the client-side reduction that picks the single
    latest period across the member is a global max over the whole member,
    not decidable per batch. It gets the same true-count-then-truncate
    treatment.

    A `"latest"` read elects its year as a mint of the request does
    (`elected_latest_year`) and reads it the same way (`_servable_rows`), one more request
    under the same guard when a series ends after that year. `elect`, when given, is the
    election in its place, for a caller reading one StatVar over several previews that
    elected over all of them. The prepared rows hold the elected year's period beside the newest one
    (`_PreparedPreview.year`), and `_finish_preview_member` serves the first when a row
    holds it. A read with `series_roles` and no `elect` elects no year.

    `mint_full_frame`, when true, also builds `_PreviewOutcome.full_frame`
    over every survivor row. That build is the same one a mint's own
    `_finish_member_table` runs, so it is what a caller needs to mint a
    preview's `content_version`. A caller reporting no version (`False`,
    the default) skips it, since the frame is never read.

    Truncation, when `total_rows > limit`, keeps the `limit` most recent
    rows (`_most_recent_rows`).

    `coverage`, when given, is forwarded to `_probe_member` (real facet counts for a
    role-bearing request) -- the same instance a mint reads, so a preview's facet selection can
    never diverge from a real mint's on this account either. A role-bearing request reaching
    `probe_facets` with `coverage=None` raises `MaterialisationUsageError` before any fetch. A
    request that binds no entity previews the scope a mint fetches: the places
    `place_scope.read_default_presence` returns for a subject-only StatVar, and coverage's
    exhaustive filter for a role-bearing one. `precomputed` candidates skip the probe, admitting
    rows as the probe would (`facets.probe_from_candidates`). The preview sends no explicit
    roles and keeps SDMX's header pre-scan.

    `series_roles` names roles the request leaves open whose every dcid is a
    series of its own. Each series' rows then pass `_servable_rows` on their own, so a
    `"latest"` window keeps each series' latest period, which a caller reading the
    StatVar's series over several previews reduces to the StatVar's one latest period
    (`window.latest_period`) once they all return, and conversion runs per series, so
    `_PreviewOutcome.series_conversion_records` holds each series' record and
    `conversion_record` their sum (`convert.merged_record`). A gap in any
    series refuses the whole read, as it does without `series_roles`. Each
    series keeps its own most recent `series_limit` rows, so a long series
    cannot crowd out a short or older one, and
    `_PreviewOutcome.series_total_rows` counts each series' survivors before
    that cut. `ceiling` still sizes the raw-fetch guard.
    """
    bare_stat_var = bare_stat_var_ref(request.stat_var)
    probe, _ = _national_currency_probe(
        request,
        (
            probe_from_candidates(
                precomputed,
                declared_roles=request.declared_roles,
                bound_entities=_flatten_bound_entities(request.role_bindings),
            )
            if precomputed is not None
            else _probe_member(
                request,
                graph=graph,
                availability=sdmx,
                coverage=coverage_index,
                deadline=deadline,
            )
        ),
        graph=graph,
        deadline=deadline,
    )
    probe = _admitted(probe, request.candidate_filter)
    election = _select_member_facets(
        request,
        probe.candidates,
        graph=graph,
        breadth=probe.breadth_in(request.window),
        variable_breadth=probe.variable_breadth,
    )
    selections = election.selections
    conversion_plan = _plan_conversion(
        request=request,
        selections=selections,
        graph=graph,
        sdmx=sdmx,
        rate_cache=rate_cache,
        deadline=deadline,
    )

    dispatch_ids = joint_dispatch_facet_ids(probe.candidates, election, member=0)
    read_year: Callable[[int], pa.Table] | None = None
    counted: Callable[[str], Sequence[str]] | None = None

    if dispatch_ids:
        raw_guard = _RAW_FETCH_ROWS_GUARD_MULTIPLIER * ceiling
        # The entity filter a mint's own fetch sends (`_fetch_member_batches`).
        entities, _ = _dispatch_entities_and_roles(
            role_bindings=fetch_role_bindings(
                request.role_bindings, request.declared_roles, places=probe.places
            ),
            declared_roles=request.declared_roles,
            bare_stat_vars=(bare_stat_var,),
            coverage=coverage_index,
        )
        stream = partial(
            stream_observation_batches,
            sdmx=sdmx,
            graph=graph,
            variables=[bare_stat_var],
            entities=entities,
            facet_ids=dispatch_ids,
            max_rows=raw_guard,
            deadline=deadline,
        )
        with stream(time_period=_time_period_for_window(request.window)) as (roles, sdmx_batches):
            _check_role_divergence(roles, request)
            schema = observation_schema(roles)
            if request.window.kind == "latest":
                # The fetch above already carries `time_period="LATEST"`,
                # so `sdmx_batches` holds one row per matched series. The
                # client-side reduction is still a global max over the
                # whole member. Buffer every batch of this (already
                # narrowed) response (still bounded by `raw_guard` above)
                # and apply the window/facet/value filters below, as a
                # separate pass over the assembled table. Other window
                # kinds fold that same filtering into the single-pass loop
                # that filters each batch as it streams.
                batches = list(sdmx_batches)
            else:
                batches = [
                    batch
                    for table in _surviving_tables(
                        sdmx_batches,
                        schema=schema,
                        window=request.window,
                        selections=selections,
                        admission=probe.admission,
                    )
                    for batch in table.to_batches()
                ]

        def read_year(year: int) -> pa.Table:
            with stream(time_period=_time_period_for_window(request.window, year=year)) as (
                _roles,
                year_batches,
            ):
                return _admitted_series_rows(
                    pa.Table.from_batches(list(year_batches), schema=schema),
                    series_roles,
                    allowed_series_keys,
                )

        if elect is None and not series_roles:
            elect = partial(
                elected_latest_year, request, coverage=coverage_index, facet_ids=dispatch_ids
            )
            counted = partial(
                elected_entities, request, coverage=coverage_index, facet_ids=dispatch_ids
            )
    else:
        # Mirrors `_generate_member`'s own no-dispatch-ids degrade: no
        # probed candidate jointly matches all three selected refs at once,
        # so the fetch is skipped and every survivor set below is trivially
        # empty.
        _log_no_joint_facet_match(request.stat_var, selections)
        schema = observation_schema(request.declared_roles)
        batches = []

    table = _admitted_series_rows(
        pa.Table.from_batches(batches, schema=schema), series_roles, allowed_series_keys
    )
    # Resolve role-shaped omissions before choosing latest, but defer all
    # arithmetic.  Search may select an exact global period after its reads
    # finish; conversion must see only that period.
    currencies: EntityCurrencies | None = None
    series_currencies: dict[tuple[str, ...], EntityCurrencies | None] = {}
    year: int | None = None
    if series_roles and table.num_rows:
        none = table.slice(0, 0)

        @cache
        def series_rows_at(year: int) -> dict[tuple[str, ...], pa.Table]:
            """Each series' rows at `year`, from one read of the member however many
            series end after it."""
            assert read_year is not None  # a table with rows was fetched
            return dict(_split_by_roles(read_year(year), series_roles))

        prepared_tables: list[pa.Table] = []
        for key, rows in _split_by_roles(table, series_roles):
            rows, series_currencies[key], year, _omitted = _servable_rows(
                rows,
                window=request.window,
                selections=selections,
                admission=probe.admission,
                conversion_plan=conversion_plan,
                elected=(
                    _ElectedRead(
                        elect=elect, read=lambda at, key=key: series_rows_at(at).get(key, none)
                    )
                    if elect is not None
                    else None
                ),
            )
            prepared_tables.append(rows)
        table = pa.concat_tables(prepared_tables)
    else:
        table, currencies, year, _omitted = _servable_rows(
            table,
            window=request.window,
            selections=selections,
            admission=probe.admission,
            conversion_plan=conversion_plan,
            elected=(
                _ElectedRead(elect=elect, read=read_year, counted=counted)
                if elect is not None and read_year is not None
                else None
            ),
        )
    return _PreparedPreview(
        request=request,
        probe=probe,
        selections=selections,
        conversion_plan=conversion_plan,
        table=table,
        currencies=currencies,
        series_currencies=series_currencies,
        limit=limit,
        ceiling=ceiling,
        mint_full_frame=mint_full_frame,
        series_roles=tuple(series_roles),
        series_limit=series_limit,
        year=year,
    )


def _finish_preview_member(
    prepared: _PreparedPreview, *, time_period: str | None = None
) -> _PreviewOutcome:
    """Apply conversion to a prepared preview's rows at `time_period`, or with none given
    at the period a mint of the request serves (`columnar.elected_rows`)."""
    request = prepared.request
    table = prepared.table
    if time_period is not None:
        table = table.filter(pc.equal(table["TIME_PERIOD"], pa.scalar(time_period)))
    else:
        table = columnar.elected_rows(table, year=prepared.year)
    conversion_plan = prepared.conversion_plan
    series_conversion_records: dict[tuple[str, ...], ConversionRecord] = {}
    if prepared.series_roles and (table.num_rows or prepared.series_currencies):
        series_tables = dict(_split_by_roles(table, prepared.series_roles))
        empty = table.slice(0, 0)
        converted: list[pa.Table] = []
        for key in dict.fromkeys([*prepared.series_currencies, *series_tables]):
            rows = series_tables.get(key, empty)
            series_table, series_record = apply_member_conversion(
                conversion_plan,
                rows,
                declared_roles=request.declared_roles,
                currencies=prepared.series_currencies.get(key),
            )
            converted.append(series_table)
            if series_record is not None and (rows.num_rows or series_record.omitted_entities):
                series_conversion_records[key] = series_record
        table = pa.concat_tables(converted)
        conversion_record = merged_record(list(series_conversion_records.values()))
    else:
        table, conversion_record = apply_member_conversion(
            conversion_plan,
            table,
            declared_roles=request.declared_roles,
            currencies=prepared.currencies,
        )
    conversion_recipe_id = (
        conversion_plan.recipe.recipe_id
        if conversion_plan is not None and conversion_plan.recipe is not None
        else None
    )

    applied_window = compute_applied_window(request.window, columnar.served_year_bounds(table))
    coverage = columnar.coverage(
        table, declared_roles=request.declared_roles, applied_window=applied_window
    )

    # Built from `table` before the truncation branch below replaces the
    # name with the delivered subset. `_PreviewOutcome.full_frame` is the
    # only frame over every survivor; a caller minting `content_version`
    # over the full survivor set reads this one, since a mint of the same
    # request hashes every survivor row rather than a `limit`-bounded slice.
    # A caller reporting no `content_version` skips it.
    full_frame = (
        build_frame_from_table(table, role_keys=request.declared_roles, stat_var=request.stat_var)
        if prepared.mint_full_frame
        else None
    )

    total_rows = table.num_rows
    series_total_rows: dict[tuple[str, ...], int] = {}
    if prepared.series_roles and table.num_rows:
        assert prepared.series_limit is not None
        series_tables = _split_by_roles(table, prepared.series_roles)
        series_total_rows = {key: rows.num_rows for key, rows in series_tables}
        truncated = any(rows > prepared.series_limit for rows in series_total_rows.values())
        delivered_table = (
            pa.concat_tables(
                [_most_recent_rows(rows, prepared.series_limit) for _key, rows in series_tables]
            )
            if truncated
            else table
        )
    else:
        truncated = total_rows > prepared.limit
        delivered_table = _most_recent_rows(table, prepared.limit)

    served_entities_by_role = columnar.served_entities_by_role(table, roles=request.declared_roles)

    frame = build_frame_from_table(
        delivered_table, role_keys=request.declared_roles, stat_var=request.stat_var
    )
    return _PreviewOutcome(
        frame=frame,
        full_frame=full_frame,
        served_entities_by_role=served_entities_by_role,
        selections=dict(prepared.selections),
        coverage=coverage,
        applied_window=applied_window,
        total_rows=total_rows,
        truncated=truncated,
        conversion_record=conversion_record,
        conversion_recipe_id=conversion_recipe_id,
        default_scope=prepared.probe.scope,
        series_conversion_records=series_conversion_records,
        series_total_rows=series_total_rows,
    )


def _check_role_divergence(roles: tuple[str, ...], request: MaterialisationRequest) -> None:
    """Raises `UpstreamResponseError` naming both sets when the SDMX
    response's own header roles disagree with `request.declared_roles` as
    sets. The two are minted at different times over the same StatVar
    (`declared_roles` at shape resolution, `roles` at this fetch), so a graph
    write landing between them (e.g. an `observationProperties` edit) is the
    reachable cause, not a caller mistake. Comparison ignores order: wire
    order carries no meaning `Frame.build`'s own `role_keys` depends on.
    """
    reported = set(roles)
    expected = set(request.declared_roles)
    if reported != expected:
        raise UpstreamResponseError(
            f"SDMX response for stat_var {request.stat_var!r} reports role "
            f"columns {sorted(reported)!r}, but the resolved shape declares "
            f"{sorted(expected)!r}"
        )


def _flatten_bound_entities(role_bindings: Mapping[str, EntityFilter]) -> tuple[str, ...]:
    """Every plain dcid across every `Sequence[str]`-bound role, deduped
    (first-seen order). A `ContainedIn`-bound role is skipped entirely (its
    expansion isn't known upfront), a documented limitation shared with
    `probe_facets`' own note.
    """
    seen: set[str] = set()
    entities: list[str] = []
    for filt in role_bindings.values():
        if isinstance(filt, ContainedIn):
            continue
        for dcid in filt:
            if dcid not in seen:
                seen.add(dcid)
                entities.append(dcid)
    return tuple(entities)
