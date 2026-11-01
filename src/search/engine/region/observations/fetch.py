"""Fetch and shape observation previews for standard and role-bearing queries.

A per-series preview reads the series its read plan serves (`read_plan.PairPlan.served`)
through materialise with the plan's candidates and a filter admitting only the plan's
group, so materialise elects nothing the plan did not. It makes one read per member and
bound role tuple (`_Dispatch`): a role the answer leaves open is read open, and the rows
are split into the plan's series by that role's dcid. Under `latest` each read uses SDMX
`LATEST`, reads the year its StatVar elected when a series ends after it, as a mint does
(`materialise.prepare_materialise_preview`), and prepares retained rows without arithmetic.
Once every read has returned, each StatVar's one period is selected across them, the
newest in its elected year when a read holds a row there and else the newest any read
holds, then each prepared read finalises that period and converts it. A StatVar that
elected no year serves a period newer than the coverage index states, as a mint's `LATEST`
read serves it. A read whose dispatched facets fall outside its planned
group is a plan mismatch: it is logged and its series count as failed. A unit request
converts the same reads. Reads run in a bounded thread pool under a process-wide semaphore
and stop at `observation_settings().obs_preview_max_pairs` series. Scope-defaulted queries
read the scope plan's one facet (`PairPlan.scope_facet_id`), and role-bearing queries over
the pair cap read the bulk plan's elected candidate over its year list (`read_plan.BulkPlan`).
A native unit request narrows the plan before either read, and a converting one converts
the points either read returns (`bulk_conversion`); neither bulk branch elects a facet itself.

`_build_observations` stays synchronous because resubmit calls it from a worker thread with
no event loop. Its per-pair fan-out uses `ThreadPoolExecutor`.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Protocol, Self, cast

from code_index.shared import get_index
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dc_sdmx import SdmxClientError
from dfo_platform.executor import submit_with_context
from dfo_platform.request_deadline import bounded_deadline
from materialise import (
    DefaultFacetUndefinedError,
    FacetBinding,
    FacetNotAvailableError,
    GraphReader,
    GraphReadError,
    MaterialisationRequest,
    PreparedPreview,
    PreviewResult,
    UpstreamResponseError,
    finish_materialise_preview,
    materialise_preview,
    prepare_materialise_preview,
)
from materialise.convert import ConversionRecord
from materialise.facets import FacetSelection, UnitRequest, merge_facets_by_series
from materialise.method_status import status_of
from materialise.rate_tables import RateTableCache
from materialise.recipes import (
    NonCurrencyUnitError,
    RateCoverageGapError,
    RecipeNotDeclaredError,
    cause_from_recipe_error,
)
from materialise.series_conversion import SeriesConversion
from materialise.window import latest_periods
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.degraded_reads import note_degraded_read
from search.engine.errors import UPSTREAM_READ_ERRORS, GraphInfraError
from search.engine.region.codes import (
    OBSERVATIONS_CLIENTS_UNAVAILABLE,
    OBSERVATIONS_NO_MATCHING_SOURCE,
    OBSERVATIONS_NO_POINTS,
    OBSERVATIONS_NOTHING_TO_PROBE,
    OBSERVATIONS_PROBE_FAILED,
    OBSERVATIONS_TIMED_OUT,
)
from search.engine.region.observations.bulk_conversion import (
    BulkConversion,
    any_combination_converts,
)
from search.engine.region.observations.bulk_role_series import _bulk_role_series
from search.engine.region.observations.bulk_scope import BulkScope, _bulk_scope_series
from search.engine.region.result import FacetAxisFilter
from search.engine.retrieval.read_plan import (
    Admits,
    BulkPlan,
    PairPlan,
    PlanSeries,
    PlanYears,
    ReadPlan,
    plan_window,
)
from search.models import (
    IdentityUnitRequest,
    SpecObservations,
    TimeWindow,
    Warning,
    omitted_observations,
)
from search.observations import (
    OBSERVATIONS_PREVIEW_TOO_LARGE,
    OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED,
    PreviewMode,
    SeriesRead,
    assemble_per_series,
    currency_unit_ref,
    observation_settings,
    observations_too_large,
    observations_windowed_scope,
    preview_mode,
)

if TYPE_CHECKING:
    from search.engine.retrieval.member_facts import MemberFacts
    from series_coverage.index import CoverageIndex
    from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)


def _standard_pairs(
    sv_dcids: list[str], entity_dcids: Sequence[str]
) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    """Build the per-pair input required by materialise's single-facet request design."""
    return [(sv, (("observationAbout", entity),)) for sv in sv_dcids for entity in entity_dcids]


def _role_bearing_pairs(
    sv_dcids: list[str], obs_entities: Mapping[str, Sequence[str]]
) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    """Build the StatVar-by-role-tuple product.

    Incomplete role tuples produce no pairs.
    """
    role_names = list(obs_entities.keys())
    role_dcid_lists = [list(obs_entities[role]) for role in role_names]
    if not role_names or any(not dcids for dcids in role_dcid_lists):
        return []
    return [
        (sv, tuple(zip(role_names, combo, strict=True)))
        for sv in sv_dcids
        for combo in itertools.product(*role_dcid_lists)
    ]


def _elected_requires_conversion(
    sv_dcids: Sequence[str],
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None,
) -> bool:
    """Whether any elected combination requires conversion, which a bulk read performs on
    its points (`bulk_conversion`).

    Missing elections conservatively require conversion. StatVars absent from `elected` add
    no series. A member with any combination converting by arithmetic
    (`bulk_conversion.any_combination_converts`) requires it.
    """
    if elected is None:
        return True
    return any(any_combination_converts(elected.get(sv, ())) for sv in sv_dcids)


def _native_unit_refs(
    sv_dcids: Sequence[str],
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None,
) -> frozenset[str] | None:
    """The native unit refs the elections chose, once conversion has been ruled out, which a
    bulk read narrows its plan to so it never mixes currencies. None when elections or
    native refs are absent. Refs come from elections, not the request string."""
    if elected is None:
        return None
    native_refs = frozenset(
        selection.ref
        for sv in sv_dcids
        for combo in elected.get(sv, ())
        if (selection := combo.get("unit")) is not None and selection.ref is not None
    )
    return native_refs or None


_OBS_PREVIEW_TOTAL_SEMAPHORE = threading.BoundedSemaphore(settings().obs_preview_total_concurrency)
"""Process-wide gate on in-flight per-pair preview fetches, sized by
settings().obs_preview_total_concurrency. Each pair's `materialise_preview` dispatch in
`_preview_one` holds a permit for the call's duration (`_preview_permit`), however many
`_build_observations` calls fan out at once. settings().obs_preview_concurrency bounds one
call's pool; this bounds the sum across calls. Bulk reads and label lookups are outside it.
Tests that need a different bound patch this attribute directly, since
`threading.BoundedSemaphore` cannot be resized.
"""


@contextmanager
def _preview_permit(deadline: float | None) -> Iterator[None]:
    """Hold one `_OBS_PREVIEW_TOTAL_SEMAPHORE` permit for the block. The wait ends after
    `graph_timeout_s`, which raises `UpstreamResponseError`, or at `deadline` narrowed by the
    bound request deadline (`bounded_deadline`) when that comes first, which raises
    `TimeoutError`, as a read past the deadline does."""
    timeout = settings().graph_timeout_s
    bound = bounded_deadline(deadline)
    remaining = None if bound is None else max(0.0, bound - time.monotonic())
    if remaining is not None and remaining < timeout:
        if not _OBS_PREVIEW_TOTAL_SEMAPHORE.acquire(timeout=remaining):
            raise TimeoutError(
                "observation-preview permit not acquired before the deadline "
                "(process-wide semaphore held by other in-flight preview calls)"
            )
    elif not _OBS_PREVIEW_TOTAL_SEMAPHORE.acquire(timeout=timeout):
        raise UpstreamResponseError(
            f"observation-preview permit not acquired within {timeout}s "
            "(process-wide semaphore held by other in-flight preview calls)"
        )
    try:
        yield
    finally:
        _OBS_PREVIEW_TOTAL_SEMAPHORE.release()


# `obs_preview_max_pairs` caps role products, the series a plan serves, and actual dispatch
# items. Overflow skips the whole preview. Partial series would misstate coverage.
#
# Reads run in a pool of at most `obs_preview_concurrency` workers (about 0.2 s per
# `materialise_preview` call) under the process-wide semaphore. Permit waits end after
# `graph_timeout_s` or at the deadline (`_preview_permit`). Bulk reads and label lookups
# are outside this gate.


_READ_GRAPH = "graph"
_READ_SDMX = "sdmx"
_READ_UPSTREAM_RESPONSE = "upstream response"
_EVERY_FAILING_READ_ORDER: tuple[str, ...] = (_READ_GRAPH, _READ_SDMX, _READ_UPSTREAM_RESPONSE)
"""The three reads a per-pair hard failure resolves to, gathered in the one
order _every_failing_read_message joins them -- a stable order that is a
property of this tuple, not of the set iteration behind `hard_failed_reads`."""


def _read_named_by(exc: BaseException) -> str:
    """Name the failing read by exception type.

    `UpstreamResponseError` names neither read.
    """
    if isinstance(exc, GraphReadError):
        return _READ_GRAPH
    if isinstance(exc, SdmxClientError):
        return _READ_SDMX
    return _READ_UPSTREAM_RESPONSE


def _every_failing_read_message(
    reads: set[str], hard_failed: int, dispatched: int, *, plan_mismatched: int = 0
) -> str:
    """Describe failed dispatches and list distinct reads in `_EVERY_FAILING_READ_ORDER`.

    `plan_mismatched` of the failures dispatched a facet outside their planned group. An
    empty read set with no mismatch means no pair had a dispatchable facet, not an upstream
    read failure.
    """
    failed = (
        f"{hard_failed} of {dispatched} requested series could not be served; "
        if dispatched > hard_failed
        else f"all {dispatched} requested series could not be served; "
    )
    if plan_mismatched:
        failed += f"{plan_mismatched} dispatched a facet outside the preview's read plan; "
    if not reads and plan_mismatched:
        return f"{failed}Spec.resolved.observations was not populated."
    if not reads:
        return (
            f"{failed}no (stat_var, entity) pair had a dispatchable facet, so "
            "Spec.resolved.observations was not populated."
        )
    named = [read for read in _EVERY_FAILING_READ_ORDER if read in reads]
    joined = " and ".join(named) if len(named) == 2 else ", ".join(named)
    noun = "read" if len(named) == 1 else "reads"
    return f"{failed}the {joined} {noun} failed, so Spec.resolved.observations was not populated."


def _preview_too_large(count: int) -> tuple[SpecObservations, Warning]:
    """The too-large omission (`search.observations.observations_too_large`) of a preview
    holding `count` series or pairs, and its warning."""
    observations = observations_too_large(count)
    message = _omission_message(observations, OBSERVATIONS_PREVIEW_TOO_LARGE)
    return observations, Warning(
        code=OBSERVATIONS_PREVIEW_TOO_LARGE, severity="warn", message=message
    )


def _series_too_large(
    count: int, sv_dcids: list[str], obs_entities: Mapping[str, Sequence[str]]
) -> tuple[SpecObservations, Warning]:
    """`_preview_too_large` for a per-series read holding `count` series, logged."""
    logger.warning(
        "observations preview skipped for sv_dcids=%r entities=%r: %d "
        "(stat_var, entity) pairs with data exceeds the %d-pair ceiling",
        sv_dcids,
        obs_entities,
        count,
        observation_settings().obs_preview_max_pairs,
    )
    return _preview_too_large(count)


def _omission_message(observations: SpecObservations, code: str) -> str:
    """The message of an omitted preview whose reason is `code`, for the warning that repeats
    it. Each caller names its code constant at its `Warning(code=...)` site."""
    assert observations.omitted is not None
    assert observations.omitted.reason == code
    return observations.omitted.message


class _Narrowable(Protocol):
    """A bulk read's plan, which a native unit request narrows (`PairPlan`, `BulkPlan`)."""

    def narrowed(self, admits: Admits) -> Self: ...


_CONVERSION_CLIENTS_UNAVAILABLE = (
    "the observation preview was not fetched: converting it needs the unit decision and "
    "the materialise graph, SDMX client and rate cache."
)


def _bulk_unit[P: _Narrowable](
    plan: P,
    sv_dcids: Sequence[str],
    *,
    unit_request: IdentityUnitRequest | None,
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    rate_cache: RateTableCache | None,
    deadline: float | None,
) -> tuple[P, BulkConversion | None] | None:
    """A bulk read's plan and conversion under `unit_request`. A request that needs
    conversion (`_elected_requires_conversion`) converts the read's points
    (`bulk_conversion`), which needs the elections, the materialise clients and the rate
    cache, else None. A natively satisfied request narrows the plan to the elected units, so
    the read never mixes currencies. No request returns the plan as it is."""
    if unit_request is None:
        return plan, None
    if not _elected_requires_conversion(sv_dcids, elected):
        if (native_refs := _native_unit_refs(sv_dcids, elected)) is not None:
            plan = plan.narrowed(lambda _sv, candidate: candidate.unit in native_refs)
        return plan, None
    if (
        elected is None
        or materialise_graph is None
        or materialise_sdmx is None
        or rate_cache is None
    ):
        return None
    # Deferred to avoid the import cycle with `convertibility`.
    from search.engine.region.convertibility import unit_request_for_materialise

    return plan, BulkConversion(
        unit_request=unit_request_for_materialise(unit_request),
        elected=elected,
        graph=materialise_graph,
        sdmx=materialise_sdmx,
        rate_cache=rate_cache,
        deadline=deadline,
    )


def _bulk_scope_dispatch(
    plan: PairPlan,
    sv_dcids: list[str],
    obs_entities: Mapping[str, Sequence[str]],
    *,
    scope: BulkScope,
    graph: EngineGraphClient,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    rate_cache: RateTableCache | None,
    facet_filter: Mapping[str, FacetAxisFilter] | None,
    unit_request: IdentityUnitRequest | None,
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None,
    member_constraints: Mapping[str, Mapping[str, str]],
    deadline: float | None,
) -> tuple[SpecObservations, Warning | None]:
    """Dispatch a scope-defaulted standard query through the one-facet bulk branch, serving
    the facet `plan` elected.

    A range plan returns `OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED`. `latest` and dateless
    plans supply one latest point per place. A unit request converts those points or narrows
    the plan to the native units (`_bulk_unit`); a conversion without its inputs returns
    `OBSERVATIONS_CLIENTS_UNAVAILABLE` before any read.
    """
    if isinstance(plan.years, TimeWindow):
        observations = observations_windowed_scope()
        message = _omission_message(observations, OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED)
        return observations, Warning(
            code=OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED, severity="warn", message=message
        )
    unit = _bulk_unit(
        plan,
        sv_dcids,
        unit_request=unit_request,
        elected=elected,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        rate_cache=rate_cache,
        deadline=deadline,
    )
    if unit is None:
        return omitted_observations(
            OBSERVATIONS_CLIENTS_UNAVAILABLE, _CONVERSION_CLIENTS_UNAVAILABLE
        ), None
    plan, conversion = unit
    return _bulk_scope_series(
        plan,
        sv_dcids,
        obs_entities.get("observationAbout", ()),
        scope=scope,
        graph=graph,
        facet_filter=facet_filter,
        member_constraints=member_constraints,
        conversion=conversion,
    )


def _role_bearing_bulk_dispatch(
    plan: ReadPlan | None,
    sv_dcids: list[str],
    obs_entities: Mapping[str, Sequence[str]],
    pairs: list[tuple[str, tuple[tuple[str, str], ...]]],
    *,
    sdmx: EngineSdmxClient | None,
    graph: EngineGraphClient,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    rate_cache: RateTableCache | None,
    facet_filter: Mapping[str, FacetAxisFilter] | None,
    unit_request: IdentityUnitRequest | None,
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None,
    member_constraints: Mapping[str, Mapping[str, str]],
    deadline: float | None,
) -> tuple[SpecObservations, Warning | None]:
    """Read a role-bearing answer in bulk (`PreviewMode.ROLE_BULK`).

    Only a bulk plan (`read_plan.BulkPlan`) reads. Without one, as for an answer with no plan
    whose pair product passes `obs_preview_max_pairs`, it returns
    `OBSERVATIONS_PREVIEW_TOO_LARGE`. A bulk read needs `sdmx`, and a range plan needs its
    year list (`BulkPlan.read_years`); otherwise it returns `OBSERVATIONS_PREVIEW_TOO_LARGE`
    rather than a partial preview. A unit request converts the read's points or narrows the
    plan to the native units (`_bulk_unit`); a conversion without its inputs returns
    `OBSERVATIONS_CLIENTS_UNAVAILABLE`. A `latest` read serves each member its own latest
    period (`_bulk_role_series`).
    """
    readable = (
        isinstance(plan, BulkPlan)
        and sdmx is not None
        and not (isinstance(plan.years, TimeWindow) and plan.read_years is None)
    )
    if not readable:
        # Skip the whole preview. A partial group would misstate coverage.
        logger.warning(
            "observations preview skipped for sv_dcids=%r entities=%r: %d "
            "(stat_var, entity) pairs exceeds the %d-pair ceiling",
            sv_dcids,
            obs_entities,
            len(pairs),
            observation_settings().obs_preview_max_pairs,
        )
        return _preview_too_large(len(pairs))
    assert isinstance(plan, BulkPlan) and sdmx is not None  # `readable`
    unit = _bulk_unit(
        plan,
        sv_dcids,
        unit_request=unit_request,
        elected=elected,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        rate_cache=rate_cache,
        deadline=deadline,
    )
    if unit is None:
        return omitted_observations(
            OBSERVATIONS_CLIENTS_UNAVAILABLE, _CONVERSION_CLIENTS_UNAVAILABLE
        ), None
    plan, conversion = unit
    return _bulk_role_series(
        plan,
        sv_dcids,
        sdmx=sdmx,
        graph=graph,
        facet_filter=facet_filter,
        deadline=deadline,
        member_constraints=member_constraints,
        conversion=conversion,
    )


_RoleTuple = tuple[tuple[str, str], ...]
_SeriesKey = tuple[str, _RoleTuple]


@dataclass(frozen=True)
class _Dispatch:
    """One preview read: a member at one bound role tuple, and every planned series the
    read serves. `bound` holds the roles the answer's entities name. A role a series binds
    beyond them is open on the read, so the series of one bound tuple share the read and
    each takes the rows carrying its own dcid on that role (`_split_series`). A fully bound
    tuple is one series, read alone. `years` is the window the read takes its rows in, and
    `elected_year` the year the member elected under `latest`, None when it elected none."""

    stat_var: str
    bound: _RoleTuple
    series: tuple[PlanSeries, ...]
    years: PlanYears
    elected_year: int | None = None

    @property
    def group_ids(self) -> frozenset[str]:
        """The facet ids the read may dispatch: the member's group, which its series share."""
        return frozenset(facet_id for s in self.series for facet_id in s.group_ids)

    @property
    def open_roles(self) -> tuple[str, ...]:
        """The roles the read leaves open: those its series bind beyond `bound`. A member's
        series bind the same roles, so the first speaks for all."""
        bound = {role for role, _dcid in self.bound}
        return tuple(role for role, _dcid in self.series[0].role_tuple if role not in bound)

    def bindings(self, unit_request: UnitRequest | None) -> Mapping[str, FacetBinding]:
        """The bindings that dispatch the planned group: none, or under a unit request the
        unit axis in `state="request"`, which converts the group's rows."""
        if unit_request is None:
            return {}
        return {"unit": FacetBinding(state="request", request=unit_request)}


def _dispatches(
    series: Sequence[PlanSeries],
    obs_entities: Mapping[str, Sequence[str]],
    *,
    years: PlanYears,
    elected: Mapping[str, int],
) -> list[_Dispatch]:
    """Group the planned series into reads: one per member and tuple of the roles
    `obs_entities` names, in series order, each over `years`.

    Under `latest` materialise asks SDMX for `LATEST` rows, and for the rows in the year
    the member elected (`elected`, from `PairPlan.elected_years`) when a series ends after
    it, as a mint does. The caller later finalises the StatVar's one period across its
    prepared reads, before it converts."""
    bound_roles = {role for role, dcids in obs_entities.items() if dcids}
    grouped: dict[tuple[str, _RoleTuple], list[PlanSeries]] = {}
    for s in series:
        bound = tuple((role, dcid) for role, dcid in s.role_tuple if role in bound_roles)
        grouped.setdefault((s.stat_var, bound), []).append(s)
    return [
        _Dispatch(
            stat_var=sv,
            bound=bound,
            series=tuple(members),
            years=years,
            elected_year=elected.get(sv) if years == "latest" else None,
        )
        for (sv, bound), members in grouped.items()
    ]


@dataclass(frozen=True)
class _SeriesRead:
    """One planned series' share of a read: its points in date order, the rows it holds
    before the read's row limit or any display truncation, the read's selections and its
    own conversion record, and whether a row it serves carries an estimated method
    (`_holds_estimates`)."""

    points: list[tuple[str, float | None]]
    total_rows: int
    selections: Mapping[str, FacetSelection]
    conversion_record: ConversionRecord | None
    holds_estimates: bool = False


def _split_series(dispatch: _Dispatch, result: PreviewResult) -> dict[_SeriesKey, _SeriesRead]:
    """Split a read's rows into its planned series. A read with no open role serves one
    series, which takes every row. Otherwise each series takes the rows carrying its dcids
    on the open roles, with the latest period, the row count before the read's limit and the
    conversion record materialise kept for that series (`materialise_preview`'s
    `series_roles`). Rows of a role tuple the plan does not serve are dropped."""
    open_roles = dispatch.open_roles
    if not open_roles:
        (series,) = dispatch.series
        return {
            (series.stat_var, series.role_tuple): _SeriesRead(
                points=sorted(_frame_points(result.frame.rows, result), key=lambda p: p[0]),
                total_rows=result.total_rows,
                selections=result.selections,
                conversion_record=result.conversion_record,
                holds_estimates=_holds_estimates(result.frame.rows, columns=result.frame.columns),
            )
        }
    columns = result.frame.columns
    indices = [columns.index(role) for role in open_roles]
    rows_by_dcids: dict[tuple[str, ...], list[tuple[object, ...]]] = {}
    for row in result.frame.rows:
        rows_by_dcids.setdefault(tuple(cast(str, row[i]) for i in indices), []).append(row)
    reads: dict[_SeriesKey, _SeriesRead] = {}
    for s in dispatch.series:
        by_role = dict(s.role_tuple)
        dcids = tuple(by_role[role] for role in open_roles)
        rows = rows_by_dcids.get(dcids, [])
        points = sorted(_frame_points(rows, result), key=lambda p: p[0])
        reads[(s.stat_var, s.role_tuple)] = _SeriesRead(
            points=points,
            total_rows=result.series_total_rows.get(dcids, 0),
            selections=result.selections,
            conversion_record=result.series_conversion_records.get(dcids),
            holds_estimates=_holds_estimates(rows, columns=columns),
        )
    return reads


def _holds_estimates(rows: Sequence[tuple[object, ...]], *, columns: Sequence[str]) -> bool:
    """Whether a row of `rows` carries a measurement method the measurement-method
    classification classes estimated under the row's provenance
    (`materialise.method_status.status_of`), as the rows of a continuation
    (`materialise.facets.FacetSelection.continued_by`) do."""
    method, provenance = columns.index("measurementMethod"), columns.index("provenance")
    return any(
        status_of(cast("str | None", row[method]), cast("str | None", row[provenance]))
        == "estimated"
        for row in rows
    )


def _at_latest_periods(
    reads: Mapping[_SeriesKey, _SeriesRead],
) -> tuple[dict[_SeriesKey, _SeriesRead], tuple[_SeriesKey, ...]]:
    """`reads` under a `latest` window, each StatVar reduced to its one latest period across
    every read of it (`materialise.window.latest_periods` over their points' dates), as a
    mint keeps a StatVar's rows at that period alone (`materialise.columnar.latest_rows`).
    The reads of a StatVar arrive finalised at one period (`_collect_pair_results`), the
    one in its elected year when a read holds it. Each read keeps its point at that
    period; a read holding none, because its series holds no row there, is dropped.
    Returns the kept reads and the dropped keys."""
    latest = latest_periods(
        (sv, date) for (sv, _t), read in reads.items() for date, _v in read.points
    )
    kept: dict[_SeriesKey, _SeriesRead] = {}
    dropped: list[_SeriesKey] = []
    for key, read in reads.items():
        points = [point for point in read.points if point[0] == latest.get(key[0])]
        if points:
            kept[key] = replace(read, points=points, total_rows=len(points))
        else:
            dropped.append(key)
    return kept, tuple(dropped)


def _every_row_omitted(read: _SeriesRead) -> bool:
    """Whether a conversion left out every row `read` held, because no single currency names
    its entity (`ConversionRecord.omitted_entities`) or the deflator table lacks its values
    (`ConversionRecord.deflator_omitted`)."""
    record = read.conversion_record
    return (
        read.total_rows == 0
        and record is not None
        and bool(record.omitted_entities or record.deflator_omitted)
    )


def _frame_points(
    rows: Sequence[tuple[object, ...]], result: PreviewResult
) -> list[tuple[str, float | None]]:
    """(time_period, value) pairs off `rows` of `result.frame`, already window-filtered and
    value-valid (materialise's own `columnar.observed_mask` gate)."""
    columns = result.frame.columns
    date_idx, value_idx = columns.index("time_period"), columns.index("value")
    return [(cast(str, row[date_idx]), cast("float | None", row[value_idx])) for row in rows]


def _dispatched_facet_ids(result: PreviewResult) -> frozenset[str]:
    """The facet ids materialise's selections dispatched."""
    return frozenset(
        selection.dispatch_facet_id
        for selection in result.selections.values()
        if selection.dispatch_facet_id is not None
    )


def _dispatched_selection_ids(selections: Mapping[str, FacetSelection]) -> frozenset[str]:
    """Dispatch facet ids from a prepared or finished materialise selection."""
    return frozenset(
        selection.dispatch_facet_id
        for selection in selections.values()
        if selection.dispatch_facet_id is not None
    )


def _finish_preview_one(prepared: PreparedPreview, *, time_period: str | None) -> PreviewResult:
    """Finalise under the same process-wide permit as preparation.

    Role-shaped conversion can lazily load its tables during finalisation.
    """
    with _preview_permit(None):
        return finish_materialise_preview(prepared, time_period=time_period)


def _preview_one(
    dispatch: _Dispatch,
    *,
    unit_request: UnitRequest | None,
    materialise_sdmx: MaterialiseSdmxClient,
    materialise_graph: GraphReader,
    deadline: float | None,
    rate_cache: RateTableCache | None,
    coverage_index: CoverageIndex,
    member_facts: Mapping[str, MemberFacts] | None,
    prepare_only: bool = False,
) -> PreviewResult | PreparedPreview:
    """Build and dispatch one materialisation request on a pool worker.

    Materialise reads the plan's candidates for the read's series instead of probing,
    merged by facet id and narrowed to the member's planned group, so it elects nothing the
    plan did not. The request binds the read's bound roles and leaves any other role open,
    naming it in `series_roles` so that materialise keeps each series' own conversion record.
    It reads the dispatch's `years`; under `latest` it prepares SDMX `LATEST` rows, and the
    rows in the dispatch's `elected_year` when a series ends after it, which the caller
    finalises at the StatVar's one period before converting. Each series the read
    returns keeps its most recent `obs_preview_max_rows` rows (`series_limit`), and the read's
    `limit` and `ceiling`, which sizes its raw fetch, are that many per planned series. Shared
    inputs are passed explicitly. Only the permit wait reads the bound request deadline. Resolve
    `materialise_preview` at call time so tests can patch this module global. The process-wide
    permit covers only the upstream call (`_preview_permit`), and a wait that runs out fails
    this read, while other reads continue. A read whose conversion
    omitted every row returns them empty, with the omitted entities on its conversion records:
    whether the request refuses is the bound-unit gate's decision
    (`convertibility.unit_coverage`).
    """
    sv = dispatch.stat_var
    role_bindings: dict[str, list[str]] = {}
    for role, dcid in dispatch.bound:
        role_bindings.setdefault(role, []).append(dcid)
    mf = (member_facts or {}).get(sv)
    declared_roles = (
        mf.read_declared_roles()
        if mf is not None
        else tuple(role for role, _dcid in dispatch.series[0].role_tuple)
    )
    group_ids = dispatch.group_ids
    by_series = {"|".join(s.key): s.candidates for s in dispatch.series}
    request = MaterialisationRequest(
        stat_var=f"dcid:{sv}",
        role_bindings=role_bindings,
        declared_roles=declared_roles,
        facet_bindings=dispatch.bindings(unit_request),
        window=plan_window(dispatch.years),
        candidate_filter=lambda facet: facet.facet_id in group_ids,
        # An answer may state that its facet names no source.
        requires_provenance=False,
    )
    with _preview_permit(deadline):
        if prepare_only:
            return prepare_materialise_preview(
                request,
                sdmx=materialise_sdmx,
                graph=materialise_graph,
                limit=settings().obs_preview_max_rows * len(dispatch.series),
                ceiling=settings().obs_preview_max_rows * len(dispatch.series),
                series_limit=settings().obs_preview_max_rows,
                rate_cache=rate_cache,
                coverage=coverage_index,
                deadline=deadline,
                precomputed=merge_facets_by_series(by_series, list(by_series)),
                series_roles=dispatch.open_roles,
                allowed_series_keys=(
                    frozenset(
                        tuple(dict(series.role_tuple)[role] for role in dispatch.open_roles)
                        for series in dispatch.series
                    )
                    if dispatch.open_roles
                    else None
                ),
                elected_year=dispatch.elected_year,
            )
        return materialise_preview(
            request,
            sdmx=materialise_sdmx,
            graph=materialise_graph,
            limit=settings().obs_preview_max_rows * len(dispatch.series),
            ceiling=settings().obs_preview_max_rows * len(dispatch.series),
            series_limit=settings().obs_preview_max_rows,
            rate_cache=rate_cache,
            coverage=coverage_index,
            deadline=deadline,
            precomputed=merge_facets_by_series(by_series, list(by_series)),
            series_roles=dispatch.open_roles,
            refuse_when_every_row_omitted=False,
        )


@dataclass(frozen=True)
class _PairFetchOutcome:
    """Series reads and rejection counts from one dispatch round, counted per series."""

    pair_results: dict[_SeriesKey, _SeriesRead]
    hard_failed_pairs: int
    hard_failed_reads: frozenset[str]
    timed_out_pairs: int
    facet_rejected_pairs: int
    conversion_rejected_pairs: int
    conversion_rejected_cause: str | None
    unfiltered_empty_pairs: int
    plan_mismatched_pairs: int = 0
    # Planned series a `latest` read returned no point for at their StatVar's latest period
    # (`_at_latest_periods`): coverage and SDMX disagree, so they are not served.
    short_of_latest_pairs: int = 0


def _collect_pair_results(
    pool: ThreadPoolExecutor,
    dispatches: list[_Dispatch],
    *,
    unit_request: UnitRequest | None,
    materialise_sdmx: MaterialiseSdmxClient,
    materialise_graph: GraphReader,
    deadline: float | None,
    rate_cache: RateTableCache | None,
    coverage_index: CoverageIndex,
    member_facts: Mapping[str, MemberFacts] | None,
    facet_filter: Mapping[str, FacetAxisFilter] | None,
) -> _PairFetchOutcome:
    """Run every read and return its series and the per-series failure counts.

    Futures are read in dispatch order for deterministic results. One failure does not stop
    other reads, and a failed read counts each series it would have served.

    - `FacetNotAvailableError` is a facet rejection with a filter or a unit request.
      Otherwise it is an unfiltered empty read. `DefaultFacetUndefinedError` means no
      candidate is viable: a rejection under a unit request, which can empty the pool, and
      otherwise a hard failure. Materialise prefers it when both errors arise.
    - `RecipeNotDeclaredError`, `RateCoverageGapError`, and `NonCurrencyUnitError` are
      conversion rejections. Report recipe causes or the unsupported unit. The aggregate
      warning keeps the last rejection cause. `RateTableInvalidError` and
      `UpstreamVersionChangedError` propagate because they invalidate shared recipe tables.
      `MaterialisationUsageError` propagates as a code error.
    - `UPSTREAM_READ_ERRORS` (`GraphReadError`, `SdmxClientError`, `UpstreamResponseError`)
      count as hard failures and identify the failing read. `TimeoutError` has a separate count for upstream
      deadline expiry. Engine checkpoints occur outside the worker and propagate.

    Materialise elects among the member's planned group alone, with or without a unit
    request. A result whose selections dispatched a facet outside that group is a plan
    mismatch: logged at WARNING, counted as failed series (`plan_mismatched_pairs`), and
    never served. A series whose rows a conversion into each entity's own currency all
    omitted is left out uncounted, since the answer's `currency_unmapped_omitted` warning
    already names its entities, and so is one whose rows a missing deflator all omitted,
    which the answer's `deflator_missing_omitted` warning names (`_every_row_omitted`).

    Under `latest` every read is prepared first, then each StatVar's reads are finalised
    at one period: the newest in the year it elected when any of its reads holds a row
    there (`PreparedPreview.elected_period`), else the newest any of them holds
    (`PreparedPreview.latest_period`), which a StatVar that elected no year serves. The
    second case, for a StatVar that elected, means coverage states a year SDMX holds no
    row in, and is logged.
    """
    pair_results: dict[_SeriesKey, _SeriesRead] = {}
    hard_failed_pairs = 0
    hard_failed_reads: set[str] = set()
    timed_out_pairs = 0
    facet_rejected_pairs = 0
    conversion_rejected_pairs = 0
    conversion_rejected_cause: str | None = None
    unfiltered_empty_pairs = 0
    plan_mismatched_pairs = 0
    prepared_reads: list[tuple[_Dispatch, PreparedPreview]] = []

    futures = [
        submit_with_context(
            pool,
            _preview_one,
            dispatch,
            unit_request=unit_request,
            materialise_sdmx=materialise_sdmx,
            materialise_graph=materialise_graph,
            deadline=deadline,
            rate_cache=rate_cache,
            coverage_index=coverage_index,
            member_facts=member_facts,
            prepare_only=dispatch.years == "latest",
        )
        for dispatch in dispatches
    ]
    for dispatch, future in zip(dispatches, futures, strict=True):
        sv, bound, count = dispatch.stat_var, dispatch.bound, len(dispatch.series)
        try:
            result = future.result()
        except FacetNotAvailableError:
            if facet_filter or unit_request is not None:
                facet_rejected_pairs += count
            else:
                unfiltered_empty_pairs += count
            continue
        except DefaultFacetUndefinedError:
            if unit_request is not None:
                facet_rejected_pairs += count
            else:
                logger.warning(
                    "observation preview read sv=%r role_tuple=%r found no "
                    "dispatchable facet axis (no bound axis and no declared "
                    "default)",
                    sv,
                    bound,
                    exc_info=True,
                )
                hard_failed_pairs += count
            continue
        except (RecipeNotDeclaredError, RateCoverageGapError) as exc:
            conversion_rejected_pairs += count
            conversion_rejected_cause = cause_from_recipe_error(exc).code
            logger.warning(
                "observation preview read sv=%r role_tuple=%r refused the "
                "requested currency conversion (cause=%r)",
                sv,
                bound,
                conversion_rejected_cause,
            )
            continue
        except NonCurrencyUnitError as exc:
            conversion_rejected_pairs += count
            conversion_rejected_cause = f"facet_binding_unsupported:{exc.unit}"
            logger.warning(
                "observation preview read sv=%r role_tuple=%r refused the "
                "requested currency conversion (unit %r is not a CurrencyUnit)",
                sv,
                bound,
                exc.unit,
            )
            continue
        except UPSTREAM_READ_ERRORS as exc:
            logger.warning(
                "observation preview read sv=%r role_tuple=%r failed",
                sv,
                bound,
                exc_info=True,
            )
            hard_failed_pairs += count
            hard_failed_reads.add(_read_named_by(exc))
            continue
        except TimeoutError:
            logger.warning(
                "observation preview read sv=%r role_tuple=%r crossed the request deadline",
                sv,
                bound,
                exc_info=True,
            )
            timed_out_pairs += count
            continue
        if dispatch.years == "latest":
            assert isinstance(result, PreparedPreview)
            if not _dispatched_selection_ids(result.selections) <= dispatch.group_ids:
                plan_mismatched_pairs += count
                logger.warning(
                    "observation preview read sv=%r role_tuple=%r dispatched facets %r outside "
                    "its planned group %r; not served",
                    sv,
                    bound,
                    sorted(_dispatched_selection_ids(result.selections)),
                    sorted(dispatch.group_ids),
                )
                continue
            prepared_reads.append((dispatch, result))
            continue
        assert isinstance(result, PreviewResult)
        if not _dispatched_facet_ids(result) <= dispatch.group_ids:
            plan_mismatched_pairs += count
            logger.warning(
                "observation preview read sv=%r role_tuple=%r dispatched facets %r outside "
                "its planned group %r; not served (%d plan mismatches in this preview)",
                sv,
                bound,
                sorted(_dispatched_facet_ids(result)),
                sorted(dispatch.group_ids),
                plan_mismatched_pairs,
            )
            continue
        pair_results.update(
            (key, read)
            for key, read in _split_series(dispatch, result).items()
            if not _every_row_omitted(read)
        )

    if prepared_reads:
        newest = latest_periods(
            (dispatch.stat_var, prepared.latest_period) for dispatch, prepared in prepared_reads
        )
        elected = latest_periods(
            (dispatch.stat_var, prepared.elected_period) for dispatch, prepared in prepared_reads
        )
        fell_back = sorted(
            {
                dispatch.stat_var
                for dispatch, _prepared in prepared_reads
                if dispatch.elected_year is not None
                and dispatch.stat_var in newest
                and dispatch.stat_var not in elected
            }
        )
        if fell_back:
            logger.warning(
                "coverage index and the observations read disagree: %r returned no row in "
                "the year coverage elects; serving their newest period",
                fell_back,
            )
        finish_futures = [
            submit_with_context(
                pool,
                _finish_preview_one,
                prepared,
                time_period=elected.get(dispatch.stat_var, newest.get(dispatch.stat_var)),
            )
            for dispatch, prepared in prepared_reads
        ]
        for (dispatch, _prepared), future in zip(prepared_reads, finish_futures, strict=True):
            count = len(dispatch.series)
            try:
                result = future.result()
            except (RecipeNotDeclaredError, RateCoverageGapError) as exc:
                conversion_rejected_pairs += count
                conversion_rejected_cause = cause_from_recipe_error(exc).code
                continue
            except NonCurrencyUnitError as exc:
                conversion_rejected_pairs += count
                conversion_rejected_cause = f"facet_binding_unsupported:{exc.unit}"
                continue
            except UPSTREAM_READ_ERRORS:
                hard_failed_pairs += count
                hard_failed_reads.add(_READ_UPSTREAM_RESPONSE)
                continue
            except TimeoutError:
                timed_out_pairs += count
                continue
            if not _dispatched_facet_ids(result) <= dispatch.group_ids:
                plan_mismatched_pairs += count
                continue
            pair_results.update(
                (key, read)
                for key, read in _split_series(dispatch, result).items()
                if not _every_row_omitted(read)
            )

    return _PairFetchOutcome(
        pair_results=pair_results,
        hard_failed_pairs=hard_failed_pairs + plan_mismatched_pairs,
        hard_failed_reads=frozenset(hard_failed_reads),
        timed_out_pairs=timed_out_pairs,
        facet_rejected_pairs=facet_rejected_pairs,
        conversion_rejected_pairs=conversion_rejected_pairs,
        conversion_rejected_cause=conversion_rejected_cause,
        unfiltered_empty_pairs=unfiltered_empty_pairs,
        plan_mismatched_pairs=plan_mismatched_pairs,
    )


def _run_pair_fetch_pool(
    series: Sequence[PlanSeries],
    sv_dcids: list[str],
    obs_entities: Mapping[str, Sequence[str]],
    *,
    years: PlanYears,
    elected: Mapping[str, int],
    unit_request: UnitRequest | None,
    materialise_graph: GraphReader,
    deadline: float | None,
    materialise_sdmx: MaterialiseSdmxClient,
    rate_cache: RateTableCache | None,
    coverage_index: CoverageIndex,
    member_facts: Mapping[str, MemberFacts] | None,
    facet_filter: Mapping[str, FacetAxisFilter] | None,
) -> _PairFetchOutcome | tuple[SpecObservations, Warning]:
    """Dispatch the planned series' reads and own their worker-pool lifecycle.

    The pool has at most `obs_preview_concurrency` workers. Check the series count against
    `obs_preview_max_pairs` before grouping them into reads (`_dispatches`). A role-bearing
    plan counted its series from coverage at planning, and a `latest` read dispatches every
    held series, which can be more. Overflow returns `OBSERVATIONS_PREVIEW_TOO_LARGE`
    without partial results. Shutdown cancels queued work and waits for active reads, whose
    permit wait is bounded by `graph_timeout_s`, so no fetch outlives this call.
    """
    if len(series) > observation_settings().obs_preview_max_pairs:
        return _series_too_large(len(series), sv_dcids, obs_entities)
    dispatches = _dispatches(series, obs_entities, years=years, elected=elected)
    with ThreadPoolExecutor(
        max_workers=max(1, min(settings().obs_preview_concurrency, len(dispatches))),
        thread_name_prefix="obs-preview",
    ) as pool:
        try:
            return _collect_pair_results(
                pool,
                dispatches,
                unit_request=unit_request,
                materialise_sdmx=materialise_sdmx,
                materialise_graph=materialise_graph,
                deadline=deadline,
                rate_cache=rate_cache,
                coverage_index=coverage_index,
                member_facts=member_facts,
                facet_filter=facet_filter,
            )
        finally:
            # Cancel queued reads and wait for active ones, even if a fetch raises.
            pool.shutdown(wait=True, cancel_futures=True)


def _no_pairs_succeeded_omission(
    requested: int,
    outcome: _PairFetchOutcome,
    *,
    facet_filter: Mapping[str, FacetAxisFilter] | None,
    elected_mode: bool,
) -> tuple[SpecObservations, Warning | None]:
    """Classify a round with no successful pair, prioritising hard failures over timeouts.

    No requested series returns `OBSERVATIONS_NOTHING_TO_PROBE` without a warning. Hard
    failures return `OBSERVATIONS_PROBE_FAILED` and name failed reads. Otherwise timeouts
    return `OBSERVATIONS_TIMED_OUT`. Active filters or elected conversions with rejected
    pairs return `OBSERVATIONS_NO_MATCHING_SOURCE`, including any conversion cause.
    Remaining empty fetches return `OBSERVATIONS_NO_POINTS` without a warning.
    """
    if not requested:
        return (
            omitted_observations(
                OBSERVATIONS_NOTHING_TO_PROBE,
                "the observation preview has nothing to fetch: no (stat_var, "
                "entity) pair confirmed a candidate to dispatch.",
            ),
            None,
        )
    if outcome.hard_failed_pairs:
        _message = _every_failing_read_message(
            set(outcome.hard_failed_reads),
            outcome.hard_failed_pairs,
            requested,
            plan_mismatched=outcome.plan_mismatched_pairs,
        )
        return (
            omitted_observations(
                OBSERVATIONS_PROBE_FAILED,
                _message,
                dispatched=requested,
                hard_failed=outcome.hard_failed_pairs,
            ),
            Warning(code=OBSERVATIONS_PROBE_FAILED, severity="warn", message=_message),
        )
    if outcome.timed_out_pairs:
        _message = (
            "the observation fetch did not finish before the request deadline; "
            "coverage stands, series were not populated."
        )
        return (
            omitted_observations(
                OBSERVATIONS_TIMED_OUT,
                _message,
                dispatched=requested,
                hard_failed=outcome.hard_failed_pairs or None,
            ),
            Warning(code=OBSERVATIONS_TIMED_OUT, severity="warn", message=_message),
        )
    if (facet_filter or elected_mode) and (
        outcome.facet_rejected_pairs or outcome.conversion_rejected_pairs
    ):
        message = "no observations available from the requested source(s)."
        if outcome.conversion_rejected_pairs:
            message += (
                f" {outcome.conversion_rejected_pairs} pair(s) refused the requested "
                f"currency conversion (cause={outcome.conversion_rejected_cause})."
            )
        return (
            omitted_observations(
                OBSERVATIONS_NO_MATCHING_SOURCE,
                message,
                dispatched=requested,
                facet_rejected=outcome.facet_rejected_pairs + outcome.conversion_rejected_pairs,
            ),
            Warning(code=OBSERVATIONS_NO_MATCHING_SOURCE, severity="warn", message=message),
        )
    return (
        omitted_observations(
            OBSERVATIONS_NO_POINTS,
            f"{requested} requested series were fetched; none carried an "
            "observation on their own selected facet.",
            dispatched=requested,
            empty=outcome.unfiltered_empty_pairs,
            hard_failed=0,
            facet_rejected=0,
        ),
        None,
    )


def _series_read(read: _SeriesRead) -> SeriesRead:
    """`read` as the assembly's `SeriesRead`: its selections' refs per axis and its
    conversion record narrowed to the series (`SeriesConversion.from_record`)."""
    return SeriesRead(
        points=read.points,
        total_rows=read.total_rows,
        facets={
            axis: selection.ref
            for axis, selection in read.selections.items()
            if selection.ref is not None
        },
        conversion=(
            SeriesConversion.from_record(read.conversion_record)
            if read.conversion_record is not None
            else None
        ),
        holds_estimates=read.holds_estimates,
    )


def _assemble_pair_series(
    outcome: _PairFetchOutcome,
    requested: int,
    sv_dcids: list[str],
    obs_entities: Mapping[str, Sequence[str]],
    *,
    member_constraints: Mapping[str, Mapping[str, str]],
    graph: EngineGraphClient,
    target_currency_ref: str | None,
    facet_filter: Mapping[str, FacetAxisFilter] | None,
    elected_mode: bool,
) -> tuple[SpecObservations, Warning | None]:
    """Build series from successful pairs (`search.observations.assemble_per_series`) and
    add any partial-result warning.

    A `GraphInfraError` from the label batch returns `OBSERVATIONS_PROBE_FAILED` rather than
    failing the resolved answer. Warning priority is hard failure, timeout, then rejected
    filter/conversion when other pairs succeeded.
    """
    pair_results = outcome.pair_results
    try:
        observations = assemble_per_series(
            {key: _series_read(read) for key, read in pair_results.items()},
            label_source=graph,
            on_names_unread=partial(note_degraded_read, "provenance names"),
            index=get_index(),
            member_constraints=member_constraints,
            target_currency_ref=target_currency_ref,
        )
    except GraphInfraError:
        logger.warning(
            "observations fetch failed for sv_dcids=%r entities=%r; returning no observations",
            sv_dcids,
            obs_entities,
            exc_info=True,
        )
        _message = "observations fetch failed; Spec.resolved.observations was not populated."
        return (
            omitted_observations(OBSERVATIONS_PROBE_FAILED, _message),
            Warning(code=OBSERVATIONS_PROBE_FAILED, severity="warn", message=_message),
        )

    if unserved := outcome.hard_failed_pairs + outcome.short_of_latest_pairs:
        # Covers upstream failures, pairs with no dispatchable facet and series a
        # `latest` read found short of their StatVar's latest period.
        warn = Warning(
            code=OBSERVATIONS_PROBE_FAILED,
            severity="warn",
            message=(
                f"{len(pair_results)} of {requested} requested series are "
                f"included; {unserved} could not be served."
            ),
        )
    elif outcome.timed_out_pairs:
        # Some pairs succeeded. Report the rest that crossed the deadline.
        warn = Warning(
            code=OBSERVATIONS_TIMED_OUT,
            severity="warn",
            message=(
                f"{len(pair_results)} of {requested} requested series are "
                f"included; {outcome.timed_out_pairs} crossed the request deadline."
            ),
        )
    elif (facet_filter or elected_mode) and (
        outcome.facet_rejected_pairs or outcome.conversion_rejected_pairs
    ):
        # Reuse the no-matching-source code for partial filter and conversion rejects.
        rejected_total = outcome.facet_rejected_pairs + outcome.conversion_rejected_pairs
        message = (
            f"{rejected_total} of {requested} requested series had "
            "no observations available from the requested source(s)."
        )
        if outcome.conversion_rejected_pairs:
            message += (
                f" {outcome.conversion_rejected_pairs} refused the requested currency "
                f"conversion (cause={outcome.conversion_rejected_cause})."
            )
        warn = Warning(
            code=OBSERVATIONS_NO_MATCHING_SOURCE,
            severity="warn",
            message=message,
        )
    else:
        warn = None
    return observations, warn


def _dispatched_series(plan: PairPlan, sv_dcids: Sequence[str]) -> list[PlanSeries]:
    """The series of `sv_dcids` a per-series read of `plan` dispatches. Every served one
    counts, and under `latest` every held one, since the year or the winner the coverage
    index predicts can be stale and preparation finds the period SDMX holds before
    conversion."""
    members = set(sv_dcids)
    return [
        s for s in (plan.held if plan.years == "latest" else plan.served) if s.stat_var in members
    ]


def _preview_mode(
    plan: ReadPlan | None,
    sv_dcids: Sequence[str],
    *,
    role_bearing: bool,
    scope: BulkScope | None,
    bound_pairs: int,
) -> PreviewMode:
    """The mode `_build_observations` previews `plan` in (`search.observations.preview_mode`).

    A role-bearing plan carries its mode. `read_plan.plan_role_read` decided it through
    `preview_mode` over the coverage index's series count, which this read does not hold, and
    recorded `ROLE_BULK` as a `BulkPlan` and `PER_SERIES` as a `PairPlan`. Any other answer
    decides here. Its places are defaulted when its plan reads a plural default scope
    (`PairPlan.scope`) and the answer has a bulk scope read (`bulk_scope_for`), and its
    series are those a per-series read dispatches (`_dispatched_series`).
    """
    if role_bearing and plan is not None:
        return PreviewMode.ROLE_BULK if isinstance(plan, BulkPlan) else PreviewMode.PER_SERIES
    return preview_mode(
        role_bearing=role_bearing,
        places_defaulted=isinstance(plan, PairPlan) and plan.scope and scope is not None,
        bound_pairs=bound_pairs,
        series_count=lambda: (
            len(_dispatched_series(plan, sv_dcids)) if isinstance(plan, PairPlan) else 0
        ),
    )


def _fetch_pair_observations(
    plan: PairPlan,
    sv_dcids: list[str],
    obs_entities: Mapping[str, Sequence[str]],
    *,
    graph: EngineGraphClient,
    materialise_graph: GraphReader,
    materialise_sdmx: MaterialiseSdmxClient,
    facet_filter: Mapping[str, FacetAxisFilter] | None,
    deadline: float | None,
    member_facts: Mapping[str, MemberFacts] | None,
    rate_cache: RateTableCache | None,
    unit_request: IdentityUnitRequest | None,
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None,
    coverage: CoverageSnapshot,
    member_constraints: Mapping[str, Mapping[str, str]],
) -> tuple[SpecObservations, Warning | None]:
    """Fetch each series `plan` serves over the plan's years, one read per member and bound
    role tuple (`_dispatches`), reduce a `latest` read's series to each StatVar's one
    period (`_at_latest_periods`), then assemble the result. A planned series that reduction
    drops is logged and counted among the series the partial-result warning says could not
    be served.

    Each `materialise_preview` call reads its series' planned candidates and elects among
    the member's group alone, over `obs_preview_max_rows` rows per series, separate from the
    bulk `role_series` row cap and the response's `obs_inline_max` display limit. A series whose candidates a
    facet filter removed entirely counts as a rejected series without a read, and a series
    with data but no dispatchable group (`PlanSeries.unservable`) as a failed series without
    a read, so the warning says no pair had a dispatchable facet. When the planned groups
    hold no point in the plan's years the preview is complete and empty.

    With both `unit_request` and `elected`, every read binds `unit` in `state="request"`
    and converts the planned group's rows. The caller narrows `plan` to the elected units
    first (`region.facets.FacetAdmission`), so a StatVar with no election keeps no
    candidate and counts as rejected series.
    """
    elected_mode = unit_request is not None and elected is not None
    translated_unit_request = None
    if unit_request is not None and elected is not None:
        # Deferred to avoid the import cycle with `convertibility`.
        from search.engine.region.convertibility import unit_request_for_materialise

        translated_unit_request = unit_request_for_materialise(unit_request)
    target_currency_ref = currency_unit_ref(
        translated_unit_request.currency if translated_unit_request is not None else None
    )

    members = set(sv_dcids)
    planned = [s for s in plan.series if s.stat_var in members]
    series = _dispatched_series(plan, sv_dcids)
    # A narrowed plan keeps a series whose every candidate the filter removed.
    filter_rejected = sum(1 for s in planned if not s.candidates)
    # A series with data and no dispatchable group fails without a read, as coverage counts it.
    undispatchable = sum(1 for s in plan.unservable if s.stat_var in members)
    if not series and not undispatchable and any(s.group for s in planned):
        # The planned groups hold no point in the plan's years: a complete, empty preview.
        return SpecObservations(series=[], total_obs=0, truncated=False), None

    pool_result = _run_pair_fetch_pool(
        series,
        sv_dcids,
        obs_entities,
        years=plan.years,
        elected=plan.elected_years or {},
        unit_request=translated_unit_request,
        materialise_graph=materialise_graph,
        deadline=deadline,
        materialise_sdmx=materialise_sdmx,
        rate_cache=rate_cache,
        coverage_index=coverage.index,
        member_facts=member_facts,
        facet_filter=facet_filter,
    )
    if not isinstance(pool_result, _PairFetchOutcome):
        return pool_result
    pair_results, short_of_latest = pool_result.pair_results, ()
    if plan.years == "latest":
        pair_results, short_of_latest = _at_latest_periods(pair_results)
        expected = {(s.stat_var, s.role_tuple) for s in plan.served if s.stat_var in members}
        # Held series that coverage already knew were older are intentionally
        # fetched to find the actual winner, but are not partial failures.
        short_of_latest = tuple(key for key in short_of_latest if key in expected)
        if short_of_latest:
            logger.warning(
                "coverage index and the observations read disagree: %d planned series "
                "held no point at their StatVar's latest period and are not served: %r",
                len(short_of_latest),
                sorted(short_of_latest),
            )
    outcome = replace(
        pool_result,
        pair_results=pair_results,
        facet_rejected_pairs=pool_result.facet_rejected_pairs + filter_rejected,
        hard_failed_pairs=pool_result.hard_failed_pairs + undispatchable,
        short_of_latest_pairs=len(short_of_latest),
    )
    requested = (
        max(
            len(series),
            len(pair_results) + len(short_of_latest),
        )
        + filter_rejected
        + undispatchable
    )

    if not outcome.pair_results:
        return _no_pairs_succeeded_omission(
            requested, outcome, facet_filter=facet_filter, elected_mode=elected_mode
        )
    return _assemble_pair_series(
        outcome,
        requested,
        sv_dcids,
        obs_entities,
        member_constraints=member_constraints,
        graph=graph,
        target_currency_ref=target_currency_ref,
        facet_filter=facet_filter,
        elected_mode=elected_mode,
    )


def _build_observations(
    plan: ReadPlan | None,
    sv_dcids: list[str],
    obs_entities: Mapping[str, Sequence[str]],
    *,
    graph: EngineGraphClient,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    facet_filter: Mapping[str, FacetAxisFilter] | None = None,
    scope: BulkScope | None = None,
    sdmx: EngineSdmxClient | None = None,
    deadline: float | None = None,
    member_facts: Mapping[str, MemberFacts] | None = None,
    rate_cache: RateTableCache | None = None,
    unit_request: IdentityUnitRequest | None = None,
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None = None,
    coverage: CoverageSnapshot,
) -> tuple[SpecObservations, Warning | None]:
    """Fetch a resolved query's observation preview from its read plan. Every path returns
    `SpecObservations`; omissions carry a reason code, which matches any accompanying
    warning.

    `plan` is resolve's read plan (`retrieval.read_plan`): the antichain-minimal members the
    preview reads, the years it reads, and per series or in bulk the facets it serves. The
    preview reads `plan.members` among `sv_dcids`, and elects nothing the plan did not. The
    preview mode rule picks the read (`_preview_mode`). A scope-defaulted plan with `scope`
    reads one facet across every place (`_bulk_scope_dispatch`); a bulk role plan reads its facet across every pair
    (`_role_bearing_bulk_dispatch`, which needs `sdmx`); a per-series plan reads each series
    it serves (`_fetch_pair_observations`, which needs the materialise clients and returns
    `OBSERVATIONS_CLIENTS_UNAVAILABLE` without a warning when they are missing). An answer
    without a plan read no facets: its preview returns `OBSERVATIONS_NOTHING_TO_PROBE`, or for
    a role-bearing product over the pair cap `OBSERVATIONS_PREVIEW_TOO_LARGE`.

    `obs_entities` maps declared roles to confirmed entity dcids; missing StatVars or
    entities return `OBSERVATIONS_NOTHING_TO_PROBE` without a warning. `facet_filter` tests
    provenance, measurement method, or unit; the plan is already narrowed to it, and it
    classifies empty reads as no matching source. `deadline` is passed to upstream reads; a
    read past it raises `TimeoutError`. `member_facts` supplies row labels and declared
    roles. `rate_cache` shares conversion tables across pair reads.

    `unit_request` and `elected` convert the plan's groups on the per-series path, whose
    plan the caller narrows to the elected units, and the read's points on both bulk paths
    (`_bulk_unit`), which narrow the plan to the elected native units when nothing needs
    converting.
    `coverage` supplies the index role-bearing materialisation reads.
    """
    if not sv_dcids or not obs_entities:
        return (
            omitted_observations(
                OBSERVATIONS_NOTHING_TO_PROBE,
                "the observation preview has nothing to fetch: no confirmed "
                "StatVar or no entity was resolved to observe.",
            ),
            None,
        )
    role_bearing = "observationAbout" not in obs_entities
    member_constraints: Mapping[str, Mapping[str, str]] = {
        sv: mf.read_constraints() for sv, mf in (member_facts or {}).items()
    }
    # A planned read names the members it reads; an answer without a plan names the members
    # its resolver elected.
    if plan is not None:
        requested = set(sv_dcids)
        sv_dcids = [sv for sv in plan.members if sv in requested]
    pairs = (
        _role_bearing_pairs(sv_dcids, obs_entities)
        if role_bearing
        else _standard_pairs(sv_dcids, obs_entities.get("observationAbout", ()))
    )
    if not pairs:
        return (
            omitted_observations(
                OBSERVATIONS_NOTHING_TO_PROBE,
                "the observation preview has nothing to fetch: no (stat_var, "
                "entity) pair could be built from the resolved members and "
                "entities.",
            ),
            None,
        )
    mode = _preview_mode(
        plan, sv_dcids, role_bearing=role_bearing, scope=scope, bound_pairs=len(pairs)
    )
    if mode is PreviewMode.SCOPE_BULK:
        assert isinstance(plan, PairPlan) and scope is not None  # `_preview_mode`
        return _bulk_scope_dispatch(
            plan,
            sv_dcids,
            obs_entities,
            scope=scope,
            graph=graph,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=rate_cache,
            facet_filter=facet_filter,
            unit_request=unit_request,
            elected=elected,
            member_constraints=member_constraints,
            deadline=deadline,
        )
    if mode is PreviewMode.ROLE_BULK:
        return _role_bearing_bulk_dispatch(
            plan,
            sv_dcids,
            obs_entities,
            pairs,
            sdmx=sdmx,
            graph=graph,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=rate_cache,
            facet_filter=facet_filter,
            unit_request=unit_request,
            elected=elected,
            member_constraints=member_constraints,
            deadline=deadline,
        )
    if not isinstance(plan, PairPlan):
        # Every resolver that reads facets plans its read; an answer without a plan read none.
        return (
            omitted_observations(
                OBSERVATIONS_NOTHING_TO_PROBE,
                "the observation preview has nothing to fetch: the answer read no facets "
                "to plan its series from.",
            ),
            None,
        )
    if materialise_graph is None or materialise_sdmx is None:
        # Missing clients are not an upstream failure. Avoid a false probe-failed warning.
        logger.info(
            "observations fetch skipped for sv_dcids=%r entities=%r: no materialise "
            "graph/sdmx client was supplied; returning no observations",
            sv_dcids,
            obs_entities,
        )
        return (
            omitted_observations(
                OBSERVATIONS_CLIENTS_UNAVAILABLE,
                "the observation preview was not fetched: no materialise "
                "graph/sdmx client was supplied.",
            ),
            None,
        )
    if mode is PreviewMode.TOO_LARGE:
        return _series_too_large(len(_dispatched_series(plan, sv_dcids)), sv_dcids, obs_entities)
    return _fetch_pair_observations(
        plan,
        sv_dcids,
        obs_entities,
        graph=graph,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        facet_filter=facet_filter,
        deadline=deadline,
        member_facts=member_facts,
        rate_cache=rate_cache,
        unit_request=unit_request,
        elected=elected,
        coverage=coverage,
        member_constraints=member_constraints,
    )
