"""Convert a bulk read's points as a per-series read and a mint convert their rows.

A bulk read serves one facet's native values across the series it dispatches: a role read
(`bulk_role_series`) at most `observation_settings().role_bulk_series_max` pairs, a scope read
(`bulk_scope`) each place's latest point. Under a unit request (`BulkConversion`) it then
converts them, one StatVar at a time, through the materialise
steps every other read takes: `plan_member_conversion` decides the StatVar's conversion
from the selections the answer's gate checked for the read's facet
(`convertibility.facet_selections`: the selection its unit decision elected for the facet's
unit, under the facet's own provenance) and loads the recipe's tables, `omit_unmapped_rows`
leaves out the rows of entities no single currency names when the request names a role's
currency, and `apply_member_conversion` converts the rest. The StatVars' conversion arcs
come in one graph read (`conversion_graph_reader`), and the tables from one request's view
of the process's rate cache (`RateTableCache.for_request`).

The answer's bound-unit gate has already checked that the plan's rows convert
(`convertibility.unit_coverage`) under the same selections, so a StatVar the conversion
refuses here means the tables changed under the request, or that its unit decision elected
no selection for the read's facet's unit. Its pairs are left out, and the read reports it
under `observations_converted_scope_unsupported` (`left_out_warning`, `left_out_omission`).
A graph, SDMX or upstream-response failure, or a crossed deadline, while the conversion
reads (`UPSTREAM_READ_ERRORS`, `TimeoutError`) propagates to the bulk read, which omits
its preview (`conversion_failed_omission`), as a per-series read does.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, cast

import pyarrow as pa

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dc_sdmx.results import VALUE_COLUMN
from materialise import FacetBinding, GraphReader, MaterialisationRequest, RequestedWindow
from materialise.columnar import observed_values
from materialise.convert import ConversionRecord
from materialise.facets import FacetSelection, UnitRequest
from materialise.member_conversion import (
    apply_member_conversion,
    conversion_graph_reader,
    omit_unmapped_rows,
    plan_member_conversion,
)
from materialise.rate_tables import RateTableCache
from materialise.recipes import NonCurrencyUnitError, RateCoverageGapError, RecipeNotDeclaredError
from materialise.series_conversion import SeriesConversion
from search.engine.clients.sdmx import RoleTuple
from search.engine.region.codes import OBSERVATIONS_BULK_FETCH_FAILED, OBSERVATIONS_TIMED_OUT
from search.models import SpecObservations, Warning, omitted_observations
from search.observations import OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED, currency_unit_ref

logger = logging.getLogger(__name__)

type Pair = tuple[str, RoleTuple]
type Points = list[tuple[str, float | None]]

_TIME_PERIOD = "TIME_PERIOD"
_UNIT = "unit"
_FACET_ID = "facetId"
_REFUSALS = (NonCurrencyUnitError, RateCoverageGapError, RecipeNotDeclaredError)
_FACET_NOT_ELECTED = "facet_not_elected"


class ServedFacet(Protocol):
    """The facet a bulk read serves: a role read's `RoleBulkCandidate` or a scope read's
    graph `Facet`. Its unit picks the StatVar's elected selection, and its provenance is
    the one the rows convert under (`convertibility.facet_selections`)."""

    @property
    def facet_id(self) -> str | None: ...
    @property
    def provenance_id(self) -> str | None: ...
    @property
    def unit(self) -> str | None: ...


@dataclass(frozen=True)
class BulkConversion:
    """A unit request a bulk read converts its points to: the request in materialise's
    form, each StatVar's elected selections (`convertibility.decide_part_unit`), and the
    clients and rate cache the conversion reads through, under `deadline`."""

    unit_request: UnitRequest
    elected: Mapping[str, tuple[Mapping[str, FacetSelection], ...]]
    graph: GraphReader
    sdmx: MaterialiseSdmxClient
    rate_cache: RateTableCache
    deadline: float | None = None

    @property
    def target_unit_ref(self) -> str | None:
        """The `CurrencyUnit` every converted series is in, None for a role's currency,
        which differs per series."""
        return currency_unit_ref(self.unit_request.currency)


@dataclass(frozen=True)
class ConvertedPairs:
    """The pairs' converted points, each with its StatVar's conversion record narrowed to
    the fields a series reports (`SeriesConversion.from_record`). A pair whose rows the
    conversion left out is absent from both. `left_out` maps each StatVar left out whole to
    its cause: `facet_not_elected` when its unit decision elected no selection for the
    read's facet's unit, else the refusal's code and cause (`rate_coverage_gap: rate_missing`)."""

    points: dict[Pair, Points]
    conversions: dict[Pair, SeriesConversion]
    left_out: dict[str, str]


def convert_pairs(
    pair_points: Mapping[Pair, Points],
    *,
    facet: ServedFacet,
    conversion: BulkConversion,
) -> ConvertedPairs:
    """Convert `pair_points`, read from `facet`, to `conversion`'s unit, one StatVar at a
    time, under the selections the answer's gate checked (`convertibility.facet_selections`).
    A StatVar with no elected selection for `facet`'s unit, or whose conversion refuses, is
    logged and left out, with its cause in `left_out`. A read failure raises one of
    `UPSTREAM_READ_ERRORS` or `TimeoutError`."""
    # Deferred to avoid the import cycle with `convertibility`.
    from search.engine.region.convertibility import facet_selections

    by_stat_var: dict[str, list[Pair]] = {}
    for pair in pair_points:
        by_stat_var.setdefault(pair[0], []).append(pair)
    graph = conversion_graph_reader(conversion.graph, stat_vars=list(by_stat_var))
    # One request's view of the cache, so every StatVar converts against the same tables.
    conversion = replace(conversion, rate_cache=conversion.rate_cache.for_request())
    points: dict[Pair, Points] = {}
    conversions: dict[Pair, SeriesConversion] = {}
    left_out: dict[str, str] = {}
    for stat_var, pairs in by_stat_var.items():
        selections = facet_selections(
            conversion.elected.get(stat_var, ()), (facet.unit, facet.provenance_id)
        )
        if selections is None:
            logger.warning(
                "bulk preview of %s left out: no unit decision elected unit %s of facet %s",
                stat_var,
                facet.unit,
                facet.facet_id,
            )
            left_out[stat_var] = _FACET_NOT_ELECTED
            continue
        try:
            converted, record = _convert_member(
                stat_var, pairs, pair_points, selections, conversion=conversion, graph=graph
            )
        except _REFUSALS as exc:
            logger.warning(
                "bulk preview of %s left out: its conversion refused after the unit gate passed",
                stat_var,
                exc_info=True,
            )
            cause = getattr(exc, "cause", None)
            left_out[stat_var] = f"{exc.code}: {cause}" if cause else exc.code
            continue
        points.update(converted)
        if record is not None:
            conversions.update(dict.fromkeys(converted, SeriesConversion.from_record(record)))
    return ConvertedPairs(points=points, conversions=conversions, left_out=left_out)


def _left_out_text(left_out: Mapping[str, str]) -> str:
    """The StatVars `left_out` names, each with its cause, in dcid order."""
    return ", ".join(f"{sv} ({cause})" for sv, cause in sorted(left_out.items()))


def left_out_warning(left_out: Mapping[str, str], also: Warning | None = None) -> Warning | None:
    """The warning a bulk read that converted some StatVars returns: one naming the StatVars
    the conversion left out (`ConvertedPairs.left_out`), whose message carries `also`'s
    after its own, since a read returns one warning. `also` alone when nothing was left out."""
    if not left_out:
        return also
    message = (
        f"the preview leaves out {len(left_out)} measure(s) whose points did not convert to "
        f"the unit asked: {_left_out_text(left_out)}."
    )
    if also is not None:
        message = f"{message} Also: {also.message}"
    return Warning(code=OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED, severity="warn", message=message)


def left_out_omission(left_out: Mapping[str, str]) -> tuple[SpecObservations, Warning]:
    """The omitted preview, and its matching warning, of a bulk read whose conversion left
    out every StatVar that held points (`ConvertedPairs.left_out`, non-empty)."""
    message = (
        "the observation preview shows no series: no measure's points converted to the unit "
        f"asked: {_left_out_text(left_out)}."
    )
    return (
        omitted_observations(OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED, message),
        Warning(code=OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED, severity="warn", message=message),
    )


def conversion_failed_omission(exc: Exception) -> tuple[SpecObservations, Warning]:
    """The omitted preview, and its matching warning, of a bulk read whose conversion
    raised `exc`, one of `UPSTREAM_READ_ERRORS` or `TimeoutError`: `observations_timed_out`
    for the deadline, else `observations_bulk_fetch_failed`, the codes the bulk read's own
    fetch failures take. Coverage stands."""
    logger.warning("bulk preview conversion failed; delivering no observations", exc_info=exc)
    if isinstance(exc, TimeoutError):
        message = (
            "the conversion of the bulk observation preview did not finish before the "
            "request deadline; coverage stands, series were not populated."
        )
        return (
            omitted_observations(OBSERVATIONS_TIMED_OUT, message),
            Warning(code=OBSERVATIONS_TIMED_OUT, severity="warn", message=message),
        )
    message = (
        "a read the conversion of the bulk observation preview needs failed; coverage "
        "stands, series were not populated."
    )
    return (
        omitted_observations(OBSERVATIONS_BULK_FETCH_FAILED, message),
        Warning(code=OBSERVATIONS_BULK_FETCH_FAILED, severity="warn", message=message),
    )


def _convert_member(
    stat_var: str,
    pairs: Sequence[Pair],
    pair_points: Mapping[Pair, Points],
    selections: Mapping[str, FacetSelection],
    *,
    conversion: BulkConversion,
    graph: GraphReader,
) -> tuple[dict[Pair, Points], ConversionRecord | None]:
    """One StatVar's pairs converted, with the conversion's record."""
    # A bulk pair binds every role its StatVar declares (`bulk_role_series`).
    declared_roles = tuple(role for role, _dcid in pairs[0][1])
    request = MaterialisationRequest(
        stat_var=f"dcid:{stat_var}",
        role_bindings={},
        declared_roles=declared_roles,
        facet_bindings={"unit": FacetBinding(state="request", request=conversion.unit_request)},
        # The points are read; the decision reads no window.
        window=RequestedWindow(kind="all"),
    )
    plan = plan_member_conversion(
        request=request,
        selections=selections,
        graph=graph,
        sdmx=conversion.sdmx,
        rate_cache=conversion.rate_cache,
        deadline=conversion.deadline,
    )
    table = _rows(pairs, pair_points, declared_roles=declared_roles, selections=selections)
    kept, currencies = omit_unmapped_rows(plan, table, served=table)
    converted, record = apply_member_conversion(
        plan, kept, declared_roles=declared_roles, currencies=currencies
    )
    pair_of = {tuple(dict(pair[1])[role] for role in declared_roles): pair for pair in pairs}
    entities = zip(*(converted[role].to_pylist() for role in declared_roles), strict=True)
    values = observed_values(converted[VALUE_COLUMN]).to_pylist()
    # `_rows` writes a period on every row.
    dates = cast("list[str]", converted[_TIME_PERIOD].to_pylist())
    out: dict[Pair, Points] = {}
    for bound, date, value in zip(entities, dates, values, strict=True):
        out.setdefault(pair_of[bound], []).append((date, value))
    return out, record


def _rows(
    pairs: Sequence[Pair],
    pair_points: Mapping[Pair, Points],
    *,
    declared_roles: tuple[str, ...],
    selections: Mapping[str, FacetSelection],
) -> pa.Table:
    """The pairs' points as the observation rows materialise converts: a column per
    declared role, the period, the value as SDMX states it, the native unit, and the
    facet."""
    columns: dict[str, list[str | None]] = {role: [] for role in declared_roles}
    periods: list[str] = []
    values: list[str] = []
    for pair in pairs:
        bound = dict(pair[1])
        for date, value in pair_points[pair]:
            for role in declared_roles:
                columns[role].append(bound[role])
            periods.append(date)
            values.append("" if value is None else repr(value))
    unit = selections["unit"].ref
    facet_id = selections["unit"].dispatch_facet_id
    return pa.table(
        {
            **{role: pa.array(column, type=pa.string()) for role, column in columns.items()},
            _TIME_PERIOD: pa.array(periods, type=pa.string()),
            VALUE_COLUMN: pa.array(values, type=pa.string()),
            _UNIT: pa.array([unit] * len(periods), type=pa.string()),
            _FACET_ID: pa.array([facet_id] * len(periods), type=pa.string()),
        }
    )
