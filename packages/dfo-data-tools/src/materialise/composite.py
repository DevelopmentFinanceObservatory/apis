"""Composes a composite definition's component materialisations into one frame.

A composite provenance (`code_index.types.CompositeDeclaration`) names the series it is
served from. Each component materialises through `pipeline.materialise_group` alone, and
`compose_components` makes one result of them:

- A stacked composite serves the years every component's table covers in the coverage
  snapshot (`table_years`, `stacked_years`), inside the requested window, whatever rows this
  read returns. Rows of other years are dropped, and each component's summary names the
  years it lost. Search reads the same years (`search.scope.composite`). A spliced composite
  serves each component's rows in its own declared years.
- Inside those years, each member of a component whose absence reads as zero
  (`ComponentRecipe.zero_absent`) gets a row of value 0 in every cell, a role tuple and
  period, that another component fills and it does not. Outside them no zero row is
  written, so a year a component does not cover is missing. A zero row states the unit its
  cell's rows state, its own component's where that component holds the cell, and the
  member's measurement method, so a cell's rows share one unit when the components'
  conversions agree on it. A zero-filling component holding no row at all copies another
  component's row, with the one method its facet selection names, or none. Each component is converted before it is composed, and a
  converted zero is a zero in its cell's unit. A component without
  the flag contributes nothing where it holds no row, and neither does a member its facet
  group election left out (`MaterialisationResult.unserved_members`), whose series sit under
  another source, unit or method.
- Every row carries the composite provenance. `stat_var` keeps naming the component's member,
  so the rows of two components never share a grain and no row is a total.

The frame keeps `row_sort_key` order by construction. Every row of one `stat_var` comes from
one component, and `stat_var` leads the sort key (`dfo_frame_grammar.grammar.derive_columns`),
so the composed frame is each component's rows of one member at a time, merged across
components by `stat_var` alone. It is composed once: the pass that hashes it also counts it,
folds its series statistics and output profile, and writes it as Parquet, one row group per
member, which every later pass reads (`frame_build.ParquetFrameChunks`).
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.compute as pc
from dfo_frame_grammar import (
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    VALUE_COLUMN,
    Frame,
    stream_frame_sha256,
)
from dfo_frame_grammar.arrow import build_frame_from_arrow

from code_index.members import SourceNotInFamily, identity_family_query, index_family
from code_index.types import ComponentRecipe, ComponentRole, CompositeDeclaration
from materialise import pipeline
from materialise.codes import MaterialisationWarning
from materialise.content_version import mint_version_from_digest
from materialise.continued_series import ContinuedSeries, continued_series_of_trace
from materialise.dependencies import Dependencies, members_digest
from materialise.enrichment import EnrichmentLookup
from materialise.errors import MaterialisationUsageError
from materialise.finish import (
    _clamped_years,
    _conversion_trace,
    _coverage_below_scope_warning,
    _plain_role_bindings,
    _warning,
    currency_unmapped_omitted_warning,
    deflator_missing_omitted_warning,
)
from materialise.frame_build import ParquetFrameChunks, ParquetFrameSink, build_frame
from materialise.profile import OutputProfileFold
from materialise.statistics import SeriesStatisticsSink, compute_series_statistics_table
from materialise.trace import ManifestTrace
from materialise.window import AppliedWindow, Coverage, RequestedWindow, compute_applied_window

if TYPE_CHECKING:
    from code_index.query import CodeIndex
    from materialise.pipeline import MaterialisationRequest, MaterialisationResult
    from series_coverage.index import CoverageIndex

_PROVENANCE_AXIS = "provenance"
PROVENANCE_COLUMN = "provenance"
_UNIT_AXIS = "unit"
_METHOD_AXIS = "measurementMethod"
_UNIT_COLUMN = "unit"
_METHOD_COLUMN = "measurementMethod"
_YEARS_LABEL = "years"
# The codes `compose_components` states anew over the composed frame. Every other component
# warning passes through once per distinct code and message.
_RESTATED_CODES = frozenset(
    (
        "empty_result",
        "coverage_below_scope",
        "currency_unmapped_omitted",
        "deflator_missing_omitted",
    )
)

_EMPTY_RUN = pa.table({STAT_VAR_COLUMN: pa.array([], type=pa.string())})
"""A member run with no row, which `_composed_chunks` skips."""

type _Cell = tuple[tuple[str, ...], str]
"""A row's role values and period."""


@dataclass(frozen=True)
class ComponentResult:
    """One component's materialisation: its role and recipe, the requests it read, the
    result `pipeline.materialise_group` returned for them, and the years its table covers
    (`table_years`)."""

    role: ComponentRole
    recipe: ComponentRecipe
    requests: Sequence[MaterialisationRequest]
    result: MaterialisationResult
    covered_years: frozenset[int]


def table_years(
    index: CodeIndex, coverage: CoverageIndex, identity: Mapping[str, object]
) -> frozenset[int]:
    """The years the table a component `identity` reads covers: those any series of its
    index family (`code_index.members.index_family`) holds in `coverage`, under any place.
    Empty for a family the index does not decide, or one its named dataset does not
    publish."""
    try:
        family = index_family(index, identity_family_query(index, identity))
    except SourceNotInFamily:
        return frozenset()
    if family is None or not family.rows:
        return frozenset()
    counts = coverage.series_by_year(
        variables=[row.dcid for row in family.rows],
        first_year=coverage.horizon.epoch_year,
        last_year=coverage.horizon.max_year,
    )
    return frozenset(year for year, count in counts.items() if count)


def stacked_years(covered: Iterable[frozenset[int]]) -> frozenset[int]:
    """The years a stacked composite serves: those every component's table covers, each
    component's `covered` (`table_years`)."""
    return frozenset.intersection(*covered)


def rows_in_years(frame_chunks: Callable[[], Iterator[Frame]], years: frozenset[int]) -> int:
    """How many rows of `frame_chunks` fall in `years`."""
    return sum(sum(1 for year in _years(frame.table) if year in years) for frame in frame_chunks())


@dataclass(frozen=True)
class ComponentSummary:
    """What one component contributed to a composed frame: the provenance it was served from,
    its own content version, the years it serves in the frame as closed spans, the years
    dropped from it outside the composite's years, its members, its rows, the zero rows
    written for it, the codes of its own warnings, and its facet selections."""

    role: ComponentRole
    provenance: str | None
    content_version: str
    periods: tuple[tuple[int, int], ...]
    dropped_years: tuple[int, ...]
    stat_vars: tuple[str, ...]
    rows: int
    zero_rows: int
    warnings: tuple[str, ...]
    facets: tuple[Mapping[str, object], ...]

    def as_trace_entry(self) -> dict[str, object]:
        return {
            "role": self.role,
            "provenance": self.provenance,
            "content_version": self.content_version,
            "periods": [list(span) for span in self.periods],
            "dropped_years": list(self.dropped_years),
            "stat_vars": list(self.stat_vars),
            "rows": self.rows,
            "zero_rows": self.zero_rows,
            "warnings": list(self.warnings),
            "facets": [dict(facet) for facet in self.facets],
        }


@dataclass
class _Plan:
    """What the first pass over the components settles: the years each component's rows are
    kept in, the cells of those years, each zero-filling member's own cells and the row its
    zero rows copy their measurement method from, and the unit of each covered cell's zero
    rows for each zero-filling component (`_zero_units`)."""

    kept_years: dict[ComponentRole, frozenset[int]]
    held_years: dict[ComponentRole, frozenset[int]]
    covered_cells: frozenset[_Cell]
    member_cells: dict[str, set[_Cell]] = field(default_factory=dict)
    templates: dict[str, pa.Table] = field(default_factory=dict)
    zero_members: dict[ComponentRole, tuple[str, ...]] = field(default_factory=dict)
    zero_units: dict[ComponentRole, dict[_Cell, object]] = field(default_factory=dict)


@dataclass
class _Fold:
    """What one pass over the composed chunks counts."""

    rows: int = 0
    by_stat_var: dict[str, int] = field(default_factory=dict)
    zero_rows: dict[ComponentRole, int] = field(default_factory=dict)
    years: dict[ComponentRole, set[int]] = field(default_factory=dict)
    served: dict[ComponentRole, dict[str, set[str]]] = field(default_factory=dict)
    role_values: dict[str, set[str]] = field(default_factory=dict)
    all_years: set[int] = field(default_factory=set)


def compose_components(
    components: Sequence[ComponentResult],
    *,
    composite: CompositeDeclaration,
    window: RequestedWindow,
    role_columns: Sequence[str],
    enrichment: EnrichmentLookup,
) -> MaterialisationResult:
    """One result composed from `components`, each materialised under its own role, as the
    module docstring states. `window` is the composite's requested window and `role_columns`
    the role columns every component's frame holds.

    The frame's hash and `content_version` come from the composed chunks, in the one pass
    that composes them. The trace states
    the composite's provenance as its one provenance selection, keeps each component's
    selections in its summary (`ManifestTrace.components`), and lists every member served. A
    warning code is stated once: `coverage_below_scope` counts the requested entities any
    component leaves out of the composed frame and names, per component, how many it leaves
    out; `empty_result` stands for a composed frame with no row.

    Raises `MaterialisationUsageError` for no components, two components sharing a member,
    components with different role columns, a zero-filling member that needs zero rows and
    whose method neither its own rows nor its component's one method states, or a cell
    needing zero rows whose rows state more than one unit (`_zero_units`)."""
    if not components:
        raise MaterialisationUsageError("compose_components requires at least one component")
    role_columns = tuple(role_columns)
    _check_components(components, role_columns=role_columns)
    plan = _plan(components, composite=composite)
    fold = _Fold()
    owner = {
        request.stat_var: component.role
        for component in components
        for request in component.requests
    }
    statistics = SeriesStatisticsSink(continuing=_continuations(components, composite=composite))
    profile = OutputProfileFold(role_columns=role_columns, enrichment=enrichment)
    sink = ParquetFrameSink()
    columns: list[tuple[str, ...]] = []

    def counted(chunks: Iterator[Frame]) -> Iterator[Frame]:
        for frame in chunks:
            _fold(fold, frame, plan=plan, owner=owner, role_columns=role_columns)
            statistics.write(compute_series_statistics_table(frame))
            profile.add(frame)
            sink.write(frame.table)
            columns[:] = [frame.columns]
            yield frame

    frame_digest = stream_frame_sha256(
        counted(
            _composed_chunks(components, plan=plan, composite=composite, role_columns=role_columns)
        )
    )
    frame_chunks = ParquetFrameChunks(sink.finish(), columns=columns[0])
    output_profile = profile.finish(frame_chunks)
    served_bounds = (min(fold.all_years), max(fold.all_years)) if fold.all_years else None
    applied = compute_applied_window(window, served_bounds)
    coverage = Coverage(
        rows=fold.rows,
        dimensions=(
            *(
                {"label": role, "count": len(fold.role_values.get(role, set()))}
                for role in role_columns
            ),
            {"label": _YEARS_LABEL, "count": len(fold.all_years)},
        ),
        window=applied.as_trace_entry() if applied is not None else None,
        entity_refs={
            **{role: frozenset(fold.role_values.get(role, set())) for role in role_columns},
            _YEARS_LABEL: frozenset(fold.all_years),
        },
    )
    summaries = tuple(_summary(component, plan=plan, fold=fold) for component in components)
    first = components[0].result
    traces = [component.result.trace for component in components]
    trace = ManifestTrace(
        applied_window=coverage.window,
        graph_revision=first.trace.graph_revision,
        source_revisions=_union_entries(trace.source_revisions for trace in traces),
        resolved_stat_vars=tuple(
            {**entry, "observation_count": fold.by_stat_var.get(cast(str, entry["ref"]), 0)}
            for entry in sorted(
                (entry for trace in traces for entry in trace.resolved_stat_vars),
                key=lambda entry: cast(str, entry["ref"]),
            )
        ),
        facets=_composite_facets(traces, composite=composite),
        normalisations=(),
        conversion=_conversion_trace(
            (stat_var, record)
            for component in components
            for stat_var, record in component.result.member_conversion_records.items()
            if record is not None
        ),
        frame_grammar_version=first.trace.frame_grammar_version,
        materialiser_build=first.trace.materialiser_build,
        capability_version=first.trace.capability_version,
        family=None,
        stub_candidates=tuple(
            dict.fromkeys(ref for trace in traces for ref in trace.stub_candidates)
        ),
        coverage_gaps=_union_entries(trace.coverage_gaps for trace in traces),
        choices=_union_entries(trace.choices for trace in traces),
        continuations=_union_entries(trace.continuations for trace in traces),
        components=tuple(summary.as_trace_entry() for summary in summaries),
    )
    all_requests = [request for component in components for request in component.requests]
    return pipeline.MaterialisationResult(
        frame_chunks=frame_chunks,
        frame_sha256=frame_digest,
        content_version=mint_version_from_digest(frame_digest),
        trace=trace,
        coverage=coverage,
        warnings=_warnings(components, fold=fold, window=window, applied=applied),
        series_statistics=statistics.finish(),
        output_profile=output_profile,
        conversion_record=None,
        member_conversion_records={
            stat_var: record
            for component in components
            for stat_var, record in component.result.member_conversion_records.items()
        },
        dependencies=_dependencies(components, requests=all_requests),
        default_scope=tuple(
            dict.fromkeys(
                place for component in components for place in component.result.default_scope
            )
        ),
        series_conversions={
            key: conversion
            for component in components
            for key, conversion in component.result.series_conversions.items()
        },
        probe_empty=all(component.result.probe_empty for component in components),
        countries_fallback=fold.rows > 0
        and any(component.result.countries_fallback for component in components),
    )


def _check_components(
    components: Sequence[ComponentResult], *, role_columns: tuple[str, ...]
) -> None:
    """Refuse components sharing a member or a component whose requests declare other roles."""
    owners: dict[str, ComponentRole] = {}
    for component in components:
        for request in component.requests:
            if request.declared_roles != role_columns:
                raise MaterialisationUsageError(
                    f"component {component.role!r} declares roles {request.declared_roles!r}, "
                    f"not the composite's {role_columns!r}"
                )
            holder = owners.setdefault(request.stat_var, component.role)
            if holder != component.role:
                raise MaterialisationUsageError(
                    f"components {holder!r} and {component.role!r} both serve "
                    f"{request.stat_var!r}; a composite's components serve disjoint members"
                )


def _plan(components: Sequence[ComponentResult], *, composite: CompositeDeclaration) -> _Plan:
    """The first pass: each component's years, the composite's years, the cells of those
    years, each zero-filling member's cells and template row, and the units of its zero
    rows."""
    held: dict[ComponentRole, set[int]] = {component.role: set() for component in components}
    cells_by_year: dict[int, set[_Cell]] = {}
    cell_units: dict[_Cell, dict[ComponentRole, list[object]]] = {}
    member_cells: dict[str, set[_Cell]] = {}
    templates: dict[str, pa.Table] = {}
    component_templates: dict[ComponentRole, dict[object, pa.Table]] = {}
    sample: pa.Table | None = None
    roles = _role_columns(components)
    for component in components:
        zero_fill = component.recipe.zero_absent
        for frame in component.result.frame_chunks():
            table = frame.table
            if table.num_rows == 0:
                continue
            if sample is None:
                sample = table.take([0])
            years = _years(table)
            role_values = [_strings(table, role) for role in roles]
            periods = _strings(table, TIME_PERIOD_COLUMN)
            units = table[_UNIT_COLUMN].to_pylist()
            stat_vars = _strings(table, STAT_VAR_COLUMN) if zero_fill else ()
            for position, (year, period) in enumerate(zip(years, periods, strict=True)):
                held[component.role].add(year)
                cell = (tuple(column[position] for column in role_values), period)
                cells_by_year.setdefault(year, set()).add(cell)
                stated = cell_units.setdefault(cell, {}).setdefault(component.role, [])
                if units[position] not in stated:
                    stated.append(units[position])
                if zero_fill:
                    member = stat_vars[position]
                    member_cells.setdefault(member, set()).add(cell)
            if zero_fill:
                _note_templates(
                    table, templates, component_templates.setdefault(component.role, {})
                )
    if composite.kind == "stacked":
        shared = stacked_years(component.covered_years for component in components)
        kept = {component.role: shared for component in components}
    else:
        kept = {
            component.role: frozenset(
                year
                for year in held[component.role]
                if component.recipe.periods is None or component.recipe.periods.holds(year)
            )
            for component in components
        }
    covered = frozenset(
        cell
        for year, cells in cells_by_year.items()
        if any(year in years for years in kept.values())
        for cell in cells
    )
    zero_members: dict[ComponentRole, tuple[str, ...]] = {}
    for component in components:
        if not component.recipe.zero_absent:
            continue
        left_out = frozenset(component.result.unserved_members)
        members = tuple(
            sorted(
                request.stat_var
                for request in component.requests
                if request.stat_var not in left_out
            )
        )
        zero_members[component.role] = members
        by_method = component_templates.get(component.role, {})
        if not by_method and sample is not None:
            by_method = {None: _borrowed_template(component, sample)}
        for member in members:
            if member in templates or not covered:
                continue
            if len(by_method) != 1:
                raise MaterialisationUsageError(
                    f"component {component.role!r} member {member!r} holds no row, and its "
                    f"rows state {len(by_method)} measurement methods, so its zero rows have "
                    "no method"
                )
            templates[member] = next(iter(by_method.values()))
    return _Plan(
        kept_years=kept,
        held_years={role: frozenset(years) for role, years in held.items()},
        covered_cells=covered,
        member_cells=member_cells,
        templates=templates,
        zero_members=zero_members,
        zero_units={
            role: _zero_units(
                role,
                _zero_cells(covered, members, member_cells=member_cells),
                cell_units,
            )
            for role, members in zero_members.items()
        },
    )


def _borrowed_template(component: ComponentResult, row: pa.Table) -> pa.Table:
    """`row`, another component's, as the template of zero-filling `component`'s zero rows
    when it holds no row: its measurement method is the one `component`'s facet selection
    names, or none."""
    methods = {
        facet.get("ref")
        for facet in component.result.trace.facets
        if facet.get("axis") == _METHOD_AXIS
    }
    method = methods.pop() if len(methods) == 1 else None
    position = row.schema.get_field_index(_METHOD_COLUMN)
    return row.set_column(
        position, _METHOD_COLUMN, pa.array([method], type=row.schema.field(position).type)
    )


def _zero_cells(
    covered: frozenset[_Cell], members: Sequence[str], *, member_cells: Mapping[str, set[_Cell]]
) -> set[_Cell]:
    """The cells of `covered` some one of `members` holds no row in, so a zero row fills."""
    holders: dict[_Cell, int] = {}
    for member in members:
        for cell in member_cells.get(member, ()):
            holders[cell] = holders.get(cell, 0) + 1
    return {cell for cell in covered if holders.get(cell, 0) < len(members)}


def _zero_units(
    role: ComponentRole,
    cells: Iterable[_Cell],
    cell_units: Mapping[_Cell, Mapping[ComponentRole, Sequence[object]]],
) -> dict[_Cell, object]:
    """The unit of component `role`'s zero rows in each of `cells`: the unit its own rows
    state in the cell, else the unit every other component's rows state there. Every covered
    cell holds a row of some component, and each component is converted to the unit its
    request settles for the cell before it is composed, so a zero converts to that unit.
    Raises `MaterialisationUsageError` for a cell whose rows state more than one unit."""
    units: dict[_Cell, object] = {}
    for cell in cells:
        by_role = cell_units[cell]
        stated = by_role.get(role)
        if stated is None:
            stated = []
            for other in by_role.values():
                stated.extend(unit for unit in other if unit not in stated)
        if len(stated) != 1:
            raise MaterialisationUsageError(
                f"the cell {cell!r} states {len(stated)} units, so component {role!r}'s zero "
                "rows there have no unit"
            )
        units[cell] = stated[0]
    return units


def _note_templates(
    table: pa.Table,
    templates: dict[str, pa.Table],
    by_method: dict[object, pa.Table],
) -> None:
    """Record each member's last row in `table` as its template, and a row of each distinct
    measurement method of the component."""
    stat_vars = _strings(table, STAT_VAR_COLUMN)
    methods = table[_METHOD_COLUMN].to_pylist()
    for position, member in enumerate(stat_vars):
        is_last = position + 1 == len(stat_vars) or stat_vars[position + 1] != member
        if is_last:
            templates[member] = table.take([position])
        if methods[position] not in by_method:
            by_method[methods[position]] = table.take([position])


def _composed_chunks(
    components: Sequence[ComponentResult],
    *,
    plan: _Plan,
    composite: CompositeDeclaration,
    role_columns: tuple[str, ...],
) -> Iterator[Frame]:
    """The composed frame, one chunk per member, in `stat_var` order. Yields one empty chunk
    when no row is served."""
    streams = [
        _component_runs(component, plan=plan, role_columns=role_columns) for component in components
    ]
    produced = False
    for _stat_var, table in heapq.merge(*streams, key=lambda run: run[0]):
        if table.num_rows == 0:
            continue
        table = table.set_column(
            table.schema.get_field_index(PROVENANCE_COLUMN),
            PROVENANCE_COLUMN,
            pa.array([composite.provenance] * table.num_rows, type=pa.string()),
        )
        table = table.drop_columns([STATUS_COLUMN])
        produced = True
        yield build_frame_from_arrow(table, role_keys=role_columns)
    if not produced:
        yield build_frame(role_keys=role_columns, rows=())


def _component_runs(
    component: ComponentResult, *, plan: _Plan, role_columns: tuple[str, ...]
) -> Iterator[tuple[str, pa.Table]]:
    """`component`'s rows in its kept years, one member at a time in `stat_var` order, each
    member of a zero-filling component with its zero rows, including members holding no row."""
    kept = plan.kept_years[component.role]
    runs = _member_runs(component.result.frame_chunks, kept=kept)
    if not component.recipe.zero_absent:
        yield from runs
        return
    members = plan.zero_members[component.role]
    real = dict.fromkeys(members, False)
    for stat_var, table in heapq.merge(
        runs, ((member, None) for member in members), key=lambda run: run[0]
    ):
        if table is None:
            if not real.get(stat_var, False):
                yield (
                    stat_var,
                    _zero_rows(
                        stat_var,
                        None,
                        plan=plan,
                        units=plan.zero_units[component.role],
                        role_columns=role_columns,
                    ),
                )
            continue
        real[stat_var] = True
        yield (
            stat_var,
            _zero_rows(
                stat_var,
                table,
                plan=plan,
                units=plan.zero_units[component.role],
                role_columns=role_columns,
            ),
        )


def _member_runs(
    frame_chunks: Callable[[], Iterator[Frame]], *, kept: frozenset[int]
) -> Iterator[tuple[str, pa.Table]]:
    """Each member's rows in `kept` years, one table per member in `stat_var` order. A
    member's rows may span chunks, and they are joined before it is yielded."""
    pending: str | None = None
    parts: list[pa.Table] = []
    for frame in frame_chunks():
        table = frame.table
        if table.num_rows == 0:
            continue
        table = table.filter(pa.array([year in kept for year in _years(table)], type=pa.bool_()))
        for stat_var, part in _stat_var_slices(table):
            if stat_var != pending and parts:
                yield cast(str, pending), pa.concat_tables(parts)
                parts = []
            pending = stat_var
            parts.append(part)
    if parts:
        yield cast(str, pending), pa.concat_tables(parts)


def _stat_var_slices(table: pa.Table) -> Iterator[tuple[str, pa.Table]]:
    """`table`, sorted by `stat_var`, sliced into one table per `stat_var`."""
    if table.num_rows == 0:
        return
    offset = 0
    for stat_var, run in itertools.groupby(_strings(table, STAT_VAR_COLUMN)):
        length = sum(1 for _ in run)
        yield stat_var, table.slice(offset, length)
        offset += length


def _zero_rows(
    stat_var: str,
    table: pa.Table | None,
    *,
    plan: _Plan,
    units: Mapping[_Cell, object],
    role_columns: tuple[str, ...],
) -> pa.Table:
    """`table`, the member's rows in its kept years, with a zero row for every covered cell
    the member holds no row in, in row sort order. Each zero row states its cell's unit in
    `units` and copies the member's template row's method. Zero cells differ from the
    member's own cells in their roles or period, so a stable sort on those keeps the order
    of the member's own rows."""
    held = plan.member_cells.get(stat_var, set())
    missing = sorted(plan.covered_cells - held)
    if not missing:
        return table if table is not None else _EMPTY_RUN
    template = plan.templates[stat_var]
    zeros = template.take([0] * len(missing))
    zeros = zeros.set_column(
        zeros.schema.get_field_index(STAT_VAR_COLUMN),
        STAT_VAR_COLUMN,
        pa.array([stat_var] * len(missing), type=pa.string()),
    )
    for position, role in enumerate(role_columns):
        zeros = zeros.set_column(
            zeros.schema.get_field_index(role),
            role,
            pa.array([cell[0][position] for cell in missing], type=pa.string()),
        )
    zeros = zeros.set_column(
        zeros.schema.get_field_index(TIME_PERIOD_COLUMN),
        TIME_PERIOD_COLUMN,
        pa.array([cell[1] for cell in missing], type=pa.string()),
    )
    zeros = zeros.set_column(
        zeros.schema.get_field_index(VALUE_COLUMN),
        VALUE_COLUMN,
        pa.array([0.0] * len(missing), type=pa.float64()),
    )
    unit_index = zeros.schema.get_field_index(_UNIT_COLUMN)
    zeros = zeros.set_column(
        unit_index,
        _UNIT_COLUMN,
        pa.array([units[cell] for cell in missing], type=zeros.schema.field(unit_index).type),
    )
    if table is None:
        return zeros
    joined = pa.concat_tables([table, zeros.cast(table.schema)])
    order = pc.sort_indices(
        joined,
        sort_keys=[(column, "ascending") for column in (*role_columns, TIME_PERIOD_COLUMN)],
        null_placement="at_start",
    )
    return joined.take(order)


def _strings(table: pa.Table, column: str) -> list[str]:
    """`table`'s `column` as Python strings: a frame's `stat_var`, role and period cells."""
    return cast("list[str]", table[column].to_pylist())


def _years(table: pa.Table) -> list[int]:
    """Each row's leading year."""
    return [int(period[:4]) for period in _strings(table, TIME_PERIOD_COLUMN)]


def _role_columns(components: Sequence[ComponentResult]) -> tuple[str, ...]:
    return components[0].requests[0].declared_roles


def _fold(
    fold: _Fold,
    frame: Frame,
    *,
    plan: _Plan,
    owner: Mapping[str, ComponentRole],
    role_columns: tuple[str, ...],
) -> None:
    """Count `frame`'s rows into `fold`: per member, per component the zero rows, years and
    the entities served on each role, and the frame's distinct role values and years. `owner`
    names each member's component."""
    table = frame.table
    if table.num_rows == 0:
        return
    fold.rows += table.num_rows
    stat_vars = _strings(table, STAT_VAR_COLUMN)
    years = _years(table)
    role_values = {role: _strings(table, role) for role in role_columns}
    periods = _strings(table, TIME_PERIOD_COLUMN)
    for position, stat_var in enumerate(stat_vars):
        role = owner[stat_var]
        fold.by_stat_var[stat_var] = fold.by_stat_var.get(stat_var, 0) + 1
        cell = (tuple(role_values[column][position] for column in role_columns), periods[position])
        if role in plan.zero_members and cell not in plan.member_cells.get(stat_var, set()):
            fold.zero_rows[role] = fold.zero_rows.get(role, 0) + 1
        fold.years.setdefault(role, set()).add(years[position])
        fold.all_years.add(years[position])
        served = fold.served.setdefault(role, {})
        for column in role_columns:
            value = role_values[column][position]
            served.setdefault(column, set()).add(value)
            fold.role_values.setdefault(column, set()).add(value)


def _summary(component: ComponentResult, *, plan: _Plan, fold: _Fold) -> ComponentSummary:
    facets = tuple(component.result.trace.facets)
    provenance = next(
        (
            cast("str | None", facet.get("ref"))
            for facet in facets
            if facet.get("axis") == _PROVENANCE_AXIS
        ),
        component.recipe.provenance,
    )
    served_years = fold.years.get(component.role, set())
    return ComponentSummary(
        role=component.role,
        provenance=provenance,
        content_version=component.result.content_version,
        periods=spans(served_years),
        dropped_years=tuple(
            sorted(plan.held_years[component.role] - plan.kept_years[component.role])
        ),
        stat_vars=tuple(sorted(request.stat_var for request in component.requests)),
        rows=sum(fold.by_stat_var.get(request.stat_var, 0) for request in component.requests),
        zero_rows=fold.zero_rows.get(component.role, 0),
        warnings=tuple(dict.fromkeys(warning["code"] for warning in component.result.warnings)),
        facets=facets,
    )


def spans(years: Iterable[int]) -> tuple[tuple[int, int], ...]:
    """`years` as closed spans of consecutive years, in order, as the mint's component
    summaries and Search's composite answers state them."""
    spans: list[list[int]] = []
    for year in sorted(set(years)):
        if spans and year == spans[-1][1] + 1:
            spans[-1][1] = year
        else:
            spans.append([year, year])
    return tuple((first, last) for first, last in spans)


def _composite_facets(
    traces: Sequence[ManifestTrace], *, composite: CompositeDeclaration
) -> tuple[dict[str, object], ...]:
    """The composite's provenance as its one provenance selection, then each distinct unit
    and method selection of its components, in component order."""
    entries: list[dict[str, object]] = [
        {
            "axis": _PROVENANCE_AXIS,
            "ref": composite.provenance,
            "basis": "bound",
            "default_policy": None,
            "default_policy_version": None,
            "selection_reason": f"the composite {composite.label}",
            "dispatch_facet_id": composite.provenance,
        }
    ]
    seen: set[tuple[object, object]] = set()
    for trace in traces:
        for facet in trace.facets:
            key = (facet.get("axis"), facet.get("ref"))
            if facet.get("axis") in (_UNIT_AXIS, _METHOD_AXIS) and key not in seen:
                seen.add(key)
                entries.append(dict(facet))
    return tuple(entries)


def _continuations(
    components: Sequence[ComponentResult], *, composite: CompositeDeclaration
) -> tuple[ContinuedSeries, ...]:
    """The continuations of every component's series, scoped to the composite provenance its
    rows carry in the composed frame."""
    return tuple(
        dict.fromkeys(
            replace(series, provenance=composite.provenance)
            if series.provenance is not None
            else series
            for component in components
            for series in continued_series_of_trace(
                {
                    "facets": component.result.trace.facets,
                    "continuations": component.result.trace.continuations,
                }
            )
        )
    )


def _union_entries(groups: Iterable[Iterable[dict[str, object]]]) -> tuple[dict[str, object], ...]:
    """Every distinct entry of `groups`, in first-seen order."""
    seen: list[dict[str, object]] = []
    for group in groups:
        for entry in group:
            if entry not in seen:
                seen.append(entry)
    return tuple(seen)


def _dependencies(
    components: Sequence[ComponentResult], *, requests: Sequence[MaterialisationRequest]
) -> Dependencies:
    """Every component's fingerprint as one: its members over every component, and each
    source, conversion source, recipe and latest year any component recorded. The window
    is the first component's, as `dependencies.refreshed_dependencies` reads the first
    request's."""
    fingerprints = [component.result.dependencies for component in components]

    def merged(name: str) -> tuple[tuple[str, object], ...]:
        pairs: dict[str, object] = {}
        for fingerprint in fingerprints:
            pairs.update(getattr(fingerprint, name))
        return tuple(sorted(pairs.items()))

    return Dependencies(
        members=members_digest(request.stat_var for request in requests),
        sources=cast("tuple[tuple[str, str | None], ...]", merged("sources")),
        conversion_sources=cast("tuple[tuple[str, str | None], ...]", merged("conversion_sources")),
        recipes=cast("tuple[tuple[str, str | None], ...]", merged("recipes")),
        window=fingerprints[0].window,
        latest_years=cast("tuple", merged("latest_years")),
    )


def _warnings(
    components: Sequence[ComponentResult],
    *,
    fold: _Fold,
    window: RequestedWindow,
    applied: AppliedWindow | None,
) -> tuple[MaterialisationWarning, ...]:
    """The composed frame's warnings, each code once (`compose_components`)."""
    warnings: list[MaterialisationWarning] = []
    if fold.rows == 0:
        warnings.append(_warning("empty_result"))
    unserved_by_role: dict[ComponentRole, list[str]] = {}
    for component in components:
        bound = _plain_role_bindings(component.requests[0].role_bindings)
        served = fold.served.get(component.role, {})
        unserved_by_role[component.role] = sorted(
            {
                entity
                for role, entities in bound.items()
                for entity in entities
                if entity not in served.get(role, set())
            }
        )
    unserved = sorted({entity for entities in unserved_by_role.values() for entity in entities})
    off_cadence, out_of_window, unserved_members, member_count = _member_warnings(components)
    clamped = _clamped_years(window, applied)
    countries_fallback = fold.rows > 0 and any(
        component.result.countries_fallback for component in components
    )
    if fold.rows and (
        unserved
        or off_cadence
        or out_of_window
        or unserved_members
        or clamped
        or countries_fallback
    ):
        warning = _coverage_below_scope_warning(
            unserved,
            countries_fallback=countries_fallback,
            off_cadence=off_cadence,
            out_of_window=out_of_window,
            unserved_members=unserved_members,
            member_count=member_count,
            years=clamped,
        )
        if unserved:
            counts = ", ".join(
                f"the {role} component leaves out {len(entities)}"
                for role, entities in unserved_by_role.items()
            )
            warning = cast(
                MaterialisationWarning,
                {**warning, "message": f"{warning['message']} Of these, {counts}."},
            )
        warnings.append(warning)
    omitted = {
        entity
        for component in components
        for record in component.result.member_conversion_records.values()
        if record is not None
        for entity in record.omitted_entities
    }
    if omitted:
        warnings.append(currency_unmapped_omitted_warning(omitted))
    deflated = [
        (entity, year)
        for component in components
        for record in component.result.member_conversion_records.values()
        if record is not None
        for entity, years in record.deflator_omitted
        for year in years
    ]
    if deflated:
        warnings.append(deflator_missing_omitted_warning(deflated))
    seen = {(warning["code"], warning["message"]) for warning in warnings}
    for component in components:
        for warning in component.result.warnings:
            key = (warning["code"], warning["message"])
            if warning["code"] in _RESTATED_CODES or key in seen:
                continue
            seen.add(key)
            warnings.append(warning)
    return tuple(warnings)


def _member_warnings(
    components: Sequence[ComponentResult],
) -> tuple[list[str], list[str], list[str], int]:
    """The members any component's `coverage_below_scope` counts as off cadence, out of the
    requested years, or left out of its elected facet group, from the refs each names, and
    the members those counts are out of."""
    counted: dict[str, list[str]] = {
        "off_cadence_members": [],
        "out_of_window_members": [],
        "unserved_members": [],
    }
    member_count = 0
    for component in components:
        for warning in component.result.warnings:
            if warning["code"] != "coverage_below_scope":
                continue
            fields = cast("Mapping[str, object]", warning)
            found = {
                key: cast("Mapping[str, object]", fields[key]) for key in counted if key in fields
            }
            if found:
                member_count += cast(int, fields.get("member_count", 0))
            for key, refs in found.items():
                counted[key].extend(_counted_refs(refs))
    return (
        counted["off_cadence_members"],
        counted["out_of_window_members"],
        counted["unserved_members"],
        member_count,
    )


def _counted_refs(counted: Mapping[str, object]) -> list[str]:
    """The refs a counted field carries, followed by an empty ref for each member it counts
    and does not name, so the merged count stays exact. A field names its first five
    members, and a count past five is the only one padded, so the five refs the merged
    warning names (`finish._counted`) come first."""
    refs = list(cast("Sequence[str]", counted.get("refs", ())))
    count = cast(int, counted.get("count", len(refs)))
    return refs + [""] * (count - len(refs))
