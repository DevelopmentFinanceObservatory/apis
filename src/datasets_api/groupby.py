"""Validate `groupby.by` and derive each frame's collapse axes.

The collapse set is the decoded, declared-summable axes minus `by`. Unknown
names refuse before per-frame checks. An undeclared axis absent from one frame
is a no-op. `stat_var` joins the collapse only when a CRS axis collapses or a
kept axis rolls up to a level. A decoded axis not declared summable stays out
of the collapse. Each decoded `stat_var_<axis>` cell holds the code's dcid
(`datasets_api.decode`), so a kept axis groups by code and two codes sharing a
label stay two rows. `groupby.levels` rewrites a kept hierarchical axis's decoded
codes to their ancestors at one depth (`roll_up_levels`), so the sum groups each
code under its ancestor's dcid.

The guards refuse ambiguous measures, unknown axes, decode gaps, and rows of
several datasets. They use code-index declarations and frame data. The reshape engine
performs the sum.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal, cast

import pyarrow as pa
import pyarrow.compute as pc
from dfo_frame_grammar import unit_codes

from code_index.composites import component_of, components_splice, components_sum
from code_index.query import CodeIndex
from code_index.types import (
    CodeSystemDeclaration,
    CompositeDeclaration,
    StatVarRow,
    summable_axes,
)
from dfo_platform.columns import PROVENANCE_COLUMN
from materialise.reshape import RollupNotSummableError, TableLike
from materialise.roles import KNOWN_ROLE_NAMES

_STAT_VAR_COLUMN = "stat_var"
_STAT_VAR_REF_PREFIX = "dcid:"
"""Mirrors `datasets_api.decode`'s own constants of the same name -- see
that module's docstring for why the prefix has to be stripped before a
table ref matches `index.statvars`."""

# Mirrors the fixed frame columns in `materialise.reshape`. Keep them local
# to avoid importing the grammar package. `value` stays valid in `by` as the
# summed column, which appears on every frame.
_ALWAYS_PRESENT_COLUMNS: frozenset[str] = frozenset(
    ("time_period", "value", "unit", "measurementMethod", PROVENANCE_COLUMN, "status")
)


def _bare_dcid(ref: str) -> str:
    return ref.removeprefix(_STAT_VAR_REF_PREFIX)


def declared_axes_present(
    *,
    decoded_code_systems: Sequence[str],
    declarations: Mapping[str, CodeSystemDeclaration],
) -> frozenset[str]:
    """Return declared-summable code systems decoded in this frame."""
    return summable_axes(declarations) & frozenset(decoded_code_systems)


def collapse_axes(
    *,
    by: Sequence[str],
    decoded_code_systems: Sequence[str],
    declarations: Mapping[str, CodeSystemDeclaration],
) -> tuple[str, ...]:
    """Return sorted decoded summable axes absent from `by`.

    `stat_var` is added by the caller only when this tuple is non-empty.
    Unknown names are checked separately. An axis absent from this frame adds
    no collapse entry.
    """
    keep = frozenset(by)
    present = declared_axes_present(
        decoded_code_systems=decoded_code_systems, declarations=declarations
    )
    return tuple(sorted(present - keep))


def unsummed_axes(
    *,
    decoded_code_systems: Sequence[str],
    declarations: Mapping[str, CodeSystemDeclaration],
) -> tuple[str, ...]:
    """Return sorted decoded code systems not declared summable.

    A sum never collapses these, so each stays a grain column of the summed rows.
    """
    return tuple(sorted(frozenset(decoded_code_systems) - summable_axes(declarations)))


def levels_in_frame(
    *, levels: Mapping[str, int] | None, decoded_code_systems: Sequence[str]
) -> dict[str, int]:
    """Return the `groupby.levels` entries whose axis this frame decodes. A level axis the
    frame's StatVars do not carry rolls nothing up, as an absent `by` axis collapses nothing."""
    decoded = frozenset(decoded_code_systems)
    return {axis: depth for axis, depth in (levels or {}).items() if axis in decoded}


def selected_stat_vars(
    lookup: Mapping[str, Mapping[str, str]],
    *,
    codes: Mapping[str, Sequence[str]] | None,
    decoded_code_systems: Sequence[str],
) -> frozenset[str] | None:
    """Return the StatVar refs of `lookup` whose decoded code on each `groupby.codes` axis this
    frame decodes is one of that axis's listed codes, or `None` when `codes` names no such
    axis. `lookup` is the decode lookup after any `levels` roll-up (`roll_up_levels`), so a
    StatVar is selected by its rolled code. A row's decoded cell is its StatVar's lookup cell,
    so a row is kept exactly when its `stat_var` is in the result; a StatVar with no cell on an
    axis reads an empty cell, which no code matches. An axis outside `decoded_code_systems`
    selects on nothing; a render refuses a selection on one (`render._decoded_selection`)."""
    decoded = frozenset(decoded_code_systems)
    in_frame = {
        axis: frozenset(listed) for axis, listed in (codes or {}).items() if axis in decoded
    }
    if not in_frame:
        return None
    return frozenset(
        ref
        for ref, cells in lookup.items()
        if all(cells.get(axis) in listed for axis, listed in in_frame.items())
    )


def selected_contributors(
    contributors: Iterable[str],
    *,
    index: CodeIndex,
    codes: Mapping[str, Sequence[str]] | None,
    levels: Mapping[str, int] | None = None,
) -> list[str]:
    """Return `contributors` (`dcid:` refs or bare dcids), in their order, that a groupby with
    the selection `codes` and the roll-up `levels` sums: each one whose code-index code on every
    `codes` axis, after any `levels` roll-up (`roll_up_levels`), is listed (`selected_stat_vars`),
    as the render sums only the rows of those codes. An axis no contributor carries selects on
    nothing."""
    refs = list(contributors)
    if not codes:
        return refs
    lookup: dict[str, dict[str, str]] = {}
    for ref in refs:
        statvar = index.statvar_by_dcid(_bare_dcid(ref))
        lookup[ref] = dict(statvar.constraint_properties) if statvar is not None else {}
    if levels:
        lookup = roll_up_levels(lookup, index=index, levels=levels)
    carried = sorted({axis for cells in lookup.values() for axis in cells})
    selected = selected_stat_vars(lookup, codes=codes, decoded_code_systems=carried)
    return refs if selected is None else [ref for ref in refs if ref in selected]


def sums_components(*, by: Sequence[str], composite: CompositeDeclaration | None) -> bool:
    """Return true when the groupby adds the components of `composite`, the composite the
    frame's rows carry as their provenance or `None`, into one total: the composite's
    components sum (`code_index.composites.components_sum`) and `by` keeps no `stat_var`.

    A stacked composite's components are parts of one total by declaration, so their StatVars
    sum whether or not they decode a summable axis. DAC2a's total ODA and its imputed
    multilateral ODA differ only on the aid type, which is not summable."""
    return composite is not None and components_sum(composite) and _STAT_VAR_COLUMN not in by


def splices_components(*, by: Sequence[str], composite: CompositeDeclaration | None) -> bool:
    """Return true when the groupby splices the components of `composite`, the composite the
    frame's rows carry as their provenance or `None`, into one series per cell: the
    composite's components splice (`code_index.composites.components_splice`) and `by` keeps
    no `stat_var`. Each cell takes the members of the one component holding it, and
    `SpliceCheck` refuses a frame in which it cannot."""
    return composite is not None and components_splice(composite) and _STAT_VAR_COLUMN not in by


def sums_stat_vars(
    *,
    by: Sequence[str],
    levels: Mapping[str, int] | None,
    decoded_code_systems: Sequence[str],
    declarations: Mapping[str, CodeSystemDeclaration],
    composite: CompositeDeclaration | None = None,
) -> bool:
    """Return true when the groupby collapses this frame's StatVars: a decoded summable axis
    collapses, `by` leaves out `stat_var` on a frame decoding a summable axis
    (`rekeys_stat_vars`), a kept axis rolls up to a level, the frame's stacked composite adds
    its components (`sums_components`), or its spliced composite splices them
    (`splices_components`)."""
    return (
        bool(
            collapse_axes(
                by=by, decoded_code_systems=decoded_code_systems, declarations=declarations
            )
        )
        or rekeys_stat_vars(
            by=by, decoded_code_systems=decoded_code_systems, declarations=declarations
        )
        or bool(levels_in_frame(levels=levels, decoded_code_systems=decoded_code_systems))
        or sums_components(by=by, composite=composite)
        or splices_components(by=by, composite=composite)
    )


_COMPONENT_COLUMN = "__component"
_CODES_COLUMN = "__codes"


class SpliceCheck:
    """The rows of a spliced composite's frame, noted chunk by chunk as Arrow group-bys before
    its `stat_var` collapse, so the collapse serves each cell from one component and adds no
    two components' values.

    A cell is a role tuple and time period (`cell_columns`). A row's component is the one of
    `composite` that serves its StatVar (`code_index.composites.component_of`). Each noted
    chunk keeps only its distinct cells with their components, and its distinct cells with
    the kept codes and StatVars of the members `selection` keeps, so the check holds at most
    a few short columns per frame row, beside the fold's `_ARROW_FOLD_BYTES_PER_ROW`, and no
    Python object per row.

    `refuse_structure` reads every row, whatever `selection` keeps, and raises
    `RollupNotSummableError` (`rollup_not_summable`): `spliced_component_unknown` for a
    StatVar no one component serves, among the resolved `members` or the rows, else
    `spliced_overlap` for the first cell holding rows of two components.

    While a splice runs, every fold of the collapse reads an absent member as no row, since a
    spliced cell holds one component's members and none of the others', and its codes on a
    collapsed axis can differ from another component's. `refuse_missing`, run after the
    `stat_var` fold's own preconditions, states the folds' missing-member rule for the
    component holding each cell instead, unless `missing_reads_zero`: a cell's group of kept
    codes (`kept_code_systems`, the decoded axes the collapse does not sum, read from
    `lookup`) expects every member of that component that `selection` keeps with those codes,
    so every code of a collapsed axis that component's members hold. A row with a null or
    non-string `stat_var` refuses `missing_members` too, where the fold would read it: with no
    `selection`, since a selection keeps no row without a member."""

    def __init__(
        self,
        composite: CompositeDeclaration,
        *,
        index: CodeIndex,
        cell_columns: Sequence[str],
        members: Iterable[str],
        lookup: Mapping[str, Mapping[str, str]],
        kept_code_systems: Sequence[str],
        missing_reads_zero: bool,
        selection: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        self._composite = composite
        self._index = index
        self.cell_columns = tuple(cell_columns)
        self._lookup = lookup
        self._kept = tuple(kept_code_systems)
        self._missing_reads_zero = missing_reads_zero
        self._selection = selection
        self._component_of: dict[str, str | None] = {}
        self._unknown: str | None = None
        self._expected: dict[tuple[str, str], set[str]] = {}
        for ref in members:
            component = self._component(ref)
            if component is not None and self._selected(ref):
                self._expected.setdefault((component, self._codes(ref)), set()).add(ref)
        self._cell_parts: list[pa.Table] = []
        self._member_parts: list[pa.Table] = []
        self._null_rows = 0

    def note(self, stat_vars: pa.Array | pa.ChunkedArray, cells: pa.Table) -> None:
        """Note the rows whose `stat_var` refs are `stat_vars`, strings with a null for a row
        holding none, and whose cells are the rows of `cells`, its `cell_columns`."""
        stat_vars = pc.cast(stat_vars, pa.string())
        if stat_vars.null_count:
            if self._selection is None:
                self._null_rows += stat_vars.null_count
            valid = pc.is_valid(stat_vars)
            stat_vars, cells = stat_vars.filter(valid), cells.filter(valid)
        if len(stat_vars) == 0:
            return
        refs = cast("list[str]", pc.unique(stat_vars).to_pylist())
        positions = pc.index_in(stat_vars, value_set=pa.array(refs, pa.string()))
        components = pa.array([self._component(ref) for ref in refs], pa.string()).take(positions)
        keyed = cells.select(list(self.cell_columns)).append_column(_COMPONENT_COLUMN, components)
        self._cell_parts.append(
            keyed.group_by([*self.cell_columns, _COMPONENT_COLUMN], use_threads=False).aggregate([])
        )
        if self._missing_reads_zero:
            return
        selected = pa.array([self._selected(ref) for ref in refs], pa.bool_()).take(positions)
        codes = pa.array([self._codes(ref) for ref in refs], pa.string()).take(positions)
        members = (
            keyed.append_column(_CODES_COLUMN, codes)
            .append_column(_STAT_VAR_COLUMN, stat_vars)
            .filter(selected)
        )
        self._member_parts.append(
            members.group_by(
                [*self.cell_columns, _COMPONENT_COLUMN, _CODES_COLUMN, _STAT_VAR_COLUMN],
                use_threads=False,
            ).aggregate([])
        )

    def refuse_structure(self) -> None:
        """Raise `RollupNotSummableError` for a StatVar no one component serves, else for a
        cell holding rows of two components."""
        if self._unknown is not None:
            raise RollupNotSummableError(
                f"groupby axis {_STAT_VAR_COLUMN!r} refused: {self._unknown!r} is a member of "
                f"no one component of the spliced composite {self._composite.provenance!r}, "
                "so no cell can be read as one component's",
                axis=_STAT_VAR_COLUMN,
                precondition="spliced_component_unknown",
            )
        if not self._cell_parts:
            return
        held = (
            pa.concat_tables(self._cell_parts, promote_options="default")
            .group_by(list(self.cell_columns), use_threads=False)
            .aggregate([(_COMPONENT_COLUMN, "distinct")])  # ty: ignore[invalid-argument-type]  # pyarrow-stubs widen the aggregation name
        )
        lengths = cast(
            "pa.Int32Array",
            pc.list_value_length(held.column(f"{_COMPONENT_COLUMN}_distinct").combine_chunks()),
        )
        overlapping = held.filter(
            cast("pa.BooleanArray", pc.greater(lengths, pa.scalar(1, pa.int32())))
        )
        if overlapping.num_rows == 0:
            return
        first = overlapping.slice(0, 1).to_pylist()[0]
        roles = sorted(first.pop(f"{_COMPONENT_COLUMN}_distinct"))
        raise RollupNotSummableError(
            f"groupby axis {_STAT_VAR_COLUMN!r} refused: the spliced composite "
            f"{self._composite.provenance!r} serves one component in each cell, but the cell "
            f"{first!r} holds rows of {roles!r}, so the frame's components overlap and their "
            "values are not added",
            axis=_STAT_VAR_COLUMN,
            precondition="spliced_overlap",
        )

    def refuse_missing(self) -> None:
        """Raise `RollupNotSummableError` (`missing_members`) for a row with no `stat_var`, or
        a cell lacking a member of the component holding it, unless `missing_reads_zero`."""
        if self._missing_reads_zero:
            return
        if self._null_rows:
            raise RollupNotSummableError(
                f"groupby axis {_STAT_VAR_COLUMN!r} refused: {self._null_rows} row(s) of the "
                f"spliced composite {self._composite.provenance!r} hold no {_STAT_VAR_COLUMN!r} "
                "(a null/suppressed member), and the source has not declared missing members "
                "as zero",
                axis=_STAT_VAR_COLUMN,
                precondition="missing_members",
            )
        if not self._member_parts:
            return
        groups = (
            pa.concat_tables(self._member_parts, promote_options="default")
            .group_by([*self.cell_columns, _COMPONENT_COLUMN, _CODES_COLUMN], use_threads=False)
            .aggregate([(_STAT_VAR_COLUMN, "distinct")])  # ty: ignore[invalid-argument-type]  # pyarrow-stubs widen the aggregation name
        )
        for group in groups.to_pylist():
            observed = set(group.pop(f"{_STAT_VAR_COLUMN}_distinct"))
            component = group.pop(_COMPONENT_COLUMN)
            expected = self._expected.get((component, group.pop(_CODES_COLUMN)), set())
            if absent := expected - observed:
                raise RollupNotSummableError(
                    f"groupby axis {_STAT_VAR_COLUMN!r} refused: expected member(s) "
                    f"{tuple(sorted(absent))!r} of the spliced composite's component "
                    f"{component!r} never appear in the cell {group!r}, and the source has "
                    "not declared missing members as zero",
                    axis=_STAT_VAR_COLUMN,
                    precondition="missing_members",
                )

    def _codes(self, ref: str) -> str:
        codes = self._lookup.get(ref, {})
        return json.dumps([codes.get(code_system) for code_system in self._kept])

    def _selected(self, ref: str) -> bool:
        if self._selection is None:
            return True
        codes = self._lookup.get(ref, {})
        return all(codes.get(axis) in kept for axis, kept in self._selection.items())

    def _component(self, ref: str) -> str | None:
        """The role of the one component serving `ref`, noting `ref` as unknown when none
        does."""
        if ref not in self._component_of:
            row = self._index.statvar_by_dcid(_bare_dcid(ref))
            role = component_of(self._composite, row) if row is not None else None
            self._component_of[ref] = role
            if role is None and self._unknown is None:
                self._unknown = ref
        return self._component_of[ref]


_REPORTER_COLUMN = "__reporter"
_TOTAL_COLUMN = "__total"


class ReporterCheck:
    """The rows a collapse of a dataset declaring `members_report_overlapping_accounts` sums,
    noted chunk by chunk as Arrow group-bys before its `stat_var` collapse: each row's cell,
    kept codes, StatVar and reporting organisation (`reporting_organisation`, the member
    provenance that filed it).

    A cell is a role tuple and time period (`cell_columns`). A total is a cell and the codes
    of the decoded axes the collapse does not sum (`kept_code_systems`, read from `lookup`).
    Only the rows of the StatVars `selection` keeps are noted, and a row with no `stat_var`
    is not, so the check holds a few short columns per distinct row.

    `summed_reporters` counts the reporting organisations whose rows the collapse sums, and
    reads zero unless some total folds rows of two or more of them, the figure the
    `reporters_summed` warning describes.

    `refuse_unknown_absence`, run with `reads_zero` after the `stat_var` fold's own
    preconditions, refuses a total missing a member whose absence the dataset reads as
    unknown in the cell's year. The expected members of a total are the `members` that
    `selection` keeps with its codes. An absent member's provenance is the reporting
    organisation of its own rows at the cell's role tuple, the series the absent row belongs
    to, else that of the total's rows, and `reads_zero(provenance, year)` judges it
    (`code_index.members.absent_reads_zero_in`)."""

    def __init__(
        self,
        *,
        cell_columns: Sequence[str],
        members: Iterable[str],
        lookup: Mapping[str, Mapping[str, str]],
        kept_code_systems: Sequence[str],
        selection: Mapping[str, frozenset[str]] | None = None,
        reads_zero: Callable[[str | None, int | None], bool] | None = None,
    ) -> None:
        self.cell_columns = tuple(cell_columns)
        self._lookup = lookup
        self._kept = tuple(kept_code_systems)
        self._selection = selection
        self._reads_zero = reads_zero
        self._expected: dict[str, set[str]] = {}
        for ref in members:
            if self._selected(ref):
                self._expected.setdefault(self._codes(ref), set()).add(ref)
        self._parts: list[pa.Table] = []

    def note(
        self,
        stat_vars: pa.Array | pa.ChunkedArray,
        reporters: pa.Array | pa.ChunkedArray,
        cells: pa.Table,
    ) -> None:
        """Note the rows whose `stat_var` refs are `stat_vars`, whose reporting organisations
        are `reporters` and whose cells are the rows of `cells`, its `cell_columns`."""
        stat_vars = pc.cast(stat_vars, pa.string())
        reporters = pc.cast(reporters, pa.string())
        valid = pc.is_valid(stat_vars)
        stat_vars, reporters, cells = (
            stat_vars.filter(valid),
            reporters.filter(valid),
            cells.filter(valid),
        )
        if len(stat_vars) == 0:
            return
        refs = cast("list[str]", pc.unique(stat_vars).to_pylist())
        positions = pc.index_in(stat_vars, value_set=pa.array(refs, pa.string()))
        selected = pa.array([self._selected(ref) for ref in refs], pa.bool_()).take(positions)
        codes = pa.array([self._codes(ref) for ref in refs], pa.string()).take(positions)
        noted = (
            cells.select(list(self.cell_columns))
            .append_column(_CODES_COLUMN, codes)
            .append_column(_STAT_VAR_COLUMN, stat_vars)
            .append_column(_REPORTER_COLUMN, reporters)
            .filter(selected)
        )
        self._parts.append(
            noted.group_by(
                [*self.cell_columns, _CODES_COLUMN, _STAT_VAR_COLUMN, _REPORTER_COLUMN],
                use_threads=False,
            ).aggregate([])
        )

    def summed_reporters(self) -> int:
        """How many reporting organisations the totals folding rows of two or more of them
        name, else 0. A row naming no reporting organisation counts as none."""
        if not self._parts:
            return 0
        noted = pa.concat_tables(self._parts, promote_options="default")
        totals = _row_codes(noted, [*self.cell_columns, _CODES_COLUMN])
        reporters = noted.column(_REPORTER_COLUMN)
        aggregation: tuple[str, Literal["count_distinct"], pc.CountOptions] = (
            _REPORTER_COLUMN,
            "count_distinct",
            pc.CountOptions(mode="only_valid"),
        )
        per_total = (
            pa.table({_TOTAL_COLUMN: totals, _REPORTER_COLUMN: reporters})
            .group_by(_TOTAL_COLUMN, use_threads=False)
            .aggregate([aggregation])
        )
        mixing = per_total.filter(
            cast(
                "pa.BooleanArray",
                pc.greater_equal(
                    per_total.column(f"{_REPORTER_COLUMN}_count_distinct"),
                    pa.scalar(2, pa.int64()),
                ),
            )
        ).column(_TOTAL_COLUMN)
        if len(mixing) == 0:
            return 0
        summed = reporters.filter(pc.is_in(totals, value_set=mixing.combine_chunks()))
        return len(pc.unique(summed).drop_null())

    def refuse_unknown_absence(self) -> None:
        """Raise `RollupNotSummableError` (`missing_members`) for a total missing a member
        whose absence reads unknown in its cell's year. A no-op without `reads_zero`.

        The totals missing an expected member are found in Arrow, by counting each total's
        distinct expected StatVars against its codes' expected count, and only their rows,
        and the rows of each absent member's own series, are read into Python."""
        if self._reads_zero is None or not self._parts:
            return
        noted = pa.concat_tables(self._parts, promote_options="default")
        missing = self._missing_totals(noted)
        if missing is None:
            return
        time_column = self.cell_columns[-1]
        role_columns = self.cell_columns[:-1]
        totals: dict[tuple[object, ...], tuple[set[str], set[str | None]]] = {}
        for row in _python_rows(noted.filter(missing)):
            roles = tuple(row[column] for column in role_columns)
            key = (*roles, row[time_column], row[_CODES_COLUMN])
            observed, reporters = totals.setdefault(key, (set(), set()))
            observed.add(cast("str", row[_STAT_VAR_COLUMN]))
            reporters.add(cast("str | None", row[_REPORTER_COLUMN]))
        absent = {
            key: sorted(self._expected.get(key[-1], set()) - observed)
            for key, (observed, _) in totals.items()
        }
        series = self._series_reporters(
            noted,
            {(*key[:-2], member) for key, members in absent.items() for member in members},
        )
        for key, (_, reporters) in totals.items():
            *roles, period, _codes = key
            year = _leading_year(period)
            for member in absent[key]:
                filed = series.get((*roles, member)) or reporters
                if all(self._reads_zero(provenance, year) for provenance in filed):
                    continue
                cell = dict(zip(self.cell_columns, (*roles, period), strict=True))
                raise RollupNotSummableError(
                    f"groupby axis {_STAT_VAR_COLUMN!r} refused: expected member {member!r} "
                    f"never appears in the cell {cell!r}, and its reporting organisation "
                    "holds no observation in that year, so its absence there is unknown "
                    "rather than zero",
                    axis=_STAT_VAR_COLUMN,
                    precondition="missing_members",
                )

    def _missing_totals(self, noted: pa.Table) -> pa.BooleanArray | None:
        """Whether each of `noted`'s rows belongs to a total holding fewer distinct expected
        StatVars than its codes expect, or None when no total does."""
        pair_columns = [_CODES_COLUMN, _STAT_VAR_COLUMN]
        pairs = [(codes, ref) for codes, refs in self._expected.items() for ref in refs]
        expected_pairs = pa.table(
            {
                _CODES_COLUMN: pa.array([codes for codes, _ in pairs], pa.string()),
                _STAT_VAR_COLUMN: pa.array([ref for _, ref in pairs], pa.string()),
            }
        )
        pair_codes = _row_codes(
            pa.concat_tables(
                [noted.select(pair_columns).cast(expected_pairs.schema), expected_pairs]
            ),
            pair_columns,
        )
        rows = noted.num_rows
        expected_pair = pc.is_in(pair_codes.slice(0, rows), value_set=pair_codes.slice(rows))
        codes_order = list(self._expected)
        expected_count = pa.array(
            [len(self._expected[codes]) for codes in codes_order], pa.int64()
        ).take(
            pc.index_in(noted.column(_CODES_COLUMN), value_set=pa.array(codes_order, pa.string()))
        )
        totals = _row_codes(noted, [*self.cell_columns, _CODES_COLUMN])
        member = pc.if_else(expected_pair, pair_codes.slice(0, rows), pa.scalar(None, pa.int64()))
        counted: tuple[str, Literal["count_distinct"], pc.CountOptions] = (
            "member",
            "count_distinct",
            pc.CountOptions(mode="only_valid"),
        )
        highest: tuple[str, Literal["max"], None] = ("expected", "max", None)
        per_total = (
            pa.table({_TOTAL_COLUMN: totals, "member": member, "expected": expected_count})
            .group_by(_TOTAL_COLUMN, use_threads=False)
            .aggregate([counted, highest])
        )
        short = per_total.filter(
            cast(
                "pa.BooleanArray",
                pc.less(
                    per_total.column("member_count_distinct"), per_total.column("expected_max")
                ),
            )
        ).column(_TOTAL_COLUMN)
        if len(short) == 0:
            return None
        return pc.is_in(totals, value_set=short.combine_chunks())

    def _series_reporters(
        self, noted: pa.Table, wanted: set[tuple[object, ...]]
    ) -> dict[tuple[object, ...], set[str | None]]:
        """The reporting organisations of `noted`'s rows of each `wanted` series, a role tuple
        and StatVar, read into Python for those series' rows alone."""
        if not wanted:
            return {}
        series_columns = [*self.cell_columns[:-1], _STAT_VAR_COLUMN]
        query = pa.Table.from_pylist(
            [dict(zip(series_columns, key, strict=True)) for key in wanted],
            schema=noted.select(series_columns).schema,
        )
        codes = _row_codes(pa.concat_tables([noted.select(series_columns), query]), series_columns)
        rows = noted.num_rows
        kept = pc.is_in(codes.slice(0, rows), value_set=codes.slice(rows))
        series: dict[tuple[object, ...], set[str | None]] = {}
        for row in _python_rows(noted.select([*series_columns, _REPORTER_COLUMN]).filter(kept)):
            key = tuple(row[column] for column in series_columns)
            series.setdefault(key, set()).add(cast("str | None", row[_REPORTER_COLUMN]))
        return series

    def _codes(self, ref: str) -> str:
        codes = self._lookup.get(ref, {})
        return json.dumps([codes.get(code_system) for code_system in self._kept])

    def _selected(self, ref: str) -> bool:
        if self._selection is None:
            return True
        codes = self._lookup.get(ref, {})
        return all(codes.get(axis) in kept for axis, kept in self._selection.items())


def _row_codes(table: pa.Table, columns: Sequence[str]) -> pa.Int64Array:
    """One code per row of `table`, equal exactly where every one of `columns` holds the same
    value or null (`dfo_frame_grammar.unit_codes`)."""
    return cast(
        "pa.Int64Array",
        unit_codes(
            pa.StructArray.from_arrays(
                [table.column(column).combine_chunks() for column in columns], names=list(columns)
            )
        ).codes,
    )


def _python_rows(table: pa.Table) -> list[dict[str, object]]:
    """`table`'s rows as Python dicts, the one place `ReporterCheck` reads rows into Python."""
    return table.to_pylist()


def _leading_year(period: object) -> int | None:
    """The year a `time_period` cell begins with, `None` for a cell holding none."""
    if not isinstance(period, str):
        return None
    try:
        return int(period[:4])
    except ValueError:
        return None


def rekeys_stat_vars(
    *,
    by: Sequence[str],
    decoded_code_systems: Sequence[str],
    declarations: Mapping[str, CodeSystemDeclaration],
) -> bool:
    """Return true when `by` leaves out `stat_var` and the frame decodes a declared-summable
    axis: the groupby reads one `stat_var` keyed by the axes `by` keeps, summing those it
    leaves out. With every such axis kept it sums nothing, each kept tuple holding one StatVar,
    and still reads the frame keyed by them (conformance C118)."""
    return _STAT_VAR_COLUMN not in by and bool(
        declared_axes_present(decoded_code_systems=decoded_code_systems, declarations=declarations)
    )


def kept_axes(
    *,
    by: Sequence[str],
    decoded_code_systems: Sequence[str],
    declarations: Mapping[str, CodeSystemDeclaration],
) -> tuple[str, ...]:
    """Return sorted decoded summable axes named in `by`, the code axes a sum keeps as grain."""
    present = declared_axes_present(
        decoded_code_systems=decoded_code_systems, declarations=declarations
    )
    return tuple(sorted(present & frozenset(by)))


def yields_grammar_frame(
    *,
    decoded_code_systems: Sequence[str],
    declarations: Mapping[str, CodeSystemDeclaration],
    lookup: Mapping[str, Mapping[str, str]],
    components: bool = False,
) -> bool:
    """Return true when the sum leaves a grammar frame: this frame decodes a summable axis, or
    the collapse adds a stacked composite's components (`sums_components`) or splices a
    spliced one's (`splices_components`), either of which `components` states, and every
    unsummed axis holds one value across the resolved members in `lookup`.

    The frame carries one `stat_var` on every row and keeps each summable axis named in `by`
    (`kept_axes`) as a string grain column. An unsummed axis holding one value describes the
    whole family, so its constant column is left out of that frame. One holding several values
    keeps its column and the rows stay a table grouped by it. An empty summable axis set is
    false unless `components`.
    """
    present = declared_axes_present(
        decoded_code_systems=decoded_code_systems, declarations=declarations
    )
    if not present and not components:
        return False
    return all(
        len({cells.get(code_system) for cells in lookup.values()}) == 1
        for code_system in unsummed_axes(
            decoded_code_systems=decoded_code_systems, declarations=declarations
        )
    )


def roll_up_levels(
    lookup: Mapping[str, Mapping[str, str]],
    *,
    index: CodeIndex,
    levels: Mapping[str, int],
) -> dict[str, dict[str, str]]:
    """Return `lookup` with each level axis's code replaced by its depth ancestor's dcid.

    A StatVar's code maps to `CodeIndex.ancestor_at_depth`. A code shallower than its depth has
    no ancestor there and keeps its own dcid, so its row names it. A declared absence and a
    missing cell stay as they are, and an ancestor with no code row drops the cell, which
    `refuse_decode_gaps` then refuses.
    """
    bare_dcids = {_bare_dcid(ref) for ref in lookup}
    statvar_by_dcid = {row.dcid: row for row in index.statvars if row.dcid in bare_dcids}
    rolled: dict[str, dict[str, str]] = {}
    for ref, cells in lookup.items():
        new_cells = dict(cells)
        statvar = statvar_by_dcid.get(_bare_dcid(ref))
        for code_system, depth in levels.items():
            code = statvar.constraint_properties.get(code_system) if statvar else None
            if code is None or code_system not in cells:
                continue
            ancestor = index.ancestor_at_depth(code, depth)
            if ancestor is None:
                continue
            if index.code_by_dcid(ancestor) is None:
                del new_cells[code_system]
            else:
                new_cells[code_system] = ancestor
        rolled[ref] = new_cells
    return rolled


def contributing_dcids(table: TableLike) -> frozenset[str]:
    """Return distinct StatVars with rows, before `stat_var` collapses.

    Only members with rows contribute. The count drives the `stat_vars_summed`
    warning for full collapses.
    """
    if _STAT_VAR_COLUMN not in table.columns:
        return frozenset()
    pos = table.columns.index(_STAT_VAR_COLUMN)
    return frozenset(cast(str, row[pos]) for row in table.rows if isinstance(row[pos], str))


def refuse_stat_var_collision(
    table: TableLike,
    *,
    index: CodeIndex,
    decoded_code_systems: Sequence[str],
    composite: CompositeDeclaration | None = None,
) -> None:
    """Raise `RollupNotSummableError` (`rollup_not_summable`,
    `stat_var_ambiguous`) when distinct StatVars share values across all decoded axes.

    Once `stat_var` collapses, decoded axes alone distinguish rows. Population
    type, measured property, and related identity fields sit outside the frame,
    so matching axes can merge distinct measures. Check every StatVar before
    collapsing, even when rows differ elsewhere.

    `composite` is the declared composite the frame's rows carry as their provenance, or
    `None`. Its components' members are parts of one total, so a group of StatVars passes
    when each is a member of a different component (`code_index.composites.component_of`),
    as the bilateral and imputed parts of one sector are, or the net and grant-equivalent
    years of one aid type, which share no year.
    """
    if _STAT_VAR_COLUMN not in table.columns:
        return
    pos = table.columns.index(_STAT_VAR_COLUMN)
    bare_dcids: set[str] = set()
    for row in table.rows:
        cell = row[pos]
        if isinstance(cell, str):
            bare_dcids.add(_bare_dcid(cell))
    statvar_by_dcid = {row.dcid: row for row in index.statvars if row.dcid in bare_dcids}

    sorted_systems = sorted(decoded_code_systems)
    groups: dict[tuple[str | None, ...], set[str]] = {}
    for row in statvar_by_dcid.values():
        key = tuple(row.constraint_properties.get(code_system) for code_system in sorted_systems)
        groups.setdefault(key, set()).add(row.dcid)

    for dcids in groups.values():
        if len(dcids) > 1 and not _one_per_component(dcids, composite, statvar_by_dcid):
            named = ", ".join(sorted(dcids))
            raise RollupNotSummableError(
                f"groupby axis 'stat_var' refused: StatVar(s) {named} decode to the same "
                f"values across {tuple(sorted_systems)!r} but are distinct measures; "
                "collapsing stat_var would merge them with no discriminating column left",
                axis=_STAT_VAR_COLUMN,
                precondition="stat_var_ambiguous",
            )


def adds_stat_vars(
    contributors: Iterable[str],
    *,
    index: CodeIndex,
    kept: Sequence[str],
    levels: Mapping[str, int],
    components: bool,
    composite: CompositeDeclaration | None = None,
) -> bool:
    """Whether a groupby collapse adds StatVars together: a level roll-up or a stacked
    composite's component sum ran, or two of `contributors` (`dcid:` refs or bare dcids), the
    StatVars with rows, hold one tuple of codes on the `kept` axes. A collapse where each kept
    tuple holds one StatVar re-keys the frame under one `stat_var` and adds nothing
    (`rekeys_stat_vars`).

    `composite` is the composite the frame's rows carry, or `None`. A spliced composite's
    collapse is a splice: each cell holds one component's members (`SpliceCheck`), so its
    contributors are keyed on their component (`code_index.composites.component_of`) as well
    as their kept codes, and only two members of one component sharing kept codes add. The
    headline's net and grant-equivalent members add nothing. A StatVar no one component
    serves reads as adding, since nothing tells it apart; the mint refuses it
    (`SpliceCheck`, `spliced_component_unknown`)."""
    spliced = composite is not None and composite.kind == "spliced"
    if levels or (components and not spliced):
        return True
    seen: set[tuple[str | None, ...]] = set()
    for ref in contributors:
        statvar = index.statvar_by_dcid(_bare_dcid(ref))
        codes = statvar.constraint_properties if statvar is not None else {}
        key = tuple(codes.get(axis) for axis in kept)
        if spliced:
            assert composite is not None
            role = component_of(composite, statvar) if statvar is not None else None
            if role is None:
                # A StatVar no one component serves cannot be told apart from the others, so
                # the collapse is read as adding it (`SpliceCheck` refuses it at the mint).
                return True
            key = (role, *key)
        if key in seen:
            return True
        seen.add(key)
    return False


def _one_per_component(
    dcids: Iterable[str],
    composite: CompositeDeclaration | None,
    statvar_by_dcid: Mapping[str, StatVarRow],
) -> bool:
    """Whether each of `dcids` is a member of a different component of `composite`."""
    if composite is None:
        return False
    roles = [component_of(composite, statvar_by_dcid[dcid]) for dcid in dcids]
    return None not in roles and len(set(roles)) == len(roles)


def refuse_unknown_by_axes(
    *, by: Sequence[str], declarations: Mapping[str, CodeSystemDeclaration]
) -> None:
    """Raise `RollupNotSummableError` (`rollup_not_summable`,
    `not_declared_summable`) for the first `by` name outside declared
    summable axes, known roles, `stat_var`, and always-present columns.

    This check is deployment-wide. A heterogeneous frame can omit a known axis.
    That name is valid across the deployment.
    """
    known = (
        summable_axes(declarations)
        | KNOWN_ROLE_NAMES
        | {_STAT_VAR_COLUMN}
        | _ALWAYS_PRESENT_COLUMNS
    )
    for name in by:
        if name not in known:
            raise RollupNotSummableError(
                f"groupby.by names {name!r}, which this deployment declares neither a "
                "summable code system, a role name, nor a column every frame carries",
                axis=name,
                precondition="not_declared_summable",
            )


def refuse_decode_gaps(
    *, lookup: Mapping[str, Mapping[str, str]], refs: Iterable[str], code_systems: Sequence[str]
) -> None:
    """Raise `RollupNotSummableError` (`rollup_not_summable`, `index_gap`) if a
    ref lacks a decoded value for a requested code system.

    A declared absence has a value. A lookup miss becomes an empty cell and
    could merge distinct StatVars after collapse. Check every ref, including
    zero-row members, across every decoded code system.
    """
    missing: list[str] = []
    for ref in sorted(set(refs)):
        cells = lookup.get(ref, {})
        missing.extend(
            f"{ref}/{code_system}" for code_system in code_systems if code_system not in cells
        )
    if missing:
        raise RollupNotSummableError(
            f"groupby axis 'stat_var' refused: {', '.join(missing)} have no decoded value "
            "(a genuine code-index gap rather than a declared absence), and merging an "
            "undecoded StatVar into a rollup could silently combine distinct measures",
            axis=_STAT_VAR_COLUMN,
            precondition="index_gap",
        )


def refuse_provenance_disagreement(*, provenance_refs: Sequence[str]) -> None:
    """Raise `RollupNotSummableError` (`rollup_not_summable`,
    `provenance_disagreement`) when the frame spans multiple datasets.

    `provenance_refs` are dataset ids. The rows of a declared dataset's member provenances
    carry the dataset's id (`materialise.columnar.dataset_cells`), so the members of one
    dataset agree. Each dataset declares whether absent observations count as zero or stay
    unknown. Mixed datasets can disagree, so refuse before selecting a rule. A materialised
    group carries one dataset, so this guards other callers.
    """
    distinct = sorted(set(provenance_refs))
    if len(distinct) > 1:
        raise RollupNotSummableError(
            f"groupby refused: this frame's rows come from {len(distinct)} distinct "
            f"datasets ({', '.join(distinct)}), and whether an absent member counts as "
            "zero is a per-source declaration -- a summed group spanning sources that could "
            "disagree on it cannot be resolved to one answer",
            axis="provenance",
            precondition="provenance_disagreement",
        )
