"""Pure engines for decode, groupby, column selection, enrichment, and pivot.
`apply_reshapes` runs these steps in a fixed order before format writing. Graph,
code-index, hierarchy, and format-cap inputs arrive as parameters; the engines perform no I/O
and keep no identity state. The input frame's content version is minted before reshape.

Engines accept a structural `TableLike` with `.columns` and `.rows`. Reshape
results use this module's frozen `Table` because grouping and pivoting can break
`Frame`'s long-form identity invariants.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence, Set
from dataclasses import dataclass, field
from typing import Protocol

from dfo_platform.codes import CodeCatalogue
from dfo_platform.problem import Alternative, ProblemIssue
from materialise.codes import MaterialisationWarning
from materialise.enrichment import PARTY_KIND_FIELD, is_party_role
from materialise.errors import MaterialisationError, MaterialisationUsageError
from materialise.method_status import method_status_classes

# Mirrors dfo_frame_grammar.grammar's fixed column names (frame_build.py's
# `_STAT_VAR_REF_PREFIX` is this same codebase's precedent for mirroring a
# grammar constant here rather than importing dfo_frame_grammar for it --
# this module's own independence rule forbids that import).
_STAT_VAR_COLUMN = "stat_var"
_TIME_PERIOD_COLUMN = "time_period"
_VALUE_COLUMN = "value"
_UNIT_COLUMN = "unit"
_STATUS_COLUMN = "status"
_MEASUREMENT_METHOD_COLUMN = "measurementMethod"
_PROVENANCE_COLUMN = "provenance"
_FACET_IDENTITY_COLUMNS: tuple[str, ...] = (_MEASUREMENT_METHOD_COLUMN, _PROVENANCE_COLUMN)

SUPPORTED_ENRICHMENTS: frozenset[str] = frozenset({"iso2", "iso3", "label", "party_kind", "type"})
"""The closed enrichment field vocabulary this engine can fill. Capabilities
publishing reads this constant directly rather than duplicating it."""

TABLE_WIDE_ENRICHMENTS: frozenset[str] = frozenset({"party_kind"})
"""The enrichment fields a bare token requests for every party role of the
table (`materialise.enrichment.is_party_role`). A sum across any party role needs
each party's kind, so a caller asks for it without knowing which roles the table
carries. Every other field needs its entity-column prefix."""

_MAX_NAMED_DUPLICATE_KEYS = 10
"""Cap on how many duplicate pivot keys a `pivot_not_unique` message names,
mirroring `facets._MAX_ALTERNATIVES_PER_ISSUE`'s capped-list convention."""

_MAX_HOMOGENEITY_VALUES_PER_GROUP = 3
"""Cap on how many distinct values `pivot`'s homogeneity check tracks per
`(column, index_value)` group as it scans the table. `len(values) > 1` is
the only thing the check itself needs, so a cap at or above 2 preserves the
check exactly; the extra room above 2 gives `pivot_not_homogeneous`'s own
message a small, genuine sample of the offending values rather than just
proof that more than one exists."""


class TableLike(Protocol):
    """The structural table contract every reshape engine accepts: the
    grammar `Frame` satisfies this as-is, without either side
    importing the other. Declared as read-only properties rather than
    plain attributes -- a plain-attribute `Protocol` member is implicitly
    read-write, which a frozen dataclass (this module's own `Table`, and
    the grammar `Frame`) can never satisfy since neither allows attribute
    assignment at all.
    """

    @property
    def columns(self) -> tuple[str, ...]: ...

    @property
    def rows(self) -> tuple[tuple[object, ...], ...]: ...


@dataclass(frozen=True)
class Table:
    """The module-local table shape every reshape engine returns: exactly
    `.columns`/`.rows`, matching `TableLike`. A grouped or pivoted table
    cannot satisfy `Frame`'s long-form invariants, so this is always the
    module's own type, never a `Frame`.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class ColumnClassification:
    """Classify actual table column names, independent of declared roles.
    `facet_axis_columns` contains present `unit`, `measurementMethod`, and
    `provenance` fields; groupby refuses these axes. `non_droppable_columns`
    contains grain, unit, and provenance, while `measurementMethod` can be
    dropped after homogeneity checks. `apply_reshapes` derives this before
    decoding, so decoded annotations stay droppable.
    """

    facet_axis_columns: frozenset[str]
    non_droppable_columns: frozenset[str]


def classify_columns(columns: Sequence[str]) -> ColumnClassification:
    """Classify the table's present names, even after earlier column selection.
    Every nonfixed, nonfacet column belongs to the grain.
    """
    present = set(columns)
    facet_axis_columns = frozenset(
        name for name in (_UNIT_COLUMN, *_FACET_IDENTITY_COLUMNS) if name in present
    )
    grain_columns = frozenset(
        name
        for name in columns
        if name not in facet_axis_columns and name not in (_VALUE_COLUMN, _STATUS_COLUMN)
    )
    non_droppable_columns = grain_columns | (present & {_UNIT_COLUMN, _PROVENANCE_COLUMN})
    return ColumnClassification(
        facet_axis_columns=facet_axis_columns, non_droppable_columns=non_droppable_columns
    )


# The Datasets mint `options` keys that request each reshape step. A refusal points at the key
# whose value it refuses.
_GROUPBY_POINTER = "/options/groupby"
_COLUMNS_POINTER = "/options/columns"
_PIVOT_POINTER = "/options/pivot"


class RollupNotSummableError(MaterialisationError):
    """422, code="rollup_not_summable". Raised by `groupby_sum` when `axis`
    fails one of its preconditions. `axis` names the column; `precondition`
    is one of `"facet_axis"`, `"not_declared_summable"`,
    `"unique_input_grain"`, `"facet_homogeneity"`, `"parent_child_overlap"`,
    `"missing_members"`, or `"finite_total"` -- the message always names
    both.
    """

    code = "rollup_not_summable"
    status = 422

    def __init__(self, *args: object, axis: str, precondition: str) -> None:
        # The structured issue lets clients branch on the axis without parsing the detail.
        super().__init__(
            *args,
            pointer=_GROUPBY_POINTER,
            issues=(ProblemIssue(pointer=_GROUPBY_POINTER, axis=axis),),
        )
        self.axis = axis
        self.precondition = precondition


class ColumnNotDroppableError(MaterialisationError):
    """422, code="column_not_droppable". Raised by `select_columns` when a
    caller's `keep` list omits one or more grain, unit, or provenance
    columns that are still present on the table. `columns` names every
    such column, in `table.columns` order.
    """

    code = "column_not_droppable"
    status = 422

    def __init__(self, *args: object, columns: tuple[str, ...]) -> None:
        super().__init__(
            *args,
            issues=tuple(
                ProblemIssue(pointer=_COLUMNS_POINTER, member=column) for column in columns
            ),
        )
        self.columns = columns


class ColumnUnknownError(MaterialisationError):
    """422, code="column_unknown". Raised by `select_columns` when a
    caller's `keep` list names a column the table does not carry,
    including an axis a prior `groupby_sum` call already collapsed. Also
    raised by `pivot` for a missing index/columns/value column, and by
    `parse_column_tokens` for a token that neither selects a column nor
    parses as a qualified enrichment token. `alternatives` names every
    qualified enrichment token that would parse on this table, each as an
    `Alternative(ref=token, label=token)` (an enrichment token has no
    separate label). It is populated only by `parse_column_tokens`. Every
    other raise site leaves it `None`, through the base
    `MaterialisationError.alternatives`, which this class leaves unshadowed.
    """

    code = "column_unknown"
    status = 422

    def __init__(self, *args: object, column: str, alternatives: tuple[str, ...] = ()) -> None:
        super().__init__(
            *args,
            pointer=_COLUMNS_POINTER,
            alternatives=[Alternative(ref=token, label=token) for token in alternatives] or None,
        )
        self.column = column


class PivotNotUniqueError(MaterialisationError):
    """422, code="pivot_not_unique". Raised by `pivot` for either of two
    uniqueness failures: an `(index, columns)` pair repeating across more
    than one input row (pivoting would otherwise silently pick one value
    and drop the rest), or distinct `columns` values that stringify to the
    same output column name (pivoting would otherwise silently produce a
    `Table` with duplicate column names holding different values).
    `duplicate_keys` names the offending group per collision -- a
    two-tuple `(index_value, columns_value)` for the first failure, or the
    tuple of colliding raw `columns` values (prefixed with `index` itself
    when one of them collides with the retained index column name) for the
    second -- capped at `_MAX_NAMED_DUPLICATE_KEYS`; `truncated` is `True`
    when more existed than the cap could name. `varying` names the columns
    that differ inside duplicate key groups, with their frame-wide distinct
    counts, as `PivotAnalysis.varying` does; it is empty for a
    stringification collision.
    """

    code = "pivot_not_unique"
    status = 422

    def __init__(
        self,
        *args: object,
        duplicate_keys: tuple[tuple[object, ...], ...],
        truncated: bool,
        varying: tuple[tuple[str, int], ...] = (),
    ) -> None:
        # One issue per duplicate key, then one per column varying inside the key groups.
        super().__init__(
            *args,
            issues=(
                *(ProblemIssue(pointer=_PIVOT_POINTER, member=str(key)) for key in duplicate_keys),
                *(ProblemIssue(pointer=_PIVOT_POINTER, axis=column) for column, _ in varying),
            ),
        )
        self.duplicate_keys = duplicate_keys
        self.truncated = truncated
        self.varying = varying


class PivotOverColumnCapError(MaterialisationError):
    """422, code="pivot_over_column_cap". Raised by `pivot` when the
    pivot's output width (the index column plus one column per distinct
    `columns` value) exceeds the caller-injected `column_cap`. This is the
    reshape's own output shape, checked before any format writer runs,
    against a cap the caller reads from the target format (this module
    never imports `formats.py` to learn it) -- distinct from
    `formats.excel_limit`.
    """

    code = "pivot_over_column_cap"
    status = 422

    def __init__(self, *args: object, count: int, cap: int) -> None:
        super().__init__(*args, pointer=_PIVOT_POINTER)
        self.count = count
        self.cap = cap


class PivotOverCellCapError(MaterialisationError):
    """422, code="pivot_over_cell_cap". Raised by `pivot` when the pivot's
    total output cell count (the number of distinct `index` values times
    the output width) exceeds the caller-injected `cell_cap`. Distinct
    from `pivot_over_column_cap`, which bounds only the output's width: a
    table with a narrow, capped width can still carry an unbounded number
    of cells once its `index` axis is wide too -- csv and parquet report no
    column cap at all. Checked immediately after the column-cap check,
    against a cap the caller reads from its own deployment configuration
    (this module never imports `datasets_api.config` to learn it).
    """

    code = "pivot_over_cell_cap"
    status = 422

    def __init__(self, *args: object, count: int, cap: int) -> None:
        super().__init__(*args, pointer=_PIVOT_POINTER)
        self.count = count
        self.cap = cap


class PivotNotHomogeneousError(MaterialisationError):
    """422, code="pivot_not_homogeneous". Raised by `pivot` when a column
    `pivot` is about to drop -- one of `classification.non_droppable_columns`
    present on the table but not `index`/`columns`/`value_column` -- carries
    more than one distinct value within a single `index` group. Pivoting
    would otherwise silently blend those rows together under one output
    row without deciding which value to keep. A dropped column outside
    `classification.non_droppable_columns` never trips this check:
    it is expected to vary with the pivot key by construction, and is
    simply dropped like any other unselected column (see `pivot`'s own
    docstring). `offenders` names each failing `(column, index_value,
    sorted_distinct_values)` tuple, capped at `_MAX_NAMED_DUPLICATE_KEYS`;
    `truncated` is `True` when more existed than the cap could name --
    mirrors `PivotNotUniqueError`'s own offender/truncation shape.
    """

    code = "pivot_not_homogeneous"
    status = 422

    def __init__(
        self,
        *args: object,
        offenders: tuple[tuple[str, object, tuple[object, ...]], ...],
        truncated: bool,
    ) -> None:
        super().__init__(
            *args,
            issues=tuple(
                ProblemIssue(pointer=_PIVOT_POINTER, member=f"{column}={index_value!r}")
                for column, index_value, _values in offenders
            ),
        )
        self.offenders = offenders
        self.truncated = truncated


RESHAPE_CODE_CATALOGUE: CodeCatalogue = {
    "rollup_not_summable": (
        "Rollup Not Summable",
        "This groupby axis cannot be summed under the current preconditions.",
    ),
    "column_not_droppable": (
        "Column Not Droppable",
        "This column is grain, unit, or provenance and cannot be dropped.",
    ),
    "column_unknown": (
        "Column Unknown",
        "This column name is not present on the table.",
    ),
    "pivot_not_unique": (
        "Pivot Not Unique",
        "The pivot's index and columns pair is not unique across the input.",
    ),
    "pivot_over_column_cap": (
        "Pivot Over Column Cap",
        "The pivot's output width exceeds the target format's column cap.",
    ),
    "pivot_over_cell_cap": (
        "Pivot Over Cell Cap",
        "The pivot's total output cell count exceeds this deployment's cap.",
    ),
    "pivot_not_homogeneous": (
        "Pivot Not Homogeneous",
        "The pivot would merge rows that differ on a grain, unit, or provenance column it drops.",
    ),
    "enrichment_unavailable": (
        "Enrichment Unavailable",
        "A name, code, or requested enrichment column has no value for some rows: the "
        "graph names nothing for the value, the name read failed, or two codes of the "
        "column's code system share its label. Those cells are left empty rather than "
        "fabricated. measurementMethod_label holds only a name that differs from the "
        "method's dcid, and its empty cells never raise this warning.",
    ),
}
"""This module's own code-catalogue fragment. A later integration can merge
this into the served catalogue alongside `MATERIALISE_CODE_CATALOGUE`,
`FORMATS_CODE_CATALOGUE`, and `BUNDLE_CODE_CATALOGUE`."""


class EnrichmentUnavailableWarning(MaterialisationWarning):
    """`enrichment_unavailable`'s own wider warning shape, one entry per
    affected `(entity_column, field)` request. `column` is that request's
    own output column name (`{entity_column}_{field}`), and `count` is how
    many rows left that column's cell empty.
    `datasets_api.manifest.reshape_warning_models` reads both to name the
    affected column and count in the wire `message` it composes.
    """

    column: str
    count: int


def enrichment_unavailable_warning(*, column: str, count: int) -> EnrichmentUnavailableWarning:
    """Builds one `enrichment_unavailable` warning naming `column` (the
    request's own `f"{entity_column}_{field}"` output name) and `count`
    (how many rows left that column's cell empty). Public: `_fill_lookup_columns`
    is the whole-table caller, and `datasets_api.render._stream_bare_pivot`
    is a second caller that never builds a `Table` for `_fill_lookup_columns`
    to scan -- a streamed pivot's enrichment request only ever needs this
    warning (`pivot`'s own docstring: the enrichment column itself never
    survives into a pivot's output), gathered from the distinct refs and
    per-ref row counts it collects during its own one pass instead.
    """
    _title, detail = RESHAPE_CODE_CATALOGUE["enrichment_unavailable"]
    return {
        "code": "enrichment_unavailable",
        "severity": "warn",
        "message": detail,
        "column": column,
        "count": count,
    }


def _cell_sort_key(cell: object) -> tuple:
    """A lightweight local collation for reshape output row order --
    independent of the grammar's own `cell_sort_key` (imported nowhere in
    this module). Once a table has been reshaped it is no longer a `Frame`,
    and no reshape output's row order carries identity meaning (see the
    module docstring). `None` sorts first; every other cell sorts by its
    own string form.
    """
    if cell is None:
        return (0, "")
    return (1, str(cell))


def _first_ancestor_in(
    hierarchy: Mapping[object, object], start: object, candidates: frozenset[object]
) -> object | None:
    """Return the first ancestor of `start` in `candidates`, excluding
    `start`, or `None`. The visited set stops cycles and self-loops. Reusing
    `candidates` avoids a copy for every value in one group.
    """
    visited: set[object] = set()
    current = start
    while current in hierarchy:
        parent = hierarchy[current]
        if parent != start and parent in candidates:
            return parent
        if parent in visited:
            return None
        visited.add(parent)
        current = parent
    return None


@dataclass(frozen=True)
class ExpectedPopulation:
    """Graph-declared population for a `groupby_sum` axis, scoped by named
    columns. `columns` includes the collapsed axis and may include grouping
    columns that scope members to each output group. `members` includes
    declared members with no input row, allowing absent observations to be
    distinguished from undeclared entities.

    Construction requires unique, non-empty column names and one value per
    column in each member. `groupby_sum` checks that these columns apply to
    its axis and grouping keys.
    """

    columns: tuple[str, ...]
    members: frozenset[tuple[object, ...]]

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("ExpectedPopulation.columns must be non-empty")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError(
                f"ExpectedPopulation.columns must not repeat a name, got {self.columns!r}"
            )
        width = len(self.columns)
        for member in self.members:
            if len(member) != width:
                raise ValueError(
                    f"ExpectedPopulation member {member!r} does not have one value per "
                    f"column in {self.columns!r}"
                )


def _estimated_total(column: str, values: Set[object]) -> str | None:
    """The `measurementMethod` a total carries when the rows it sums hold a series' method
    and the estimated method continuing it (`facets.continuation`): the estimated one
    (`method_status.method_status_classes`), since a total containing an estimate is an
    estimate. `None` for another column, or for any other mix, such as no estimated method,
    two, or two other methods beside one, which stays refused (`facet_homogeneity`)."""
    if column != _MEASUREMENT_METHOD_COLUMN or len(values) != 2:
        return None
    classes = method_status_classes()
    estimated = {v for v in values if isinstance(v, str) and classes.get(v) == "estimated"}
    return next(iter(estimated)) if len(estimated) == 1 else None


def groupby_sum(
    table: TableLike,
    *,
    axis: str,
    classification: ColumnClassification,
    summable_axes: frozenset[str],
    hierarchy: Mapping[object, object] | None = None,
    missing_declared_as_zero: bool = False,
    expected_population: ExpectedPopulation | None = None,
    value_column: str = _VALUE_COLUMN,
) -> Table:
    """Sum `value_column` by the remaining grain columns. Facet columns and
    `status` pass through and must be homogeneous within each output group, except that a
    group mixing one estimated `measurementMethod` with other methods sums to a total
    carrying the estimated one (`_estimated_total`).
    Parent-child overlap in `hierarchy` refuses a group that would double-count
    an ancestor.

    `expected_population.columns` must include `axis` and may include other
    grouping columns. Every output group must match a population key, even when
    missing members count as zero. The population detects suppressed members
    absent from the table. Missing or null axis members raise
    `rollup_not_summable` unless `missing_declared_as_zero=True`; then both
    count as zero. Refusal reports name at most 10 absent members.

    `RollupNotSummableError` checks in order: facet axis, missing or
    non-summable axis, duplicate input grain, inhomogeneous pass-through
    columns, ancestor overlap, missing members, and non-finite totals. A
    missing `value_column` raises `ColumnUnknownError` before row processing.
    Invalid population columns or undescribed group keys raise `ValueError`.
    """
    if axis in classification.facet_axis_columns:
        raise RollupNotSummableError(
            f"groupby axis {axis!r} is a facet axis and cannot be collapsed by summing",
            axis=axis,
            precondition="facet_axis",
        )
    if axis not in table.columns or axis not in summable_axes:
        raise RollupNotSummableError(
            f"groupby axis {axis!r} is not declared summable",
            axis=axis,
            precondition="not_declared_summable",
        )

    columns = table.columns
    if value_column not in columns:
        raise ColumnUnknownError(
            f"column {value_column!r} is not present on this table", column=value_column
        )
    axis_pos = columns.index(axis)
    value_pos = columns.index(value_column)
    hierarchy = hierarchy or {}

    full_grain_positions = [i for i in range(len(columns)) if i != value_pos]
    seen_full_grain: set[tuple] = set()
    for row in table.rows:
        key = tuple(row[i] for i in full_grain_positions)
        if key in seen_full_grain:
            raise RollupNotSummableError(
                f"groupby axis {axis!r} refused: input rows are not uniquely keyed by "
                f"their full grain (excluding {value_column!r})",
                axis=axis,
                precondition="unique_input_grain",
            )
        seen_full_grain.add(key)

    passthrough_positions = [
        i
        for i, c in enumerate(columns)
        if c in classification.facet_axis_columns or c == _STATUS_COLUMN
    ]
    grouping_positions = [
        i
        for i in range(len(columns))
        if i != axis_pos and i != value_pos and i not in passthrough_positions
    ]

    population_other_key_positions: tuple[int, ...] = ()
    population_index: dict[tuple, list[object]] = {}
    if expected_population is not None:
        if axis not in expected_population.columns:
            raise ValueError(
                f"expected_population.columns {expected_population.columns!r} must include "
                f"the collapsed axis {axis!r}"
            )
        grouping_column_names = {columns[i]: i for i in grouping_positions}
        for name in expected_population.columns:
            if name != axis and name not in grouping_column_names:
                raise ValueError(
                    f"expected_population column {name!r} is not a grouping column of this "
                    f"groupby_sum call (axis={axis!r}); grouping columns here are "
                    f"{sorted(grouping_column_names)!r}"
                )
        population_axis_pos = expected_population.columns.index(axis)
        other_population_columns = tuple(c for c in expected_population.columns if c != axis)
        other_member_positions = tuple(
            expected_population.columns.index(c) for c in other_population_columns
        )
        # Position of each kept column WITHIN a group's own key tuple
        # (`grouping_positions` order), so a group's key can be projected
        # onto the same columns without re-deriving them per group.
        population_other_key_positions = tuple(
            grouping_positions.index(grouping_column_names[c]) for c in other_population_columns
        )
        for member in expected_population.members:
            other_values = tuple(member[i] for i in other_member_positions)
            population_index.setdefault(other_values, []).append(member[population_axis_pos])

    groups: dict[tuple, list[tuple]] = {}
    group_order: list[tuple] = []
    for row in table.rows:
        key = tuple(row[i] for i in grouping_positions)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(row)

    estimated_totals: dict[tuple, dict[int, str]] = {}
    for key in group_order:
        rows_in_group = groups[key]
        for pos in passthrough_positions:
            values = {row[pos] for row in rows_in_group}
            if len(values) > 1 and (method := _estimated_total(columns[pos], values)) is not None:
                estimated_totals.setdefault(key, {})[pos] = method
            elif len(values) > 1:
                raise RollupNotSummableError(
                    f"groupby axis {axis!r} refused: column {columns[pos]!r} is not "
                    "homogeneous across the rows this axis would collapse together "
                    f"(values {sorted(values, key=_cell_sort_key)!r})",
                    axis=axis,
                    precondition="facet_homogeneity",
                )
        if hierarchy:
            # Built ONCE per group and reused across every value's walk below
            # -- `_first_ancestor_in` excludes `start` from a match inline,
            # so this same frozenset is passed as-is to every call rather
            # than a fresh `axis_values - {value}` copy built per value.
            axis_values = frozenset(
                row[axis_pos] for row in rows_in_group if row[axis_pos] is not None
            )
            for value in axis_values:
                ancestor = _first_ancestor_in(hierarchy, value, axis_values)
                if ancestor is not None:
                    raise RollupNotSummableError(
                        f"groupby axis {axis!r} refused: {ancestor!r} and {value!r} are both "
                        "present in the same slice, and one is an ancestor of the other in "
                        "the declared hierarchy",
                        axis=axis,
                        precondition="parent_child_overlap",
                    )

    if expected_population is not None and other_population_columns:
        # Runs regardless of `missing_declared_as_zero`: a row group the
        # population has no member for at all is not a missingness question
        # -- the population simply does not describe this table -- so it is
        # refused here even when absent members elsewhere count as zero.
        for key in group_order:
            group_key_values = tuple(key[i] for i in population_other_key_positions)
            if group_key_values not in population_index:
                group_key = tuple(zip(other_population_columns, group_key_values, strict=True))
                raise ValueError(
                    f"expected_population has no member for group {group_key!r} on its kept "
                    f"columns {other_population_columns!r} -- every row group in this table "
                    "must be described by the expected population"
                )

    if not missing_declared_as_zero:
        for row in table.rows:
            if row[axis_pos] is None:
                raise RollupNotSummableError(
                    f"groupby axis {axis!r} refused: a null/suppressed member is present "
                    "and the source has not declared missing members as zero",
                    axis=axis,
                    precondition="missing_members",
                )
        if expected_population is not None:
            for key in group_order:
                group_key_values = tuple(key[i] for i in population_other_key_positions)
                expected = frozenset(population_index.get(group_key_values, ()))
                observed = {row[axis_pos] for row in groups[key] if row[axis_pos] is not None}
                absent = expected - observed
                if absent:
                    named = sorted(absent, key=_cell_sort_key)
                    truncated = len(named) > _MAX_NAMED_DUPLICATE_KEYS
                    named = named[:_MAX_NAMED_DUPLICATE_KEYS]
                    group_key = tuple(zip(other_population_columns, group_key_values, strict=True))
                    raise RollupNotSummableError(
                        f"groupby axis {axis!r} refused: expected member(s) {tuple(named)!r} "
                        f"never appear in this slice for group {group_key!r}"
                        f"{' (truncated)' if truncated else ''}, and "
                        "the source has not declared missing members as zero",
                        axis=axis,
                        precondition="missing_members",
                    )

    remaining_positions = [i for i in range(len(columns)) if i != axis_pos]
    value_pos_in_remaining = remaining_positions.index(value_pos)
    output_columns = tuple(columns[i] for i in remaining_positions)

    output_rows: list[tuple] = []
    for key in group_order:
        rows_in_group = groups[key]
        total = sum(row[value_pos] for row in rows_in_group)
        if not math.isfinite(total):
            raise RollupNotSummableError(
                f"groupby axis {axis!r} refused: the summed total for this slice is not a "
                f"finite number ({total!r})",
                axis=axis,
                precondition="finite_total",
            )
        representative = [rows_in_group[0][i] for i in remaining_positions]
        representative[value_pos_in_remaining] = total
        for pos, method in estimated_totals.get(key, {}).items():
            representative[remaining_positions.index(pos)] = method
        output_rows.append(tuple(representative))
    output_rows.sort(key=lambda r: tuple(_cell_sort_key(c) for c in r))

    return Table(columns=output_columns, rows=tuple(output_rows))


class _GroupAccumulator:
    """Per-group running sum, representative row, and distinct passthrough
    and axis values. The sets support homogeneity, hierarchy, and population
    checks; their size follows distinct group values rather than row count.
    """

    __slots__ = ("axis_values", "observed_axis_values", "passthrough", "representative", "total")

    def __init__(self) -> None:
        self.total: float = 0.0
        self.representative: list[object] | None = None
        self.passthrough: dict[int, set[object]] = {}
        self.axis_values: set[object] = set()
        self.observed_axis_values: set[object] = set()


@dataclass(frozen=True)
class GroupbySumChunksOutcome:
    """Exactly one of `table` and `violation` is set. The fold stores the
    whole-table refusal so callers can run higher-precedence checks after
    scanning the chunks.
    """

    table: Table | None
    violation: RollupNotSummableError | ValueError | None


@dataclass(frozen=True)
class _GroupbySumFoldPlan:
    """Every fact `fold_groupby_sum_chunks` derives once, before any row is
    folded: the column positions the fold reads by index, and (when
    `expected_population` is given) the population lookup its membership
    check reads from. Built by `_plan_groupby_sum_fold`.
    """

    axis: str
    value_column: str
    missing_declared_as_zero: bool
    hierarchy: Mapping[object, object]
    columns: tuple[str, ...]
    axis_pos: int
    value_pos: int
    full_grain_positions: tuple[int, ...]
    passthrough_positions: tuple[int, ...]
    grouping_positions: tuple[int, ...]
    remaining_positions: tuple[int, ...]
    value_pos_in_remaining: int
    output_columns: tuple[str, ...]
    has_expected_population: bool
    population_other_key_positions: tuple[int, ...]
    population_index: dict[tuple, list[object]]
    other_population_columns: tuple[str, ...]


def _plan_groupby_sum_fold(
    columns: Sequence[str],
    *,
    axis: str,
    classification: ColumnClassification,
    summable_axes: frozenset[str],
    hierarchy: Mapping[object, object] | None,
    missing_declared_as_zero: bool,
    value_column: str,
    expected_population: ExpectedPopulation | None,
) -> _GroupbySumFoldPlan:
    """Validates `axis`/`value_column` against `columns`, the two
    row-content-independent preconditions `fold_groupby_sum_chunks` checks
    before any row is read (`not_declared_summable`: `axis` absent from
    `columns` or not in `summable_axes`; a missing `value_column`), and
    derives every column position the fold needs.

    `expected_population`, when given, must name `axis` among its own
    columns and no column outside this call's grouping columns (raises
    `ValueError` otherwise). `population_index` is then
    `expected_population.members` keyed by each member's non-axis column
    values, with every axis member sharing a key collected into that key's
    own list: what the membership check in `_resolve_groupby_sum_outcome`
    reads to tell "this member never appeared" from "this member was never
    expected".
    """
    if axis not in columns or axis not in summable_axes:
        raise RollupNotSummableError(
            f"groupby axis {axis!r} is not declared summable",
            axis=axis,
            precondition="not_declared_summable",
        )
    if value_column not in columns:
        raise ColumnUnknownError(
            f"column {value_column!r} is not present on this table", column=value_column
        )

    axis_pos = columns.index(axis)
    value_pos = columns.index(value_column)

    full_grain_positions = tuple(i for i in range(len(columns)) if i != value_pos)
    passthrough_positions = tuple(
        i
        for i, c in enumerate(columns)
        if c in classification.facet_axis_columns or c == _STATUS_COLUMN
    )
    grouping_positions = tuple(
        i
        for i in range(len(columns))
        if i != axis_pos and i != value_pos and i not in passthrough_positions
    )
    remaining_positions = tuple(i for i in range(len(columns)) if i != axis_pos)
    value_pos_in_remaining = remaining_positions.index(value_pos)
    output_columns = tuple(columns[i] for i in remaining_positions)

    population_other_key_positions: tuple[int, ...] = ()
    population_index: dict[tuple, list[object]] = {}
    other_population_columns: tuple[str, ...] = ()
    if expected_population is not None:
        if axis not in expected_population.columns:
            raise ValueError(
                f"expected_population.columns {expected_population.columns!r} must include "
                f"the collapsed axis {axis!r}"
            )
        grouping_column_names = {columns[i]: i for i in grouping_positions}
        for name in expected_population.columns:
            if name != axis and name not in grouping_column_names:
                raise ValueError(
                    f"expected_population column {name!r} is not a grouping column of this "
                    f"groupby_sum call (axis={axis!r}); grouping columns here are "
                    f"{sorted(grouping_column_names)!r}"
                )
        population_axis_pos = expected_population.columns.index(axis)
        other_population_columns = tuple(c for c in expected_population.columns if c != axis)
        other_member_positions = tuple(
            expected_population.columns.index(c) for c in other_population_columns
        )
        population_other_key_positions = tuple(
            grouping_positions.index(grouping_column_names[c]) for c in other_population_columns
        )
        for member in expected_population.members:
            other_values = tuple(member[i] for i in other_member_positions)
            population_index.setdefault(other_values, []).append(member[population_axis_pos])

    return _GroupbySumFoldPlan(
        axis=axis,
        value_column=value_column,
        missing_declared_as_zero=missing_declared_as_zero,
        hierarchy=hierarchy or {},
        columns=tuple(columns),
        axis_pos=axis_pos,
        value_pos=value_pos,
        full_grain_positions=full_grain_positions,
        passthrough_positions=passthrough_positions,
        grouping_positions=grouping_positions,
        remaining_positions=remaining_positions,
        value_pos_in_remaining=value_pos_in_remaining,
        output_columns=output_columns,
        has_expected_population=expected_population is not None,
        population_other_key_positions=population_other_key_positions,
        population_index=population_index,
        other_population_columns=other_population_columns,
    )


class _GroupbySumFold:
    """One accumulator per output group plus the prior full-grain key.
    Row-level violations are recorded during the fold; group checks resolve
    after all rows arrive in `_resolve_groupby_sum_outcome`.
    """

    def __init__(self, plan: _GroupbySumFoldPlan) -> None:
        self._plan = plan
        self.groups: dict[tuple, _GroupAccumulator] = {}
        self.group_order: list[tuple] = []
        self._previous_full_grain: tuple | None = None
        self.unique_violation: RollupNotSummableError | None = None
        self.population_violation: ValueError | None = None
        self.null_member_violation: RollupNotSummableError | None = None

    def process_row(self, row: tuple[object, ...]) -> None:
        """Fold one row. Chunks must follow non-decreasing `row_sort_key`
        order (all columns except `value_column`), as materialised frame
        chunks do. That order makes duplicate full-grain rows adjacent; a
        descending key raises `MaterialisationUsageError`. Other group checks
        resolve after folding and do not depend on row order.
        """
        plan = self._plan
        full_key = tuple(_cell_sort_key(row[i]) for i in plan.full_grain_positions)
        if self._previous_full_grain is not None:
            if full_key == self._previous_full_grain:
                if self.unique_violation is None:
                    self.unique_violation = RollupNotSummableError(
                        f"groupby axis {plan.axis!r} refused: input rows are not uniquely "
                        f"keyed by their full grain (excluding {plan.value_column!r})",
                        axis=plan.axis,
                        precondition="unique_input_grain",
                    )
            elif full_key < self._previous_full_grain:
                raise MaterialisationUsageError(
                    "groupby_sum_chunks() requires chunks in non-decreasing row_sort_key "
                    "order (every column but value_column); this precondition is what a "
                    "genuine materialised frame chunk stream always satisfies -- a caller "
                    "supplying an out-of-order chunk stream has a caller-shape bug, not a "
                    "malformed request"
                )
        self._previous_full_grain = full_key

        key = tuple(row[i] for i in plan.grouping_positions)
        group = self.groups.get(key)
        if group is None:
            group = _GroupAccumulator()
            group.representative = [row[i] for i in plan.remaining_positions]
            self.groups[key] = group
            self.group_order.append(key)
            if (
                plan.has_expected_population
                and plan.other_population_columns
                and self.population_violation is None
            ):
                group_key_values = tuple(key[i] for i in plan.population_other_key_positions)
                if group_key_values not in plan.population_index:
                    group_key = tuple(
                        zip(plan.other_population_columns, group_key_values, strict=True)
                    )
                    self.population_violation = ValueError(
                        f"expected_population has no member for group {group_key!r} on its "
                        f"kept columns {plan.other_population_columns!r} -- every row group in "
                        "this table must be described by the expected population"
                    )

        for pos in plan.passthrough_positions:
            group.passthrough.setdefault(pos, set()).add(row[pos])

        axis_value = row[plan.axis_pos]
        if plan.hierarchy and axis_value is not None:
            group.axis_values.add(axis_value)

        if (
            not plan.missing_declared_as_zero
            and axis_value is None
            and self.null_member_violation is None
        ):
            self.null_member_violation = RollupNotSummableError(
                f"groupby axis {plan.axis!r} refused: a null/suppressed member is present "
                "and the source has not declared missing members as zero",
                axis=plan.axis,
                precondition="missing_members",
            )

        if axis_value is not None:
            group.observed_axis_values.add(axis_value)

        assert group.representative is not None
        cell_value = row[plan.value_pos]
        assert isinstance(cell_value, int | float), (
            f"groupby_sum_chunks() found a non-numeric {plan.value_column!r} cell: {cell_value!r}"
        )
        group.total += cell_value


def _resolve_groupby_sum_outcome(
    fold: _GroupbySumFold,
    plan: _GroupbySumFoldPlan,
    *,
    expected_population: ExpectedPopulation | None,
) -> GroupbySumChunksOutcome:
    """Resolve the same violation as whole-table `groupby_sum` in this order:
    unique input grain; per-group facet homogeneity, then hierarchy overlap;
    invalid population-group key; null members; absent population members;
    non-finite total. Groups are checked in first-appearance order. Return
    exactly one of `table` or `violation`.
    """
    violation: RollupNotSummableError | ValueError | None = fold.unique_violation
    estimated_totals: dict[tuple, dict[int, str]] = {}

    if violation is None:
        for key in fold.group_order:
            group = fold.groups[key]
            facet_violation: RollupNotSummableError | None = None
            for pos in plan.passthrough_positions:
                values = group.passthrough.get(pos, set())
                if (
                    len(values) > 1
                    and (method := _estimated_total(plan.columns[pos], values)) is not None
                ):
                    estimated_totals.setdefault(key, {})[pos] = method
                elif len(values) > 1:
                    facet_violation = RollupNotSummableError(
                        f"groupby axis {plan.axis!r} refused: column {plan.columns[pos]!r} is "
                        "not homogeneous across the rows this axis would collapse together "
                        f"(values {sorted(values, key=_cell_sort_key)!r})",
                        axis=plan.axis,
                        precondition="facet_homogeneity",
                    )
                    break
            if facet_violation is not None:
                violation = facet_violation
                break
            if plan.hierarchy:
                axis_values = frozenset(group.axis_values)
                hierarchy_violation: RollupNotSummableError | None = None
                for value in axis_values:
                    ancestor = _first_ancestor_in(plan.hierarchy, value, axis_values)
                    if ancestor is not None:
                        hierarchy_violation = RollupNotSummableError(
                            f"groupby axis {plan.axis!r} refused: {ancestor!r} and {value!r} "
                            "are both present in the same slice, and one is an ancestor of "
                            "the other in the declared hierarchy",
                            axis=plan.axis,
                            precondition="parent_child_overlap",
                        )
                        break
                if hierarchy_violation is not None:
                    violation = hierarchy_violation
                    break

    if violation is None:
        violation = fold.population_violation

    if violation is None:
        violation = fold.null_member_violation

    if violation is None and not plan.missing_declared_as_zero and expected_population is not None:
        for key in fold.group_order:
            group_key_values = tuple(key[i] for i in plan.population_other_key_positions)
            expected = frozenset(plan.population_index.get(group_key_values, ()))
            observed = fold.groups[key].observed_axis_values
            absent = expected - observed
            if absent:
                named = sorted(absent, key=_cell_sort_key)
                truncated = len(named) > _MAX_NAMED_DUPLICATE_KEYS
                named = named[:_MAX_NAMED_DUPLICATE_KEYS]
                group_key = tuple(zip(plan.other_population_columns, group_key_values, strict=True))
                violation = RollupNotSummableError(
                    f"groupby axis {plan.axis!r} refused: expected member(s) {tuple(named)!r} "
                    f"never appear in this slice for group {group_key!r}"
                    f"{' (truncated)' if truncated else ''}, and "
                    "the source has not declared missing members as zero",
                    axis=plan.axis,
                    precondition="missing_members",
                )
                break

    output_rows: list[tuple] = []
    if violation is None:
        for key in fold.group_order:
            group = fold.groups[key]
            if not math.isfinite(group.total):
                violation = RollupNotSummableError(
                    f"groupby axis {plan.axis!r} refused: the summed total for this slice is "
                    f"not a finite number ({group.total!r})",
                    axis=plan.axis,
                    precondition="finite_total",
                )
                break
            assert group.representative is not None
            representative = list(group.representative)
            representative[plan.value_pos_in_remaining] = group.total
            for pos, method in estimated_totals.get(key, {}).items():
                representative[plan.remaining_positions.index(pos)] = method
            output_rows.append(tuple(representative))

    if violation is not None:
        return GroupbySumChunksOutcome(table=None, violation=violation)

    output_rows.sort(key=lambda r: tuple(_cell_sort_key(c) for c in r))
    return GroupbySumChunksOutcome(
        table=Table(columns=plan.output_columns, rows=tuple(output_rows)), violation=None
    )


def fold_groupby_sum_chunks(
    chunks: Iterator[TableLike],
    *,
    axis: str,
    classification: ColumnClassification,
    summable_axes: frozenset[str],
    hierarchy: Mapping[object, object] | None = None,
    missing_declared_as_zero: bool = False,
    expected_population: ExpectedPopulation | None = None,
    value_column: str = _VALUE_COLUMN,
) -> GroupbySumChunksOutcome:
    """Fold chunks once, holding one accumulator per output group and one
    previous grain key. Content-dependent violations are stored in the same
    precedence order as `groupby_sum`; facet-axis and axis-declaration checks
    raise before reading rows. An external higher-precedence check can run
    before the wrapper raises a stored violation. Requires at least one chunk.
    """
    if axis in classification.facet_axis_columns:
        raise RollupNotSummableError(
            f"groupby axis {axis!r} is a facet axis and cannot be collapsed by summing",
            axis=axis,
            precondition="facet_axis",
        )

    try:
        first_chunk = next(chunks)
    except StopIteration:
        raise MaterialisationUsageError(
            "groupby_sum_chunks() requires at least one chunk"
        ) from None
    columns = first_chunk.columns

    plan = _plan_groupby_sum_fold(
        columns,
        axis=axis,
        classification=classification,
        summable_axes=summable_axes,
        hierarchy=hierarchy,
        missing_declared_as_zero=missing_declared_as_zero,
        value_column=value_column,
        expected_population=expected_population,
    )
    fold = _GroupbySumFold(plan)
    for chunk in (first_chunk, *chunks):
        for row in chunk.rows:
            fold.process_row(row)

    return _resolve_groupby_sum_outcome(fold, plan, expected_population=expected_population)


def groupby_sum_chunks(
    chunks: Iterator[TableLike],
    *,
    axis: str,
    classification: ColumnClassification,
    summable_axes: frozenset[str],
    hierarchy: Mapping[object, object] | None = None,
    missing_declared_as_zero: bool = False,
    expected_population: ExpectedPopulation | None = None,
    value_column: str = _VALUE_COLUMN,
) -> Table:
    """Stream `groupby_sum` with the same result and refusal order. This
    wrapper raises a stored violation. Call `fold_groupby_sum_chunks` when a
    caller must run a higher-precedence check after the scan but before the
    roll-up refusal is raised.
    """
    outcome = fold_groupby_sum_chunks(
        chunks,
        axis=axis,
        classification=classification,
        summable_axes=summable_axes,
        hierarchy=hierarchy,
        missing_declared_as_zero=missing_declared_as_zero,
        expected_population=expected_population,
        value_column=value_column,
    )
    if outcome.violation is not None:
        raise outcome.violation
    assert outcome.table is not None
    return outcome.table


def select_columns(
    table: TableLike, *, keep: Sequence[str], classification: ColumnClassification
) -> Table:
    """Select columns in `keep` order. Missing names raise `ColumnUnknownError`;
    omitted present grain, unit, or provenance columns raise
    `ColumnNotDroppableError` in table-column order.
    """
    present = set(table.columns)
    for name in keep:
        if name not in present:
            raise ColumnUnknownError(f"column {name!r} is not present on this table", column=name)

    keep_set = set(keep)
    missing_non_droppable = [
        c for c in table.columns if c in classification.non_droppable_columns and c not in keep_set
    ]
    if missing_non_droppable:
        columns = tuple(missing_non_droppable)
        raise ColumnNotDroppableError(
            f"column(s) {columns!r} are grain, unit, or provenance and cannot be dropped",
            columns=columns,
        )

    index_of = {name: i for i, name in enumerate(table.columns)}
    positions = [index_of[name] for name in keep]
    rows = tuple(tuple(row[i] for i in positions) for row in table.rows)
    return Table(columns=tuple(keep), rows=rows)


def _fill_lookup_columns(
    table: TableLike,
    *,
    requests: Sequence[tuple[str, str]],
    lookup: Mapping[str, Mapping[str, str]],
) -> tuple[Table, tuple[MaterialisationWarning, ...]]:
    """Append one `{entity_column}_{field}` column per request, in request
    order. Missing refs or fields become empty cells; each affected request
    gets an `enrichment_unavailable` warning with its empty-cell count.
    Callers validate entity columns, field names, and duplicate requests.
    """
    entity_positions = {
        entity_column: table.columns.index(entity_column) for entity_column, _field in requests
    }
    missing_counts: dict[tuple[str, str], int] = {}
    new_rows = []
    for row in table.rows:
        extra: list[object] = []
        for entity_column, field_name in requests:
            ref = row[entity_positions[entity_column]]
            entry = lookup.get(ref, {}) if isinstance(ref, str) else {}
            value = entry.get(field_name, "")
            if value == "":
                request = (entity_column, field_name)
                missing_counts[request] = missing_counts.get(request, 0) + 1
            extra.append(value)
        new_rows.append((*row, *extra))

    output_columns = (
        *table.columns,
        *(f"{entity_column}_{field_name}" for entity_column, field_name in requests),
    )
    warnings = tuple(
        enrichment_unavailable_warning(
            column=f"{entity_column}_{field_name}",
            count=missing_counts[(entity_column, field_name)],
        )
        for entity_column, field_name in requests
        if (entity_column, field_name) in missing_counts
    )
    return Table(columns=output_columns, rows=tuple(new_rows)), warnings


def _refuse_unknown_entity_columns(
    table: TableLike, *, requests: Sequence[tuple[str, str]], caller_name: str
) -> None:
    unknown_columns = sorted(
        {entity_column for entity_column, _field in requests if entity_column not in table.columns}
    )
    if unknown_columns:
        raise MaterialisationUsageError(
            f"{caller_name}() requires every request's entity_column to be present on "
            f"the table; table columns are {table.columns!r}, got unknown "
            f"column(s) {unknown_columns!r}"
        )


def _refuse_duplicate_requests(
    requests: Sequence[tuple[str, str]], *, caller_name: str, field_noun: str
) -> None:
    seen_requests: set[tuple[str, str]] = set()
    duplicates: list[tuple[str, str]] = []
    for request in requests:
        if request in seen_requests:
            duplicates.append(request)
        else:
            seen_requests.add(request)
    if duplicates:
        raise MaterialisationUsageError(
            f"{caller_name}() requires distinct (entity_column, {field_noun}) requests, "
            f"got duplicate(s) {duplicates!r}"
        )


def enrich_columns(
    table: TableLike,
    *,
    requests: Sequence[tuple[str, str]],
    lookup: Mapping[str, Mapping[str, str]],
) -> tuple[Table, tuple[MaterialisationWarning, ...]]:
    """Append `{entity_column}_{field}` columns in request order.
    Fields must be supported, entity columns must exist, and requests must
    be unique. Invalid input raises `MaterialisationUsageError` before rows
    are processed.
    """
    unsupported = sorted(
        {
            field_name
            for _entity_column, field_name in requests
            if field_name not in SUPPORTED_ENRICHMENTS
        }
    )
    if unsupported:
        raise MaterialisationUsageError(
            f"enrich_columns() requires fields drawn from {sorted(SUPPORTED_ENRICHMENTS)!r}, "
            f"got unsupported field(s) {unsupported!r}"
        )
    _refuse_unknown_entity_columns(table, requests=requests, caller_name="enrich_columns")
    _refuse_duplicate_requests(requests, caller_name="enrich_columns", field_noun="field")

    return _fill_lookup_columns(table, requests=requests, lookup=lookup)


def decode_columns(
    table: TableLike,
    *,
    requests: Sequence[tuple[str, str]],
    lookup: Mapping[str, Mapping[str, str]],
) -> tuple[Table, tuple[MaterialisationWarning, ...]]:
    """Append `{entity_column}_{code_system}` columns in request order,
    filling them from the caller's code-index lookup. The caller chooses the
    code systems per frame; entity columns must exist and requests must be
    unique, or `MaterialisationUsageError` is raised before row processing.

    Decode runs before groupby so a decoded StatVar axis survives collapse.
    The caller stores a declared-absent reading as an explicit sentinel.
    A declared code missing from the index stays absent from `lookup`, so it
    yields an empty cell and an `enrichment_unavailable` warning.
    """
    _refuse_unknown_entity_columns(table, requests=requests, caller_name="decode_columns")
    _refuse_duplicate_requests(requests, caller_name="decode_columns", field_noun="code_system")

    return _fill_lookup_columns(table, requests=requests, lookup=lookup)


@dataclass(frozen=True)
class PivotAnalysis:
    """Analysis facts for `pivot` before cells are built. `unique` fails for
    duplicate key pairs or stringified-name collisions; separate fields
    preserve each refusal's details and truncation status. `homogeneous`
    checks dropped non-droppable columns within each index. `width` and
    `cells` give output dimensions. `varying` lists non-key, non-value,
    non-status columns that differ inside duplicate-key groups, with their
    frame-wide distinct counts; it is empty for stringification-only collisions.
    """

    index: str
    columns: str
    unique: bool
    homogeneous: bool
    width: int
    cells: int
    varying: tuple[tuple[str, int], ...]
    duplicate_keys: tuple[tuple[object, ...], ...]
    duplicate_keys_truncated: bool
    offenders: tuple[tuple[str, object, tuple[object, ...]], ...]
    offenders_truncated: bool
    stringification_collisions: tuple[tuple[object, ...], ...]
    stringification_truncated: bool


def analyse_pivot(
    table: TableLike,
    *,
    index: str,
    columns: str,
    classification: ColumnClassification,
    value_column: str = _VALUE_COLUMN,
) -> PivotAnalysis:
    """Computes `PivotAnalysis` for a candidate `index`/`columns` pivot
    pair, assuming `index`, `columns`, and `value_column` are already
    known to be present on `table.columns`. A caller asking about a
    candidate pair already knows this. `pivot` checks it before calling
    this function, because a caller of `pivot` may not know it.

    Tracks a set of seen `(index, columns)` keys to detect duplicates,
    since this function builds no output table and needs no pivoted cell
    values. The homogeneity, duplicate-key, and stringification-collision
    bookkeeping run in one accumulation pass, so `pivot` can raise its
    refusals from this function's output unchanged.
    `varying`'s frame-wide distinct counts run a second, narrower pass
    restricted to rows whose key repeats. The large majority of rows,
    which belong to no duplicated key, stay out of memory for this pass.
    """
    index_pos = table.columns.index(index)
    columns_pos = table.columns.index(columns)

    # Each group's value set is capped as it accumulates, so it stays
    # bounded on a column with many distinct values per group.
    homogeneity_columns = tuple(
        c
        for c in table.columns
        if c in classification.non_droppable_columns and c not in (index, columns, value_column)
    )
    homogeneity_positions = tuple(table.columns.index(c) for c in homogeneity_columns)
    homogeneity_seen: dict[str, dict[object, set]] = {c: {} for c in homogeneity_columns}
    homogeneity_value_truncated = False

    # Tracks whether an `(index, columns)` key has already been seen, to
    # detect repeats.
    seen_keys: set[tuple[object, object]] = set()
    distinct_duplicate_keys: list[tuple[object, object]] = []
    seen_duplicate_keys: set[tuple[object, object]] = set()
    duplicate_keys_truncated = False
    pivot_values: list[object] = []
    seen_pivot_values: set[object] = set()
    index_values: list[object] = []
    seen_index_values: set[object] = set()

    for row in table.rows:
        index_value = row[index_pos]
        pivot_value = row[columns_pos]
        key = (index_value, pivot_value)
        if key in seen_keys:
            if key not in seen_duplicate_keys:
                seen_duplicate_keys.add(key)
                if len(distinct_duplicate_keys) < _MAX_NAMED_DUPLICATE_KEYS:
                    distinct_duplicate_keys.append(key)
                else:
                    duplicate_keys_truncated = True
        else:
            seen_keys.add(key)
        if pivot_value not in seen_pivot_values:
            seen_pivot_values.add(pivot_value)
            pivot_values.append(pivot_value)
        if index_value not in seen_index_values:
            seen_index_values.add(index_value)
            index_values.append(index_value)
        for column, pos in zip(homogeneity_columns, homogeneity_positions, strict=True):
            values = homogeneity_seen[column].setdefault(index_value, set())
            cell = row[pos]
            if cell not in values:
                if len(values) < _MAX_HOMOGENEITY_VALUES_PER_GROUP:
                    values.add(cell)
                else:
                    homogeneity_value_truncated = True

    found_offenders: list[tuple[str, object, tuple[object, ...]]] = []
    for column in homogeneity_columns:
        for index_value in index_values:
            values = homogeneity_seen[column].get(index_value, set())
            if len(values) > 1:
                found_offenders.append(
                    (column, index_value, tuple(sorted(values, key=_cell_sort_key)))
                )
    offenders_truncated = (
        len(found_offenders) > _MAX_NAMED_DUPLICATE_KEYS or homogeneity_value_truncated
    )
    offenders = tuple(found_offenders[:_MAX_NAMED_DUPLICATE_KEYS])

    pivot_column_names = tuple(v if isinstance(v, str) else str(v) for v in pivot_values)
    raw_values_by_name: dict[str, list[object]] = {}
    for raw, name in zip(pivot_values, pivot_column_names, strict=True):
        raw_values_by_name.setdefault(name, []).append(raw)
    name_collisions: list[tuple[object, ...]] = []
    for name, raws in raw_values_by_name.items():
        collides_with_index = name == index
        if len(raws) > 1 or collides_with_index:
            name_collisions.append((index, *raws) if collides_with_index else tuple(raws))
    stringification_truncated = len(name_collisions) > _MAX_NAMED_DUPLICATE_KEYS
    stringification_collisions = tuple(name_collisions[:_MAX_NAMED_DUPLICATE_KEYS])

    unique = not distinct_duplicate_keys and not stringification_collisions

    varying: tuple[tuple[str, int], ...] = ()
    if not unique and seen_duplicate_keys:
        candidate_columns = tuple(
            c for c in table.columns if c not in (index, columns, value_column, _STATUS_COLUMN)
        )
        candidate_positions = tuple(table.columns.index(c) for c in candidate_columns)
        first_value_by_key: dict[tuple[object, object], dict[str, object]] = {}
        differing_columns: set[str] = set()
        for row in table.rows:
            key = (row[index_pos], row[columns_pos])
            if key not in seen_duplicate_keys:
                continue
            first_values = first_value_by_key.setdefault(key, {})
            for column, pos in zip(candidate_columns, candidate_positions, strict=True):
                if column in differing_columns:
                    continue
                cell = row[pos]
                if column not in first_values:
                    first_values[column] = cell
                elif first_values[column] != cell:
                    differing_columns.add(column)
        if differing_columns:
            positions_by_column = dict(zip(candidate_columns, candidate_positions, strict=True))
            distinct_sets: dict[str, set] = {c: set() for c in differing_columns}
            for row in table.rows:
                for column in differing_columns:
                    distinct_sets[column].add(row[positions_by_column[column]])
            varying = tuple(
                (c, len(distinct_sets[c])) for c in candidate_columns if c in differing_columns
            )

    return PivotAnalysis(
        index=index,
        columns=columns,
        unique=unique,
        homogeneous=not offenders,
        width=1 + len(pivot_values),
        cells=len(index_values) * (1 + len(pivot_values)),
        varying=varying,
        duplicate_keys=tuple(distinct_duplicate_keys),
        duplicate_keys_truncated=duplicate_keys_truncated,
        offenders=offenders,
        offenders_truncated=offenders_truncated,
        stringification_collisions=stringification_collisions,
        stringification_truncated=stringification_truncated,
    )


def pivot(
    table: TableLike,
    *,
    index: str,
    columns: str,
    column_cap: int,
    cell_cap: int,
    classification: ColumnClassification,
    value_column: str = _VALUE_COLUMN,
) -> Table:
    """Pivot `value_column` from rows indexed by `index` into columns from
    distinct `columns` values. The output keeps only `index` and pivoted
    columns; select any enrichment that must survive as an index or pivot
    column. Pivoting follows column selection, so its new columns cannot be
    selected in the same plan.

    All three input columns must exist. Refusals run in order: duplicate
    keys, inhomogeneous dropped columns, width cap, cell cap, then
    stringification collisions. Output columns follow first-seen row order;
    absent `(index, columns)` pairs contain `None`.
    """
    for name in (index, columns, value_column):
        if name not in table.columns:
            raise ColumnUnknownError(f"column {name!r} is not present on this table", column=name)

    analysis = analyse_pivot(
        table,
        index=index,
        columns=columns,
        classification=classification,
        value_column=value_column,
    )

    if analysis.duplicate_keys:
        raise PivotNotUniqueError(
            f"pivot on index={index!r}, columns={columns!r} is not unique: duplicate "
            f"keys {analysis.duplicate_keys!r}"
            f"{' (truncated)' if analysis.duplicate_keys_truncated else ''}",
            duplicate_keys=analysis.duplicate_keys,
            truncated=analysis.duplicate_keys_truncated,
            varying=analysis.varying,
        )

    if analysis.offenders:
        raise PivotNotHomogeneousError(
            f"pivot on index={index!r} would merge rows that differ on column(s) it drops: "
            f"{analysis.offenders!r}{' (truncated)' if analysis.offenders_truncated else ''}",
            offenders=analysis.offenders,
            truncated=analysis.offenders_truncated,
        )

    if analysis.width > column_cap:
        raise PivotOverColumnCapError(
            f"pivot output would carry {analysis.width} columns, exceeding the format's "
            f"column cap of {column_cap}",
            count=analysis.width,
            cap=column_cap,
        )

    if analysis.cells > cell_cap:
        index_count = analysis.cells // analysis.width
        raise PivotOverCellCapError(
            f"pivot output would carry {analysis.cells} cells ({index_count} rows x "
            f"{analysis.width} columns), exceeding this deployment's cap of {cell_cap}",
            count=analysis.cells,
            cap=cell_cap,
        )

    if analysis.stringification_collisions:
        raise PivotNotUniqueError(
            f"pivot on index={index!r}, columns={columns!r} is not unique: distinct "
            f"columns values collide once stringified into the same output column name: "
            f"{analysis.stringification_collisions!r}"
            f"{' (truncated)' if analysis.stringification_truncated else ''}",
            duplicate_keys=analysis.stringification_collisions,
            truncated=analysis.stringification_truncated,
        )

    index_pos = table.columns.index(index)
    columns_pos = table.columns.index(columns)
    value_pos = table.columns.index(value_column)

    cell_by_key: dict[tuple[object, object], object] = {}
    pivot_values: list[object] = []
    seen_pivot_values: set[object] = set()
    index_values: list[object] = []
    seen_index_values: set[object] = set()
    for row in table.rows:
        index_value = row[index_pos]
        pivot_value = row[columns_pos]
        key = (index_value, pivot_value)
        if key not in cell_by_key:
            cell_by_key[key] = row[value_pos]
        if pivot_value not in seen_pivot_values:
            seen_pivot_values.add(pivot_value)
            pivot_values.append(pivot_value)
        if index_value not in seen_index_values:
            seen_index_values.add(index_value)
            index_values.append(index_value)

    pivot_column_names = tuple(v if isinstance(v, str) else str(v) for v in pivot_values)
    output_columns = (index, *pivot_column_names)
    output_rows = tuple(
        (
            index_value,
            *(cell_by_key.get((index_value, pv)) for pv in pivot_values),
        )
        for index_value in index_values
    )
    return Table(columns=output_columns, rows=output_rows)


class PivotFacts(Protocol):
    """Facts about one `(index, columns)` pair over the rows a pivot fold
    reads, known before the fold runs. `materialise.profile.PivotFeasibility`
    satisfies this as-is, without this module importing it.
    """

    @property
    def duplicate_keys(self) -> bool: ...

    @property
    def homogeneous(self) -> bool: ...

    @property
    def width(self) -> int: ...

    @property
    def cells(self) -> int: ...

    @property
    def varying(self) -> tuple[tuple[str, int], ...]: ...


@dataclass(frozen=True)
class PivotChunksOutcome:
    """Exactly one of `table` and `violation` is set. The fold stores
    refusals so callers can run higher-priority checks after scanning chunks.
    """

    table: Table | None
    violation: (
        PivotNotUniqueError
        | PivotOverColumnCapError
        | PivotOverCellCapError
        | PivotNotHomogeneousError
        | None
    )


def known_pivot_cap_violation(
    known: PivotFacts | None, *, column_cap: int, cell_cap: int
) -> PivotOverColumnCapError | PivotOverCellCapError | None:
    """The cap refusal `fold_pivot_chunks` returns before its scan for the facts `known`.

    `None` when there are no facts, when they report duplicate keys or a group that is not
    homogeneous (those refusals outrank the caps and need the scan), or when the pivot fits
    both caps.
    """
    if known is None or known.duplicate_keys or not known.homogeneous:
        return None
    return _pivot_cap_violation(
        width=known.width, cells=known.cells, column_cap=column_cap, cell_cap=cell_cap
    )


def fold_pivot_chunks(
    chunks: Iterator[TableLike],
    *,
    index: str,
    columns: str,
    column_cap: int,
    cell_cap: int,
    classification: ColumnClassification,
    value_column: str = _VALUE_COLUMN,
    known: PivotFacts | None = None,
) -> PivotChunksOutcome:
    """Fold chunks in one pass into distinct output-key state. The fold keeps
    the same refusal order as `pivot` and stores violations so callers can run
    higher-priority checks after the scan. Otherwise it returns the same table
    as whole-table `pivot`.

    Rows sharing an `index` may be separated by `row_sort_key`, so the fold
    reads the full stream before building rows. Memory scales with distinct
    `(index, columns)` keys, distinct axis values, and capped homogeneity
    samples.

    `known`, when given, must describe the rows `chunks` yields, as the stored
    output profile does for a frame pivoted without a collapsing groupby. When
    it reports unique keys and homogeneous groups, the column and cell caps are
    checked against its `width` and `cells` before any key is stored, which
    keeps the refusal order. A duplicate-key refusal carries its `varying`.
    Without `known` the fold finds `varying` itself, as `analyse_pivot` does:
    it keeps each key's first row of non-key columns and the distinct values
    of every such column, so memory grows with those columns as well.
    """
    first_chunk = next(chunks, None)
    if first_chunk is None:
        raise MaterialisationUsageError("pivot_chunks() requires at least one chunk")
    table_columns = first_chunk.columns
    for name in (index, columns, value_column):
        if name not in table_columns:
            raise ColumnUnknownError(f"column {name!r} is not present on this table", column=name)

    known_violation = known_pivot_cap_violation(known, column_cap=column_cap, cell_cap=cell_cap)
    if known_violation is not None:
        return PivotChunksOutcome(table=None, violation=known_violation)

    index_pos = table_columns.index(index)
    columns_pos = table_columns.index(columns)
    value_pos = table_columns.index(value_column)

    homogeneity_columns = tuple(
        c
        for c in table_columns
        if c in classification.non_droppable_columns and c not in (index, columns, value_column)
    )
    homogeneity_positions = tuple(table_columns.index(c) for c in homogeneity_columns)
    homogeneity_seen: dict[str, dict[object, set]] = {c: {} for c in homogeneity_columns}
    homogeneity_value_truncated = False

    distinct_duplicate_keys: list[tuple[object, object]] = []
    seen_duplicate_keys: set[tuple[object, object]] = set()
    duplicate_keys_truncated = False

    # Without stored facts, `varying` comes from the duplicate groups. `distinct_by_column` also
    # canonicalises each cell, so a key's first row costs a pointer per column.
    varying_columns = (
        tuple(c for c in table_columns if c not in (index, columns, value_column, _STATUS_COLUMN))
        if known is None
        else ()
    )
    varying_positions = tuple(table_columns.index(c) for c in varying_columns)
    distinct_by_column: dict[str, dict[object, object]] = {c: {} for c in varying_columns}
    first_cells_by_key: dict[tuple[object, object], tuple[object, ...]] = {}
    differing_columns: set[str] = set()

    cell_by_key: dict[tuple[object, object], object] = {}
    pivot_values: list[object] = []
    seen_pivot_values: set[object] = set()
    index_values: list[object] = []
    seen_index_values: set[object] = set()

    for chunk in (first_chunk, *chunks):
        for row in chunk.rows:
            index_value = row[index_pos]
            pivot_value = row[columns_pos]
            key = (index_value, pivot_value)
            if varying_columns:
                cells = tuple(
                    distinct_by_column[column].setdefault(row[pos], row[pos])
                    for column, pos in zip(varying_columns, varying_positions, strict=True)
                )
                first_cells = first_cells_by_key.setdefault(key, cells)
                differing_columns.update(
                    column
                    for column, first, cell in zip(varying_columns, first_cells, cells, strict=True)
                    if first != cell
                )
            if key in cell_by_key:
                if key not in seen_duplicate_keys:
                    seen_duplicate_keys.add(key)
                    if len(distinct_duplicate_keys) < _MAX_NAMED_DUPLICATE_KEYS:
                        distinct_duplicate_keys.append(key)
                    else:
                        duplicate_keys_truncated = True
            else:
                cell_by_key[key] = row[value_pos]
            if pivot_value not in seen_pivot_values:
                seen_pivot_values.add(pivot_value)
                pivot_values.append(pivot_value)
            if index_value not in seen_index_values:
                seen_index_values.add(index_value)
                index_values.append(index_value)
            for column, pos in zip(homogeneity_columns, homogeneity_positions, strict=True):
                values = homogeneity_seen[column].setdefault(index_value, set())
                cell = row[pos]
                if cell not in values:
                    if len(values) < _MAX_HOMOGENEITY_VALUES_PER_GROUP:
                        values.add(cell)
                    else:
                        homogeneity_value_truncated = True

    found_offenders: list[tuple[str, object, tuple[object, ...]]] = []
    for column in homogeneity_columns:
        for index_value in index_values:
            values = homogeneity_seen[column].get(index_value, set())
            if len(values) > 1:
                found_offenders.append(
                    (column, index_value, tuple(sorted(values, key=_cell_sort_key)))
                )
    offenders_truncated = (
        len(found_offenders) > _MAX_NAMED_DUPLICATE_KEYS or homogeneity_value_truncated
    )
    offenders = tuple(found_offenders[:_MAX_NAMED_DUPLICATE_KEYS])

    pivot_column_names = tuple(v if isinstance(v, str) else str(v) for v in pivot_values)
    raw_values_by_name: dict[str, list[object]] = {}
    for raw, name in zip(pivot_values, pivot_column_names, strict=True):
        raw_values_by_name.setdefault(name, []).append(raw)
    name_collisions: list[tuple[object, ...]] = []
    for name, raws in raw_values_by_name.items():
        collides_with_index = name == index
        if len(raws) > 1 or collides_with_index:
            name_collisions.append((index, *raws) if collides_with_index else tuple(raws))
    stringification_truncated = len(name_collisions) > _MAX_NAMED_DUPLICATE_KEYS
    stringification_collisions = tuple(name_collisions[:_MAX_NAMED_DUPLICATE_KEYS])

    width = 1 + len(pivot_values)
    cells = len(index_values) * width

    violation: (
        PivotNotUniqueError
        | PivotOverColumnCapError
        | PivotOverCellCapError
        | PivotNotHomogeneousError
        | None
    ) = None

    if distinct_duplicate_keys:
        violation = PivotNotUniqueError(
            f"pivot on index={index!r}, columns={columns!r} is not unique: duplicate "
            f"keys {tuple(distinct_duplicate_keys)!r}"
            f"{' (truncated)' if duplicate_keys_truncated else ''}",
            duplicate_keys=tuple(distinct_duplicate_keys),
            truncated=duplicate_keys_truncated,
            varying=known.varying
            if known is not None
            else tuple(
                (c, len(distinct_by_column[c])) for c in varying_columns if c in differing_columns
            ),
        )
    elif offenders:
        violation = PivotNotHomogeneousError(
            f"pivot on index={index!r} would merge rows that differ on column(s) it drops: "
            f"{offenders!r}{' (truncated)' if offenders_truncated else ''}",
            offenders=offenders,
            truncated=offenders_truncated,
        )
    elif (
        cap_violation := _pivot_cap_violation(
            width=width, cells=cells, column_cap=column_cap, cell_cap=cell_cap
        )
    ) is not None:
        violation = cap_violation
    elif stringification_collisions:
        violation = PivotNotUniqueError(
            f"pivot on index={index!r}, columns={columns!r} is not unique: distinct "
            f"columns values collide once stringified into the same output column name: "
            f"{stringification_collisions!r}"
            f"{' (truncated)' if stringification_truncated else ''}",
            duplicate_keys=stringification_collisions,
            truncated=stringification_truncated,
        )

    if violation is not None:
        return PivotChunksOutcome(table=None, violation=violation)

    output_columns = (index, *pivot_column_names)
    output_rows = tuple(
        (
            index_value,
            *(cell_by_key.get((index_value, pv)) for pv in pivot_values),
        )
        for index_value in index_values
    )
    return PivotChunksOutcome(table=Table(columns=output_columns, rows=output_rows), violation=None)


def _pivot_cap_violation(
    *, width: int, cells: int, column_cap: int, cell_cap: int
) -> PivotOverColumnCapError | PivotOverCellCapError | None:
    """The column-cap refusal, else the cell-cap refusal, for a pivot of this shape."""
    if width > column_cap:
        return PivotOverColumnCapError(
            f"pivot output would carry {width} columns, exceeding the format's "
            f"column cap of {column_cap}",
            count=width,
            cap=column_cap,
        )
    if cells > cell_cap:
        return PivotOverCellCapError(
            f"pivot output would carry {cells} cells ({cells // width} rows x "
            f"{width} columns), exceeding this deployment's cap of {cell_cap}",
            count=cells,
            cap=cell_cap,
        )
    return None


def pivot_chunks(
    chunks: Iterator[TableLike],
    *,
    index: str,
    columns: str,
    column_cap: int,
    cell_cap: int,
    classification: ColumnClassification,
    value_column: str = _VALUE_COLUMN,
) -> Table:
    """`pivot`'s streaming twin: a thin, always-raising wrapper over
    `fold_pivot_chunks` (`outcome.violation`, when present, is raised
    here; otherwise `outcome.table` is returned) -- see that function's
    own docstring for the one-pass accumulation this wraps.

    A caller not needing to interleave an external, higher-precedence
    check between reading `chunks` and this function's own violation
    always gets this direct, raising contract; `datasets_api.render
    ._stream_bare_pivot` is the one caller that does, and calls
    `fold_pivot_chunks` directly instead.
    """
    outcome = fold_pivot_chunks(
        chunks,
        index=index,
        columns=columns,
        column_cap=column_cap,
        cell_cap=cell_cap,
        classification=classification,
        value_column=value_column,
    )
    if outcome.violation is not None:
        raise outcome.violation
    assert outcome.table is not None
    return outcome.table


@dataclass(frozen=True)
class DecodePlan:
    """One `decode_columns` invocation's parameters, mirroring
    `EnrichmentPlan`'s own shape. `requests` is a tuple of `(entity_column,
    code_system)` pairs, in the order `decode_columns` should append their
    output columns -- `stat_var` paired with every code system this frame's
    StatVars declare that the code index also declares. `lookup` is
    `{ref: {code_system: label_or_absent_reading}}`, built by the caller
    from the code index: see `decode_columns`'s own docstring for how a
    declared-absent reading and a genuine index miss both reach this
    mapping, kept distinguishable.
    """

    requests: tuple[tuple[str, str], ...]
    lookup: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class GroupbyPlan:
    """One `groupby_sum` invocation's parameters, everything but `table`
    and `classification` -- `apply_reshapes` supplies those from its own
    inputs.
    """

    axis: str
    summable_axes: frozenset[str]
    hierarchy: Mapping[object, object] = field(default_factory=dict)
    missing_declared_as_zero: bool = False
    expected_population: ExpectedPopulation | None = None


@dataclass(frozen=True)
class EnrichmentPlan:
    """One `enrich_columns` invocation's parameters. `requests` is a tuple
    of `(entity_column, field)` pairs, in the order `enrich_columns` should
    append their output columns.
    """

    requests: tuple[tuple[str, str], ...]
    lookup: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class ParsedColumns:
    """The result of splitting a caller's `columns` token list into a
    selection tuple and enrichment requests, per `parse_column_tokens`.
    """

    select: tuple[str, ...]
    enrichments: tuple[tuple[str, str], ...]


def parse_column_tokens(
    tokens: Sequence[str],
    *,
    table_columns: Sequence[str],
    entity_columns: Sequence[str],
    supported_fields: frozenset[str],
) -> ParsedColumns:
    """Split tokens into selected columns and enrichment requests. Exact
    table-column matches take priority; other tokens match an entity-column
    prefix plus a supported field, since names can contain underscores. A bare
    field in `TABLE_WIDE_ENRICHMENTS` (`party_kind`) requests that field for every
    party role of the table (`materialise.enrichment.is_party_role`), and a qualified
    token the bare one already covers adds nothing. `party_kind` belongs to party roles
    only, so a token qualifying a subject role with it (`observationAbout_party_kind`)
    does not parse.
    Unknown tokens raise `ColumnUnknownError` with all qualified alternatives,
    independent of field enablement. Preserve token order; callers reject
    duplicates.
    """
    table_column_set = set(table_columns)
    select: list[str] = []
    enrichments: list[tuple[str, str]] = []
    alternatives: tuple[str, ...] | None = None
    for token in tokens:
        if token in table_column_set:
            select.append(token)
            continue

        if token in supported_fields and token in TABLE_WIDE_ENRICHMENTS:
            enrichments.extend(
                (entity_column, token)
                for entity_column in entity_columns
                if is_party_role(entity_column) and (entity_column, token) not in enrichments
            )
            continue

        matched_request: tuple[str, str] | None = None
        for entity_column in entity_columns:
            prefix = f"{entity_column}_"
            if not token.startswith(prefix):
                continue
            field_name = token[len(prefix) :]
            if field_name in supported_fields and enrichment_qualifies(entity_column, field_name):
                matched_request = (entity_column, field_name)
                break
        if matched_request is not None:
            if matched_request not in enrichments:
                enrichments.append(matched_request)
            continue

        if alternatives is None:
            alternatives = tuple(
                f"{entity_column}_{field_name}"
                for entity_column in entity_columns
                for field_name in sorted(supported_fields)
                if enrichment_qualifies(entity_column, field_name)
            )
        raise ColumnUnknownError(
            f"column {token!r} is not present on this table and does not parse as a "
            "qualified enrichment token; a token must name a column of this table or "
            "read as {entity_column}_{field}",
            column=token,
            alternatives=alternatives,
        )

    return ParsedColumns(select=tuple(select), enrichments=tuple(enrichments))


def enrichment_qualifies(entity_column: str, field_name: str) -> bool:
    """Whether `field_name` can qualify `entity_column`: every field but `party_kind`, which
    qualifies a party role only."""
    return field_name != PARTY_KIND_FIELD or is_party_role(entity_column)


@dataclass(frozen=True)
class PivotPlan:
    """One `pivot` invocation's parameters, everything but `table` and
    `classification` -- `apply_reshapes` supplies those from its own
    inputs, the same way it does for `GroupbyPlan`.
    """

    index: str
    columns: str
    column_cap: int
    cell_cap: int
    value_column: str = _VALUE_COLUMN


@dataclass(frozen=True)
class ReshapePlan:
    """The closed reshape list a caller wants applied. `apply_reshapes`
    always runs the steps present here in this module's ONE fixed order --
    decode, then groupby, then column selection, then enrichment, then
    pivot -- whatever order these fields are written in below; a step left
    `None` is skipped entirely.
    """

    decode: DecodePlan | None = None
    groupby: GroupbyPlan | None = None
    select: tuple[str, ...] | None = None
    enrichment: EnrichmentPlan | None = None
    pivot: PivotPlan | None = None


def apply_reshapes(
    table: TableLike, *, plan: ReshapePlan
) -> tuple[Table, tuple[MaterialisationWarning, ...]]:
    """Apply requested steps in fixed order: decode, groupby, select,
    enrichment, and pivot. Derive column classification once from the input
    before decoding; groupby and selection share it, so decoded columns stay
    droppable and collapsed grain columns remain protected. Content-version
    minting precedes reshape, which never changes identity.
    """
    classification = classify_columns(table.columns)
    warnings: list[MaterialisationWarning] = []
    current = Table(columns=table.columns, rows=table.rows)

    if plan.decode is not None:
        current, decode_warnings = decode_columns(
            current,
            requests=plan.decode.requests,
            lookup=plan.decode.lookup,
        )
        warnings.extend(decode_warnings)

    if plan.groupby is not None:
        current = groupby_sum(
            current,
            axis=plan.groupby.axis,
            classification=classification,
            summable_axes=plan.groupby.summable_axes,
            hierarchy=plan.groupby.hierarchy,
            missing_declared_as_zero=plan.groupby.missing_declared_as_zero,
            expected_population=plan.groupby.expected_population,
        )

    if plan.select is not None:
        current = select_columns(current, keep=plan.select, classification=classification)

    if plan.enrichment is not None:
        current, enrichment_warnings = enrich_columns(
            current,
            requests=plan.enrichment.requests,
            lookup=plan.enrichment.lookup,
        )
        warnings.extend(enrichment_warnings)

    if plan.pivot is not None:
        current = pivot(
            current,
            index=plan.pivot.index,
            columns=plan.pivot.columns,
            column_cap=plan.pivot.column_cap,
            cell_cap=plan.pivot.cell_cap,
            classification=classification,
            value_column=plan.pivot.value_column,
        )

    return current, tuple(warnings)
