"""`compare` and `sum_of` over two slices paired member by member.

`on` maps each member column of the call's `data` side to the column of its
`other` side that holds the same entities: `{"destination":
"observationAbout"}` pairs a CRS recipient with a WDI country. `data` is
`sum_of`'s left side and `compare`'s right side, since `compare` gives right
minus left (`dfo_analysis_engine.engine.compare`), so either call returns `data`
plus or minus `other` with its rows keyed as `data` names them.

Each side is sliced as the single-series form slices it
(`dfo_analysis_engine.engine.compare.slice_pair`), and must hold one row per member
and period, so nothing is summed within a side (`CompareAmbiguousSliceError`).
Units and facets follow each verb's single-series policy: both refuse two
different units, `compare` carries its left side's facets less the axes the
sides declare apart (`compare.compare_facets`), and `sum_of` carries the axes
both sides agree on (`dfo_analysis_engine.engine.common.agreed_facets`) and refuses a
composed unit.

Members pair as `quotient_of` pairs its keys
(`dfo_analysis_engine.engine.two_sided.split_keys` and `pair_periods`). A member one
side holds and the other lacks is left out and recorded as `unmatched_key`
with its side. A paired member's periods the other side lacks are recorded as
`period_unmatched`, and a paired member sharing no period with its partner is
dropped from both sides. Each result row is one paired member and one period
both sides hold.

`on` naming a column a side lacks refuses (`PairedUnknownColumnError`).
Three calls have no row to return and refuse: a side holding no row, or no
paired member sharing a period (`ComparisonPeriodsDisjointError`); and no
member pairing at all (`PairedNoKeyMatchedError`). A sum or difference that
overflows refuses (`CompareNonFiniteResultError`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

import pyarrow as pa
from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import (
    STAT_VAR_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    ComposedUnit,
    UnitCell,
)

from dfo_analysis_engine.engine.aggregate import ComposedUnitSumError
from dfo_analysis_engine.engine.columnar import distinct, repeated
from dfo_analysis_engine.engine.common import (
    Exclusions,
    FacetSelection,
    UnknownEntityColumnError,
    agreed_facets,
    combine_exclusions,
    facet_columns,
    frame_keeping_role_types,
    record_exclusion,
    validate_entity_columns,
)
from dfo_analysis_engine.engine.compare import (
    CompareNonFiniteResultError,
    ComparisonPeriodsDisjointError,
    SideSlice,
    compare_facets,
    held_entities,
    slice_pair,
)
from dfo_analysis_engine.engine.two_sided import (
    Key,
    KeyPairs,
    KeyPeriods,
    key_sample,
    pair_periods,
    split_keys,
)

type PairedVerb = Literal["compare", "sum_of"]
type SliceSide = Literal["left", "right"]


class PairedUnknownColumnError(Exception):
    """`on` names `columns` that `side`'s frame holds no member column for.
    `available` is what a retry could name there."""

    def __init__(
        self, *, side: SliceSide, columns: Sequence[str], available: Sequence[str]
    ) -> None:
        self.side = side
        self.columns = tuple(columns)
        self.available = tuple(available)
        super().__init__(f"on names column(s) the {side} side lacks: {self.columns!r}")


class PairedNoKeyMatchedError(Exception):
    """Both sides hold members and none pairs under `on`. `left_keys` and
    `right_keys` are each side's sample of members
    (`dfo_analysis_engine.engine.two_sided.key_sample`), as `(column, value)` pairs
    under that side's own column names; `left_key_count` and
    `right_key_count` are how many members each side holds."""

    def __init__(
        self,
        *,
        left_keys: Sequence[KeyPairs],
        right_keys: Sequence[KeyPairs],
        left_key_count: int,
        right_key_count: int,
    ) -> None:
        self.left_keys = tuple(left_keys)
        self.right_keys = tuple(right_keys)
        self.left_key_count = left_key_count
        self.right_key_count = right_key_count
        super().__init__(
            f"none of the left side's {left_key_count} member(s) pairs with any of the "
            f"right side's {right_key_count}"
        )


@dataclass(frozen=True, kw_only=True)
class PairedCell:
    """One paired member and period: both sides' values and the result's.
    `key` names the member as `data` names it."""

    key: KeyPairs
    time_period: str
    left_value: float
    right_value: float
    value: float


@dataclass(frozen=True, kw_only=True)
class PairedResult:
    """Everything `dfo_analysis_engine.calls.call_verb` needs to mint identity, write the
    store and render the digest for a paired `compare` or `sum_of`.

    `data_side` is the side the call's `data` fills, whose column names key
    `frame` and every `KeyPairs` below but the unmatched members. Periods and
    coverage read the members that pair: `left_periods` and `right_periods`
    are every period those members hold on each side, `matched_periods` every
    period some member shares with its partner, and a side's coverage
    fraction the (member, period) cells it shares over those it holds.

    The dropped members are those the frame holds no row for:
    `left_unmatched_keys` and `right_unmatched_keys`, with no partner, each
    under its side's own column names, and `keys_without_shared_period`,
    paired and sharing no period, which both sides lose.
    `left_only_periods` and `right_only_periods` are, for each member with a
    row, the periods it holds on that side alone.
    """

    frame: Frame
    stat_var: str
    data_side: SliceSide
    left_stat_vars: tuple[str, ...]
    right_stat_vars: tuple[str, ...]
    left_entities: Mapping[str, str]
    right_entities: Mapping[str, str]
    unit: UnitCell | None
    facet_selections: tuple[FacetSelection, ...]
    left_facet_selections: tuple[FacetSelection, ...]
    right_facet_selections: tuple[FacetSelection, ...]
    warnings: tuple[str, ...]
    left_periods: tuple[str, ...]
    right_periods: tuple[str, ...]
    matched_periods: tuple[str, ...]
    left_coverage_fraction: float
    right_coverage_fraction: float
    cells: tuple[PairedCell, ...]
    left_unmatched_keys: tuple[KeyPairs, ...] = ()
    right_unmatched_keys: tuple[KeyPairs, ...] = ()
    keys_without_shared_period: tuple[KeyPairs, ...] = ()
    left_only_periods: tuple[KeyPeriods, ...] = ()
    right_only_periods: tuple[KeyPeriods, ...] = ()
    exclusions: Exclusions = field(default_factory=Exclusions)

    @property
    def left_dropped_key_count(self) -> int:
        return len(self.left_unmatched_keys) + len(self.keys_without_shared_period)

    @property
    def right_dropped_key_count(self) -> int:
        return len(self.right_unmatched_keys) + len(self.keys_without_shared_period)


@dataclass(frozen=True, kw_only=True)
class _Members:
    """One side's slice by member: each member's value in each period it
    holds, and the row of the slice each value sits in."""

    slice: SideSlice
    frame: Frame
    columns: tuple[str, ...]
    by_key: dict[Key, dict[str, float]]
    rows: dict[tuple[Key, str], int]

    def taken(self, cells: Iterable[tuple[Key, str]]) -> pa.Table:
        """The slice's rows holding `cells`, in order."""
        indices = [self.rows[cell] for cell in cells]
        return self.slice.table.take(pa.array(indices, type=pa.int64()))

    def key_pairs(self, keys: Iterable[Key]) -> tuple[KeyPairs, ...]:
        return tuple(tuple(zip(self.columns, key, strict=True)) for key in keys)


def _members(side: SideSlice, frame: Frame, columns: tuple[str, ...]) -> _Members:
    """`side`, a slice holding one row per member of `columns` and period, by member."""
    table = side.table
    key_columns = [cast("list[str]", table.column(column).to_pylist()) for column in columns]
    periods = cast("list[str]", table.column(TIME_PERIOD_COLUMN).to_pylist())
    values = cast("list[float]", table.column(VALUE_COLUMN).to_pylist())
    by_key: dict[Key, dict[str, float]] = {}
    rows: dict[tuple[Key, str], int] = {}
    for index, (period, value) in enumerate(zip(periods, values, strict=True)):
        key = tuple(column[index] for column in key_columns)
        by_key.setdefault(key, {})[period] = value
        rows[key, period] = index
    return _Members(slice=side, frame=frame, columns=columns, by_key=by_key, rows=rows)


def _record(
    exclusions: Exclusions,
    members: _Members,
    *,
    reason: Literal["unmatched_key", "period_unmatched"],
    side: SliceSide,
    cells: Iterable[tuple[Key, str]],
) -> Exclusions:
    return record_exclusion(
        exclusions, frame=members.frame, reason=reason, dropped=members.taken(cells), side=side
    )


def run_paired_compare(
    left_frame: Frame,
    right_frame: Frame,
    *,
    stat_var: str,
    on: Mapping[str, str],
    left_entity: Mapping[str, str],
    right_entity: Mapping[str, str],
    left_pinned_unit: UnitCell | None = None,
    right_pinned_unit: UnitCell | None = None,
) -> PairedResult:
    """`right_frame` minus `left_frame` for each member and period both hold.
    `on` maps each member column of `right_frame`, the call's `data`, to the
    column of `left_frame` holding the same entities. See the module docstring."""
    return _run_paired(
        "compare",
        left_frame,
        right_frame,
        stat_var=stat_var,
        on=on,
        left_entity=left_entity,
        right_entity=right_entity,
        left_pinned_unit=left_pinned_unit,
        right_pinned_unit=right_pinned_unit,
    )


def run_paired_sum_of(
    left_frame: Frame,
    right_frame: Frame,
    *,
    stat_var: str,
    on: Mapping[str, str],
    left_entity: Mapping[str, str],
    right_entity: Mapping[str, str],
    left_pinned_unit: UnitCell | None = None,
    right_pinned_unit: UnitCell | None = None,
) -> PairedResult:
    """`left_frame` plus `right_frame` for each member and period both hold.
    `on` maps each member column of `left_frame`, the call's `data`, to the
    column of `right_frame` holding the same entities. See the module docstring."""
    return _run_paired(
        "sum_of",
        left_frame,
        right_frame,
        stat_var=stat_var,
        on=on,
        left_entity=left_entity,
        right_entity=right_entity,
        left_pinned_unit=left_pinned_unit,
        right_pinned_unit=right_pinned_unit,
    )


def _run_paired(
    verb: PairedVerb,
    left_frame: Frame,
    right_frame: Frame,
    *,
    stat_var: str,
    on: Mapping[str, str],
    left_entity: Mapping[str, str],
    right_entity: Mapping[str, str],
    left_pinned_unit: UnitCell | None,
    right_pinned_unit: UnitCell | None,
) -> PairedResult:
    data_side: SliceSide = "left" if verb == "sum_of" else "right"
    data_columns, other_columns = tuple(on), tuple(on.values())
    left_by, right_by = (
        (data_columns, other_columns) if data_side == "left" else (other_columns, data_columns)
    )
    for side, frame, by in (("left", left_frame, left_by), ("right", right_frame, right_by)):
        try:
            validate_entity_columns(frame, by)
        except UnknownEntityColumnError as exc:
            raise PairedUnknownColumnError(
                side=side, columns=exc.columns, available=exc.available
            ) from exc
    left_slice, right_slice = slice_pair(
        left_frame,
        right_frame,
        left_entity=left_entity,
        right_entity=right_entity,
        left_pinned_unit=left_pinned_unit,
        right_pinned_unit=right_pinned_unit,
        left_by=left_by,
        right_by=right_by,
    )
    if verb == "sum_of":
        for side, side_slice in (("left", left_slice), ("right", right_slice)):
            if isinstance(side_slice.unit, ComposedUnit):
                raise ComposedUnitSumError(side_slice.unit, side=side)
        facets: tuple[FacetSelection, ...] = ()
        facet_warnings: tuple[str, ...] = ()
        if left_slice.populated and right_slice.populated:
            facets, facet_warnings = agreed_facets(
                left_slice.facet_selections,
                right_slice.facet_selections,
                sides=("left", "right"),
                result="sum",
            )
    else:
        facets, facet_warnings = compare_facets(left_slice, right_slice)

    left = _members(left_slice, left_frame, left_by)
    right = _members(right_slice, right_frame, right_by)
    if not (left.by_key and right.by_key):
        raise ComparisonPeriodsDisjointError(
            left_periods=left_slice.periods(), right_periods=right_slice.periods()
        )
    paired, left_unmatched = split_keys(sorted(left.by_key), right, pooled=False)
    _, right_unmatched = split_keys(sorted(right.by_key), left, pooled=False)
    if not paired:
        raise PairedNoKeyMatchedError(
            left_keys=key_sample(left, left_by),
            right_keys=key_sample(right, right_by),
            left_key_count=len(left.by_key),
            right_key_count=len(right.by_key),
        )
    pairing = pair_periods(left, right, numerator_by=data_columns, paired_keys=paired, pooled=False)
    if not pairing.matched_periods:
        raise ComparisonPeriodsDisjointError(
            left_periods=pairing.numerator_periods, right_periods=pairing.denominator_periods
        )

    cells: list[PairedCell] = []
    kept: list[tuple[Key, str]] = []
    for key, periods in pairing.shared.items():
        key_pairs = tuple(zip(data_columns, key, strict=True))
        for period in periods:
            left_value, right_value = left.by_key[key][period], right.by_key[key][period]
            value = left_value + right_value if verb == "sum_of" else right_value - left_value
            if not math.isfinite(value):
                raise CompareNonFiniteResultError(
                    time_period=period, quantity="sum" if verb == "sum_of" else "delta"
                )
            kept.append((key, period))
            cells.append(
                PairedCell(
                    key=key_pairs,
                    time_period=period,
                    left_value=left_value,
                    right_value=right_value,
                    # Signed zero normalised, as every verb's output is.
                    value=value + 0.0,
                )
            )

    exclusions = combine_exclusions(left_slice.exclusions, right_slice.exclusions)
    for side, members, unmatched in (
        ("left", left, left_unmatched),
        ("right", right, right_unmatched),
    ):
        exclusions = _record(
            exclusions,
            members,
            reason="unmatched_key",
            side=side,
            cells=((key, period) for key in unmatched for period in members.by_key[key]),
        )
    for side, members, outside in (
        ("left", left, pairing.numerator_unmatched),
        ("right", right, pairing.denominator_unmatched),
    ):
        exclusions = _record(
            exclusions, members, reason="period_unmatched", side=side, cells=outside
        )

    data = left if data_side == "left" else right
    data_rows = data.taken(kept)
    count = len(cells)
    output_frame = frame_keeping_role_types(
        {
            STAT_VAR_COLUMN: repeated(stat_var, count),
            TIME_PERIOD_COLUMN: pa.array([cell.time_period for cell in cells], type=pa.string()),
            VALUE_COLUMN: pa.array([cell.value for cell in cells], type=pa.float64()),
            UNIT_COLUMN: data_rows.column(UNIT_COLUMN),
            **facet_columns(facets, count=count),
            **{column: data_rows.column(column) for column in data_columns},
        },
        role_keys=data_columns,
        source=data.frame,
    )

    return PairedResult(
        frame=output_frame,
        stat_var=stat_var,
        data_side=data_side,
        left_stat_vars=tuple(distinct(left.taken(kept), STAT_VAR_COLUMN)),
        right_stat_vars=tuple(distinct(right.taken(kept), STAT_VAR_COLUMN)),
        left_entities=held_entities(left_frame, left_slice.table, entity=left_entity),
        right_entities=held_entities(right_frame, right_slice.table, entity=right_entity),
        unit=left_slice.unit,
        facet_selections=facets,
        left_facet_selections=left_slice.facet_selections,
        right_facet_selections=right_slice.facet_selections,
        warnings=tuple(
            dict.fromkeys((*left_slice.warnings, *right_slice.warnings, *facet_warnings))
        ),
        left_periods=pairing.numerator_periods,
        right_periods=pairing.denominator_periods,
        matched_periods=pairing.matched_periods,
        left_coverage_fraction=pairing.numerator_coverage_fraction,
        right_coverage_fraction=pairing.denominator_coverage_fraction,
        cells=tuple(cells),
        left_unmatched_keys=left.key_pairs(left_unmatched),
        right_unmatched_keys=right.key_pairs(right_unmatched),
        keys_without_shared_period=pairing.keys_without_shared_period,
        left_only_periods=pairing.numerator_only_periods,
        right_only_periods=pairing.denominator_only_periods,
        exclusions=exclusions,
    )
