"""Refusals: the stable code, retry mode, message and `fix` of every call the
engine declines, and the one mapping from an engine exception to its refusal.

A refusal is a `Refusal`, built by one of its typed factories, and raised as
`AnalysisRefused`. `refusal_for_engine_error` maps the typed exceptions a
verb's `run_<verb>` raises to the refusal that verb gives; an exception outside
the verb's own set is no refusal and propagates. `Refusal.as_json` is the body
a service returns: `{code, retry_mode, message, fix}`. The messages name the
call's parameters (`frame_ref`, `group_by`, `get_data`), since the caller that
reads them states a call in those terms.

Each code has one `RetryMode` (`RETRY_MODE_BY_CODE`). `do_not_retry` codes
share one shape: nothing about the call's arguments was wrong, so retrying it
unchanged reaches the same failure. `non_finite_result` names an overflow,
`mixed_time_grain` a frame whose periods do not change between calls, and
`rolling_grain_unsupported` root frames whose periods are not calendar years.
Every other code names something a caller can change: a different
`frame_ref`, a supported aggregation, a narrower `group_by`, a unit pin.

One frame carrying more than one unit or facet value refuses
`unit_heterogeneous` or `facet_heterogeneous`, with the candidates in
`fix["disambiguation"]`; two sides that each resolve to one value and
disagree refuse `cross_frame_unit_mismatch`, `cross_frame_unit_unbound` (one
side declares no unit) or `cross_frame_facet_mismatch` (one StatVar from two
facet selections). A unit refusal's `fix["units"]` lists each candidate in the
form the pin parameter accepts: a unit ref as its string, a converted unit as
its request object, a composed unit as its `{numerator, denominator}` object.
`role_ambiguous` names the calling verb's own column parameter in
`fix["parameter"]`.

`rolling_anchor_unresolved` names, in `fix["detail"]`, the check that left the
latest period unestablished; `rolling_populations_differ` refuses a recipe
whose rolling op-specs carry more than one population list, which no single
latest period serves, even when the lists give the same period.

`composite_refusal` turns a `stat_var_unresolved` over the members of a
composite frame into the refusal naming its components and the `get_data`
read that sums them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Self

from dfo_frame_grammar.grammar import ComposedUnit, UnitCell, UnitRequest

from dfo_analysis_engine.engine.aggregate import (
    AGGREGATIONS,
    ComposedUnitSumError,
    NonFiniteAggregateResultError,
    UnknownGroupByColumnError,
    UnsupportedAggregationError,
)
from dfo_analysis_engine.engine.aggregate import (
    StatVarUnresolvedError as AggregateStatVarUnresolvedError,
)
from dfo_analysis_engine.engine.common import (
    FacetHeterogeneousError,
    PlaceLevelsMixedError,
    StatVarUnresolvedError,
    UnitHeterogeneousError,
    UnitNotPresentError,
    UnknownEntityColumnError,
)
from dfo_analysis_engine.engine.compare import (
    CompareAmbiguousSliceError,
    CompareFacetMismatchError,
    CompareNonFiniteResultError,
    CompareUnitMismatchError,
    CompareUnitUnboundError,
    ComparisonPeriodsDisjointError,
)
from dfo_analysis_engine.engine.filter import (
    FilterPlaceLevelUnknownError,
    FilterPredicateTypeMismatchError,
    FilterPredicateValueTypeMismatchError,
    FilterUnitConversionUnsupportedError,
    FilterUnknownColumnError,
)
from dfo_analysis_engine.engine.growth import (
    GrowthAmbiguousEntityError,
    GrowthEndpointsMissingError,
    GrowthNonFiniteResultError,
)
from dfo_analysis_engine.engine.join import (
    JoinAmbiguousKeyError,
    JoinNoCommonKeyColumnsError,
    JoinOutputCollisionError,
    JoinResultTooLargeError,
    JoinUnknownKeyError,
)
from dfo_analysis_engine.engine.paired import PairedNoKeyMatchedError, PairedUnknownColumnError
from dfo_analysis_engine.engine.quotient_of import (
    QuotientOfNoKeyMatchedError,
    QuotientOfPeriodsDisjointError,
    QuotientOfSideUnitUnsupportedError,
    QuotientOfZeroDenominatorError,
)
from dfo_analysis_engine.engine.rank import RankAmbiguousEntityError, RankNoRowsInWindowError
from dfo_analysis_engine.engine.share_of import (
    ShareOfFacetMismatchError,
    ShareOfGrainMismatchError,
    ShareOfKeyUnknownError,
    ShareOfKeyUnmatchedError,
    ShareOfNonFiniteResultError,
    ShareOfUnitMismatchError,
    ShareOfUnitUnboundError,
)
from dfo_analysis_engine.engine.two_sided import NonFiniteCellError
from dfo_analysis_engine.place_type import (
    PARTY_ROLES,
    PLACE_LEVELS,
    WHOLE_CELL,
    WHOLE_SIDE_POPULATIONS,
    PartyKind,
    class_column,
    enrichment_role,
    whole_column,
)
from dfo_analysis_engine.units import unit_json, unit_name

__all__ = [
    "RETRY_MODE_BY_CODE",
    "AnalysisRefused",
    "CompositeComponent",
    "Refusal",
    "RefusalCode",
    "RetryMode",
    "composite_refusal",
    "refusal_for_engine_error",
]


class RetryMode(StrEnum):
    """How a caller should react to a refusal, apart from the code that caused it."""

    CORRECT_AND_RETRY = "correct_and_retry"
    RETRY_AFTER_DELAY = "retry_after_delay"
    DO_NOT_RETRY = "do_not_retry"


class RefusalCode(StrEnum):
    """The codes the engine refuses with, each with exactly one `RetryMode`."""

    OP_SPEC_INVALID = "op_spec_invalid"
    STAT_VAR_UNRESOLVED = "stat_var_unresolved"
    ROLE_AMBIGUOUS = "role_ambiguous"
    OPERATION_UNCOVERED = "operation_uncovered"
    FRAME_TOO_LARGE = "frame_too_large"
    UNIT_HETEROGENEOUS = "unit_heterogeneous"
    FACET_HETEROGENEOUS = "facet_heterogeneous"
    COVERAGE_INSUFFICIENT = "coverage_insufficient"
    NON_FINITE_RESULT = "non_finite_result"
    MIXED_TIME_GRAIN = "mixed_time_grain"
    COMPARISON_PERIODS_DISJOINT = "comparison_periods_disjoint"
    GROWTH_GAP_YEAR = "growth_gap_year"
    CROSS_FRAME_UNIT_MISMATCH = "cross_frame_unit_mismatch"
    CROSS_FRAME_UNIT_UNBOUND = "cross_frame_unit_unbound"
    CROSS_FRAME_FACET_MISMATCH = "cross_frame_facet_mismatch"
    SHARE_OF_GRAIN_MISMATCH = "share_of_grain_mismatch"
    FILTER_PREDICATE_TYPE_MISMATCH = "filter_predicate_type_mismatch"
    FILTER_UNIT_CONVERSION_UNSUPPORTED = "filter_unit_conversion_unsupported"
    SHARE_OF_ZERO_DENOMINATOR = "share_of_zero_denominator"
    SHARE_OF_KEY_UNKNOWN = "share_of_key_unknown"
    SHARE_OF_KEY_UNMATCHED = "share_of_key_unmatched"
    QUOTIENT_OF_ZERO_DENOMINATOR = "quotient_of_zero_denominator"
    QUOTIENT_OF_NO_KEY_MATCHED = "quotient_of_no_key_matched"
    COMPARE_NO_KEY_MATCHED = "compare_no_key_matched"
    SUM_OF_NO_KEY_MATCHED = "sum_of_no_key_matched"
    QUOTIENT_OF_SIDE_UNIT_UNSUPPORTED = "quotient_of_side_unit_unsupported"
    COMPOSED_UNIT_SUM_UNSUPPORTED = "composed_unit_sum_unsupported"
    JOIN_OUTPUT_COLLISION = "join_output_collision"
    JOIN_NO_COMMON_KEY_COLUMNS = "join_no_common_key_columns"
    DERIVED_INPUT_UNUSABLE = "derived_input_unusable"
    DERIVED_INPUT_ORDINAL = "derived_input_ordinal"
    PLACE_LEVELS_MIXED = "place_levels_mixed"
    ROLLING_ANCHOR_UNRESOLVED = "rolling_anchor_unresolved"
    ROLLING_GRAIN_UNSUPPORTED = "rolling_grain_unsupported"
    ROLLING_POPULATIONS_DIFFER = "rolling_populations_differ"


_DO_NOT_RETRY = frozenset(
    {
        RefusalCode.NON_FINITE_RESULT,
        RefusalCode.MIXED_TIME_GRAIN,
        RefusalCode.ROLLING_GRAIN_UNSUPPORTED,
    }
)

RETRY_MODE_BY_CODE: Mapping[RefusalCode, RetryMode] = {
    code: RetryMode.DO_NOT_RETRY if code in _DO_NOT_RETRY else RetryMode.CORRECT_AND_RETRY
    for code in RefusalCode
}

# Caps the candidates a disambiguation payload carries in a refusal's `fix`.
_DISAMBIGUATION_CANDIDATE_CAP = 10

# The longest label a disambiguation candidate carries.
MAX_LABEL_CHARS = 200

# How many of each side's sampled keys a key-matching refusal's message lists;
# `fix` carries the whole sample.
_KEY_SAMPLE_IN_MESSAGE = 5

# The party kinds a filter keeps to compare or rank a flow side's members: private parties
# are of kind `party` too, told apart by their class.
_MEMBER_KINDS: tuple[PartyKind, ...] = ("party",)
# The kinds a filter keeps to sum the members of a role alone: its parties and the
# unallocated residuals, which add to a sum though no ranking compares them.
_POOLED_MEMBER_KINDS: tuple[PartyKind, ...] = ("party", "residual")


# How the `filter` refusal over a converted unit tells a caller to get a
# frame with one unit.
_REFETCH_ONE_UNIT = (
    "call get_data again with scope.facets binding a single unit, then retry with the new frame_ref"
)

# How the same refusal over a composed unit tells a caller to select one
# unit: a verb wrote the frame, so no `get_data` call returns it in one unit.
_PIN_ONE_UNIT = (
    "set the unit parameter of the verb that reads this frame_ref to that unit's object, "
    "as a unit_heterogeneous refusal lists it"
)


def _quoted_unit(unit: UnitCell) -> str:
    """`unit` as a message quotes it: the `repr` of its `unit_name`."""
    return repr(unit_name(unit))


def _pin_values(units: Sequence[UnitCell]) -> str:
    """`units` as a message lists them for a retry: each as the JSON value
    the unit-pin parameter accepts, a ref as a JSON string and a converted
    or composed unit as its object."""
    return ", ".join(json.dumps(unit_json(unit)) for unit in units)


def _disambiguation_payload(candidates: Sequence[UnitCell]) -> dict[str, Any] | None:
    """The candidate set a caller chooses from, or `None` when `candidates` is
    empty, as a frame whose every declared unit is `None` has none to offer.
    Each candidate's `id` is the value the pin parameter accepts (a ref as its
    string, a converted or composed unit as its object) and its label its
    `unit_name`, cut to `MAX_LABEL_CHARS`."""
    if not candidates:
        return None
    return {
        "candidates": [
            {
                "id": value if isinstance(value, str) else value.as_json(),
                "label": unit_name(value)[:MAX_LABEL_CHARS],
                "detail": {},
            }
            for value in candidates[:_DISAMBIGUATION_CANDIDATE_CAP]
        ]
    }


@dataclass(frozen=True, kw_only=True)
class Refusal:
    """A refusal: its code, the message a caller reads, and `fix`, the JSON
    object naming what to change. Built by one of the typed factories below."""

    code: RefusalCode
    message: str
    fix: dict[str, Any]

    @property
    def retry_mode(self) -> RetryMode:
        return RETRY_MODE_BY_CODE[self.code]

    def as_json(self) -> dict[str, Any]:
        """`{code, retry_mode, message, fix}`, the refusal body a service returns."""
        return {
            "code": self.code.value,
            "retry_mode": self.retry_mode.value,
            "message": self.message,
            "fix": self.fix,
        }

    @classmethod
    def op_spec_invalid(cls, *, detail: str) -> Self:
        return cls(
            code=RefusalCode.OP_SPEC_INVALID,
            message=f"The op-spec was invalid: {detail}. Correct it and retry.",
            fix={"detail": detail},
        )

    @classmethod
    def stat_var_unresolved(cls, *, reason: str, distinct_stat_vars: list[str]) -> Self:
        return cls(
            code=RefusalCode.STAT_VAR_UNRESOLVED,
            message=f"Could not resolve a single StatVar to aggregate: {reason}.",
            fix={"reason": reason, "distinct_stat_vars": distinct_stat_vars},
        )

    @classmethod
    def composite_stat_vars_unsummed(
        cls, *, distinct_stat_vars: list[str], components: list[dict[str, object]]
    ) -> Self:
        """`stat_var_unresolved` for a composite frame read with one series per StatVar:
        `components` names each component's role and member count
        (`composite_refusal`), and `get_data` the read that adds
        them into one series per party and year."""
        parts = " and ".join(
            f"{component['role']} ({component['stat_vars']} StatVars)" for component in components
        )
        reason = f"the input frame is a composite of {parts}, read with one series per StatVar"
        return cls(
            code=RefusalCode.STAT_VAR_UNRESOLVED,
            message=(
                f"Could not resolve a single StatVar: {reason}. Read the spec again with "
                'get_data stat_vars="sum", which adds every component\'s StatVars into one '
                "series per party and year, and pass that frame_ref."
            ),
            fix={
                "reason": reason,
                "distinct_stat_vars": distinct_stat_vars,
                "components": components,
                "get_data": {"stat_vars": "sum"},
            },
        )

    @classmethod
    def role_ambiguous(
        cls,
        *,
        parameter: str,
        columns: list[str],
        available: list[str],
        side: str | None = None,
    ) -> Self:
        """`parameter` names the calling verb's own column-list parameter
        this refusal is about (`group_by`, `entity_by`, `rank_by`,
        `join_keys`, `predicates`, `on`, or an `entity` binding such as
        `left_entity`) -- never hardcoded to one verb's own vocabulary,
        since every verb but `aggregate` reaches this factory too. `side`
        names the side of a two-sided verb whose frame lacks the columns,
        where `parameter` reads both sides.
        """
        frame = "this frame" if side is None else f"the {side} side's frame"
        fix: dict[str, Any] = {"parameter": parameter, "columns": columns, "available": available}
        if side is not None:
            fix["side"] = side
        return cls(
            code=RefusalCode.ROLE_AMBIGUOUS,
            message=(
                f"{parameter} names column(s) {frame} has no grain for: {columns}. "
                f"Available: {available}."
            ),
            fix=fix,
        )

    @classmethod
    def paired_member_ambiguous(
        cls, *, side: str, members: list[str], entity: list[str], available: list[str]
    ) -> Self:
        """`role_ambiguous` for a side of a paired `compare` or `sum_of` that holds
        more than one row for one member and period: its `{side}_entity` binding
        (`entity`, the columns it binds) leaves a role unbound, and binding one more
        is the fix. `members` are the side's member columns under `on`."""
        parameter = f"{side}_entity"
        return cls(
            code=RefusalCode.ROLE_AMBIGUOUS,
            message=(
                f"The {side} side holds more than one row for one member of {members} in "
                f"one period, so {parameter} {entity} leaves a role unbound. Bind one more "
                f"of {available} in {parameter}."
            ),
            fix={
                "parameter": parameter,
                "side": side,
                "columns": entity,
                "members": members,
                "available": available,
            },
        )

    @classmethod
    def operation_uncovered(cls, *, aggregation: str, supported: list[str]) -> Self:
        return cls(
            code=RefusalCode.OPERATION_UNCOVERED,
            message=(f"{aggregation!r} is not a supported aggregation. Supported: {supported}."),
            fix={"aggregation": aggregation, "supported": supported},
        )

    @classmethod
    def frame_too_large_on_write(cls, *, reason: str) -> Self:
        """`frame_too_large` for this call's own result, over the row ceiling
        it may write, before any frame is built. `reason` states the row count
        and the ceiling. Narrowing this call's inputs is the fix."""
        return cls(
            code=RefusalCode.FRAME_TOO_LARGE,
            message=(
                f"This call's own result was too large to store: {reason}. Narrow the "
                "inputs to this operation (a smaller frame_ref, a coarser group_by, or an "
                "entity/predicate that selects fewer rows) and retry -- no result was stored."
            ),
            fix={"reason": reason},
        )

    @classmethod
    def unit_heterogeneous(cls, *, parameter: str, units: Sequence[UnitCell]) -> Self:
        """`parameter` names the calling verb's own unit-pin parameter
        (`unit`, or, for a two-sided verb, both sides' own names together,
        e.g. `"left_unit or right_unit"`), since `compare`/`share_of`/`join`
        have no single `unit` field.

        `filter` refuses `category_match` on a converted or composed unit,
        so when `units` holds one the message offers only the pin, listing
        each unit as the JSON value `parameter` accepts.
        """
        fix: dict[str, Any] = {"parameter": parameter, "units": [unit_json(u) for u in units]}
        disambiguation = _disambiguation_payload(units)
        if disambiguation is not None:
            fix["disambiguation"] = disambiguation
        names = [unit_name(unit) for unit in units]
        if all(isinstance(unit, str) for unit in units):
            retry = (
                f"Retry with `{parameter}` set to one of these to disambiguate, or filter this "
                "frame_ref with a category_match predicate on unit naming one of these "
                "values first, then retry with the resulting pair as frame_ref."
            )
        else:
            retry = f"Retry with `{parameter}` set to one of these values: {_pin_values(units)}."
        return cls(
            code=RefusalCode.UNIT_HETEROGENEOUS,
            message=f"The input frame carries more than one unit: {names}. {retry}",
            fix=fix,
        )

    @classmethod
    def unit_not_present(
        cls,
        *,
        parameter: str,
        pinned_unit: UnitCell,
        units: Sequence[UnitCell],
    ) -> Self:
        """`parameter` names the calling verb's own unit-pin parameter that
        `pinned_unit` was passed to. `unit_heterogeneous` documents the
        handling of a converted or composed unit in `units`."""
        fix: dict[str, Any] = {
            "parameter": parameter,
            "pinned_unit": unit_json(pinned_unit),
            "units": [unit_json(unit) for unit in units],
        }
        disambiguation = _disambiguation_payload(units)
        if disambiguation is not None:
            fix["disambiguation"] = disambiguation
        names = [unit_name(unit) for unit in units]
        if all(isinstance(unit, str) for unit in units):
            retry = "Retry with one of these instead."
        else:
            retry = f"Retry with `{parameter}` set to one of these values: {_pin_values(units)}."
        return cls(
            code=RefusalCode.UNIT_HETEROGENEOUS,
            message=(
                f"{parameter}={_quoted_unit(pinned_unit)} is not one of the units this frame "
                f"declares: {names}. {retry}"
            ),
            fix=fix,
        )

    @classmethod
    def facet_heterogeneous(cls, *, axis: str, values: list[str]) -> Self:
        fix: dict[str, Any] = {"axis": axis, "values": values}
        disambiguation = _disambiguation_payload(values)
        if disambiguation is not None:
            fix["disambiguation"] = disambiguation
        return cls(
            code=RefusalCode.FACET_HETEROGENEOUS,
            message=(
                f"The input frame carries more than one declared {axis!r}: {values}. "
                "There is no way to pin this from a parameter yet -- retry with a "
                "frame_ref already scoped to one of these."
            ),
            fix=fix,
        )

    @classmethod
    def coverage_insufficient(
        cls, *, realised_coverage_fraction: float, floor: float, skipped_cell_count: int
    ) -> Self:
        return cls(
            code=RefusalCode.COVERAGE_INSUFFICIENT,
            message=(
                f"Only {realised_coverage_fraction:.1%} of the expected grid is present "
                f"({skipped_cell_count} cell(s) missing), below the {floor:.0%} floor. "
                "Narrow group_by to a coarser grain and retry."
            ),
            fix={
                "realised_coverage_fraction": realised_coverage_fraction,
                "floor": floor,
                "skipped_cell_count": skipped_cell_count,
            },
        )

    @classmethod
    def non_finite_result(cls, *, group_key: dict[str, str], aggregation: str) -> Self:
        return cls(
            code=RefusalCode.NON_FINITE_RESULT,
            message=(
                f"The {aggregation} for group {group_key} is not finite (overflow). "
                "Retrying the same call would reach the same result; narrow the input instead."
            ),
            fix={"group_key": group_key, "aggregation": aggregation},
        )

    @classmethod
    def mixed_time_grain(cls, *, grains: list[str], periods: list[str]) -> Self:
        """`grains` names every distinct grain the resolved input's own
        `time_period` column carries, in `year`/`month`/`day` order, and
        `periods` the cells that conflict: one per grain for a grain mix, or
        the spans and a point beside them for a span mix within one grain.
        """
        return cls(
            code=RefusalCode.MIXED_TIME_GRAIN,
            message=(
                f"The input frame's own time_period column mixes {periods} "
                f"(grains {grains}). Every verb here reads a frame whose periods are all "
                "points at one grain, or exactly one span such as 2018/2022, so this frame "
                "cannot be resolved. Retrying the identical frame_ref reaches the identical "
                "column; retry with a frame_ref whose time_period values are points at one "
                "grain, or one span alone."
            ),
            fix={"grains": grains, "periods": periods},
        )

    @classmethod
    def comparison_periods_disjoint(
        cls,
        *,
        left_periods: list[str],
        right_periods: list[str],
        left_label: str = "left",
        right_label: str = "right",
    ) -> Self:
        """`left_label`/`right_label` name the calling verb's two sides in
        the message and in `fix`'s keys: `left` and `right` for `compare` and
        `sum_of`, `numerator` and `denominator` for `quotient_of`."""
        return cls(
            code=RefusalCode.COMPARISON_PERIODS_DISJOINT,
            message=(
                "The two sides share no overlapping time_period. "
                f"{left_label.capitalize()} side has {left_periods}; "
                f"{right_label} side has {right_periods}. Retry with a frame_ref (or entity "
                "binding) whose periods actually overlap the other side's."
            ),
            fix={f"{left_label}_periods": left_periods, f"{right_label}_periods": right_periods},
        )

    @classmethod
    def cross_frame_unit_mismatch(
        cls,
        *,
        left_label: str,
        left_unit: UnitCell,
        right_label: str,
        right_unit: UnitCell,
    ) -> Self:
        """Two independently-resolved sides (`compare`'s left/right,
        `share_of`'s numerator/denominator) carry different units. Distinct
        from `unit_heterogeneous`, which fires when *one* frame itself
        carries more than one unit -- here each side resolved to exactly
        one unit on its own, and it's the pairing across sides that's
        unsafe to combine. When either side holds a converted unit, the
        message asks for both sides again in one unit. A composed unit is
        written by `quotient_of`, so no `get_data` binding and no pin changes
        it; when either side holds one, the message asks for sides divided
        from inputs in matching units.
        """
        units = (left_unit, right_unit)
        if any(isinstance(unit, ComposedUnit) for unit in units):
            retry = (
                "A unit of the form 'X per Y' is the unit quotient_of gives one frame "
                "divided by another, and no unit pin or get_data binding changes it. Retry "
                "with two sides whose units match: each a quotient_of result over inputs in "
                "the same units as the other."
            )
        elif any(isinstance(unit, UnitRequest) for unit in units):
            retry = (
                "Call get_data for both sides with scope.facets binding the same unit, "
                "then retry with the new frame_refs."
            )
        else:
            retry = "Retry with both sides pinned to the same unit."
        return cls(
            code=RefusalCode.CROSS_FRAME_UNIT_MISMATCH,
            message=(
                f"{left_label} resolved to unit {_quoted_unit(left_unit)}, {right_label} to "
                f"{_quoted_unit(right_unit)} -- combining values across different units is "
                f"silently wrong, not merely inconvenient. {retry}"
            ),
            fix={
                left_label: unit_json(left_unit),
                right_label: unit_json(right_unit),
            },
        )

    @classmethod
    def cross_frame_unit_unbound(
        cls, *, unbound_label: str, bound_label: str, bound_unit: UnitCell
    ) -> Self:
        """One side's own rows leave `unit` unbound entirely, while the
        other resolved to a real declared unit. Distinct from
        `cross_frame_unit_mismatch`, which fires when both sides resolve to
        a real, differing unit. Here one side has no declared unit at all,
        so there is no second unit to compare.
        """
        return cls(
            code=RefusalCode.CROSS_FRAME_UNIT_UNBOUND,
            message=(
                f"{unbound_label}'s own rows leave `unit` unbound entirely; {bound_label} "
                f"resolved to unit {_quoted_unit(bound_unit)}. Retry with a {unbound_label} whose "
                "own rows actually declare a unit."
            ),
            fix={
                "unbound_side": unbound_label,
                "bound_side": bound_label,
                "bound_unit": unit_json(bound_unit),
            },
        )

    @classmethod
    def cross_frame_facet_mismatch(
        cls,
        *,
        left_label: str,
        right_label: str,
        axis: str,
        left_ref: str,
        right_ref: str,
        stat_vars: Sequence[str],
    ) -> Self:
        """Two independently-resolved sides (`compare`'s left/right,
        `share_of`'s numerator/denominator) hold the same `stat_vars` and
        disagree on one facet axis: one measure from two sources or methods.
        Distinct from `facet_heterogeneous`, which fires when one side's own
        slice carries more than one declared value. Here each side is
        internally uniform, and it's the pairing across sides that's unsafe.
        Sides holding different StatVars never raise it.
        """
        return cls(
            code=RefusalCode.CROSS_FRAME_FACET_MISMATCH,
            message=(
                f"{left_label} and {right_label} both hold {', '.join(stat_vars)}, but "
                f"{left_label} resolved facet {axis!r} to {left_ref!r} and {right_label} to "
                f"{right_ref!r} -- one measure from two different {axis!r} selections, whose "
                f"values do not combine. Retry with {', '.join(stat_vars)} read under one "
                f"{axis!r} on both sides, {left_ref!r} or {right_ref!r}: with get_data, bind "
                f"scope.facets {axis!r} to that one ref for both sides' frames."
            ),
            fix={"axis": axis, left_label: left_ref, right_label: right_ref},
        )

    @classmethod
    def growth_gap_year(
        cls,
        *,
        window_start: str,
        window_end: str,
        entity_count: int,
        missing_start: int,
        missing_end: int,
        suggested_window: list[str],
        suggested_window_entities: int,
    ) -> Self:
        """No entity has a row in both endpoint years. The counts cover every
        entity the input holds. `suggested_window` is a window
        `suggested_window_entities` entities hold both ends of, `[]` when no
        entity holds two years inside the requested window.
        """
        if suggested_window:
            retry = (
                f"{suggested_window_entities} entit(y/ies) have rows in both "
                f"{suggested_window[0]} and {suggested_window[1]}; retry with window_start and "
                "window_end set to those years."
            )
        else:
            retry = (
                "No entity has rows in two years inside the window; retry with window_start "
                "and window_end set to years the entities report."
            )
        return cls(
            code=RefusalCode.GROWTH_GAP_YEAR,
            message=(
                f"No entity has rows in both {window_start} and {window_end}, and growth reads "
                f"each entity's value in those two endpoint years. Of {entity_count} "
                f"entit(y/ies), {missing_start} have no row in {window_start} and {missing_end} "
                f"have no row in {window_end}. {retry}"
            ),
            fix={
                "window_start": window_start,
                "window_end": window_end,
                "entity_count": entity_count,
                "missing_start": missing_start,
                "missing_end": missing_end,
                "suggested_window": suggested_window,
                "suggested_window_entities": suggested_window_entities,
            },
        )

    @classmethod
    def growth_non_finite_result(
        cls, *, entity_count: int, zero_or_sign_crossing_base: int, non_finite: int
    ) -> Self:
        """Every entity holding both endpoints was excluded for a growth value
        that is undefined or not finite. `non_finite_result`'s own code,
        with growth's per-cause entity counts in place of one group key."""
        causes = []
        if zero_or_sign_crossing_base:
            causes.append(
                f"{zero_or_sign_crossing_base} have a zero or sign-crossing base, which leaves "
                "a relative change undefined (change='absolute' is defined there)"
            )
        if non_finite:
            causes.append(f"{non_finite} compute to a non-finite value")
        return cls(
            code=RefusalCode.NON_FINITE_RESULT,
            message=(
                f"No entity has a finite growth value. Of the {entity_count} entit(y/ies) with "
                f"rows in both endpoint years, {' and '.join(causes)}. Retrying the same call "
                "would reach the same result; change the request instead."
            ),
            fix={
                "entity_count": entity_count,
                "zero_or_sign_crossing_base": zero_or_sign_crossing_base,
                "non_finite": non_finite,
            },
        )

    @classmethod
    def share_of_grain_mismatch(
        cls,
        *,
        numerator_periods: list[str],
        denominator_periods: list[str],
        key: dict[str, str] | None = None,
    ) -> Self:
        """No numerator key shares a period with its denominator group.
        share_of divides each key over the periods both sides report and
        leaves out and names a key sharing none, so this fires only when
        that leaves no key at all. `key` names the call's one key when it
        has exactly one, `{}` (the default) otherwise.
        """
        key = key or {}
        # Both lists empty is a distinct fault: neither side's entity
        # binding matched a row, so there are no two period sets to compare.
        if not numerator_periods and not denominator_periods:
            message = (
                "Neither the numerator's nor the denominator's own entity binding matched "
                "any row -- both sides summed to zero periods. Retry with a numerator_entity "
                "and/or denominator_entity that actually narrows to rows this frame_ref "
                "carries, or a frame_ref that does."
            )
        else:
            for_key = f" for {key}" if key else ""
            message = (
                f"The numerator's periods {numerator_periods} share none with the "
                f"denominator's periods {denominator_periods}{for_key}, so there is no period "
                "to divide. Retry with a numerator and denominator whose years overlap: "
                "narrow either side's frame_ref or entity binding to data covering the "
                "other's years."
            )
        return cls(
            code=RefusalCode.SHARE_OF_GRAIN_MISMATCH,
            message=message,
            fix={
                "numerator_periods": numerator_periods,
                "denominator_periods": denominator_periods,
                "key": key,
            },
        )

    @classmethod
    def share_of_zero_denominator(cls, *, time_period: str) -> Self:
        """A summed-to-exactly-zero denominator is retryable (a different
        `frame_ref` or entity binding for the denominator can produce a
        nonzero sum) -- distinct from `non_finite_result`, where retrying
        the identical call reaches the identical, genuine overflow.
        """
        return cls(
            code=RefusalCode.SHARE_OF_ZERO_DENOMINATOR,
            message=(
                f"The denominator summed to exactly zero for time_period {time_period!r}, "
                "making the share undefined. Retry with a denominator_frame_ref or "
                "denominator_entity that sums to a nonzero value for this period."
            ),
            fix={"time_period": time_period},
        )

    @classmethod
    def share_of_key_unknown(
        cls, *, column: str, numerator_by: list[str], denominator_by_keys: list[str]
    ) -> Self:
        """`denominator_by` is non-empty but leaves `column` (one of
        `numerator_by`'s own columns) out -- a partial mapping has no
        defined pairing for the column(s) it omits."""
        return cls(
            code=RefusalCode.SHARE_OF_KEY_UNKNOWN,
            message=(
                f"denominator_by does not cover numerator_by column {column!r} -- a "
                f"partial mapping has no defined pairing for the column(s) it omits. "
                f"denominator_by names: {denominator_by_keys}; numerator_by: "
                f"{numerator_by}. Retry with denominator_by either empty (a pooled "
                "denominator, one total shared by every key) or naming every "
                "numerator_by column."
            ),
            fix={
                "column": column,
                "numerator_by": numerator_by,
                "denominator_by_keys": denominator_by_keys,
            },
        )

    @classmethod
    def share_of_key_unmatched(
        cls,
        *,
        numerator_keys: list[dict[str, str]],
        denominator_keys: list[dict[str, str]],
        numerator_key_count: int,
        denominator_key_count: int,
    ) -> Self:
        """No numerator key has a denominator group. `numerator_keys` and
        `denominator_keys` are a sorted sample of each side's keys, each a
        `{column: value}` object under that side's own column names, and the
        two counts are how many keys each side holds in all, as
        `quotient_of_no_key_matched` reports them."""
        return cls(
            code=RefusalCode.SHARE_OF_KEY_UNMATCHED,
            message=(
                f"None of the numerator's {numerator_key_count} key(s) has a denominator group "
                f"among the denominator's {denominator_key_count}. Numerator keys include "
                f"{numerator_keys[:_KEY_SAMPLE_IN_MESSAGE]}; denominator keys include "
                f"{denominator_keys[:_KEY_SAMPLE_IN_MESSAGE]}. denominator_by must map each "
                "numerator_by column to the denominator column holding the same entities. "
                "Retry with that mapping, with a denominator_frame_ref (or denominator_entity) "
                "that holds the numerator's entities, or without denominator_by to divide "
                "every key by one total."
            ),
            fix={
                "numerator_keys": numerator_keys,
                "denominator_keys": denominator_keys,
                "numerator_key_count": numerator_key_count,
                "denominator_key_count": denominator_key_count,
            },
        )

    @classmethod
    def quotient_of_zero_denominator(cls, *, time_periods: list[str], cell_count: int) -> Self:
        """Every matched (key, period) has a denominator that sums to exactly
        zero. `cell_count` is how many cells that is and `time_periods` the
        periods they fall in. A call with some nonzero cell returns those and
        records the rest, so it never reaches here."""
        return cls(
            code=RefusalCode.QUOTIENT_OF_ZERO_DENOMINATOR,
            message=(
                f"The denominator summed to exactly zero in all {cell_count} matched cell(s), "
                f"over time_period(s) {time_periods}, so the call has no quotient to return. "
                "Retry with a denominator_frame_ref or denominator_entity that sums to a "
                "nonzero value in these periods."
            ),
            fix={"time_periods": time_periods, "cell_count": cell_count},
        )

    @classmethod
    def quotient_of_no_key_matched(
        cls,
        *,
        numerator_keys: list[dict[str, str]],
        denominator_keys: list[dict[str, str]],
        numerator_key_count: int,
        denominator_key_count: int,
    ) -> Self:
        """No numerator key has a denominator group. `numerator_keys` and
        `denominator_keys` are a sorted sample of each side's keys, each a
        `{column: value}` object under that side's own column names, and the
        two counts are how many keys each side holds in all."""
        return cls(
            code=RefusalCode.QUOTIENT_OF_NO_KEY_MATCHED,
            message=(
                f"None of the numerator's {numerator_key_count} key(s) matches any of the "
                f"denominator's {denominator_key_count}. Numerator keys include "
                f"{numerator_keys[:_KEY_SAMPLE_IN_MESSAGE]}; denominator keys include "
                f"{denominator_keys[:_KEY_SAMPLE_IN_MESSAGE]}. denominator_by must map each "
                "numerator_by column to the denominator column holding the same entities. "
                "Retry with that mapping, with a denominator_frame_ref that holds the "
                "numerator's entities, or without denominator_by to divide every key by one "
                "total."
            ),
            fix={
                "numerator_keys": numerator_keys,
                "denominator_keys": denominator_keys,
                "numerator_key_count": numerator_key_count,
                "denominator_key_count": denominator_key_count,
            },
        )

    @classmethod
    def slice_pair_no_key_matched(
        cls,
        *,
        verb: Literal["compare", "sum_of"],
        left_keys: list[dict[str, str]],
        right_keys: list[dict[str, str]],
        left_key_count: int,
        right_key_count: int,
    ) -> Self:
        """A `compare` or `sum_of` paired by `on` where no member of one side
        pairs with a member of the other. `left_keys` and `right_keys` are a
        sorted sample of each side's members, each a `{column: value}` object
        under that side's own column names, and the two counts are how many
        members each side holds in all, as `quotient_of_no_key_matched`
        reports them."""
        code = (
            RefusalCode.COMPARE_NO_KEY_MATCHED
            if verb == "compare"
            else RefusalCode.SUM_OF_NO_KEY_MATCHED
        )
        return cls(
            code=code,
            message=(
                f"None of the left side's {left_key_count} member(s) pairs with any of the "
                f"right side's {right_key_count}. Left members include "
                f"{left_keys[:_KEY_SAMPLE_IN_MESSAGE]}; right members include "
                f"{right_keys[:_KEY_SAMPLE_IN_MESSAGE]}. on must map each column of data to "
                "the column of other holding the same entities. Retry with that mapping, or "
                "with frames (or entity bindings) that hold the same entities."
            ),
            fix={
                "left_keys": left_keys,
                "right_keys": right_keys,
                "left_key_count": left_key_count,
                "right_key_count": right_key_count,
            },
        )

    @classmethod
    def quotient_of_side_unit_unsupported(
        cls,
        *,
        side: str,
        reason: Literal["dimensionless", "composed", "operation_stat_var"],
        unit: UnitCell | None = None,
        stat_var: str | None = None,
    ) -> Self:
        """`side` (`numerator` or `denominator`) can be no part of the
        result's composed unit. `reason` says why and decides the retry:

        - `dimensionless`: the side is a share or a relative growth rate, or
          a result derived from one. Its values are a ratio already, a per
          cent of its own denominator or a rate of change, not an amount.
        - `composed`: the side resolved to `unit`, a composed unit, and a
          composed unit holds no composed part.
        - `operation_stat_var`: the side's rows declare no unit and their
          stat_var is `stat_var`, a `sum_of` result's own ref, which no graph
          node or stored definition names.

        `fix` carries `unit` for `composed` and `stat_var` for
        `operation_stat_var`.
        """
        fix: dict[str, Any] = {"side": side, "reason": reason}
        if reason == "dimensionless":
            why = (
                f"The {side} is a share or a relative growth rate, or a result derived from "
                "one: its values are a ratio already, not an amount to divide"
            )
            retry = (
                f"Retry with a {side} frame_ref that holds a measured quantity, such as the "
                "frame the share_of or growth call read."
            )
        elif reason == "composed":
            assert unit is not None
            fix["unit"] = unit_json(unit)
            why = (
                f"The {side} resolved to unit {_quoted_unit(unit)}, a quotient already, and "
                "a unit of the form 'X per Y' holds one unit on each side"
            )
            retry = (
                f"Retry with a {side} frame_ref that holds a measure in a single unit, such "
                "as one of the frames the earlier quotient_of call read."
            )
        else:
            fix["stat_var"] = stat_var
            why = (
                f"The {side}'s rows declare no unit and their stat_var {stat_var!r} is a "
                "sum_of result's own ref, which no dataset names"
            )
            retry = (
                f"Retry with a {side} frame_ref get_data returned, whose stat_var names what "
                "it counts; a get_data call can sum several variables into one series itself."
            )
        return cls(
            code=RefusalCode.QUOTIENT_OF_SIDE_UNIT_UNSUPPORTED,
            message=(f"{why}, so it cannot be the {side} part of the result's unit. {retry}"),
            fix=fix,
        )

    @classmethod
    def composed_unit_sum_unsupported(
        cls, *, verb: str, unit: UnitCell, side: str | None = None, sums_a_cell: bool = False
    ) -> Self:
        """`verb` was asked to add values in `unit`, a composed unit. `side`
        is the side that resolved to it (`left` or `right` for `sum_of`,
        `numerator` or `denominator` for `share_of`), and `None` for a verb
        with one input. `fix` carries `side` only when it is set.
        `sums_a_cell` is whether `side` computes over one row per key and
        period and refuses only a cell holding several, as a `share_of`
        side does; the message then names the `<side>_entity`,
        `numerator_by` and `denominator_by` arguments that leave one row."""
        fix: dict[str, Any] = {"verb": verb, "unit": unit_json(unit)}
        if side is None:
            what = f'{verb} with aggregation="sum" adds values'
            instead = 'Retry with aggregation="mean", "max" or "min"'
        elif sums_a_cell:
            fix["side"] = side
            what = f"The {side} side of {verb} sums more than one row for a period, with values"
            instead = (
                f"Narrow that side to one entity with {side}_entity, or key the share with "
                "numerator_by and denominator_by so each key reads one row, or take a mean, "
                "max or min of the figures with aggregate first"
            )
        else:
            fix["side"] = side
            what = f"The {side} side of {verb} holds values"
            instead = "Take a mean, max or min of the figures with aggregate"
        return cls(
            code=RefusalCode.COMPOSED_UNIT_SUM_UNSUPPORTED,
            message=(
                f"{what} in unit {_quoted_unit(unit)}, a quotient, and quotients added "
                f"together are not a quotient in that unit. {instead}. For one figure pooled "
                "across entities, total each measure across them first, then divide the "
                "totals with quotient_of."
            ),
            fix=fix,
        )

    @classmethod
    def filter_predicate_type_mismatch(
        cls, *, column: str, predicate_kind: str, actual_type: str
    ) -> Self:
        return cls(
            code=RefusalCode.FILTER_PREDICATE_TYPE_MISMATCH,
            message=(
                f"A {predicate_kind!r} predicate on column {column!r} doesn't match that "
                f"column's own declared cell type ({actual_type}). Retry with the correct "
                "predicate kind for this column, or a different column."
            ),
            fix={"column": column, "predicate_kind": predicate_kind, "actual_type": actual_type},
        )

    @classmethod
    def filter_predicate_value_type_mismatch(
        cls, *, column: str, actual_type: str, expected_type: str
    ) -> Self:
        """An `ordered_threshold` predicate targeting a legal column, whose
        own `value` carries the wrong operand type for it -- reuses
        `filter_predicate_type_mismatch`'s own code and retry posture: a
        different operand type for the same column is the same kind of
        correction a wrong predicate kind or column is.
        """
        operand_words = {
            "float": "a number",
            "str": "a period string",
            "point": "one period such as 2018",
            "interval": "an interval such as 2018/2022",
        }
        return cls(
            code=RefusalCode.FILTER_PREDICATE_TYPE_MISMATCH,
            message=(
                f"An 'ordered_threshold' predicate's value on column {column!r} must be "
                f"{operand_words.get(expected_type, expected_type)}, not "
                f"{operand_words.get(actual_type, actual_type)}. Retry with "
                f"{operand_words.get(expected_type, expected_type)} for this column."
            ),
            fix={"column": column, "actual_type": actual_type, "expected_type": expected_type},
        )

    @classmethod
    def filter_unit_conversion_unsupported(
        cls, *, column: str, predicate_kind: str, actual_type: str
    ) -> Self:
        """`actual_type` names what the frame's `unit` cells hold beside
        refs: `"UnitRequest"`, `"ComposedUnit"`, or both joined by `" | "`.
        A frame holding a composed unit came from a verb, so its retry pins
        the unit in the verb that reads the frame, where a converted frame's
        retry fetches one unit."""
        if actual_type == "UnitRequest":
            held = "a converted unit"
            select_one = _REFETCH_ONE_UNIT
        else:
            held = (
                "a composed unit"
                if actual_type == "ComposedUnit"
                else "converted and composed units"
            )
            select_one = _PIN_ONE_UNIT
        return cls(
            code=RefusalCode.FILTER_UNIT_CONVERSION_UNSUPPORTED,
            message=(
                f"A {predicate_kind!r} predicate on column {column!r} cannot match this "
                f"frame's units: its unit cells hold {held}, and category_match "
                "compares unit strings. Predicates on value, time_period and role columns "
                "still apply to this frame. Retry without the unit predicate, or, to select "
                f"one unit, {select_one}."
            ),
            fix={"column": column, "predicate_kind": predicate_kind, "actual_type": actual_type},
        )

    @classmethod
    def join_output_collision(cls, *, left_stat_var: str, right_stat_var: str) -> Self:
        return cls(
            code=RefusalCode.JOIN_OUTPUT_COLLISION,
            message=(
                f"Both sides resolved to StatVar {left_stat_var!r} with the same unit and "
                "facets -- the two rows a matched key would emit are indistinguishable, "
                "which Frame.build cannot represent. Retry with a right_frame_ref (or entity "
                "binding) whose stat_var, unit, or facets differ from the left side's."
            ),
            fix={"left_stat_var": left_stat_var, "right_stat_var": right_stat_var},
        )

    @classmethod
    def join_no_common_key_columns(
        cls, *, left_columns: list[str], right_columns: list[str]
    ) -> Self:
        return cls(
            code=RefusalCode.JOIN_NO_COMMON_KEY_COLUMNS,
            message=(
                f"join_keys was left empty, and auto-detection found no role column both "
                f"sides declare: left declares {left_columns}, right declares "
                f"{right_columns}. Joining on time_period alone would pair unrelated "
                "rows. Retry with a right_frame_ref whose own role columns actually "
                "overlap the left side's, or an explicit join_keys naming a column both "
                "sides genuinely share."
            ),
            fix={"left_columns": left_columns, "right_columns": right_columns},
        )

    @classmethod
    def derived_input_unusable(cls, *, span: str) -> Self:
        return cls(
            code=RefusalCode.DERIVED_INPUT_UNUSABLE,
            message=(
                f"This frame_ref names a result whose rows carry the span {span}. growth "
                "reads one row per period, so it has no years to window over a span. Retry "
                "growth over the frame this result was computed from."
            ),
            fix={"span": span},
        )

    @classmethod
    def derived_input_ordinal(cls, *, reason: str) -> Self:
        return cls(
            code=RefusalCode.DERIVED_INPUT_ORDINAL,
            message=(
                "This frame_ref names a derived result whose own value column holds a rank "
                f"position, not a measured value: {reason}. A ratio, a relative growth rate or "
                "a ranking over a rank position is not a meaningful number. growth retries with "
                'change="absolute" over the identical frame_ref -- a rank moves by places, not '
                "by percentage; rank retries over the frame_ref the per-period rank itself "
                "read, with limit set on that one call to keep a top or bottom N; share_of and "
                "quotient_of have no such retry and need a frame_ref that isn't itself a "
                "per-period rank's own output."
            ),
            fix={"reason": reason},
        )

    @classmethod
    def place_levels_mixed(
        cls,
        *,
        column: str,
        types: Sequence[str],
        whole_total: str | None = None,
        whole_class: str | None = None,
        member_count: int = 0,
        members: Sequence[str] = (),
    ) -> Self:
        """`types` are the distinct cells of `column`, sorted and capped by
        the engine that found them, and empty when the frame carries no such
        column. A flow party role's type or kind column (`PARTY_ROLES`) gets a
        message naming the role and the group-total exclusion, since a
        `place_level` filter is not what its refusal asks for. It names a
        search naming the population for a total across the whole side, a
        filter keeping party and residual for a sum of members, and for a
        comparison of members a filter keeping the kinds a kind column's
        `fix.member_kinds` lists. `whole_total` names a whole total of
        `whole_class`, or of no class when None, pooled beside `member_count`
        members it may hold, of which `members` are examples. The message asks
        to keep the whole totals alone, by a filter on the role's whole column,
        or the members alone, by a filter on `column`, and `fix.filters` holds
        both."""
        role = enrichment_role(column, roles=PARTY_ROLES)
        if role is not None and whole_total is not None:
            counted = f"{member_count} member{'' if member_count == 1 else 's'}"
            held = (
                f"a whole total of the {whole_class} {role} parties, and {counted} beside "
                "it are of that class or of none"
                if whole_class is not None
                else f"a whole total of no one class, and it may hold the {counted} beside it"
            )
            whole = whole_column(role)
            return cls(
                code=RefusalCode.PLACE_LEVELS_MIXED,
                message=(
                    f"The {role} parties in this frame cannot be pooled: {whole_total} is "
                    f"{held}, such as {', '.join(members)}. A sum across them would count "
                    "those members twice. Filter the frame with a category_match on "
                    f"{whole!r} keeping {WHOLE_CELL}, to sum the whole totals alone, or a "
                    f"category_match on {column!r} keeping {' and '.join(_POOLED_MEMBER_KINDS)}, "
                    "to sum the members alone, then retry with that result's frame_ref."
                ),
                fix={
                    "column": column,
                    "types": list(types),
                    "whole_total": whole_total,
                    "whole_class": whole_class,
                    "member_count": member_count,
                    "example_members": list(members),
                    "filters": [
                        {"column": whole, "values": [WHOLE_CELL]},
                        {"column": column, "values": list(_POOLED_MEMBER_KINDS)},
                    ],
                },
            )
        if role is not None and not types:
            return cls(
                code=RefusalCode.PLACE_LEVELS_MIXED,
                message=(
                    f"The {role} parties in this frame cannot be pooled or ranked: the frame "
                    f"carries no {column!r} column, so its parties cannot be told apart from "
                    "the group totals that hold them. Retrieve the data again with get_data "
                    f"and place_types=true, which adds {column!r}, and retry with that "
                    f"frame_ref, or filter the frame to a single {role} party."
                ),
                fix={"column": column, "types": []},
            )
        if role is not None:
            return cls(
                code=RefusalCode.PLACE_LEVELS_MIXED,
                message=(
                    f"The {role} parties in this frame cannot be pooled: {column!r} holds "
                    f"{', '.join(types)}. Parties of kind total, such as a DAC regional total, "
                    "are left out of sums and ranks beside their members, and totals alone are "
                    f"summed when {whole_column(role)!r} is true on every row. So a party here has "
                    "no declared kind, or every party is a total and not all are whole. For a "
                    "total across the whole side, search naming the population "
                    f'("{WHOLE_SIDE_POPULATIONS[role]}") with no operation over that side. '
                    f"Otherwise filter the frame with a category_match on {role!r} naming the "
                    f"parties to keep, or a category_match on {column!r}: to sum the members, "
                    f"keep party and residual; to compare or rank them, keep "
                    f"{', '.join(_MEMBER_KINDS)} alone. Then retry with that result's frame_ref; "
                    f"{class_column(role)!r} tells private parties from official ones."
                ),
                fix={"column": column, "types": list(types), "member_kinds": list(_MEMBER_KINDS)},
            )
        return cls(
            code=RefusalCode.PLACE_LEVELS_MIXED,
            message=(
                f"This frame's places share no level a filter can keep: {column!r} holds "
                f"{', '.join(types)}, and no one of {', '.join(sorted(PLACE_LEVELS))} is "
                "among every place's types. A frame that names no place holds the world, "
                "regions, World Bank groups, countries and their subdivisions side by side, "
                "and ranking or summing them together counts a place beside the places inside "
                "it. Filter the frame first with a place_level predicate on "
                f"{column!r} naming one of those levels, such as Country, then retry with that "
                "result's frame_ref. A type outside those levels, such as Place or "
                "ProvisionalNode, names no level, and no place_level filter keeps the places "
                "that carry it alone."
            ),
            fix={"column": column, "types": list(types)},
        )

    @classmethod
    def rolling_anchor_unresolved(
        cls, *, detail: str, reason: str, context: Mapping[str, Any] | None = None
    ) -> Self:
        """`detail` is the machine-readable name of the check that failed and
        `reason` says it in words. `context` adds the values the check read."""
        return cls(
            code=RefusalCode.ROLLING_ANCHOR_UNRESOLVED,
            message=(
                f"The latest period of the data is not established: {reason}. Call "
                "resolve_anchor with the populations and every root frame of the question, "
                "then retry with the anchor_key and the periods it gives."
            ),
            fix={"detail": detail, **(context or {})},
        )

    @classmethod
    def rolling_grain_unsupported(cls, *, reason: str, periods: Sequence[str]) -> Self:
        return cls(
            code=RefusalCode.ROLLING_GRAIN_UNSUPPORTED,
            message=(
                f"A latest period can be taken over calendar years only: {reason}. "
                "State the periods as fixed years instead."
            ),
            fix={"periods": list(periods)},
        )

    @classmethod
    def rolling_populations_differ(
        cls, *, populations: Sequence[Sequence[Mapping[str, Any]]]
    ) -> Self:
        """The rolling op-specs of one recipe carry `populations`, more than one
        distinct list. One latest period is taken over one list, so the recipe
        names no period, whatever each list would give."""
        return cls(
            code=RefusalCode.ROLLING_POPULATIONS_DIFFER,
            message=(
                f"The recipe's rolling op-specs carry {len(populations)} different population "
                "lists, and one latest period is taken over one list. State the same "
                "rolling.population on every rolling op-spec of the recipe and retry."
            ),
            fix={"populations": [list(entry) for entry in populations]},
        )


class AnalysisRefused(Exception):
    """The engine declined the call. `refusal` holds its code, retry mode,
    message and `fix`."""

    def __init__(self, refusal: Refusal) -> None:
        self.refusal = refusal
        super().__init__(f"{refusal.code.value}: {refusal.message}")

    @property
    def code(self) -> RefusalCode:
        return self.refusal.code


@dataclass(frozen=True, kw_only=True)
class CompositeComponent:
    """One component of a composite root frame: what it holds (`role`) and its
    member StatVars, as the root's manifest lists them under `trace.components`."""

    role: str
    stat_vars: tuple[str, ...]


def composite_refusal(
    refusal: Refusal, components: Iterable[Sequence[CompositeComponent]]
) -> Refusal:
    """`refusal`, or, when it refuses an input frame's several StatVars and those are
    members of one of `components`' composites, the refusal naming that composite's
    components and the `get_data` read that sums them.

    `components` holds each input's composite components, empty for an input of
    one table. A verb reading two frames refuses one side's StatVars, so the
    composite whose components hold them is the side refused; a refusal of a plain
    frame beside a composite stays the plain one.
    """
    if refusal.code is not RefusalCode.STAT_VAR_UNRESOLVED:
        return refusal
    distinct = refusal.fix.get("distinct_stat_vars") or []
    if len(distinct) < 2:
        return refusal
    for composite in components:
        members = {stat_var for component in composite for stat_var in component.stat_vars}
        if members.intersection(distinct):
            return Refusal.composite_stat_vars_unsummed(
                distinct_stat_vars=list(distinct),
                components=[
                    {"role": component.role, "stat_vars": len(component.stat_vars)}
                    for component in composite
                ],
            )
    return refusal


@dataclass(frozen=True, kw_only=True)
class _Vocabulary:
    """What a verb refuses and how it names its own parameters.

    `errors` are the engine exceptions the verb refuses. `unit` names its unit
    pin, `entity` the parameter an unknown entity column is blamed on, and
    `composed` how a composed-unit sum names its side: not at all for a verb of
    one input, by `side` for `sum_of`, and by `side` over one cell for
    `share_of`, whose side computes over one row per key and period."""

    errors: tuple[type[Exception], ...]
    unit: str = "unit"
    entity: str = ""
    composed: Literal["none", "side", "cell"] = "none"


_LEFT_OR_RIGHT_UNIT = "left_unit or right_unit"
_NUMERATOR_OR_DENOMINATOR_UNIT = "numerator_unit or denominator_unit"
_NUMERATOR_OR_DENOMINATOR_ENTITY = "numerator_entity or denominator_entity"

_SLICE_PAIR_ERRORS: tuple[type[Exception], ...] = (
    CompareAmbiguousSliceError,
    ComparisonPeriodsDisjointError,
    CompareUnitMismatchError,
    CompareUnitUnboundError,
    CompareFacetMismatchError,
    CompareNonFiniteResultError,
    PairedNoKeyMatchedError,
    PairedUnknownColumnError,
    UnitNotPresentError,
    UnitHeterogeneousError,
    UnknownEntityColumnError,
    FacetHeterogeneousError,
)
_SLICE_PAIR = _Vocabulary(
    errors=_SLICE_PAIR_ERRORS, unit=_LEFT_OR_RIGHT_UNIT, entity="left_entity or right_entity"
)

# Each verb's refused exceptions. `run_aggregate` raises its own
# `StatVarUnresolvedError`; every other verb raises `engine.common`'s.
_VERBS: Mapping[str, _Vocabulary] = {
    "aggregate": _Vocabulary(
        errors=(
            UnsupportedAggregationError,
            ComposedUnitSumError,
            UnknownGroupByColumnError,
            AggregateStatVarUnresolvedError,
            UnitNotPresentError,
            UnitHeterogeneousError,
            FacetHeterogeneousError,
            NonFiniteAggregateResultError,
            PlaceLevelsMixedError,
        )
    ),
    "growth": _Vocabulary(
        errors=(
            UnknownEntityColumnError,
            StatVarUnresolvedError,
            UnitNotPresentError,
            UnitHeterogeneousError,
            FacetHeterogeneousError,
            GrowthAmbiguousEntityError,
            GrowthEndpointsMissingError,
            GrowthNonFiniteResultError,
        ),
        entity="entity_by",
    ),
    "rank": _Vocabulary(
        errors=(
            UnsupportedAggregationError,
            ComposedUnitSumError,
            UnknownEntityColumnError,
            StatVarUnresolvedError,
            UnitNotPresentError,
            UnitHeterogeneousError,
            FacetHeterogeneousError,
            RankNoRowsInWindowError,
            RankAmbiguousEntityError,
            NonFiniteAggregateResultError,
            PlaceLevelsMixedError,
        ),
        entity="rank_by",
    ),
    "filter": _Vocabulary(
        errors=(
            FilterUnknownColumnError,
            FilterPredicateTypeMismatchError,
            FilterPredicateValueTypeMismatchError,
            FilterPlaceLevelUnknownError,
            FilterUnitConversionUnsupportedError,
        )
    ),
    "compare": _SLICE_PAIR,
    "sum_of": _Vocabulary(
        errors=(*_SLICE_PAIR_ERRORS, ComposedUnitSumError),
        unit=_SLICE_PAIR.unit,
        entity=_SLICE_PAIR.entity,
        composed="side",
    ),
    "share_of": _Vocabulary(
        errors=(
            ShareOfGrainMismatchError,
            ShareOfKeyUnknownError,
            ShareOfKeyUnmatchedError,
            ShareOfNonFiniteResultError,
            ShareOfUnitMismatchError,
            ShareOfUnitUnboundError,
            ShareOfFacetMismatchError,
            StatVarUnresolvedError,
            UnitHeterogeneousError,
            UnitNotPresentError,
            UnknownEntityColumnError,
            FacetHeterogeneousError,
            PlaceLevelsMixedError,
            ComposedUnitSumError,
        ),
        unit=_NUMERATOR_OR_DENOMINATOR_UNIT,
        entity=_NUMERATOR_OR_DENOMINATOR_ENTITY,
        composed="cell",
    ),
    "quotient_of": _Vocabulary(
        errors=(
            QuotientOfSideUnitUnsupportedError,
            QuotientOfNoKeyMatchedError,
            QuotientOfPeriodsDisjointError,
            QuotientOfZeroDenominatorError,
            NonFiniteCellError,
            StatVarUnresolvedError,
            UnitNotPresentError,
            UnitHeterogeneousError,
            UnknownEntityColumnError,
            FacetHeterogeneousError,
            PlaceLevelsMixedError,
        ),
        unit=_NUMERATOR_OR_DENOMINATOR_UNIT,
        entity=_NUMERATOR_OR_DENOMINATOR_ENTITY,
    ),
    "join": _Vocabulary(
        errors=(
            JoinUnknownKeyError,
            JoinAmbiguousKeyError,
            JoinOutputCollisionError,
            JoinNoCommonKeyColumnsError,
            JoinResultTooLargeError,
            StatVarUnresolvedError,
            UnitNotPresentError,
            UnitHeterogeneousError,
            FacetHeterogeneousError,
        ),
        unit=_LEFT_OR_RIGHT_UNIT,
    ),
}


def refusal_for_engine_error(verb: str, exc: Exception) -> Refusal | None:
    """The refusal `verb` gives for `exc`, an exception its `run_<verb>` raised, or
    `None` when `exc` is none of the exceptions `verb` refuses, which then
    propagates as a failure of the engine.

    Raises:
        KeyError: `verb` is no verb.
    """
    vocabulary = _VERBS[verb]
    if not isinstance(exc, vocabulary.errors):
        return None
    return _refusal(verb, vocabulary, exc)


def _refusal(verb: str, vocabulary: _Vocabulary, exc: Exception) -> Refusal:
    """`refusal_for_engine_error` for an exception `vocabulary` refuses."""
    match exc:
        case UnsupportedAggregationError():
            return Refusal.operation_uncovered(
                aggregation=exc.aggregation, supported=list(AGGREGATIONS)
            )
        case UnknownGroupByColumnError():
            return Refusal.role_ambiguous(
                parameter="group_by", columns=list(exc.columns), available=list(exc.available)
            )
        case UnknownEntityColumnError():
            return Refusal.role_ambiguous(
                parameter=vocabulary.entity,
                columns=list(exc.columns),
                available=list(exc.available),
            )
        case AggregateStatVarUnresolvedError() | StatVarUnresolvedError():
            return Refusal.stat_var_unresolved(
                reason=exc.reason, distinct_stat_vars=list(exc.distinct_stat_vars)
            )
        case UnitNotPresentError():
            return Refusal.unit_not_present(
                parameter=vocabulary.unit, pinned_unit=exc.pinned_unit, units=list(exc.units)
            )
        case UnitHeterogeneousError():
            return Refusal.unit_heterogeneous(parameter=vocabulary.unit, units=list(exc.units))
        case FacetHeterogeneousError():
            return Refusal.facet_heterogeneous(axis=exc.axis, values=list(exc.values))
        case PlaceLevelsMixedError():
            return Refusal.place_levels_mixed(
                column=exc.column,
                types=list(exc.types),
                whole_total=exc.whole_total,
                whole_class=exc.whole_class,
                member_count=exc.member_count,
                members=list(exc.members),
            )
        case ComposedUnitSumError():
            return Refusal.composed_unit_sum_unsupported(
                verb=verb,
                unit=exc.unit,
                side=None if vocabulary.composed == "none" else exc.side,
                sums_a_cell=vocabulary.composed == "cell",
            )
        case NonFiniteAggregateResultError():
            return Refusal.non_finite_result(group_key=exc.group_key, aggregation=exc.aggregation)
        case GrowthAmbiguousEntityError():
            return Refusal.role_ambiguous(
                parameter="entity_by", columns=list(exc.entity_by), available=list(exc.available)
            )
        case GrowthEndpointsMissingError():
            return Refusal.growth_gap_year(
                window_start=exc.window_start,
                window_end=exc.window_end,
                entity_count=exc.entity_count,
                missing_start=exc.missing_start,
                missing_end=exc.missing_end,
                suggested_window=list(exc.suggested_window),
                suggested_window_entities=exc.suggested_window_entities,
            )
        case GrowthNonFiniteResultError():
            return Refusal.growth_non_finite_result(
                entity_count=exc.entity_count,
                zero_or_sign_crossing_base=exc.zero_or_sign_crossing_base,
                non_finite=exc.non_finite,
            )
        case RankNoRowsInWindowError():
            return Refusal.stat_var_unresolved(reason=exc.reason, distinct_stat_vars=[])
        case RankAmbiguousEntityError():
            return Refusal.role_ambiguous(
                parameter="rank_by", columns=list(exc.rank_by), available=list(exc.available)
            )
        case FilterUnknownColumnError():
            return Refusal.role_ambiguous(
                parameter="predicates", columns=list(exc.columns), available=list(exc.available)
            )
        case FilterPredicateTypeMismatchError():
            return Refusal.filter_predicate_type_mismatch(
                column=exc.column, predicate_kind=exc.predicate_kind, actual_type=exc.actual_type
            )
        case FilterPredicateValueTypeMismatchError():
            return Refusal.filter_predicate_value_type_mismatch(
                column=exc.column, actual_type=exc.actual_type, expected_type=exc.expected_type
            )
        case FilterPlaceLevelUnknownError():
            return Refusal.filter_predicate_type_mismatch(
                column=exc.column, predicate_kind="place_level", actual_type="untyped place entity"
            )
        case FilterUnitConversionUnsupportedError():
            return Refusal.filter_unit_conversion_unsupported(
                column=exc.column, predicate_kind=exc.predicate_kind, actual_type=exc.actual_type
            )
        case PairedNoKeyMatchedError():
            return Refusal.slice_pair_no_key_matched(
                verb="compare" if verb == "compare" else "sum_of",
                left_keys=[dict(key) for key in exc.left_keys],
                right_keys=[dict(key) for key in exc.right_keys],
                left_key_count=exc.left_key_count,
                right_key_count=exc.right_key_count,
            )
        case _:
            return _two_sided_refusal(exc)


def _two_sided_refusal(exc: Exception) -> Refusal:
    """`_refusal` for the exceptions of the verbs that read two sides."""
    match exc:
        case CompareAmbiguousSliceError() if exc.by:
            return Refusal.paired_member_ambiguous(
                side=exc.side,
                members=list(exc.by),
                entity=list(exc.entity),
                available=list(exc.available),
            )
        case CompareAmbiguousSliceError():
            return Refusal.role_ambiguous(
                parameter=f"{exc.side}_entity",
                columns=list(exc.entity),
                available=list(exc.available),
            )
        case PairedUnknownColumnError():
            return Refusal.role_ambiguous(
                parameter="on",
                columns=list(exc.columns),
                available=list(exc.available),
                side=exc.side,
            )
        case ComparisonPeriodsDisjointError():
            return Refusal.comparison_periods_disjoint(
                left_periods=list(exc.left_periods), right_periods=list(exc.right_periods)
            )
        case CompareUnitMismatchError():
            return Refusal.cross_frame_unit_mismatch(
                left_label="left",
                left_unit=exc.left_unit,
                right_label="right",
                right_unit=exc.right_unit,
            )
        case CompareUnitUnboundError() | ShareOfUnitUnboundError():
            return Refusal.cross_frame_unit_unbound(
                unbound_label=exc.unbound_side,
                bound_label=exc.bound_side,
                bound_unit=exc.bound_unit,
            )
        case CompareFacetMismatchError():
            return Refusal.cross_frame_facet_mismatch(
                left_label="left",
                right_label="right",
                axis=exc.axis,
                left_ref=exc.left_ref,
                right_ref=exc.right_ref,
                stat_vars=exc.stat_vars,
            )
        case CompareNonFiniteResultError():
            return Refusal.non_finite_result(
                group_key={"time_period": exc.time_period}, aggregation=exc.quantity
            )
        case ShareOfGrainMismatchError():
            return Refusal.share_of_grain_mismatch(
                numerator_periods=list(exc.numerator_periods),
                denominator_periods=list(exc.denominator_periods),
                key=dict(exc.key),
            )
        case ShareOfKeyUnknownError():
            return Refusal.share_of_key_unknown(
                column=exc.column,
                numerator_by=list(exc.numerator_by),
                denominator_by_keys=list(exc.denominator_by_keys),
            )
        case ShareOfKeyUnmatchedError():
            return Refusal.share_of_key_unmatched(
                numerator_keys=[dict(key) for key in exc.numerator_keys],
                denominator_keys=[dict(key) for key in exc.denominator_keys],
                numerator_key_count=exc.numerator_key_count,
                denominator_key_count=exc.denominator_key_count,
            )
        case ShareOfNonFiniteResultError():
            # A zero denominator is retryable over another denominator; an
            # overflow is not.
            if exc.reason == "zero denominator":
                return Refusal.share_of_zero_denominator(time_period=exc.time_period)
            return Refusal.non_finite_result(
                group_key={"time_period": exc.time_period}, aggregation="share_of"
            )
        case ShareOfUnitMismatchError():
            return Refusal.cross_frame_unit_mismatch(
                left_label="numerator",
                left_unit=exc.numerator_unit,
                right_label="denominator",
                right_unit=exc.denominator_unit,
            )
        case ShareOfFacetMismatchError():
            return Refusal.cross_frame_facet_mismatch(
                left_label="numerator",
                right_label="denominator",
                axis=exc.axis,
                left_ref=exc.numerator_ref,
                right_ref=exc.denominator_ref,
                stat_vars=(exc.stat_var,),
            )
        case QuotientOfSideUnitUnsupportedError():
            return Refusal.quotient_of_side_unit_unsupported(
                side=exc.side, reason=exc.reason, unit=exc.unit
            )
        case QuotientOfNoKeyMatchedError():
            return Refusal.quotient_of_no_key_matched(
                numerator_keys=[dict(key) for key in exc.numerator_keys],
                denominator_keys=[dict(key) for key in exc.denominator_keys],
                numerator_key_count=exc.numerator_key_count,
                denominator_key_count=exc.denominator_key_count,
            )
        case QuotientOfPeriodsDisjointError():
            return Refusal.comparison_periods_disjoint(
                left_periods=list(exc.numerator_periods),
                right_periods=list(exc.denominator_periods),
                left_label="numerator",
                right_label="denominator",
            )
        case QuotientOfZeroDenominatorError():
            return Refusal.quotient_of_zero_denominator(
                time_periods=list(exc.time_periods), cell_count=exc.cell_count
            )
        case NonFiniteCellError():
            # quotient_of leaves a zero denominator out, so this is an overflow.
            return Refusal.non_finite_result(
                group_key={"time_period": exc.time_period}, aggregation="quotient_of"
            )
        case JoinUnknownKeyError():
            return Refusal.role_ambiguous(
                parameter=exc.parameter, columns=list(exc.columns), available=list(exc.available)
            )
        case JoinAmbiguousKeyError():
            return Refusal.role_ambiguous(
                parameter="join_keys", columns=list(exc.key_columns), available=list(exc.available)
            )
        case JoinOutputCollisionError():
            return Refusal.join_output_collision(
                left_stat_var=exc.left_stat_var, right_stat_var=exc.right_stat_var
            )
        case JoinNoCommonKeyColumnsError():
            return Refusal.join_no_common_key_columns(
                left_columns=list(exc.left_columns), right_columns=list(exc.right_columns)
            )
        case JoinResultTooLargeError():
            return Refusal.frame_too_large_on_write(
                reason=f"frame has {exc.row_count} rows, over this store's {exc.ceiling}-row ceiling"
            )
    raise TypeError(f"no refusal for {type(exc).__name__}")
