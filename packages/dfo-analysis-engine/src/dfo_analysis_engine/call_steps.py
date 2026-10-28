"""The step record of one verb call (`steps.Step`): what the call did to its inputs, built from
its engine result as each of dfo-ai's verb tools built it (`steps.build_step`).

A call's step names its inputs by the frame pairs its op-spec's slots hold, its id is the
call's `ops_` id, and its policy is the verb's one fixed policy (`VERB_POLICIES`). A call's
values are a ratio already when its derived marker says so (`calls.DerivedMarker`): a
single-input verb and `sum_of` read the call's own marker, and `compare` and `join` each
input's, since their sides keep their own units.

`compare` and `sum_of` paired member by member (`engine.paired`) record the periods both sides
match, the paired keys, and the keys either side holds that the other does not match or holds
in no shared period, as entities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import JsonValue
from pydantic.experimental.missing_sentinel import MISSING

from dfo_analysis_engine.engine.aggregate import AggregateResult
from dfo_analysis_engine.engine.compare import CompareResult
from dfo_analysis_engine.engine.filter import FilterResult
from dfo_analysis_engine.engine.growth import GrowthResult
from dfo_analysis_engine.engine.join import JoinResult
from dfo_analysis_engine.engine.paired import PairedResult
from dfo_analysis_engine.engine.quotient_of import QuotientOfResult
from dfo_analysis_engine.engine.rank import RankResult
from dfo_analysis_engine.engine.share_of import ShareOfResult
from dfo_analysis_engine.engine.sum_of import SumOfResult
from dfo_analysis_engine.frame_ref import FrameRef
from dfo_analysis_engine.identity import OpSpecVariant
from dfo_analysis_engine.op_spec import (
    AggregateOpSpec,
    CompareOpSpec,
    FilterOpSpec,
    GrowthOpSpec,
    JoinOpSpec,
    QuotientOfOpSpec,
    RankOpSpec,
    ShareOfOpSpec,
    SumOfOpSpec,
)
from dfo_analysis_engine.steps import (
    LeftOut,
    Step,
    build_step,
    caller_params,
    entity_ref,
    left_out_from_exclusions,
    left_out_keys,
    shared_step_unit,
    step_unit,
    window_params,
)
from dfo_analysis_engine.unit_refs import PERCENT_UNIT

__all__ = ["VERB_POLICIES", "call_step"]

VERB_POLICIES: Mapping[str, str] = {
    "aggregate": "aggregate_skip_absent_grain_cells",
    "growth": "growth_endpoint_years",
    "rank": "rank_what_is_present",
    "filter": "filter_row_value_predicates",
    "compare": "compare_intersect_periods",
    "sum_of": "sum_of_intersect_periods",
    "share_of": "share_of_intersect_periods",
    "quotient_of": "quotient_of_intersect_periods",
    "join": "join_inner",
}
"""Each verb's one fixed policy, as dfo-ai's verb tools name it on a trace."""


@dataclass
class _Policy:
    policy_name: str | None


@dataclass
class _Record:
    content: str


@dataclass(frozen=True, kw_only=True)
class _CallTrace:
    """What `build_step` reads of a call (`steps.Trace`)."""

    op_spec_hash: str
    applied_policy: _Policy
    execution_record: _Record


def call_step(
    op_spec: OpSpecVariant,
    result: object,
    *,
    op_spec_hash: str,
    canonical_op_spec: str,
    ratio: bool,
    input_ratios: Sequence[bool],
) -> Step:
    """The step of the call that ran `op_spec` and gave `result`, its engine result.

    Args:
        op_spec: The call's op-spec, its slots holding its inputs' pairs.
        result: The engine's result for `op_spec`.
        op_spec_hash: The call's `ops_` id.
        canonical_op_spec: The op-spec's RFC 8785 canonical JSON text.
        ratio: Whether the call's own values are a ratio already (its derived marker).
        input_ratios: The same of each input, the first slot first.
    """
    trace = _CallTrace(
        op_spec_hash=op_spec_hash,
        applied_policy=_Policy(VERB_POLICIES[str(op_spec.verb)]),
        execution_record=_Record(canonical_op_spec),
    )
    sides_ratio = (input_ratios[0], input_ratios[-1])
    match op_spec, result:
        case AggregateOpSpec(), AggregateResult():
            return _aggregate(op_spec, result, trace=trace, ratio=ratio)
        case GrowthOpSpec(), GrowthResult():
            return _growth(op_spec, result, trace=trace, ratio=ratio)
        case RankOpSpec(), RankResult():
            return _rank(op_spec, result, trace=trace, ratio=ratio)
        case FilterOpSpec(), FilterResult():
            return _filter(op_spec, result, trace=trace, ratio=ratio)
        case CompareOpSpec() | SumOfOpSpec(), PairedResult():
            return _paired(op_spec, result, trace=trace, ratio=ratio)
        case CompareOpSpec(), CompareResult():
            return _compare(op_spec, result, trace=trace, ratio=sides_ratio)
        case SumOfOpSpec(), SumOfResult():
            return _sum(op_spec, result, trace=trace, ratio=ratio)
        case ShareOfOpSpec(), ShareOfResult():
            return _share(op_spec, result, trace=trace)
        case QuotientOfOpSpec(), QuotientOfResult():
            return _quotient(op_spec, result, trace=trace)
        case JoinOpSpec(), JoinResult():
            return _join(op_spec, result, trace=trace, ratio=sides_ratio)
        case _:
            raise TypeError(f"no step for {op_spec.verb} giving {type(result).__name__}")


def _two(first: FrameRef, second: FrameRef | None) -> tuple[FrameRef, FrameRef]:
    """A two-sided call's sides, the first read twice when the op-spec names no second."""
    return first, second or first


def _one_sided(left: Sequence[str], right: Sequence[str]) -> list[LeftOut]:
    """The periods one side alone reports, as one count when there are any."""
    count = len(set(left) ^ set(right))
    return [LeftOut(reason="period_unmatched", count=count, of="periods")] if count else []


def _aggregate(
    op_spec: AggregateOpSpec, result: AggregateResult, *, trace: _CallTrace, ratio: bool
) -> Step:
    return build_step(
        op="aggregate",
        trace=trace,
        inputs={"input": op_spec.frame_ref},
        kept={
            "groups": result.present_cell_count,
            "entities": result.entity_count,
            "periods": len(result.collapsed_periods),
        },
        left_out=left_out_from_exclusions(result.exclusions, row_reasons=("unit_unbound",)),
        unit=step_unit(result.resolved_unit, ratio=ratio),
        period_entities=result.period_entities,
    )


def _growth(op_spec: GrowthOpSpec, result: GrowthResult, *, trace: _CallTrace, ratio: bool) -> Step:
    return build_step(
        op="growth",
        trace=trace,
        inputs={"input": op_spec.frame_ref},
        params=window_params(caller_params(trace.execution_record.content)),
        kept={"entities": len(result.entity_growths)},
        left_out=[
            *(
                LeftOut(reason=reason, count=count, of="entities")
                for reason, count in result.entities_left_out.items()
            ),
            *left_out_from_exclusions(result.exclusions, row_reasons=("unit_unbound",)),
        ],
        unit=step_unit(result.resolved_unit if result.change == "absolute" else None, ratio=ratio),
    )


def _rank(op_spec: RankOpSpec, result: RankResult, *, trace: _CallTrace, ratio: bool) -> Step:
    params = window_params(caller_params(trace.execution_record.content))
    params["order"] = "ascending" if params.pop("ascending", False) else "descending"
    coverage = {
        entity_rank.entity_key: (
            entity_ref(entity_rank.entity_key),
            entity_rank.period_count,
            result.window_period_count,
        )
        for entity_rank in result.kept_ranks
    }
    return build_step(
        op="rank",
        trace=trace,
        inputs={"input": op_spec.frame_ref},
        params=params,
        kept={"entities": len(coverage)},
        left_out=left_out_from_exclusions(result.exclusions, row_reasons=("unit_unbound",)),
        unit=None if result.ordinal else step_unit(result.resolved_unit, ratio=ratio),
        entity_coverage=coverage.values(),
    )


def _kit_predicate(predicate: JsonValue) -> JsonValue:
    """One canonical predicate as a step names it: its kind left implicit in its members, and
    a threshold's `value` as its `operand`."""
    assert isinstance(predicate, dict)
    return {
        "operand" if name == "value" else name: value
        for name, value in predicate.items()
        if name != "kind"
    }


def _filter(op_spec: FilterOpSpec, result: FilterResult, *, trace: _CallTrace, ratio: bool) -> Step:
    params = caller_params(trace.execution_record.content)
    if isinstance(predicates := params.get("predicates"), list):
        params["predicates"] = [_kit_predicate(predicate) for predicate in predicates]
    return build_step(
        op="filter",
        trace=trace,
        inputs={"input": op_spec.frame_ref},
        params=params,
        kept={"entity_periods": result.kept_row_count},
        left_out=[],
        unit=step_unit(result.units[0], ratio=ratio) if len(result.units) == 1 else MISSING,
    )


def _compare(
    op_spec: CompareOpSpec,
    result: CompareResult,
    *,
    trace: _CallTrace,
    ratio: tuple[bool, bool],
) -> Step:
    left, right = _two(op_spec.frame_ref, op_spec.right_frame_ref)
    return build_step(
        op="compare",
        trace=trace,
        inputs={"left": left, "right": right},
        kept={"periods": len(result.intersected_periods)},
        left_out=[
            *_one_sided(result.left_periods, result.right_periods),
            *left_out_from_exclusions(result.exclusions, row_reasons=("unit_unbound",)),
        ],
        unit=shared_step_unit(result.left_resolved_unit, result.right_resolved_unit, ratio=ratio),
        sides={"left": result.left_entities, "right": result.right_entities},
    )


def _sum(op_spec: SumOfOpSpec, result: SumOfResult, *, trace: _CallTrace, ratio: bool) -> Step:
    left, right = _two(op_spec.frame_ref, op_spec.right_frame_ref)
    one_sided = len({entry.time_period for entry in result.left_out})
    return build_step(
        op="sum",
        trace=trace,
        inputs={"left": left, "right": right},
        kept={"periods": len(result.intersected_periods)},
        left_out=[
            *(
                [LeftOut(reason="period_unmatched", count=one_sided, of="periods")]
                if one_sided
                else []
            ),
            *left_out_from_exclusions(result.exclusions, row_reasons=("unit_unbound",)),
        ],
        unit=step_unit(result.unit, ratio=ratio),
    )


def _paired(
    op_spec: CompareOpSpec | SumOfOpSpec,
    result: PairedResult,
    *,
    trace: _CallTrace,
    ratio: bool,
) -> Step:
    left, right = _two(op_spec.frame_ref, op_spec.right_frame_ref)
    compare = isinstance(op_spec, CompareOpSpec)
    return build_step(
        op="compare" if compare else "sum",
        trace=trace,
        inputs={"left": left, "right": right},
        kept={
            "entities": len({cell.key for cell in result.cells}),
            "periods": len(result.matched_periods),
        },
        left_out=[
            *_one_sided(result.left_periods, result.right_periods),
            *left_out_keys(
                (
                    ("unmatched_key", (*result.left_unmatched_keys, *result.right_unmatched_keys)),
                    ("period_unmatched", result.keys_without_shared_period),
                )
            ),
            *left_out_from_exclusions(result.exclusions, row_reasons=("unit_unbound",)),
        ],
        unit=step_unit(result.unit, ratio=ratio),
        sides={"left": result.left_entities, "right": result.right_entities} if compare else None,
    )


def _share(op_spec: ShareOfOpSpec, result: ShareOfResult, *, trace: _CallTrace) -> Step:
    numerator, denominator = _two(op_spec.frame_ref, op_spec.denominator_frame_ref)
    coverage = result.coverage
    return build_step(
        op="share",
        trace=trace,
        inputs={"numerator": numerator, "denominator": denominator},
        kept={
            "entities": coverage.entities,
            "periods": coverage.periods,
            "entities_with_fewer_periods": coverage.entities_with_fewer_periods,
        },
        left_out=[
            *left_out_keys(
                (
                    ("unmatched_key", result.unmatched_keys),
                    ("period_unmatched", result.keys_without_shared_period),
                )
            ),
            *left_out_from_exclusions(result.exclusions),
        ],
        unit=step_unit(PERCENT_UNIT),
        period_entities=coverage.period_entities,
    )


def _quotient(op_spec: QuotientOfOpSpec, result: QuotientOfResult, *, trace: _CallTrace) -> Step:
    numerator, denominator = _two(op_spec.frame_ref, op_spec.denominator_frame_ref)
    coverage = result.coverage
    return build_step(
        op="quotient",
        trace=trace,
        inputs={"numerator": numerator, "denominator": denominator},
        kept={
            "entities": coverage.entities,
            "periods": coverage.periods,
            "entities_with_fewer_periods": coverage.entities_with_fewer_periods,
        },
        left_out=[
            *left_out_keys(
                (
                    ("unmatched_key", result.unmatched_keys),
                    ("period_unmatched", result.keys_without_shared_period),
                    ("zero_denominator", result.keys_with_zero_denominator),
                )
            ),
            *left_out_from_exclusions(result.exclusions, row_reasons=("unit_unbound",)),
        ],
        unit=step_unit(result.unit),
        period_entities=coverage.period_entities,
    )


def _join(
    op_spec: JoinOpSpec, result: JoinResult, *, trace: _CallTrace, ratio: tuple[bool, bool]
) -> Step:
    return build_step(
        op="join",
        trace=trace,
        inputs={"left": op_spec.frame_ref, "right": op_spec.right_frame_ref},
        kept={"entity_periods": len(result.matched_keys)},
        left_out=left_out_from_exclusions(
            result.exclusions, row_reasons=("unit_unbound", "unmatched_key")
        ),
        unit=shared_step_unit(result.left_resolved_unit, result.right_resolved_unit, ratio=ratio),
    )
