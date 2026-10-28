"""One verb call over admitted frames: the checks a verb runs around its engine,
the engine call itself, the `ops_` and `ocv_` ids of its result, and the facts
a caller renders the result from.

`call_verb` takes a validated op-spec and its input frames by slot (`frame_ref`,
and `right_frame_ref` or `denominator_frame_ref` when the op-spec sets one), and
runs the call as the verb's MCP tool runs it once its inputs are resolved:

1. Array parameters are canonicalised as `identity` normalises them before
   minting the `ops_` id (`canonical_columns`, `canonical_predicates`), and
   entity bindings get their keys sorted, so the engine runs in the order the id
   was minted from. A duplicate entry refuses `op_spec_invalid`.
2. The inputs' derived markers are checked (`DerivedMarker`): `rank`, a
   relative `growth`, `share_of` and `quotient_of` refuse a rank position
   (`derived_input_ordinal`), `quotient_of` a side that is a ratio already, and
   `growth` an input labelled with one span (`derived_input_unusable`).
3. The engine runs. A typed exception it raises becomes the verb's refusal
   (`refusal.refusal_for_engine_error`), rewritten for a composite input
   (`refusal.composite_refusal`). `aggregate` and `rank` then refuse a result
   below `coverage_floor` (`coverage_insufficient`).
4. The `ops_` id is minted over the op-spec, and the `ocv_` content version over
   that id, each distinct input `cv_` once, and the engine's execution identity.

A two-sided verb with no second slot reads its first input twice and mints over
one `cv_`. Every refusal is raised as `AnalysisRefused`.

`admit` makes a frame an input: its `time_period` cells must be points at one
grain or exactly one span (`mixed_time_grain`). An input carries what the next
verb reads beside its rows: its `cv_`, its time kind, its derived marker, and
the composite components of the root it descends from through each first slot.

`VerbCall.summary` is a JSON object of the facts a caller renders a digest
from: the `result.VerbResult` fields, the verb, the canonical op-spec's
parameters, the execution identity, the derived marker, and `result`, every
field of the engine's result dataclass but its frame. A `compare` or `sum_of`
summary adds `sides`, the engine side each of the call's `data` and `other`
fills: `compare` gives right minus left, so its `data` is `right`.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, assert_never, cast

import rfc8785
from dfo_frame_grammar import Frame
from dfo_frame_grammar.content_version import frame_sha256, mint_content_version
from dfo_frame_grammar.grammar import (
    TIME_PERIOD_COLUMN,
    ComposedUnit,
    StatVarPart,
    UnitRequest,
    parse_time_period,
)
from pydantic import ValidationError

from dfo_analysis_engine.artifact import Exclusion, Interval, Observation, Time
from dfo_analysis_engine.engine.aggregate import run_aggregate
from dfo_analysis_engine.engine.common import Exclusions
from dfo_analysis_engine.engine.compare import run_compare
from dfo_analysis_engine.engine.filter import run_filter
from dfo_analysis_engine.engine.growth import run_growth
from dfo_analysis_engine.engine.join import MatchedKeys, run_join
from dfo_analysis_engine.engine.paired import run_paired_compare, run_paired_sum_of
from dfo_analysis_engine.engine.quotient_of import run_quotient_of
from dfo_analysis_engine.engine.rank import run_rank
from dfo_analysis_engine.engine.share_of import run_share_of
from dfo_analysis_engine.engine.sum_of import run_sum_of
from dfo_analysis_engine.execution_identity import ExecutionIdentity, current_execution_identity
from dfo_analysis_engine.frame_ref import FrameRef
from dfo_analysis_engine.identity import (
    OP_SPEC_HASH_PATTERN,
    OpSpecIdentityError,
    OpSpecVariant,
    _sorted_unique_predicates,
    _sorted_unique_strings,
    canonicalize_op_spec,
    mint_toolkit_content_version,
    op_spec_hash_for,
)
from dfo_analysis_engine.op_spec import (
    AggregateOpSpec,
    CompareOpSpec,
    FilterOpSpec,
    GrowthOpSpec,
    JoinOpSpec,
    OpSpecAdapter,
    Predicate,
    PredicateAdapter,
    QuotientOfOpSpec,
    RankOpSpec,
    ShareOfOpSpec,
    SumOfOpSpec,
)
from dfo_analysis_engine.refusal import (
    AnalysisRefused,
    CompositeComponent,
    Refusal,
    composite_refusal,
    refusal_for_engine_error,
)
from dfo_analysis_engine.result import VerbResult, observed_shape
from dfo_analysis_engine.time_grain import TimeGrain, distinct_time_grains, is_interval, time_grain

__all__ = [
    "DEFAULT_COVERAGE_FLOOR",
    "DEFAULT_MAX_FRAME_ROWS",
    "DerivedMarker",
    "Input",
    "VerbCall",
    "admit",
    "build_op_spec",
    "call_verb",
    "canonical_columns",
    "canonical_predicates",
    "derive_frame_content_version",
    "mint_identity",
    "op_spec_hash_checked",
]

# The least share of the expected grid an `aggregate` or `rank` result covers.
DEFAULT_COVERAGE_FLOOR = 0.5

# The most rows a `join` may write.
DEFAULT_MAX_FRAME_ROWS = 250_000

_ORDINAL_REASON = (
    "names a per-period rank's own output (or a result derived from one), whose value "
    "column holds a rank position"
)


@dataclass(frozen=True, kw_only=True)
class DerivedMarker:
    """What a verb's result carries for the verbs that read it: whether its values
    are a ratio already (`dimensionless`: a relative growth rate or a share),
    whether it kept its input's StatVar (`inherited_stat_var`, a growth), and
    whether its values are rank positions (`ordinal`, a per-period rank with no
    limit). Every verb carries its inputs' flags onward, OR-ed."""

    dimensionless: bool = False
    inherited_stat_var: bool = False
    ordinal: bool = False
    reason: str = ""

    def detail(self) -> dict[str, object]:
        """The marker as a JSON object, naming only the flags that are set."""
        detail: dict[str, object] = {}
        if self.dimensionless:
            detail["dimensionless"] = True
        if self.inherited_stat_var:
            detail["inherited_stat_var"] = True
        if self.ordinal:
            detail["ordinal"] = True
        if self.reason:
            detail["reason"] = self.reason
        return detail


def _merged_marker(
    inherited: Sequence[DerivedMarker | None],
    *,
    dimensionless: bool = False,
    inherited_stat_var: bool = False,
    ordinal: bool = False,
) -> DerivedMarker | None:
    """The marker of a result whose inputs carry `inherited`, with its own flags
    set, or `None` when no flag is set."""
    present = [marker for marker in inherited if marker is not None]
    merged = DerivedMarker(
        dimensionless=dimensionless or any(marker.dimensionless for marker in present),
        inherited_stat_var=inherited_stat_var
        or any(marker.inherited_stat_var for marker in present),
        ordinal=ordinal or any(marker.ordinal for marker in present),
        reason="; ".join(dict.fromkeys(marker.reason for marker in present if marker.reason)),
    )
    if not (merged.dimensionless or merged.inherited_stat_var or merged.ordinal):
        return None
    return merged


@dataclass(frozen=True, kw_only=True)
class Input:
    """A frame a verb reads, admitted (`admit`).

    `content_version` is the frame-grammar `cv_` the reading verb's `ocv_` is
    minted over. `span` is the one span its periods carry, `None` for points.
    `time` is its time kind, `marker` its derived marker, and `components` the
    composite components of the root it descends from."""

    frame: Frame
    content_version: str
    span: str | None
    time: Time
    marker: DerivedMarker | None = None
    components: tuple[CompositeComponent, ...] = ()


# How many spans a `mixed_time_grain` refusal names when a frame carries more than one.
_NAMED_SPANS = 2


def admit(
    frame: Frame,
    *,
    content_version: str,
    time: Time | None = None,
    marker: DerivedMarker | None = None,
    components: Sequence[CompositeComponent] = (),
) -> Input:
    """`frame` as an input, or the `mixed_time_grain` refusal unless its periods
    are points at one grain or one span alone. `time` defaults to the time kind
    its cells state: an `Interval` of unknown contiguity for one span, since
    nothing says which periods a source collapsed into it, else an `Observation`.

    Raises:
        AnalysisRefused: `mixed_time_grain`.
    """
    periods = cast("list[str]", frame.table.column(TIME_PERIOD_COLUMN).unique().to_pylist())
    _check_periods(periods)
    span = periods[0] if len(periods) == 1 and is_interval(periods[0]) else None
    if time is None:
        if span is None:
            time = Observation()
        else:
            cell = parse_time_period(span)
            time = Interval(start=cell.start, end=cell.end, contiguous=None)
    return Input(
        frame=frame,
        content_version=content_version,
        span=span,
        time=time,
        marker=marker,
        components=tuple(components),
    )


def _check_periods(periods: Sequence[str]) -> None:
    """Refuse `mixed_time_grain` unless `periods` (distinct cells) are points at
    one grain or one span alone, naming one period per grain for a grain mix, or
    two spans and one point for a span beside other periods."""
    first_by_grain: dict[TimeGrain, str] = {}
    spans: list[str] = []
    points: list[str] = []
    for period in sorted(periods):
        cell = parse_time_period(period)
        first_by_grain.setdefault(cell.grain, period)
        if cell.start != cell.end:
            if len(spans) < _NAMED_SPANS:
                spans.append(period)
        elif not points:
            points.append(period)
    grains = distinct_time_grains(first_by_grain.values())
    if len(grains) > 1:
        raise AnalysisRefused(
            Refusal.mixed_time_grain(
                grains=list(grains), periods=[first_by_grain[grain] for grain in grains]
            )
        )
    if spans and len(periods) > 1:
        raise AnalysisRefused(
            Refusal.mixed_time_grain(grains=list(grains), periods=[*spans, *points])
        )


def derive_frame_content_version(frame: Frame) -> str:
    """`frame`'s frame-grammar `cv_`: a hash of its canonical bytes alone."""
    return mint_content_version(frame_sha256=frame_sha256(frame))


_MAX_RENDERED_INPUT_CHARS = 80


def _validation_detail(exc: ValidationError) -> str:
    """One line per error in `exc`: its dotted location and message, and the
    offending input when its `repr` is short enough to read."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        line = f"{location}: {error['msg']}"
        if "input" in error:
            rendered_input = repr(error["input"])
            if len(rendered_input) <= _MAX_RENDERED_INPUT_CHARS:
                line = f"{line} (got {rendered_input})"
        lines.append(line)
    return "\n".join(lines)


def build_op_spec(arguments: Mapping[str, Any]) -> OpSpecVariant:
    """The op-spec `arguments` state, `verb` included.

    Raises:
        AnalysisRefused: `op_spec_invalid`.
    """
    try:
        return OpSpecAdapter.validate_python(dict(arguments))
    except ValidationError as exc:
        raise AnalysisRefused(Refusal.op_spec_invalid(detail=_validation_detail(exc))) from exc


def op_spec_hash_checked(op_spec: OpSpecVariant) -> str:
    """`op_spec`'s `ops_` id.

    Raises:
        AnalysisRefused: `op_spec_invalid`, for a duplicate entry in an array
            parameter or a non-finite number that reached canonicalisation.
    """
    try:
        return op_spec_hash_for(op_spec)
    except (OpSpecIdentityError, rfc8785.CanonicalizationError) as exc:
        raise AnalysisRefused(Refusal.op_spec_invalid(detail=str(exc))) from exc


def mint_identity(
    op_spec: OpSpecVariant, *, input_content_versions: Sequence[str]
) -> tuple[str, ExecutionIdentity, str]:
    """`(ops_ id, execution identity, ocv_ content version)` of `op_spec` over
    `input_content_versions`.

    Raises:
        AnalysisRefused: `op_spec_invalid`, as `op_spec_hash_checked` raises it.
    """
    op_spec_hash = op_spec_hash_checked(op_spec)
    execution_identity = current_execution_identity()
    content_version = mint_toolkit_content_version(
        op_spec_hash=op_spec_hash,
        input_content_versions=input_content_versions,
        execution_identity=execution_identity,
    )
    return op_spec_hash, execution_identity, content_version


def canonical_columns(columns: Sequence[str], *, description: str) -> tuple[str, ...]:
    """`columns` NFC-normalised and sorted, as `identity` normalises the field
    named `description` before minting the `ops_` id.

    Raises:
        AnalysisRefused: `op_spec_invalid`, for a duplicate entry.
    """
    try:
        return tuple(_sorted_unique_strings(columns, description=description))
    except OpSpecIdentityError as exc:
        raise AnalysisRefused(Refusal.op_spec_invalid(detail=str(exc))) from exc


def canonical_predicates(predicates: Sequence[Predicate]) -> tuple[Predicate, ...]:
    """`predicates` in the order `identity` sorts them before minting the `ops_` id.

    Raises:
        AnalysisRefused: `op_spec_invalid`, for a duplicate predicate.
    """
    try:
        canonical = _sorted_unique_predicates(
            [predicate.model_dump(mode="json") for predicate in predicates]
        )
    except OpSpecIdentityError as exc:
        raise AnalysisRefused(Refusal.op_spec_invalid(detail=str(exc))) from exc
    return tuple(PredicateAdapter.validate_python(entry) for entry in canonical)


def _binding(entity: Mapping[str, str]) -> dict[str, str]:
    """An entity binding with its keys sorted, so a refusal lists them in one order."""
    return dict(sorted(entity.items()))


def _input_content_versions(first: Input, second: Input, *, second_set: bool) -> list[str]:
    """The `cv_` values a two-sided verb mints over: the first input's, and the
    second's when its slot is set and it differs."""
    if not second_set or second.content_version == first.content_version:
        return [first.content_version]
    return [first.content_version, second.content_version]


def _run[T](verb: str, engine: Callable[[], T], inputs: Sequence[Input]) -> T:
    """`engine()`, with an exception `verb` refuses raised as its refusal, which
    names the composite components of `inputs` when it refuses their StatVars."""
    try:
        return engine()
    except Exception as exc:
        refusal = refusal_for_engine_error(verb, exc)
        if refusal is None:
            raise
        components = (frame.components for frame in inputs)
        raise AnalysisRefused(composite_refusal(refusal, components)) from exc


def _require_measured(inputs: Sequence[tuple[str, Input]]) -> None:
    """Refuse `derived_input_ordinal` for the first of `inputs` holding rank positions."""
    for label, frame in inputs:
        if frame.marker is not None and frame.marker.ordinal:
            raise AnalysisRefused(
                Refusal.derived_input_ordinal(reason=f"{label} {_ORDINAL_REASON}")
            )


def _check_coverage(fraction: float, *, floor: float, skipped: int) -> None:
    if fraction < floor:
        raise AnalysisRefused(
            Refusal.coverage_insufficient(
                realised_coverage_fraction=fraction, floor=floor, skipped_cell_count=skipped
            )
        )


# Bound on how many missing periods `Interval.absent` names.
_ABSENT_CAP = 10


def _period_count(*, start: str, end: str, grain: TimeGrain) -> int | None:
    """How many periods `start` to `end` spans at `grain`, `None` for a day that
    is no calendar date."""
    if grain == "year":
        return int(end) - int(start) + 1
    if grain == "month":
        return (int(end[:4]) - int(start[:4])) * 12 + (int(end[5:7]) - int(start[5:7])) + 1
    try:
        return date.fromisoformat(end).toordinal() - date.fromisoformat(start).toordinal() + 1
    except ValueError:
        return None


def _periods(*, start: str, end: str, grain: TimeGrain) -> Iterator[str]:
    """Every period from `start` to `end`, at `grain`, one at a time."""
    if grain == "year":
        yield from (str(year) for year in range(int(start), int(end) + 1))
        return
    if grain == "month":
        year, month = int(start[:4]), int(start[5:7])
        last = (int(end[:4]), int(end[5:7]))
        while (year, month) <= last:
            yield f"{year:04d}-{month:02d}"
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return
    current, last_day = date.fromisoformat(start), date.fromisoformat(end)
    while True:
        yield current.isoformat()
        if current >= last_day:
            return
        current += timedelta(days=1)


def _interval(*, start: str, end: str, observed: Sequence[str]) -> Interval:
    """The `Interval` from `start` to `end`, contiguous when `observed` holds
    every period between them, else naming up to `_ABSENT_CAP` absent periods."""
    grain = time_grain(start)
    total = _period_count(start=start, end=end, grain=grain)
    if total is None:
        return Interval(start=start, end=end, contiguous=None)
    observed_set = set(observed)
    absent_count = total - len(observed_set)
    if absent_count <= 0:
        return Interval(start=start, end=end, contiguous=True)
    absent: list[str] = []
    for period in _periods(start=start, end=end, grain=grain):
        if len(absent) >= _ABSENT_CAP:
            break
        if period not in observed_set:
            absent.append(period)
    return Interval(
        start=start, end=end, contiguous=False, absent=tuple(absent), absent_count=absent_count
    )


def _merged_time(own: Interval | None, inputs: Sequence[Input]) -> Time:
    """`own` when the call collapses periods into one span, else the first
    input's `Interval`, else an `Observation`."""
    if own is not None:
        return own
    return next((frame.time for frame in inputs if isinstance(frame.time, Interval)), Observation())


@dataclass(frozen=True, kw_only=True)
class _Outcome:
    """What one verb's engine gave, and what the call adds to it."""

    result: Any
    input_content_versions: list[str]
    marker: DerivedMarker | None
    own_interval: Interval | None = None
    dropped_key_count: int = 0
    right_dropped_key_count: int = 0
    ranked_before_limit: int | None = None
    sides: Mapping[str, str] | None = None


@dataclass(frozen=True, kw_only=True)
class VerbCall:
    """One verb call's result.

    `ref` is the result's `{ops_, ocv_}` pair, `frame_content_version` its
    frame's `cv_`, and `canonical_op_spec` the parsed RFC 8785 op-spec its id is
    minted over. `time`, `marker` and `components` are what a next call reads
    beside the rows (`as_input`). `summary` is described in the module docstring."""

    ref: FrameRef
    frame: Frame
    frame_content_version: str
    canonical_op_spec: dict[str, Any]
    execution_identity: ExecutionIdentity
    exclusions: tuple[Exclusion, ...]
    time: Time
    marker: DerivedMarker | None
    components: tuple[CompositeComponent, ...]
    summary: dict[str, Any] = field(repr=False)

    def as_input(self) -> Input:
        """This result, admitted as the input of a next call.

        Raises:
            AnalysisRefused: `mixed_time_grain`.
        """
        return admit(
            self.frame,
            content_version=self.frame_content_version,
            time=self.time,
            marker=self.marker,
            components=self.components,
        )


def call_verb(
    op_spec: OpSpecVariant,
    inputs: Mapping[str, Input],
    *,
    coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
    max_frame_rows: int = DEFAULT_MAX_FRAME_ROWS,
) -> VerbCall:
    """Run `op_spec` over `inputs`, keyed by frame slot. See the module docstring.

    Raises:
        AnalysisRefused: The call is refused.
    """
    first = inputs["frame_ref"]
    match op_spec:
        case AggregateOpSpec():
            outcome = _aggregate(op_spec, first, coverage_floor=coverage_floor)
        case GrowthOpSpec():
            outcome = _growth(op_spec, first)
        case RankOpSpec():
            outcome = _rank(op_spec, first, coverage_floor=coverage_floor)
        case FilterOpSpec():
            outcome = _filter(op_spec, first)
        case CompareOpSpec() | SumOfOpSpec():
            second = inputs.get("right_frame_ref", first)
            outcome = _slice_pair(op_spec, first, second)
        case ShareOfOpSpec():
            outcome = _share_of(op_spec, first, inputs.get("denominator_frame_ref", first))
        case QuotientOfOpSpec():
            outcome = _quotient_of(op_spec, first, inputs.get("denominator_frame_ref", first))
        case JoinOpSpec():
            outcome = _join(op_spec, first, inputs["right_frame_ref"], max_rows=max_frame_rows)
        case _:
            assert_never(op_spec)
    # The first slot before the second, whatever order the tree lists them in.
    ordered = [first, *(frame for slot, frame in inputs.items() if slot != "frame_ref")]
    return _finished(op_spec, outcome, ordered, components=first.components)


def _finished(
    op_spec: OpSpecVariant,
    outcome: _Outcome,
    inputs: Sequence[Input],
    *,
    components: tuple[CompositeComponent, ...],
) -> VerbCall:
    op_spec_hash, execution_identity, content_version = mint_identity(
        op_spec, input_content_versions=outcome.input_content_versions
    )
    frame: Frame = outcome.result.frame
    frame_content_version = derive_frame_content_version(frame)
    exclusions = _exclusions(outcome.result.exclusions)
    time = _merged_time(outcome.own_interval, inputs)
    role_keys, stat_var_count = observed_shape(frame)
    canonical = json.loads(canonicalize_op_spec(op_spec))
    shape = VerbResult(
        spec_id=op_spec_hash,
        content_version=content_version,
        frame_content_version=frame_content_version,
        rows=frame.num_rows,
        role_keys=role_keys,
        stat_var_count=stat_var_count,
        dropped_key_count=outcome.dropped_key_count,
        right_dropped_key_count=outcome.right_dropped_key_count,
        ranked_before_limit=outcome.ranked_before_limit,
        exclusions=exclusions,
        time=time,
    )
    summary = {
        "verb": str(op_spec.verb),
        **shape.model_dump(mode="json"),
        "parameters": canonical["parameters"],
        "execution_identity": dataclasses.asdict(execution_identity),
        "derived": None if outcome.marker is None else outcome.marker.detail(),
        "result": _facts(outcome.result, top=True),
    }
    if outcome.sides is not None:
        summary["sides"] = dict(outcome.sides)
    return VerbCall(
        ref=FrameRef(spec_id=op_spec_hash, content_version=content_version),
        frame=frame,
        frame_content_version=frame_content_version,
        canonical_op_spec=canonical,
        execution_identity=execution_identity,
        exclusions=exclusions,
        time=time,
        marker=outcome.marker,
        components=components,
        summary=summary,
    )


def _exclusions(exclusions: Exclusions) -> tuple[Exclusion, ...]:
    return tuple(
        Exclusion(
            reason=record.reason,
            rows=record.row_count,
            entities=record.entities,
            entities_truncated=record.entities_truncated,
            side=record.side,
        )
        for record in exclusions.records
    )


def _facts(value: object, *, top: bool = False) -> object:
    """`value`, an engine result or a part of one, as JSON. A frame is left out
    of the top-level result; a unit is its pin form; join's matched keys are
    their rows."""
    if isinstance(value, MatchedKeys):
        value = value.rows()
    if isinstance(value, UnitRequest | ComposedUnit):
        return value.as_json()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _facts(getattr(value, item.name))
            for item in dataclasses.fields(value)
            if not (top and isinstance(getattr(value, item.name), Frame))
        }
    if isinstance(value, Mapping):
        return {str(key): _facts(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_facts(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"no JSON form for {type(value)!r}")


def _aggregate(op_spec: AggregateOpSpec, frame: Input, *, coverage_floor: float) -> _Outcome:
    group_by = canonical_columns(op_spec.group_by, description="group_by")
    result = _run(
        "aggregate",
        lambda: run_aggregate(
            frame.frame,
            group_by=group_by,
            aggregation=op_spec.aggregation,
            pinned_unit=op_spec.unit,
        ),
        [frame],
    )
    _check_coverage(
        result.realised_coverage_fraction, floor=coverage_floor, skipped=result.skipped_cell_count
    )
    collapses = result.period_label is not None and len(result.collapsed_periods) > 1
    return _Outcome(
        result=result,
        input_content_versions=[frame.content_version],
        marker=_merged_marker([frame.marker]),
        own_interval=_interval(
            start=result.collapsed_periods[0],
            end=result.collapsed_periods[-1],
            observed=result.collapsed_periods,
        )
        if collapses
        else None,
    )


def _growth(op_spec: GrowthOpSpec, frame: Input) -> _Outcome:
    entity_by = canonical_columns(op_spec.entity_by, description="entity_by")
    if frame.span is not None:
        raise AnalysisRefused(Refusal.derived_input_unusable(span=frame.span))
    if op_spec.change == "relative":
        _require_measured([("this frame_ref", frame)])
    result = _run(
        "growth",
        lambda: run_growth(
            frame.frame,
            window_start=op_spec.window_start,
            window_end=op_spec.window_end,
            entity_by=entity_by,
            change=op_spec.change,
            pinned_unit=op_spec.unit,
        ),
        [frame],
    )
    return _Outcome(
        result=result,
        input_content_versions=[frame.content_version],
        marker=_merged_marker(
            [frame.marker], dimensionless=op_spec.change == "relative", inherited_stat_var=True
        ),
        own_interval=_interval(
            start=op_spec.window_start, end=op_spec.window_end, observed=result.observed_periods
        ),
    )


def _rank(op_spec: RankOpSpec, frame: Input, *, coverage_floor: float) -> _Outcome:
    rank_by = canonical_columns(op_spec.rank_by, description="rank_by")
    _require_measured([("this frame_ref", frame)])
    result = _run(
        "rank",
        lambda: run_rank(
            frame.frame,
            rank_by=rank_by,
            aggregation=op_spec.aggregation,
            window_start=op_spec.window_start,
            window_end=op_spec.window_end,
            ascending=op_spec.ascending,
            pinned_unit=op_spec.unit,
            per_period=op_spec.per_period,
            limit=op_spec.limit,
        ),
        [frame],
    )
    _check_coverage(
        result.overall_coverage_fraction,
        floor=coverage_floor,
        skipped=result.expected_cell_count - result.present_cell_count,
    )
    window = result.periods_in_window
    collapses = not result.per_period and len(window) > 1
    return _Outcome(
        result=result,
        input_content_versions=[frame.content_version],
        marker=_merged_marker([frame.marker], ordinal=result.ordinal),
        own_interval=_interval(start=min(window), end=max(window), observed=window)
        if collapses
        else None,
        ranked_before_limit=len(result.entity_ranks) if result.limit is not None else None,
    )


def _filter(op_spec: FilterOpSpec, frame: Input) -> _Outcome:
    predicates = canonical_predicates(op_spec.predicates)
    result = _run(
        "filter",
        lambda: run_filter(frame.frame, predicates=predicates, span=frame.span),
        [frame],
    )
    return _Outcome(
        result=result,
        input_content_versions=[frame.content_version],
        marker=_merged_marker([frame.marker]),
    )


def _slice_pair(op_spec: CompareOpSpec | SumOfOpSpec, left: Input, right: Input) -> _Outcome:
    """`compare` or `sum_of`, one series a side or, with `on`, paired member by
    member (`engine.paired`). `data` is compare's right side and sum_of's left."""
    stat_var = f"dcid:{op_spec_hash_checked(op_spec)}"
    compare = isinstance(op_spec, CompareOpSpec)
    arguments: dict[str, Any] = {
        "stat_var": stat_var,
        "left_entity": _binding(op_spec.left_entity),
        "right_entity": _binding(op_spec.right_entity),
        "left_pinned_unit": op_spec.left_unit,
        "right_pinned_unit": op_spec.right_unit,
    }
    if op_spec.on:
        paired = run_paired_compare if compare else run_paired_sum_of
        result = _run(
            str(op_spec.verb),
            lambda: paired(left.frame, right.frame, on=_binding(op_spec.on), **arguments),
            [left, right],
        )
        dropped = (result.left_dropped_key_count, result.right_dropped_key_count)
    else:
        engine = run_compare if compare else run_sum_of
        result = _run(
            str(op_spec.verb), lambda: engine(left.frame, right.frame, **arguments), [left, right]
        )
        dropped = (0, 0)
    return _Outcome(
        result=result,
        input_content_versions=_input_content_versions(
            left, right, second_set=op_spec.right_frame_ref is not None
        ),
        marker=_merged_marker([left.marker, right.marker]),
        dropped_key_count=dropped[0],
        right_dropped_key_count=dropped[1],
        sides={"data": "right", "other": "left"} if compare else {"data": "left", "other": "right"},
    )


def _share_of(op_spec: ShareOfOpSpec, numerator: Input, denominator: Input) -> _Outcome:
    _require_measured(
        [("the numerator frame_ref", numerator), ("the denominator frame_ref", denominator)]
    )
    numerator_by = canonical_columns(op_spec.numerator_by, description="numerator_by")
    stat_var = f"dcid:{op_spec_hash_checked(op_spec)}"
    result = _run(
        "share_of",
        lambda: run_share_of(
            numerator.frame,
            denominator.frame,
            stat_var=stat_var,
            numerator_entity=_binding(op_spec.numerator_entity),
            denominator_entity=_binding(op_spec.denominator_entity),
            numerator_by=numerator_by,
            denominator_by=_binding(op_spec.denominator_by),
            numerator_pinned_unit=op_spec.numerator_unit,
            denominator_pinned_unit=op_spec.denominator_unit,
        ),
        [numerator, denominator],
    )
    return _Outcome(
        result=result,
        input_content_versions=_input_content_versions(
            numerator, denominator, second_set=op_spec.denominator_frame_ref is not None
        ),
        marker=_merged_marker([numerator.marker, denominator.marker], dimensionless=True),
        dropped_key_count=result.dropped_key_count,
    )


def _quotient_of(op_spec: QuotientOfOpSpec, numerator: Input, denominator: Input) -> _Outcome:
    for side, frame in (("numerator", numerator), ("denominator", denominator)):
        _require_measured([(f"the {side} frame_ref", frame)])
        if frame.marker is not None and frame.marker.dimensionless:
            raise AnalysisRefused(
                Refusal.quotient_of_side_unit_unsupported(side=side, reason="dimensionless")
            )
    numerator_by = canonical_columns(op_spec.numerator_by, description="numerator_by")
    stat_var = f"dcid:{op_spec_hash_checked(op_spec)}"
    result = _run(
        "quotient_of",
        lambda: run_quotient_of(
            numerator.frame,
            denominator.frame,
            stat_var=stat_var,
            numerator_entity=_binding(op_spec.numerator_entity),
            denominator_entity=_binding(op_spec.denominator_entity),
            numerator_by=numerator_by,
            denominator_by=_binding(op_spec.denominator_by),
            numerator_pinned_unit=op_spec.numerator_unit,
            denominator_pinned_unit=op_spec.denominator_unit,
        ),
        [numerator, denominator],
    )
    # A side with no unit under an operation's own StatVar names nothing a unit can.
    for side, part in (
        ("numerator", result.unit.numerator),
        ("denominator", result.unit.denominator),
    ):
        if isinstance(part, StatVarPart) and OP_SPEC_HASH_PATTERN.fullmatch(
            part.stat_var.removeprefix("dcid:")
        ):
            raise AnalysisRefused(
                Refusal.quotient_of_side_unit_unsupported(
                    side=side, reason="operation_stat_var", stat_var=part.stat_var
                )
            )
    return _Outcome(
        result=result,
        input_content_versions=_input_content_versions(
            numerator, denominator, second_set=op_spec.denominator_frame_ref is not None
        ),
        marker=_merged_marker([numerator.marker, denominator.marker]),
        dropped_key_count=result.dropped_key_count,
    )


def _join(op_spec: JoinOpSpec, left: Input, right: Input, *, max_rows: int) -> _Outcome:
    join_keys = canonical_columns(op_spec.join_keys, description="join_keys")
    result = _run(
        "join",
        lambda: run_join(
            left.frame,
            right.frame,
            join_keys=join_keys,
            right_keys=op_spec.right_keys,
            left_pinned_unit=op_spec.left_unit,
            right_pinned_unit=op_spec.right_unit,
            max_output_rows=max_rows,
        ),
        [left, right],
    )
    return _Outcome(
        result=result,
        input_content_versions=_input_content_versions(left, right, second_set=True),
        marker=_merged_marker([left.marker, right.marker]),
        dropped_key_count=result.left_unmatched_entity_count,
        right_dropped_key_count=result.right_unmatched_entity_count,
    )
