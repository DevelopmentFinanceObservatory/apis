"""`evaluate`: a recipe tree computed over its root frames.

A recipe tree is the `tree` of a recipe (`recipe`): a derived node is
`{"op_spec", "inputs"}`, `op_spec` a parsed canonical op-spec and `inputs` its
frame slots, each to the node of the frame in it; a root node is `{"root": i}`.
A node's `frame` pair, when present, is not read. `input_frames` gives each
root index its frame (`RootFrame`).

The tree is computed from its roots up. Each derived node's op-spec is rebuilt
with its slots naming its inputs' new pairs, and run as the verb's call runs it
(`calls.call_verb`), so each node's frame, `ops_` id and `ocv_` content version
are those the verbs give called one at a time on the same frames.

A tree whose op-specs hold a rolling overlay has its latest period found once,
before any verb runs (`anchor.latest_period`), over the root frames a path
through a rolling op-spec reaches (`rolling.rolling_roots`) and the one
population list every rolling op-spec carries. Rolling op-specs that carry
different lists refuse `rolling_populations_differ`, even where the lists give
one period, since the recipe then names no single latest period. Under
`Policy(latest=False)` each rolling literal must already be the latest period
at its offset, else the tree refuses `rolling_anchor_unresolved` with detail
`literal_differs`. Under `Policy(latest=True)` each literal is rewritten to it,
so the tree follows newer data.

Every refusal is raised as `refusal.AnalysisRefused`.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from dfo_frame_grammar import Frame, FrameError, RollingError, shift_period
from pydantic import TypeAdapter, ValidationError

from dfo_analysis_engine.anchor import ResolvedPeriods, latest_period
from dfo_analysis_engine.artifact import Exclusion
from dfo_analysis_engine.calls import (
    DEFAULT_COVERAGE_FLOOR,
    DEFAULT_MAX_FRAME_ROWS,
    Input,
    VerbCall,
    admit,
    build_op_spec,
    call_verb,
)
from dfo_analysis_engine.execution_identity import ExecutionIdentity
from dfo_analysis_engine.frame_ref import FrameRef
from dfo_analysis_engine.op_spec import Populations
from dfo_analysis_engine.recipe import derived_nodes, pair
from dfo_analysis_engine.refusal import AnalysisRefused, CompositeComponent, Refusal
from dfo_analysis_engine.rolling import canonical_populations, rolling_roots

__all__ = ["Evaluation", "Policy", "RootFrame", "evaluate"]

_ROLLING = "rolling"

_POPULATIONS: TypeAdapter[Populations] = TypeAdapter(Populations)


@dataclass(frozen=True, kw_only=True)
class RootFrame:
    """A root frame: the `{spec_id, content_version}` pair a tree's op-specs
    name it by, its rows, and, for a composite frame, its components as its
    manifest lists them. The pair's `content_version` is the `cv_` a verb
    reading the frame mints over."""

    ref: FrameRef
    frame: Frame
    components: tuple[CompositeComponent, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Policy:
    """How a tree is evaluated.

    `latest` rewrites each rolling literal to the latest period of the roots,
    where `False` requires it to be that period already. `coverage_floor` is the
    least share of the expected grid an `aggregate` or `rank` result covers, and
    `max_frame_rows` the most rows a `join` writes."""

    latest: bool = False
    coverage_floor: float = DEFAULT_COVERAGE_FLOOR
    max_frame_rows: int = DEFAULT_MAX_FRAME_ROWS


_EXACT = Policy()


@dataclass(frozen=True, kw_only=True)
class Evaluation:
    """A tree's result.

    `ref` and `frame` are the top node's `{ops_, ocv_}` pair and frame, and
    `exclusions`, `execution_identity` and `summary` (`calls.VerbCall.summary`)
    its call's. `resolved_periods` is the latest period the rolling op-specs
    count from, `None` for a tree of fixed periods. `tree` is the tree as
    computed: each derived node with its `frame` pair and its canonical op-spec
    over its inputs' pairs, and each root node with its root frame's pair."""

    ref: FrameRef
    frame: Frame
    resolved_periods: ResolvedPeriods | None
    exclusions: tuple[Exclusion, ...]
    execution_identity: ExecutionIdentity
    summary: dict[str, Any] = field(repr=False)
    tree: dict[str, Any] = field(repr=False)


def evaluate(
    recipe: Mapping[str, Any],
    input_frames: Mapping[int, RootFrame],
    *,
    policy: Policy = _EXACT,
) -> Evaluation:
    """Compute the recipe tree `recipe` over `input_frames`. See the module docstring.

    Args:
        recipe: The recipe tree. Its top node is derived.
        input_frames: The frame of every root index the tree names.
        policy: How rolling literals, coverage and output size are treated.

    Raises:
        AnalysisRefused: A node, or the tree's latest period, is refused.
        ValueError: The top node is a root, or a root index has no frame.
    """
    if "root" in recipe:
        raise ValueError("a recipe tree's top node is derived")
    roots = _admitted_roots(recipe, input_frames)
    periods = _latest_period(recipe, roots)
    walk = _Walk(roots=roots, periods=periods, policy=policy)
    call, tree = walk.derived(recipe)
    return Evaluation(
        ref=call.ref,
        frame=call.frame,
        resolved_periods=periods,
        exclusions=call.exclusions,
        execution_identity=call.execution_identity,
        summary=call.summary,
        tree=tree,
    )


@dataclass(frozen=True, kw_only=True)
class _Root:
    ref: FrameRef
    input: Input


def _root_indices(tree: Mapping[str, Any]) -> Iterator[int]:
    if "root" in tree:
        yield tree["root"]
        return
    for node in tree["inputs"].values():
        yield from _root_indices(node)


def _admitted_roots(
    tree: Mapping[str, Any], input_frames: Mapping[int, RootFrame]
) -> dict[int, _Root]:
    """Every root of `tree`, admitted, by index."""
    admitted: dict[int, _Root] = {}
    for index in _root_indices(tree):
        if index in admitted:
            continue
        if index not in input_frames:
            raise ValueError(f"no frame is given for root {index}")
        root = input_frames[index]
        admitted[index] = _Root(
            ref=root.ref,
            input=admit(
                root.frame, content_version=root.ref.content_version, components=root.components
            ),
        )
    return admitted


def _latest_period(tree: Mapping[str, Any], roots: Mapping[int, _Root]) -> ResolvedPeriods | None:
    """The latest period of a rolling `tree`, `None` for a tree of fixed periods."""
    stated: list[Any] = []
    for node in derived_nodes(tree):
        overlay = node["op_spec"]["parameters"].get(_ROLLING)
        if isinstance(overlay, Mapping) and overlay.get("population") not in stated:
            stated.append(overlay.get("population"))
    if not stated:
        return None
    if len(stated) > 1:
        raise AnalysisRefused(Refusal.rolling_populations_differ(populations=stated))
    try:
        populations = _POPULATIONS.validate_python(canonical_populations(stated[0]))
    except (ValidationError, ValueError, KeyError, TypeError) as exc:
        raise AnalysisRefused(Refusal.op_spec_invalid(detail=f"rolling.population: {exc}")) from exc
    reached: dict[tuple[str, str], tuple[str, Input]] = {}
    for node in rolling_roots(tree):
        root = roots[node["root"]]
        reached.setdefault(
            (root.ref.spec_id, root.ref.content_version), (root.ref.spec_id, root.input)
        )
    return latest_period(populations, list(reached.values()))


# The window bounds a rolling overlay can govern, in the order they are checked.
_WINDOW_BOUNDS = ("window_start", "window_end")


def _is_rolling_ref(value: object) -> bool:
    return isinstance(value, Mapping) and "anchor" in value


def _anchored(parameters: Mapping[str, Any], *, period: str, latest: bool) -> dict[str, Any]:
    """`parameters` of a rolling op-spec on the latest `period`: each literal a
    rolling reference governs (a window bound named in `rolling`, and a rolling
    predicate's `value`) rewritten to `period` at its offset under `latest`, else
    checked to be it.

    Raises:
        AnalysisRefused: `rolling_anchor_unresolved`, with detail `literal_differs`
            for a literal that is not the period at its offset, or
            `literal_unshiftable` for an offset that moves the period off the calendar.
    """
    anchored = copy.deepcopy(dict(parameters))
    overlay = anchored[_ROLLING]
    for name in _WINDOW_BOUNDS:
        if _is_rolling_ref(reference := overlay.get(name)):
            _anchor(anchored, name, reference, place=name, period=period, latest=latest)
    for predicate in anchored.get("predicates", ()):
        if _is_rolling_ref(reference := predicate.get(_ROLLING)):
            place = f"the {predicate['op']} threshold on {predicate['column']}"
            _anchor(predicate, "value", reference, place=place, period=period, latest=latest)
    return anchored


def _anchor(
    holder: dict[str, Any],
    key: str,
    reference: Mapping[str, Any],
    *,
    place: str,
    period: str,
    latest: bool,
) -> None:
    """Rewrite or check `holder[key]`, the literal at `place`. See `_anchored`."""
    offset = int(reference["offset"])
    try:
        expected: str | None = shift_period(period, offset)
    except (RollingError, FrameError):
        expected = None
    literal = holder[key]
    if latest and expected is not None:
        holder[key] = expected
        return
    if literal == expected:
        return
    raise AnalysisRefused(
        Refusal.rolling_anchor_unresolved(
            detail="literal_unshiftable" if expected is None else "literal_differs",
            reason=(
                f"{place} is {literal}, and the latest period {period} at offset "
                f"{offset} is {expected}"
            ),
            context={"period": period, "offset": offset, "literal": literal, "expected": expected},
        )
    )


@dataclass(kw_only=True)
class _Walk:
    roots: Mapping[int, _Root]
    periods: ResolvedPeriods | None
    policy: Policy

    def derived(self, node: Mapping[str, Any]) -> tuple[VerbCall, dict[str, Any]]:
        """`node` computed, and its computed tree."""
        inputs: dict[str, Input] = {}
        pairs: dict[str, dict[str, str]] = {}
        trees: dict[str, dict[str, Any]] = {}
        for slot, child in node["inputs"].items():
            if "root" in child:
                root = self.roots[child["root"]]
                inputs[slot] = root.input
                pairs[slot] = pair(root.ref)
                trees[slot] = {"frame": pair(root.ref), "root": child["root"]}
            else:
                call, trees[slot] = self.derived(child)
                inputs[slot] = call.as_input()
                pairs[slot] = pair(call.ref)
        op_spec = node["op_spec"]
        parameters: Mapping[str, Any] = op_spec["parameters"]
        if self.periods is not None and isinstance(parameters.get(_ROLLING), Mapping):
            parameters = _anchored(
                parameters, period=self.periods.period, latest=self.policy.latest
            )
        built = build_op_spec({**parameters, **pairs, "verb": op_spec["verb"]})
        call = call_verb(
            built,
            inputs,
            coverage_floor=self.policy.coverage_floor,
            max_frame_rows=self.policy.max_frame_rows,
        )
        tree = {"frame": pair(call.ref), "op_spec": call.canonical_op_spec, "inputs": trees}
        return call, tree
