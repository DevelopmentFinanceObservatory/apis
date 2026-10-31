"""Every recorded engine case, run as a one-node tree through `evaluate`, gives the
frame the engine gives and the refusal code it recorded.

A case's engine arguments become the verb's op-spec (`pinned_unit` is `unit`,
and so on). A case whose arguments state no valid op-spec, such as a growth
window bound that is no calendar year, or whose input frame mixes time grains,
which a verb refuses before its engine runs, has no tree and is left out.
`compare` and `share_of` label their result with the op-spec's own `ops_` id,
so their frames are compared with the engine called under that label.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from dfo_frame_grammar import frame_sha256
from differential.cases import frame_from_spec
from differential.harness import IMPLEMENTATIONS, _engine_kwargs, input_frames, load_fixture
from pydantic import ValidationError

from dfo_analysis_engine.calls import admit
from dfo_analysis_engine.evaluate import Policy, RootFrame, evaluate
from dfo_analysis_engine.frame_ref import FrameRef
from dfo_analysis_engine.identity import OpSpecIdentityError, canonicalize_op_spec
from dfo_analysis_engine.op_spec import OpSpecAdapter
from dfo_analysis_engine.refusal import AnalysisRefused

_PARAMETERS = {
    "pinned_unit": "unit",
    "left_pinned_unit": "left_unit",
    "right_pinned_unit": "right_unit",
    "numerator_pinned_unit": "numerator_unit",
    "denominator_pinned_unit": "denominator_unit",
}
_SECOND_SLOT = {
    "compare": "right_frame_ref",
    "share_of": "denominator_frame_ref",
    "join": "right_frame_ref",
}
_LABELLED = frozenset({"compare", "share_of"})
_ROOTS = [FrameRef(spec_id=f"spec_{c * 32}", content_version=f"cv_{c * 32}") for c in "ab"]


def _tree(record: dict[str, Any]) -> dict[str, Any] | None:
    """`record` as a one-node tree, or `None` when it states no op-spec or reads a
    frame no verb admits."""
    verb = record["verb"]
    parameters = {
        _PARAMETERS.get(name, name): value
        for name, value in record["params"].items()
        if name != "stat_var"
    }
    inputs: dict[str, Any] = {"frame_ref": {"root": 0}}
    if len(record["frames"]) == 2 or verb == "join":
        inputs[_SECOND_SLOT[verb]] = {"root": len(record["frames"]) - 1}
    slots = {slot: _ROOTS[node["root"]].model_dump() for slot, node in inputs.items()}
    try:
        op_spec = OpSpecAdapter.validate_python({"verb": verb, **parameters, **slots})
        canonical = json.loads(canonicalize_op_spec(op_spec))
        for spec in record["frames"]:
            admit(frame_from_spec(spec), content_version="")
    except (ValidationError, OpSpecIdentityError, AnalysisRefused):
        return None
    return {"op_spec": canonical, "inputs": inputs}


_CASES = [
    pytest.param(record, tree, id=record["id"])
    for record in load_fixture()
    if next(iter(record["outcome"])) in ("result", "refusal")
    and (tree := _tree(record)) is not None
]


def test_most_recorded_cases_state_a_tree() -> None:
    assert len(_CASES) >= 400


@pytest.mark.parametrize(("record", "tree"), _CASES)
def test_evaluate_matches_the_engine(record: dict[str, Any], tree: dict[str, Any]) -> None:
    roots = {
        index: RootFrame(ref=_ROOTS[index], frame=frame_from_spec(spec))
        for index, spec in enumerate(record["frames"])
    }
    outcome = record["outcome"]
    if "refusal" in outcome:
        with pytest.raises(AnalysisRefused) as refused:
            evaluate(tree, roots, policy=Policy(coverage_floor=0.0))
        assert refused.value.code.value == outcome["refusal"]["code"]
        return

    evaluation = evaluate(tree, roots, policy=Policy(coverage_floor=0.0))

    if record["verb"] in _LABELLED:
        arguments = {**_engine_kwargs(record), "stat_var": f"dcid:{evaluation.ref.spec_id}"}
        engine = IMPLEMENTATIONS[record["verb"]][0]
        expected = frame_sha256(engine(*input_frames(record), **arguments).frame)
    else:
        expected = outcome["result"]["frame"]["frame_sha256"]
    assert frame_sha256(evaluation.frame) == expected
    assert json.loads(json.dumps(evaluation.summary)) == evaluation.summary
