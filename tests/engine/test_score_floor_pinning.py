"""Issue #5: low-confidence recall and shapes must end as no data, never false candidates.

The recall floor filters candidates before shape creation; the unknown-score threshold filters
shapes individually. These tests cover both pipeline dead ends, including mixed and empty pools.
"""

from __future__ import annotations

import asyncio

from search.engine.bind import _UniversalBindOutput
from search.engine.config import settings
from search.engine.region import RegionResult, resolve_variable
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

# ---------------------------------------------------------------------------
# Shared node fixtures: two SVs with distinct five-tuples, so each forms its
# own ShapeDraft under derive_shapes' five-tuple grouping.
# ---------------------------------------------------------------------------

_SV_WEAK = "Count_Person_WeakSignalMetric"
_SV_STRONG = "Amount_EconomicActivity_StrongSignalMetric"

_WEAK_LABEL = "Weak Signal Metric"
_STRONG_LABEL = "Strong Signal Metric"

_NODES: dict = {
    _SV_WEAK: {
        "label": _WEAK_LABEL,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": _WEAK_LABEL}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    _SV_STRONG: {
        "label": _STRONG_LABEL,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": _STRONG_LABEL}]},
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    "country/KEN": {
        "label": "Kenya",
        "type": "Country",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "Country"}]},
            "name": {"nodes": [{"value": "Kenya"}]},
        },
    },
}

_RESOLVE: dict = {"Kenya": "country/KEN"}

# Derived from the config constant rather than hardcoded, so recalibrating
# settings().unknown_score_threshold keeps these tests meaningful.
_BELOW_THRESHOLD = settings().unknown_score_threshold - 0.1
_ANOTHER_BELOW_THRESHOLD = settings().unknown_score_threshold - 0.2
_AT_OR_ABOVE_THRESHOLD = settings().unknown_score_threshold + 0.1


class _UnreachableLLM:
    """Fails if the pipeline calls bind after reaching a dead end."""

    async def generate_structured(self, **_kwargs):
        raise AssertionError("bind_universal must not run once the pipeline has already dead-ended")


class _RecordingBindLLM:
    """Elects capped_shapes[0] unconditionally and records every prompt it is
    called with, so a test can assert on exactly which/how many shapes made
    it into the candidate set bind_universal was shown (the eligibility
    filter's real, external effect -- see bind._build_universal_bind_prompt's
    "CANDIDATE SHAPES (N total)" header and per-shape "label:" line).
    """

    def __init__(self):
        self.prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        self.prompts.append(prompt)
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[],
            also_plausible=None,
            ask=None,
        ), None


# ---------------------------------------------------------------------------
# Mechanism 2 (settings().unknown_score_threshold): the mixed case that proves
# eligible_shapes is a PER-SHAPE filter, not a check on the single top-scoring
# shape. A future refactor that collapsed the list comprehension in
# pipeline.py:513-520 into "look only at the best shape" would still pass
# test_unknown_gate.py's single-shape cases but must fail this one.
# ---------------------------------------------------------------------------


def _run_mixed_shapes() -> tuple[RegionResult, _RecordingBindLLM]:
    detect_query = "mixed confidence signal metric"
    detect = {
        detect_query: {
            "svs": [_SV_STRONG, _SV_WEAK],
            "cosine_scores": [_AT_OR_ABOVE_THRESHOLD, _BELOW_THRESHOLD],
        }
    }
    graph = FakeGraph(nodes=_NODES, detect=detect, resolve=_RESOLVE, obs={})
    llm = _RecordingBindLLM()
    result = asyncio.run(
        resolve_variable(
            detect_query,
            entities=["Kenya"],
            date_request=None,
            detect_query=detect_query,
            role_query=f"{detect_query} in Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=llm,
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )
    return result, llm


class TestMixedShapesEligibilityIsPerShape:
    """Two candidate shapes recalled for the same variable: one below
    settings().unknown_score_threshold, one at/above it. Pins that the query
    proceeds (the below-threshold sibling does not drag the whole variable
    down to variable_not_resolved) AND that the below-threshold shape never
    reaches bind_universal at all -- proving each shape is filtered
    independently rather than the query being gated on the pool's single best
    (or single first) shape.
    """

    def test_proceeds_past_the_gate(self):
        result, _llm = _run_mixed_shapes()
        assert result.no_data_reason != "variable_not_resolved"

    def test_only_the_qualifying_shape_reaches_bind(self):
        _result, llm = _run_mixed_shapes()
        assert len(llm.prompts) == 1
        prompt = llm.prompts[0]
        assert "CANDIDATE SHAPES (1 total)" in prompt
        assert _STRONG_LABEL in prompt
        assert _WEAK_LABEL not in prompt


# ---------------------------------------------------------------------------
# Mechanism 2: every shape below threshold (multiple shapes) -> eligible_shapes
# ends up empty -> variable_not_resolved, and bind never runs. Complements
# test_unknown_gate.py's single-shape case: with 2+ shapes
# in the pool, a regression that only checked "at least one shape formed"
# (rather than "at least one ELIGIBLE shape") would wrongly proceed here.
# ---------------------------------------------------------------------------


def _run_all_below_threshold() -> RegionResult:
    detect_query = "uniformly weak signal metric"
    detect = {
        detect_query: {
            "svs": [_SV_STRONG, _SV_WEAK],
            "cosine_scores": [_BELOW_THRESHOLD, _ANOTHER_BELOW_THRESHOLD],
        }
    }
    graph = FakeGraph(nodes=_NODES, detect=detect, resolve=_RESOLVE, obs={})
    return asyncio.run(
        resolve_variable(
            detect_query,
            entities=[],
            date_request=None,
            detect_query=detect_query,
            role_query=detect_query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_UnreachableLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


class TestAllShapesBelowThresholdDeadEnds:
    def test_status_is_no_data_variable_not_resolved(self):
        result = _run_all_below_threshold()
        assert result.status == "no_data"
        assert result.no_data_reason == "variable_not_resolved"
        assert result.status != "candidates"

    def test_bind_and_materialise_never_run(self):
        result = _run_all_below_threshold()
        assert "bind" not in result.timing_by_step
        assert "materialise" not in result.timing_by_step
        assert "answer" not in result.timing_by_step


# ---------------------------------------------------------------------------
# Mechanism 1 (settings().recall_score_floor): the floor empties recall()'s own
# candidate pool before derive_shapes ever runs. This is the OTHER
# variable_not_resolved dead-end (pipeline.py:498, the "not shapes_found"
# branch), never reaching the eligible_shapes gate at all. Distinct code path
# from every test above; TestRelevanceFloor in test_interpret.py already pins
# recall()'s own candidate-dropping behaviour in isolation, this pins that the
# emptied pool actually propagates to the pipeline's own dead-end.
# ---------------------------------------------------------------------------

_BELOW_RECALL_FLOOR = settings().recall_score_floor / 2


def _run_recall_floor_empties_pool() -> RegionResult:
    detect_query = "unresolvable noise metric"
    detect = {detect_query: {"svs": ["sv/PureNoise"], "cosine_scores": [_BELOW_RECALL_FLOOR]}}
    graph = FakeGraph(nodes={}, detect=detect, resolve={}, obs={})
    return asyncio.run(
        resolve_variable(
            detect_query,
            entities=[],  # no entity named in this query
            date_request=None,
            detect_query=detect_query,
            role_query=detect_query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_UnreachableLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


class TestRecallFloorEmptiesPoolDeadEnds:
    def test_status_is_no_data_variable_not_resolved(self):
        result = _run_recall_floor_empties_pool()
        assert result.status == "no_data"
        assert result.no_data_reason == "variable_not_resolved"
        assert result.status != "candidates"

    def test_bind_and_materialise_never_run(self):
        result = _run_recall_floor_empties_pool()
        assert "bind" not in result.timing_by_step
        assert "materialise" not in result.timing_by_step
        assert "answer" not in result.timing_by_step
