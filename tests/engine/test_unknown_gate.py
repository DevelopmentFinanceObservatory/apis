"""A STANDARD shape below SEARCH_UNKNOWN_SCORE_THRESHOLD returns variable_not_resolved.

Tests call resolve_variable with an inline FakeGraph and skip LLM binding.
"""

from __future__ import annotations

import asyncio

from search.engine.bind import _UniversalBindOutput
from search.engine.quantity_read import QuantityOutput
from search.engine.region import RegionResult, resolve_variable
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

# The fixture SV uses the standard resolve path.
_SV = "Count_Person_LeftHanded"

_NODES: dict = {
    _SV: {
        "label": "Count of left-handed persons",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Count of left-handed persons"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    # EntityResolver confirms resolved dcids with a typeOf/name graph read.
    "country/FRA": {"label": "France", "type": "Country"},
}

_RESOLVE: dict = {"France": "country/FRA"}


class _DeclineLLM:
    """Declines the bind menu and names no not-electable reading."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, ask="no reading is a rate"), None


def _run(cosine_score: float, llm=None) -> RegionResult:
    detect: dict = {"left-handedness rate": {"svs": [_SV], "cosine_scores": [cosine_score]}}
    graph = FakeGraph(nodes=_NODES, detect=detect, resolve=_RESOLVE, obs={})

    return asyncio.run(
        resolve_variable(
            "left-handedness rate",
            entities=["France"],
            date_request=None,
            detect_query="left-handedness rate",
            role_query="left-handedness rate in France",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=llm or FakeLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def test_below_threshold_returns_variable_not_resolved():
    """A score below the 0.65 threshold returns variable_not_resolved."""
    result = _run(cosine_score=0.60)
    assert result.status == "no_data"
    assert result.no_data_reason == "variable_not_resolved"


def test_below_threshold_skips_materialise():
    """The gate returns before materialise or answer, so timing_by_step omits both."""
    result = _run(cosine_score=0.60)
    assert "materialise" not in result.timing_by_step
    assert "answer" not in result.timing_by_step


def test_at_or_above_threshold_refuses_a_rate_without_a_denominator_shape():
    """A 0.70 score clears the floor, but the count serves no rate, so no reading is
    admissible and the part refuses on the form. A model that names no not-electable
    reading leaves nothing flagged."""
    result = _run(cosine_score=0.70, llm=_DeclineLLM())
    assert result.no_data_reason == "denominator_not_available"
    assert not result.flagged_match
