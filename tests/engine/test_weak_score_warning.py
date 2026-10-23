"""RETRIEVAL_SCORE_WEAK applies to scored standard shapes between the relevance and weak thresholds.

A fixed bind stub isolates warning logic from shape election.
"""

from __future__ import annotations

import asyncio

from search.engine.quantity_read import QuantityOutput
from search.engine.region import RETRIEVAL_SCORE_WEAK, RegionResult, resolve_variable
from search.engine.stated_words_read import StatedWordsOutput
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY, NO_RATING
from tests.fixtures import FakeGraph, make_resolver


class _FixedBindLLM:
    """Elects the sole candidate shape with no constraint bindings, unconditionally."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is StatedWordsOutput:
            return NO_RATING
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, also_plausible=None), None


# The SV carries the populationType and measuredProperty arcs required by derive_shapes.
_SV = "Count_Person"

_NODES: dict = {
    _SV: {
        "label": "Count Person",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Count Person"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
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


def _run(cosine_score: float | None) -> RegionResult:
    """Run resolve_variable; None omits cosine_scores and gives the SV the 1.0 sentinel."""
    detect_entry: dict = {"svs": [_SV]}
    if cosine_score is not None:
        detect_entry["cosine_scores"] = [cosine_score]

    # Use a fixed detect query key; supply one entity so the standard resolver
    # probes obs and returns no_data (obs={}), carrying any accumulated warnings.
    detect: dict = {"exports": detect_entry}
    graph = FakeGraph(nodes=_NODES, detect=detect, resolve=_RESOLVE, obs={})

    return asyncio.run(
        resolve_variable(
            "exports",
            entities=["Kenya"],
            date_request=None,
            detect_query="exports",
            role_query="exports Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def test_weak_score_fires():
    """Score 0.70 is above the 0.65 relevance floor and below the 0.75 weak threshold."""
    result = _run(cosine_score=0.70)
    assert any(w.code == RETRIEVAL_SCORE_WEAK for w in result.warnings)


def test_clean_score_silent():
    """Score 0.80 clears both thresholds and emits no warning."""
    result = _run(cosine_score=0.80)
    assert not any(w.code == RETRIEVAL_SCORE_WEAK for w in result.warnings)


def test_sentinel_1_0_silent():
    """The 1.0 default for omitted cosine_scores suppresses the warning."""
    result = _run(cosine_score=None)
    assert not any(w.code == RETRIEVAL_SCORE_WEAK for w in result.warnings)


_TOPIC = "dc/topic/Exports"
_WEAK_RECALL = "Count_Household"


def _run_expansion(topic_score: float, *, weak_recall_score: float | None = None) -> RegionResult:
    """Run resolve_variable on a recall where `Count_Person` arrives only through a Topic, with
    no score of its own. `weak_recall_score` adds a recalled SV under the floor, which caps the
    expansion's rank below it while its root keeps `topic_score`."""
    nodes: dict = {
        **_NODES,
        _TOPIC: {
            "arcs": {
                "relevantVariable": {"nodes": [{"dcid": _SV, "types": ["StatisticalVariable"]}]}
            }
        },
        _WEAK_RECALL: {
            "label": "Count Household",
            "type": "StatisticalVariable",
            "arcs": {
                "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                "name": {"nodes": [{"value": "Count Household"}]},
                "populationType": {"nodes": [{"dcid": "Household"}]},
                "measuredProperty": {"nodes": [{"dcid": "count"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
            },
        },
    }
    svs, scores = [_TOPIC], [topic_score]
    if weak_recall_score is not None:
        svs.append(_WEAK_RECALL)
        scores.append(weak_recall_score)
    detect: dict = {"exports": {"svs": svs, "cosine_scores": scores}}
    graph = FakeGraph(nodes=nodes, detect=detect, resolve=_RESOLVE, obs={})
    return asyncio.run(
        resolve_variable(
            "exports",
            entities=["Kenya"],
            date_request=None,
            detect_query="exports",
            role_query="exports Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def test_weak_expansion_root_fires():
    """An expansion whose root recalls at 0.70 falls in the weak band as a scored shape does."""
    result = _run_expansion(topic_score=0.70)
    assert any(w.code == RETRIEVAL_SCORE_WEAK for w in result.warnings)


def test_strong_expansion_root_silent():
    result = _run_expansion(topic_score=0.80)
    assert not any(w.code == RETRIEVAL_SCORE_WEAK for w in result.warnings)


def test_weak_root_warns_under_a_capped_rank():
    """A 0.60 recall caps the expansion's rank at 0.5999; its 0.70 root still admits it and
    reads weak."""
    result = _run_expansion(topic_score=0.70, weak_recall_score=0.60)
    assert result.no_data_reason != "variable_not_resolved"
    assert any(w.code == RETRIEVAL_SCORE_WEAK for w in result.warnings)


def test_strong_root_stays_silent_under_a_capped_rank():
    result = _run_expansion(topic_score=0.80, weak_recall_score=0.60)
    assert result.no_data_reason != "variable_not_resolved"
    assert not any(w.code == RETRIEVAL_SCORE_WEAK for w in result.warnings)
