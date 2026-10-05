"""Materialisation can elect a close-scoring qualifier variant when a shape has no data.

Tests cover score margins, semantic guards, and unscored expansion shapes.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.region import phase_answer, resolve_variable
from search.engine.region.codes import (
    ELECTED_SHAPE_NO_DATA,
    MEMBERSHIP_UNVERIFIED,
    PLACE_SCOPE_DEFAULTED,
)
from search.engine.region.state import EngineOptions
from search.models import DatasetResubmitInput, ResolveRequest
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.engine.test_dataset_resubmit import _identity_entry_from_spec
from tests.fixtures import FakeGraph, make_resolver

_QUERY = "mortality count in Kenya"
_KEN_NODE = {"country/KEN": {"label": "Kenya", "type": "Country"}}

_ELECTED_SV = "Count_MortalityEventZeroObsGuard"
_SIBLING_SV = "Count_MortalityEventEstimatedZeroObsGuard"
_SIBLING_SV_2 = "Count_MortalityEventProjectedZeroObsGuard"
_FAR_SV = "Count_UnrelatedTopicZeroObsGuard"


def _arcs(
    label: str,
    *,
    pop_type: str,
    meas_prop: str,
    denom: str | None = None,
    stat_type: str = "measuredValue",
    meas_qual: str | None = None,
) -> dict:
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": pop_type}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": stat_type}]},
    }
    if denom:
        arcs["measurementDenominator"] = {"nodes": [{"dcid": denom}]}
    if meas_qual:
        arcs["measurementQualifier"] = {"nodes": [{"dcid": meas_qual}]}
    return arcs


_NODES = {
    **_KEN_NODE,
    _ELECTED_SV: {
        "label": "Mortality event count",
        "type": "StatisticalVariable",
        "arcs": _arcs("Mortality event count", pop_type="MortalityEvent", meas_prop="count"),
    },
    # Same population, property, stat type, and denominator; different qualifier.
    _SIBLING_SV: {
        "label": "Estimated mortality event count",
        "type": "StatisticalVariable",
        "arcs": _arcs(
            "Estimated mortality event count",
            pop_type="MortalityEvent",
            meas_prop="count",
            meas_qual="Estimated",
        ),
    },
    _SIBLING_SV_2: {
        "label": "Projected mortality event count",
        "type": "StatisticalVariable",
        "arcs": _arcs(
            "Projected mortality event count",
            pop_type="MortalityEvent",
            meas_prop="count",
            meas_qual="Projected",
        ),
    },
    _FAR_SV: {
        "label": "Unrelated topic count",
        "type": "StatisticalVariable",
        "arcs": _arcs("Unrelated topic count", pop_type="UnrelatedThing", meas_prop="count"),
    },
}

_QUALIFIED_QUERY = "GDP growth in Ghana"
_GHA_NODE = {"country/GHA": {"label": "Ghana", "type": "Country"}}
# The bare growth-rate sibling differs only by measurementQualifier.
_QUALIFIED_ELECTED_SV = "RealValueGrowthRateZeroObsGuard"
_BARE_STAT_TYPE_SIBLING_SV = "BareGrowthRateZeroObsGuard"
# The rural label triggers alternate_guard's population-narrowing rule.
_RURAL_QUALIFIER_SIBLING_SV = "RuralBareGrowthRateZeroObsGuard"
_QUALIFIED_NODES = {
    **_GHA_NODE,
    _QUALIFIED_ELECTED_SV: {
        "label": "Growth rate amount of economic activity (real value)",
        "type": "StatisticalVariable",
        "arcs": _arcs(
            "Growth rate amount of economic activity (real value)",
            pop_type="EconomicActivity",
            meas_prop="amount",
            stat_type="growthRate",
            meas_qual="RealValue",
        ),
    },
    _BARE_STAT_TYPE_SIBLING_SV: {
        "label": "Gross domestic product growth rate",
        "type": "StatisticalVariable",
        "arcs": _arcs(
            "Gross domestic product growth rate",
            pop_type="EconomicActivity",
            meas_prop="amount",
            stat_type="growthRate",
        ),
    },
    _RURAL_QUALIFIER_SIBLING_SV: {
        "label": "Rural gross domestic product growth rate",
        "type": "StatisticalVariable",
        "arcs": _arcs(
            "Rural gross domestic product growth rate",
            pop_type="EconomicActivity",
            meas_prop="amount",
            stat_type="growthRate",
        ),
    },
}


class _NoAmbiguityBindLLM:
    """Elects the chosen shape and reports no also_plausible indices unless given some."""

    def __init__(self, chosen_shape_index: int, also_plausible: list[int] | None = None):
        self._chosen_shape_index = chosen_shape_index
        self._also_plausible = also_plausible

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=self._chosen_shape_index,
            bindings=[],
            also_plausible=self._also_plausible,
            ask=None,
        ), None


class _RecordingWalkReader(FakeMintGraphReader):
    """A member walk reader that keeps the filters of each reverse query."""

    def __init__(self, graph: FakeGraph) -> None:
        super().__init__(arcs=graph.member_arcs())
        self.walked: list[dict[str, str]] = []
        self._lock = threading.Lock()

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        with self._lock:
            self.walked.append(dict(filters or {}))
        return super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )


def _run(
    *,
    svs: list[str],
    scores: list[float],
    obs: dict,
    chosen_shape_index: int = 0,
    member_graph=None,
):
    detect = {_QUERY: {"svs": svs, "cosine_scores": scores}}
    graph = FakeGraph(nodes=_NODES, detect=detect, resolve={"Kenya": "country/KEN"}, obs=obs)
    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_NoAmbiguityBindLLM(chosen_shape_index),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(
                materialise_graph=member_graph(graph) if member_graph is not None else None
            ),
        )
    )


_KEN_OBS = [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}]


def _walk_reader(readers: list[_RecordingWalkReader]):
    def member_graph(graph: FakeGraph) -> _RecordingWalkReader:
        readers.append(_RecordingWalkReader(graph))
        return readers[0]

    return member_graph


def test_a_reelected_sibling_is_walked_before_it_answers():
    """The sibling is walked before it is materialised, so the answer it gives lists the
    members the mint serves and carries no `membership_unverified`."""
    readers: list[_RecordingWalkReader] = []

    result = _run(
        svs=[_ELECTED_SV, _SIBLING_SV],
        scores=[0.90, 0.85],
        obs={f"{_SIBLING_SV}|country/KEN": _KEN_OBS},
        member_graph=_walk_reader(readers),
    )

    assert result.status == "definite"
    assert [sv.ref for sv in result.specs[0].resolved.stat_vars.items] == [_SIBLING_SV]
    assert sorted(walk.get("measurementQualifier") or "" for walk in readers[0].walked) == [
        "",
        "Estimated",
    ]
    assert MEMBERSHIP_UNVERIFIED not in [w.code for w in result.warnings]


def test_siblings_offered_as_candidates_are_each_walked():
    """Two siblings hold data, so neither answers alone: each is walked before it is
    offered, so each lists the members its own resubmit serves."""
    readers: list[_RecordingWalkReader] = []

    result = _run(
        svs=[_ELECTED_SV, _SIBLING_SV, _SIBLING_SV_2],
        scores=[0.90, 0.85, 0.84],
        obs={
            f"{_SIBLING_SV}|country/KEN": _KEN_OBS,
            f"{_SIBLING_SV_2}|country/KEN": _KEN_OBS,
        },
        member_graph=_walk_reader(readers),
    )

    assert result.status == "candidates"
    assert len(readers[0].walked) == 3
    assert MEMBERSHIP_UNVERIFIED not in [w.code for w in result.warnings]


def test_single_sibling_within_margin_reelects_to_definite():
    result = _run(
        svs=[_ELECTED_SV, _SIBLING_SV],
        scores=[0.90, 0.85],  # 0.05 margin, above near margin and within sibling bound
        obs={f"{_SIBLING_SV}|country/KEN": _KEN_OBS},
    )
    assert result.status == "definite"
    sv_dcids = [sv.ref for sv in result.specs[0].resolved.stat_vars.items]
    assert sv_dcids == [_SIBLING_SV]


def test_single_sibling_reelection_warns_elected_shape_no_data():
    result = _run(
        svs=[_ELECTED_SV, _SIBLING_SV],
        scores=[0.90, 0.85],
        obs={f"{_SIBLING_SV}|country/KEN": _KEN_OBS},
    )
    (warning,) = (w for w in result.warnings if w.code == ELECTED_SHAPE_NO_DATA)
    assert warning.params == {
        "no_data": ["Mortality event count"],
        "elected": ["Estimated mortality event count"],
    }


def test_two_siblings_within_margin_become_candidates():
    result = _run(
        svs=[_ELECTED_SV, _SIBLING_SV, _SIBLING_SV_2],
        scores=[0.90, 0.85, 0.82],  # Margins 0.05 and 0.08 are within the sibling bound.
        obs={
            f"{_SIBLING_SV}|country/KEN": _KEN_OBS,
            f"{_SIBLING_SV_2}|country/KEN": _KEN_OBS,
        },
    )
    assert result.status == "candidates"
    assert len(result.specs) == 2
    sv_dcids = {sv.ref for spec in result.specs for sv in spec.resolved.stat_vars.items}
    assert sv_dcids == {_SIBLING_SV, _SIBLING_SV_2}


def test_two_siblings_candidates_warns_elected_shape_no_data():
    result = _run(
        svs=[_ELECTED_SV, _SIBLING_SV, _SIBLING_SV_2],
        scores=[0.90, 0.85, 0.82],
        obs={
            f"{_SIBLING_SV}|country/KEN": _KEN_OBS,
            f"{_SIBLING_SV_2}|country/KEN": _KEN_OBS,
        },
    )
    (warning,) = (w for w in result.warnings if w.code == ELECTED_SHAPE_NO_DATA)
    assert warning.params == {
        "no_data": ["Mortality event count"],
        "elected": ["Estimated mortality event count", "Projected mortality event count"],
    }


def test_no_data_bearing_sibling_stays_no_observations():
    result = _run(
        svs=[_ELECTED_SV, _SIBLING_SV],
        scores=[0.90, 0.85],
        obs={},
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"


def test_sibling_outside_score_margin_is_never_probed():
    # An unrelated-domain recall artefact with data must not steal the election.
    result = _run(
        svs=[_ELECTED_SV, _FAR_SV],
        scores=[0.90, 0.60],  # 0.30 margin exceeds the sibling bound
        obs={f"{_FAR_SV}|country/KEN": _KEN_OBS},
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"


def test_weak_chosen_score_exempts_the_mechanism():
    result = _run(
        svs=[_ELECTED_SV, _SIBLING_SV],
        scores=[0.70, 0.66],  # Chosen score is within the weak-score band
        obs={f"{_SIBLING_SV}|country/KEN": _KEN_OBS},
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"


_DIFFERENT_MEAS_PROP_SIBLING_SV = "MortalityRateZeroObsGuard"
_DIFFERENT_MEAS_PROP_GROWTH_SIBLING_SV = "MortalityRateGrowthZeroObsGuard"
_DIFFERENT_MEAS_PROP_NODES = {
    **_KEN_NODE,
    _ELECTED_SV: _NODES[_ELECTED_SV],
    _DIFFERENT_MEAS_PROP_SIBLING_SV: {
        "label": "Mortality rate",
        "type": "StatisticalVariable",
        "arcs": _arcs("Mortality rate", pop_type="Person", meas_prop="mortalityRate"),
    },
    _DIFFERENT_MEAS_PROP_GROWTH_SIBLING_SV: {
        "label": "Mortality rate growth",
        "type": "StatisticalVariable",
        "arcs": _arcs(
            "Mortality rate growth",
            pop_type="Person",
            meas_prop="mortalityRate",
            stat_type="growthRate",
        ),
    },
}


def test_different_measured_property_siblings_stay_no_observations():
    detect = {
        _QUERY: {
            "svs": [
                _ELECTED_SV,
                _DIFFERENT_MEAS_PROP_SIBLING_SV,
                _DIFFERENT_MEAS_PROP_GROWTH_SIBLING_SV,
            ],
            "cosine_scores": [0.90, 0.85, 0.83],  # Both are within the sibling bound
        }
    }
    graph = FakeGraph(
        nodes=_DIFFERENT_MEAS_PROP_NODES,
        detect=detect,
        resolve={"Kenya": "country/KEN"},
        obs={
            f"{_DIFFERENT_MEAS_PROP_SIBLING_SV}|country/KEN": _KEN_OBS,
            f"{_DIFFERENT_MEAS_PROP_GROWTH_SIBLING_SV}|country/KEN": _KEN_OBS,
        },
    )
    result = asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_NoAmbiguityBindLLM(0),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"


_SINGLE_SIBLING_DIFF_MEAS_PROP_SV = "GrowthRateMortalityDifferentPropZeroObsGuard"
_SINGLE_SIBLING_DIFF_MEAS_PROP_NODES = {
    **_KEN_NODE,
    _ELECTED_SV: _NODES[_ELECTED_SV],
    _SINGLE_SIBLING_DIFF_MEAS_PROP_SV: {
        "label": "Mortality growth rate",
        "type": "StatisticalVariable",
        "arcs": _arcs(
            "Mortality growth rate",
            pop_type="MortalityEvent",
            meas_prop="mortalityRate",
            stat_type="growthRate",
        ),
    },
}


def test_single_sibling_different_measured_property_stays_no_observations():
    detect = {
        _QUERY: {
            "svs": [_ELECTED_SV, _SINGLE_SIBLING_DIFF_MEAS_PROP_SV],
            "cosine_scores": [0.90, 0.85],  # margin 0.05: within the sibling bound
        }
    }
    graph = FakeGraph(
        nodes=_SINGLE_SIBLING_DIFF_MEAS_PROP_NODES,
        detect=detect,
        resolve={"Kenya": "country/KEN"},
        obs={f"{_SINGLE_SIBLING_DIFF_MEAS_PROP_SV}|country/KEN": _KEN_OBS},
    )
    result = asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_NoAmbiguityBindLLM(0),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"


def _run_qualified(*, svs: list[str], scores: list[float], obs: dict, chosen_shape_index: int = 0):
    detect = {_QUALIFIED_QUERY: {"svs": svs, "cosine_scores": scores}}
    graph = FakeGraph(
        nodes=_QUALIFIED_NODES, detect=detect, resolve={"Ghana": "country/GHA"}, obs=obs
    )
    return asyncio.run(
        resolve_variable(
            _QUALIFIED_QUERY,
            entities=["Ghana"],
            date_request=None,
            detect_query=_QUALIFIED_QUERY,
            role_query=_QUALIFIED_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_NoAmbiguityBindLLM(chosen_shape_index),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )


_GHA_OBS = [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}]


def test_stat_type_distinct_sibling_reelects_to_definite():
    result = _run_qualified(
        svs=[_QUALIFIED_ELECTED_SV, _BARE_STAT_TYPE_SIBLING_SV],
        scores=[0.90, 0.85],  # margin 0.05: > SEARCH_NEAR_MARGIN, <= the sibling bound
        # elected SV: no entry, zero obs; the bare-stat_type sibling has data.
        obs={f"{_BARE_STAT_TYPE_SIBLING_SV}|country/GHA": _GHA_OBS},
    )
    assert result.status == "definite"
    sv_dcids = [sv.ref for sv in result.specs[0].resolved.stat_vars.items]
    assert sv_dcids == [_BARE_STAT_TYPE_SIBLING_SV]
    assert any(w.code == ELECTED_SHAPE_NO_DATA for w in result.warnings)


def test_rural_qualifier_variant_rejected_by_guard_stays_no_data():
    result = _run_qualified(
        svs=[_QUALIFIED_ELECTED_SV, _RURAL_QUALIFIER_SIBLING_SV],
        scores=[0.90, 0.85],  # same margin as the positive control
        obs={f"{_RURAL_QUALIFIER_SIBLING_SV}|country/GHA": _GHA_OBS},
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"


_DIFFERENT_STAT_TYPE_SIBLING_SV = "GrowthRateMortalityEventZeroObsGuard"
_DIFFERENT_STAT_TYPE_NODES = {
    **_NODES,
    _DIFFERENT_STAT_TYPE_SIBLING_SV: {
        "label": "Mortality event growth rate",
        "type": "StatisticalVariable",
        "arcs": _arcs(
            "Mortality event growth rate",
            pop_type="MortalityEvent",
            meas_prop="count",
            stat_type="growthRate",
        ),
    },
}


def test_stat_type_differing_sibling_is_never_probed():
    detect = {
        _QUERY: {
            "svs": [_ELECTED_SV, _DIFFERENT_STAT_TYPE_SIBLING_SV],
            "cosine_scores": [0.90, 0.85],  # margin 0.05: within the sibling bound
        }
    }
    graph = FakeGraph(
        nodes=_DIFFERENT_STAT_TYPE_NODES,
        detect=detect,
        resolve={"Kenya": "country/KEN"},
        obs={f"{_DIFFERENT_STAT_TYPE_SIBLING_SV}|country/KEN": _KEN_OBS},
    )
    result = asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_NoAmbiguityBindLLM(0),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"


_EXPANSION_TOPIC_ROOT = "dc/topic/ZeroObsGuardExpansion"
_EXPANDED_SIBLING_SV = "Count_MortalityEventExpandedZeroObsGuard"
_EXPANSION_NODES = {
    **_NODES,
    _EXPANSION_TOPIC_ROOT: {
        "arcs": {
            "relevantVariable": {
                "nodes": [{"dcid": _EXPANDED_SIBLING_SV, "types": ["StatisticalVariable"]}]
            }
        }
    },
    _EXPANDED_SIBLING_SV: {
        "label": "Expanded estimated mortality event count",
        "type": "StatisticalVariable",
        "arcs": _arcs(
            "Expanded estimated mortality event count",
            pop_type="MortalityEvent",
            meas_prop="count",
            meas_qual="Estimated",
        ),
    },
}


def test_unscored_expansion_sibling_is_never_probed():
    detect = {
        _QUERY: {
            "svs": [_ELECTED_SV, _EXPANSION_TOPIC_ROOT],
            # Equal scores keep the topic root within SEARCH_TOPIC_EXPAND_MARGIN.
            "cosine_scores": [0.90, 0.90],
            "types": {_EXPANSION_TOPIC_ROOT: "Topic"},
        }
    }
    graph = FakeGraph(
        nodes=_EXPANSION_NODES,
        detect=detect,
        resolve={"Kenya": "country/KEN"},
        # elected SV: no entry, zero obs; the expanded sibling has data.
        obs={f"{_EXPANDED_SIBLING_SV}|country/KEN": _KEN_OBS},
    )
    result = asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_NoAmbiguityBindLLM(0),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"


_ELECTED_EXPANSION_TOPIC_ROOT = "dc/topic/ZeroObsGuardElectedExpansion"
_UNSCORED_ELECTED_SV = "Count_MortalityEventUnscoredElectedZeroObsGuard"
_UNSCORED_ELECTED_NODES = {
    **_NODES,
    _ELECTED_EXPANSION_TOPIC_ROOT: {
        "arcs": {
            "relevantVariable": {
                "nodes": [{"dcid": _UNSCORED_ELECTED_SV, "types": ["StatisticalVariable"]}]
            }
        }
    },
    _UNSCORED_ELECTED_SV: {
        "label": "Mortality event count (expansion-only)",
        "type": "StatisticalVariable",
        "arcs": _arcs(
            "Mortality event count (expansion-only)",
            pop_type="MortalityEvent",
            meas_prop="count",
        ),
    },
}


def test_unscored_elected_shape_with_scored_data_bearing_sibling_is_never_probed():
    """Index 1 elects an unscored expansion leaf; index 0 is a data-bearing qualifier variant.

    Their tied root scores allow expansion. The guard skips the margin calculation when the
    elected shape has no score, leaving its no_observations result intact."""
    detect = {
        _QUERY: {
            "svs": [_ELECTED_EXPANSION_TOPIC_ROOT, _SIBLING_SV],
            # Tied scores: the topic root sits within SEARCH_TOPIC_EXPAND_MARGIN
            # of the pool's own best score, so it qualifies as an expansion
            # root without widening any threshold.
            "cosine_scores": [0.90, 0.90],
            "types": {_ELECTED_EXPANSION_TOPIC_ROOT: "Topic"},
        }
    }
    graph = FakeGraph(
        nodes=_UNSCORED_ELECTED_NODES,
        detect=detect,
        resolve={"Kenya": "country/KEN"},
        # elected SV: no entry, zero obs; the scored sibling has data.
        obs={f"{_SIBLING_SV}|country/KEN": _KEN_OBS},
    )
    result = asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_NoAmbiguityBindLLM(1),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"


# A qualifier variant measures the elected quantity only when it holds the elected bindings.
_MEASURE_QUERY = "GDP in Kenya"
_REAL_GDP_SV = "InflationAdjustedGdpZeroObsGuard"
_NOMINAL_GDP_SV = "NominalGdpZeroObsGuard"
_PPP_GNI_SV = "PppGniZeroObsGuard"


def _activity_arcs(label: str, *, qualifier: str, activity: str) -> dict:
    return {
        **_arcs(label, pop_type="EconomicActivity", meas_prop="amount", meas_qual=qualifier),
        "constraintProperties": {"nodes": [{"dcid": "activitySource"}]},
        "activitySource": {"nodes": [{"dcid": activity}]},
    }


_MEASURE_NODES = {
    **_KEN_NODE,
    _REAL_GDP_SV: {
        "label": "Inflation-adjusted gross domestic product",
        "type": "StatisticalVariable",
        "arcs": _activity_arcs(
            "Inflation-adjusted gross domestic product",
            qualifier="InflationAdjusted",
            activity="GrossDomesticProduction",
        ),
    },
    _NOMINAL_GDP_SV: {
        "label": "Nominal gross domestic product",
        "type": "StatisticalVariable",
        "arcs": _activity_arcs(
            "Nominal gross domestic product",
            qualifier="Nominal",
            activity="GrossDomesticProduction",
        ),
    },
    _PPP_GNI_SV: {
        "label": "Gross national income at purchasing power parity",
        "type": "StatisticalVariable",
        "arcs": _activity_arcs(
            "Gross national income at purchasing power parity",
            qualifier="PurchasingPowerParity",
            activity="GrossNationalIncome",
        ),
    },
}


class _BindsActivityLLM:
    """Elects shape 0 and binds its activity source, as the model does for a named measure."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="what",
                    property_dcid="activitySource",
                    kind="value",
                    value_dcids=["GrossDomesticProduction"],
                    evidence="mentioned",
                )
            ],
            also_plausible=None,
            ask=None,
        ), None


def _run_measure(*, svs: list[str], scores: list[float], obs: dict):
    detect = {_MEASURE_QUERY: {"svs": svs, "cosine_scores": scores}}
    graph = FakeGraph(
        nodes=_MEASURE_NODES, detect=detect, resolve={"Kenya": "country/KEN"}, obs=obs
    )
    return asyncio.run(
        resolve_variable(
            _MEASURE_QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_MEASURE_QUERY,
            role_query=_MEASURE_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindsActivityLLM(),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )


def test_a_qualifier_variant_of_another_measure_is_not_elected_in_the_empty_shapes_place():
    """GNI at purchasing power parity differs from inflation-adjusted GDP in its qualifier
    and in the activity it measures, so its data does not answer a GDP question."""
    result = _run_measure(
        svs=[_REAL_GDP_SV, _PPP_GNI_SV],
        scores=[0.90, 0.85],
        obs={f"{_PPP_GNI_SV}|country/KEN": _KEN_OBS},
    )

    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"


def test_a_qualifier_variant_holding_the_bound_values_is_elected():
    """Nominal GDP covers the inflation-adjusted reading at Kenya, so the default reading
    removes the empty inflation-adjusted shape before bind and Nominal is elected directly,
    with no re-election warning."""
    result = _run_measure(
        svs=[_REAL_GDP_SV, _NOMINAL_GDP_SV, _PPP_GNI_SV],
        scores=[0.90, 0.86, 0.85],
        obs={
            f"{_NOMINAL_GDP_SV}|country/KEN": _KEN_OBS,
            f"{_PPP_GNI_SV}|country/KEN": _KEN_OBS,
        },
    )

    assert result.status == "definite"
    assert [sv.ref for sv in result.specs[0].resolved.stat_vars.items] == [_NOMINAL_GDP_SV]
    assert not any(w.code == ELECTED_SHAPE_NO_DATA for w in result.warnings)


_PLACELESS_QUERY = "mortality count"
_GHA_COUNTRY = "country/GHA"
_EARTH_CHILDREN = {"Earth": {"Country": ["country/KEN", _GHA_COUNTRY, "country/TZA"]}}
_SIBLING_OBS = {
    f"{_SIBLING_SV}|country/KEN": _KEN_OBS,
    f"{_SIBLING_SV}|{_GHA_COUNTRY}": _KEN_OBS,
}


_TWO_SIBLINGS = ([_ELECTED_SV, _SIBLING_SV, _SIBLING_SV_2], [0.90, 0.85, 0.82])
_SIBLING_AND_FAR = ([_ELECTED_SV, _SIBLING_SV, _FAR_SV], [0.90, 0.85, 0.84])


class _ReadRecordingGraph(FakeGraph):
    """Records the default-scope and facet-batch reads the engine makes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reads: list[tuple[str, list[str]]] = []

    def child_places_batch(self, dcids, child_type, *args, **kwargs):
        self.reads.append(("child_places_batch", [*dcids, child_type]))
        return super().child_places_batch(dcids, child_type, *args, **kwargs)

    def entities_with_data_by_variable(self, stat_vars, entities, *args, **kwargs):
        self.reads.append(("entities_with_data_by_variable", list(stat_vars)))
        return super().entities_with_data_by_variable(stat_vars, entities, *args, **kwargs)

    def observation_facets_batch(self, stat_vars, entities, *args, **kwargs):
        self.reads.append(("observation_facets_batch", list(stat_vars)))
        return super().observation_facets_batch(stat_vars, entities, *args, **kwargs)


def _placeless_graph(
    obs: dict, svs: list[str] | None = None, scores: list[float] | None = None
) -> _ReadRecordingGraph:
    detect = {
        _PLACELESS_QUERY: {
            "svs": svs or [_ELECTED_SV, _SIBLING_SV],
            "cosine_scores": scores or [0.90, 0.85],
        }
    }
    return _ReadRecordingGraph(
        nodes=_NODES, detect=detect, resolve={}, obs=obs, children=_EARTH_CHILDREN
    )


def _run_placeless(
    obs: dict,
    *,
    graph: FakeGraph | None = None,
    also_plausible: list[int] | None = None,
    sdmx: FakeSdmxClient | None = None,
    options: EngineOptions | None = None,
):
    graph = graph or _placeless_graph(obs)
    return asyncio.run(
        resolve_variable(
            _PLACELESS_QUERY,
            entities=[],
            date_request=None,
            detect_query=_PLACELESS_QUERY,
            role_query=_PLACELESS_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_NoAmbiguityBindLLM(0, also_plausible),
            base_steps=[],
            base_timing={},
            sdmx=sdmx
            or FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
            options=options,
        )
    )


def _resubmit_placeless(spec, obs: dict):
    graph = _placeless_graph(obs)
    replay = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(entries=[_identity_entry_from_spec(spec, _PLACELESS_QUERY)])
        ),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, scope_availability=graph.scope_availability),
    )
    return replay.parts[0]


def test_placeless_sibling_reelects_over_its_own_default_scope(monkeypatch):
    """With no place named, the sibling is probed over the default scope of its own StatVar.
    The elected StatVar has data nowhere, the sibling at two of Earth's three countries, so
    the answer covers and reads those two, warns once that the scope was defaulted, and
    resubmits to the same spec."""
    read_entities: list[dict[str, list[str]]] = []
    observation_entities = phase_answer._observation_entities

    def recording_observation_entities(shape, **kwargs):
        entities_read, warning = observation_entities(shape, **kwargs)
        read_entities.append(entities_read)
        return entities_read, warning

    monkeypatch.setattr(phase_answer, "_observation_entities", recording_observation_entities)
    result = _run_placeless(_SIBLING_OBS)

    assert result.status == "definite", result
    (spec,) = result.specs
    assert [sv.ref for sv in spec.resolved.stat_vars.items] == [_SIBLING_SV]
    codes = [w.code for w in result.warnings]
    assert codes.count(PLACE_SCOPE_DEFAULTED) == 1, codes
    assert [w.severity for w in result.warnings if w.code == PLACE_SCOPE_DEFAULTED] == ["warn"]
    assert ELECTED_SHAPE_NO_DATA in codes
    assert read_entities == [{"observationAbout": [_GHA_COUNTRY, "country/KEN"]}]
    coverage = spec.resolved.coverage
    assert coverage.kind == "exact"
    assert coverage.observation_count == 10
    assert {"label": "places", "count": 2} in [
        dim.model_dump() for dim in coverage.dimensions or []
    ]

    replay = _resubmit_placeless(spec, _SIBLING_OBS)
    assert replay.status == "definite", replay
    assert replay.spec.spec_id == spec.spec_id
    assert replay.spec.resolved.coverage == coverage


def test_placeless_sibling_without_data_stays_no_observations():
    """With no place named and no data anywhere, the sibling's default scope is empty, so the
    first answer is no_observations and names no re-election. A resubmit of the sibling's
    identity against the same graph answers no_data too."""
    result = _run_placeless({})

    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    codes = [w.code for w in result.warnings]
    assert codes.count(PLACE_SCOPE_DEFAULTED) == 1, codes
    assert ELECTED_SHAPE_NO_DATA not in codes

    (sibling_spec,) = _run_placeless(_SIBLING_OBS).specs
    replay = _resubmit_placeless(sibling_spec, {})
    assert replay.status == "no_data", replay


def test_adopted_placeless_sibling_reads_observations_over_its_own_default_scope():
    """The elected StatVar has data nowhere, so its default scope is empty, and the sibling's
    is two countries. The preview of the adopted sibling takes the one bulk read a defaulted
    country scope allows."""
    facet = [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5, "facetId": "F1"}]
    obs = {f"{_SIBLING_SV}|country/KEN": facet, f"{_SIBLING_SV}|{_GHA_COUNTRY}": facet}
    graph = _placeless_graph(obs)
    sdmx = FakeSdmxClient(
        observations={}, availability={}, scope_availability=graph.scope_availability
    )
    result = _run_placeless(
        obs, graph=graph, sdmx=sdmx, options=EngineOptions(include_observations=True)
    )

    assert result.status == "definite", result
    assert [sv.ref for sv in result.specs[0].resolved.stat_vars.items] == [_SIBLING_SV]
    assert [
        (call["variables"], call["places"], call["facet_ids"])
        for call in sdmx.subject_observations_calls
    ] == [([_SIBLING_SV], [_GHA_COUNTRY, "country/KEN"], ["F1"])]


def _observation_counts(result) -> dict[str, int]:
    return {
        spec.resolved.stat_vars.items[0].ref: spec.resolved.coverage.observation_count
        for spec in result.specs
    }


def _country_counts(result) -> dict[str, int | None]:
    return {
        spec.resolved.stat_vars.items[0].ref: next(
            (dim.count for dim in spec.resolved.coverage.dimensions or [] if dim.label == "places"),
            None,
        )
        for spec in result.specs
    }


def test_two_placeless_siblings_become_candidates_over_their_own_data():
    """With no place named, two siblings hold data in different countries. Each is probed in
    the default scope of its own StatVar and they answer as candidates: the first over its
    two countries, the second at its one country as a single source. A resubmit of either
    reports the coverage its candidate did."""
    obs = {**_SIBLING_OBS, f"{_SIBLING_SV_2}|country/TZA": _KEN_OBS}
    result = _run_placeless(obs, graph=_placeless_graph(obs, *_TWO_SIBLINGS))

    assert result.status == "candidates", result
    assert _observation_counts(result) == {_SIBLING_SV: 10, _SIBLING_SV_2: 5}
    assert _country_counts(result) == {_SIBLING_SV: 2, _SIBLING_SV_2: None}
    codes = [w.code for w in result.warnings]
    assert codes.count(PLACE_SCOPE_DEFAULTED) == 1, codes
    assert ELECTED_SHAPE_NO_DATA in codes
    for spec in result.specs:
        replay = _resubmit_placeless(spec, obs)
        assert replay.status == "definite", replay
        assert replay.spec.resolved.coverage == spec.resolved.coverage


def test_two_placeless_siblings_share_one_default_scope_read():
    """The placeless menu of two siblings costs one listing of Earth's countries and no
    presence read, since availability gives every StatVar's scope. The elected shape's
    empty scope needs no facet batch; the batches are a prewarm and one per sibling for the
    re-election, whose candidates the menu takes without a second read."""
    obs = {f"{_SIBLING_SV}|country/KEN": _KEN_OBS, f"{_SIBLING_SV_2}|country/KEN": _KEN_OBS}
    graph = _placeless_graph(obs, *_TWO_SIBLINGS)
    result = _run_placeless(obs, graph=graph)

    assert result.status == "candidates", result
    both = [_SIBLING_SV, _SIBLING_SV_2]
    assert graph.reads == [
        ("child_places_batch", ["Earth", "Country"]),
        ("observation_facets_batch", both),
        ("observation_facets_batch", [_SIBLING_SV]),
        ("observation_facets_batch", [_SIBLING_SV_2]),
    ]


@pytest.mark.parametrize(
    ("alternate_countries", "alternate_observations"),
    [
        pytest.param(["TZA"], 5, id="data-elsewhere"),
        pytest.param(["KEN", "TZA"], 10, id="data-overlapping"),
    ],
)
def test_flagged_alternate_beside_an_adopted_placeless_sibling_keeps_its_own_data(
    alternate_countries, alternate_observations
):
    """The bind step flags an unrelated reading beside an empty elected shape, and the
    sibling adopted in its place holds data in Kenya and Ghana. The sibling's default scope
    spans the alternate's StatVar too, so the alternate answers as a candidate over all its
    own countries and a resubmit reports the same coverage."""
    obs = {
        **_SIBLING_OBS,
        **{f"{_FAR_SV}|country/{country}": _KEN_OBS for country in alternate_countries},
    }
    result = _run_placeless(obs, graph=_placeless_graph(obs, *_SIBLING_AND_FAR), also_plausible=[2])

    assert result.status == "candidates", result
    assert _observation_counts(result) == {_SIBLING_SV: 10, _FAR_SV: alternate_observations}
    for spec in result.specs:
        replay = _resubmit_placeless(spec, obs)
        assert replay.status == "definite", replay
        assert (
            replay.spec.resolved.coverage.observation_count
            == spec.resolved.coverage.observation_count
        )
