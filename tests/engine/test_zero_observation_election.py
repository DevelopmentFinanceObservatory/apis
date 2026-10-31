"""Status rule 1: a qualifier variant with data answers when the elected shape has none.

Tests cover the semantic guards, unscored expansion shapes, and the readings that do not
count as a variant.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading

import pytest

from search.engine.assemble import build_no_data_part
from search.engine.bind import SlotBindingDraft
from search.engine.binders import Binder
from search.engine.quantity_read import QuantityOutput
from search.engine.region import (
    candidate_collapse,
    phase_answer,
    phase_place_materialise,
    resolve_variable,
)
from search.engine.region.candidates import _AlternatesRead
from search.engine.region.codes import (
    ELECTED_SHAPE_NO_DATA,
    MEMBERSHIP_UNVERIFIED,
    PLACE_SCOPE_DEFAULTED,
)
from search.engine.region.state import EngineOptions
from search.engine.stated_words_read import StatedWordsOutput
from search.models import DatasetResubmitInput, Operation, ResolveRequest
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine._binders import ReplyBinder, elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.engine._quantity import NO_QUANTITY, NO_RATING
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
        "label": "Gross domestic product growth rate (real value)",
        "type": "StatisticalVariable",
        "arcs": _arcs(
            "Gross domestic product growth rate (real value)",
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
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is StatedWordsOutput:
            return NO_RATING
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(self._chosen_shape_index, also_plausible=self._also_plausible), None


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
    binder: Binder | None = None,
    svs: list[str],
    scores: list[float],
    obs: dict,
    chosen_shape_index: int = 0,
    member_graph=None,
    nodes: dict = _NODES,
    also_plausible: list[int] | None = None,
):
    detect = {_QUERY: {"svs": svs, "cosine_scores": scores}}
    graph = FakeGraph(nodes=nodes, detect=detect, resolve={"Kenya": "country/KEN"}, obs=obs)
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
            binder=binder,
            llm=_NoAmbiguityBindLLM(chosen_shape_index, also_plausible),
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
    assert [sv.ref.ref for sv in result.specs[0].resolved.members] == [_SIBLING_SV]
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
    sv_dcids = [sv.ref.ref for sv in result.specs[0].resolved.members]
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


_FEMALE_SIBLING_SV = "Count_MortalityEventEstimatedFemaleZeroObsGuard"
_FEMALE_SIBLING_NAME = "Estimated female mortality event count"


@pytest.mark.parametrize(
    ("far_obs", "also_plausible", "status", "binder"),
    [
        pytest.param({}, None, "definite", None, id="stand-in-alone"),
        pytest.param(
            {f"{_FAR_SV}|country/KEN": _KEN_OBS},
            [2],
            "candidates",
            ReplyBinder(),
            id="beside-an-alternate",
        ),
    ],
)
def test_single_sibling_reelection_names_the_stand_in_specs_shape_label(
    far_obs, also_plausible, status, binder
):
    """The sibling shape holds two series. Recall ranks the female one first, so the shape
    is drafted under its name, and only the other holds data in Kenya. The warning's
    `elected` is the shape label of the Spec built for the stand-in, the served series'
    name, whether the stand-in answers alone or is offered beside an also-plausible
    alternate with data."""
    female_arcs = _arcs(
        _FEMALE_SIBLING_NAME, pop_type="MortalityEvent", meas_prop="count", meas_qual="Estimated"
    )
    female_arcs["gender"] = {"nodes": [{"dcid": "Female"}]}
    nodes = {
        **_NODES,
        _FEMALE_SIBLING_SV: {
            "label": _FEMALE_SIBLING_NAME,
            "type": "StatisticalVariable",
            "arcs": female_arcs,
        },
    }

    result = _run(
        svs=[_ELECTED_SV, _FEMALE_SIBLING_SV, _SIBLING_SV, _FAR_SV],
        scores=[0.90, 0.86, 0.85, 0.84],
        obs={f"{_SIBLING_SV}|country/KEN": _KEN_OBS, **far_obs},
        nodes=nodes,
        also_plausible=also_plausible,
        binder=binder,
    )

    assert result.status == status
    (stand_in,) = (
        spec
        for spec in result.specs
        if [sv.ref.ref for sv in spec.resolved.members] == [_SIBLING_SV]
    )
    (warning,) = (w for w in result.warnings if w.code == ELECTED_SHAPE_NO_DATA)
    assert warning.params == {
        "no_data": ["Mortality event count"],
        "elected": [stand_in.resolved.shape.label],
    }
    assert stand_in.resolved.shape.label == "Estimated mortality event count"


def test_a_sibling_beside_an_unevaluated_sibling_keeps_the_elected_no_data_answer(
    monkeypatch,
) -> None:
    """The estimated count holds data at Kenya, as in
    `test_single_sibling_within_margin_reelects_to_definite`, but another sibling's presence
    read after unbinding times out, so it is unevaluated rather than disproven. The
    estimated count is not installed in the elected shape's place: the elected no-data
    answer stands, as it does beside an over-cap sibling."""
    materialise = phase_place_materialise._materialise_alt_shapes

    def beside_a_timed_out_sibling(alternates, **kwargs) -> _AlternatesRead:
        read = materialise(alternates, **kwargs)
        timed_out = dataclasses.replace(alternates[0], bindings=[])
        return read._replace(unevaluated=(*read.unevaluated, (timed_out, "timed_out")))

    monkeypatch.setattr(
        phase_place_materialise, "_materialise_alt_shapes", beside_a_timed_out_sibling
    )

    result = _run(
        svs=[_ELECTED_SV, _SIBLING_SV],
        scores=[0.90, 0.85],
        obs={f"{_SIBLING_SV}|country/KEN": _KEN_OBS},
    )

    assert result.status == "no_data", result
    assert ELECTED_SHAPE_NO_DATA not in [w.code for w in result.warnings]


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
    sv_dcids = {sv.ref.ref for spec in result.specs for sv in spec.resolved.members}
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


def test_an_unrelated_reading_with_data_does_not_answer_for_an_empty_election():
    # Status rule 1: only a qualifier variant answers; an unrelated reading nobody
    # nominated takes no part, whatever its score.
    result = _run(
        svs=[_ELECTED_SV, _FAR_SV],
        scores=[0.90, 0.60],  # 0.30 margin exceeds the sibling bound
        obs={f"{_FAR_SV}|country/KEN": _KEN_OBS},
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"


def test_a_weak_chosen_score_sets_no_status():
    """Status rule 5: a weak retrieval score warns and leaves rule 1 to answer from the
    qualifier sibling with data."""
    result = _run(
        svs=[_ELECTED_SV, _SIBLING_SV],
        scores=[0.70, 0.66],  # Chosen score is within the weak-score band
        obs={f"{_SIBLING_SV}|country/KEN": _KEN_OBS},
    )
    assert result.status == "definite"
    assert [sv.ref.ref for sv in result.specs[0].resolved.members] == [_SIBLING_SV]


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
    sv_dcids = [sv.ref.ref for sv in result.specs[0].resolved.members]
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


def test_an_unscored_expansion_sibling_answers_for_an_empty_election():
    """Status rule 1 reads no score: a qualifier sibling reached through topic expansion
    answers for the empty elected shape like any other."""
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
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "definite"
    assert ELECTED_SHAPE_NO_DATA in [w.code for w in result.warnings]


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


def test_an_unscored_empty_election_is_answered_by_its_scored_sibling():
    """Index 1 elects an unscored expansion leaf; index 0 is a data-bearing qualifier variant.
    Status rule 1 reads no score, so the sibling answers for the empty election."""
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
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "definite"
    assert ELECTED_SHAPE_NO_DATA in [w.code for w in result.warnings]


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
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is StatedWordsOutput:
            return NO_RATING
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(
            0,
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
    assert [sv.ref.ref for sv in result.specs[0].resolved.members] == [_NOMINAL_GDP_SV]
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
    assert [sv.ref.ref for sv in spec.resolved.members] == [_SIBLING_SV]
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
    assert [sv.ref.ref for sv in result.specs[0].resolved.members] == [_SIBLING_SV]
    assert [
        (call["variables"], call["places"], call["facet_ids"])
        for call in sdmx.subject_observations_calls
    ] == [([_SIBLING_SV], [_GHA_COUNTRY, "country/KEN"], ["F1"])]


def _observation_counts(result) -> dict[str, int]:
    return {
        spec.resolved.members[0].ref.ref: spec.resolved.coverage.observation_count
        for spec in result.specs
    }


def _country_counts(result) -> dict[str, int | None]:
    return {
        spec.resolved.members[0].ref.ref: next(
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
    re-election, whose candidates the menu takes without a second read. Each candidate's
    facet options then read its own StatVar's facets, in either order."""
    obs = {f"{_SIBLING_SV}|country/KEN": _KEN_OBS, f"{_SIBLING_SV_2}|country/KEN": _KEN_OBS}
    graph = _placeless_graph(obs, *_TWO_SIBLINGS)
    result = _run_placeless(obs, graph=graph)

    assert result.status == "candidates", result
    both = [_SIBLING_SV, _SIBLING_SV_2]
    assert graph.reads[:4] == [
        ("child_places_batch", ["Earth", "Country"]),
        ("observation_facets_batch", both),
        ("observation_facets_batch", [_SIBLING_SV]),
        ("observation_facets_batch", [_SIBLING_SV_2]),
    ]
    assert sorted(graph.reads[4:]) == [
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


_LITERACY_QUERY = "female literacy rate"
_LITERACY_SVS = (
    "Count_Person_Female_LiterateZeroObsGuard",
    "Count_Person_Male_LiterateZeroObsGuard",
)
_FEMALE_LITERACY_SV, _MALE_LITERACY_SV = _LITERACY_SVS
_ESTIMATED_FEMALE_LITERACY_SV = "Count_Person_Female_Literate_EstimatedZeroObsGuard"
_PROJECTED_FEMALE_LITERACY_SV = "Count_Person_Female_Literate_ProjectedZeroObsGuard"
_LITERACY_NO_GENDER_SV = "Count_Person_Literate_AdultZeroObsGuard"
_LITERACY_BY_SEX_SV = "Count_Person_Female_Literate_BySexZeroObsGuard"


def _literacy_arcs(
    label: str,
    gender: str | None,
    meas_qual: str | None,
    denom: str = "Count_Person",
    meas_prop: str = "count",
) -> dict:
    arcs = {
        **_arcs(label, pop_type="Person", meas_prop=meas_prop, denom=denom, meas_qual=meas_qual),
        "constraintProperties": {"nodes": [{"dcid": "literacyStatus"}]},
        "literacyStatus": {"nodes": [{"dcid": "Literate"}]},
    }
    if gender is not None:
        arcs["constraintProperties"] = {"nodes": [{"dcid": "gender"}, {"dcid": "literacyStatus"}]}
        arcs["gender"] = {"nodes": [{"dcid": gender}]}
    return arcs


def _literacy_node(
    label: str,
    gender: str | None,
    meas_qual: str | None = None,
    denom: str = "Count_Person",
    meas_prop: str = "count",
) -> dict:
    return {
        "label": label,
        "type": "StatisticalVariable",
        "arcs": _literacy_arcs(label, gender, meas_qual, denom, meas_prop),
    }


_LITERACY_NODES = {
    "country/BGD": {"label": "Bangladesh", "type": "Country"},
    "Female": {"label": "Female", "type": "GenderType"},
    "Male": {"label": "Male", "type": "GenderType"},
    "Literate": {"label": "Literate", "type": "LiteracyStatusEnum"},
    _FEMALE_LITERACY_SV: _literacy_node("Female literacy rate", "Female"),
    _MALE_LITERACY_SV: _literacy_node("Male literacy rate", "Male"),
    _ESTIMATED_FEMALE_LITERACY_SV: _literacy_node(
        "Estimated female literacy rate", "Female", "Estimated"
    ),
    _PROJECTED_FEMALE_LITERACY_SV: _literacy_node(
        "Projected female literacy rate", "Female", "Projected"
    ),
    # Another measured property over another denominator, so it is a reading of another
    # measure with no gender slot.
    _LITERACY_NO_GENDER_SV: _literacy_node(
        "Adult literacy rate", None, denom="Count_Person_15OrMoreYears", meas_prop="literacy"
    ),
    # A female literacy rate whose label does not name the sex.
    _LITERACY_BY_SEX_SV: _literacy_node("Literacy rate by sex", "Female"),
}


def _unmet_value_outcome(result) -> tuple:
    """The part's status and no-data reason, the unmet value terms each answer's trace
    records (a refusal's flagged match for a miss), by its first member, and what departs
    `measure_not_applied`."""
    answers = result.flagged_match or () if result.status == "no_data" else result.specs
    return (
        result.status,
        result.no_data_reason,
        {
            spec.resolved.members[0].ref.ref: [
                q.text
                for q in spec.resolved.trace.query_qualifiers
                if q.disposition == "no_vocabulary"
            ]
            for spec in answers
        },
        sorted(
            fact.asked
            for fact in result.departure_facts.values()
            if fact.code == "measure_not_applied"
        ),
    )


@pytest.mark.parametrize(
    ("svs", "scores", "with_data", "also_plausible", "outcome"),
    [
        # The bind leaves the elected literacy rate's gender slot open, so that answer would
        # hold every gender, and it has no data. Its estimated sibling stands in (status rule
        # 1), and its one member holds Female, so "female" is met and the part is definite.
        pytest.param(
            [*_LITERACY_SVS, _ESTIMATED_FEMALE_LITERACY_SV],
            [0.90, 0.90, 0.85],
            [_ESTIMATED_FEMALE_LITERACY_SV],
            None,
            ("definite", None, {_ESTIMATED_FEMALE_LITERACY_SV: []}, []),
            id="reelected-sibling",
        ),
        # Two such siblings hold data, so they answer as a candidate set. Each applies
        # Female, so the set departs nothing for "female", though the elected shape leaves
        # it unmet.
        pytest.param(
            [*_LITERACY_SVS, _ESTIMATED_FEMALE_LITERACY_SV, _PROJECTED_FEMALE_LITERACY_SV],
            [0.90, 0.90, 0.85, 0.84],
            [_ESTIMATED_FEMALE_LITERACY_SV, _PROJECTED_FEMALE_LITERACY_SV],
            None,
            (
                "candidates",
                None,
                {_ESTIMATED_FEMALE_LITERACY_SV: [], _PROJECTED_FEMALE_LITERACY_SV: []},
                [],
            ),
            id="sibling-set-applies-it",
        ),
        # The elected literacy rate applies Female, and a nominated alternate of another
        # measure with no gender slot answers beside it (status rule 2(b)). The alternate leaves "female" unmet, so its trace records it and
        # the set departs it.
        pytest.param(
            [_LITERACY_BY_SEX_SV, _LITERACY_NO_GENDER_SV],
            [0.90, 0.85],
            [_LITERACY_BY_SEX_SV, _LITERACY_NO_GENDER_SV],
            [1],
            (
                "candidates",
                None,
                {_LITERACY_BY_SEX_SV: [], _LITERACY_NO_GENDER_SV: ["female"]},
                ["female"],
            ),
            id="alternate-in-a-set-leaves-it-unmet",
        ),
    ],
)
def test_each_answer_is_judged_on_the_requested_value(
    svs, scores, with_data, also_plausible, outcome
):
    """ "female literacy rate in Bangladesh": the value "female" is judged on each answer's
    own shape and bindings, a stand-in sibling, each member of a candidate set and a
    surviving alternate, and each answer's trace records it when that answer leaves it
    unmet."""
    graph = FakeGraph(
        nodes=_LITERACY_NODES,
        detect={_LITERACY_QUERY: {"svs": svs, "cosine_scores": scores}},
        resolve={"Bangladesh": "country/BGD"},
        obs={f"{sv}|country/BGD": _KEN_OBS for sv in with_data},
    )
    result = asyncio.run(
        resolve_variable(
            _LITERACY_QUERY,
            entities=["Bangladesh"],
            date_request=None,
            detect_query=_LITERACY_QUERY,
            role_query=_LITERACY_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_NoAmbiguityBindLLM(0, also_plausible),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(),
        )
    )

    assert _unmet_value_outcome(result) == outcome


_LITERACY_UNSEXED_SV = "Count_Person_Literate_UnsexedZeroObsGuard"
_ESTIMATED_MALE_LITERACY_SV = "Count_Person_Male_Literate_EstimatedZeroObsGuard"


class _StrayFemaleBindLLM:
    """Elects the first shape and drafts `gender` Female on it."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is StatedWordsOutput:
            return NO_RATING
        draft = SlotBindingDraft(
            axis="how", property_dcid="gender", kind="value", value_dcids=["Female"]
        )
        return elected_reply(0, bindings=[draft], also_plausible=None), None


@pytest.mark.parametrize(
    ("sibling", "outcome"),
    [
        (_ESTIMATED_FEMALE_LITERACY_SV, ("definite", None, _ESTIMATED_FEMALE_LITERACY_SV, [])),
        (
            _ESTIMATED_MALE_LITERACY_SV,
            ("no_data", "qualifier_not_bound", _LITERACY_UNSEXED_SV, ["Female"]),
        ),
    ],
    ids=["sibling-holds-the-stray-value", "sibling-holds-another-value"],
)
def test_a_stray_value_is_met_by_a_stand_in_holding_it(sibling, outcome):
    """ "female literacy rate in Bangladesh" elects a literacy rate with no gender slot and
    no data, and the bind drafts Female on it, a stray value (`_stray_requirements`). The
    alternate guard reads it as a value the request names (`requirements.stray_values`), so
    an estimated sibling holding Female stands in (status rule 1), meets it, and nothing
    refuses or departs for it. A sibling holding Male is pinned out, and the part refuses
    `qualifier_not_bound` on the elected reading, naming exactly its trace's terms."""
    svs = [_LITERACY_UNSEXED_SV, sibling]
    nodes = {
        **_LITERACY_NODES,
        _LITERACY_UNSEXED_SV: _literacy_node("Literacy rate", None),
        _ESTIMATED_MALE_LITERACY_SV: _literacy_node(
            "Estimated male literacy rate", "Male", "Estimated"
        ),
    }
    graph = FakeGraph(
        nodes=nodes,
        detect={_LITERACY_QUERY: {"svs": svs, "cosine_scores": [0.90, 0.85]}},
        resolve={"Bangladesh": "country/BGD"},
        obs={f"{sibling}|country/BGD": _KEN_OBS},
    )
    result = asyncio.run(
        resolve_variable(
            _LITERACY_QUERY,
            entities=["Bangladesh"],
            date_request=None,
            detect_query=_LITERACY_QUERY,
            role_query=_LITERACY_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_StrayFemaleBindLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(),
        )
    )

    (answer,) = result.flagged_match or () if result.status == "no_data" else result.specs
    refused = [q.text for q in result.unbound_qualifiers or ()]
    assert (
        result.status,
        result.no_data_reason,
        answer.resolved.members[0].ref.ref,
        refused,
    ) == outcome
    assert [q.text for q in answer.resolved.trace.query_qualifiers] == refused
    assert result.departure_facts == {}


_ADULT_FEMALE_LITERACY_SV = "Literacy_Person_15OrMore_Female_RiderZeroObsGuard"


class _StrayFemaleNominatingLLM:
    """Elects the first shape, drafts `gender` Female on it and nominates the second."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is StatedWordsOutput:
            return NO_RATING
        draft = SlotBindingDraft(
            axis="how", property_dcid="gender", kind="value", value_dcids=["Female"]
        )
        return elected_reply(0, bindings=[draft], also_plausible=[1]), None


def test_a_rider_holding_the_stray_value_keeps_the_refusal_it_rides_on():
    """ "female literacy rate in Bangladesh" elects a literacy rate with no gender slot and
    no data, with Female drafted on it as a stray value. The nominated adult female
    literacy rate, another measure, holds Female and has data, so it rides on the
    `no_observations` refusal (`only_alternate_has_data`), judged on its own shape: it
    meets Female, so its trace is empty and nothing departs. The refusal judged its match,
    so the pipeline's hook for refusals that judged none (`settle_unjudged_refusal`)
    leaves it alone, though the elected answer leaves Female unmet."""
    svs = [_LITERACY_UNSEXED_SV, _ADULT_FEMALE_LITERACY_SV]
    nodes = {
        **_LITERACY_NODES,
        _LITERACY_UNSEXED_SV: _literacy_node("Literacy rate", None),
        _ADULT_FEMALE_LITERACY_SV: _literacy_node(
            "Adult female literacy rate",
            "Female",
            denom="Count_Person_15OrMoreYears",
            meas_prop="literacy",
        ),
    }
    graph = FakeGraph(
        nodes=nodes,
        detect={_LITERACY_QUERY: {"svs": svs, "cosine_scores": [0.90, 0.85]}},
        resolve={"Bangladesh": "country/BGD"},
        obs={f"{_ADULT_FEMALE_LITERACY_SV}|country/BGD": _KEN_OBS},
    )
    result = asyncio.run(
        resolve_variable(
            _LITERACY_QUERY,
            entities=["Bangladesh"],
            date_request=None,
            detect_query=_LITERACY_QUERY,
            role_query=_LITERACY_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_StrayFemaleNominatingLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(),
        )
    )

    assert (result.status, result.no_data_reason, result.gate) == (
        "no_data",
        "no_observations",
        "only_alternate_has_data",
    )
    assert result.alternate_match is not None
    assert [m.ref.ref for m in result.alternate_match.resolved.members] == [
        _ADULT_FEMALE_LITERACY_SV
    ]
    assert list(result.alternate_match.resolved.trace.query_qualifiers) == []
    assert result.unbound_qualifiers == ()
    assert result.departure_facts == {}


_ADULT_MALE_LITERACY_SV = "Literacy_Person_15OrMore_Male_RiderZeroObsGuard"
_YOUTH_FEMALE_LITERACY_SV = "LiteracyYouth_Person_15To24_Female_RiderZeroObsGuard"
_YOUTH_MALE_LITERACY_SV = "LiteracyYouth_Person_15To24_Male_RiderZeroObsGuard"
_ESTIMATED_LITERACY_UNSEXED_SV = "Count_Person_Literate_EstimatedUnsexedZeroObsGuard"


class _StrayGenderNominatingTwoLLM:
    """Elects the first shape, drafts `values` on `gender` on it and nominates the next
    two."""

    def __init__(self, values: list[str]) -> None:
        self.values = values

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is StatedWordsOutput:
            return NO_RATING
        draft = SlotBindingDraft(
            axis="how",
            property_dcid="gender",
            kind="value" if len(self.values) == 1 else "set",
            value_dcids=self.values,
        )
        return elected_reply(0, bindings=[draft], also_plausible=[1, 2]), None


def _resolve_stray_gender(
    elected: str,
    others: list[str],
    drafted: list[str],
    query: str = _LITERACY_QUERY,
    operations: tuple[Operation, ...] = (),
):
    """Resolve `query` in Bangladesh over `elected` and `others`, all with
    data, with `drafted` drafted on `gender` on the elected shape and the others
    nominated."""
    nodes = {
        **_LITERACY_NODES,
        _LITERACY_UNSEXED_SV: _literacy_node("Literacy rate", None),
        _ESTIMATED_LITERACY_UNSEXED_SV: _literacy_node(
            "Estimated literacy rate", None, "Estimated"
        ),
        _ADULT_FEMALE_LITERACY_SV: _literacy_node(
            "Adult female literacy rate",
            "Female",
            denom="Count_Person_15OrMoreYears",
            meas_prop="literacy",
        ),
        _ADULT_MALE_LITERACY_SV: _literacy_node(
            "Adult male literacy rate",
            "Male",
            denom="Count_Person_15OrMoreYears",
            meas_prop="literacy",
        ),
        _YOUTH_FEMALE_LITERACY_SV: _literacy_node(
            "Youth female literacy rate",
            "Female",
            denom="Count_Person_15To24Years",
            meas_prop="youthLiteracy",
        ),
        _YOUTH_MALE_LITERACY_SV: _literacy_node(
            "Youth male literacy rate",
            "Male",
            denom="Count_Person_15To24Years",
            meas_prop="youthLiteracy",
        ),
    }
    svs = [elected, *others]
    graph = FakeGraph(
        nodes=nodes,
        detect={query: {"svs": svs, "cosine_scores": [0.90, 0.85, 0.84]}},
        resolve={"Bangladesh": "country/BGD"},
        obs={f"{sv}|country/BGD": _KEN_OBS for sv in svs},
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=["Bangladesh"],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_StrayGenderNominatingTwoLLM(drafted),
            operations=operations,
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(),
        )
    )


@pytest.mark.parametrize(
    ("elected", "others", "drafted", "outcome"),
    [
        (
            _LITERACY_UNSEXED_SV,
            [_ADULT_FEMALE_LITERACY_SV, _YOUTH_FEMALE_LITERACY_SV],
            ["Female"],
            (
                "candidates",
                None,
                sorted([_ADULT_FEMALE_LITERACY_SV, _YOUTH_FEMALE_LITERACY_SV]),
                [],
            ),
        ),
        (
            _LITERACY_UNSEXED_SV,
            [_ADULT_FEMALE_LITERACY_SV, _ADULT_MALE_LITERACY_SV],
            ["Female"],
            (
                "no_data",
                "qualifier_not_bound",
                [_LITERACY_UNSEXED_SV],
                ["Female"],
            ),
        ),
        (
            _ESTIMATED_LITERACY_UNSEXED_SV,
            [_PROJECTED_FEMALE_LITERACY_SV, _ADULT_MALE_LITERACY_SV],
            ["Female"],
            ("definite", None, [_PROJECTED_FEMALE_LITERACY_SV], []),
        ),
        (
            _LITERACY_UNSEXED_SV,
            [_ADULT_MALE_LITERACY_SV, _LITERACY_NO_GENDER_SV],
            ["Female"],
            ("no_data", "qualifier_not_bound", [_LITERACY_UNSEXED_SV], ["Female"]),
        ),
        (
            _LITERACY_UNSEXED_SV,
            [_ADULT_FEMALE_LITERACY_SV, _YOUTH_MALE_LITERACY_SV],
            ["Female", "Male"],
            (
                "candidates",
                None,
                sorted([_ADULT_FEMALE_LITERACY_SV, _YOUTH_MALE_LITERACY_SV]),
                [],
            ),
        ),
        (
            _LITERACY_UNSEXED_SV,
            [_LITERACY_NO_GENDER_SV, _ESTIMATED_LITERACY_UNSEXED_SV],
            ["Female", "Male"],
            (
                "no_data",
                "qualifier_not_bound",
                [_LITERACY_UNSEXED_SV],
                ["Female and Male"],
            ),
        ),
    ],
    ids=[
        "two-nominees-apply-it",
        "one-nominee-applies-it",
        "one-price-basis-sibling-applies-it",
        "no-reading-applies-it",
        "each-nominee-holds-a-value-of-a-set",
        "no-reading-holds-a-value-of-a-set",
    ],
)
def test_a_stray_value_binds_every_candidate_the_elected_reading_included(
    elected, others, drafted, outcome
):
    """ "female literacy rate in Bangladesh" elects a literacy rate with no gender slot that
    has data, and the bind drafts `drafted` on it, stray values (`_stray_requirements`),
    and nominates the two other readings, which have data. Each answer is held to every
    drafted value: the alternate guard pins out a reading that holds another gender or none
    (`AlternateJudge._stray_refusal`), and the elected reading, which cannot apply it,
    leaves the answer when another does (`candidate_collapse._meeting_stray_values`).

    Two nominees holding Female are the candidates. One nominee holding Female, a reading
    of another measure, does not answer by elimination: the part refuses
    `qualifier_not_bound` naming Female with the elected reading as flagged match and
    nothing else, a response the wire model builds (`NoData`). A sibling of the elected
    measure on another price basis, the axis an elected reading off the default leaves
    open, answers alone. With no reading applying the value the part refuses on the elected
    reading. A drafted set of Female and Male is read by membership: the female and the
    male nominees are the candidates, without the unsexed elected reading, and with no
    reading holding a value of the set the part refuses naming the set."""
    query = _LITERACY_QUERY if drafted == ["Female"] else "male and female literacy rate"
    result = _resolve_stray_gender(elected, others, drafted, query)

    answers = result.flagged_match or () if result.status == "no_data" else result.specs
    assert (
        result.status,
        result.no_data_reason,
        sorted(member.ref.ref for spec in answers for member in spec.resolved.members),
        sorted(q.text for q in result.unbound_qualifiers or ()),
    ) == outcome
    if result.status == "no_data":
        part = build_no_data_part(
            result.no_data_reason,
            variable_text=result.variable_text,
            warnings=list(result.warnings),
            flagged_match=answers[0],
            gate=result.gate,
            alternate_match=result.alternate_match,
        )
        assert (part.no_data.gate, part.no_data.alternate_match) == (None, None)
        assert part.no_data.flagged_match.spec_id == answers[0].spec_id
    # The drafted set departs nothing. Each candidate of the set leaves out the other sex,
    # which the candidates part discloses once per value.
    departed = ["female", "male"] if result.status == "candidates" and len(drafted) > 1 else []
    assert sorted(fact.asked.casefold() for fact in result.departure_facts.values()) == departed
    assert sum(w.code == "query_qualifier_dropped" for w in result.warnings) == len(departed)


def test_the_measure_fold_reads_only_the_answers_that_meet_a_stray_value(monkeypatch):
    """The stray filter runs before the measure-sibling fold
    (`candidate_collapse.assemble_candidates`), so a sibling that applies the value is
    never folded into an elected reading that leaves for it: the fold is handed the two
    nominees holding Female and not the elected reading."""
    folded: list[list[str]] = []
    fold = candidate_collapse._fold_measure_sibling_candidates

    def _recording(specs, shapes, partitions):
        folded.append(sorted(m.ref.ref for spec in specs for m in spec.resolved.members))
        return fold(specs, shapes, partitions)

    monkeypatch.setattr(candidate_collapse, "_fold_measure_sibling_candidates", _recording)

    _resolve_stray_gender(
        _LITERACY_UNSEXED_SV, [_ADULT_FEMALE_LITERACY_SV, _YOUTH_FEMALE_LITERACY_SV], ["Female"]
    )

    assert folded == [sorted([_ADULT_FEMALE_LITERACY_SV, _YOUTH_FEMALE_LITERACY_SV])]


def test_a_stray_value_holds_the_candidates_of_a_part_with_an_operation():
    """An operation on the part changes nothing of the stray rule: "female literacy rate in
    Bangladesh" asked as a growth over years still offers the two nominees holding Female
    and leaves the unsexed elected reading out."""
    result = _resolve_stray_gender(
        _LITERACY_UNSEXED_SV,
        [_ADULT_FEMALE_LITERACY_SV, _YOUTH_FEMALE_LITERACY_SV],
        ["Female"],
        operations=(Operation(verb="growth", phrase="growth in", over="year"),),
    )

    assert result.status == "candidates"
    assert sorted(m.ref.ref for spec in result.specs for m in spec.resolved.members) == sorted(
        [_ADULT_FEMALE_LITERACY_SV, _YOUTH_FEMALE_LITERACY_SV]
    )


def test_a_lone_reading_of_a_drafted_set_discloses_the_value_it_leaves_out():
    """ "male and female literacy rate in Bangladesh" drafts Female and Male on an elected
    estimated literacy rate with no gender slot. The projected female literacy rate, a
    sibling of the elected measure on the open price-basis axis, meets the set by
    membership and answers alone; its members hold no Male, so the answer departs "Male"
    (`measure_not_applied`, `query_qualifier_dropped`) and refuses nothing."""
    result = _resolve_stray_gender(
        _ESTIMATED_LITERACY_UNSEXED_SV,
        [_PROJECTED_FEMALE_LITERACY_SV, _LITERACY_NO_GENDER_SV],
        ["Female", "Male"],
        "male and female literacy rate",
    )

    assert result.status == "definite"
    assert [m.ref.ref for spec in result.specs for m in spec.resolved.members] == [
        _PROJECTED_FEMALE_LITERACY_SV
    ]
    assert [(fact.code, fact.asked) for fact in result.departure_facts.values()] == [
        ("measure_not_applied", "Male")
    ]
    assert sum(w.code == "query_qualifier_dropped" for w in result.warnings) == 1


@pytest.mark.parametrize(
    ("basis", "answers"),
    [
        ("structural", True),
        ("twin", True),
        ("nominated", False),
        ("direction", False),
        (None, False),
    ],
)
def test_a_lone_survivor_answers_for_the_elected_reading_only_on_its_measure_and_direction(
    basis, answers
):
    """When the elected reading has data and leaves for a stray value, a lone survivor
    answers only when it reads the elected measure in the asked direction: a structural
    sibling or a registry twin. A nominated reading of another measure, the other
    direction, and a reading offered on no basis leave the part refusing on the elected
    reading (`candidate_collapse.answers_for_the_elected_reading`)."""
    assert candidate_collapse.answers_for_the_elected_reading(basis) is answers


_ADULT_FEMALE_BY_SEX = "Literacy_Person_15OrMore_Female_BySexZeroObsGuard"
_ADULT_MALE_BY_SEX = "Literacy_Person_15OrMore_Male_BySexZeroObsGuard"
_ADULT_TOTAL_BY_SEX = "Literacy_Person_15OrMore_BySexZeroObsGuard_Total"
_YOUTH_FEMALE_BY_SEX = "LiteracyYouth_Person_15To24_Female_BySexZeroObsGuard"
_YOUTH_MALE_BY_SEX = "LiteracyYouth_Person_15To24_Male_BySexZeroObsGuard"
_YOUTH_TOTAL_BY_SEX = "LiteracyYouth_Person_15To24_BySexZeroObsGuard_Total"
_BY_SEX_NODES = {
    sv: _literacy_node(label, sex, denom=denom, meas_prop=meas_prop)
    for sv, label, sex, denom, meas_prop in (
        (
            _ADULT_FEMALE_BY_SEX,
            "Adult female literacy rate",
            "Female",
            "Count_Person_15OrMoreYears",
            "literacy",
        ),
        (
            _ADULT_MALE_BY_SEX,
            "Adult male literacy rate",
            "Male",
            "Count_Person_15OrMoreYears",
            "literacy",
        ),
        (
            _ADULT_TOTAL_BY_SEX,
            "Adult literacy rate total",
            None,
            "Count_Person_15OrMoreYears",
            "literacy",
        ),
        (
            _YOUTH_FEMALE_BY_SEX,
            "Youth female literacy rate",
            "Female",
            "Count_Person_15To24Years",
            "youthLiteracy",
        ),
        (
            _YOUTH_MALE_BY_SEX,
            "Youth male literacy rate",
            "Male",
            "Count_Person_15To24Years",
            "youthLiteracy",
        ),
        (
            _YOUTH_TOTAL_BY_SEX,
            "Youth literacy rate total",
            None,
            "Count_Person_15To24Years",
            "youthLiteracy",
        ),
    )
}


class _NoDraftNominatingTwoLLM:
    """Elects the first shape, drafts nothing and nominates the next two."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is StatedWordsOutput:
            return NO_RATING
        return elected_reply(0, also_plausible=[1, 2]), None


def _resolve_by_sex(
    others: list[str],
    drafted: list[str],
    *,
    query: str = "male and female literacy rate",
    with_data: list[str] | None = None,
):
    """Resolve `query` in Bangladesh over the unsexed literacy rate, elected, and the
    by-sex literacy members `others`, with `drafted` drafted on `gender` and the next two
    shapes nominated (nothing drafted when `drafted` is empty). Every StatVar has data
    unless `with_data` names those that do."""
    svs = [_LITERACY_UNSEXED_SV, *others]
    graph = FakeGraph(
        nodes={
            **_LITERACY_NODES,
            _LITERACY_UNSEXED_SV: _literacy_node("Literacy rate", None),
            **{sv: _BY_SEX_NODES[sv] for sv in others},
        },
        detect={query: {"svs": svs, "cosine_scores": [0.90 - 0.01 * i for i in range(len(svs))]}},
        resolve={"Bangladesh": "country/BGD"},
        obs={f"{sv}|country/BGD": _KEN_OBS for sv in (svs if with_data is None else with_data)},
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=["Bangladesh"],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_StrayGenderNominatingTwoLLM(drafted) if drafted else _NoDraftNominatingTwoLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(),
        )
    )


def _members_by_answer(result) -> list[list[str]]:
    answers = result.flagged_match or () if result.status == "no_data" else result.specs
    return sorted(sorted(m.ref.ref for m in spec.resolved.members) for spec in answers)


_ADULT_BY_SEX = [_ADULT_FEMALE_BY_SEX, _ADULT_MALE_BY_SEX]
_YOUTH_BY_SEX = [_YOUTH_FEMALE_BY_SEX, _YOUTH_MALE_BY_SEX]


@pytest.mark.parametrize(
    ("others", "members", "departed"),
    [
        (
            [*_ADULT_BY_SEX, *_YOUTH_BY_SEX],
            [sorted(_ADULT_BY_SEX), sorted(_YOUTH_BY_SEX)],
            [],
        ),
        (
            [*_ADULT_BY_SEX, _ADULT_TOTAL_BY_SEX, *_YOUTH_BY_SEX, _YOUTH_TOTAL_BY_SEX],
            [sorted(_ADULT_BY_SEX), sorted(_YOUTH_BY_SEX)],
            [],
        ),
        (
            [*_ADULT_BY_SEX, _YOUTH_FEMALE_BY_SEX],
            [sorted(_ADULT_BY_SEX), [_YOUTH_FEMALE_BY_SEX]],
            ["Male"],
        ),
    ],
    ids=["families-holding-each-value", "families-with-a-total", "one-family-holds-one-value"],
)
def test_a_drafted_set_reaches_readings_whose_families_hold_each_value(others, members, departed):
    """ "male and female literacy rate in Bangladesh" drafts Female and Male on an unsexed
    elected reading and nominates adult and youth literacy. Each nominee binds its gender
    slot to the drafted values its members hold (`candidates._with_stray_values`), so it is
    represented and read with exactly those members: a family of a Female and a Male member
    serves both sexes, a family's total, which holds no sex, is pinned out and never read,
    and a family holding only Female serves it and departs "Male" once."""
    result = _resolve_by_sex(others, ["Female", "Male"])

    assert result.status == "candidates"
    assert _members_by_answer(result) == sorted(members)
    assert [fact.asked for fact in result.departure_facts.values()] == departed


def test_a_drafted_set_mints_one_identity_whatever_order_it_was_drafted_in():
    """Female then Male and Male then Female bind the same set on each nominee, so the
    candidates' identities are the same."""
    spec_ids = [
        sorted(
            spec.spec_id
            for spec in _resolve_by_sex([*_ADULT_BY_SEX, *_YOUTH_BY_SEX], drafted).specs
        )
        for drafted in (["Female", "Male"], ["Male", "Female"])
    ]

    assert spec_ids[0] == spec_ids[1]


@pytest.mark.parametrize(
    ("others", "outcome"),
    [
        (
            [*_ADULT_BY_SEX, *_YOUTH_BY_SEX],
            ("no_data", "no_observations", [], sorted(_ADULT_BY_SEX)),
        ),
        (
            [_ADULT_TOTAL_BY_SEX, _YOUTH_TOTAL_BY_SEX],
            ("no_data", "qualifier_not_bound", ["Female and Male"], None),
        ),
    ],
    ids=["a-nominee-serves-the-set", "no-nominee-meets-the-set"],
)
def test_a_drafted_set_beside_an_elected_reading_with_no_data(others, outcome):
    """With no data on the unsexed elected reading, a nominee serving both sexes rides on
    the `no_observations` refusal as its alternate match; with nominees that hold no sex,
    the part refuses `qualifier_not_bound` naming the drafted set once."""
    result = _resolve_by_sex(others, ["Female", "Male"], with_data=others)

    alternate = result.alternate_match
    assert (
        result.status,
        result.no_data_reason,
        [q.text for q in result.unbound_qualifiers or ()],
        alternate and sorted(m.ref.ref for m in alternate.resolved.members),
    ) == outcome


def test_a_part_with_no_stray_value_keeps_its_answer_and_identities():
    """Nothing drafted on a missing slot changes nothing: "literacy rate in Bangladesh" over
    the unsexed reading, nominating the two by-sex totals, offers the three readings with
    the identities the engine minted before stray values were judged. The elected reading
    binds the literacy status every member of its sample asserts as the slot's sole value."""
    result = _resolve_by_sex(
        [*_ADULT_BY_SEX, _ADULT_TOTAL_BY_SEX, *_YOUTH_BY_SEX, _YOUTH_TOTAL_BY_SEX],
        [],
        query="literacy rate",
    )

    assert result.status == "candidates"
    assert [spec.spec_id for spec in result.specs] == [
        "spec_d0b1081b996808e0d3be7af90df63194",
        "spec_9e7c933a8ebea23245636e6c69f73dc5",
        "spec_ac60a40b1496bf76499ce6503248c19e",
    ]
