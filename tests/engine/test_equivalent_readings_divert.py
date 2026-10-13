"""Tests when registered twin StatVars move from candidates to `equivalent_readings`.

Offline scenarios use hand-built graph and LLM doubles with registry dcids.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from search.engine.bind import _UniversalBindOutput
from search.engine.config import SearchSettings
from search.engine.conjoin import assemble_region
from search.engine.envelope import now_ms
from search.engine.quantity_read import QuantityOutput
from search.engine.region import NEAR_NEIGHBOR_AMBIGUOUS, resolve_variable
from search.engine.region.codes import ELECTED_SHAPE_NO_DATA
from search.engine.region.state import EngineOptions
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_KEN_NODE = {"country/KEN": {"label": "Kenya", "type": "Country"}}

# The registry's electricity-access cluster (region/equivalent_series.py):
# exactly two members, used for the plain divert/no-divert scenarios below.
_ELEC_WB = "worldBank/EG_ELC_ACCS_ZS"
_ELEC_SDG = "undata/sdg/EG_ACS_ELEC"

# The registry's under-five mortality cluster: its only three-member cluster.
_MORT_WB = "worldBank/SH_DYN_MORT"
_MORT_SDG = "undata/sdg/SH_DYN_MORT.AGE--Y0T4"
_MORT_DFO = "MortalityRate_Person_Upto4Years_AsFractionOf_Count_BirthEvent_LiveBirth"

# An unregistered StatVar serves as a competing alternate.
_UNRELATED_SV = "Count_Person_Employed"


def _arcs(*, pop_type: str, meas_prop: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": f"{pop_type}/{meas_prop}"}]},
        "populationType": {"nodes": [{"dcid": pop_type}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }


def _node(pop_type: str, meas_prop: str) -> dict:
    return {
        "label": f"{pop_type} {meas_prop}",
        "type": "StatisticalVariable",
        "arcs": _arcs(pop_type=pop_type, meas_prop=meas_prop),
    }


# Every SV used below gets its own five-tuple (a distinct populationType/
# measuredProperty pair) so each becomes a structurally distinct shape --
# real cross-vocabulary twins are exactly this: different source vocabularies
# assert different five-tuple arcs for the same real-world indicator.
_NODES = {
    **_KEN_NODE,
    _ELEC_WB: _node("Household", "electricityAccessWorldBank"),
    _ELEC_SDG: _node("Household", "electricityAccessSdg"),
    _MORT_WB: _node("Person", "mortalityWorldBank"),
    _MORT_SDG: _node("Person", "mortalitySdg"),
    _MORT_DFO: _node("Person", "mortalityDfo"),
    _UNRELATED_SV: _node("Person", "employedCount"),
}


class _CandidateBindLLM:
    """Elects `chosen_shape_index` and flags `also_plausible` -- the model's
    own ambiguity signal -- for the universal choose+bind call. No
    constraint slots on any candidate shape here."""

    def __init__(self, *, chosen_shape_index: int, also_plausible: list[int] | None):
        self._chosen_shape_index = chosen_shape_index
        self._also_plausible = also_plausible

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=self._chosen_shape_index,
            bindings=[],
            also_plausible=self._also_plausible,
            ask=None,
        ), None


def _facet(*, provenance_id: str | None) -> dict:
    return {
        "earliestDate": "2015",
        "latestDate": "2020",
        "obsCount": 5,
        "provenanceId": provenance_id,
    }


def _run(
    *,
    svs: list[str],
    scores: list[float],
    chosen_shape_index: int,
    also_plausible: list[int] | None,
    obs: dict,
    search_max_candidates: int | None = None,
    search_settings: Callable[..., SearchSettings] | None = None,
    include_option_coverage: bool = False,
):
    if search_max_candidates is not None:
        assert search_settings is not None
        search_settings(max_candidates=search_max_candidates)
    graph = FakeGraph(
        nodes=_NODES,
        detect={"electricity or mortality in Kenya": {"svs": svs, "cosine_scores": scores}},
        resolve={"Kenya": "country/KEN"},
        obs=obs,
    )
    return asyncio.run(
        resolve_variable(
            "electricity or mortality in Kenya",
            entities=["Kenya"],
            date_request=None,
            detect_query="electricity or mortality in Kenya",
            role_query="electricity or mortality in Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_CandidateBindLLM(
                chosen_shape_index=chosen_shape_index, also_plausible=also_plausible
            ),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(include_option_coverage=include_option_coverage),
        )
    )


def _sv_refs(spec) -> set[str]:
    return {sv.ref.ref for sv in spec.resolved.members}


def _warning_codes(result) -> set[str]:
    return {w.code for w in result.warnings}


# A genuine twin diverts: definite, with the twin named in equivalent_readings.


def test_genuine_twin_diverts_to_definite():
    result = _run(
        svs=[_ELEC_WB, _ELEC_SDG],
        scores=[0.90, 0.85],
        chosen_shape_index=0,
        also_plausible=[1],
        obs={
            f"{_ELEC_WB}|country/KEN": [_facet(provenance_id="source/WorldBank")],
            f"{_ELEC_SDG}|country/KEN": [_facet(provenance_id="source/SDG")],
        },
    )
    assert result.status == "definite"
    assert len(result.specs) == 1
    elected = result.specs[0]
    assert _sv_refs(elected) == {_ELEC_WB}
    twins = elected.resolved.equivalent_readings
    assert twins is not None and len(twins) == 1
    assert _sv_refs(twins[0]) == {_ELEC_SDG}


def test_same_provenance_split_does_not_divert():
    result = _run(
        svs=[_ELEC_WB, _ELEC_SDG],
        scores=[0.90, 0.85],
        chosen_shape_index=0,
        also_plausible=[1],
        obs={
            f"{_ELEC_WB}|country/KEN": [_facet(provenance_id="source/Shared")],
            f"{_ELEC_SDG}|country/KEN": [_facet(provenance_id="source/Shared")],
        },
    )
    assert result.status == "candidates"
    assert len(result.specs) == 2
    elected = next(s for s in result.specs if _sv_refs(s) == {_ELEC_WB})
    assert elected.resolved.equivalent_readings is None


def test_missing_provenance_does_not_divert():
    """The gate requires distinct provenance. Missing ids keep both candidates."""
    result = _run(
        svs=[_ELEC_WB, _ELEC_SDG],
        scores=[0.90, 0.85],
        chosen_shape_index=0,
        also_plausible=[1],
        obs={
            f"{_ELEC_WB}|country/KEN": [_facet(provenance_id=None)],
            f"{_ELEC_SDG}|country/KEN": [_facet(provenance_id=None)],
        },
    )
    assert result.status == "candidates"
    assert len(result.specs) == 2


# The under-five mortality triplet: two non-elected members both survive and
# both divert -- this is what equivalent_readings's list type exists for.


def test_mortality_triplet_diverts_both_twins():
    result = _run(
        svs=[_MORT_WB, _MORT_SDG, _MORT_DFO],
        scores=[0.90, 0.85, 0.80],
        chosen_shape_index=0,
        also_plausible=[1, 2],
        obs={
            f"{_MORT_WB}|country/KEN": [_facet(provenance_id="source/WorldBank")],
            f"{_MORT_SDG}|country/KEN": [_facet(provenance_id="source/SDG")],
            f"{_MORT_DFO}|country/KEN": [_facet(provenance_id="source/DFO")],
        },
    )
    assert result.status == "definite"
    assert len(result.specs) == 1
    elected = result.specs[0]
    assert _sv_refs(elected) == {_MORT_WB}
    twins = elected.resolved.equivalent_readings
    assert twins is not None
    assert {sv for t in twins for sv in _sv_refs(t)} == {_MORT_SDG, _MORT_DFO}


# Candidate scores track surviving specs. A diverted twin has no score entry.


def test_diverted_twin_pruned_from_candidate_scores():
    result = _run(
        svs=[_ELEC_WB, _ELEC_SDG, _UNRELATED_SV],
        scores=[0.90, 0.85, 0.80],
        chosen_shape_index=0,
        also_plausible=[1, 2],
        obs={
            f"{_ELEC_WB}|country/KEN": [_facet(provenance_id="source/WorldBank")],
            f"{_ELEC_SDG}|country/KEN": [_facet(provenance_id="source/SDG")],
            f"{_UNRELATED_SV}|country/KEN": [_facet(provenance_id="source/Other")],
        },
    )
    assert result.status == "candidates"
    assert len(result.specs) == 2
    elected = next(s for s in result.specs if _sv_refs(s) == {_ELEC_WB})
    twins = elected.resolved.equivalent_readings
    assert twins is not None and _sv_refs(twins[0]) == {_ELEC_SDG}
    assert result.candidate_scores is not None
    assert set(result.candidate_scores) == {s.spec_id for s in result.specs}


# A twin with data answers for an elected shape with no observations.


@pytest.mark.parametrize(
    ("svs", "obs", "status"),
    [
        pytest.param(
            [_ELEC_WB, _ELEC_SDG],
            {f"{_ELEC_SDG}|country/KEN": [_facet(provenance_id="source/SDG")]},
            "definite",
            id="twin-answers",
        ),
        pytest.param(
            [_ELEC_WB, _UNRELATED_SV],
            {f"{_UNRELATED_SV}|country/KEN": [_facet(provenance_id="source/Other")]},
            "no_data",
            id="unrelated-alternate-stays-alternate",
        ),
    ],
)
def test_elected_no_data_twin_answers(svs, obs, status):
    """The twin reads the elected indicator, so it answers definite and names the elected
    shape's empty read in an info warning. It lists nothing, since the elected shape holds
    nothing to read. An unrelated alternate stays the no-data answer's alternate match."""
    result = _run(
        svs=svs,
        scores=[0.90, 0.85],
        chosen_shape_index=0,
        also_plausible=[1],
        # No entry for the elected StatVar: confirmed structurally, zero observations.
        obs=obs,
    )
    assert result.status == status
    if status == "no_data":
        assert result.alternate_match is not None
        assert _sv_refs(result.alternate_match) == {_UNRELATED_SV}
        return
    (answer,) = result.specs
    assert _sv_refs(answer) == {_ELEC_SDG}
    assert not answer.resolved.equivalent_readings
    assert [(w.code, w.severity) for w in result.warnings if w.code == ELECTED_SHAPE_NO_DATA] == [
        (ELECTED_SHAPE_NO_DATA, "info")
    ]


def test_divert_under_tightest_candidate_budget(search_settings):
    result = _run(
        svs=[_ELEC_WB, _ELEC_SDG, _UNRELATED_SV],
        scores=[0.90, 0.85, 0.80],
        chosen_shape_index=0,
        also_plausible=[1, 2],
        obs={
            f"{_ELEC_WB}|country/KEN": [_facet(provenance_id="source/WorldBank")],
            f"{_ELEC_SDG}|country/KEN": [_facet(provenance_id="source/SDG")],
            f"{_UNRELATED_SV}|country/KEN": [_facet(provenance_id="source/Other")],
        },
        search_max_candidates=2,
        search_settings=search_settings,
    )
    assert result.status == "definite"
    assert len(result.specs) == 1
    elected = result.specs[0]
    assert _sv_refs(elected) == {_ELEC_WB}
    twins = elected.resolved.equivalent_readings
    assert twins is not None and len(twins) == 1
    assert _sv_refs(twins[0]) == {_ELEC_SDG}


# A diverted shape becomes an `equivalent_readings` annotation, so the candidate warning would be inaccurate.


def test_gate_fires_on_diverted_shape_suppresses_warning():
    result = _run(
        # The 0.01 margin is below the 0.025 threshold. The query names neither label,
        # so the gate fires without a model ambiguity signal.
        svs=[_ELEC_WB, _ELEC_SDG],
        scores=[0.90, 0.89],
        chosen_shape_index=0,
        also_plausible=None,
        obs={
            f"{_ELEC_WB}|country/KEN": [_facet(provenance_id="source/WorldBank")],
            f"{_ELEC_SDG}|country/KEN": [_facet(provenance_id="source/SDG")],
        },
    )
    assert result.status == "definite"
    assert len(result.specs) == 1
    elected = result.specs[0]
    assert _sv_refs(elected) == {_ELEC_WB}
    twins = elected.resolved.equivalent_readings
    assert twins is not None and len(twins) == 1
    assert _sv_refs(twins[0]) == {_ELEC_SDG}
    assert NEAR_NEIGHBOR_AMBIGUOUS not in _warning_codes(result)


def test_diverted_gate_twin_clears_confidence_and_ordering_state():
    """A gated twin leaves elected and unrelated candidate confidence unchanged."""
    result = _run(
        svs=[_ELEC_WB, _ELEC_SDG, _UNRELATED_SV],
        scores=[0.90, 0.89, 0.85],
        chosen_shape_index=0,
        also_plausible=[2],
        obs={
            f"{_ELEC_WB}|country/KEN": [_facet(provenance_id="source/WorldBank")],
            f"{_ELEC_SDG}|country/KEN": [_facet(provenance_id="source/SDG")],
            f"{_UNRELATED_SV}|country/KEN": [_facet(provenance_id="source/Other")],
        },
    )

    assert result.status == "candidates"
    assert [_sv_refs(spec) for spec in result.specs] == [{_ELEC_WB}, {_UNRELATED_SV}]
    elected, unrelated = result.specs
    twins = elected.resolved.equivalent_readings
    assert twins is not None and _sv_refs(twins[0]) == {_ELEC_SDG}
    assert elected.resolved.confidence == "high"
    assert twins[0].resolved.confidence == "high"
    assert unrelated.resolved.confidence == "high"
    assert NEAR_NEIGHBOR_AMBIGUOUS not in _warning_codes(result)
    assert result.candidate_near_neighbor_spec_ids == frozenset()
    assert result.near_neighbor_fired is False


# The 0.89 unrelated candidate triggers the near-neighbor warning.
# The lower-scoring twin diverts through the model's `also_plausible` signal.


def test_gate_fires_on_surviving_candidate_leaves_warning_when_other_shape_diverts():
    result = _run(
        # The unrelated StatVar scores 0.89 against 0.90 and triggers the gate.
        # The model's `also_plausible` signal includes the lower-scoring twin.
        svs=[_ELEC_WB, _UNRELATED_SV, _ELEC_SDG],
        scores=[0.90, 0.89, 0.70],
        chosen_shape_index=0,
        also_plausible=[2],
        obs={
            f"{_ELEC_WB}|country/KEN": [_facet(provenance_id="source/WorldBank")],
            f"{_UNRELATED_SV}|country/KEN": [_facet(provenance_id="source/Other")],
            f"{_ELEC_SDG}|country/KEN": [_facet(provenance_id="source/SDG")],
        },
    )
    assert result.status == "candidates"
    assert len(result.specs) == 2
    elected = next(s for s in result.specs if _sv_refs(s) == {_ELEC_WB})
    twins = elected.resolved.equivalent_readings
    assert twins is not None and _sv_refs(twins[0]) == {_ELEC_SDG}
    assert any(_sv_refs(s) == {_UNRELATED_SV} for s in result.specs)
    assert NEAR_NEIGHBOR_AMBIGUOUS in _warning_codes(result)


def test_divert_without_near_neighbor_warning_is_unaffected():
    result = _run(
        svs=[_ELEC_WB, _ELEC_SDG],
        scores=[0.90, 0.85],
        chosen_shape_index=0,
        also_plausible=[1],
        obs={
            f"{_ELEC_WB}|country/KEN": [_facet(provenance_id="source/WorldBank")],
            f"{_ELEC_SDG}|country/KEN": [_facet(provenance_id="source/SDG")],
        },
    )
    assert result.status == "definite"
    elected = result.specs[0]
    assert elected.resolved.equivalent_readings is not None
    assert NEAR_NEIGHBOR_AMBIGUOUS not in _warning_codes(result)


# A twin carries no rendered sentence and no facet_options, even when
# include_option_coverage is on for the request: with_sentence (search.engine.assemble) only ever renders
# interpretations[0]/extras/candidates/nearest_real/flagged_match, and the
# facet_options probe (_attach_candidate_facet_options, region/pipeline.py)
# only ever runs on post-divert survivors -- a twin is neither.


def test_diverted_twin_carries_no_sentence_or_facet_options_with_both_flags_on():
    region = _run(
        svs=[_ELEC_WB, _ELEC_SDG],
        scores=[0.90, 0.85],
        chosen_shape_index=0,
        also_plausible=[1],
        obs={
            f"{_ELEC_WB}|country/KEN": [_facet(provenance_id="source/WorldBank")],
            f"{_ELEC_SDG}|country/KEN": [_facet(provenance_id="source/SDG")],
        },
        include_option_coverage=True,
    )
    assert region.status == "definite"

    response = assemble_region(
        region,
        query="electricity or mortality in Kenya",
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=5,
    )

    elected = response.parts[0].spec
    # Prove both flags actually took effect on the elected spec itself -- a
    # test that passed because they'd silently defaulted off would prove
    # nothing about the twin below.
    assert elected.resolved.rendered_sentence is not None
    assert elected.resolved.sentence_plan is not None
    assert elected.resolved.facet_options != []

    twins = elected.resolved.equivalent_readings
    assert twins is not None and len(twins) == 1
    twin = twins[0]
    assert twin.resolved.rendered_sentence is None
    assert twin.resolved.sentence_plan is None
    assert twin.resolved.facet_options == []
