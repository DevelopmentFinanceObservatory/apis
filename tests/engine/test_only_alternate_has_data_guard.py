"""Tests the no-data guard for alternate shapes when the elected shape has no observations.

A lone alternate yields `no_data` with `ONLY_ALTERNATE_HAS_DATA`; two alternates yield
`candidates` with an `ELECTED_SHAPE_NO_DATA` warning."""

from __future__ import annotations

import asyncio

from search.engine.bind import _UniversalBindOutput
from search.engine.region import resolve_variable
from search.engine.region.codes import ELECTED_SHAPE_NO_DATA, ONLY_ALTERNATE_HAS_DATA
from search.engine.region.state import _elected_stat_vars
from search.engine.retrieval.retrieve import NoDataDraft
from search.engine.shape import ShapeDraft
from search.models import GraphRef
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.fixtures import FakeGraph, make_resolver

_ELECTED_SV = "Count_Person_HighAltitude"
_STD_SV = "HealthcareExpenditureAsFractionOfGDP"
_STD_SV_2 = "Count_MedicalDoctor_PerCapita"


def _std_arcs(label: str, *, pop_type: str, meas_prop: str, denom: str | None = None) -> dict:
    # A plain standard SV with no constraint properties at all.
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": pop_type}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if denom:
        arcs["measurementDenominator"] = {"nodes": [{"dcid": denom}]}
    return arcs


_KEN_NODE = {"country/KEN": {"label": "Kenya", "type": "Country"}}
_ELECTED_NODE = {
    _ELECTED_SV: {
        "label": "Count of people at high altitude",
        "type": "StatisticalVariable",
        "arcs": _std_arcs(
            "Count of people at high altitude", pop_type="Person", meas_prop="countHighAltitude"
        ),
    },
}


class _GuardBindLLM:
    """Elects the data-empty shape and marks the standard alternatives as plausible."""

    def __init__(self, also_plausible: list[int], *, chosen_shape_index: int):
        self._also_plausible = also_plausible
        self._chosen_shape_index = chosen_shape_index

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=self._chosen_shape_index,
            # No constraint slots on any candidate shape here -- every
            # binding-carrying variant of this scenario lives in
            # test_option_coverage.py instead.
            bindings=[],
            also_plausible=self._also_plausible,
            ask=None,
        ), None


def _run(
    *,
    nodes: dict,
    detect: dict,
    obs: dict,
    also_plausible: list[int],
    chosen_shape_index: int,
):
    graph = FakeGraph(nodes=nodes, detect=detect, resolve={"Kenya": "country/KEN"}, obs=obs)
    return asyncio.run(
        resolve_variable(
            "high altitude population or health spending in Kenya",
            entities=["Kenya"],
            date_request=None,
            detect_query="high altitude population or health spending in Kenya",
            role_query="high altitude population or health spending in Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_GuardBindLLM(also_plausible, chosen_shape_index=chosen_shape_index),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


# ---------------------------------------------------------------------------
# One alternate misses CandidateSet's minimum size and yields no_data.
# ---------------------------------------------------------------------------

_ONE_ALT_NODES: dict = {
    **_KEN_NODE,
    **_ELECTED_NODE,
    _STD_SV: {
        "label": "Healthcare expenditure as fraction of GDP",
        "type": "StatisticalVariable",
        "arcs": _std_arcs(
            "Healthcare expenditure as fraction of GDP",
            pop_type="EconomicActivity",
            meas_prop="amount",
            denom="GrossDomesticProduction",
        ),
    },
}
# Standard SV ranks first (higher retrieval score) -- the model must elect
# the data-empty shape despite this.
_ONE_ALT_DETECT = {
    "high altitude population or health spending in Kenya": {
        "svs": [_STD_SV, _ELECTED_SV],
        "cosine_scores": [0.85, 0.82],
    },
}
_ONE_ALT_OBS = {
    # The standard alt shape has real data at Kenya.
    f"{_STD_SV}|country/KEN": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
    # Deliberately NO entry for the elected SV -- confirmed structurally
    # (graph.node_arcs) but zero real observations, the "honest limitation"
    # graph_confirm_resolve's own no_observations relaxation path carries.
}


def _run_one_alt_scenario():
    # 2 shapes ranked by score: STD_SV(0.85) index 0, ELECTED_SV(0.82) index 1.
    return _run(
        nodes=_ONE_ALT_NODES,
        detect=_ONE_ALT_DETECT,
        obs=_ONE_ALT_OBS,
        also_plausible=[0],
        chosen_shape_index=1,
    )


def test_one_alt_survivor_is_no_data_not_definite():
    result = _run_one_alt_scenario()
    assert result.status == "no_data", (
        f"expected no_data (a lone non-elected alt can't fill CandidateSet's "
        f"min_length=2, and must never silently become definite), got {result.status!r}"
    )


def test_one_alt_survivor_carries_no_diagnostics_warning():
    """The no-data result carries the gate without a diagnostics warning."""
    result = _run_one_alt_scenario()
    assert not any(w.code == ONLY_ALTERNATE_HAS_DATA for w in result.warnings), (
        f"expected no ONLY_ALTERNATE_HAS_DATA warning, got {[w.code for w in result.warnings]}"
    )


def test_one_alt_survivor_carried_as_alternate_match():
    """The plausible alternative is exposed without a Spec for the data-empty elected shape."""
    result = _run_one_alt_scenario()
    assert result.specs == ()
    assert result.flagged_match is None
    assert result.alternate_match is not None
    sv_dcids = [sv.ref.dcid for sv in result.alternate_match.resolved.members]
    assert sv_dcids == [_STD_SV]


def test_one_alt_survivor_carries_only_alternate_has_data_gate():
    """Only the alternate has data, so the gate is `ONLY_ALTERNATE_HAS_DATA`."""
    result = _run_one_alt_scenario()
    assert result.gate == ONLY_ALTERNATE_HAS_DATA


def test_one_alt_survivor_elected_stat_vars_names_the_elected_shape():
    """`elected_stat_vars` names the structurally confirmed, data-empty elected shape."""
    result = _run_one_alt_scenario()
    assert result.elected_stat_vars == [
        GraphRef(dcid=_ELECTED_SV, label="Count of people at high altitude")
    ]


def test_elected_stat_vars_none_when_mat_result_is_no_data_draft():
    """A role-incompatible `NoDataDraft` has no graph-confirmed identity to publish."""
    shape = ShapeDraft(
        shape_id="shape_placeholder",
        label="shape placeholder",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_ELECTED_SV: _ELECTED_NODE[_ELECTED_SV]["arcs"]},
    )
    assert _elected_stat_vars(NoDataDraft(reason="role_incompatible"), shape) is None


# ---------------------------------------------------------------------------
# Role-bearing elected shapes also publish their identity on the bare no-data path.
# ---------------------------------------------------------------------------

_ROLE_SV = "TESTFIXTURE/OnlyAlternateHasDataGuard/RoleBearingFlow"
# The subject-only alternate shares population and stat type with _ROLE_SV.
# Its denominator and entity roles distinguish it after five-tuple grouping. The query
# names a share, which the alternate honours, so the guard admits it beside the level.
_AID_SV = "AidReceivedAsFractionOfGDP"
_ROLE_NODE = {
    _ROLE_SV: {
        "label": "aid flow",
        "type": "StatisticalVariable",
        "arcs": {
            **_std_arcs("aid flow", pop_type="EconomicActivity", meas_prop="amount"),
            "observationProperties": {"nodes": [{"dcid": "creditor"}, {"dcid": "debtor"}]},
        },
    },
    _AID_SV: {
        "label": "Aid received as a fraction of GDP",
        "type": "StatisticalVariable",
        "arcs": _std_arcs(
            "Aid received as a fraction of GDP",
            pop_type="EconomicActivity",
            meas_prop="amount",
            denom="GrossDomesticProduction",
        ),
    },
}
_ROLE_GUARD_QUERY = "aid to Kenya as a share of GDP"
_ROLE_GUARD_DETECT = {
    _ROLE_GUARD_QUERY: {
        "svs": [_AID_SV, _ROLE_SV],
        "cosine_scores": [0.85, 0.82],
    },
}
_ROLE_GUARD_OBS = {
    f"{_AID_SV}|country/KEN": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
}


def _run_role_bearing_one_alt_scenario():
    # AID_SV ranks first at 0.85 and ROLE_SV second at 0.82.
    # ROLE_SV's creditor/debtor roles bind Kenya on the destination axis.
    graph = FakeGraph(
        nodes={
            **_KEN_NODE,
            **_ROLE_NODE,
        },
        detect=_ROLE_GUARD_DETECT,
        resolve={"Kenya": "country/KEN"},
        obs=_ROLE_GUARD_OBS,
    )
    return asyncio.run(
        resolve_variable(
            _ROLE_GUARD_QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_ROLE_GUARD_QUERY,
            role_query=_ROLE_GUARD_QUERY,
            entity_roles=door_roles_for_test_query(_ROLE_GUARD_QUERY, ["Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_GuardBindLLM([0], chosen_shape_index=1),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def test_role_bearing_elected_shape_no_observations_elects_stat_vars():
    """The role-bearing no-data path names its elected StatVar after bound-constraint confirmation."""
    result = _run_role_bearing_one_alt_scenario()
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert result.flagged_match is None
    assert result.alternate_match is not None
    assert result.gate == ONLY_ALTERNATE_HAS_DATA
    assert result.elected_stat_vars == [GraphRef(dcid=_ROLE_SV, label="aid flow")]


# ---------------------------------------------------------------------------
# Two alternates meet CandidateSet's minimum and yield candidates with a warning.
# ---------------------------------------------------------------------------

_TWO_ALT_NODES: dict = {
    **_KEN_NODE,
    **_ELECTED_NODE,
    _STD_SV: {
        "label": "Healthcare expenditure as fraction of GDP",
        "type": "StatisticalVariable",
        "arcs": _std_arcs(
            "Healthcare expenditure as fraction of GDP",
            pop_type="EconomicActivity",
            meas_prop="amount",
            denom="GrossDomesticProduction",
        ),
    },
    # Distinct five-tuples need different populationType/measuredProperty values.
    # Shared five-tuples deduplicate to one shape.
    _STD_SV_2: {
        "label": "Physicians per 1000 people",
        "type": "StatisticalVariable",
        "arcs": _std_arcs(
            "Physicians per 1000 people",
            pop_type="MedicalDoctor",
            meas_prop="count",
        ),
    },
}
_TWO_ALT_DETECT = {
    "high altitude population or health spending in Kenya": {
        "svs": [_STD_SV, _STD_SV_2, _ELECTED_SV],
        "cosine_scores": [0.86, 0.85, 0.82],
    },
}
_TWO_ALT_OBS = {
    f"{_STD_SV}|country/KEN": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
    f"{_STD_SV_2}|country/KEN": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
}


def _run_two_alt_scenario():
    # 3 shapes ranked by score: STD_SV(0.86) index 0, STD_SV_2(0.85) index 1,
    # ELECTED_SV(0.82) index 2.
    return _run(
        nodes=_TWO_ALT_NODES,
        detect=_TWO_ALT_DETECT,
        obs=_TWO_ALT_OBS,
        also_plausible=[0, 1],
        chosen_shape_index=2,
    )


def test_two_alt_survivors_is_candidates_not_definite():
    result = _run_two_alt_scenario()
    assert result.status == "candidates", (
        f"expected candidates (2+ survivors satisfy CandidateSet's min_length=2), "
        f"got {result.status!r}"
    )
    assert len(result.specs) == 2


def test_two_alt_survivors_warning_names_the_elected_failure():
    result = _run_two_alt_scenario()
    warning = next((w for w in result.warnings if w.code == ELECTED_SHAPE_NO_DATA), None)
    assert warning is not None, (
        f"expected ELECTED_SHAPE_NO_DATA warning, got {[w.code for w in result.warnings]}"
    )
