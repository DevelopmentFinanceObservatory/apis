"""Checks per-StatVar facet enforcement through a resubmit with a two-member `how` set.

A binding that only one StatVar satisfies must refuse `facet_not_available`.
"""

from __future__ import annotations

import pytest

from materialise.graph import SubjectPage
from search.engine.errors import EngineInputError
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.models import IdentityBindingValue, ResubmitIdentityPayload
from tests.engine._base_identity import make_identity, subject_entity
from tests.engine._builders import binding_set_slot as _binding_set_slot
from tests.engine._builders import value_slot as _value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import materialise_double_from_obs, walk_options
from tests.fixtures import FakeGraph

_ENTITY = "country/USA"
# Mirrors _FIVE_TUPLE_ARCS below, the five-tuple the resubmit's member walk filters on.
_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "TestExpenditureActivity",
    "measured_property": "testExpenditure",
    "stat_type": "measuredValue",
}
_SV_A = "Test_Expenditure_SourceA"
_SV_B = "Test_Expenditure_SourceB"
_PROP = "financingSource"
_VAL_A = "SourceA"
_VAL_B = "SourceB"

_FIVE_TUPLE_ARCS = {
    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
    "populationType": {"nodes": [{"dcid": "TestExpenditureActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "testExpenditure"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
}


def _sv_arcs(value: str) -> dict:
    return {
        **_FIVE_TUPLE_ARCS,
        "constraintProperties": {"nodes": [{"dcid": _PROP}]},
        _PROP: {"nodes": [{"dcid": value}]},
    }


_NODES: dict = {
    _SV_A: {
        "label": "Test Expenditure SourceA",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(_VAL_A),
    },
    _SV_B: {
        "label": "Test Expenditure SourceB",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(_VAL_B),
    },
    _ENTITY: {"label": "United States", "type": "Country"},
}

_HOW_SET_SLOT = _binding_set_slot("how", _PROP, "financing source", [_VAL_A, _VAL_B])


def _member_arcs(node_arcs: dict) -> dict[str, tuple[str, ...]]:
    """Flatten node arcs to the tuple form read by both graph doubles."""
    return {
        prop: tuple(n.get("dcid") or n.get("value") for n in entry["nodes"])
        for prop, entry in node_arcs.items()
    }


def _obs(*, method_a: str, method_b: str) -> dict:
    return {
        f"{_SV_A}|{_ENTITY}": [
            {
                "earliestDate": "2015",
                "latestDate": "2020",
                "obsCount": 3,
                "provenanceId": "dc/base/SourceA",
                "measurementMethod": method_a,
            }
        ],
        f"{_SV_B}|{_ENTITY}": [
            {
                "earliestDate": "2015",
                "latestDate": "2020",
                "obsCount": 2,
                "provenanceId": "dc/base/SourceB",
                "measurementMethod": method_b,
            }
        ],
    }


def _resubmit(obs: dict, facets: dict):
    graph = FakeGraph(nodes=_NODES, obs=obs)
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = {
        _SV_A: _member_arcs(_NODES[_SV_A]["arcs"]),
        _SV_B: _member_arcs(_NODES[_SV_B]["arcs"]),
    }
    materialise_graph._subjects = {
        ("testExpenditure", "measuredProperty"): SubjectPage(dcids=(_SV_A, _SV_B), truncated=False),
    }
    entities = [subject_entity(_ENTITY)]
    base_identity = make_identity(_SHAPE, slots=[_HOW_SET_SLOT], entities=entities)
    inp = ResubmitIdentityPayload(
        identity=base_identity.model_copy(update={"facets": facets}),
    )
    return resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            graph,
            EngineOptions(
                include_observations=True,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
            ),
        ),
    )


def test_binding_satisfied_by_only_one_stat_var_refuses_through_resubmit():
    """The resubmit refuses when only SV_A has the bound measurement method."""
    obs = _obs(method_a="GrossDisbursement", method_b="Commitment")
    with pytest.raises(EngineInputError) as exc_info:
        _resubmit(obs, {"measurementMethod": IdentityBindingValue(ref="GrossDisbursement")})
    assert exc_info.value.code == "facet_not_available"
    assert exc_info.value.pointer == "/input/identity/facets/measurementMethod"


def test_binding_satisfied_by_every_stat_var_answers_definite_with_all_series():
    """Both StatVars satisfy the binding and contribute a series."""
    obs = _obs(method_a="GrossDisbursement", method_b="GrossDisbursement")
    region = _resubmit(obs, {"measurementMethod": IdentityBindingValue(ref="GrossDisbursement")})
    assert region.status == "definite", f"expected definite, got {region.status!r}"
    spec = region.specs[0]
    assert {sv.ref.dcid for sv in spec.resolved.members} == {_SV_A, _SV_B}
    assert spec.resolved.observations is not None
    assert {s.stat_var.ref for s in spec.resolved.observations.series} == {_SV_A, _SV_B}


def test_single_stat_var_behavior_unaffected_by_the_per_sv_check():
    """A single-StatVar resubmit still returns a definite result."""
    graph = FakeGraph(nodes=_NODES, obs=_obs(method_a="GrossDisbursement", method_b="Commitment"))
    entities = [subject_entity(_ENTITY)]
    value_slot_a = _value_slot("how", _PROP, "financing source", _VAL_A, _VAL_A)
    base_identity = make_identity(_SHAPE, slots=[value_slot_a], entities=entities)
    inp = ResubmitIdentityPayload(
        identity=base_identity.model_copy(
            update={"facets": {"measurementMethod": IdentityBindingValue(ref="GrossDisbursement")}}
        ),
    )
    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(graph, EngineOptions(include_observations=True)),
    )
    assert region.status == "definite", f"expected definite, got {region.status!r}"
    assert {sv.ref.dcid for sv in region.specs[0].resolved.members} == {_SV_A}
