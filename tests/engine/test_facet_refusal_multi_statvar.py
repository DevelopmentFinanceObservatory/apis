"""Checks per-StatVar facet enforcement through a resubmit with a two-member `how` set.

A binding that only one StatVar satisfies must refuse `facet_not_available`.
"""

from __future__ import annotations

import pytest

from materialise.graph import SubjectPage
from search.engine.clients.graph import Facet
from search.engine.errors import EngineInputError
from search.engine.facet_refusals import candidate_facet_refusal
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.models import IdentityBindingValue, ResubmitIdentityPayload
from tests.engine._base_identity import make_identity, subject_entity
from tests.engine._builders import binding_set_slot as _binding_set_slot
from tests.engine._builders import value_slot as _value_slot
from tests.engine._conjoin_fixtures import member_departures
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


def _obs(*, method_a: str, method_b: str, provenance_b: str = "dc/base/SourceB") -> dict:
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
                "provenanceId": provenance_b,
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


@pytest.mark.parametrize(
    ("provenance_b", "series", "left_out"),
    [
        pytest.param("dc/base/SourceB", {_SV_A}, ["Test Expenditure SourceB"], id="two-sources"),
        pytest.param("dc/base/SourceA", {_SV_A, _SV_B}, [], id="one-source"),
    ],
)
def test_binding_satisfied_by_every_stat_var_serves_the_one_group(
    provenance_b: str, series: set[str], left_out: list[str]
):
    """Both StatVars satisfy the bound method, so the resubmit answers definite with both
    members. Under two sources the members of one Spec serve one group: SourceA's, which
    holds more observations, so SV_A alone contributes a series and the part names SV_B in a
    `member_unavailable` departure. Under one source both serve and none departs."""
    obs = _obs(
        method_a="GrossDisbursement", method_b="GrossDisbursement", provenance_b=provenance_b
    )
    region = _resubmit(obs, {"measurementMethod": IdentityBindingValue(ref="GrossDisbursement")})
    assert region.status == "definite", f"expected definite, got {region.status!r}"
    spec = region.specs[0]
    assert {sv.ref.ref for sv in spec.resolved.members} == {_SV_A, _SV_B}
    assert spec.resolved.observations is not None
    assert {s.stat_var.ref for s in spec.resolved.observations.series} == series
    assert [asked for _ref, asked, _shown in member_departures(region)] == left_out


def test_lifetime_source_does_not_mask_windowed_method_missing_from_one_stat_var():
    """Every StatVar must hold method pins in-window even when its source exists lifetime."""
    lifetime = {
        sv: [
            Facet(
                earliest_date="2015",
                latest_date="2020",
                obs_count=1,
                provenance_id="dc/base/Common",
                measurement_method="GrossDisbursement",
            )
        ]
        for sv in (_SV_A, _SV_B)
    }
    windowed = {_SV_A: lifetime[_SV_A], _SV_B: []}
    refusal = candidate_facet_refusal(
        FakeGraph(),
        windowed,
        {
            "provenance": IdentityBindingValue(ref="dc/base/Common"),
            "measurementMethod": IdentityBindingValue(ref="GrossDisbursement"),
        },
        provenance_facets_by_sv=lifetime,
        deadline=None,
    )
    assert refusal is not None
    assert refusal.pointer == "/input/facets/measurementMethod"


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
    assert {sv.ref.ref for sv in region.specs[0].resolved.members} == {_SV_A}
