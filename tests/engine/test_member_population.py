"""#893: an unbound standard answer ships one elected StatVar but measures the
full confirmed population. These tests pin population counts and exactness at
the resolver and response levels, including capped recall samples.
"""

from __future__ import annotations

import asyncio

from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.region import resolve_variable
from search.engine.retrieval.retrieve import Materialised, materialise
from search.engine.retrieval.role_resolve import graph_confirm_resolve
from search.engine.shape import ShapeDraft
from series_coverage.schema import RoleNames
from tests.engine import test_index_sourced_grounding as _idx
from tests.engine._fake_coverage import FakeCoverageService, empty_coverage_snapshot, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._index_slots import index_slot_keys
from tests.engine.test_standard_resolve import (
    _RECIPIENT,
    _SET_SV_A,
    _SET_SV_B,
    _graph_with_data_for,
    _two_member_set_shape,
)
from tests.fixtures import FakeGraph, make_resolver

_MEMBER_A = "TestPopulationMember_A"
_MEMBER_B = "TestPopulationMember_B"
_MEMBER_C = "TestPopulationMember_C"


def _three_member_shape() -> ShapeDraft:
    """Three members of one shape, none declaring a constraint property, so
    an entirely unbound query (bindings=[]) excludes none of them."""
    return ShapeDraft(
        shape_id="test_member_population_shape",
        label="test population shape",
        pop_type_dcid="TestPopulationActivity",
        meas_prop_dcid="testPopulationCount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_MEMBER_A: {}, _MEMBER_B: {}, _MEMBER_C: {}},
        representative_score=0.9,
    )


def test_standard_resolve_confirmed_population_exceeds_the_elected_winner():
    """All three unbound members have data, but the single-match branch ships
    only the first. `population.confirmed` counts all three (#893).
    """
    shape = _three_member_shape()
    graph = _graph_with_data_for(_MEMBER_A, _MEMBER_B, _MEMBER_C)

    result = materialise(
        shape,
        [],
        {"observationAbout": [_RECIPIENT]},
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_MEMBER_A]
    assert result.population is not None
    assert result.population.admitted == 3
    assert result.population.confirmed == 3
    assert result.population.exact is True


def test_set_binding_collected_count_matches_population_confirmed():
    """A set binding ships every confirmed member, so both counts agree."""
    shape = _two_member_set_shape()
    bindings = [
        SlotBindingDraft(
            axis="how",
            property_dcid="financingSource",
            kind="set",
            value_dcids=["SourceA", "SourceB"],
        ),
    ]
    graph = _graph_with_data_for(_SET_SV_A, _SET_SV_B)

    result = materialise(
        shape,
        bindings,
        {"observationAbout": [_RECIPIENT]},
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SET_SV_A, _SET_SV_B]
    assert result.population is not None
    assert result.population.confirmed == len(result.sv_dcids) == 2


def _index_sourced_shape() -> ShapeDraft:
    """The packaged index contains this CRS five-tuple. The one-member recall
    pool is incomplete; `declared_roles` are fixture labels and do not affect
    five-tuple matching.
    """
    return ShapeDraft(
        shape_id="test_population_exact_index_sourced_shape",
        label="test index-sourced shape",
        pop_type_dcid=_idx._POP_TYPE,
        meas_prop_dcid=_idx._MEAS_PROP,
        stat_type_dcid=_idx._STAT_TYPE,
        meas_qual_dcid=_idx._MEAS_QUAL,
        meas_denom_dcid=None,
        slot_keys=index_slot_keys(
            population_type=_idx._POP_TYPE,
            measured_property=_idx._MEAS_PROP,
            stat_type=_idx._STAT_TYPE,
            measurement_qualifier=_idx._MEAS_QUAL,
        ),
        sv_arc_facts={_idx._RECALLED_SV: {}},
        sv_arc_facts_complete=False,
        declared_roles=("creditor", "debtor"),
        representative_score=0.9,
    )


def test_index_sourced_population_is_exact(monkeypatch):
    """The packaged index carries this CRS five-tuple, so its full population
    is exact despite an incomplete one-member recall pool (#893).
    """
    shape = _index_sourced_shape()
    bindings = [
        SlotBindingDraft(
            axis="how",
            property_dcid=_idx._PROP_PURPOSE,
            kind="value",
            value_dcids=[_idx._EXTRA_PURPOSE],
        )
    ]
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_idx._EXTRA_SV, "f1", "country/FRA", _idx._KEN_DCID, [2020])],
        role_names={
            _idx._EXTRA_SV: RoleNames(donor="financingProvider", recipient="financingRecipient")
        },
    )

    result = graph_confirm_resolve(
        shape=shape,
        bindings=bindings,
        subject_dcids=[_idx._KEN_DCID],
        origin_dcids=["country/FRA"],
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage_snapshot=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.population is not None
    assert result.population.exact is True
    assert result.population.admitted >= 1
    assert result.population.confirmed == 1
    assert result.sv_dcids == [_idx._EXTRA_SV]


_RECALL_MEMBER = "TestRecallPoolMember_A"


def _recall_pool_shape() -> ShapeDraft:
    """A fabricated five-tuple this deployment's packaged index does not
    carry, with sv_arc_facts_complete=False -- derive_shapes' own stamp for
    every shape its capped confirm fetch produces, regardless of how many
    members that fetch actually returned."""
    return ShapeDraft(
        shape_id="test_population_exact_recall_pool_shape",
        label="test recall-pool shape",
        pop_type_dcid="TestRecallPoolActivity",
        meas_prop_dcid="testRecallPoolCount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_RECALL_MEMBER: {}},
        sv_arc_facts_complete=False,
        representative_score=0.9,
    )


def test_recall_pool_population_is_inexact():
    """This synthetic five-tuple misses the index, so the capped recall pool
    supplies only a floor and `exact` is false."""
    shape = _recall_pool_shape()
    graph = _graph_with_data_for(_RECALL_MEMBER)

    result = materialise(
        shape,
        [],
        {"observationAbout": [_RECIPIENT]},
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.population is not None
    assert result.population.exact is False
    assert result.population.admitted == 1
    assert result.population.confirmed == 1


_QUERY_POP = "population"
_PLACE_TEXT = "Testland"
_PLACE_DCID = "country/TST"

_BARE_SV = "TestCount_Person"
_MALE_SV = "TestCount_Person_Male"
_FEMALE_SV = "TestCount_Person_Female"


def _population_arcs(gender: str | None) -> dict:
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": "TestPerson"}]},
        "measuredProperty": {"nodes": [{"dcid": "testCount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if gender is None:
        arcs["constraintProperties"] = {"nodes": []}
    else:
        arcs["constraintProperties"] = {"nodes": [{"dcid": "gender"}]}
        arcs["gender"] = {"nodes": [{"dcid": gender}]}
    return arcs


_NODES_POP = {
    _PLACE_DCID: {"label": "Testland", "type": "Country"},
    _BARE_SV: {
        "label": "test bare population",
        "type": "StatisticalVariable",
        "arcs": _population_arcs(None),
    },
    _MALE_SV: {
        "label": "test male population",
        "type": "StatisticalVariable",
        "arcs": _population_arcs("Male"),
    },
    _FEMALE_SV: {
        "label": "test female population",
        "type": "StatisticalVariable",
        "arcs": _population_arcs("Female"),
    },
}
_DETECT_POP = {
    _QUERY_POP: {"svs": [_BARE_SV, _MALE_SV, _FEMALE_SV], "cosine_scores": [0.9, 0.85, 0.8]}
}
_OBS_POP = {
    f"{sv}|{_PLACE_DCID}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}]
    for sv in (_BARE_SV, _MALE_SV, _FEMALE_SV)
}


class _NoBindingLLM:
    """Returns no bindings, so no member is excluded by a bound value."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return (
            _UniversalBindOutput(chosen_shape_index=0, bindings=[], also_plausible=None, ask=None),
            None,
        )


def _run_population_query():
    graph = FakeGraph(
        nodes=_NODES_POP, obs=_OBS_POP, detect=_DETECT_POP, resolve={_PLACE_TEXT: _PLACE_DCID}
    )
    return asyncio.run(
        resolve_variable(
            _QUERY_POP,
            entities=[_PLACE_TEXT],
            date_request=None,
            detect_query=_QUERY_POP,
            role_query=_QUERY_POP,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_NoBindingLLM(),
            base_steps=[],
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )


def test_definite_standard_response_reports_the_confirmed_population_not_just_the_elected_winner():
    """Three unbound siblings have data, while the single-match branch ships
    only the bare aggregate. The response counts all three and marks its one
    shipped item as truncated, reproducing #893.
    """
    result = _run_population_query()

    assert result.status == "definite"
    spec = result.specs[0]
    resolved = spec.resolved

    assert [sv.ref for sv in resolved.stat_vars.items] == [_BARE_SV]
    assert len(resolved.stat_vars.items) == 1
    assert resolved.stat_vars.member_count == 3
    assert resolved.stat_vars.truncated is True
    assert resolved.shape.member_count == 3
