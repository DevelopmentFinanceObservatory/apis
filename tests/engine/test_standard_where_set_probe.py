"""A standard answer over a `where` set probes every place in the set, so a set whose data
sits only at a place after the first still answers with it."""

from __future__ import annotations

import asyncio

from search.engine.bind import SlotBindingDraft
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.retrieval.read_plan import PairPlan
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.standard import standard_resolve
from search.engine.shape import ShapeDraft
from search.models import BindingSet, CoverageExact
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_SV = "Count_Person"
_GTM, _SLV = "country/GTM", "country/SLV"
_SOURCE = "dc/base/WhereSetFixture"

# El Salvador alone holds data; Guatemala, named first, holds none.
_OBS = {
    f"{_SV}|{_SLV}": [
        {"earliestDate": "2021", "latestDate": "2021", "obsCount": 1, "provenanceId": _SOURCE}
    ]
}
_ARCS = {
    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
    "name": {"nodes": [{"value": "Count Person"}]},
    "populationType": {"nodes": [{"dcid": "Person"}]},
    "measuredProperty": {"nodes": [{"dcid": "count"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
}
_NODES: dict = {
    _SV: {"label": "Count Person", "type": "StatisticalVariable", "arcs": _ARCS},
    _GTM: {"label": "Guatemala", "type": "Country"},
    _SLV: {"label": "El Salvador", "type": "Country"},
}


def _shape() -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_where_set_shape",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_SV: {}},
        representative_score=0.9,
    )


def test_standard_resolve_probes_every_place_of_a_where_set():
    """The set binds Guatemala and El Salvador. A probe of Guatemala alone finds nothing;
    the plan probes both and serves El Salvador's series."""
    result = standard_resolve(
        shape=_shape(),
        bindings=[
            SlotBindingDraft(axis="where", property_dcid=None, kind="set", value_dcids=[_GTM, _SLV])
        ],
        entities={"observationAbout": [_GTM, _SLV]},
        graph=FakeGraph(nodes={}, obs=_OBS, detect={}, resolve={}),
        place_dcids=[_GTM, _SLV],
    )

    assert isinstance(result, Materialised)
    assert result.has_data is True
    plan = result.read_plan
    assert isinstance(plan, PairPlan)
    assert plan.places == (_GTM, _SLV)
    assert [s.role_tuple for s in plan.served] == [(("observationAbout", _SLV),)]


class _FixedBindLLM:
    """Elects the sole candidate shape with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, also_plausible=None), None


def test_a_first_answer_over_two_named_places_serves_the_second_places_data():
    """ "Population of Guatemala and El Salvador" binds both as a set, and the answer serves
    El Salvador's data although Guatemala, named first, holds none."""
    query = "population"
    graph = FakeGraph(
        nodes=_NODES,
        detect={query: {"svs": [_SV], "entities": []}},
        resolve={"Guatemala": _GTM, "El Salvador": _SLV},
        obs=_OBS,
    )
    result = asyncio.run(
        resolve_variable(
            query,
            entities=["Guatemala", "El Salvador"],
            date_request=None,
            detect_query=query,
            role_query="population Guatemala El Salvador",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.status == "definite", result
    resolved = result.specs[0].resolved
    where = next(s for s in resolved.slots if s.key.axis == "where")
    assert isinstance(where.binding, BindingSet)
    assert [v.ref.ref for v in where.binding.values] == [_GTM, _SLV]
    assert isinstance(resolved.coverage, CoverageExact)
    assert resolved.coverage.has_data is True
    assert resolved.coverage.observation_count == 1
