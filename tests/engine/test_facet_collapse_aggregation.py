"""Checks that a single surviving candidate aggregates `resolved.facets` without a bound unit."""

from __future__ import annotations

import asyncio

from search.engine.bind import _UniversalBindOutput
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.state import EngineOptions
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import materialise_double_from_obs
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_SV_SURVIVOR = "Test_FacetCollapse_SV_Survivor"
_SV_DROPPED = "Test_FacetCollapse_SV_Dropped"
_QUERY = "facet collapse aggregation test variable in Kenya"


def _std_arcs(label: str, *, meas_prop: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }


_NODES = {
    "country/KEN": {"label": "Kenya", "type": "Country"},
    _SV_SURVIVOR: {
        "label": "Facet collapse SV survivor",
        "type": "StatisticalVariable",
        "arcs": _std_arcs("Facet collapse SV survivor", meas_prop="amountSurvivor"),
    },
    _SV_DROPPED: {
        "label": "Facet collapse SV dropped",
        "type": "StatisticalVariable",
        "arcs": _std_arcs("Facet collapse SV dropped", meas_prop="amountDropped"),
    },
}
_DETECT = {
    _QUERY: {
        "svs": [_SV_SURVIVOR, _SV_DROPPED],
        "cosine_scores": [0.9, 0.88],
    },
}
# SV_SURVIVOR has one in-window USD observation. SV_DROPPED has none.
# The surviving StatVar reaches the collapse-to-definite branch.
_OBS = {
    f"{_SV_SURVIVOR}|country/KEN": [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/Src",
            "unit": "USD",
            "dates": ["2020"],
            "values": [10.0],
        }
    ],
}


class _CollapseBindLLM:
    """Elects SV_SURVIVOR (index 0) with no constraint bindings and flags
    SV_DROPPED (index 1) as also_plausible -- the ambiguity signal that
    drives region/pipeline.py's candidates fan-out."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[],
            also_plausible=[1],
            ask=None,
        ), None


def test_collapse_to_definite_aggregates_resolved_facets_with_no_bound_unit():
    """A single surviving candidate aggregates facets without a bound unit."""
    graph = FakeGraph(
        nodes=_NODES,
        detect=_DETECT,
        resolve={"Kenya": "country/KEN"},
        obs=_OBS,
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_OBS)
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
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_CollapseBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(
                include_observations=True,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
            ),
        )
    )
    assert result.status == "definite"
    spec = result.specs[0]
    assert spec.resolved.facets, (
        "resolved.facets must be aggregated on the collapse-to-definite path, "
        "not left empty just because no unit was bound"
    )
    assert spec.resolved.facets["unit"].ref == "USD"
    assert spec.resolved.facets["unit"].basis == "fallback"
