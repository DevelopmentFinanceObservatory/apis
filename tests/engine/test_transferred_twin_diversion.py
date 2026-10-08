"""A UN series shape reached through a successor diverts with its registered base twin.

Our resolver recalls the World Bank variable and the superseded base SDG variable, which
stands as the UN headline variable the registry clusters with the World Bank's. The series'
breakdown members, which recall did not find, stay out of the shape, so the twin diversion
compares the headline with the World Bank variable."""

from __future__ import annotations

import asyncio

from origin_catalogue import BASE, UN_SYSTEM
from search.engine.bind import _UniversalBindOutput
from search.engine.region.pipeline import resolve_variable
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver
from tests.origin_catalogue.fake_catalogue import (
    provenance_row,
    restrictive_catalogue,
    un_mirror,
    un_variable,
)

_RWA = "country/RWA"
_QUERY = "access to electricity"
_WDI = "dc/base/WorldDevelopmentIndicators"
_WB = "worldBank/EG_ELC_ACCS_ZS"
_BASE_SDG = "sdg/EG_ACS_ELEC"
_SERIES = "undata/sdg/SERIES-EG_ACS_ELEC"
_HEADLINE = "undata/sdg/EG_ACS_ELEC"
_URBAN = "undata/sdg/EG_ACS_ELEC.URBANISATION--U"
_RURAL = "undata/sdg/EG_ACS_ELEC.URBANISATION--R"
_URBANISATION = "undataUrbanisation"


def _node(dcid: str, *, population_type: str, urbanisation: str | None = None) -> dict:
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": dcid}]},
        "populationType": {"nodes": [{"dcid": population_type}]},
        "measuredProperty": {"nodes": [{"dcid": "value"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": []},
    }
    if urbanisation is not None:
        arcs["constraintProperties"] = {"nodes": [{"dcid": _URBANISATION}]}
        arcs[_URBANISATION] = {"nodes": [{"dcid": urbanisation}]}
    return {"label": dcid, "type": "StatisticalVariable", "arcs": arcs}


_NODES = {
    _RWA: {"label": "Rwanda", "type": "Country"},
    _WB: _node(_WB, population_type="Household"),
    _HEADLINE: _node(_HEADLINE, population_type=_SERIES),
    _URBAN: _node(_URBAN, population_type=_SERIES, urbanisation="undata/URBANISATION-U"),
    _RURAL: _node(_RURAL, population_type=_SERIES, urbanisation="undata/URBANISATION-R"),
}


def _catalogue():
    breakdown = {
        dcid: un_variable(
            _SERIES, name=dcid, constraints={_URBANISATION: ("Urbanisation", value, label)}
        )
        for dcid, value, label in (
            (_URBAN, "undata/URBANISATION-U", "Urban"),
            (_RURAL, "undata/URBANISATION-R", "Rural"),
        )
    }
    return restrictive_catalogue(
        provenances=[
            provenance_row(_WDI, BASE),
            provenance_row("undata/p/SDG", UN_SYSTEM, segment="sdg"),
        ],
        base_variables={_WB: ([_WDI], [])},
        withheld_variables=[_BASE_SDG],
        successors={_BASE_SDG: _HEADLINE},
        un_mirror=un_mirror({_HEADLINE: un_variable(_SERIES, name=_HEADLINE), **breakdown}),
    )


def _facet(provenance_id: str) -> dict:
    return {
        "earliestDate": "2015",
        "latestDate": "2020",
        "obsCount": 5,
        "provenanceId": provenance_id,
    }


class _ElectFirstNamingSecond:
    """Elects the first shape and names the second as also plausible, binding nothing."""

    async def generate_structured(self, *, prompt, system, schema):
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=[1], ask=None
        ), None


def _resolve() -> RegionResult:
    graph = FakeGraph(
        nodes=_NODES,
        obs={
            f"{_WB}|{_RWA}": [_facet(_WDI)],
            **{f"{dcid}|{_RWA}": [_facet("undata/p/SDG")] for dcid in (_HEADLINE, _URBAN, _RURAL)},
        },
        detect={_QUERY: {"svs": [_WB, _BASE_SDG], "cosine_scores": [0.9, 0.85]}},
        un_detect={_QUERY: {"svs": [], "cosine_scores": []}},
        resolve={"Rwanda": _RWA},
    )
    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Rwanda"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_ElectFirstNamingSecond(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(deadline=None),
        )
    )


def _members(spec) -> set[str]:
    return {sv.ref.dcid for sv in spec.resolved.members}


def test_a_transferred_series_with_breakdown_members_diverts_with_its_base_twin(
    bind_catalogue,
) -> None:
    bind_catalogue(_catalogue())

    result = _resolve()

    assert result.status == "definite"
    (answer,) = result.specs
    assert _members(answer) == {_WB}
    (twin,) = answer.resolved.equivalent_readings or ()
    assert _members(twin) == {_HEADLINE}
    assert twin.resolved.shape.member_count == 1
