"""An unavailable-place carrier must pass through the bound-unit filter so the resolver can return
`NO_MEMBER_HAS_DATA_AT_PLACE`, not the generic `facet_not_available` refusal.

Two distinct shapes drive the candidates path. Neither has data, and the unavailable carrier alone
survives deduplication.
"""

from __future__ import annotations

import asyncio

from search.engine.bind import _UniversalBindOutput
from search.engine.region import (
    NO_MEMBER_HAS_DATA_AT_PLACE,
    PLACE_SCOPE_DEFAULTED,
    resolve_variable,
)
from search.engine.region.state import EngineOptions
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import sdmx_double
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise.conftest import FakeGraphReader

_DETECT_QUERY = "test place scope population or count"
_SV_A = "TEST/AvailAtPlaceUnitFilter_PopulationA"
_SV_B = "TEST/AvailAtPlaceUnitFilter_PopulationB"
_EARTH_CHILDREN = {
    "Earth": {"Country": ["country/A1", "country/A2", "country/A3"]},
}


def _sv_arcs(label: str, *, meas_prop: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }


_NODES: dict = {
    _SV_A: {
        "label": "Test Population A",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs("A", meas_prop="count"),
    },
    _SV_B: {
        "label": "Test Population B",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs("B", meas_prop="countAdult"),
    },
}
_DETECT: dict = {_DETECT_QUERY: {"svs": [_SV_A, _SV_B], "cosine_scores": [0.9, 0.85]}}


class _CandidatesBindLLM:
    """Elect shape 0 and send shape 1 through the candidates path."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=[1], ask=None
        ), None


def _run():
    graph = FakeGraph(
        nodes=_NODES,
        detect=_DETECT,
        resolve={},
        obs={},
        children=_EARTH_CHILDREN,
    )
    return asyncio.run(
        resolve_variable(
            _DETECT_QUERY,
            entities=[],
            date_request=None,
            detect_query=_DETECT_QUERY,
            role_query=_DETECT_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_CandidatesBindLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            unit_text="USD",
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(
                # Empty role bindings skip unit selection. The client satisfies the request guard.
                materialise_graph=FakeGraphReader(),
                materialise_sdmx=sdmx_double(),
            ),
        )
    )


def test_the_flagged_unavailable_carrier_still_reports_no_member_has_data_at_place():
    """The bound-unit filter must preserve the unavailable carrier so its specific refusal fires."""
    result = _run()
    assert result.status == "no_data", result
    assert result.no_data_reason == "no_observations"
    assert result.gate == NO_MEMBER_HAS_DATA_AT_PLACE
    assert result.flagged_match is not None and len(result.flagged_match) == 1
    assert any(w.code == PLACE_SCOPE_DEFAULTED for w in result.warnings)
