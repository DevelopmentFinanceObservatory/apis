"""A standard query with two entities resolves definite, and its identity
re-hashes to the returned `spec_id`. The fixed binder isolates identity
reproduction from shape choice.
"""

from __future__ import annotations

import asyncio

from dfo_spec_kit import identity as _identity_core

from search.engine.bind import _UniversalBindOutput
from search.engine.region import RegionResult, resolve_variable
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

_SV = "Count_Person"
_DETECT_QUERY = "population"

_NODES: dict = {
    _SV: {
        "label": "Count Person",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Count Person"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    "country/FRA": {"label": "France", "type": "Country"},
    "country/DEU": {"label": "Germany", "type": "Country"},
}
_DETECT: dict = {_DETECT_QUERY: {"svs": [_SV], "entities": []}}
_RESOLVE: dict = {
    "France": "country/FRA",
    "Germany": "country/DEU",
}


class _FixedBindLLM:
    """Elects the sole candidate shape with no constraint bindings, unconditionally."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[],
            also_plausible=None,
            ask=None,
        ), None


def _run(entities: list[str], *, obs: dict) -> RegionResult:
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve=_RESOLVE, obs=obs)
    return asyncio.run(
        resolve_variable(
            _DETECT_QUERY,
            entities=entities,
            date_request=None,
            detect_query=_DETECT_QUERY,
            role_query=f"population {' '.join(entities)}",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def test_two_entity_part_reproduces_its_own_spec_id():
    """Both entities reach the identity and resolved entity list, and the
    returned identity re-hashes to its minted `spec_id`."""
    obs = {
        f"{_SV}|country/FRA": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1},
        ],
        f"{_SV}|country/DEU": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1},
        ],
    }
    result = _run(["France", "Germany"], obs=obs)
    assert result.status == "definite", f"expected definite, got {result.status!r}"
    spec = result.specs[0]
    rehashed = _identity_core.spec_id_from_value(
        spec.definition.identity.model_dump(exclude_none=True)
    )
    assert rehashed == spec.definition.spec_id
    assert spec.spec_id == spec.definition.spec_id

    where_set = {dcid for dcids in spec.definition.identity.entities.values() for dcid in dcids}
    assert where_set == {"country/FRA", "country/DEU"}

    resolved_dcids = {entity.ref for entity in spec.resolved.entities}
    assert resolved_dcids == {"country/FRA", "country/DEU"}
