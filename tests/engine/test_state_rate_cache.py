"""`resolve_variable` passes `rate_cache` through `ResolveState` to observation building."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from materialise.rate_tables import RateTableCache
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.observations import dispatch as preview_dispatch
from search.engine.region.state import EngineOptions
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import materialise_double_from_obs
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver


class _SoleShapeLLM:
    """Elects the sole (unconstrained) candidate shape, no bindings needed."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return (
            elected_reply(0, also_plausible=None),
            None,
        )


_ENTITY = "country/KEN"
_SV = "Test_RateCache_Expenditure"
_QUERY = "rate cache test variable in Kenya"

_SV_ARCS: dict = {
    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
    "populationType": {"nodes": [{"dcid": "TestRateCacheActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "testRateCacheExpenditure"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
}

_NODES: dict = {
    _ENTITY: {"label": "Kenya", "type": "Country"},
    _SV: {"label": "Rate cache expenditure", "type": "StatisticalVariable", "arcs": _SV_ARCS},
}


def _obs_facet() -> dict:
    return {
        "earliestDate": "2020",
        "latestDate": "2020",
        "obsCount": 1,
        "provenanceId": "dc/base/RateCacheSrc",
        "dates": ["2020"],
        "values": [10.0],
        "facetId": "f1",
    }


def _run(*, rate_cache: RateTableCache | None):
    obs = {f"{_SV}|{_ENTITY}": [_obs_facet()]}
    graph = FakeGraph(
        nodes=_NODES,
        detect={_QUERY: {"svs": [_SV], "cosine_scores": [0.95]}},
        resolve={"Kenya": _ENTITY},
        obs=obs,
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    return asyncio.run(
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
            llm=_SoleShapeLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(
                include_observations=True,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=rate_cache,
            ),
        )
    )


def test_rate_cache_reaches_build_observations():
    """rate_cache passed to resolve_variable lands on the ResolveState it
    builds, and reaches _build_observations through the include_observations
    fetch site exactly as materialise_graph/materialise_sdmx do."""
    sentinel = RateTableCache()
    captured: list[object] = []
    real_build_observations = preview_dispatch._build_observations

    def _spy(*args, **kwargs):
        captured.append(kwargs.get("rate_cache"))
        return real_build_observations(*args, **kwargs)

    with patch.object(preview_dispatch, "_build_observations", _spy):
        result = _run(rate_cache=sentinel)

    assert result.status == "definite", f"expected definite, got {result.status!r}"
    assert captured, "expected the include_observations fetch to call _build_observations"
    assert all(c is sentinel for c in captured)


def test_resolve_variable_with_no_rate_cache_still_serves():
    """None (every caller today) reaches _build_observations as None and
    the resolve path serves exactly as it did before ResolveState carried
    this field."""
    captured: list[object] = []
    real_build_observations = preview_dispatch._build_observations

    def _spy(*args, **kwargs):
        captured.append(kwargs.get("rate_cache"))
        return real_build_observations(*args, **kwargs)

    with patch.object(preview_dispatch, "_build_observations", _spy):
        result = _run(rate_cache=None)

    assert result.status == "definite", f"expected definite, got {result.status!r}"
    assert captured, "expected the include_observations fetch to call _build_observations"
    assert all(c is None for c in captured)
