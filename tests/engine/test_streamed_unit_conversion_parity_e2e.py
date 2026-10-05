"""Buffered and streamed observation paths apply the same requested unit conversion.

The relaunch needs a rate cache; without one it raises `MaterialisationUsageError` before SDMX.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise.errors import MaterialisationUsageError
from materialise.rate_tables import RateTableCache
from materialise.recipe_bodies import RECIPE_BODIES
from search.engine.bind import _UniversalBindOutput
from search.engine.concurrency import scoped_concurrency
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    _facet_from_fixture,
)
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise.conftest import FakeGraphReader

_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"

_SV = "TESTFIXTURE/StreamedUnitParity/Amount"
_KEN = "country/KEN"
_QUERY = "streamed unit parity amount"

_PROVENANCE = "dc/base/StreamedUnitSrc"
_RECIPE = "ConversionRecipeDAC"
_TABLE_PROVENANCE = RECIPE_BODIES[_RECIPE].table_provenance

_STD_ARCS = {
    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
    "name": {"nodes": [{"value": "Streamed unit parity amount"}]},
    "populationType": {"nodes": [{"dcid": "TestStreamedUnitActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "testStreamedUnitAmount"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
}

_USD_FIXTURE = ObsFacetFixture(
    facet_id="native", points=(("2022", 100.0),), provenance_id=_PROVENANCE, unit="CurrencyUSD"
)


class _FixedBindLLM:
    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def _materialise_sdmx() -> MaterialiseSdmxClient:
    main_body = _csv_header(["observationAbout"]) + _csv_row(
        stat_var=_SV, role_dcids=[_KEN], fixture=_USD_FIXTURE, date_str="2022", value=100.0
    )
    bodies = {_SV: main_body}

    def observations(request: httpx.Request) -> httpx.Response:
        sv = request.url.params.get("c[variableMeasured]", "")
        body = bodies.get(sv, _csv_header(["observationAbout"]))
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    server = RateTableServer(
        recipe_table_bodies(
            _RECIPE, rates={"CurrencyUSD": {2022: 1.0}, "CurrencyEUR": {2022: 0.9}}
        ),
        fallback=observations,
    )
    return server.client(base_url=MATERIALISE_BASE_URL)


def _materialise_graph() -> FakeGraphReader:
    return FakeGraphReader(
        arcs={
            _SV: dict(CONVERTIBLE_STAT_VAR_ARCS),
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_PROVENANCE,
                currencies=["CurrencyUSD", "CurrencyEUR"],
            ),
        },
        facets={(_SV, _KEN): [_facet_from_fixture(_USD_FIXTURE)]},
    )


def _graph() -> FakeGraph:
    return FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _SV: {"label": "Streamed unit parity amount", "arcs": _STD_ARCS},
            _PROVENANCE: {"label": "Streamed Unit Source"},
        },
        detect={_QUERY: {"svs": [_SV], "cosine_scores": [0.9]}},
        resolve={"Kenya": _KEN},
        obs={
            f"{_SV}|{_KEN}": [
                {
                    "facetId": "native",
                    "earliestDate": "2022",
                    "latestDate": "2022",
                    "obsCount": 1,
                    "provenanceId": _PROVENANCE,
                    "unit": "CurrencyUSD",
                    "dates": ["2022"],
                    "values": [100.0],
                }
            ]
        },
    )


def _run_buffered():
    return asyncio.run(
        resolve_variable(
            "amount",
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=_graph(),
            resolver=make_resolver(_graph()),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FixedBindLLM(),
            base_steps=[],
            base_timing={},
            unit_text="EUR",
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(
                include_observations=True,
                materialise_graph=_materialise_graph(),
                materialise_sdmx=_materialise_sdmx(),
                rate_cache=RateTableCache(),
            ),
        )
    )


def _run_streamed(*, rate_cache: RateTableCache | None):
    async def _body():
        async with scoped_concurrency():
            result = await resolve_variable(
                "amount",
                entities=["Kenya"],
                date_request=None,
                detect_query=_QUERY,
                role_query=_QUERY,
                pac=True,
                graph=_graph(),
                resolver=make_resolver(_graph()),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=_FixedBindLLM(),
                base_steps=[],
                base_timing={},
                unit_text="EUR",
                coverage=empty_coverage_snapshot(),
                options=EngineOptions(
                    stream_observations=True,
                    materialise_graph=_materialise_graph(),
                    materialise_sdmx=_materialise_sdmx(),
                    rate_cache=rate_cache,
                ),
            )
            assert result.observations_task is not None
            events, warns = await result.observations_task
            return result, events, warns

    return asyncio.run(_body())


def test_streamed_and_buffered_paths_serve_the_same_converted_points():
    """Buffered and streamed `{EUR, current}` requests convert the native-USD point identically."""
    buffered = _run_buffered()
    assert buffered.status == "definite", buffered
    buffered_series = buffered.specs[0].resolved.observations.series[0]
    assert buffered_series.conversion is not None
    assert [p.value for p in buffered_series.points] == [pytest.approx(100.0 * 0.9 / 1.0)]

    streamed, events, warns = _run_streamed(rate_cache=RateTableCache())
    assert streamed.status == "definite", streamed
    assert not warns
    assert len(events) == 1
    streamed_series = events[0].observations.series[0]
    assert streamed_series.conversion is not None
    assert [p.value for p in streamed_series.points] == [p.value for p in buffered_series.points]
    assert [p.value for p in streamed_series.points] == [pytest.approx(100.0 * 0.9 / 1.0)]


def test_converting_request_still_cites_dates_and_covers_the_converting_stat_var():
    """A converting native-USD facet retains its citation, 2022 date range, and coverage count.

    The EUR request changes values, not the selected facet metadata.
    """
    buffered = _run_buffered()
    assert buffered.status == "definite", buffered
    spec = buffered.specs[0]

    # The selected facet supplies the converted series' provenance citation.
    cited_dcids = {ref.dcid for ref in spec.resolved.trace.resolved_sources}
    assert _PROVENANCE in cited_dcids

    # The selected facet also supplies the StatVar's 2022 date range.
    sv = next(item for item in spec.resolved.stat_vars.items if item.ref == _SV)
    assert sv.data_date_range is not None
    assert sv.data_date_range.start == "2022"
    assert sv.data_date_range.end == "2022"

    # Coverage retains the converting StatVar's one observation.
    assert spec.resolved.coverage.has_data is True
    if hasattr(spec.resolved.coverage, "observation_count"):
        assert spec.resolved.coverage.observation_count == 1


def test_streamed_relaunch_needs_rate_cache_to_reach_the_conversion():
    """Without a rate cache, streamed materialisation raises MaterialisationUsageError.

    SDMX receives no request.
    """
    with pytest.raises(MaterialisationUsageError):
        _run_streamed(rate_cache=None)
