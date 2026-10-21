"""Buffered and streamed answers both report projected exchange rates, as a warning and as
the `modelled_values` departure.

The EUR rate fixture uses the IMFProjection method, so both paths must emit `deflator_projected`.
"""

from __future__ import annotations

import asyncio

import httpx

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise.rate_tables import RateTableCache
from search.engine.concurrency import scoped_concurrency
from search.engine.conjoin import assemble_region
from search.engine.enrichment import late_departures, part_sources
from search.engine.quantity_read import QuantityOutput
from search.engine.region.codes import DEFLATOR_PROJECTED
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    _facet_from_fixture,
)
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise.conftest import FakeGraphReader

_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"

_SV = "TESTFIXTURE/StreamedDeflatorProjected/Amount"
_KEN = "country/KEN"
_QUERY = "streamed deflator projected amount"

_PROVENANCE = "dc/base/StreamedDeflatorProjectedSrc"
_RECIPE = "ConversionRecipeDAC"
_EUR = "CurrencyEUR"

_STD_ARCS = {
    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
    "name": {"nodes": [{"value": "Streamed deflator projected amount"}]},
    "populationType": {"nodes": [{"dcid": "TestStreamedDeflatorProjectedActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "testStreamedDeflatorProjectedAmount"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
}

_USD_FIXTURE = ObsFacetFixture(
    facet_id="native", points=(("2022", 100.0),), provenance_id=_PROVENANCE, unit="CurrencyUSD"
)


class _FixedBindLLM:
    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0), None


def _materialise_sdmx() -> MaterialiseSdmxClient:
    main_body = _csv_header(["observationAbout"]) + _csv_row(
        stat_var=_SV, role_dcids=[_KEN], fixture=_USD_FIXTURE, date_str="2022", value=100.0
    )
    bodies = {_SV: main_body}

    def observations(request: httpx.Request) -> httpx.Response:
        sv = request.url.params.get("c[variableMeasured]", "")
        body = bodies.get(sv, _csv_header(["observationAbout"]))
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    # The EUR rate is served as an `IMFProjection` row.
    server = RateTableServer(
        recipe_table_bodies(
            _RECIPE,
            rates={"CurrencyUSD": {2022: 1.0}, _EUR: {2022: 0.9}},
            projected={(_EUR, 2022)},
        ),
        fallback=observations,
    )
    return server.client(base_url=MATERIALISE_BASE_URL)


def _materialise_graph() -> FakeGraphReader:
    return FakeGraphReader(
        arcs={
            _SV: dict(CONVERTIBLE_STAT_VAR_ARCS),
            **conversion_arcs(
                recipe_id=_RECIPE, provenance_id=_PROVENANCE, currencies=["CurrencyUSD", _EUR]
            ),
        },
        facets={(_SV, _KEN): [_facet_from_fixture(_USD_FIXTURE)]},
    )


def _graph() -> FakeGraph:
    return FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _SV: {"label": "Streamed deflator projected amount", "arcs": _STD_ARCS},
            _PROVENANCE: {"label": "Streamed Deflator Projected Source"},
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


def _run_streamed():
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
                base_timing={},
                unit_text="EUR",
                coverage=empty_coverage_snapshot(),
                options=EngineOptions(
                    stream_observations=True,
                    materialise_graph=_materialise_graph(),
                    materialise_sdmx=_materialise_sdmx(),
                    rate_cache=RateTableCache(),
                ),
            )
            assert result.observations_task is not None
            events, warns = await result.observations_task
            return result, events, warns

    return asyncio.run(_body())


def test_streamed_and_buffered_answers_carry_the_same_deflator_projected_warning():
    """A `{EUR, current}` request-form binding whose exchange rate at the
    observation year has no actual row for the target currency, only a
    projected (IMFProjection) one: the buffered path (_attach_observations,
    via region/pipeline.py's eager _resolve_spec_observations call) and the
    streamed path (_observations_events, via result.observations_task) must
    carry the same deflator_projected warning -- both derive it from the
    same _deflator_projected_warning helper in region/observations/
    dispatch.py."""
    buffered = _run_buffered()
    assert buffered.status == "definite", buffered
    buffered_matches = [w for w in buffered.warnings if w.code == DEFLATOR_PROJECTED]
    assert len(buffered_matches) == 1, buffered.warnings

    streamed, events, warns = _run_streamed()
    assert streamed.status == "definite", streamed
    assert len(events) == 1
    streamed_matches = [w for w in warns if w.code == DEFLATOR_PROJECTED]
    assert len(streamed_matches) == 1, warns
    assert streamed_matches[0].message == buffered_matches[0].message
    assert streamed_matches[0].severity == buffered_matches[0].severity


def test_a_streamed_answer_states_projected_values_on_done_as_the_buffered_part_does():
    """Projected rates make some converted points estimates. The buffered part states
    `modelled_values`. The streamed part ships before its read, so its complete departures
    arrive in the `done` event's late departures, equal to the buffered part's."""
    buffered = _run_buffered()
    (part,) = assemble_region(
        buffered,
        extra_warnings=[],
        start_ms=0,
        engine_build="test",
        max_candidates=5,
        include_observations=True,
    ).parts
    buffered_modelled = [d for d in part.departures if d.code == "modelled_values"]
    assert [d.text for d in buffered_modelled] == ["Some of these values are modelled estimates."]

    streamed, events, warns = _run_streamed()
    response = assemble_region(
        streamed,
        extra_warnings=[],
        start_ms=0,
        engine_build="test",
        max_candidates=5,
    )
    (streamed_part,) = response.parts
    assert [d.code for d in streamed_part.departures if d.code == "modelled_values"] == []
    (source,) = part_sources(response, answered=[streamed], streaming=[streamed])
    late = late_departures(source, events=events, warnings=warns)

    assert late is not None
    assert (late.part_index, late.spec_id) == (0, streamed.spec.spec_id)
    assert late.departures == part.departures
