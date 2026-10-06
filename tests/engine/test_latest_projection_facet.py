"""Check that a `latest` request serves the recorded facet's latest point when a projection
facet runs past it, and that the answer names the served point before any observation is
read, so a streamed `response` event and a buffered response agree.

Kenya's fertility rate holds recorded values to 2024 and an `Estimated` facet to 2032, both
from one provenance. Ethiopia's recorded values end in 2019. The SDMX double honours
`c[facetId]` and `c[TIME_PERIOD]=LATEST` as the service does, returning one latest row per
dispatched facet.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from search.engine.core import resolve_async
from search.models import (
    CoverageExact,
    DefinitePart,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    ResolveOptions,
    ResolveRequest,
    Spec,
    TimeWindow,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    _SDMX_CSV_CONTENT_TYPE,
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    _facet_from_fixture,
    parse_sse,
)
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, _load_json, make_resolver
from tests.materialise.conftest import FakeGraphReader

_SV = "FertilityRate_Person_Female"
_KEN = "country/KEN"
_ETH = "country/ETH"
_PROVENANCE = "dc/base/WorldBankIDS"


def _yearly(facet_id: str, years: range, *, method: str | None = None) -> ObsFacetFixture:
    return ObsFacetFixture(
        facet_id=facet_id,
        points=tuple((str(y), float(y)) for y in years),
        provenance_id=_PROVENANCE,
        measurement_method=method,
    )


_FIXTURES: dict[str, tuple[ObsFacetFixture, ...]] = {
    _KEN: (
        _yearly("recorded", range(1970, 2025)),
        _yearly("estimated", range(2025, 2033), method="Estimated"),
    ),
    _ETH: (_yearly("recorded", range(1970, 2020)),),
}


def _graph_facet(fixture: ObsFacetFixture) -> dict:
    dates = [d for d, _ in fixture.points]
    facet: dict = {
        "earliestDate": dates[0],
        "latestDate": dates[-1],
        "obsCount": len(dates),
        "provenanceId": fixture.provenance_id,
        "facetId": fixture.facet_id,
        "dates": dates,
        "values": [v for _, v in fixture.points],
    }
    if fixture.measurement_method is not None:
        facet["measurementMethod"] = fixture.measurement_method
    return facet


def _sdmx() -> MaterialiseSdmxClient:
    """Serve each dispatched facet's rows, or its latest row under `LATEST`."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        body = _csv_header(["observationAbout"])
        entity = params.get("c[observationAbout]", "")
        if params.get("c[variableMeasured]") == _SV and entity in _FIXTURES:
            facet_ids = params.get("c[facetId]")
            latest = params.get("c[TIME_PERIOD]") == "LATEST"
            for fixture in _FIXTURES[entity]:
                if facet_ids is not None and fixture.facet_id not in facet_ids.split(","):
                    continue
                for date_str, value in fixture.points[-1:] if latest else fixture.points:
                    body += _csv_row(
                        stat_var=_SV,
                        role_dcids=[entity],
                        fixture=fixture,
                        date_str=date_str,
                        value=value,
                    )
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    return MaterialiseSdmxClient(
        base_url=MATERIALISE_BASE_URL, transport=httpx.MockTransport(handler)
    )


def _clients() -> tuple[FakeGraph, FakeGraphReader, MaterialiseSdmxClient]:
    graph = FakeGraph(
        obs={
            **_load_json("graph_obs.json"),
            **{
                f"{_SV}|{entity}": [_graph_facet(f) for f in fixtures]
                for entity, fixtures in _FIXTURES.items()
            },
        }
    )
    materialise_graph = FakeGraphReader(
        facets={
            (_SV, entity): [_facet_from_fixture(f) for f in fixtures]
            for entity, fixtures in _FIXTURES.items()
        }
    )
    return graph, materialise_graph, _sdmx()


def _latest_request(*places: str, time_text: str | None = "latest") -> ResolveRequest:
    return ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(
                    variable_text="fertility rate",
                    entities=[ParsedEntity(text=place) for place in places],
                    time_text=time_text,
                )
            ]
        ),
        response_options=ResolveOptions(include_observations=True),
    )


def _resolve_buffered(*places: str) -> Spec:
    return _resolve_part(*places).spec


def _resolve_part(*places: str) -> DefinitePart:
    graph, materialise_graph, materialise_sdmx = _clients()
    result = asyncio.run(
        resolve_async(
            _latest_request(*places),
            graph=graph,
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        )
    )
    part = result.parts[0]
    assert isinstance(part, DefinitePart), part
    return part


def test_buffered_latest_serves_the_recorded_point_and_names_its_year():
    spec = _resolve_buffered("Kenya")

    assert spec.definition.identity.requested_window.kind == "latest"
    observations = spec.resolved.observations
    assert observations is not None
    (series,) = observations.series
    assert [(p.date, p.value) for p in series.points] == [("2024", 2024.0)]
    assert "measurementMethod" not in series.facet

    served_year = TimeWindow(start_year=2024, end_year=2024)
    coverage = spec.resolved.coverage
    assert isinstance(coverage, CoverageExact)
    assert coverage.window == served_year
    assert coverage.observation_count == 1
    assert coverage.evidence == "observed"
    when_slot = next(slot for slot in spec.resolved.slots if slot.key.axis == "when")
    assert when_slot.binding.value.time_window == served_year


def test_sse_response_event_names_the_recorded_year_before_the_points_arrive():
    """The `response` event ships before the observation read, and already names 2024; the
    trailing `observations` event then delivers the 2024 point."""
    graph, materialise_graph, materialise_sdmx = _clients()
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/search/resolve",
        json=_latest_request("Kenya").model_dump(mode="json", exclude_none=True),
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    frames = parse_sse(response.text)
    events = [event for event, _ in frames]

    part = frames[events.index("response")][1]["response"]["parts"][0]
    resolved = part["spec"]["resolved"]
    served_year = {"start_year": 2024, "end_year": 2024}
    assert resolved["coverage"]["window"] == served_year
    when_slot = next(slot for slot in resolved["slots"] if slot["key"]["axis"] == "when")
    assert when_slot["binding"]["value"]["time_window"] == served_year
    assert "in 2024" in resolved["rendered_sentence"]
    assert "2032" not in resolved["rendered_sentence"]

    trailing = frames[events.index("observations")][1]["observations"]
    assert [[p["date"] for p in s["points"]] for s in trailing["series"]] == [["2024"]]


def test_latest_across_places_serves_the_variables_latest_period():
    """Ethiopia's latest recorded point is 2019 and Kenya's 2024, so the variable's latest
    period is 2024 and Kenya alone serves it. Coverage names 2024, and the sentence names it
    and Ethiopia's departure says it holds no data in it."""
    part = _resolve_part("Kenya", "Ethiopia")
    spec = part.spec

    observations = spec.resolved.observations
    assert observations is not None
    served = {
        next(e.ref for e in s.entities): [p.date for p in s.points] for s in observations.series
    }
    assert served == {_KEN: ["2024"]}
    assert spec.resolved.coverage.window == TimeWindow(start_year=2024, end_year=2024)
    sentence = spec.resolved.sentence_plan
    assert sentence is not None
    assert "2024. Latest year with data" in "".join(token.text for token in sentence)
    assert [(d.code, d.text) for d in part.departures] == [
        ("entity_unavailable", "You asked for Ethiopia. The data holds no values for it in 2024.")
    ]


# The door parses no open-start phrasing; `test_read_plan_invariants` covers one.
@pytest.mark.parametrize(
    "time_text", ["latest", "2015 to 2020", "since 2018", "2026 to 2030", None]
)
def test_a_streamed_answer_counts_the_points_its_trailing_event_serves(time_text):
    """Coverage ships in the `response` event before the preview runs; the trailing
    `observations` event then serves exactly the points it counted, over the span it
    states. A dateless answer states no span."""
    graph, materialise_graph, materialise_sdmx = _clients()
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )
    request = _latest_request("Kenya", "Ethiopia", time_text=time_text)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/search/resolve",
        json=request.model_dump(mode="json", exclude_none=True),
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    frames = parse_sse(response.text)
    events = [event for event, _ in frames]
    coverage = frames[events.index("response")][1]["response"]["parts"][0]["spec"]["resolved"][
        "coverage"
    ]
    trailing = frames[events.index("observations")][1]["observations"]
    years = [int(p["date"][:4]) for s in trailing["series"] for p in s["points"]]

    assert coverage["observation_count"] == len(years)
    assert years, trailing
    if time_text is None:
        assert "window" not in coverage
    else:
        assert coverage["window"]["start_year"] <= min(years)
        assert coverage["window"]["end_year"] == max(years)
