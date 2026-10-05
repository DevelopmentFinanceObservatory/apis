"""Tests the parsed door's HTTP schema, refusals, status parity, and SSE events."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from materialise.errors import UpstreamVersionChangedError
from materialise.rate_tables import RateTableInvalidError
from search.engine.config import SEARCH_MAX_PARTS
from search.engine.unit.base_year import TABLE_PROVENANCES
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import PINNED_DATE, materialise_double_from_obs
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, _load_json, make_resolver

_FERTILITY_SV = "FertilityRate_Person_Female"
_FERTILITY_KEN_KEY = f"{_FERTILITY_SV}|country/KEN"


def _fertility_kenya_graph(
    *, currency_units: dict | None = None, extra_nodes: dict | None = None
) -> FakeGraph:
    """Builds two provenance facets. `currency_units` can override the default roster, and
    `extra_nodes` adds graph nodes."""
    nodes = _load_json("graph_nodes.json")
    obs = _load_json("graph_obs.json")
    currency_kwargs = {} if currency_units is None else {"currency_units": currency_units}
    return FakeGraph(
        nodes={
            **nodes,
            "dc/base/FertilitySourceOne": {"label": "Fertility Source One"},
            "dc/base/FertilitySourceTwo": {"label": "Fertility Source Two"},
            **(extra_nodes or {}),
        },
        **currency_kwargs,
        obs={
            **obs,
            _FERTILITY_KEN_KEY: [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "dates": ["2020"],
                    "values": [4.5],
                    "provenanceId": "dc/base/FertilitySourceOne",
                    "importName": "FertilitySourceOne",
                },
                {
                    "earliestDate": "2019",
                    "latestDate": "2019",
                    "obsCount": 1,
                    "dates": ["2019"],
                    "values": [4.6],
                    "provenanceId": "dc/base/FertilitySourceTwo",
                    "importName": "FertilitySourceTwo",
                },
            ],
        },
    )


def make_client() -> TestClient:
    graph = _fertility_kenya_graph()
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
    )
    return TestClient(app, raise_server_exceptions=False)


def _fertility_kenya_payload(**overrides) -> dict:
    payload = {
        "input": {
            "kind": "parsed",
            "parts": [
                {
                    "variable_text": "fertility rate",
                    "entities": [{"text": "Kenya"}],
                }
            ],
        }
    }
    payload["input"].update(overrides)
    return payload


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into [(event_type, data_dict), ...], skipping `: ping` comments."""
    frames: list[tuple[str, dict]] = []
    event_type: str | None = None
    for line in body.splitlines():
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:") and event_type is not None:
            frames.append((event_type, json.loads(line[len("data:") :].strip())))
            event_type = None
    return frames


class TestBufferedAndSSETransports:
    def test_buffered_json_resolves(self):
        client = make_client()
        resp = client.post("/api/search/resolve", json=_fertility_kenya_payload())
        assert resp.status_code == 200
        assert resp.json()["status"] in ("definite", "candidates", "no_data")

    def test_sse_resolves(self):
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json=_fertility_kenya_payload(),
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        frames = _parse_sse(resp.text)
        response_frames = [data for ev, data in frames if ev == "response"]
        assert len(response_frames) == 1
        assert response_frames[0]["response"]["status"] in ("definite", "candidates", "no_data")

    def test_sse_event_vocabulary_matches_raw_text(self):
        # Parsed and raw-text requests share the default SSE events.
        # Coverage events require opt-in.
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json=_fertility_kenya_payload(),
            headers={"Accept": "text/event-stream"},
        )
        frames = _parse_sse(resp.text)
        event_types = [ev for ev, _ in frames]
        for required in ("interpretation", "shape", "sentence", "response", "done"):
            assert required in event_types, event_types
        assert event_types.index("interpretation") < event_types.index("response")
        assert "facet_options" not in event_types
        terminals = [ev for ev in event_types if ev in ("done", "error")]
        assert terminals == ["done"]

    def test_sse_trailing_coverage_events_fire_under_include_option_coverage(self):
        # Coverage events trail the response. This standard family has no entity options.
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                **_fertility_kenya_payload(),
                "response_options": {"include_option_coverage": True},
            },
            headers={"Accept": "text/event-stream"},
        )
        frames = _parse_sse(resp.text)
        event_types = [ev for ev, _ in frames]
        assert "slot_options" in event_types
        assert "entity_options" not in event_types
        assert "facet_options" in event_types
        facet_events = [data for ev, data in frames if ev == "facet_options"]
        assert any(fe["axis"] == "provenance" for fe in facet_events)
        for fe in facet_events:
            assert fe["options"], fe
            # Facet options have no confirmation step, so SSE coverage remains null.
            assert all(opt["coverage"] is None for opt in fe["options"])

    def test_facet_options_buffered_and_sse_carry_the_same_content(self):
        # Transport split must match slot_options exactly: same content,
        # inline on the buffered path, trailing on SSE.
        client = make_client()
        payload = {
            **_fertility_kenya_payload(),
            "response_options": {"include_option_coverage": True},
        }

        buffered = client.post("/api/search/resolve", json=payload)
        assert buffered.status_code == 200
        buffered_facet_options = buffered.json()["parts"][0]["spec"]["resolved"]["facet_options"]
        assert buffered_facet_options

        sse = client.post(
            "/api/search/resolve", json=payload, headers={"Accept": "text/event-stream"}
        )
        sse_facet_events = [data for ev, data in _parse_sse(sse.text) if ev == "facet_options"]
        assert sse_facet_events

        # Ignore transport-only fields: SSE `spec_id` and omitted-versus-null coverage.
        def _shape(axis: str, options: list[dict], truncated: bool) -> dict:
            return {
                "axis": axis,
                "values": [opt["value"] for opt in options],
                "truncated": truncated,
            }

        buffered_shapes = sorted(
            (_shape(fo["axis"], fo["options"], fo["truncated"]) for fo in buffered_facet_options),
            key=lambda d: d["axis"],
        )
        sse_shapes = sorted(
            (_shape(ev["axis"], ev["options"], ev["truncated"]) for ev in sse_facet_events),
            key=lambda d: d["axis"],
        )
        assert buffered_shapes == sse_shapes


class TestCapRefusalsEndToEnd:
    @pytest.mark.parametrize(
        "parts",
        [
            pytest.param(
                [{"variable_text": ""}],
                id="empty-variable-text",
            ),
            pytest.param(
                [{"variable_text": "fertility rate"}] * (SEARCH_MAX_PARTS + 1),
                id="oversized-parts",
            ),
        ],
    )
    def test_invalid_parsed_request_refuses_request_invalid(self, parts: list[dict]) -> None:
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "parsed", "parts": parts}},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "request_invalid"


class TestQueryEchoShape:
    def test_echo_reflects_parsed_entry_path(self):
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json=_fertility_kenya_payload(raw_query="fertility rate in Kenya"),
        )
        assert resp.status_code == 200
        body = resp.json()
        echo = body["query_echo"]
        assert echo["entry_path"] == "parsed"
        assert echo["extract_skipped"] is True
        assert [p["variable_text"] for p in body["parts"]] == ["fertility rate"]
        assert echo["raw_query"] == "fertility rate in Kenya"


class TestRefusalStatusParityWithRawText:
    def test_same_utterance_resolves_the_same_status(self):
        client = make_client()
        with patch("search.engine.extract.date") as mock_date:
            mock_date.today.return_value = PINNED_DATE
            mock_date.side_effect = None
            raw_resp = client.post(
                "/api/search/resolve",
                json={
                    "input": {
                        "kind": "raw_text",
                        "query": "fertility rate in Kenya",
                    }
                },
            )
        parsed_resp = client.post("/api/search/resolve", json=_fertility_kenya_payload())
        assert raw_resp.status_code == parsed_resp.status_code == 200
        assert raw_resp.json()["status"] == parsed_resp.json()["status"]


def _post_unit_text(graph: FakeGraph, unit_text: str):
    client = TestClient(
        search_app(
            graph=graph, resolver=make_resolver(graph), llm=FakeLLM(), sdmx=FakeSdmxClient()
        ),
        raise_server_exceptions=False,
    )
    return client.post(
        "/api/search/resolve",
        json=_fertility_kenya_payload(
            parts=[
                {
                    "variable_text": "fertility rate",
                    "entities": [{"text": "Kenya"}],
                    "unit_text": unit_text,
                }
            ]
        ),
    )


def _table_provenance(base_years: list[str]) -> dict:
    return {
        "label": "Conversion tables",
        "arcs": {
            "lastDataRefreshDate": {"nodes": [{"value": "2026-10-01"}]},
            "deflatorBaseYear": {"nodes": [{"value": year} for year in base_years]},
        },
    }


class TestUnitNotResolvedCarriesUnitAlternatives:
    def test_an_ambiguous_currency_name_offers_its_iso_codes_as_resubmittable_values(self):
        resp = _post_unit_text(_fertility_kenya_graph(), "francs")
        assert resp.status_code == 200
        no_data = resp.json()["parts"][0]["no_data"]
        assert no_data["reason"] == "unit_not_resolved"
        assert no_data["unit_alternatives"] == [
            {"currency": "CHF", "label": "Swiss franc"},
            {"currency": "XAF", "label": "Central African CFA franc"},
            {"currency": "XOF", "label": "West African CFA franc"},
        ]
        assert no_data["unit_alternatives_truncated"] is False

    def test_an_unparseable_phrase_carries_no_unit_alternatives(self):
        resp = _post_unit_text(_fertility_kenya_graph(), "zorkmids")
        assert resp.status_code == 200
        no_data = resp.json()["parts"][0]["no_data"]
        assert no_data["reason"] == "unit_not_resolved"
        assert "unit_alternatives" not in no_data
        assert "unit_alternatives_truncated" not in no_data


class TestConstantPricesCompleteFromTheConversionTables:
    def test_disagreeing_table_base_years_refuse_capability_unavailable(self):
        dac, imf = TABLE_PROVENANCES
        graph = _fertility_kenya_graph(
            extra_nodes={dac: _table_provenance(["2024"]), imf: _table_provenance(["2023"])}
        )
        resp = _post_unit_text(graph, "constant prices")
        assert resp.status_code == 501
        body = resp.json()
        assert body["code"] == "capability_unavailable"
        assert "deflatorBaseYear" in body["detail"]

    def test_an_absent_table_provenance_refuses_capability_unavailable(self):
        resp = _post_unit_text(_fertility_kenya_graph(), "constant prices")
        assert resp.status_code == 501
        assert resp.json()["code"] == "capability_unavailable"


def _post_two_parts(graph: FakeGraph, unit_text: str):
    """Post two parsed parts, so the request runs the degrading multi-part fan-out."""
    part = {"variable_text": "fertility rate", "entities": [{"text": "Kenya"}]}
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {_FERTILITY_KEN_KEY: graph._obs[_FERTILITY_KEN_KEY]}
    )
    client = TestClient(
        search_app(
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        ),
        raise_server_exceptions=False,
    )
    return client.post(
        "/api/search/resolve",
        json=_fertility_kenya_payload(parts=[part, {**part, "unit_text": unit_text}]),
    )


class TestMultiPartRateTableFailuresRefuseTheRequest:
    """A conversion-table failure in one leg refuses the request. No part degrades to a
    placeholder no_data."""

    def test_an_absent_table_provenance_refuses_capability_unavailable(self):
        resp = _post_two_parts(_fertility_kenya_graph(), "constant prices")
        assert resp.status_code == 501
        assert resp.json()["code"] == "capability_unavailable"

    @pytest.mark.parametrize(
        ("exc", "status", "code"),
        [
            pytest.param(
                RateTableInvalidError("table has no lastDataRefreshDate"),
                501,
                "capability_unavailable",
                id="capability_unavailable",
            ),
            pytest.param(
                UpstreamVersionChangedError("table moved during its load"),
                503,
                "upstream_version_changed",
                id="upstream_version_changed",
            ),
        ],
    )
    def test_a_conversion_decision_failure_serves_its_problem(self, exc, status, code):
        def _raise(*_args, **_kwargs):
            raise exc

        with patch("search.engine.region.phase_answer.decide_part_unit", _raise):
            resp = _post_two_parts(_fertility_kenya_graph(), "US dollars")
        assert resp.status_code == status
        assert resp.json()["code"] == code


class TestCapabilitiesPublishLiveDoorAndCaps:
    def test_capabilities_publish_parsed_live_and_the_new_caps(self):
        client = make_client()
        body = client.get("/api/search/capabilities").json()
        doors = {d["name"]: d["live"] for d in body["doors"]}
        assert doors["parsed"] is True
        assert "max_parts" in body["caps"]
        assert "max_variable_text_chars" in body["caps"]


class TestSchemaReflectsPartsShape:
    def test_schema_request_defines_parsed_input_with_parts(self):
        client = make_client()
        body = client.get("/api/search/schema").json()
        defs = body["request"]["$defs"]
        assert "ParsedInput" in defs
        parsed_input = defs["ParsedInput"]
        assert "parts" in parsed_input["properties"]
        part_ref_schema = defs["ParsedPart"]
        assert set(part_ref_schema["properties"]) == {
            "variable_text",
            "entities",
            "time_text",
            "unit_text",
            "source_text",
            "operations",
        }
        assert defs["ParsedInput"]["additionalProperties"] is False
        assert defs["ParsedPart"]["additionalProperties"] is False
