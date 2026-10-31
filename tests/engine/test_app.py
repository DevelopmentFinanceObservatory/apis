"""FastAPI app tests using TestClient."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from logging.handlers import BufferingHandler

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from dfo_platform.build import build_commit
from dfo_platform.request_deadline import bounded_deadline
from search.engine.config import settings
from search.engine.core import RequestBudget, request_budget
from search.engine.extract import Extraction, ExtractionResult
from search.engine.region import RegionResult
from search.models import Resolution
from tests.engine._base_identity import GDP_SHAPE, make_identity, subject_entity
from tests.engine._builders import value_slot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import member_walk_reader, resolve_log_lines
from tests.engine._refusal_detail import assert_names_plainly
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver


def make_client(*, graph=None, llm=None, sdmx=None) -> TestClient:
    """Build an offline client. Callers can replace any fake dependency. A resubmit walks
    the mint's member enumeration over the graph's nodes."""
    _graph = graph or FakeGraph()
    app = search_app(
        graph=_graph,
        resolver=make_resolver(_graph),
        llm=llm or FakeLLM(),
        sdmx=sdmx or FakeSdmxClient(),
        materialise_graph=member_walk_reader(_graph) if isinstance(_graph, FakeGraph) else None,
    )
    return TestClient(app, raise_server_exceptions=False)


def _assert_error(
    resp,
    *,
    status: int,
    code: str,
    detail: str | None = None,
    retryable: bool | None = None,
) -> dict:
    """Check problem-details fields and return the parsed body."""
    assert resp.status_code == status
    body = resp.json()
    assert body["code"] == code
    if detail is not None:
        assert body["detail"] == detail
    if retryable is not None:
        assert body["retryable"] is retryable
    return body


class TestHealthz:
    def test_healthz_ok(self):
        client = make_client()
        resp = client.get("/api/search/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["build"] == build_commit()


class TestResolveEndpoint:
    def test_200_on_valid_request(self):
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "raw_text",
                    "query": "health ODA grants from USA to Ethiopia",
                }
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("definite", "candidates", "no_data")

    def test_400_on_malformed_body(self):
        # Unknown top-level fields return request_invalid.
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={"not_valid": "body"},
        )
        _assert_error(resp, status=400, code="request_invalid")

    def test_200_on_kind_parsed(self):
        # Parsed input with parts uses the shared resolve pipeline.
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "parsed",
                    "parts": [
                        {
                            "variable_text": "health ODA grants",
                            "entities": [
                                {"text": "USA", "role_text": "from"},
                                {"text": "Ethiopia", "role_text": "to"},
                            ],
                        }
                    ],
                }
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("definite", "candidates", "no_data")

    def test_400_on_old_flat_parsed_shape(self):
        # Flat parsed input fails schema validation.
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "parsed",
                    "variable_text": ["health ODA"],
                }
            },
        )
        _assert_error(resp, status=400, code="request_invalid")

    def test_removed_singular_origin_field_is_rejected_at_its_wire_pointer(self):
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": _base_identity_with_facets({}),
                            "origin_dcid": "country/FRA",
                        }
                    ],
                }
            },
        )
        body = _assert_error(resp, status=400, code="request_invalid")
        assert body["pointer"] == "/input/entries/0/origin_dcid"

    def test_422_when_query_exceeds_max_chars(self):
        client = make_client()
        long_query = "x" * (settings().max_query_chars + 1)
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": long_query}},
        )
        assert resp.status_code == 422

    def test_503_on_graph_infra_error(self):
        # A graph that raises GraphInfraError at detect_svs routes to 503
        from search.engine.errors import GraphInfraError

        class ErrorGraph(FakeGraph):
            # Keep FakeGraph's base state for resolver batch confirmation; the intended
            # detect_svs failure below still controls this test's error path.
            def __init__(self):
                super().__init__()

            def node_label(self, dcid):
                raise GraphInfraError("simulated error")

            def node_arcs(self, dcid):
                raise GraphInfraError("simulated error")

            def node_type(self, dcid):
                raise GraphInfraError("simulated error")

            def resolve_entity(self, name):
                raise GraphInfraError("simulated error")

            def detect_svs(self, query):
                raise GraphInfraError("simulated error")

            def child_dcids(self, parent_dcid):
                raise GraphInfraError("simulated error")

            def observation_facets(self, *, stat_var, entity, needs_dates=False):
                raise GraphInfraError("simulated error")

        client = make_client(graph=ErrorGraph())

        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "raw_text",
                    "query": "health ODA grants from USA to Ethiopia",
                }
            },
        )
        _assert_error(
            resp,
            status=503,
            code="service_unavailable",
            detail="Service temporarily unavailable",
            retryable=True,
        )

    def test_503_on_llm_infra_error(self):
        # An LLM that raises LLMInfraError at extract routes to 503
        from search.engine.errors import LLMInfraError

        class _ErrorLLM:
            async def generate_structured(self, *, prompt, system, schema):
                raise LLMInfraError("simulated LLM error")

        client = make_client(llm=_ErrorLLM())

        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "raw_text",
                    "query": "health ODA grants from USA to Ethiopia",
                }
            },
        )
        _assert_error(
            resp,
            status=503,
            code="service_unavailable",
            detail="Service temporarily unavailable",
            retryable=True,
        )

    def test_500_on_generic_error(self):
        # A non-infra exception falls through to the generic handler → 500
        class _BoomGraph(FakeGraph):
            # Skip fixture loading so the overrides handle each graph call.
            def __init__(self):
                pass

            def node_label(self, dcid):
                raise RuntimeError("boom")

            def node_arcs(self, dcid):
                raise RuntimeError("boom")

            def node_type(self, dcid):
                raise RuntimeError("boom")

            def resolve_entity(self, name):
                raise RuntimeError("boom")

            def detect_svs(self, query):
                raise RuntimeError("boom")

            def child_dcids(self, parent_dcid):
                raise RuntimeError("boom")

            def observation_facets(self, *, stat_var, entity, needs_dates=False):
                raise RuntimeError("boom")

        client = make_client(graph=_BoomGraph())

        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "raw_text",
                    "query": "health ODA grants from USA to Ethiopia",
                }
            },
        )
        assert resp.headers["content-type"].startswith("application/problem+json")
        body = _assert_error(resp, status=500, code="internal_error", retryable=False)
        assert isinstance(body["trace_id"], str) and body["trace_id"]

    def test_whitespace_query_returns_no_data(self):
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "   "}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "no_data"
        assert body["parts"][0]["no_data"]["reason"] == "variable_not_resolved"

    def test_504_on_timeout(self, monkeypatch):
        # asyncio.TimeoutError -> 504
        async def _slow_resolve(*args, **kwargs):
            raise TimeoutError()

        monkeypatch.setattr("search.engine.app.resolve_async", _slow_resolve)

        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "health ODA"}},
        )
        _assert_error(resp, status=504, code="timeout", detail="Request timed out", retryable=True)

    def test_502_on_upstream_4xx(self):
        # GraphInfraError with a 4xx upstream_status maps to 502 Bad Gateway
        from search.engine.errors import GraphInfraError

        class _UpstreamErrorGraph(FakeGraph):
            # Keep FakeGraph's base state for resolver batch confirmation; the intended
            # detect_svs failure below still controls this test's error path.
            def __init__(self):
                super().__init__()

            def node_label(self, dcid):
                raise GraphInfraError("not found", upstream_status=404)

            def node_arcs(self, dcid):
                raise GraphInfraError("not found", upstream_status=404)

            def node_type(self, dcid):
                raise GraphInfraError("not found", upstream_status=404)

            def resolve_entity(self, name):
                raise GraphInfraError("not found", upstream_status=404)

            def detect_svs(self, query):
                raise GraphInfraError("not found", upstream_status=404)

            def child_dcids(self, parent_dcid):
                raise GraphInfraError("not found", upstream_status=404)

            def observation_facets(self, *, stat_var, entity, needs_dates=False):
                raise GraphInfraError("not found", upstream_status=404)

        client = make_client(graph=_UpstreamErrorGraph())

        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "raw_text",
                    "query": "health ODA grants from USA to Ethiopia",
                }
            },
        )
        _assert_error(
            resp,
            status=502,
            code="upstream_error",
            detail="Bad gateway: upstream returned an error",
            retryable=True,
        )

    def test_503_on_graph_5xx_upstream(self):
        # GraphInfraError with a 5xx upstream_status maps to 503.
        from search.engine.errors import GraphInfraError

        class _ServerErrorGraph(FakeGraph):
            # Keep FakeGraph's base state for resolver batch confirmation; the intended
            # detect_svs failure below still controls this test's error path.
            def __init__(self):
                super().__init__()

            def node_label(self, dcid):
                raise GraphInfraError("server error", upstream_status=500)

            def node_arcs(self, dcid):
                raise GraphInfraError("server error", upstream_status=500)

            def node_type(self, dcid):
                raise GraphInfraError("server error", upstream_status=500)

            def resolve_entity(self, name):
                raise GraphInfraError("server error", upstream_status=500)

            def detect_svs(self, query):
                raise GraphInfraError("server error", upstream_status=500)

            def child_dcids(self, parent_dcid):
                raise GraphInfraError("server error", upstream_status=500)

            def observation_facets(self, *, stat_var, entity, needs_dates=False):
                raise GraphInfraError("server error", upstream_status=500)

        client = make_client(graph=_ServerErrorGraph())

        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "raw_text",
                    "query": "health ODA grants from USA to Ethiopia",
                }
            },
        )
        assert resp.status_code == 503


class TestResolveOpenAPIResponseModel:
    """The resolve route's wire body must match its documented ResolveResponse schema."""

    def test_definite_response_validates_against_resolve_response(self):
        # "fertility rate in Kenya" is the canned query TestSSETransport's own
        # _resolve_sse helper relies on for a definite, data-bearing result.
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "raw_text",
                    "query": "fertility rate in Kenya",
                }
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["kind"] == "resolution"
        assert body["status"] == "definite"
        parsed = Resolution.model_validate(body)
        assert len(parsed.parts) == 1
        assert parsed.parts[0].status == "definite"

    def test_definite_response_wire_bytes_match_a_resolution_rebuild(self):
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "raw_text",
                    "query": "fertility rate in Kenya",
                }
            },
            headers={"accept-encoding": "identity"},
        )
        assert resp.status_code == 200
        rebuilt = Resolution.model_validate(resp.json()).model_dump(mode="json", exclude_none=True)
        assert resp.content == JSONResponse(rebuilt).body

    def test_resolve_operation_declares_json_and_sse_200_schemas(self):
        client = make_client()
        spec = client.get("/openapi.json").json()
        op = spec["paths"]["/api/search/resolve"]["post"]
        content = op["responses"]["200"]["content"]
        union = content["application/json"]["schema"]
        assert union["discriminator"] == {
            "propertyName": "kind",
            "mapping": {
                "resolution": "#/components/schemas/Resolution",
                "catalogue": "#/components/schemas/CatalogueAnswer",
                "declined": "#/components/schemas/DeclinedAnswer",
            },
        }
        assert union["oneOf"] == [
            {"$ref": "#/components/schemas/Resolution"},
            {"$ref": "#/components/schemas/CatalogueAnswer"},
            {"$ref": "#/components/schemas/DeclinedAnswer"},
        ]
        sse_schema = content["text/event-stream"]["schema"]
        assert sse_schema["type"] == "string"
        assert sse_schema["format"] == "event-stream"
        assert "response" in sse_schema["description"].lower()

        sse_description = op["responses"]["200"]["description"].lower()
        for marker in ("response", "done", "error"):
            assert marker in sse_description
        resolution_schema = spec["components"]["schemas"]["Resolution"]
        assert "status" in resolution_schema["required"]


class TestSchema:
    def test_schema_version(self):
        client = make_client()
        resp = client.get("/api/search/schema")
        body = resp.json()
        assert body["schema_version"] == 1

    def test_schema_has_request_and_response(self):
        client = make_client()
        resp = client.get("/api/search/schema")
        body = resp.json()
        assert isinstance(body["request"], dict)
        assert isinstance(body["response"], dict)

    def test_request_schema_version_is_const_integer_one(self):
        client = make_client()
        body = client.get("/api/search/schema").json()
        schema_version = body["request"]["properties"]["schema_version"]
        assert schema_version["type"] == "integer"
        assert schema_version["const"] == 1

    def test_request_properties_use_response_options_not_options(self):
        client = make_client()
        body = client.get("/api/search/schema").json()
        properties = body["request"]["properties"]
        assert "response_options" in properties
        assert "options" not in properties

    def test_response_options_schema_has_exactly_the_contract_fields(self):
        client = make_client()
        body = client.get("/api/search/schema").json()
        resolve_options = body["request"]["$defs"]["ResolveOptions"]
        assert set(resolve_options["properties"]) == {
            "include_observations",
            "include_option_coverage",
            "include_skills",
        }

    def test_include_observations_schema_describes_the_opt_in_default(self):
        client = make_client()
        body = client.get("/api/search/schema").json()
        description = body["request"]["$defs"]["ResolveOptions"]["properties"][
            "include_observations"
        ]["description"]
        assert "default is false" in description
        assert "multi-candidate" in description

    def test_role_key_descriptions_use_only_clean_contract_vocabulary(self):
        client = make_client()
        schema = client.get("/api/search/schema").json()
        response_defs = schema["response"]["$defs"]
        descriptions = [
            response_defs["ResolvedScope"]["description"],
            response_defs["ResolvedSpec"]["properties"]["scope"]["description"],
            client.get("/openapi.json").json()["paths"]["/api/search/options"]["post"][
                "description"
            ],
        ]

        for description in descriptions:
            assert "origin" in description
            assert "destination" in description
            assert "provider" not in description
            assert "recipient" not in description

    def test_truncated_flag_present_on_option_and_observation_schemas(self):
        client = make_client()
        body = client.get("/api/search/schema").json()
        defs = body["response"]["$defs"]
        for def_name in ("SlotOptions", "EntityOptions", "SpecObservations"):
            assert "truncated" in defs[def_name]["properties"], def_name

    def test_schema_json_has_no_bare_options_include_leak(self):
        # Reject bare options wording while allowing response_options.include_*.
        client = make_client()
        raw = client.get("/api/search/schema").text
        assert not re.search(r"(?<!response_)options\.include", raw)

    def test_schema_links_to_capabilities(self):
        # The schema link must resolve to the capabilities endpoint.
        client = make_client()
        resp = client.get("/api/search/schema")
        body = resp.json()
        assert body["capabilities"] == "/api/search/capabilities"
        assert client.get(body["capabilities"]).status_code == 200


class TestSchemaAndCapabilitiesOpenAPIResponseModels:
    """Both pre-built response routes declare a non-empty OpenAPI 200 schema."""

    def test_schema_operation_declares_a_non_empty_200_schema(self):
        client = make_client()
        spec = client.get("/openapi.json").json()
        op = spec["paths"]["/api/search/schema"]["get"]
        body_schema = op["responses"]["200"]["content"]["application/json"]["schema"]
        assert body_schema["type"] == "object"
        assert set(body_schema["properties"]) == {
            "schema_version",
            "request",
            "response",
            "capabilities",
        }

    def test_capabilities_operation_declares_a_non_empty_200_schema(self):
        client = make_client()
        spec = client.get("/openapi.json").json()
        op = spec["paths"]["/api/search/capabilities"]["get"]
        body_schema = op["responses"]["200"]["content"]["application/json"]["schema"]
        assert body_schema == {"$ref": "#/components/schemas/Capabilities"}


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse SSE frames into event/data pairs and skip `: ping` comments."""
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


class TestSSETransport:
    def _resolve_sse(self, client: TestClient) -> str:
        """Request text/event-stream and return the raw body."""
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "raw_text",
                    "query": "fertility rate in Kenya",
                }
            },
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        return resp.text

    def test_sse_returns_event_stream_content_type(self):
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "raw_text",
                    "query": "health ODA grants from USA to Ethiopia",
                }
            },
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_sse_ends_with_exactly_one_terminal_event(self):
        client = make_client()
        body = self._resolve_sse(client)
        frames = _parse_sse(body)
        terminals = [ev for ev, _ in frames if ev in ("done", "error")]
        assert terminals == ["done"]

    def test_sse_emits_interpretation_before_response(self):
        client = make_client()
        body = self._resolve_sse(client)
        frames = _parse_sse(body)
        event_types = [ev for ev, _ in frames]
        assert "interpretation" in event_types
        assert event_types.index("interpretation") < event_types.index("response")

    def test_sse_response_event_carries_full_resolve_response(self):
        # `done` is a terminal marker. The full response travels in `response`.
        client = make_client()
        body = self._resolve_sse(client)
        frames = _parse_sse(body)
        response_frames = [data for ev, data in frames if ev == "response"]
        assert len(response_frames) == 1
        parsed = Resolution.model_validate(response_frames[0]["response"])
        assert parsed.status == "definite"
        interpretation = parsed.parts[0].spec
        assert interpretation.resolved.coverage.has_data is True
        done_frames = [data for ev, data in frames if ev == "done"]
        assert len(done_frames) == 1
        assert "status" not in done_frames[0]

    def test_sse_response_matches_json_path(self):
        # Both paths must return the same ResolveResponse body.
        client = make_client()
        payload = {"input": {"kind": "raw_text", "query": "health ODA grants from USA to Ethiopia"}}

        json_resp = client.post("/api/search/resolve", json=payload)

        sse_resp = client.post(
            "/api/search/resolve",
            json=payload,
            headers={"Accept": "text/event-stream"},
        )

        frames = _parse_sse(sse_resp.text)
        response_payload = next(data for ev, data in frames if ev == "response")["response"]

        # LLM mock ordering can change other response fields between calls.
        assert response_payload["status"] == json_resp.json()["status"]

    def test_default_accept_returns_json(self):
        # Without text/event-stream in Accept, the response is buffered JSON.
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "raw_text",
                    "query": "health ODA grants from USA to Ethiopia",
                }
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("application/json")
        body = resp.json()
        assert body["status"] in ("definite", "candidates", "no_data")

    def test_wildcard_accept_does_not_trigger_sse(self):
        # The wildcard Accept header selects buffered JSON.
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "raw_text",
                    "query": "health ODA grants from USA to Ethiopia",
                }
            },
            headers={"Accept": "*/*"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("application/json")

    def test_sse_q0_does_not_trigger_sse(self):
        # q=0 makes text/event-stream unacceptable.
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "raw_text",
                    "query": "health ODA grants from USA to Ethiopia",
                }
            },
            headers={"Accept": "text/event-stream;q=0"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("application/json")


class TestEngineInputError:
    def test_engine_input_error_returns_400(self, monkeypatch):
        # EngineInputError without a status maps to 400.
        from search.engine.errors import EngineInputError

        async def _raise_input_error(*args, **kwargs):
            raise EngineInputError("unknown shape_id")

        monkeypatch.setattr("search.engine.app.resolve_async", _raise_input_error)
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "test"}},
        )
        _assert_error(resp, status=400, code="request_invalid", detail="Bad request")

    def test_engine_input_error_with_code_includes_code(self, monkeypatch):
        # Preserve an explicit error code and status in the response.
        from search.engine.errors import EngineInputError

        async def _raise_with_code(*args, **kwargs):
            raise EngineInputError(
                "recipient not found in the graph", code="invalid_role_ref", status=422
            )

        monkeypatch.setattr("search.engine.app.resolve_async", _raise_with_code)
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "test"}},
        )
        _assert_error(
            resp, status=422, code="invalid_role_ref", detail="recipient not found in the graph"
        )

    def test_engine_input_error_routing_failure_falls_back_to_request_invalid(self, monkeypatch):
        # Routing errors use a generic detail without exposing available shape ids.
        from search.engine.errors import EngineInputError

        async def _raise_routing(*args, **kwargs):
            raise EngineInputError("shape_id mismatch")  # no code kwarg

        monkeypatch.setattr("search.engine.app.resolve_async", _raise_routing)
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "test"}},
        )
        _assert_error(resp, status=400, code="request_invalid", detail="Bad request")


def test_resubmit_identity_naming_no_measured_property_answers_variable_not_resolved():
    """A resubmitted identity whose shape names no measured property anchors no member walk.
    The mint refuses it, so the part answers `variable_not_resolved`."""
    client = make_client()
    resp = client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "dataset_resubmit",
                "entries": [
                    {
                        "entry_kind": "identity",
                        "variable_text": "a resubmitted identity",
                        "identity": {
                            "shape": {
                                "kind": "statvar_schema",
                                "population_type": "nonexistent_shape_xyz",
                            },
                            "slots": [],
                            "entities": {},
                            "requested_window": {"kind": "all"},
                        },
                    }
                ],
            }
        },
    )
    assert resp.status_code == 200, resp.text
    (part,) = resp.json()["parts"]
    assert part["no_data"]["reason"] == "variable_not_resolved"


def test_resubmit_identity_missing_role_refs_name_every_missing_dcid():
    """The resubmit door preserves every rejected role ref in
    its invalid_role_ref problem detail."""
    missing_dcids = ["country/ZZZ_NOT_A_PLACE", "country/ZZZ_ALSO_NOT_A_PLACE"]
    entities = [subject_entity(dcid) for dcid in missing_dcids]
    gdp = value_slot("how", "activitySource", "activity source", "GrossDomesticProduction")
    identity = make_identity(GDP_SHAPE, [gdp], entities=entities)

    client = make_client()
    resp = client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "dataset_resubmit",
                "entries": [
                    {
                        "entry_kind": "identity",
                        "variable_text": "a resubmitted identity",
                        "identity": identity.model_dump(mode="json"),
                    }
                ],
            }
        },
    )

    body = _assert_error(resp, status=422, code="invalid_role_ref")
    assert body["pointer"] == "/input/entries/0/identity/entities/observationAbout"
    assert_names_plainly(body["detail"], sorted(missing_dcids))


def test_resubmit_identity_shape_with_unknown_member_returns_400():
    """Both described and pinned shape arms reject unrecognised members during validation."""
    client = make_client()
    for shape in (
        {"kind": "statvar_schema", "measured_property": "X", "bogus": "y"},
        {"kind": "stat_var", "id": "X", "bogus": "y"},
    ):
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": {
                                "shape": shape,
                                "slots": [],
                                "entities": {},
                                "requested_window": {"kind": "all"},
                            },
                        }
                    ],
                }
            },
        )
        _assert_error(resp, status=400, code="request_invalid")


def _base_identity_with_facets(facets: dict) -> dict:
    return {
        "shape": {"kind": "statvar_schema", "population_type": "some_shape"},
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
        "facets": facets,
    }


class TestFacetAxisVersioning:
    """Unknown facet axes return request_invalid/400.

    Known axes above the deployed schema version return unsupported_schema_version/422.
    """

    def test_unknown_facet_axis_is_request_invalid_400(self):
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": _base_identity_with_facets(
                                {"bogus_axis": {"state": "unbound"}}
                            ),
                        }
                    ],
                }
            },
        )
        _assert_error(resp, status=400, code="request_invalid")

    def test_higher_schema_version_facet_axis_is_unsupported_schema_version_422(self, monkeypatch):
        import search.models as models

        monkeypatch.setitem(models._FACET_AXIS_SCHEMA_VERSION, "futureAxis", 99)
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": _base_identity_with_facets(
                                {"futureAxis": {"state": "value", "ref": "x"}}
                            ),
                        }
                    ],
                }
            },
        )
        _assert_error(resp, status=422, code="unsupported_schema_version")

    def test_bad_schema_version_and_higher_facet_axis_both_present_lands_one_422(self, monkeypatch):
        import search.models as models

        monkeypatch.setitem(models._FACET_AXIS_SCHEMA_VERSION, "futureAxis", 99)
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "schema_version": 999,
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": _base_identity_with_facets(
                                {"futureAxis": {"state": "value", "ref": "x"}}
                            ),
                        }
                    ],
                },
            },
        )
        _assert_error(resp, status=422, code="unsupported_schema_version")

    def test_resubmit_facets_unknown_axis_is_request_invalid_400(self):
        """The top-level facets map uses the same axis validation as base_identity.facets."""
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": _base_identity_with_facets(
                                {"bogus_axis": {"state": "unbound"}}
                            ),
                        }
                    ],
                }
            },
        )
        _assert_error(resp, status=400, code="request_invalid")

    def test_resubmit_facets_higher_schema_version_axis_is_unsupported_schema_version_422(
        self, monkeypatch
    ):
        import search.models as models

        monkeypatch.setitem(models._FACET_AXIS_SCHEMA_VERSION, "futureAxis", 99)
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": _base_identity_with_facets(
                                {"futureAxis": {"state": "value", "ref": "x"}}
                            ),
                        }
                    ],
                }
            },
        )
        _assert_error(resp, status=422, code="unsupported_schema_version")


class TestCORS:
    def test_cors_preflight_allowed_origin(self, monkeypatch):
        # An origin in DFO_CORS_ORIGINS receives Access-Control-Allow-Origin.
        monkeypatch.setenv("DFO_CORS_ORIGINS", "https://app.example.com")
        client = make_client()
        resp = client.options(
            "/api/search/resolve",
            headers={
                "Origin": "https://app.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "https://app.example.com"

    def test_cors_no_header_for_empty_allowlist(self, monkeypatch):
        # An empty allowlist adds no CORS headers.
        monkeypatch.delenv("DFO_CORS_ORIGINS", raising=False)
        client = make_client()
        resp = client.options(
            "/api/search/resolve",
            headers={
                "Origin": "https://attacker.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" not in resp.headers


@pytest.mark.parametrize("transport", ["buffered", "sse"])
def test_resolve_logs_one_line_with_caller_flags_transport_cache_and_timing(transport):
    """Each resolve logs one `resolve` line whose timing, trace id and outcome match the
    response it served."""
    headers = {"User-Agent": "dfo-test/1"}
    if transport == "sse":
        headers["Accept"] = "text/event-stream"
    with resolve_log_lines() as lines:
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {"kind": "raw_text", "query": "fertility rate in Kenya"},
                "response_options": {"include_observations": True},
            },
            headers=headers,
        )
    assert resp.status_code == 200
    body = (
        resp.json()
        if transport == "buffered"
        else next(data["response"] for event, data in _parse_sse(resp.text) if event == "response")
    )

    [logged] = lines
    assert logged["trace_id"] == resp.headers["x-dfo-trace-id"] == body["diagnostics"]["trace_id"]
    assert logged["kind"] == "raw_text"
    assert logged["caller"] == "dfo-test/1"
    assert logged["include_observations"] is True
    assert "include_skills" not in logged
    assert logged["transport"] == transport
    assert logged["cache"] in {"inactive", "miss", "hit layer1", "hit layer2"}
    timing = body["diagnostics"]["timing_ms"]
    assert logged["total_ms"] == timing["total"]
    assert logged["by_step"] == timing.get("by_step")
    assert logged["parts_by_step"] == [
        (part.get("timing_ms") or {}).get("by_step") for part in body["parts"]
    ]
    assert any(logged["parts_by_step"])
    assert logged["outcome"] == body["status"]
    assert logged["part_statuses"] == [part["status"] for part in body["parts"]]
    assert logged["llm_usage"] == body["diagnostics"].get("llm_usage")
    assert logged["timed_out"] is False
    assert logged["degraded_leg"] is False
    assert logged["query_hmac"] is None


def test_sse_pipeline_failure_logs_its_traceback_once(monkeypatch):
    """An unhandled SSE pipeline failure ends in one `internal_error` event, and exactly one
    log record carries its traceback, as on the JSON route."""
    boom = RuntimeError("boom")

    async def _failing_events(*args, **kwargs):
        raise boom
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr("search.engine.app.resolve_events", _failing_events)
    # Handlers on the emitting loggers, since the `search` logger may stop propagation.
    handler = BufferingHandler(capacity=1000)
    loggers = [logging.getLogger("search.engine.app"), logging.getLogger("dfo_platform.refusal")]
    for each in loggers:
        each.addHandler(handler)
    try:
        resp = make_client().post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "fertility rate in Kenya"}},
            headers={"Accept": "text/event-stream"},
        )
    finally:
        for each in loggers:
            each.removeHandler(handler)

    frames = _parse_sse(resp.text)
    assert [(event, data["code"]) for event, data in frames] == [("error", "internal_error")]
    assert len([r for r in handler.buffer if r.exc_info and r.exc_info[1] is boom]) == 1


# ---------------------------------------------------------------------------
# Request budget: the soft deadline's partial answer, then the hard cap
# ---------------------------------------------------------------------------


def _extract_two_variables(query: str, *, llm=None):
    async def _inner() -> tuple[ExtractionResult, None]:
        extraction = Extraction(
            kind="resolution", variables=["fast", "slow"], entities=[], dates=[]
        )
        return ExtractionResult(extraction=extraction, qualifiers={}), None

    return _inner()


def test_json_route_answers_the_partial_200_when_a_leg_runs_past_the_soft_deadline(
    monkeypatch, search_settings
):
    """A leg blocked in a worker thread at the soft deadline is cancelled, and the partial
    answer lands inside the grace, before the hard cap's 504. The thread sees the request deadline,
    so its reads are bounded by the hard cap. The 2 s grace bounds only a slow runner's cancel and
    thread hop, which fails the test red with a 504 and never passes it falsely."""
    search_settings(request_timeout_s=0.3, finalise_grace_s=2.0)
    monkeypatch.setattr("search.engine.entry.raw_text.extract", _extract_two_variables)
    release = threading.Event()
    deadlines_seen: list[float | None] = []
    budgets: list[RequestBudget] = []

    def _recording_budget(*, kind: str) -> RequestBudget:
        budgets.append(request_budget(kind=kind))
        return budgets[-1]

    monkeypatch.setattr("search.engine.app.request_budget", _recording_budget)

    def _blocked_read() -> None:
        deadlines_seen.append(bounded_deadline(None))
        release.wait(timeout=10)

    async def _leg(variable: str, **_kwargs: object) -> RegionResult:
        if variable == "slow":
            try:
                await asyncio.to_thread(_blocked_read)
            finally:
                # The thread outlives the cancelled await. Releasing it lets TestClient's
                # loop shut its executor down without waiting for the timeout.
                release.set()
        return RegionResult(
            variable_text=variable,
            status="no_data",
            specs=(),
            no_data_reason="variable_not_resolved",
            warnings=(),
            timing_by_step={},
        )

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _leg)
    client = make_client()
    try:
        with resolve_log_lines() as lines:
            resp = client.post(
                "/api/search/resolve", json={"input": {"kind": "raw_text", "query": "two things"}}
            )
    finally:
        release.set()

    # A 200 means the partial answer landed before the hard cap, which answers 504.
    assert resp.status_code == 200, resp.text
    assert resp.json()["diagnostics"]["timed_out"] is True
    # The finished leg keeps its own reason and the cut leg reads `timed_out`.
    [logged] = lines
    assert logged["outcome"] == "no_data"
    assert logged["no_data_reasons"] == ["variable_not_resolved", "timed_out"]
    assert logged["timed_out"] is True
    [budget] = budgets
    assert budget.hard - budget.soft == pytest.approx(2.0)
    assert deadlines_seen == [budget.hard]


async def _stalled_pipeline(*_args: object, **_kwargs: object):
    """A pipeline that never finishes, cut from outside. Bounded so a test whose cut does not
    arrive fails instead of hanging."""
    await asyncio.sleep(5)
    return
    yield  # pragma: no cover - makes this an async generator


@pytest.mark.parametrize("use_uvloop", [False, True], ids=["asyncio", "uvloop"])
@pytest.mark.parametrize("transport", ["buffered", "sse"])
def test_the_hard_cap_reads_as_a_timeout_on_the_resolve_line(
    monkeypatch, search_settings, transport, use_uvloop
):
    """The hard cap reaches `resolve_events` as cancellation on both transports and on either
    loop. Production runs uvloop, whose timers fire ahead of the instant they were set for."""
    search_settings(request_timeout_s=0.05, finalise_grace_s=0.05)
    monkeypatch.setattr("search.engine.core._resolve_pipeline_events", _stalled_pipeline)
    headers = {"Accept": "text/event-stream"} if transport == "sse" else {}
    client = TestClient(
        make_client().app,
        raise_server_exceptions=False,
        backend_options={"use_uvloop": use_uvloop},
    )
    with resolve_log_lines() as lines:
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "fertility rate in Kenya"}},
            headers=headers,
        )

    if transport == "buffered":
        assert resp.status_code == 504
    else:
        assert [event for event, _data in _parse_sse(resp.text)] == ["error"]
    [logged] = lines
    assert (logged["error_code"], logged["timed_out"]) == ("timeout", True)
    assert logged["outcome"] is None


@pytest.mark.parametrize("unwinds_s", [0, 0.2], ids=["before_the_cap", "crossing_the_cap"])
def test_a_client_disconnect_reads_as_cancelled_under_the_request_trace_id(
    monkeypatch, search_settings, unwinds_s
):
    """A disconnect unwinds the pipeline in the stream's producer task, whose copied context
    still holds the id the response header carries. One still unwinding when the hard cap
    fires stays a disconnect, as `asyncio.timeout` decides."""
    search_settings(request_timeout_s=0.05, finalise_grace_s=0.05)

    async def _slowly_cut_pipeline(*_args: object, **_kwargs: object):
        try:
            await asyncio.sleep(5)
        finally:
            await asyncio.shield(asyncio.sleep(unwinds_s))
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr("search.engine.core._resolve_pipeline_events", _slowly_cut_pipeline)
    app = make_client().app
    body = json.dumps({"input": {"kind": "raw_text", "query": "fertility rate in Kenya"}})
    started = asyncio.Event()
    headers: dict[bytes, bytes] = {}
    requested = False

    async def receive() -> dict:
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": body.encode(), "more_body": False}
        await started.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            headers.update(message["headers"])
            started.set()

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/search/resolve",
        "raw_path": b"/api/search/resolve",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"accept", b"text/event-stream"),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "scheme": "http",
        "root_path": "",
        "http_version": "1.1",
    }
    with resolve_log_lines() as lines:
        asyncio.run(asyncio.wait_for(app(scope, receive, send), timeout=4))

    [logged] = lines
    assert (logged["error_code"], logged["timed_out"]) == ("cancelled", None)
    assert logged["trace_id"].encode() == headers[b"x-dfo-trace-id"]


def test_sse_route_ends_at_the_hard_cap_with_the_timeout_error_event(monkeypatch, search_settings):
    """A pipeline that never finishes ends at soft deadline plus grace with the 504 `timeout`
    error event, and a worker thread inside the stream sees the hard cap as its deadline. The
    stream arms its own timer at the hard cap too, not at the soft deadline."""
    search_settings(request_timeout_s=0.1, finalise_grace_s=1.0)
    deadlines_seen: list[float | None] = []
    budgets: list[RequestBudget] = []
    armed: list[float] = []

    class _RecordingAsyncio:
        """`search.engine.app.asyncio` with `timeout` recording when each timer fires, on the
        loop clock, which is `time.monotonic()`."""

        def __getattr__(self, name: str) -> object:
            return getattr(asyncio, name)

        def timeout(self, delay: float | None) -> asyncio.Timeout:
            timer = asyncio.timeout(delay)
            when = timer.when()
            assert when is not None
            armed.append(when)
            return timer

    monkeypatch.setattr("search.engine.app.asyncio", _RecordingAsyncio())

    async def _hanging_events(*_args: object, budget: RequestBudget, **_kwargs: object):
        budgets.append(budget)
        deadlines_seen.append(await asyncio.to_thread(bounded_deadline, None))
        # Long past the cap. Bounded so a stream without the cap fails instead of hanging.
        await asyncio.sleep(5)
        return
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr("search.engine.app.resolve_events", _hanging_events)
    client = make_client()
    resp = client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": "fertility rate in Kenya"}},
        headers={"Accept": "text/event-stream"},
    )

    frames = _parse_sse(resp.text)
    assert [(event, data["code"], data["status"]) for event, data in frames] == [
        ("error", "timeout", 504)
    ]
    [budget] = budgets
    assert budget.hard - budget.soft == pytest.approx(1.0)
    assert deadlines_seen == [budget.hard]
    # The timer is armed at the budget's hard deadline, less the time since the route read it.
    assert armed == [pytest.approx(budget.hard, abs=0.1)]
