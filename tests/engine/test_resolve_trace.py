"""Pins the Langfuse trace `send_resolve_trace` queues for each resolve a route runs."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from dfo_platform.caller import (
    UNRESOLVED_IDENTITY,
    DeliverySurface,
    reset_delivery_identity,
    set_delivery_identity,
)
from dfo_platform.trace import reset_trace_id, set_trace_id
from search.engine import resolve_trace
from search.engine.app import _served_code
from search.engine.core import ResolveRecord, resolve_line
from search.engine.envelope import make_diagnostics, make_query_echo
from search.models import (
    RESOLVE_RESPONSE_ADAPTER,
    CandidateSet,
    CandidatesPart,
    NoData,
    NoDataPart,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    Resolution,
    ResolveRequest,
)
from tests.engine._harness import EMPTY_SDMX, make_request, resolve_log_lines
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_TRACE_ID = "0123456789abcdef0123456789abcdef"
_KEYS = {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"}


def _drain() -> None:
    """Wait for every trace queued so far, since the worker runs them in order."""
    resolve_trace._EXECUTOR.submit(lambda: None).result(timeout=10)


@pytest.fixture
def exported(monkeypatch) -> Iterator[Callable[[], list[ReadableSpan]]]:
    """Turn tracing on with a client that exports to memory, and yield a function returning
    the spans exported once every queued trace has been recorded."""
    for name, value in _KEYS.items():
        monkeypatch.setenv(name, value)
    exporter = InMemorySpanExporter()
    # The SDK keeps one resource manager per public key, so each test gets its own key.
    client = Langfuse(
        public_key=f"pk-{uuid.uuid4().hex}",
        secret_key="sk-test",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
    )
    monkeypatch.setattr(resolve_trace, "_client", lambda: client)

    def finished() -> list[ReadableSpan]:
        _drain()
        client.flush()
        return list(exporter.get_finished_spans())

    yield finished
    client.shutdown()


def _run_line(request: ResolveRequest, record: ResolveRecord, *, surface: DeliverySurface) -> None:
    """Run one resolve's log block under `_TRACE_ID` and a delivery identity on `surface`."""
    trace_token = set_trace_id(_TRACE_ID)
    identity_token = set_delivery_identity(replace(UNRESOLVED_IDENTITY, surface=surface))
    try:
        with resolve_line(
            request, record, caller=None, transport="buffered", served_code=_served_code
        ):
            pass
    finally:
        reset_delivery_identity(identity_token)
        reset_trace_id(trace_token)


def _resolution(worked_example_response: dict) -> Resolution:
    definite = RESOLVE_RESPONSE_ADAPTER.validate_python(worked_example_response).parts[0]
    ranked = [
        definite.spec.model_copy(
            update={
                "spec_id": f"spec-{n}",
                "resolved": definite.spec.resolved.model_copy(
                    update={"rendered_sentence": f"reading {n} " + "x" * 600}
                ),
            }
        )
        for n in range(12)
    ]
    candidates = CandidatesPart(
        status="candidates",
        candidates=CandidateSet(
            ordering="intent_rank", max_candidates=12, specs=ranked, truncated=False
        ),
    )
    no_data = NoDataPart(status="no_data", no_data=NoData(reason="entity_not_resolved"))
    return Resolution(
        query_echo=make_query_echo("aid to Kenya", extract_skipped=False),
        diagnostics=make_diagnostics("test-build", [], {}, 5),
        parts=[definite, candidates, no_data],
    )


def test_a_resolve_sends_one_trace_under_its_own_trace_id(exported, worked_example_response):
    response = _resolution(worked_example_response)
    query = "aid to Kenya " + "y" * 2000

    _run_line(make_request(query), ResolveRecord(response=response), surface=DeliverySurface.mcp)

    [span] = exported()
    attributes = dict(span.attributes or {})
    assert span.name == "search.resolve"
    assert format(span.context.trace_id, "032x") == _TRACE_ID
    assert attributes["langfuse.trace.name"] == "search.resolve"
    assert attributes["langfuse.trace.metadata.surface"] == "mcp"
    assert json.loads(attributes["langfuse.observation.input"]) == {
        "kind": "raw_text",
        "query": query[:500],
    }
    definite = response.parts[0]
    assert json.loads(attributes["langfuse.observation.output"]) == {
        "kind": "resolution",
        "status": response.status,
        "parts": [
            {
                "status": "definite",
                "specs": [
                    {
                        "spec_id": definite.spec.spec_id,
                        "definition": definite.spec.resolved.rendered_sentence,
                    }
                ],
            },
            {
                "status": "candidates",
                "specs": [
                    {"spec_id": f"spec-{n}", "definition": (f"reading {n} " + "x" * 600)[:500]}
                    for n in range(10)
                ],
            },
            {"status": "no_data", "specs": []},
        ],
        "error_code": None,
    }


def test_a_parsed_request_traces_its_whole_input(exported):
    request = ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(
                    variable_text="health aid",
                    entities=[ParsedEntity(text="Kenya")],
                    time_text="since 2015",
                )
            ]
        )
    )

    _run_line(request, ResolveRecord(error_code="cancelled"), surface=DeliverySurface.website)

    [span] = exported()
    attributes = dict(span.attributes or {})
    assert json.loads(attributes["langfuse.observation.input"]) == {
        "kind": "parsed",
        "parts": [
            {
                "variable_text": "health aid",
                "entities": [{"text": "Kenya"}],
                "time_text": "since 2015",
                "sources": [],
                "operations": [],
            }
        ],
    }
    assert json.loads(attributes["langfuse.observation.output"]) == {
        "kind": None,
        "status": None,
        "parts": [],
        "error_code": "cancelled",
    }
    assert attributes["langfuse.trace.metadata.surface"] == "website"


@pytest.mark.parametrize(
    ("keys", "trace_id", "surface"),
    [
        pytest.param({}, _TRACE_ID, "website", id="no_keys"),
        pytest.param({"LANGFUSE_PUBLIC_KEY": "pk-test"}, _TRACE_ID, "mcp", id="one_key"),
        pytest.param(_KEYS, None, "website", id="no_trace_id"),
        pytest.param(_KEYS, _TRACE_ID, "studio", id="studio"),
        pytest.param(_KEYS, _TRACE_ID, "api", id="api"),
        pytest.param(_KEYS, _TRACE_ID, "direct", id="direct"),
        pytest.param(_KEYS, _TRACE_ID, "unknown", id="unknown"),
    ],
)
def test_nothing_is_sent_without_both_keys_a_trace_id_and_a_traced_surface(
    monkeypatch, keys, trace_id, surface
):
    for name in _KEYS:
        monkeypatch.delenv(name, raising=False)
    for name, value in keys.items():
        monkeypatch.setenv(name, value)
    calls: list[str] = []
    monkeypatch.setattr(resolve_trace, "_client", lambda: calls.append("client"))

    resolve_trace.send_resolve_trace(
        trace_id=trace_id,
        inp=make_request("aid to Kenya").input,
        response=None,
        error_code=None,
        surface=surface,
    )

    _drain()
    assert calls == []


@pytest.mark.parametrize("transport", ["buffered", "sse"])
def test_a_failing_langfuse_client_leaves_the_resolve_standing(monkeypatch, transport):
    for name, value in _KEYS.items():
        monkeypatch.setenv(name, value)
    calls: list[str] = []

    def _failing_client():
        calls.append("client")
        raise RuntimeError("langfuse down")

    monkeypatch.setattr(resolve_trace, "_client", _failing_client)
    # The test app vouches for no caller, so the request is named as the website's.
    website = replace(UNRESOLVED_IDENTITY, surface=DeliverySurface.website)
    monkeypatch.setattr("search.engine.core.current_delivery_identity", lambda: website)
    graph = FakeGraph()
    app = search_app(graph=graph, resolver=make_resolver(graph), llm=FakeLLM(), sdmx=EMPTY_SDMX)
    headers = {"Accept": "text/event-stream"} if transport == "sse" else {}

    with resolve_log_lines() as lines:
        resp = TestClient(app, raise_server_exceptions=False).post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "fertility rate in Kenya"}},
            headers=headers,
        )

    _drain()
    assert resp.status_code == 200
    [logged] = lines
    assert logged["error_code"] is None
    assert calls == ["client"]
