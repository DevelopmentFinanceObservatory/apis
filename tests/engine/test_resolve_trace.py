"""Pins the Langfuse trace `queue_resolve_trace` queues for each resolve a route runs, and the
`trace` field of the `resolve` log line."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from dfo_platform.caller import (
    UNRESOLVED_IDENTITY,
    DeliverySurface,
    reset_delivery_identity,
    set_delivery_identity,
)
from dfo_platform.trace import reset_trace_id, set_trace_id
from search.engine import resolve_trace
from search.engine.app import _event_source, _served_code
from search.engine.core import ResolveRecord, resolve_line
from search.engine.envelope import make_diagnostics, make_query_echo
from search.engine.events import ResponseEvent
from search.models import (
    RESOLVE_RESPONSE_ADAPTER,
    CandidateSet,
    CandidatesPart,
    DatasetResubmitInput,
    NoData,
    NoDataPart,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    Resolution,
    ResolveRequest,
    ResubmitPlaceholderEntry,
)
from tests.engine._harness import EMPTY_SDMX, make_request, resolve_log_lines
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_TRACE_ID = "0123456789abcdef0123456789abcdef"
_KEYS = {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"}
_ON = {**_KEYS, "SEARCH_TRACE_ENABLED": "true"}
_ENV_VARS = (*_KEYS, "SEARCH_TRACE_ENABLED")
_RESUBMIT = ResolveRequest(
    input=DatasetResubmitInput(
        entries=[
            ResubmitPlaceholderEntry(
                variable_text="aid", status="no_data", reason="variable_not_resolved"
            )
        ]
    )
)


def _drain() -> None:
    """Wait for every trace queued so far, since the worker runs them in order."""
    resolve_trace._EXECUTOR.submit(lambda: None).result(timeout=10)


def _tracing_to(monkeypatch, exporter: SpanExporter) -> Langfuse:
    """Turn tracing on with a client exporting through `exporter`, wrapped as `_client` wraps
    its own."""
    for name, value in _ON.items():
        monkeypatch.setenv(name, value)
    results = resolve_trace._ExportResults(exporter)
    # The SDK keeps one resource manager per public key, so each client gets its own key.
    client = Langfuse(
        public_key=f"pk-{uuid.uuid4().hex}",
        secret_key="sk-test",
        tracer_provider=TracerProvider(),
        span_exporter=results,
    )
    monkeypatch.setattr(resolve_trace, "_client", lambda: resolve_trace._Tracing(client, results))
    return client


@pytest.fixture
def exported(monkeypatch) -> Iterator[Callable[[], list[ReadableSpan]]]:
    """Turn tracing on with a client that exports to memory, and yield a function returning
    the spans exported once every queued trace has been recorded."""
    exporter = InMemorySpanExporter()
    client = _tracing_to(monkeypatch, exporter)

    def finished() -> list[ReadableSpan]:
        _drain()
        client.flush()
        return list(exporter.get_finished_spans())

    yield finished
    client.shutdown()


class _ScriptedExporter(SpanExporter):
    """An exporter that answers every export with `answer`, or raises when it is None."""

    def __init__(self, answer: SpanExportResult | None) -> None:
        self.answer = answer

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self.answer is None:
            raise ConnectionError("exporter down")
        return self.answer

    def shutdown(self) -> None:
        pass


def _run_line(request: ResolveRequest, record: ResolveRecord, *, surface: DeliverySurface) -> dict:
    """Run one resolve's log block under `_TRACE_ID` and a delivery identity on `surface`, and
    return its log line."""

    async def _block() -> None:
        async with resolve_line(
            request, record, caller=None, transport="buffered", served_code=_served_code
        ):
            pass

    trace_token = set_trace_id(_TRACE_ID)
    identity_token = set_delivery_identity(replace(UNRESOLVED_IDENTITY, surface=surface))
    try:
        with resolve_log_lines() as lines:
            asyncio.run(_block())
    finally:
        reset_delivery_identity(identity_token)
        reset_trace_id(trace_token)
    [logged] = lines
    return logged


@pytest.fixture
def held_worker(monkeypatch) -> Iterator[threading.Event]:
    """Turn tracing on with a client the worker waits for until the yielded gate is set, which
    teardown sets."""
    for name, value in _ON.items():
        monkeypatch.setenv(name, value)
    gate = threading.Event()

    def _held_client():
        gate.wait(timeout=30)
        raise RuntimeError("no client in this test")

    monkeypatch.setattr(resolve_trace, "_client", _held_client)
    yield gate
    gate.set()
    _drain()


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

    logged = _run_line(
        make_request(query), ResolveRecord(response=response), surface=DeliverySurface.mcp
    )

    assert logged["trace"] == "sent"
    [span] = exported()
    attributes = dict(span.attributes or {})
    assert span.name == "search.resolve"
    assert format(span.context.trace_id, "032x") == _TRACE_ID
    assert attributes["langfuse.trace.name"] == "search.resolve"
    assert json.loads(attributes["langfuse.trace.metadata.surface"]) == "mcp"
    assert json.loads(attributes["langfuse.observation.input"]) == {
        "kind": "raw_text",
        "query": query[:500],
        "studio_routing": False,
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
    assert json.loads(attributes["langfuse.trace.metadata.surface"]) == "website"


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        pytest.param(SpanExportResult.SUCCESS, "sent", id="success"),
        pytest.param(SpanExportResult.FAILURE, "failed", id="failure"),
        pytest.param(None, "failed", id="raises"),
    ],
)
def test_the_line_reads_the_exporter_s_own_result(monkeypatch, answer, outcome):
    client = _tracing_to(monkeypatch, _ScriptedExporter(answer))
    try:
        logged = _run_line(
            make_request("aid to Kenya"), ResolveRecord(), surface=DeliverySurface.website
        )
    finally:
        client.shutdown()

    assert logged["trace"] == outcome


@pytest.mark.parametrize(("status", "outcome"), [(200, "sent"), (401, "failed")])
@pytest.mark.parametrize("export_path", [None, "custom/otel/v1/traces"])
@pytest.mark.parametrize(("compression", "encoding"), [(None, "gzip"), ("none", None)])
def test_the_process_client_posts_as_the_default_langfuse_exporter_and_reads_its_status(
    monkeypatch, status, outcome, export_path, compression, encoding
):
    """The client `_client` builds and a Langfuse client left to build its own exporter post to
    one local host, which answers every OTLP post with `status`, as Langfuse answers keys it
    rejects with 401. Both posts carry the same path and headers, the keys apart, so a Langfuse
    upgrade that changes what its exporter sends fails here."""
    posts: list[tuple[str, dict[str, str]]] = []

    class _Host(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            headers = {
                k.lower(): v for k, v in self.headers.items() if k.lower() != "content-length"
            }
            posts.append((self.path, headers))
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Host)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    ours, theirs = f"pk-{uuid.uuid4().hex}", f"pk-{uuid.uuid4().hex}"
    for name, value in _ON.items():
        monkeypatch.setenv(name, value)
    for name in (
        "LANGFUSE_OTEL_COMPRESSION",
        "OTEL_EXPORTER_OTLP_TRACES_COMPRESSION",
        "OTEL_EXPORTER_OTLP_COMPRESSION",
    ):
        monkeypatch.delenv(name, raising=False)
    if compression is not None:
        monkeypatch.setenv("LANGFUSE_OTEL_COMPRESSION", compression)
    monkeypatch.setenv("LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES", "1234567")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", ours)
    monkeypatch.setenv("LANGFUSE_BASE_URL", f"{base_url}/")
    if export_path is None:
        monkeypatch.delenv("LANGFUSE_OTEL_TRACES_EXPORT_PATH", raising=False)
    else:
        monkeypatch.setenv("LANGFUSE_OTEL_TRACES_EXPORT_PATH", export_path)
    resolve_trace._client.cache_clear()
    default_provider = TracerProvider()
    default = Langfuse(
        public_key=theirs, secret_key="sk-test", base_url=base_url, tracer_provider=default_provider
    )
    try:
        logged = _run_line(
            make_request("aid to Kenya"), ResolveRecord(), surface=DeliverySurface.mcp
        )
        with default.start_as_current_observation(name="default", as_type="span"):
            pass
        default.flush()
        ours_size = getattr(resolve_trace._client().results._exporter, "_max_request_size")  # noqa: B009
        (processor,) = default_provider._active_span_processor._span_processors
        theirs_size = getattr(processor, "span_exporter")._exporter._max_request_size  # noqa: B009
    finally:
        _drain()
        resolve_trace._client().client.shutdown()
        resolve_trace._client.cache_clear()
        default.shutdown()
        server.shutdown()

    assert logged["trace"] == outcome

    def keyed(public_key: str) -> tuple[str, dict[str, str]]:
        """The one post carrying `public_key`, with its key-bound headers taken out."""
        (post,) = [p for p in posts if p[1].get("x-langfuse-public-key") == public_key]
        path, headers = post
        auth = "Basic " + base64.b64encode(f"{public_key}:sk-test".encode()).decode()
        assert headers.pop("authorization") == auth
        headers.pop("x-langfuse-public-key")
        return path, headers

    expected_path = f"/{export_path}" if export_path else resolve_trace.OTLP_PATH
    assert {path for path, _ in posts} == {expected_path}
    our_path, our_headers = keyed(ours)
    assert (our_path, our_headers) == keyed(theirs)
    assert our_headers["x-langfuse-ingestion-version"] == "4"
    assert our_headers.get("content-encoding") == encoding
    assert ours_size == theirs_size == 1234567


def test_a_drain_past_its_deadline_cancels_the_queued_traces(monkeypatch, held_worker):
    """The held worker keeps the first trace in flight past the drain's deadline. The traces
    queued behind it never run, release their slots, and a trace queued after the shutdown
    reads `dropped`."""
    shared = resolve_trace._EXECUTOR
    worker = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(resolve_trace, "_EXECUTOR", worker)
    monkeypatch.setattr(resolve_trace, "SHUTDOWN_DRAIN_S", 0.1)
    ran: list[str] = []
    held = resolve_trace._client

    def _counted_client():
        ran.append("client")
        return held()

    monkeypatch.setattr(resolve_trace, "_client", _counted_client)

    def _queue() -> tuple[str, object]:
        return resolve_trace.queue_resolve_trace(
            trace_id=_TRACE_ID,
            inp=make_request("aid to Kenya").input,
            response=None,
            error_code=None,
            surface="website",
        )

    queued = [_queue() for _ in range(4)]
    asyncio.run(resolve_trace.drain_resolve_traces())
    late = _queue()
    held_worker.set()
    worker.shutdown(wait=True)
    monkeypatch.setattr(resolve_trace, "_EXECUTOR", shared)

    assert [outcome for outcome, _future in queued] == ["queued"] * 4
    assert [future.cancelled() for _outcome, future in queued] == [False, True, True, True]
    assert ran == ["client"]
    assert late == ("dropped", None)
    assert asyncio.run(resolve_trace.await_resolve_trace(queued[1][1])) == "dropped"
    taken = [
        resolve_trace._PENDING.acquire(blocking=False) for _ in range(resolve_trace.PENDING_CAP)
    ]
    for _ in range(sum(taken)):
        resolve_trace._PENDING.release()
    assert all(taken)


@pytest.mark.parametrize(
    ("env", "trace_id", "surface", "request_", "outcome"),
    [
        pytest.param({}, _TRACE_ID, "website", None, "off", id="nothing_set"),
        pytest.param(_KEYS, _TRACE_ID, "website", None, "off", id="keys_without_opt_in"),
        pytest.param(
            {**_KEYS, "SEARCH_TRACE_ENABLED": "yes"},
            _TRACE_ID,
            "mcp",
            None,
            "off",
            id="not_1_or_true",
        ),
        pytest.param(
            {"LANGFUSE_PUBLIC_KEY": "pk-test", "SEARCH_TRACE_ENABLED": "1"},
            _TRACE_ID,
            "mcp",
            None,
            "off",
            id="one_key",
        ),
        pytest.param(_ON, None, "website", None, "skipped", id="no_trace_id"),
        pytest.param(_ON, _TRACE_ID, "studio", None, "skipped", id="studio"),
        pytest.param(_ON, _TRACE_ID, "api", None, "skipped", id="api"),
        pytest.param(_ON, _TRACE_ID, "direct", None, "skipped", id="direct"),
        pytest.param(_ON, _TRACE_ID, "unknown", None, "skipped", id="unknown"),
        pytest.param(_ON, _TRACE_ID, "mcp", _RESUBMIT, "skipped", id="dataset_resubmit"),
    ],
)
def test_nothing_is_sent_without_the_opt_in_both_keys_a_trace_id_and_a_traced_request(
    monkeypatch, env, trace_id, surface, request_, outcome
):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    calls: list[str] = []
    monkeypatch.setattr(resolve_trace, "_client", lambda: calls.append("client"))

    queued = resolve_trace.queue_resolve_trace(
        trace_id=trace_id,
        inp=(request_ or make_request("aid to Kenya")).input,
        response=None,
        error_code=None,
        surface=surface,
    )

    _drain()
    assert queued == (outcome, None)
    assert calls == []


def test_a_trace_past_the_pending_cap_is_dropped_and_the_line_says_so(held_worker):
    queued = [
        resolve_trace.queue_resolve_trace(
            trace_id=_TRACE_ID,
            inp=make_request("aid to Kenya").input,
            response=None,
            error_code=None,
            surface="website",
        )[0]
        for _ in range(resolve_trace.PENDING_CAP)
    ]

    logged = _run_line(make_request("aid to Kenya"), ResolveRecord(), surface=DeliverySurface.mcp)

    assert set(queued) == {"queued"}
    assert logged["trace"] == "dropped"


@pytest.mark.parametrize(
    ("export_s", "outcome"),
    [
        pytest.param(0.1, "failed", id="export_within_the_wait"),
        pytest.param(None, "timed_out", id="export_past_the_wait"),
    ],
)
def test_the_request_awaits_its_export_up_to_the_cap_without_blocking_the_loop(
    monkeypatch, held_worker, export_s, outcome
):
    """The held client raises once released, so an export the request waited for reads
    `failed`, and one it gave up on reads `timed_out`."""
    monkeypatch.setattr(resolve_trace, "EXPORT_WAIT_S", 0.5)
    if export_s is not None:
        threading.Timer(export_s, held_worker.set).start()
    ticks = 0

    async def _tick() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    async def _run() -> tuple[int, float]:
        ticker = asyncio.create_task(_tick())
        await asyncio.sleep(0)
        before, started = ticks, time.monotonic()
        async with resolve_line(
            make_request("aid to Kenya"),
            ResolveRecord(),
            caller=None,
            transport="buffered",
            served_code=_served_code,
        ):
            pass
        during, waited = ticks - before, time.monotonic() - started
        ticker.cancel()
        return during, waited

    trace_token = set_trace_id(_TRACE_ID)
    identity_token = set_delivery_identity(
        replace(UNRESOLVED_IDENTITY, surface=DeliverySurface.website)
    )
    try:
        with resolve_log_lines() as lines:
            during, waited = asyncio.run(_run())
    finally:
        reset_delivery_identity(identity_token)
        reset_trace_id(trace_token)

    [logged] = lines
    assert logged["trace"] == outcome
    assert waited < 1.0
    # The loop kept running other coroutines while the request waited.
    assert during > 0


async def _never() -> None:
    await asyncio.Event().wait()


@pytest.mark.parametrize("cancelled", ["in_block", "in_wait", "closed_body"])
def test_a_cancelled_request_queues_its_trace_without_waiting(monkeypatch, held_worker, cancelled):
    monkeypatch.setattr(resolve_trace, "EXPORT_WAIT_S", 30.0)
    record = ResolveRecord()

    def _line():
        return resolve_line(
            make_request("aid to Kenya"),
            record,
            caller=None,
            transport="sse",
            served_code=_served_code,
        )

    async def _request() -> None:
        if cancelled == "closed_body":

            async def _stream():
                no_data = NoDataPart(status="no_data", no_data=NoData(reason="entity_not_resolved"))
                yield ResponseEvent(
                    response=Resolution(
                        query_echo=make_query_echo("aid to Kenya", extract_skipped=False),
                        diagnostics=make_diagnostics("test-build", [], {}, 5),
                        parts=[no_data],
                    )
                )
                await _never()

            body = _event_source(
                _stream(), hard_deadline=time.monotonic() + 60, record=record, line=_line()
            )
            await anext(body)
            await body.aclose()
            return
        async with _line():
            if cancelled == "in_block":
                await _never()

    async def _run() -> None:
        task = asyncio.create_task(_request())
        if cancelled != "closed_body":
            # Long enough to reach the block, or the wait past it.
            await asyncio.sleep(0.05)
            task.cancel()
        # A request that waited for the held export would hang here.
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2)

    trace_token = set_trace_id(_TRACE_ID)
    identity_token = set_delivery_identity(
        replace(UNRESOLVED_IDENTITY, surface=DeliverySurface.website)
    )
    try:
        with resolve_log_lines() as lines:
            asyncio.run(_run())
    finally:
        reset_delivery_identity(identity_token)
        reset_trace_id(trace_token)

    [logged] = lines
    assert logged["trace"] == "queued"


@pytest.mark.parametrize("transport", ["buffered", "sse"])
def test_a_failing_langfuse_client_leaves_the_resolve_standing(monkeypatch, transport):
    for name, value in _ON.items():
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
    assert logged["trace"] == "failed"
    assert calls == ["client"]
