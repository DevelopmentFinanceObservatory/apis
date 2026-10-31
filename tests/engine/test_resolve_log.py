"""Pins the outcome fields of the one `resolve` log line `_log_resolve` writes per request."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import unicodedata
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from dfo_platform.trace import reset_trace_id, set_trace_id
from search.engine.app import _event_source, _served_code
from search.engine.core import ResolveRecord, _log_resolve, resolve_events, resolve_line
from search.engine.envelope import make_diagnostics, make_query_echo
from search.engine.errors import EngineInfraError, EngineInputError
from search.engine.events import DoneEvent, DoneTelemetry, ResponseEvent, serialize_sse
from search.engine.response_cache import ResponseCache
from search.engine.response_cache_flow import CacheScope
from search.engine.response_cache_keys import normalise_query
from search.models import (
    RESOLVE_RESPONSE_ADAPTER,
    CatalogueAnswer,
    DatasetResubmitInput,
    DeclinedAnswer,
    NoData,
    NoDataPart,
    ParsedInput,
    ParsedPart,
    Resolution,
    ResolveRequest,
    ResubmitPlaceholderEntry,
    Scope,
    Unsupported,
    UnsupportedPart,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import EMPTY_SDMX, make_request, parse_sse, resolve_log_lines
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_RAW_BODY = {"input": {"kind": "raw_text", "query": "fertility rate in Kenya"}}
_KEY = "query-key"
_TRACE_ID = "abc123"
_USAGE = {"input_tokens": 120, "output_tokens": 30, "cached_tokens": 0}
# The response echoes this word on every row, and the line must not.
_ECHOED_WORD = "fertility"


@pytest.fixture(autouse=True)
def _bound_trace_id() -> Iterator[None]:
    """Bind the id these tests read, so none stays bound for later tests in the worker."""
    token = set_trace_id(_TRACE_ID)
    yield
    reset_trace_id(token)


def _done(*, timed_out: bool = False, degraded_leg: bool = False) -> DoneEvent:
    return DoneEvent(
        elapsed_ms=5,
        timed_out=timed_out,
        degraded_leg=degraded_leg,
        trace_id="abc123",
        telemetry=DoneTelemetry(),
    )


def _logged(request: ResolveRequest, record: ResolveRecord) -> dict:
    """The JSON object of the one line `_log_resolve` writes for `record`."""
    with resolve_log_lines() as lines:
        _log_resolve(request, record, caller=None, transport="buffered", trace_id=_TRACE_ID)
    [line] = lines
    return line


def _definite_part(worked_example: dict):
    response = RESOLVE_RESPONSE_ADAPTER.validate_python(worked_example)
    assert isinstance(response, Resolution)
    return response.parts[0]


_NO_DATA = NoDataPart(status="no_data", no_data=NoData(reason="entity_not_resolved"))
_UNSUPPORTED = UnsupportedPart(
    status="unsupported",
    unsupported=Unsupported(
        code="unit_request_unsupported", pointer="/identity", detail="per capita", definition={}
    ),
)


def _diagnostics():
    return make_diagnostics("test-build", [], {}, 5, llm_usage=_USAGE)


def _echo():
    return make_query_echo(f"{_ECHOED_WORD} rate in Kenya", extract_skipped=False)


def _resolution(parts: list) -> Resolution:
    return Resolution(query_echo=_echo(), diagnostics=_diagnostics(), parts=parts)


def _hmac(text: str, *, key: str = _KEY) -> str:
    return hmac.new(key.encode(), text.encode(), hashlib.sha256).hexdigest()


_RAW = make_request(f"  {_ECHOED_WORD.title()} Rate in KENYA ")
_RAW_HMAC = _hmac(f"{_ECHOED_WORD} rate in kenya")
_PARSED = ResolveRequest(input=ParsedInput(parts=[ParsedPart(variable_text=_ECHOED_WORD)]))
_RESUBMIT = ResolveRequest(
    input=DatasetResubmitInput(
        entries=[
            ResubmitPlaceholderEntry(
                variable_text=_ECHOED_WORD, status="no_data", reason="variable_not_resolved"
            )
        ]
    )
)


def _row(
    id: str,
    family: str,
    outcome: str | None,
    *,
    part_statuses: tuple[str, ...] = (),
    no_data_reasons: tuple[str, ...] = (),
    unsupported_codes: tuple[str, ...] = (),
    done: DoneEvent | None = None,
    request: ResolveRequest = _RAW,
    key: str = _KEY,
    query_hmac: str | None = _RAW_HMAC,
):
    return pytest.param(
        family,
        outcome,
        list(part_statuses),
        list(no_data_reasons),
        list(unsupported_codes),
        done,
        request,
        key,
        query_hmac,
        id=id,
    )


@pytest.mark.parametrize(
    (
        "family",
        "outcome",
        "part_statuses",
        "no_data_reasons",
        "unsupported_codes",
        "done",
        "request_",
        "key",
        "query_hmac",
    ),
    [
        _row("definite", "definite", "definite", part_statuses=("definite",), done=_done()),
        _row(
            "mixed",
            "mixed",
            "mixed",
            part_statuses=("definite", "no_data", "unsupported"),
            no_data_reasons=("entity_not_resolved",),
            unsupported_codes=("unit_request_unsupported",),
        ),
        _row(
            "no_data",
            "no_data",
            "no_data",
            part_statuses=("no_data",),
            no_data_reasons=("entity_not_resolved",),
        ),
        _row(
            "degraded",
            "no_data",
            "no_data",
            part_statuses=("no_data",),
            no_data_reasons=("entity_not_resolved",),
            done=_done(degraded_leg=True),
        ),
        _row(
            "soft_timeout",
            "no_data",
            "no_data",
            part_statuses=("no_data",),
            no_data_reasons=("entity_not_resolved",),
            done=_done(timed_out=True),
        ),
        _row("declined", "declined", "declined"),
        _row("catalogue", "catalogue", "catalogue"),
        _row("no_response", "no_response", None),
        _row(
            "another_key",
            "no_response",
            None,
            key="another-key",
            query_hmac=_hmac(f"{_ECHOED_WORD} rate in kenya", key="another-key"),
        ),
        _row("no_key", "no_response", None, key="", query_hmac=None),
        _row("parsed_door", "no_response", None, request=_PARSED, query_hmac=None),
        _row("resubmit_door", "no_response", None, request=_RESUBMIT, query_hmac=None),
        _row(
            "inner_whitespace",
            "no_response",
            None,
            request=make_request("aid  to\tKenya"),
            query_hmac=_hmac("aid to kenya"),
        ),
        _row(
            "nbsp",
            "no_response",
            None,
            request=make_request("aid\u00a0to Kenya"),
            query_hmac=_hmac("aid to kenya"),
        ),
        _row(
            "nfd",
            "no_response",
            None,
            request=make_request(unicodedata.normalize("NFD", "Côte d'Ivoire")),
            query_hmac=_hmac("côte d'ivoire"),
        ),
    ],
)
def test_the_resolve_line_reports_the_outcome_of_each_answer_family(
    worked_example_response,
    family,
    outcome,
    part_statuses,
    no_data_reasons,
    unsupported_codes,
    done,
    request_,
    key,
    query_hmac,
    search_settings,
):
    search_settings(query_hmac_key=SecretStr(key))
    definite = _definite_part(worked_example_response)
    response = {
        "definite": _resolution([definite]),
        "mixed": _resolution([definite, _NO_DATA, _UNSUPPORTED]),
        "no_data": _resolution([_NO_DATA]),
        "declined": DeclinedAnswer(
            query_echo=_echo(),
            diagnostics=_diagnostics(),
            reason="out_of_scope",
            scope=Scope(),
        ),
        "catalogue": CatalogueAnswer(query_echo=_echo(), diagnostics=_diagnostics(), scope=Scope()),
        "no_response": None,
    }[family]
    record = ResolveRecord(response=response, done=done)

    logged = _logged(request_, record)

    assert logged["trace_id"] == _TRACE_ID
    assert logged["outcome"] == outcome
    assert logged["part_statuses"] == part_statuses
    assert logged["no_data_reasons"] == no_data_reasons
    assert logged["unsupported_codes"] == unsupported_codes
    assert logged["llm_usage"] == (_USAGE if response is not None else None)
    assert logged["error_code"] is None
    assert logged["timed_out"] == (done.timed_out if done is not None else None)
    assert logged["degraded_leg"] == (done.degraded_leg if done is not None else None)
    assert logged["query_hmac"] == query_hmac
    assert _ECHOED_WORD not in json.dumps(logged).casefold()
    bare_text = normalise_query(getattr(request_.input, "query", "")).casefold()
    assert hashlib.sha256(bare_text.encode()).hexdigest() not in json.dumps(logged)


def _client() -> TestClient:
    graph = FakeGraph()
    app = search_app(graph=graph, resolver=make_resolver(graph), llm=FakeLLM(), sdmx=EMPTY_SDMX)
    return TestClient(app, raise_server_exceptions=False)


_BOOM = RuntimeError("boom")
_NOT_RAN = EngineInputError("no such door", code="door_not_implemented", status=501)


@pytest.mark.parametrize("transport", ["buffered", "sse"])
@pytest.mark.parametrize(
    ("raises", "error_code", "timed_out"),
    [
        pytest.param(_NOT_RAN, "door_not_implemented", None, id="refused"),
        pytest.param(EngineInfraError("down"), "service_unavailable", None, id="infra"),
        pytest.param(_BOOM, "internal_error", None, id="unhandled"),
        pytest.param(
            EngineInputError("x", code="not_a_code"), "internal_error", None, id="unknown"
        ),
        pytest.param(TimeoutError("past it"), "timeout", True, id="soft_deadline_failure"),
    ],
)
def test_the_resolve_line_names_the_code_the_route_served(
    monkeypatch, transport, raises, error_code, timed_out
):
    async def _pipeline(*_args, **_kwargs):
        raise raises
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr("search.engine.core._resolve_pipeline_events", _pipeline)
    headers = {"Accept": "text/event-stream"} if transport == "sse" else {}

    with resolve_log_lines() as lines:
        resp = _client().post("/api/search/resolve", json=_RAW_BODY, headers=headers)

    [logged] = lines
    assert logged["transport"] == transport
    assert logged["error_code"] == error_code
    assert logged["timed_out"] is timed_out
    served = parse_sse(resp.text)[-1][1] if transport == "sse" else resp.json()
    assert served["code"] == error_code


@pytest.mark.parametrize(
    ("transport", "after_done", "status", "error_code", "timed_out"),
    [
        pytest.param("buffered", "raises", 500, "internal_error", False, id="buffered_failure"),
        pytest.param("buffered", "stalls", 504, "timeout", True, id="buffered_hard_cap"),
        pytest.param("sse", "raises", 200, None, False, id="sse_failure"),
        pytest.param("sse", "stalls", 200, None, False, id="sse_hard_cap"),
    ],
)
def test_what_follows_done_ends_the_request_only_where_the_client_lacks_its_answer(
    monkeypatch, search_settings, transport, after_done, status, error_code, timed_out
):
    """An SSE client already holds `done`. A buffered client holds nothing yet."""
    search_settings(request_timeout_s=0.05, finalise_grace_s=0.05)

    async def _pipeline(*_args, **_kwargs):
        yield ResponseEvent(response=_resolution([_NO_DATA]))
        yield _done()
        if after_done == "raises":
            raise _BOOM
        await asyncio.sleep(5)

    monkeypatch.setattr("search.engine.core._resolve_pipeline_events", _pipeline)
    headers = {"Accept": "text/event-stream"} if transport == "sse" else {}

    with resolve_log_lines() as lines:
        resp = _client().post("/api/search/resolve", json=_RAW_BODY, headers=headers)

    assert resp.status_code == status
    [logged] = lines
    assert logged["outcome"] == "no_data"
    assert logged["error_code"] == error_code
    assert logged["timed_out"] is timed_out


def test_a_failure_in_the_route_s_own_work_after_the_pipeline_reads_on_the_line(monkeypatch):
    def _failing_response(*_args, **_kwargs):
        raise _BOOM

    monkeypatch.setattr("search.engine.app.JSONResponse", _failing_response)

    with resolve_log_lines() as lines:
        resp = _client().post("/api/search/resolve", json=_RAW_BODY)

    assert resp.status_code == 500
    [logged] = lines
    assert logged["outcome"] is not None
    assert logged["error_code"] == "internal_error"


def test_a_failure_classification_cannot_finish_still_reads_as_a_failure(monkeypatch):
    def _broken(_exc):
        raise ValueError("no classification")

    async def _pipeline(*_args, **_kwargs):
        raise _BOOM
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr("search.engine.app.served_problem", _broken)
    monkeypatch.setattr("search.engine.core._resolve_pipeline_events", _pipeline)

    with resolve_log_lines() as lines:
        resp = _client().post("/api/search/resolve", json=_RAW_BODY)

    assert resp.status_code == 500
    [logged] = lines
    assert logged["error_code"] == "internal_error"


@pytest.mark.parametrize("transport", ["buffered", "sse"])
def test_a_log_line_that_raises_leaves_the_response_standing(monkeypatch, transport):
    def _raising_log_line(*_args, **_kwargs):
        raise ValueError("simulated log failure")

    monkeypatch.setattr("search.engine.core._log_resolve", _raising_log_line)
    headers = {"Accept": "text/event-stream"} if transport == "sse" else {}

    resp = _client().post("/api/search/resolve", json=_RAW_BODY, headers=headers)

    assert resp.status_code == 200
    if transport == "sse":
        assert parse_sse(resp.text)[-1][0] == "done"


def test_a_frame_that_cannot_serialise_ends_the_stream_with_the_error_it_logs(monkeypatch):
    def _failing_serialize(event):
        if event.kind == "interpretation":
            raise ValueError("no frame")
        return serialize_sse(event)

    monkeypatch.setattr("search.engine.app.serialize_sse", _failing_serialize)

    with resolve_log_lines() as lines:
        resp = _client().post(
            "/api/search/resolve", json=_RAW_BODY, headers={"Accept": "text/event-stream"}
        )

    assert parse_sse(resp.text)[-1][1]["code"] == "internal_error"
    [logged] = lines
    assert logged["error_code"] == "internal_error"


def test_a_client_gone_before_done_left_reads_cancelled_under_the_request_trace_id():
    """The pipeline finished, but `done` never left. The closing body runs outside the
    request's context, as an abandoned response body does."""
    record = ResolveRecord()

    async def _stream():
        record.response = _resolution([_NO_DATA])
        yield ResponseEvent(response=record.response)
        record.done = _done()
        yield record.done

    async def _run() -> None:
        body = _event_source(
            _stream(),
            hard_deadline=time.monotonic() + 60,
            record=record,
            line=resolve_line(_RAW, record, caller=None, transport="sse", served_code=_served_code),
        )
        await anext(body)
        await asyncio.sleep(0)
        token = set_trace_id("another-request")
        try:
            await body.aclose()
        finally:
            reset_trace_id(token)

    with resolve_log_lines() as lines:
        asyncio.run(_run())

    [logged] = lines
    assert logged["error_code"] == "cancelled"
    assert logged["trace_id"] == _TRACE_ID


def test_a_client_gone_once_done_left_reads_clean_while_the_pipeline_unwinds():
    record = ResolveRecord()
    unwinding = asyncio.Event()

    async def _stream():
        record.done = _done()
        try:
            yield record.done
        finally:
            unwinding.set()
            await asyncio.sleep(60)

    async def _drain() -> None:
        body = _event_source(
            _stream(),
            hard_deadline=time.monotonic() + 60,
            record=record,
            line=resolve_line(_RAW, record, caller=None, transport="sse", served_code=_served_code),
        )
        async for _frame in body:
            pass

    async def _run() -> None:
        task = asyncio.create_task(_drain())
        await unwinding.wait()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with resolve_log_lines() as lines:
        asyncio.run(_run())

    [logged] = lines
    assert logged["error_code"] is None


def test_a_stream_closed_at_a_yield_closes_the_pipeline_first(monkeypatch):
    """A consumer's close runs the pipeline's `finally` before the stream's own cleanup, and
    is no ending of the request."""
    closed: list[str] = []

    async def _pipeline(*_args, record, **_kwargs):
        try:
            yield _done()
        finally:
            record.cache = "miss"
            closed.append("pipeline")

    class _ClosingGraph(FakeGraph):
        def close(self):
            closed.append("graph")

    class _OwnGraph:
        @staticmethod
        def from_search_settings(*_args, **_kwargs):
            return _ClosingGraph()

    monkeypatch.setattr("search.engine.core._resolve_pipeline_events", _pipeline)
    monkeypatch.setattr("search.engine.core.LiveGraphClient", _OwnGraph)
    record = ResolveRecord()

    async def _run() -> None:
        stream = resolve_events(
            make_request("q"),
            resolver=make_resolver(FakeGraph()),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            record=record,
        )
        await anext(stream)
        await stream.aclose()

    asyncio.run(_run())

    assert closed == ["pipeline", "graph"]
    assert (record.cache, record.error_code) == ("miss", None)


def test_a_request_cancelled_in_the_cache_lookup_reports_its_cache(monkeypatch, search_settings):
    search_settings(response_cache_enabled=True)
    in_lookup = asyncio.Event()

    async def _stalled_lookup(self, query):
        in_lookup.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(CacheScope, "lookup_raw", _stalled_lookup)
    graph = FakeGraph()
    record = ResolveRecord()

    async def _drain() -> None:
        async for _event in resolve_events(
            make_request("q"),
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            response_cache=ResponseCache(),
            record=record,
        ):
            pass

    async def _run() -> None:
        request = asyncio.create_task(_drain())
        await in_lookup.wait()
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)

    asyncio.run(_run())

    assert record.cache == "miss"
