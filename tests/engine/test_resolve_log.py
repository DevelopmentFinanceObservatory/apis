"""Pins the outcome fields of the one `resolve` log line `_log_resolve` writes per request."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import unicodedata
from collections.abc import Iterator

import pytest
from pydantic import SecretStr

from dfo_platform.trace import reset_trace_id, set_trace_id
from search.engine.core import _log_resolve, _ResolveRecord, resolve_events
from search.engine.envelope import make_diagnostics, make_query_echo
from search.engine.errors import EngineInfraError, EngineInputError
from search.engine.events import DoneEvent, DoneTelemetry, ResponseEvent
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
from tests.engine._harness import make_request, resolve_log_lines
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

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


def _logged(request: ResolveRequest, record: _ResolveRecord) -> dict:
    """The JSON object of the one line `_log_resolve` writes for `record`."""
    with resolve_log_lines() as lines:
        _log_resolve(request, record, caller=None, transport="buffered")
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
    record = _ResolveRecord(response=response, done=done)

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


_NEVER = 60.0
_SOON = 0.01


def _ending(
    id: str,
    error_code: str | None,
    timed_out: bool | None,
    *,
    raises: Exception | None = None,
    cap_in: float = _NEVER,
    responds: bool = False,
    done: bool = False,
    sse: bool = False,
    unwinds_s: float = 0,
):
    return pytest.param(
        error_code, timed_out, raises, cap_in, responds, done, sse, unwinds_s, id=id
    )


@pytest.mark.parametrize(
    (
        "error_code",
        "timed_out",
        "raises",
        "cap_in",
        "responds",
        "done",
        "sse",
        "unwinds_s",
    ),
    [
        _ending(
            "refused",
            "door_not_implemented",
            None,
            raises=EngineInputError("no such door", code="door_not_implemented", status=501),
        ),
        _ending("infra", "service_unavailable", None, raises=EngineInfraError("graph down")),
        _ending("unhandled", "internal_error", None, raises=RuntimeError("boom")),
        _ending("soft_deadline_failure", "timeout", True, raises=TimeoutError("past it")),
        _ending("hard_cap", "timeout", True, cap_in=_SOON),
        _ending("disconnect", "cancelled", None),
        _ending("disconnect_after_response", "cancelled", None, responds=True),
        _ending("disconnect_crossing_the_cap", "cancelled", None, cap_in=0.02, unwinds_s=0.5),
        _ending("sse_cap_after_done", None, False, cap_in=_SOON, done=True, sse=True),
        _ending("sse_teardown_after_done", None, False, done=True, sse=True),
        _ending(
            "sse_failure_after_done", None, False, raises=RuntimeError("boom"), done=True, sse=True
        ),
        _ending("buffered_cap_after_done", "timeout", True, cap_in=_SOON, done=True),
        _ending(
            "buffered_failure_after_done",
            "internal_error",
            False,
            raises=RuntimeError("boom"),
            done=True,
        ),
    ],
)
def test_the_resolve_line_says_how_a_cut_request_ended(
    monkeypatch, error_code, timed_out, raises, cap_in, responds, done, sse, unwinds_s
):
    """A request that raises ends with it. One that does not stalls until the cap fires or,
    when no cap is due, its task is cancelled."""
    stalled = asyncio.Event()

    async def _pipeline(*_args, **_kwargs):
        if responds:
            yield ResponseEvent(response=_resolution([_NO_DATA]))
        if done:
            yield _done()
        if raises is not None:
            raise raises
        stalled.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(unwinds_s)

    monkeypatch.setattr("search.engine.core._resolve_pipeline_events", _pipeline)
    graph = FakeGraph()

    async def _capped() -> None:
        async with asyncio.timeout(cap_in) as cap:
            async for _event in resolve_events(
                make_request("q"),
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(),
                stream_slot_options=sse,
                hard_cap=cap,
            ):
                pass

    async def _run() -> None:
        request = asyncio.create_task(_capped())
        await asyncio.wait(
            {request, asyncio.create_task(stalled.wait())},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cap_in != _SOON:
            request.cancel()
        await asyncio.gather(request, return_exceptions=True)

    with resolve_log_lines() as lines:
        asyncio.run(_run())

    [logged] = lines
    assert logged["error_code"] == error_code
    assert logged["timed_out"] is timed_out
    assert logged["outcome"] == ("no_data" if responds else None)


def test_a_request_closed_at_a_yield_logs_after_the_pipeline_finished(monkeypatch):
    """Closing the stream closes the pipeline first, so its `finally` reaches the line."""
    events: list[str] = []

    async def _pipeline(*_args, record, **_kwargs):
        try:
            yield _done()
        finally:
            record.cache = "miss"
            events.append("pipeline")

    monkeypatch.setattr("search.engine.core._resolve_pipeline_events", _pipeline)
    graph = FakeGraph()

    async def _run() -> None:
        stream = resolve_events(
            make_request("q"),
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
        )
        await anext(stream)
        await stream.aclose()

    with resolve_log_lines() as lines:
        asyncio.run(_run())

    [logged] = lines
    assert logged["cache"] == "miss"


def test_a_request_cancelled_in_the_cache_lookup_reports_its_cache(monkeypatch, search_settings):
    search_settings(response_cache_enabled=True)
    in_lookup = asyncio.Event()

    async def _stalled_lookup(self, query):
        in_lookup.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(CacheScope, "lookup_raw", _stalled_lookup)
    graph = FakeGraph()

    async def _drain() -> None:
        async for _event in resolve_events(
            make_request("q"),
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            response_cache=ResponseCache(),
        ):
            pass

    async def _run() -> None:
        request = asyncio.create_task(_drain())
        await in_lookup.wait()
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)

    with resolve_log_lines() as lines:
        asyncio.run(_run())

    [logged] = lines
    assert logged["cache"] == "miss"
    assert logged["error_code"] == "cancelled"
