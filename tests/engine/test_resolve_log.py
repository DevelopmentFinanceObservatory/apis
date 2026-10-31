"""Pins the outcome fields of the one `resolve` log line `_log_resolve` writes per request."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import unicodedata

import pytest

from dfo_platform.trace import get_trace_id
from search.engine.core import RequestBudget, _log_resolve, _ResolveRecord, resolve_events
from search.engine.envelope import make_diagnostics, make_query_echo, now_ms
from search.engine.errors import EngineInfraError, EngineInputError
from search.engine.events import DoneEvent, DoneTelemetry
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

_USAGE = {"input_tokens": 120, "output_tokens": 30, "cached_tokens": 0}
# The response echoes this word on every row, and the line must not.
_ECHOED_WORD = "fertility"
_HARD_AHEAD = time.monotonic() + 3600


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
        _log_resolve(request, record, caller=None, transport="buffered", hard=_HARD_AHEAD)
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


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


_RAW = make_request(f"  {_ECHOED_WORD.title()} Rate in KENYA ")
_RAW_SHA = _sha(f"{_ECHOED_WORD} rate in kenya")
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
    query_sha256: str | None = _RAW_SHA,
):
    return pytest.param(
        family,
        outcome,
        list(part_statuses),
        list(no_data_reasons),
        list(unsupported_codes),
        done,
        request,
        query_sha256,
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
        "query_sha256",
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
        _row("parsed_door", "no_response", None, request=_PARSED, query_sha256=None),
        _row("resubmit_door", "no_response", None, request=_RESUBMIT, query_sha256=None),
        _row(
            "inner_whitespace",
            "no_response",
            None,
            request=make_request("aid  to\tKenya"),
            query_sha256=_sha("aid to kenya"),
        ),
        _row(
            "nbsp",
            "no_response",
            None,
            request=make_request("aid\u00a0to Kenya"),
            query_sha256=_sha("aid to kenya"),
        ),
        _row(
            "nfd",
            "no_response",
            None,
            request=make_request(unicodedata.normalize("NFD", "Côte d'Ivoire")),
            query_sha256=_sha("côte d'ivoire"),
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
    query_sha256,
):
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

    assert logged["trace_id"] == get_trace_id()
    assert logged["outcome"] == outcome
    assert logged["part_statuses"] == part_statuses
    assert logged["no_data_reasons"] == no_data_reasons
    assert logged["unsupported_codes"] == unsupported_codes
    assert logged["llm_usage"] == (_USAGE if response is not None else None)
    assert logged["error_code"] is None
    assert logged["timed_out"] == (done.timed_out if done is not None else None)
    assert logged["degraded_leg"] == (done.degraded_leg if done is not None else None)
    assert logged["query_sha256"] == query_sha256
    assert _ECHOED_WORD not in json.dumps(logged).casefold()


@pytest.mark.parametrize(
    ("emits", "raises", "hard_in", "error_code", "timed_out"),
    [
        pytest.param(
            (),
            EngineInputError("no such door", code="door_not_implemented", status=501),
            60,
            "door_not_implemented",
            None,
            id="refused",
        ),
        pytest.param(
            (), EngineInfraError("graph down"), 60, "service_unavailable", None, id="infra"
        ),
        pytest.param((), RuntimeError("boom"), 60, "internal_error", None, id="unhandled"),
        pytest.param((), TimeoutError("past the deadline"), 60, "timeout", True, id="soft_failure"),
        pytest.param((), None, -1, "timeout", True, id="hard_cap"),
        pytest.param((), None, 60, "cancelled", None, id="disconnect"),
        pytest.param((_done(),), None, -1, None, False, id="teardown_after_done"),
    ],
)
def test_the_resolve_line_says_how_a_cut_request_ended(
    monkeypatch, emits, raises, hard_in, error_code, timed_out
):
    """A request that raises ends with it; one that does not stalls until it is cancelled."""
    reached_the_end = asyncio.Event()

    async def _pipeline(*_args, **_kwargs):
        for event in emits:
            yield event
        if raises is not None:
            raise raises
        reached_the_end.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("search.engine.core._resolve_pipeline_events", _pipeline)
    now = time.monotonic()
    budget = RequestBudget(start_ms=now_ms(), soft=now, hard=now + hard_in)
    graph = FakeGraph()

    async def _drain() -> None:
        async for _event in resolve_events(
            make_request("q"),
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            budget=budget,
        ):
            pass

    async def _run() -> None:
        stream = asyncio.create_task(_drain())
        await asyncio.wait(
            {stream, asyncio.create_task(reached_the_end.wait())},
            return_when=asyncio.FIRST_COMPLETED,
        )
        stream.cancel()
        await asyncio.gather(stream, return_exceptions=True)

    with resolve_log_lines() as lines:
        asyncio.run(_run())

    [logged] = lines
    assert logged["error_code"] == error_code
    assert logged["timed_out"] is timed_out


def test_a_cancellation_after_the_response_keeps_the_outcome_on_the_line():
    record = _ResolveRecord(response=_resolution([_NO_DATA]), raised=asyncio.CancelledError())

    logged = _logged(_RAW, record)

    assert (logged["outcome"], logged["error_code"]) == ("no_data", "cancelled")
