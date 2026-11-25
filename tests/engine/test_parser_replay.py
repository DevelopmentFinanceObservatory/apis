"""The private parser replay: its gate, the seam that carries its extraction client, the
clients' call records, and the log privacy of a replay request.

Async calls use `asyncio.run` because the suite has no pytest-asyncio plugin.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing, asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast, override

import httpx
import pytest
from fastapi import Request
from pydantic import SecretStr

import search.engine.app as app_module
from dfo_platform.caller import UNRESOLVED_IDENTITY
from dfo_platform.cloud_logging import ReplayLogFilter
from search.engine import config
from search.engine.app import _event_source, _served_code, resolve
from search.engine.binders import Fallbacks
from search.engine.clients.llm import (
    LLM,
    Agreement,
    MistralClients,
    MistralLLM,
    ReplayGeminiLLM,
    Sampled,
    SecondSamples,
    SupportsLLM,
    agreement_verdict,
    held_by,
    reseeded,
)
from search.engine.core import (
    ResolveRecord,
    _log_resolve,
    _resolve_pipeline_events,
    _settle_replay_agreement,
    resolve_line,
)
from search.engine.degraded_reads import DegradedReads
from search.engine.entry import raw_text
from search.engine.errors import EngineInfraError, EngineInputError, LLMInfraError
from search.engine.events import DoneEvent, DoneTelemetry, Event, StepEvent
from search.engine.exclusion_read import Exclusion, ExclusionRead
from search.engine.extract import Extraction
from search.engine.parser_replay import (
    FLASH_LITE_MODEL,
    MISTRAL_REPLAY_INSTRUCTION,
    PARSER_ARM_HEADER,
    PARSER_CACHE_HEADER,
    PARSER_KEY_HEADER,
    PARSER_REPLAY_HEADER,
    PARSER_STREAM_HEADER,
    Arm,
    MistralReplayExtraction,
    ParserReplay,
    ReplayRecord,
    parser_replay,
)
from search.engine.resolve_trace import TraceOutcome
from search.models import ResolveRequest
from tests.engine._harness import resolve_log_lines

_QUERY = "ODA to Kenya excluding grants in 2022"
_ANSWER = Extraction(kind="resolution", variables=["ODA"]).model_dump_json()


def _request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/search/resolve",
            "query_string": b"",
            "headers": [(name.lower().encode(), value.encode()) for name, value in headers.items()],
        }
    )


def _enable_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K_SERVICE", "dfo-dev-apis")
    monkeypatch.setenv("K_REVISION", "dfo-dev-apis-00042-test")
    monkeypatch.setattr(
        config,
        "_SETTINGS",
        config.settings().model_copy(
            update={
                "parser_replay_enabled": True,
                "parser_replay_key": SecretStr("capability"),
                "parser_replay_revision": "dfo-dev-apis-00042-test",
            }
        ),
    )


def _headers(*, arm: str = "flash-lite") -> dict[str, str]:
    return {
        PARSER_REPLAY_HEADER: "run_01",
        PARSER_ARM_HEADER: arm,
        PARSER_KEY_HEADER: "capability",
    }


def _record(*, arm: Arm = "flash-lite", stream: bool = False) -> ReplayRecord:
    return ReplayRecord(
        ParserReplay(replay_id="run_01", arm=arm, stream=stream),
        query=_QUERY,
    )


# -- Gate ---------------------------------------------------------------------------------


def test_no_replay_headers_select_no_replay():
    assert parser_replay(_request({})) is None


def test_replay_headers_fail_closed_when_the_gate_is_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("K_SERVICE", "dfo-dev-apis")

    with pytest.raises(EngineInputError) as error:
        parser_replay(_request(_headers()))

    assert (error.value.status, _served_code(error.value)) == (403, "request_invalid")


def test_replay_headers_fail_closed_outside_the_dev_service(monkeypatch: pytest.MonkeyPatch):
    _enable_replay(monkeypatch)
    monkeypatch.setenv("K_SERVICE", "dfo-apis")

    with pytest.raises(EngineInputError) as error:
        parser_replay(_request(_headers()))

    assert (error.value.status, _served_code(error.value)) == (403, "request_invalid")


@pytest.mark.parametrize("revision", [None, "dfo-dev-apis-00043-inherited"])
def test_replay_headers_fail_closed_outside_the_pinned_dev_revision(
    monkeypatch: pytest.MonkeyPatch, revision: str | None
):
    _enable_replay(monkeypatch)
    if revision is None:
        monkeypatch.delenv("K_REVISION")
    else:
        monkeypatch.setenv("K_REVISION", revision)

    with pytest.raises(EngineInputError) as error:
        parser_replay(_request(_headers()))

    assert (error.value.status, _served_code(error.value)) == (403, "request_invalid")


@pytest.mark.parametrize("arm", ["flash-lite", "ministral"])
def test_verified_replay_selects_only_a_declared_arm(monkeypatch: pytest.MonkeyPatch, arm: Arm):
    _enable_replay(monkeypatch)
    headers = _headers(arm=arm)
    headers[PARSER_CACHE_HEADER] = "bypass"
    headers[PARSER_STREAM_HEADER] = "1"

    assert parser_replay(_request(headers)) == ParserReplay(
        replay_id="run_01",
        arm=arm,
        stream=True,
    )


@pytest.mark.parametrize(
    ("header", "value"),
    [
        (PARSER_CACHE_HEADER, "isolated"),
        (PARSER_STREAM_HEADER, "yes"),
        (PARSER_ARM_HEADER, "gpt"),
        (PARSER_REPLAY_HEADER, "run 01"),
    ],
)
def test_replay_rejects_an_invalid_option(monkeypatch: pytest.MonkeyPatch, header: str, value: str):
    _enable_replay(monkeypatch)
    headers = _headers() | {header: value}

    with pytest.raises(EngineInputError) as error:
        parser_replay(_request(headers))

    assert (error.value.status, _served_code(error.value)) == (422, "request_invalid")


# -- The seam -------------------------------------------------------------------------------


class _Captured(Exception):
    pass


def _deps() -> SimpleNamespace:
    async def refresh_if_stale() -> None:
        return None

    return SimpleNamespace(
        graph=object(),
        resolver=object(),
        admission=None,
        llm=LLM(),
        sdmx=object(),
        materialise_graph=None,
        materialise_sdmx=None,
        rate_cache=None,
        population_memo=None,
        place_options_pool=None,
        skills_index=None,
        response_cache=object(),
        part_memo=object(),
        generation=SimpleNamespace(refresh_if_stale=refresh_if_stale),
    )


def _route_kwargs(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str], deps: SimpleNamespace
) -> dict[str, Any]:
    """The keywords `resolve` hands the pipeline for a raw-text request with `headers`."""
    captured: dict[str, Any] = {}

    def capture(_body: object, **kwargs: Any) -> Any:
        captured.update(kwargs)
        raise _Captured

    @asynccontextmanager
    async def no_line(*_args: object, **_kwargs: object):
        yield

    monkeypatch.setattr(app_module, "resolve_events", capture)
    monkeypatch.setattr(app_module, "resolve_async", capture)
    monkeypatch.setattr(app_module, "resolve_line", no_line)
    body = ResolveRequest.model_validate({"input": {"kind": "raw_text", "query": _QUERY}})
    with pytest.raises(_Captured):
        asyncio.run(resolve(body, _request(headers), cast(Any, deps)))
    return captured


@pytest.mark.parametrize("accept", [None, "text/event-stream"])
def test_a_request_without_replay_headers_passes_the_service_clients_and_caches(
    monkeypatch: pytest.MonkeyPatch, accept: str | None
):
    deps = _deps()
    headers = {} if accept is None else {"accept": accept}

    kwargs = _route_kwargs(monkeypatch, headers, deps)

    assert kwargs["llm"] is deps.llm
    assert kwargs["extract_llm"] is None
    assert kwargs["response_cache"] is deps.response_cache
    assert kwargs["part_memo"] is deps.part_memo
    assert kwargs["record"].parser_replay is None


@pytest.mark.parametrize(
    ("arm", "client"), [("flash-lite", ReplayGeminiLLM), ("ministral", MistralLLM)]
)
def test_a_replay_passes_its_arms_extraction_client_and_no_cache(
    monkeypatch: pytest.MonkeyPatch, arm: Arm, client: type[LLM]
):
    _enable_replay(monkeypatch)
    monkeypatch.setattr(
        config,
        "_SETTINGS",
        config.settings().model_copy(update={"mistral_api_key": SecretStr("mistral")}),
    )
    deps = _deps()

    kwargs = _route_kwargs(monkeypatch, _headers(arm=arm) | {"accept": "text/event-stream"}, deps)

    extract_llm = kwargs["extract_llm"]
    assert type(extract_llm) is client
    assert (extract_llm._memo, extract_llm.extraction_store) == (None, None)
    assert kwargs["llm"] is not deps.llm
    assert (kwargs["llm"]._memo, kwargs["llm"].extraction_store) == (None, None)
    assert (kwargs["response_cache"], kwargs["part_memo"]) == (None, None)
    assert kwargs["record"].parser_replay.replay.arm == arm


def test_the_flash_lite_arm_pins_its_model(monkeypatch: pytest.MonkeyPatch):
    _enable_replay(monkeypatch)

    kwargs = _route_kwargs(monkeypatch, _headers() | {"accept": "text/event-stream"}, _deps())

    assert kwargs["extract_llm"]._model == FLASH_LITE_MODEL


def test_a_replay_requires_sse(monkeypatch: pytest.MonkeyPatch):
    _enable_replay(monkeypatch)
    body = ResolveRequest.model_validate({"input": {"kind": "raw_text", "query": _QUERY}})

    with pytest.raises(EngineInputError) as error:
        asyncio.run(resolve(body, _request(_headers()), cast(Any, _deps())))

    assert (error.value.status, _served_code(error.value)) == (422, "request_invalid")


def test_a_reseeded_replay_client_keeps_the_request_holder_and_the_recorder():
    record = _record()
    holder = SecondSamples()

    second = reseeded(
        holder.bind(ReplayGeminiLLM(model=FLASH_LITE_MODEL, recorder=record, stream=False))
    )

    assert isinstance(second, ReplayGeminiLLM)
    assert held_by(second) is holder
    assert second._recorder is record


# -- Call records -------------------------------------------------------------------------


def _gemini_response(text: str = _ANSWER) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=11,
            candidates_token_count=4,
            cached_content_token_count=2,
            thoughts_token_count=3,
        ),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
    )


class _GeminiModels:
    def __init__(self, *, fail: Exception | None = None, cancel: bool = False) -> None:
        self.fail = fail
        self.cancel = cancel

    async def generate_content(self, **_kwargs: object) -> SimpleNamespace:
        if self.cancel:
            raise asyncio.CancelledError
        if self.fail is not None:
            raise self.fail
        return _gemini_response()

    async def generate_content_stream(self, **_kwargs: object):
        async def chunks():
            yield SimpleNamespace(text=_ANSWER[:5], usage_metadata=None, candidates=[])
            if self.fail is not None:
                raise self.fail
            yield _gemini_response(_ANSWER[5:])

        return chunks()


class _FakeGemini(ReplayGeminiLLM):
    def __init__(self, record: ReplayRecord, models: _GeminiModels, *, stream: bool) -> None:
        super().__init__(model=FLASH_LITE_MODEL, recorder=record, stream=stream)
        self._fake = SimpleNamespace(aio=SimpleNamespace(models=models))

    @override
    def _client(self) -> Any:
        return self._fake


def _gemini(record: ReplayRecord, models: _GeminiModels, *, stream: bool = False) -> LLM:
    return _FakeGemini(record, models, stream=stream)


@pytest.mark.parametrize("stream", [False, True])
def test_gemini_replay_records_usage_thoughts_finish_and_first_content(stream: bool):
    record = _record(stream=stream)
    llm = _gemini(record, _GeminiModels(), stream=stream)

    async def run() -> tuple[Extraction, dict | None]:
        return await llm.generate_structured(prompt=_QUERY, system="system", schema=Extraction)

    answer, usage = asyncio.run(run())

    assert answer == Extraction.model_validate_json(_ANSWER)
    assert usage == {"input_tokens": 11, "output_tokens": 4, "cached_tokens": 2}
    [call] = record.calls
    assert (call.provider, call.model, call.phase, call.attempt) == (
        "gemini",
        FLASH_LITE_MODEL,
        "original",
        "first",
    )
    assert call.usage == {**usage, "thought_tokens": 3}
    assert call.finish_reason == "STOP"
    assert (call.first_content_ms is not None) is stream
    assert record.prompt_digest is not None and record.schema_digest is not None


def test_gemini_replay_failure_keeps_first_content_and_the_upstream_status():
    class _Upstream(Exception):
        code = 429

    record = _record(stream=True)
    llm = _gemini(record, _GeminiModels(fail=_Upstream("quota for prompt text")), stream=True)

    async def run() -> None:
        await llm.generate_structured(prompt=_QUERY, system="system", schema=Extraction)

    with pytest.raises(LLMInfraError) as error:
        asyncio.run(run())

    assert error.value.__cause__ is None
    [call] = record.calls
    assert (call.error, call.http_status) == ("_Upstream", 429)
    assert call.first_content_ms is not None


def test_gemini_replay_cancellation_is_recorded_then_reraised():
    record = _record()
    llm = _gemini(record, _GeminiModels(cancel=True))

    async def run() -> None:
        await llm.generate_structured(prompt=_QUERY, system="system", schema=Extraction)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())

    [call] = record.calls
    assert (call.error, call.usage) == ("cancelled", None)


def _mistral(
    record: ReplayRecord, handler: Any, *, stream: bool = False
) -> tuple[MistralLLM, list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return handler(request)

    class _Clients(MistralClients):
        @override
        def get(self) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                base_url="https://api.mistral.ai", transport=httpx.MockTransport(respond)
            )

    llm = MistralLLM(
        clients=_Clients("test"),
        schema=MistralReplayExtraction,
        instruction=MISTRAL_REPLAY_INSTRUCTION,
        recorder=record,
        stream=stream,
    )
    return llm, payloads


def _mistral_body(content: str = _ANSWER) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 20},
        },
    }


def _extract(llm: SupportsLLM, prompt: str = _QUERY) -> tuple[Extraction, dict | None]:
    async def run() -> tuple[Extraction, dict | None]:
        return await llm.generate_structured(prompt=prompt, system="system", schema=Extraction)

    return asyncio.run(run())


def test_mistral_sends_its_schema_instruction_and_seed_and_validates_as_extraction():
    record = _record(arm="ministral")
    llm, payloads = _mistral(record, lambda _r: httpx.Response(200, json=_mistral_body()))

    answer, usage = _extract(reseeded(llm))

    assert answer == Extraction.model_validate_json(_ANSWER)
    assert usage == {
        "mistral_input_tokens": 30,
        "mistral_output_tokens": 6,
        "mistral_cached_tokens": 20,
    }
    [payload] = payloads
    assert payload["messages"][0]["content"] == "system" + MISTRAL_REPLAY_INSTRUCTION
    assert payload["messages"][1]["content"] == _QUERY
    assert payload["random_seed"] == (1 if llm._seed is None else llm._seed + 1)
    schema = payload["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert "entities" in schema["schema"]["required"]
    assert "max_tokens" not in payload
    assert "temperature" not in payload
    [call] = record.calls
    assert (call.provider, call.attempt, call.finish_reason) == ("mistral", "second", "stop")


def test_mistral_streams_and_records_first_content():
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": _ANSWER[:5]}}]}),
        "data: "
        + json.dumps(
            {
                "choices": [{"delta": {"content": _ANSWER[5:]}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        ),
        "data: [DONE]",
    ]
    record = _record(arm="ministral", stream=True)
    llm, payloads = _mistral(
        record, lambda _r: httpx.Response(200, text="\n\n".join(lines) + "\n\n"), stream=True
    )

    answer, _usage = _extract(llm)

    assert answer == Extraction.model_validate_json(_ANSWER)
    assert payloads[0]["stream"] is True
    [call] = record.calls
    assert call.first_content_ms is not None
    assert call.finish_reason == "stop"


@pytest.mark.parametrize(
    ("response", "error", "status"),
    [
        (httpx.Response(429, json={"message": "rate limited"}), "HTTPStatusError", 429),
        (httpx.Response(200, json={}), "KeyError", None),
        (httpx.Response(200, json={"choices": []}), "IndexError", None),
        (httpx.Response(200, text="not json"), "JSONDecodeError", None),
        (
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": _ANSWER[:5]}, "finish_reason": "error"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                },
            ),
            "provider_error",
            None,
        ),
    ],
)
def test_a_failed_mistral_call_is_one_record_with_its_class_and_status(
    response: httpx.Response, error: str, status: int | None
):
    record = _record(arm="ministral")
    llm, _payloads = _mistral(record, lambda _r: response)

    with pytest.raises(LLMInfraError) as raised:
        _extract(llm)

    assert raised.value.__cause__ is None
    [call] = record.calls
    assert (call.error, call.http_status) == (error, status)


@pytest.mark.reads_exclusions
def test_the_replay_record_reports_the_served_blanked_phase():
    """A door whose read blanks a stretch discards the parse of the query. The replay
    client records both phases' calls, and the served phase is the blanked one."""

    class _Reader(LLM):
        async def generate_structured(self, *, prompt, system, schema):
            assert schema is ExclusionRead
            read = ExclusionRead(
                exclusions=[Exclusion(phrase="excluding grants", left_out=["grants"])]
            )
            return read, None

    record = _record()

    async def run() -> raw_text._DoorRead:
        holder = SecondSamples()
        try:
            return await raw_text._read_and_extract(
                _QUERY,
                llm=holder.bind(_Reader()),
                extract_llm=holder.bind(_gemini(record, _GeminiModels())),
                deadline=time.monotonic() + 30,
            )
        finally:
            await holder.aclose()

    door = asyncio.run(run())

    assert door.reading.text != _QUERY
    assert "blanked" in {call.phase for call in record.calls}
    assert record.served_phase == "blanked"
    assert record.log_fields()["served_phase"] == "blanked"


@pytest.mark.parametrize(("judge", "expected"), [(True, True), (False, False)])
def test_a_finished_agreement_reads_its_verdict(judge: bool, expected: bool):
    async def run() -> bool | None:
        async def sample() -> Sampled[str]:
            return Sampled("second", None, None)

        agreement = Agreement(sample, stage="extraction")
        agreement.read_first(lambda _second: judge, delay_s=1)
        await agreement.ended()
        return agreement_verdict(agreement)

    assert asyncio.run(run()) is expected


def _agreement(*, second_s: float) -> Agreement:
    """An extraction agreement whose second sample agrees after `second_s`."""

    async def sample() -> Sampled[str]:
        await asyncio.sleep(second_s)
        return Sampled("second", None, None)

    agreement = Agreement(sample, stage="extraction")
    agreement.read_first(lambda _second: True, delay_s=30)
    return agreement


def test_the_pipeline_records_a_replays_agreement_before_it_yields_done(
    monkeypatch: pytest.MonkeyPatch,
):
    """The SSE consumer closes the stream once it holds `done`, so the verdict must be in
    the replay record when `done` arrives."""

    async def handle(*_args: object, scope: Any, **_kwargs: object) -> AsyncGenerator[Event]:
        scope.note_extraction(_agreement(second_s=0.05))
        yield DoneEvent(
            elapsed_ms=5, degraded_leg=False, trace_id="abc123", telemetry=DoneTelemetry()
        )

    monkeypatch.setattr(raw_text, "handle", handle)
    record = ResolveRecord(parser_replay=_record())
    request = ResolveRequest.model_validate({"input": {"kind": "raw_text", "query": _QUERY}})

    async def run() -> bool | None:
        now = time.monotonic()
        events = _resolve_pipeline_events(
            request,
            graph=cast(Any, None),
            resolver=cast(Any, None),
            admission=cast(Any, None),
            llm=cast(Any, None),
            sdmx=cast(Any, None),
            binder=cast(Any, SimpleNamespace(reads_query_text=False)),
            start_ms=0,
            deadline=now + 30,
            hard_deadline=now + 32,
            response_cache=None,
            coverage_snapshot=cast(Any, None),
            warn_index_drift=False,
            record=record,
            fallbacks=Fallbacks(),
            request_reads=DegradedReads(),
        )
        async with aclosing(events):
            async for ev in events:
                if isinstance(ev, DoneEvent):
                    assert record.parser_replay is not None
                    return record.parser_replay.extraction_agreed
        raise AssertionError("the pipeline yielded no done event")

    assert asyncio.run(run()) is True


def test_a_replay_stops_waiting_for_its_agreement_a_grace_before_the_hard_cap():
    """The `done` frame needs `finalise_grace_s` to land before the hard cap ends the
    stream, so an unfinished second sample reads as unjudged by then."""
    grace = config.settings().finalise_grace_s

    async def run() -> tuple[bool | None, float]:
        agreement = _agreement(second_s=30)
        record = _record()
        scope = SimpleNamespace(extraction_agreement=agreement)
        hard_deadline = time.monotonic() + grace + 0.1
        try:
            await _settle_replay_agreement(cast(Any, scope), record, hard_deadline=hard_deadline)
            return record.extraction_agreed, hard_deadline - time.monotonic()
        finally:
            agreement.task.cancel()

    verdict, left = asyncio.run(run())
    assert verdict is None
    assert left > grace / 2


def test_a_replay_line_carries_the_replay_verdict_and_no_query_digest(search_settings):
    search_settings(query_hmac_key=SecretStr("query-key"))
    replay = _record()
    replay.extraction_agreed = False
    request = ResolveRequest.model_validate({"input": {"kind": "raw_text", "query": _QUERY}})

    with resolve_log_lines() as lines:
        _log_resolve(
            request,
            ResolveRecord(parser_replay=replay),
            caller=None,
            transport="sse",
            identity=UNRESOLVED_IDENTITY,
            trace=TraceOutcome.skipped,
        )

    [line] = lines
    assert line["query_hmac"] is None
    assert line["extraction_agreed"] is False
    assert line["parser_replay"]["extraction_agreed"] is False


# -- Log privacy ----------------------------------------------------------------------------


def test_replay_sse_suppresses_the_terminal_error_but_logs_resolve(caplog):
    marker = "private-replay-query-canary"
    body = ResolveRequest.model_validate({"input": {"kind": "raw_text", "query": marker}})
    record = ResolveRecord(parser_replay=_record(arm="ministral"))
    log_filter = ReplayLogFilter()

    async def failing_stream() -> AsyncGenerator[Event]:
        yield StepEvent(step="extract", ran=True)
        raise EngineInfraError(marker)

    async def collect() -> list[str]:
        line = resolve_line(
            body, record, caller=None, transport="sse", served_code=lambda _: "test_error"
        )
        return [
            frame
            async for frame in _event_source(
                failing_stream(), hard_deadline=time.monotonic() + 5, record=record, line=line
            )
        ]

    caplog.handler.addFilter(log_filter)
    try:
        with caplog.at_level(logging.INFO):
            frames = asyncio.run(collect())
    finally:
        caplog.handler.removeFilter(log_filter)

    assert "event: error" in frames[-1]
    assert marker not in caplog.text
    assert _QUERY not in caplog.text
    assert '"replay_id": "run_01"' in caplog.text
