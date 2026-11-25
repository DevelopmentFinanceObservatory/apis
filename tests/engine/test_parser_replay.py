"""Private parser-replay guard and extraction-contract tests."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from fastapi import Request
from pydantic import BaseModel, SecretStr

from dfo_platform.cloud_logging import ReplayLogFilter
from search.engine import config
from search.engine.app import _event_source
from search.engine.clients.llm import (
    LLM,
    Agreement,
    MistralLLM,
    Sampled,
    SecondSamples,
    agreement_verdict,
    for_stage,
    held_by,
    uncached,
    with_replay_stage,
    with_second_samples,
)
from search.engine.core import ResolveRecord, resolve_line
from search.engine.errors import EngineInfraError, EngineInputError, LLMInfraError
from search.engine.events import Event, StepEvent
from search.engine.extract import _MISTRAL_REPLAY_ENTITY_INSTRUCTION, MistralReplayExtraction
from search.engine.extraction_store import MemoryExtractionStore
from search.engine.parser_replay import (
    PARSER_ARM_HEADER,
    PARSER_CACHE_HEADER,
    PARSER_KEY_HEADER,
    PARSER_REPLAY_HEADER,
    PARSER_STREAM_HEADER,
    ParserReplay,
    bound_parser_replay,
    parser_replay,
    replay_log,
)
from search.engine.response_cache import ResponseCache
from search.models import ResolveRequest


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


def test_no_replay_headers_leave_the_normal_request_unchanged():
    assert parser_replay(_request({})) is None


def test_replay_sse_suppresses_terminal_error_traceback_but_logs_resolve(caplog):
    marker = "private-replay-query-canary"
    body = ResolveRequest.model_validate({"input": {"kind": "raw_text", "query": marker}})
    replay = ParserReplay(replay_id="run_01", arm="ministral", cache="bypass", stream=False)
    record = ResolveRecord(parser_replay=replay)
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
    assert '"replay_id": "run_01"' in caplog.text


def test_replay_headers_fail_closed_when_the_gate_is_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("K_SERVICE", "dfo-dev-apis")

    with pytest.raises(EngineInputError) as error:
        parser_replay(_request(_headers()))

    assert error.value.status == 403


def test_replay_headers_fail_closed_outside_the_dev_service(monkeypatch: pytest.MonkeyPatch):
    _enable_replay(monkeypatch)
    monkeypatch.setenv("K_SERVICE", "dfo-apis")

    with pytest.raises(EngineInputError) as error:
        parser_replay(_request(_headers()))

    assert error.value.status == 403


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

    assert error.value.status == 403


@pytest.mark.parametrize("arm", ["flash-lite", "ministral"])
def test_verified_replay_selects_only_a_declared_arm(monkeypatch: pytest.MonkeyPatch, arm: str):
    _enable_replay(monkeypatch)
    headers = _headers(arm=arm)
    headers[PARSER_CACHE_HEADER] = "bypass"
    headers[PARSER_STREAM_HEADER] = "1"

    replay = parser_replay(_request(headers))

    assert replay is not None
    assert (replay.replay_id, replay.arm, replay.cache, replay.stream) == (
        "run_01",
        arm,
        "bypass",
        True,
    )


def test_replay_rejects_any_non_bypass_application_cache_policy(monkeypatch: pytest.MonkeyPatch):
    _enable_replay(monkeypatch)
    headers = _headers()
    headers[PARSER_CACHE_HEADER] = "isolated"

    with pytest.raises(EngineInputError) as error:
        parser_replay(_request(headers))

    assert error.value.status == 422


def test_mistral_schema_requires_entities_and_keeps_the_tested_instruction():
    assert MistralReplayExtraction.model_fields["entities"].is_required()
    assert _MISTRAL_REPLAY_ENTITY_INSTRUCTION == (
        "\nOutput the entities key for every query. Before returning the JSON, check every named\n"
        "place or named geographic group in the query against entities. Each must have an\n"
        "object even if its name is already in a variable phrase. Use an empty array only\n"
        "when the query names no eligible scope. Each array item is an object with text,\n"
        "span, variables, cue, role, and members. A place with no role still gets an object.\n"
        "For donor or creditor direction, use only a cue present in the query.\n"
    )


def test_replay_extraction_clone_keeps_request_agreement_holder_but_no_caches():
    base = LLM(
        memo=ResponseCache(max_bytes=1024, max_entry_bytes=1024, ttl_s=60),
        extraction_store=MemoryExtractionStore(max_bytes=1024),
    )
    holder = SecondSamples()
    bound = holder.bind(base)

    staged = with_replay_stage(for_stage(bound, "extract"), "extract")
    replay = uncached(staged)

    assert isinstance(replay, LLM)
    assert replay is not base
    assert held_by(replay) is holder
    assert replay._stage == "extract"
    assert replay._memo is None
    assert replay.extraction_store is None


class _StreamResult(BaseModel):
    value: str


class _MistralStreamResponse:
    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"{\\"value\\":"}}]}'
        yield 'data: {"choices":[{"delta":{"content":"\\"ok\\"}"}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}'
        yield "data: [DONE]"


class _MistralStream:
    async def __aenter__(self) -> _MistralStreamResponse:
        return _MistralStreamResponse()

    async def __aexit__(self, *args: object) -> None:
        return None


class _MalformedMistralStreamResponse(_MistralStreamResponse):
    async def aiter_lines(self):
        yield "data: []"


class _MalformedMistralStream(_MistralStream):
    async def __aenter__(self) -> _MalformedMistralStreamResponse:
        return _MalformedMistralStreamResponse()


class _MistralClient:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    def stream(self, _method: str, _url: str, *, json: dict[str, Any]) -> _MistralStream:
        self.payload = json
        return _MistralStream()


class _MalformedMistralStreamClient(_MistralClient):
    def stream(self, _method: str, _url: str, *, json: dict[str, Any]) -> _MalformedMistralStream:
        self.payload = json
        return _MalformedMistralStream()


class _MistralPostResponse:
    def __init__(self, body: object) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._body


class _MistralPostClient:
    def __init__(self, body: object | None = None, *, cancel: bool = False) -> None:
        self._body = body
        self._cancel = cancel

    async def post(self, _url: str, *, json: dict[str, object]) -> _MistralPostResponse:
        if self._cancel:
            raise asyncio.CancelledError
        return _MistralPostResponse(self._body)


def test_mistral_replay_streams_extraction_and_records_first_content(
    monkeypatch: pytest.MonkeyPatch,
):
    client = _MistralClient()
    source = SecondSamples().bind(LLM())
    llm = cast(
        MistralLLM,
        with_replay_stage(with_second_samples(MistralLLM(api_key="test"), source), "extract"),
    )
    monkeypatch.setattr(llm, "_mistral_client", lambda: client)
    replay = ParserReplay(replay_id="run_01", arm="ministral", cache="bypass", stream=True)

    async def run() -> _StreamResult:
        with bound_parser_replay(replay):
            answer, _usage = await llm.generate_structured(
                prompt="query", system="system", schema=_StreamResult
            )
            return answer

    assert asyncio.run(run()) == _StreamResult(value="ok")
    assert client.payload is not None
    assert client.payload["stream"] is True
    assert client.payload["max_tokens"] == 768
    assert client.payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "dfo_extraction",
            "strict": True,
            "schema": _StreamResult.model_json_schema() | {"title": "Extraction"},
        },
    }
    assert replay.calls[0]["first_content_ms"] is not None


def test_mistral_schema_failure_keeps_the_returned_billable_usage(
    monkeypatch: pytest.MonkeyPatch,
):
    client = _MistralPostClient(
        {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
    )
    llm = MistralLLM(api_key="test")
    monkeypatch.setattr(llm, "_mistral_client", lambda: client)
    replay = ParserReplay(replay_id="run_01", arm="ministral", cache="bypass", stream=False)

    async def run() -> None:
        with bound_parser_replay(replay), pytest.raises(LLMInfraError) as error:
            await llm.generate_structured(prompt="query", system="system", schema=_StreamResult)
        assert error.value.__cause__ is None

    asyncio.run(run())
    assert replay.calls == [
        {
            "provider": "mistral",
            "model": "ministral-3b-2512",
            "stage": None,
            "attempt": None,
            "extraction_phase": None,
            "duration_ms": replay.calls[0]["duration_ms"],
            "first_content_ms": None,
            "usage": {
                "mistral_input_tokens": 3,
                "mistral_output_tokens": 2,
                "mistral_cached_tokens": 0,
            },
            "error": "LLMInfraError",
        }
    ]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": "{}"}}], "usage": "not-an-object"},
    ],
)
def test_mistral_malformed_response_is_a_single_sanitized_attempt(
    body: object, monkeypatch: pytest.MonkeyPatch
):
    llm = MistralLLM(api_key="test")
    monkeypatch.setattr(llm, "_mistral_client", lambda: _MistralPostClient(body))
    replay = ParserReplay(replay_id="run_01", arm="ministral", cache="bypass", stream=False)

    async def run() -> None:
        with bound_parser_replay(replay), pytest.raises(LLMInfraError) as error:
            await llm.generate_structured(prompt="query", system="system", schema=_StreamResult)
        assert str(error.value) == "Mistral API request failed"
        assert error.value.__cause__ is None

    asyncio.run(run())
    assert len(replay.calls) == 1
    assert replay.calls[0]["error"] in {"KeyError", "IndexError", "AttributeError", "TypeError"}


def test_mistral_malformed_stream_chunk_is_a_single_sanitized_attempt(
    monkeypatch: pytest.MonkeyPatch,
):
    llm = MistralLLM(api_key="test")
    monkeypatch.setattr(llm, "_mistral_client", _MalformedMistralStreamClient)
    replay = ParserReplay(replay_id="run_01", arm="ministral", cache="bypass", stream=True)

    async def run() -> None:
        with bound_parser_replay(replay), pytest.raises(LLMInfraError) as error:
            await llm.generate_structured(prompt="query", system="system", schema=_StreamResult)
        assert str(error.value) == "Mistral API request failed"
        assert error.value.__cause__ is None

    asyncio.run(run())
    assert len(replay.calls) == 1
    assert replay.calls[0]["error"] == "AttributeError"


def test_mistral_cancellation_is_recorded_then_reraised(monkeypatch: pytest.MonkeyPatch):
    llm = MistralLLM(api_key="test")
    monkeypatch.setattr(llm, "_mistral_client", lambda: _MistralPostClient(cancel=True))
    replay = ParserReplay(replay_id="run_01", arm="ministral", cache="bypass", stream=False)

    async def run() -> None:
        with bound_parser_replay(replay), pytest.raises(asyncio.CancelledError):
            await llm.generate_structured(prompt="query", system="system", schema=_StreamResult)

    asyncio.run(run())
    assert len(replay.calls) == 1
    assert replay.calls[0]["usage"] is None
    assert replay.calls[0]["error"] == "cancelled"


def test_gemini_cancellation_is_recorded_then_reraised(monkeypatch: pytest.MonkeyPatch):
    class _Models:
        async def generate_content(self, **_kwargs: object) -> object:
            raise asyncio.CancelledError

    class _Client:
        class aio:
            models = _Models()

    llm = LLM()
    monkeypatch.setattr(llm, "_client", lambda: _Client())
    replay = ParserReplay(replay_id="run_01", arm="flash-lite", cache="bypass", stream=False)

    async def run() -> None:
        with bound_parser_replay(replay), pytest.raises(asyncio.CancelledError):
            await llm._fetch_text(prompt="query", config=cast(Any, None))

    asyncio.run(run())
    assert len(replay.calls) == 1
    assert replay.calls[0]["usage"] is None
    assert replay.calls[0]["error"] == "cancelled"


def test_replay_only_stage_clone_keeps_non_replay_identity():
    llm = LLM()
    assert for_stage(llm, "extract") is llm

    replay = ParserReplay(replay_id="run_01", arm="flash-lite", cache="bypass", stream=False)
    with bound_parser_replay(replay):
        staged = for_stage(llm, "extract")
    assert isinstance(staged, LLM)
    assert staged is not llm
    assert staged._stage == "extract"


def test_replay_log_uses_only_opaque_experiment_digests_and_fallback_reasons():
    replay = ParserReplay(
        replay_id="run_01",
        arm="flash-lite",
        cache="bypass",
        stream=True,
        prompt_digest="p" * 64,
        schema_digest="s" * 64,
        fallback_reasons=["timeout"],
    )

    assert replay_log(replay) == {
        "replay_id": "run_01",
        "arm": "flash-lite",
        "cache": "bypass",
        "stream": True,
        "extraction_validated_ms": None,
        "extraction_agreed": None,
        "selected_extraction_phase": None,
        "prompt_digest": "p" * 64,
        "schema_digest": "s" * 64,
        "fallback_reasons": ["timeout"],
        "calls": [],
    }


@pytest.mark.parametrize(("judge", "expected"), [(True, True), (False, False)])
def test_completed_second_sample_records_agreement_verdict(judge: bool, expected: bool):
    async def run() -> bool | None:
        async def sample() -> Sampled[str]:
            return Sampled("second", None, None)

        agreement = Agreement(sample, stage="extraction")
        agreement.read_first(lambda _second: judge, delay_s=1)
        await agreement.ended()
        return agreement_verdict(agreement)

    assert asyncio.run(run()) is expected
