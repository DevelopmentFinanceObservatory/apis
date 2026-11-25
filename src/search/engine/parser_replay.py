"""Private, dev-only parser replay controls and provider-call telemetry."""

from __future__ import annotations

import hmac
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from fastapi import Request

from dfo_platform.cloud_logging import suppress_replay_logs
from search.engine.config import settings
from search.engine.errors import EngineInputError

PARSER_REPLAY_HEADER = "X-DFO-Parser-Replay"
PARSER_ARM_HEADER = "X-DFO-Parser-Arm"
PARSER_KEY_HEADER = "X-DFO-Parser-Replay-Key"
PARSER_CACHE_HEADER = "X-DFO-Parser-Cache"
PARSER_STREAM_HEADER = "X-DFO-Parser-Stream"
PARSER_RESPONSE_HEADER = "X-DFO-Parser-Arm"

_REPLAY_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
_Arm = Literal["flash-lite", "ministral"]


@dataclass
class ParserReplay:
    """One authenticated parser replay request and its provider call records."""

    replay_id: str
    arm: _Arm
    cache: Literal["bypass"]
    stream: bool
    calls: list[dict[str, object]] = field(default_factory=list)
    extraction_started: float | None = None
    extraction_validated_ms: float | None = None
    extraction_agreed: bool | None = None
    extraction_agreement: object | None = None
    selected_extraction_phase: str | None = None
    prompt_digest: str | None = None
    schema_digest: str | None = None
    fallback_reasons: list[str] = field(default_factory=list)


_REPLAY: ContextVar[ParserReplay | None] = ContextVar("parser_replay", default=None)
_PROVIDER_ATTEMPT: ContextVar[str | None] = ContextVar("parser_replay_attempt", default=None)
_EXTRACTION_PHASE: ContextVar[str | None] = ContextVar(
    "parser_replay_extraction_phase", default=None
)


def parser_replay(request: Request) -> ParserReplay | None:
    """Validate and return the request's private replay selection, if any."""
    replay_id = request.headers.get(PARSER_REPLAY_HEADER)
    arm = request.headers.get(PARSER_ARM_HEADER)
    key = request.headers.get(PARSER_KEY_HEADER)
    parser_headers = (
        replay_id,
        arm,
        key,
        request.headers.get(PARSER_CACHE_HEADER),
        request.headers.get(PARSER_STREAM_HEADER),
    )
    if not any(value is not None for value in parser_headers):
        return None
    configured = settings().parser_replay_key.get_secret_value()
    try:
        key_matches = key is not None and hmac.compare_digest(
            key.encode("utf-8"), configured.encode("utf-8")
        )
    except UnicodeEncodeError:
        key_matches = False
    configured_revision = settings().parser_replay_revision
    current_revision = os.environ.get("K_REVISION")
    if (
        not settings().parser_replay_enabled
        or os.environ.get("K_SERVICE") != "dfo-dev-apis"
        or not configured_revision
        or not current_revision
        or current_revision != configured_revision
        or not configured
        or not key_matches
    ):
        raise EngineInputError("This parser replay request is not authorised.", status=403)
    if replay_id is None or not _REPLAY_ID.fullmatch(replay_id):
        raise EngineInputError("The parser replay identifier is invalid.", status=422)
    if arm not in ("flash-lite", "ministral"):
        raise EngineInputError("The parser replay arm is invalid.", status=422)
    cache = request.headers.get(PARSER_CACHE_HEADER, "bypass")
    if cache != "bypass":
        raise EngineInputError("The parser replay cache policy is invalid.", status=422)
    stream = request.headers.get(PARSER_STREAM_HEADER, "")
    if stream not in ("", "1"):
        raise EngineInputError("The parser replay stream option is invalid.", status=422)
    return ParserReplay(replay_id=replay_id, arm=arm, cache=cache, stream=stream == "1")


@contextmanager
def bound_parser_replay(replay: ParserReplay | None) -> Iterator[None]:
    """Make a verified replay available to extraction and provider clients."""
    token = _REPLAY.set(replay)
    try:
        with suppress_replay_logs(enabled=replay is not None):
            yield
    finally:
        _REPLAY.reset(token)


def current_parser_replay() -> ParserReplay | None:
    """Return the active replay in this request task, if any."""
    return _REPLAY.get()


@contextmanager
def provider_attempt(attempt: str) -> Iterator[None]:
    """Label one request's first or speculative second parser sample."""
    token = _PROVIDER_ATTEMPT.set(attempt)
    try:
        yield
    finally:
        _PROVIDER_ATTEMPT.reset(token)


@contextmanager
def bound_extraction_phase(phase: Literal["original", "blanked"]) -> Iterator[None]:
    """Tag a raw-text extraction branch without retaining its query text."""
    token = _EXTRACTION_PHASE.set(phase)
    try:
        yield
    finally:
        _EXTRACTION_PHASE.reset(token)


def note_selected_extraction_phase(phase: Literal["original", "blanked"]) -> None:
    """Record the raw-text branch that supplied the served extraction."""
    replay = current_parser_replay()
    if replay is not None:
        replay.selected_extraction_phase = phase


def replay_log(replay: ParserReplay | None) -> dict[str, object] | None:
    """Return the replay telemetry safe for the structured resolve log line."""
    if replay is None:
        return None
    return {
        "replay_id": replay.replay_id,
        "arm": replay.arm,
        "cache": replay.cache,
        "stream": replay.stream,
        "extraction_validated_ms": replay.extraction_validated_ms,
        "extraction_agreed": replay.extraction_agreed,
        "selected_extraction_phase": replay.selected_extraction_phase,
        "prompt_digest": replay.prompt_digest,
        "schema_digest": replay.schema_digest,
        "fallback_reasons": replay.fallback_reasons,
        "calls": replay.calls,
    }


def note_provider_call(
    *,
    provider: str,
    model: str,
    stage: str | None,
    usage: dict | None,
    started: float,
    first_content_ms: float | None = None,
    error: str | None = None,
) -> None:
    """Record one provider attempt without query, response, or secret material."""
    replay = current_parser_replay()
    if replay is None:
        return
    replay.calls.append(
        {
            "provider": provider,
            "model": model,
            "stage": stage,
            "attempt": _PROVIDER_ATTEMPT.get(),
            "extraction_phase": _EXTRACTION_PHASE.get(),
            "duration_ms": round((perf_counter() - started) * 1000, 3),
            "first_content_ms": first_content_ms,
            "usage": usage,
            "error": error,
        }
    )
