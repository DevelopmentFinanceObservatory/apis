"""The private, dev-only parser replay: its request gate, its extraction clients and the
telemetry its `resolve` log line carries.

An authenticated replay request names an arm. `app.resolve` builds that arm's extraction
client (`replay_extraction_llm`) and passes it to the pipeline as `EntryParams.extract_llm`,
so a request without replay headers runs as it does without this module. The client
records each provider call it makes in the request's `ReplayRecord`.
"""

from __future__ import annotations

import hmac
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from fastapi import Request
from pydantic import Field

from search.engine.clients.llm import LLM, MistralLLM, ReplayGeminiLLM, mistral_clients
from search.engine.config import settings
from search.engine.errors import EngineInputError
from search.engine.extract import ExtractedEntity, Extraction

PARSER_REPLAY_HEADER = "X-DFO-Parser-Replay"
PARSER_ARM_HEADER = "X-DFO-Parser-Arm"
PARSER_KEY_HEADER = "X-DFO-Parser-Replay-Key"
PARSER_CACHE_HEADER = "X-DFO-Parser-Cache"
PARSER_STREAM_HEADER = "X-DFO-Parser-Stream"

FLASH_LITE_MODEL = "gemini-3.5-flash-lite"

_REPLAY_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")

type Arm = Literal["flash-lite", "ministral"]
type ExtractionPhase = Literal["original", "blanked"]


@dataclass(frozen=True)
class ParserReplay:
    """One authenticated parser replay request: its opaque id, its arm, and whether the
    extraction client streams. The cache policy is always `bypass`."""

    replay_id: str
    arm: Arm
    stream: bool
    cache: Literal["bypass"] = "bypass"


def parser_replay(request: Request) -> ParserReplay | None:
    """The request's replay selection, None when it carries no replay header.

    Raises `EngineInputError` with status 403 unless the replay switch is on, the process is
    the `dfo-dev-apis` revision the settings name, and the key matches, and with status 422
    for an invalid id, arm, cache policy or stream option."""
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
    if arm != "flash-lite" and arm != "ministral":
        raise EngineInputError("The parser replay arm is invalid.", status=422)
    if request.headers.get(PARSER_CACHE_HEADER, "bypass") != "bypass":
        raise EngineInputError("The parser replay cache policy is invalid.", status=422)
    stream = request.headers.get(PARSER_STREAM_HEADER, "")
    if stream not in ("", "1"):
        raise EngineInputError("The parser replay stream option is invalid.", status=422)
    return ParserReplay(replay_id=replay_id, arm=arm, stream=stream == "1")


@dataclass(frozen=True)
class ProviderCall:
    """One extraction call a replay client made, holding no query, answer or key.

    `phase` is `original` for a parse of the query as sent and `blanked` for a parse of
    the text the exclusion read left. `attempt` is `second` for the reseeded sample that
    judges agreement. `usage` holds the provider's token counts, Gemini's thought tokens
    included. `error` names the exception class, or `cancelled`, and `http_status` is the
    provider's status when it refused the call."""

    provider: Literal["gemini", "mistral"]
    model: str
    phase: ExtractionPhase
    attempt: Literal["first", "second"]
    duration_ms: float
    first_content_ms: float | None
    usage: dict[str, int] | None
    finish_reason: str | None
    error: str | None
    http_status: int | None


@dataclass
class ReplayRecord:
    """What one replay request's `resolve` log line reports (`log_fields`).

    The extraction client fills `calls` and the digests. The pipeline sets
    `extraction_agreed` from the served extraction's agreement and `fallback_reasons` from
    the binder's fallbacks. `query` tells the two extraction phases apart and stays out of
    the log."""

    replay: ParserReplay
    query: str = field(repr=False)
    calls: list[ProviderCall] = field(default_factory=list)
    prompt_digest: str | None = None
    schema_digest: str | None = None
    extraction_agreed: bool | None = None
    fallback_reasons: list[str] = field(default_factory=list)

    def note_call(
        self,
        *,
        provider: Literal["gemini", "mistral"],
        model: str,
        prompt: str,
        attempt: Literal["first", "second"],
        prompt_digest: str,
        schema_digest: str,
        duration_ms: float,
        first_content_ms: float | None,
        usage: dict[str, int] | None,
        finish_reason: str | None,
        error: str | None,
        http_status: int | None,
    ) -> None:
        """Record one provider call. A prompt other than the query is the blanked text
        (`entry.raw_text._read_and_extract`)."""
        self.prompt_digest = prompt_digest
        self.schema_digest = schema_digest
        self.calls.append(
            ProviderCall(
                provider=provider,
                model=model,
                phase="original" if prompt == self.query else "blanked",
                attempt=attempt,
                duration_ms=duration_ms,
                first_content_ms=first_content_ms,
                usage=usage,
                finish_reason=finish_reason,
                error=error,
                http_status=http_status,
            )
        )

    @property
    def served_phase(self) -> ExtractionPhase | None:
        """The phase whose parse the answer used: `blanked` once a blanked parse ran, since
        the door then discards the parse of the query, else `original`, and None before
        any call."""
        if not self.calls:
            return None
        return "blanked" if any(call.phase == "blanked" for call in self.calls) else "original"

    def log_fields(self) -> dict[str, object]:
        """The replay's fields of the `resolve` log line."""
        return {
            "replay_id": self.replay.replay_id,
            "arm": self.replay.arm,
            "cache": self.replay.cache,
            "stream": self.replay.stream,
            "served_phase": self.served_phase,
            "extraction_agreed": self.extraction_agreed,
            "prompt_digest": self.prompt_digest,
            "schema_digest": self.schema_digest,
            "fallback_reasons": self.fallback_reasons,
            "calls": [asdict(call) for call in self.calls],
        }


class MistralReplayExtraction(Extraction):
    """The schema Ministral answers under: `Extraction` with `entities` required."""

    entities: list[ExtractedEntity] = Field(
        description=Extraction.model_fields["entities"].description
    )


MISTRAL_REPLAY_INSTRUCTION = """
Output the entities key for every query. Before returning the JSON, check every named
place or named geographic group in the query against entities. Each must have an
object even if its name is already in a variable phrase. Use an empty array only
when the query names no eligible scope. Each array item is an object with text,
span, variables, cue, role, and members. A place with no role still gets an object.
For donor or creditor direction, use only a cue present in the query.
"""


def replay_extraction_llm(record: ReplayRecord) -> LLM:
    """The extraction client of `record`'s arm, recording into `record`: Gemini Flash-Lite,
    or Ministral with `MistralReplayExtraction` and `MISTRAL_REPLAY_INSTRUCTION`. Neither
    has a memo or an extraction store. Raises `LLMInfraError` for the Ministral arm when
    `MISTRAL_API_KEY` is unset."""
    replay = record.replay
    if replay.arm == "flash-lite":
        return ReplayGeminiLLM(model=FLASH_LITE_MODEL, recorder=record, stream=replay.stream)
    return MistralLLM(
        clients=mistral_clients(),
        schema=MistralReplayExtraction,
        instruction=MISTRAL_REPLAY_INSTRUCTION,
        recorder=record,
        stream=replay.stream,
    )
