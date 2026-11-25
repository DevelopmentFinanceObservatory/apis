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

type Arm = Literal["flash-lite", "ministral", "mistral-small"]

FLASH_LITE_MODEL = "gemini-3.5-flash-lite"
ARM_MODELS: dict[Arm, str] = {
    "flash-lite": FLASH_LITE_MODEL,
    "ministral": "ministral-3b-2512",
    "mistral-small": "mistral-small-2603",
}

_REPLAY_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")

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
    if arm is None or arm not in ARM_MODELS:
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
    """The schema the Mistral arms answer under: `Extraction` with `entities` required."""

    entities: list[ExtractedEntity] = Field(
        description=Extraction.model_fields["entities"].description
    )


MISTRAL_REPLAY_INSTRUCTION = """
Checklist. These rules restate the instructions above in the exact JSON shape the schema needs. Apply every rule before you answer.

kind
- A query that names any measure, such as aid, GDP, debt, mortality, trade or any statistic, is "resolution", even when it starts with "show me", "what is", "data on" or "do you have". Use "catalogue" only when it asks what data exists and names no topic. Use "out_of_scope" only when no part asks for a statistic.

entities
- One object for every place, country, city, region, group or organisation the query names, each listed once. A year, a measure or a bare word such as "countries" is never an entity.
- A place named by an adjective is still an entity: "Italian aid" names Italy, "American loans" names United States, "Chinese investment" names China.
- `text` is the English name. "X countries", "countries in X" and "X nations" name the region X with `members` true: "Latin American countries" has text "Latin America" and members true.
- `span` copies the query's words for the place exactly.
- `role` and `cue` come only from the query's words. "from A to B": A is "source" with cue "from", B is "recipient" with cue "to". "aid to B", "B received": B is "recipient". "A's aid", "Italian aid", "A gave": A is "source", cue the word that says so. Lending words (loans, lent, owes, debt owed to) use "creditor" and "debtor". A place that is only located ("in A", "for A") or listed side by side ("A and B") has role null and cue null.

dates
- "in 2019" or a bare "2019" is {"kind": "point", "start": "2019", "end": "2019"}.
- "since 2010" or "from 2010" is {"kind": "range", "start": "2010", "end": null}. "2010 to 2015" and "between 2010 and 2015" are {"kind": "range", "start": "2010", "end": "2015"}.
- "latest", "most recent" and "current" are {"kind": "latest", "start": null, "end": null}. With no time words, `dates` is [].
- A future target year of a forecast belongs to the forecast operation, not to `dates`.

operations
- Most queries have no operation, so `operations` is []. Add one only for these words: "top", "most", "largest", "ranked" (rank); "total", "combined", "sum of" (aggregate); "change", "growth", "increase since" (growth); "share of", "as a share of" (share_of); "why", "explain" (explain); "forecast", "projected", or a year after the current year (forecast); "overview", "report" (report).
- "compare A and B", "in", "from", "to" and a single year are not operations.

Examples in the schema's shape:

"bilateral aid from Norway to Malawi in 2018"
{"kind": "resolution", "variables": ["bilateral aid"], "entities": [{"text": "Norway", "span": "Norway", "variables": [], "cue": "from", "role": "source", "members": false}, {"text": "Malawi", "span": "Malawi", "variables": [], "cue": "to", "role": "recipient", "members": false}], "operations": [], "dropped_qualifiers": [], "dates": [{"kind": "point", "start": "2018", "end": "2018"}]}

"child mortality in Latin American countries since 2005"
{"kind": "resolution", "variables": ["child mortality"], "entities": [{"text": "Latin America", "span": "Latin American countries", "variables": [], "cue": null, "role": null, "members": true}], "operations": [], "dropped_qualifiers": [], "dates": [{"kind": "range", "start": "2005", "end": null}]}

"compare electricity access in Bolivia and Indonesia"
{"kind": "resolution", "variables": ["electricity access"], "entities": [{"text": "Bolivia", "span": "Bolivia", "variables": [], "cue": null, "role": null, "members": false}, {"text": "Indonesia", "span": "Indonesia", "variables": [], "cue": null, "role": null, "members": false}], "operations": [], "dropped_qualifiers": [], "dates": []}

"Swedish humanitarian aid to Yemen"
{"kind": "resolution", "variables": ["humanitarian aid"], "entities": [{"text": "Sweden", "span": "Swedish", "variables": [], "cue": "Swedish", "role": "source", "members": false}, {"text": "Yemen", "span": "Yemen", "variables": [], "cue": "to", "role": "recipient", "members": false}], "operations": [], "dropped_qualifiers": [], "dates": []}

"projected population of Niger in 2050"
{"kind": "resolution", "variables": ["population"], "entities": [{"text": "Niger", "span": "Niger", "variables": [], "cue": null, "role": null, "members": false}], "operations": [{"verb": "forecast", "variable": 0, "phrase": "projected in 2050", "over": null}], "dropped_qualifiers": [], "dates": []}

"tasa de alfabetización en Bolivia en 2012"
{"kind": "resolution", "variables": ["literacy rate"], "entities": [{"text": "Bolivia", "span": "Bolivia", "variables": [], "cue": null, "role": null, "members": false}], "operations": [], "dropped_qualifiers": [], "dates": [{"kind": "point", "start": "2012", "end": "2012"}]}

Final check, after you draft the JSON and before you return it:
1. Read the query again word by word. Every country, city, region, group, organisation or place adjective ("Kenyan", "Italian", "European") must have one object in `entities`, even when the name sits inside a measure ("Kenya's debt", "US aid", "Nigerian GDP"). If one is missing, add it. `entities` is [] only when the query names no place at all.
2. For each entity, look for a flow of money (aid, ODA, finance, loans, grants, investment, disbursements, remittances) and a word that gives the place a side in it ("to", "from", "received", "gave", "by", "for", "Italian aid"). If both are there, set `role` and `cue`. A place that receives is "recipient", one that gives is "source", one that lends is "creditor", one that borrows or owes is "debtor".
3. Remove any entity that is a year or a measure, and any entity listed twice.
"""


def replay_extraction_llm(record: ReplayRecord) -> LLM:
    """The extraction client of `record`'s arm, recording into `record`: Gemini
    Flash-Lite, or a Mistral model (`ARM_MODELS`) with `MistralReplayExtraction` and
    `MISTRAL_REPLAY_INSTRUCTION`. Neither has a memo or an extraction store. Raises
    `LLMInfraError` for a Mistral arm when `MISTRAL_API_KEY` is unset."""
    replay = record.replay
    if replay.arm == "flash-lite":
        return ReplayGeminiLLM(model=ARM_MODELS[replay.arm], recorder=record, stream=replay.stream)
    return MistralLLM(
        model=ARM_MODELS[replay.arm],
        clients=mistral_clients(),
        schema=MistralReplayExtraction,
        instruction=MISTRAL_REPLAY_INSTRUCTION,
        recorder=record,
        stream=replay.stream,
    )
