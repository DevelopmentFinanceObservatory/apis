"""Substitute composed prompts by stage and record every LLM call.

`LadderLLM` subclasses `LLM` so `for_stage` can apply bind's thinking-level pin.
Its `sink` stays shared across shallow copies, preserving records from each call.
"""

from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from search.engine.clients.llm import LLM, SupportsLLM

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)

# Mapped schemas use composed stage prompts. Every other schema keeps the engine prompt.
_STAGE_BY_SCHEMA: dict[str, str] = {"Extraction": "extract", "_UniversalBindOutput": "bind"}


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


@dataclass
class CallRecord:
    """One structured LLM call. The separate system and prompt hashes can be
    recombined to match `FakeLLM` fixture keys or compared independently."""

    stage: str | None
    schema: str
    system_sha1: str
    prompt_sha1: str
    prompt: str
    output: dict
    usage: dict | None
    elapsed_ms: int
    seed: int | None
    thinking_level: str | None


class LadderLLM(LLM):
    """Substitute a composed prompt for mapped schemas and append every call to `sink`.

    Schemas outside `_STAGE_BY_SCHEMA`, or absent from `systems`, keep the engine's
    prompt. The wrapper mirrors the inner LLM's thinking level and seed so
    `for_stage` preserves explicit overrides and applies bind's default pin.
    Shallow copies share `sink`, so bind calls remain in the caller's list.
    `generate_structured` awaits the inner LLM directly."""

    def __init__(
        self,
        inner: SupportsLLM,
        *,
        systems: Mapping[str, str],
        sink: list[CallRecord],
    ) -> None:
        self._llm = inner
        self._thinking_level: str | None = inner._thinking_level if isinstance(inner, LLM) else None
        self._seed: int | None = inner._seed if isinstance(inner, LLM) else None
        self._systems = dict(systems)
        self.sink = sink

    async def generate_structured(
        self,
        *,
        prompt: str,
        system: str,
        schema: type[_SchemaT],
    ) -> tuple[_SchemaT, dict | None]:
        stage = _STAGE_BY_SCHEMA.get(schema.__name__)
        variant = self._systems.get(stage, system) if stage is not None else system

        inner = self._llm
        if self._thinking_level is not None and isinstance(inner, LLM):
            inner = copy.copy(inner)
            inner._thinking_level = self._thinking_level

        t0 = time.perf_counter()
        parsed, usage = await inner.generate_structured(
            prompt=prompt, system=variant, schema=schema
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        self.sink.append(
            CallRecord(
                stage=stage,
                schema=schema.__name__,
                system_sha1=_sha1(variant),
                prompt_sha1=_sha1(prompt),
                prompt=prompt,
                output=parsed.model_dump(mode="json"),
                usage=usage,
                elapsed_ms=elapsed_ms,
                seed=self._seed,
                thinking_level=self._thinking_level,
            )
        )
        return parsed, usage
