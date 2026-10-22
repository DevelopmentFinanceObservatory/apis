"""Checks `LadderLLM` prompt substitution, cloning, call records, and usage forwarding."""

from __future__ import annotations

import asyncio
import hashlib

from pydantic import BaseModel

from promptlab.llm import CallRecord, LadderLLM
from search.engine.clients.llm import LLM, for_stage
from search.engine.extract import Extraction


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


class _OtherOutput(BaseModel):
    """A schema with no ladder stage, used to test system-prompt passthrough."""

    x: int = 0


class _StubInner:
    """Return fixed output and usage while recording forwarded calls."""

    def __init__(self, output: BaseModel, usage: dict | None = None) -> None:
        self._output = output
        self._usage = usage
        self.calls: list[tuple[str, str, str]] = []

    async def generate_structured(self, *, prompt: str, system: str, schema):
        self.calls.append((schema.__name__, system, prompt))
        return self._output, self._usage


def test_for_stage_pins_bind_thinking_level_to_minimal():
    # The `LLM` subclass passes `for_stage`'s type guard.
    wrapper = LadderLLM(LLM(), systems={}, sink=[])

    clone = for_stage(wrapper, "bind")

    assert isinstance(clone, LadderLLM)  # proves for_stage cloned, not passed through
    assert clone._thinking_level == "minimal"


def test_clone_records_land_in_the_original_wrappers_sink():
    original_sink: list[CallRecord] = []
    wrapper = LadderLLM(
        _StubInner(Extraction(kind="resolution", variables=["gdp"])), systems={}, sink=original_sink
    )

    clone = for_stage(wrapper, "bind")
    assert isinstance(clone, LadderLLM)
    assert clone is not wrapper  # for_stage did clone, or this test proves nothing
    asyncio.run(clone.generate_structured(prompt="p", system="s", schema=Extraction))

    # The clone and original share the sink list.
    assert len(original_sink) == 1
    assert wrapper.sink is clone.sink


def test_explicit_thinking_override_outranks_the_stage_default():
    wrapper = LadderLLM(LLM(thinking_level="high"), systems={}, sink=[])

    # An explicit thinking level prevents stage-level pinning.
    assert for_stage(wrapper, "bind") is wrapper


def test_unknown_schema_passes_the_engines_system_text_through_unmodified():
    inner = _StubInner(_OtherOutput(x=1))
    sink: list[CallRecord] = []
    wrapper = LadderLLM(inner, systems={"extract": "REPLACEMENT SYSTEM TEXT"}, sink=sink)

    asyncio.run(
        wrapper.generate_structured(
            prompt="p", system="ENGINE'S OWN SYSTEM TEXT", schema=_OtherOutput
        )
    )

    assert inner.calls == [("_OtherOutput", "ENGINE'S OWN SYSTEM TEXT", "p")]
    assert sink[0].stage is None
    assert sink[0].system_sha1 == _sha1("ENGINE'S OWN SYSTEM TEXT")


def test_a_stage_schema_gets_its_composed_system_text_substituted():
    inner = _StubInner(Extraction(kind="resolution", variables=["gdp"]))
    wrapper = LadderLLM(inner, systems={"extract": "COMPOSED RUNG TEXT"}, sink=[])

    asyncio.run(
        wrapper.generate_structured(
            prompt="p", system="ENGINE'S OWN SYSTEM TEXT", schema=Extraction
        )
    )

    assert inner.calls == [("Extraction", "COMPOSED RUNG TEXT", "p")]


def test_records_carry_schema_hashes_output_usage_elapsed_and_thinking_level():
    usage = {"input_tokens": 3, "output_tokens": 5, "cached_tokens": 0}
    inner = _StubInner(Extraction(kind="resolution", variables=["gdp"]), usage=usage)
    original_sink: list[CallRecord] = []
    wrapper = LadderLLM(inner, systems={}, sink=original_sink)

    clone = for_stage(wrapper, "bind")  # pins thinking_level to "minimal"
    asyncio.run(
        clone.generate_structured(prompt="the prompt", system="the system", schema=Extraction)
    )

    assert len(original_sink) == 1
    record = original_sink[0]
    assert record.schema == "Extraction"
    assert record.system_sha1 == _sha1("the system")
    assert record.prompt_sha1 == _sha1("the prompt")
    assert record.prompt == "the prompt"
    assert record.output == {
        "kind": "resolution",
        "variables": ["gdp"],
        "entities": [],
        "operations": [],
        "measure_qualifiers": [],
        "dropped_qualifiers": [],
        "dates": [],
    }
    assert record.usage == usage
    assert isinstance(record.elapsed_ms, int)
    assert record.elapsed_ms >= 0
    assert record.thinking_level == "minimal"
