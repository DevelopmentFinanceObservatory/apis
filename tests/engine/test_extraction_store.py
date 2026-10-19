"""The extraction store answers a repeated query without a model call, and reads afresh after
an edit to the extract prompt or schema or in a new calendar year.

Async calls use `asyncio.run` because the suite has no pytest-asyncio plugin.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import date
from types import SimpleNamespace

import pytest
from pydantic import create_model

import search.engine.clients.llm as llm_module
import search.engine.extract as extract_module
from search.engine.clients.llm import LLM
from search.engine.errors import LLMInfraError, LLMOutputRefused
from search.engine.extract import (
    EXTRACT_VERSION,
    Extraction,
    _extract_version,
    _extraction_system,
    extract,
)
from search.engine.extraction_store import (
    FileExtractionStore,
    MemoryExtractionStore,
    extraction_key,
)

_QUERY = "GDP of Kenya"
_ANSWER = Extraction(kind="resolution", variables=["GDP"])
_NO_USAGE = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


class _UsageMeta:
    prompt_token_count = 11
    candidates_token_count = 22
    cached_content_token_count = 0


class _CountingModels:
    """Answers every request with `_ANSWER` and records the prompts it was sent."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_content(self, *, model, contents, config):
        self.prompts.append(contents)
        return SimpleNamespace(text=_ANSWER.model_dump_json(), usage_metadata=_UsageMeta())


def _llm(monkeypatch, store) -> tuple[LLM, _CountingModels]:
    models = _CountingModels()
    monkeypatch.setattr(
        llm_module, "_get_client", lambda: SimpleNamespace(aio=SimpleNamespace(models=models))
    )
    return LLM(extraction_store=store), models


def _extract(query: str, llm: LLM):
    return asyncio.run(extract(query, llm=llm))


def test_a_query_in_another_whitespace_and_unicode_form_calls_the_model_once(monkeypatch):
    llm, models = _llm(monkeypatch, MemoryExtractionStore(max_bytes=1024 * 1024))

    first, first_usage = _extract(_QUERY, llm)
    # A no-break space, a doubled space, and padding: the same query once NFKC-normalised.
    second, second_usage = _extract(" GDP\u00a0of  Kenya ", llm)

    assert models.prompts == [_QUERY]
    assert first_usage is not None and first_usage["input_tokens"] == 11
    assert second_usage == _NO_USAGE
    assert second == first


@pytest.mark.parametrize(
    "version",
    [
        _extract_version(extract_module._EXTRACTION_SYSTEM_PROMPT + " ", Extraction),
        _extract_version(
            extract_module._EXTRACTION_SYSTEM_PROMPT,
            create_model("Extraction", __base__=Extraction, note=(str, "")),
        ),
    ],
    ids=["changed-prompt", "new-schema-field"],
)
def test_a_new_extract_version_calls_the_model_again(monkeypatch, version):
    llm, models = _llm(monkeypatch, MemoryExtractionStore(max_bytes=1024 * 1024))
    _extract(_QUERY, llm)

    assert version != EXTRACT_VERSION
    monkeypatch.setattr(extract_module, "EXTRACT_VERSION", version)
    _extract(_QUERY, llm)

    assert models.prompts == [_QUERY, _QUERY]


def test_a_new_calendar_year_calls_the_model_again_with_that_year(monkeypatch):
    """A stored reading lives for the year its prompt named; the next year asks again."""
    llm, models = _llm(monkeypatch, MemoryExtractionStore(max_bytes=1024 * 1024))
    systems: list[str] = []
    send = extract_module.generate_structured_async

    async def recording(llm, *, prompt, system, schema):
        systems.append(system)
        return await send(llm, prompt=prompt, system=system, schema=schema)

    monkeypatch.setattr(extract_module, "generate_structured_async", recording)
    for day in (date(2026, 12, 31), date(2027, 1, 1), date(2027, 6, 1)):
        monkeypatch.setattr(extract_module, "date", SimpleNamespace(today=lambda day=day: day))
        _extract(_QUERY, llm)

    assert models.prompts == [_QUERY, _QUERY]
    assert systems == [_extraction_system(2026), _extraction_system(2027)]


@pytest.mark.parametrize("failure", [LLMOutputRefused("refused"), LLMInfraError("failed")])
def test_a_refused_or_failed_call_is_not_stored(monkeypatch, failure):
    llm, _models = _llm(monkeypatch, MemoryExtractionStore(max_bytes=1024 * 1024))
    outcomes: list[BaseException | Extraction] = [failure, _ANSWER]

    async def answer(llm, *, prompt, system, schema):
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome, None

    monkeypatch.setattr(extract_module, "generate_structured_async", answer)
    with pytest.raises(type(failure)):
        _extract(_QUERY, llm)
    _extract(_QUERY, llm)
    _extract(_QUERY, llm)

    # The failure stored nothing, so the second call asked again; the third was a hit.
    assert outcomes == []


def test_a_file_store_written_by_one_process_is_read_by_a_second(monkeypatch, tmp_path):
    path = tmp_path / "extractions.jsonl"
    llm, _models = _llm(monkeypatch, FileExtractionStore(path))
    _extract(_QUERY, llm)
    # A line another writer has not finished is skipped until its newline lands.
    with path.open("ab") as file:
        file.write(b'{"key": "extract_torn"')

    key = extraction_key(
        _QUERY,
        version=EXTRACT_VERSION,
        year=extract_module.date.today().year,
        identity=llm.sampling_identity(),
    )
    reader = (
        "import sys; from pathlib import Path;"
        "from search.engine.extraction_store import FileExtractionStore;"
        "print(FileExtractionStore(Path(sys.argv[1])).get(sys.argv[2]))"
    )
    read = subprocess.run(
        [sys.executable, "-c", reader, str(path), key],
        capture_output=True,
        text=True,
        check=True,
    )

    assert Extraction.model_validate_json(read.stdout) == _ANSWER


def test_the_memory_store_evicts_the_least_recently_read_entry_past_its_byte_budget():
    entry_bytes = len("k1") + len("v" * 10)
    store = MemoryExtractionStore(max_bytes=2 * entry_bytes)
    store.put("k1", "v" * 10)
    store.put("k2", "v" * 10)
    assert store.get("k1") is not None

    store.put("k3", "v" * 10)

    assert [store.get(key) is not None for key in ("k1", "k2", "k3")] == [True, False, True]
