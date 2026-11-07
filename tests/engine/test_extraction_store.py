"""The extraction store answers a repeated query without a model call, and reads afresh after
an edit to the extract prompt or schema or in a new calendar year. It stores only a reading
two samples of the model, drawn under different seeds, agree on, and so does the exclusion
read stored beside it.

Async calls use `asyncio.run` because the suite has no pytest-asyncio plugin.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from datetime import date
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, create_model

import search.engine.clients.llm as llm_module
import search.engine.extract as extract_module
from search.engine.clients.llm import LLM, for_stage
from search.engine.errors import LLMInfraError, LLMOutputRefused
from search.engine.exclusion_read import (
    EXCLUSION_READ_VERSION,
    Exclusion,
    ExclusionRead,
    read_exclusions,
    stored_exclusion_read,
)
from search.engine.extract import (
    EXTRACT_VERSION,
    ExtractedEntity,
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
_OTHER_ANSWER = Extraction(kind="resolution", variables=["GDP per capita"])
_NO_USAGE = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


class _UsageMeta:
    prompt_token_count = 11
    candidates_token_count = 22
    cached_content_token_count = 0


class _CountingModels:
    """Answers a request with the answer `answers` holds for its seed, `_ANSWER` by default,
    and records the prompt and seed of each request it was sent."""

    def __init__(self, answers: dict[int, BaseModel] | None = None) -> None:
        self._answers = answers or {}
        self.prompts: list[str] = []
        self.seeds: list[int] = []

    async def generate_content(self, *, model, contents, config):
        await asyncio.sleep(0)
        self.prompts.append(contents)
        self.seeds.append(config.seed)
        answer = self._answers.get(config.seed, _ANSWER)
        return SimpleNamespace(text=answer.model_dump_json(), usage_metadata=_UsageMeta())


def _llm(
    monkeypatch, store, answers: dict[int, BaseModel] | None = None
) -> tuple[LLM, _CountingModels]:
    models = _CountingModels(answers)
    monkeypatch.setattr(
        llm_module, "_get_client", lambda: SimpleNamespace(aio=SimpleNamespace(models=models))
    )
    return LLM(extraction_store=store), models


def _extract(query: str, llm: LLM):
    return asyncio.run(extract(query, llm=llm))


def test_a_repeated_query_calls_the_model_once_and_another_spelling_reads_afresh(monkeypatch):
    """`extract()` post-processes a reading against the query text it was made from, so a
    reading is stored under that exact text. A miss samples the model twice, under its own
    seed and the next one, and a stored reading counts as agreed."""
    llm, models = _llm(monkeypatch, MemoryExtractionStore(max_bytes=1024 * 1024))

    first, first_usage = _extract(_QUERY, llm)
    second, second_usage = _extract(_QUERY, llm)
    # A no-break space, a doubled space, and padding: the same query once NFKC-normalised.
    _extract(" GDP\u00a0of  Kenya ", llm)

    assert models.prompts == [_QUERY] * 2 + [" GDP\u00a0of  Kenya "] * 2
    assert sorted(models.seeds[:2]) == [1, 2]
    assert first_usage is not None and first_usage["input_tokens"] == 22
    assert second_usage == _NO_USAGE
    assert second == first
    assert first.agreed and second.agreed


# A second sample naming one distinct entity past the cap, which post-processing refuses.
_OVER_THE_ENTITY_CAP = Extraction(
    kind="resolution",
    variables=["GDP"],
    entities=[
        ExtractedEntity(text=f"Place {i}", variables=[0])
        for i in range(llm_module.config.settings().max_entities + 1)
    ],
)


@pytest.mark.parametrize(
    "second_answer",
    [_OTHER_ANSWER, _OVER_THE_ENTITY_CAP],
    ids=["reads_apart", "second_over_the_entity_cap"],
)
def test_samples_that_disagree_serve_the_first_store_nothing_and_read_again(
    monkeypatch, second_answer
):
    """Two samples whose post-processed readings differ leave the store empty, so the next
    call of the same query samples the model again. A second sample whose reading
    post-processing refuses, here for naming more entities than the deployment resolves,
    disagrees the same way and never fails the request."""
    llm, models = _llm(
        monkeypatch, MemoryExtractionStore(max_bytes=1024 * 1024), answers={2: second_answer}
    )

    first, first_usage = _extract(_QUERY, llm)
    second, _ = _extract(_QUERY, llm)

    assert first.extraction.variables == ["GDP"]
    assert not first.agreed
    assert first_usage is not None and first_usage["input_tokens"] == 22
    assert second == first
    assert models.prompts == [_QUERY] * 4


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

    assert models.prompts == [_QUERY] * 4


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

    assert models.prompts == [_QUERY] * 4
    assert systems == [_extraction_system(2026)] * 2 + [_extraction_system(2027)] * 2


_REFUSED_USAGE = {"input_tokens": 5, "output_tokens": 1, "cached_tokens": 0}


@pytest.mark.parametrize(
    ("failure", "second_usage"),
    [
        pytest.param(
            LLMOutputRefused("refused", usage=_REFUSED_USAGE), _REFUSED_USAGE, id="refused"
        ),
        pytest.param(LLMInfraError("failed"), {"uncounted_calls": 1}, id="failed"),
    ],
)
@pytest.mark.parametrize("failing_seed", [1, 2], ids=["first-sample", "second-sample"])
def test_a_refused_or_failed_sample_is_not_stored(
    monkeypatch, caplog, failure, second_usage, failing_seed
):
    """A failed first sample fails the extraction as a lone sample did and cancels the second
    in flight; a failed second sample serves the first one, not agreed, logs a warning, and
    counts its reported tokens or one uncounted call. Neither stores, so the next call asks
    again, and the call after a stored pair is a hit."""
    llm, _models = _llm(monkeypatch, MemoryExtractionStore(max_bytes=1024 * 1024))
    outcomes: dict[int, list[BaseException | Extraction]] = (
        {1: [failure, _ANSWER], 2: [_ANSWER]}
        if failing_seed == 1
        else {1: [_ANSWER, _ANSWER], 2: [failure, _ANSWER]}
    )

    async def answer(llm, *, prompt, system, schema):
        await asyncio.sleep(0)
        outcome = outcomes[llm._seed].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome, None

    monkeypatch.setattr(extract_module, "generate_structured_async", answer)
    if failing_seed == 1:
        with pytest.raises(type(failure)):
            _extract(_QUERY, llm)
    else:
        served, usage = _extract(_QUERY, llm)
        assert served.extraction.variables == ["GDP"]
        assert not served.agreed
        assert usage == second_usage
        assert [r.levelname for r in caplog.records if r.exc_info] == ["WARNING"]
    again, _ = _extract(_QUERY, llm)
    _extract(_QUERY, llm)

    assert again.agreed
    assert outcomes == {1: [], 2: []}


class _GatedModels(_CountingModels):
    """`_CountingModels` whose answer under seed 2 waits for `gate`, counting the calls in
    flight."""

    def __init__(self, answers: dict[int, BaseModel] | None = None) -> None:
        super().__init__(answers)
        self.gate = asyncio.Event()
        self.in_flight = 0

    async def generate_content(self, *, model, contents, config):
        self.in_flight += 1
        try:
            if config.seed == 2:
                await self.gate.wait()
            return await super().generate_content(model=model, contents=contents, config=config)
        finally:
            self.in_flight -= 1


@pytest.mark.parametrize("stage", ["extraction", "exclusion_read"])
def test_a_second_sample_that_outlasts_its_wait_is_cancelled_and_releases_its_call(
    monkeypatch, stage
):
    """A second sample still running `llm_second_sample_wait_s` after the first answered is
    cancelled before the first is served: the read returns with no model call in flight,
    serves the first not agreed with the second counted as one uncounted call, and memoises
    nothing for the second, so the same request again calls the model for it."""
    settings = llm_module.config.settings()
    monkeypatch.setattr(
        llm_module.config,
        "_SETTINGS",
        settings.model_copy(update={"llm_second_sample_wait_s": 0.01, "llm_timeout_s": 5.0}),
    )
    models = _GatedModels({1: _READ, 2: _READ} if stage == "exclusion_read" else None)
    monkeypatch.setattr(
        llm_module, "_get_client", lambda: SimpleNamespace(aio=SimpleNamespace(models=models))
    )
    memo = llm_module.LLMMemo(max_bytes=1024 * 1024, max_entry_bytes=256 * 1024, ttl_s=3600.0)
    # No extraction store, so the repeat reaches the model calls and only the memo serves it.
    llm = LLM(memo=memo)

    async def read() -> tuple[bool | None, dict | None]:
        if stage == "exclusion_read":
            _read, usage, agreed = await read_exclusions(_READ_QUERY, llm=llm)
            return agreed, usage
        served, usage = await extract(_QUERY, llm=llm)
        return served.agreed, usage

    async def run():
        first = await read()
        in_flight = models.in_flight
        models.gate.set()
        return first, in_flight, await read()

    (first_agreed, first_usage), in_flight, (again_agreed, again_usage) = asyncio.run(run())

    assert in_flight == 0
    assert first_agreed is False
    assert first_usage is not None and first_usage["uncounted_calls"] == 1
    assert again_agreed is True
    assert models.seeds == [1, 2]
    assert again_usage is not None and again_usage["input_tokens"] == 11


_READ_QUERY = "ODA to Kenya excluding grants"
_READ = ExclusionRead(exclusions=[Exclusion(phrase="excluding grants", left_out=["grants"])])


@pytest.mark.parametrize(
    ("second", "agreed"),
    [
        pytest.param(_READ, True, id="same-read"),
        pytest.param(
            ExclusionRead(exclusions=[Exclusion(phrase="Excluding grants", left_out=["grants"])]),
            True,
            id="same-blanks",
        ),
        pytest.param(ExclusionRead(exclusions=[]), False, id="reads-apart"),
    ],
)
def test_an_exclusion_read_is_stored_only_when_its_samples_agree(monkeypatch, second, agreed):
    """The read samples twice, under the client's seed and the next one. Samples that blank
    and disclose the same stretches agree, however the model wrote them, and the first is
    stored; samples that read apart serve the first and store nothing, so the next request
    reads again."""
    llm, models = _llm(
        monkeypatch, MemoryExtractionStore(max_bytes=1024 * 1024), answers={1: _READ, 2: second}
    )

    read, usage, read_agreed = asyncio.run(read_exclusions(_READ_QUERY, llm=llm))

    assert read == _READ
    assert read_agreed is agreed
    assert sorted(models.seeds) == [1, 2]
    assert usage is not None and usage["input_tokens"] == 22
    assert stored_exclusion_read(_READ_QUERY, llm=llm) == (_READ if agreed else None)


def _single_sample_key(query: str, *, version: str, year: int, identity: dict) -> str:
    """The key a store recorded before agreement held, whose document has no `samples`."""
    document = {"query": query, "version": version, "year": year, "identity": identity}
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "extract_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_an_extraction_stored_before_agreement_is_a_miss_and_reads_the_model(monkeypatch):
    """A reading keyed by the document without the sampling policy was never agreed, so
    replaying it as agreed would skip the check. It misses and two samples read afresh."""
    store = MemoryExtractionStore(max_bytes=1024 * 1024)
    llm, models = _llm(monkeypatch, store)
    store.put(
        _single_sample_key(
            _QUERY,
            version=EXTRACT_VERSION,
            year=extract_module.date.today().year,
            identity=llm.sampling_identity(),
        ),
        _OTHER_ANSWER.model_dump_json(),
    )

    result, _ = _extract(_QUERY, llm)

    assert models.prompts == [_QUERY] * 2
    assert result.extraction.variables == ["GDP"]
    assert result.agreed


def test_an_exclusion_read_stored_before_agreement_is_not_found(monkeypatch):
    store = MemoryExtractionStore(max_bytes=1024 * 1024)
    llm, _models = _llm(monkeypatch, store)
    staged = for_stage(llm, "exclusion_read")
    assert isinstance(staged, LLM)
    store.put(
        _single_sample_key(
            "GDP excluding grants",
            version=EXCLUSION_READ_VERSION,
            year=extract_module.date.today().year,
            identity=staged.sampling_identity(),
        ),
        ExclusionRead(exclusions=[]).model_dump_json(),
    )

    assert stored_exclusion_read("GDP excluding grants", llm=llm) is None


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


def test_a_file_store_skips_a_malformed_line_and_reads_the_lines_after_it(tmp_path, caplog):
    path = tmp_path / "extractions.jsonl"
    path.write_bytes(b'{"key": "k1", "value": "v1"}\nnot json\n{"key": "k2", "value": "v2"}\n')

    store = FileExtractionStore(path)
    # A miss reads on from past the bad line, so the line is logged once.
    store.get("k3")

    assert [store.get(key) for key in ("k1", "k2")] == ["v1", "v2"]
    assert [r.name for r in caplog.records] == ["search.engine.extraction_store"]


def test_the_memory_store_evicts_the_least_recently_read_entry_past_its_byte_budget():
    entry_bytes = len("k1") + len("v" * 10)
    store = MemoryExtractionStore(max_bytes=2 * entry_bytes)
    store.put("k1", "v" * 10)
    store.put("k2", "v" * 10)
    assert store.get("k1") is not None

    store.put("k3", "v" * 10)

    assert [store.get(key) is not None for key in ("k1", "k2", "k3")] == [True, False, True]
