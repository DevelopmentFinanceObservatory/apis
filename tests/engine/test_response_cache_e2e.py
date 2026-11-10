"""Pins both response-cache layers through the offline request path."""

from __future__ import annotations

import asyncio
import time
from datetime import date
from unittest.mock import patch

import pytest

from graph_generation import UNKNOWN_GENERATION, FakeGeneration, GenerationProvider
from search.engine import binders, response_cache_flow
from search.engine.bind import _UniversalBindOutput
from search.engine.binders import JevBinder, jev_client
from search.engine.clients.llm import LLM
from search.engine.core import ResolveRecord, resolve_async, resolve_events
from search.engine.degraded_reads import fail_open_reads, note_degraded_read
from search.engine.errors import GraphInfraError, LLMInfraError
from search.engine.events import DoneEvent, ResponseEvent
from search.engine.exclusion_read import Exclusion, ExclusionRead
from search.engine.extract import ExtractedDate, ExtractedEntity, Extraction
from search.engine.extraction_store import MemoryExtractionStore
from search.engine.quantity_read import QuantityOutput
from search.engine.response_cache import ResponseCache
from search.engine.response_cache_keys import layer2_key
from search.engine.time_bind import TIME_TEXT_NOT_PARSED
from search.models import (
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
)
from series_coverage.schema import RoleNames
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine._binders import FakeAsk, elected_reply
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import EMPTY_SDMX, make_request, member_walk_reader, offline_events
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import PINNED_DATE, FakeGraph, FakeLLM

_QUERY_A = "fertility rate in Kenya"
_QUERY_B = "fertility rate for Kenya"


class _CountingLLM:
    """Count schema calls and assign distinct usage totals to each call."""

    def __init__(self, inner: FakeLLM) -> None:
        self._inner = inner
        self.calls: list[str] = []

    async def generate_structured(self, *, prompt: str, system: str, schema):
        self.calls.append(schema.__name__)
        parsed, _usage = await self._inner.generate_structured(
            prompt=prompt, system=system, schema=schema
        )
        usage = {"input_tokens": 1000 + len(self.calls), "output_tokens": 0, "cached_tokens": 0}
        return parsed, usage


class _AliasedLLM(FakeLLM):
    """Return the canonical query's extraction for an alias query."""

    def __init__(self, *, alias_query: str, canonical_query: str) -> None:
        super().__init__()
        self._alias_query = alias_query
        self._canonical_query = canonical_query

    async def generate_structured(self, *, prompt: str, system: str, schema):
        if schema is Extraction and prompt == self._alias_query:
            prompt = self._canonical_query
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def _run(
    request: ResolveRequest,
    *,
    llm,
    cache: ResponseCache | None,
    graph: FakeGraph | None = None,
    sdmx=EMPTY_SDMX,
    generation: GenerationProvider = UNKNOWN_GENERATION,
    materialise_graph: FakeMintGraphReader | None = None,
    record: ResolveRecord | None = None,
):
    """Run `resolve_async` with offline fakes and a pinned date. The mint's member walk reads
    `graph` unless `materialise_graph` replaces it."""
    graph = graph if graph is not None else FakeGraph()
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        return asyncio.run(
            resolve_async(
                request,
                graph=graph,
                llm=llm,
                sdmx=sdmx,
                response_cache=cache,
                generation=generation,
                record=record,
                materialise_graph=(
                    materialise_graph
                    if materialise_graph is not None
                    else member_walk_reader(graph)
                ),
            )
        )


def test_identical_requests_share_one_warm_cache():
    """Two identical requests cost four LLM calls in total, not eight: two extraction samples,
    one quantity read and one bind on the first, none on the second."""
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    r1 = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    r2 = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    assert llm.calls == ["Extraction", "Extraction", "QuantityOutput", "_UniversalBindOutput"]
    assert r1.status == "definite"
    assert r2.status == "definite"
    assert r2.parts == r1.parts


def test_differently_worded_single_variable_queries_keep_separate_results():
    """The full query grounds graph values, so matching extractions can answer differently."""
    llm = _CountingLLM(_AliasedLLM(alias_query=_QUERY_B, canonical_query=_QUERY_A))
    cache = ResponseCache()
    r1 = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    r2 = _run(make_request(_QUERY_B), llm=llm, cache=cache)
    assert llm.calls == ["Extraction", "Extraction", "QuantityOutput", "_UniversalBindOutput"] * 2

    # Each response retains its request's wording.
    assert r2.query_echo.raw_query == _QUERY_B
    assert r1.query_echo.raw_query == _QUERY_A

    # Diagnostics report this request's trace and LLM usage.
    assert r2.diagnostics.trace_id != r1.diagnostics.trace_id
    assert r2.diagnostics.llm_usage != r1.diagnostics.llm_usage
    assert r2.diagnostics.llm_usage is not None and r1.diagnostics.llm_usage is not None
    assert r1.status == r2.status == "definite"
    assert r1.rendered_sentence == r2.rendered_sentence


_SECOND_USAGE = {"input_tokens": 7, "output_tokens": 3, "cached_tokens": 0}


class _SeededLLM(LLM):
    """The recorded bank answered under this client's seed. With `disagreeing` set, a sample
    under any seed but 1 reads the measure as `disagreeing`. The exclusion read finds nothing
    left out, and with `read_disagreeing` set a sample under any seed but 1 reads a stretch
    the query does not hold, which is disclosed and so changes the reading. Every sample under
    a seed but 1 reports `_SECOND_USAGE`. Records each extraction's and each read's seed, in
    lists its reseeded copies share."""

    def __init__(
        self,
        *,
        disagreeing: str | None,
        store: MemoryExtractionStore,
        read_disagreeing: bool = False,
    ) -> None:
        super().__init__(seed=1, extraction_store=store)
        self._bank = FakeLLM()
        self._disagreeing = disagreeing
        self._read_disagreeing = read_disagreeing
        self.extraction_seeds: list[int | None] = []
        self.read_seeds: list[int | None] = []

    async def generate_structured(self, *, prompt: str, system: str, schema):
        if schema is ExclusionRead:
            self.read_seeds.append(self._seed)
            usage = None if self._seed == 1 else dict(_SECOND_USAGE)
            if self._read_disagreeing and self._seed != 1:
                stretch = Exclusion(phrase="excluding grants", left_out=["grants"])
                return ExclusionRead(exclusions=[stretch]), usage
            return ExclusionRead(exclusions=[]), usage
        answer, usage = await self._bank.generate_structured(
            prompt=prompt, system=system, schema=schema
        )
        if schema is Extraction:
            self.extraction_seeds.append(self._seed)
            if self._seed != 1:
                usage = dict(_SECOND_USAGE)
            if self._disagreeing is not None and self._seed != 1:
                answer = answer.model_copy(update={"variables": [self._disagreeing]})
        return answer, usage


def _served_from(response) -> str | None:
    steps = set(response.diagnostics.timing_ms.by_step or {})
    return next((step for step in steps if step.startswith("response_cache_layer")), None)


def test_samples_that_disagree_publish_no_layer1_entry_and_store_no_reading():
    """Two samples that read the query apart serve the first. Layer 1 and the extraction
    store keep nothing under the wording, so the repeat extracts again, and layer 2, keyed on
    the reading itself, answers it."""
    store = MemoryExtractionStore(max_bytes=1024 * 1024)
    llm = _SeededLLM(disagreeing="birth rate", store=store)
    cache = ResponseCache()
    record = ResolveRecord()

    first = _run(make_request(_QUERY_A), llm=llm, cache=cache, record=record)
    second = _run(make_request(_QUERY_A), llm=llm, cache=cache)

    assert sorted(llm.extraction_seeds) == [1, 1, 2, 2]
    assert store._entries == {}
    # The response leaves the second samples out; the request's record carries them.
    assert record.second_sample_usage == _SECOND_USAGE
    assert record.extraction_agreed is False
    assert _served_from(first) is None
    assert _served_from(second) == "response_cache_layer2"
    assert second.parts == first.parts


def test_samples_that_agree_publish_both_layers_and_a_stored_reading_publishes_layer1():
    """Agreeing samples store the reading and both layers. A later request whose cache lost
    the answer reads the stored reading with no model call, which counts as agreed, so its
    answer reaches layer 1 again."""
    store = MemoryExtractionStore(max_bytes=1024 * 1024)
    llm = _SeededLLM(disagreeing=None, store=store)
    cache = ResponseCache()

    _run(make_request(_QUERY_A), llm=llm, cache=cache)
    repeat = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    assert sorted(llm.extraction_seeds) == [1, 2]
    assert len(store._entries) == 1
    assert _served_from(repeat) == "response_cache_layer1"

    fresh_cache = ResponseCache()
    from_store = _run(make_request(_QUERY_A), llm=llm, cache=fresh_cache)
    from_layer1 = _run(make_request(_QUERY_A), llm=llm, cache=fresh_cache)

    assert sorted(llm.extraction_seeds) == [1, 2], "the store answered without a model call"
    assert _served_from(from_store) is None
    assert _served_from(from_layer1) == "response_cache_layer1"
    assert from_layer1.parts == from_store.parts


@pytest.mark.reads_exclusions
@pytest.mark.parametrize("read_disagreeing", [True, False], ids=["read-disagrees", "both-agree"])
def test_an_exclusion_read_whose_samples_disagree_publishes_no_layer1_entry(read_disagreeing):
    """The exclusion read decides the text extraction reads, so its two samples must agree
    too. A disagreeing read answers from its first sample and the extraction of that text
    agrees and is stored, but neither layer 1 nor the store keeps the read, so the repeat
    reads again and layer 2 answers it. A read that agrees is stored beside the extraction
    and the repeat is a layer-1 hit."""
    store = MemoryExtractionStore(max_bytes=1024 * 1024)
    llm = _SeededLLM(disagreeing=None, store=store, read_disagreeing=read_disagreeing)
    cache = ResponseCache()

    first = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    second = _run(make_request(_QUERY_A), llm=llm, cache=cache)

    assert sorted(llm.read_seeds) == ([1, 1, 2, 2] if read_disagreeing else [1, 2])
    assert sorted(llm.extraction_seeds) == [1, 2]
    stored_reads = [value for value in store._entries.values() if "exclusions" in value]
    assert len(stored_reads) == (0 if read_disagreeing else 1)
    assert first.status == "definite"
    assert _served_from(first) is None
    assert _served_from(second) == (
        "response_cache_layer2" if read_disagreeing else "response_cache_layer1"
    )
    assert second.parts == first.parts


def test_buffered_include_observations_is_cached():
    """A buffered request carries its observations in the response it stores, so two
    identical requests cost one extraction, one quantity read and one bind, and the hit equals
    the fresh answer except for its timing."""
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    request = ResolveRequest(
        input=RawTextInput(query=_QUERY_A),
        response_options=ResolveOptions(include_observations=True),
    )
    r1 = _run(request, llm=llm, cache=cache)
    r2 = _run(request, llm=llm, cache=cache)
    assert llm.calls == ["Extraction", "Extraction", "QuantityOutput", "_UniversalBindOutput"]
    assert r1.status == "definite"
    assert r2.parts == r1.parts
    assert set(r2.diagnostics.timing_ms.by_step or {}) == {"coverage", "response_cache_layer1"}


def test_a_transport_with_trailing_events_replays_its_bundle():
    """Over SSE, observations stream after the response event. The scope stores the whole
    event bundle at `done`, and an identical streamed request replays it without a model
    call, trailing events included."""
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    request = ResolveRequest(
        input=RawTextInput(query=_QUERY_A),
        response_options=ResolveOptions(include_observations=True),
    )

    async def _drain() -> list[str]:
        return [
            event.kind
            async for event in resolve_events(
                request,
                graph=FakeGraph(),
                llm=llm,
                sdmx=EMPTY_SDMX,
                response_cache=cache,
                stream_slot_options=True,
                materialise_graph=member_walk_reader(FakeGraph()),
            )
        ]

    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        first = asyncio.run(_drain())
        second = asyncio.run(_drain())
    assert "observations" in first
    assert llm.calls == ["Extraction", "Extraction", "QuantityOutput", "_UniversalBindOutput"]
    assert second == first[first.index("response") :]
    assert cache.stats().hits == 1


class _QuantityReadFailsLLM(FakeLLM):
    """The recorded bank, with every quantity read failing as a transport error does."""

    async def generate_structured(self, *, prompt: str, system: str, schema):
        if schema is QuantityOutput:
            raise LLMInfraError("the quantity read failed")
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def _lose_withheld_re_read(monkeypatch) -> FakeLLM:
    from search.engine.region import pipeline

    async def _lost_re_read(result, graph, *, deadline):
        note_degraded_read("withheld re-read")
        return result

    monkeypatch.setattr(pipeline, "with_withheld_warning", _lost_re_read)
    return FakeLLM()


def _lose_quantity_read(monkeypatch) -> FakeLLM:
    return _QuantityReadFailsLLM()


@pytest.mark.parametrize(
    "lose", [_lose_withheld_re_read, _lose_quantity_read], ids=["withheld-re-read", "quantity-read"]
)
def test_a_first_answer_that_swallowed_a_read_is_not_stored(monkeypatch, lose):
    """A read the pipeline swallows with no warning, the withheld-registry re-read or the
    quantity read, marks the part degraded, so neither layer stores the answer and the same
    request runs the pipeline again. Without the lost read the second request is a hit."""
    llm = _CountingLLM(lose(monkeypatch))
    cache = ResponseCache()
    r1 = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    r2 = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    assert r1.status == r2.status == "definite"
    assert r1.diagnostics.warnings == r2.diagnostics.warnings == []
    assert llm.calls == [
        "Extraction",
        "Extraction",
        "QuantityOutput",
        "_UniversalBindOutput",
        "Extraction",
        "Extraction",
        "QuantityOutput",
        "_UniversalBindOutput",
    ]
    assert cache.stats().entries == 0


def _cut_withheld_re_read(monkeypatch) -> list[str]:
    """Run the withheld re-read as a read its own fail-open budget cuts. Returns the list it
    notes each cut in."""
    from search.engine.region import pipeline

    fired: list[str] = []

    async def _cut_re_read(result, graph, *, deadline):
        with fail_open_reads("withheld re-read", budget_s=0.0):
            fired.append("withheld re-read")
            raise TimeoutError("the re-read ran past its budget")
        return result

    monkeypatch.setattr(pipeline, "with_withheld_warning", _cut_re_read)
    return fired


def _buffered(cache: ResponseCache | None) -> tuple[dict, DoneEvent]:
    """The buffered answer to `_QUERY_A`, less what describes one run, and its `done`."""
    graph = FakeGraph()
    events = offline_events(
        make_request(_QUERY_A),
        graph=graph,
        materialise_graph=member_walk_reader(graph),
        response_cache=cache,
    )
    response = next(e.response for e in events if isinstance(e, ResponseEvent))
    document = response.model_dump(mode="json")
    for key in ("trace_id", "timing_ms", "llm_usage"):
        document["diagnostics"].pop(key, None)
    for part in document.get("parts", []):
        part.pop("timing_ms", None)
    done = events[-1]
    assert isinstance(done, DoneEvent)
    return document, done


def test_a_first_answer_whose_read_its_budget_cut_is_served_whole_and_not_stored(monkeypatch):
    """A read cut by its own fail-open budget changes nothing the answer sends: the body, its
    warnings and the `done` flags (`degraded_leg` sets the buffered route's header) equal a
    run whose read answered. Neither layer stores it."""
    clean, clean_done = _buffered(None)
    fired = _cut_withheld_re_read(monkeypatch)
    cache = ResponseCache()

    cut, cut_done = _buffered(cache)

    assert fired == ["withheld re-read"]
    assert cut == clean
    assert (
        (cut_done.degraded_leg, cut_done.timed_out, cut_done.warnings)
        == (
            clean_done.degraded_leg,
            clean_done.timed_out,
            clean_done.warnings,
        )
        == (False, False, [])
    )
    assert cache.stats().entries == 0


class _FailingRelatedSubjects(FakeGraph):
    """The related-measures read's reverse read of the group fails."""

    def node_subjects_many(self, *args, **kwargs):
        raise GraphInfraError("the reverse read failed")


@pytest.mark.parametrize(
    ("graph_type", "entries"),
    [(FakeGraph, 2), (_FailingRelatedSubjects, 0)],
    ids=["related-read", "related-read-failed"],
)
def test_a_first_answer_whose_related_read_failed_is_not_stored(graph_type, entries):
    """A failed related-measures read leaves the answer without related measures and no
    warning, so it marks the part degraded and neither layer stores it."""
    cache = ResponseCache()
    response = _run(
        make_request("population of Kenya"),
        llm=FakeLLM(),
        cache=cache,
        graph=graph_type(),
    )
    assert response.status == "definite"
    assert (response.parts[0].spec.resolved.related_group is None) is (entries == 0)
    assert cache.stats().entries == entries


def test_cache_disabled_by_config_behaves_as_uncached(search_settings):
    """SEARCH_RESPONSE_CACHE_ENABLED=False behaves exactly as if the cache
    did not exist, even with a real ResponseCache instance handed in: two
    identical requests still cost four LLM calls."""
    # Patched on the module object response_cache_flow.CacheScope actually
    # reads from, rather than by dotted string path: this suite's full run
    # has more than one "search.engine.config" module object in play (a
    # pre-existing property of this monorepo/worktree's import setup, not
    # something this test can fix), and a string-path monkeypatch resolves
    # via its own fresh import, which is not guaranteed to land on the same
    # object CacheScope's own `config` name is bound to.
    search_settings(response_cache_enabled=False)
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    _run(make_request(_QUERY_A), llm=llm, cache=cache)
    _run(make_request(_QUERY_A), llm=llm, cache=cache)
    assert llm.calls == [
        "Extraction",
        "Extraction",
        "QuantityOutput",
        "_UniversalBindOutput",
        "Extraction",
        "Extraction",
        "QuantityOutput",
        "_UniversalBindOutput",
    ]
    stats = cache.stats()
    assert stats.hits == 0
    assert stats.misses == 0
    assert stats.entries == 0


def test_identical_parsed_requests_hit_layer2():
    """A parsed request keys layer 2 on its whole input: the second identical request
    reads no model, keeps the parsed echo and equals the first."""
    cache = ResponseCache()
    llm = _CountingLLM(FakeLLM())
    request = ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
            ],
            raw_query="fertility rate in Kenya",
        )
    )
    r1 = _run(request, llm=llm, cache=cache)
    calls_after_first = list(llm.calls)
    r2 = _run(request, llm=llm, cache=cache)
    assert r1.status == "definite"
    assert llm.calls == calls_after_first
    assert r2.parts == r1.parts
    assert r2.query_echo == r1.query_echo
    assert r2.query_echo.entry_path == "parsed"
    assert set(r2.diagnostics.timing_ms.by_step or {}) == {"coverage", "response_cache_layer2"}
    stats = cache.stats()
    assert (stats.hits, stats.entries) == (1, 1)


def _streamed_kinds_and_response(request: ResolveRequest, *, cache: ResponseCache):
    """Stream `request` over a transport with no trailing events; its event kinds and the
    response event's response."""

    async def _drain():
        return [
            event
            async for event in resolve_events(
                request,
                graph=FakeGraph(),
                llm=FakeLLM(),
                sdmx=EMPTY_SDMX,
                response_cache=cache,
                stream_slot_options=True,
                materialise_graph=member_walk_reader(FakeGraph()),
            )
        ]

    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        events = asyncio.run(_drain())
    [response] = [event.response for event in events if event.kind == "response"]
    return [event.kind for event in events], response


@pytest.mark.parametrize(
    "request_input",
    [
        RawTextInput(query=_QUERY_A),
        ParsedInput(
            parts=[
                ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
            ],
            raw_query=_QUERY_A,
        ),
    ],
    ids=["raw_text", "parsed"],
)
def test_a_streamed_hit_emits_the_answer_without_progress_events(request_input):
    """A streamed hit sends the stored response and `done`, as a raw-text layer-1 hit
    always has: the progress events (`interpretation`, `step`, `sentence`) describe a
    resolution that did not run. The response carries every part, with its sentence plan,
    as the fresh run's does, which is what a client renders the answer from."""
    cache = ResponseCache()
    request = ResolveRequest(input=request_input)

    fresh_kinds, fresh = _streamed_kinds_and_response(request, cache=cache)
    hit_kinds, hit = _streamed_kinds_and_response(request, cache=cache)

    assert {"step", "sentence"} <= set(fresh_kinds)
    assert hit_kinds == ["response", "done"]
    assert hit.parts == fresh.parts
    assert hit.sentence_plan == fresh.sentence_plan
    assert cache.stats().hits == 1


def test_parsed_requests_differing_in_one_field_miss_layer2():
    """Two parsed requests that differ only in a part's unit text keep separate entries."""
    cache = ResponseCache()

    def _request(unit_text: str | None) -> ResolveRequest:
        return ResolveRequest(
            input=ParsedInput(
                parts=[
                    ParsedPart(
                        variable_text="fertility rate",
                        entities=[ParsedEntity(text="Kenya")],
                        unit_text=unit_text,
                    )
                ]
            )
        )

    _run(_request(None), llm=FakeLLM(), cache=cache)
    _run(_request("percent"), llm=FakeLLM(), cache=cache)
    stats = cache.stats()
    assert (stats.hits, stats.entries) == (0, 2)


# ---------------------------------------------------------------------------
# role_text.contradicts: a raw-text signal Extraction alone does not carry
# ---------------------------------------------------------------------------

_ROLE_SV_DCID = "TESTFIXTURE/ResponseCacheE2E/Aid"
_KEN = "country/KEN"


class _FixedExtractionLLM:
    """Ignores the prompt for both LLM stages: Extraction always names one
    role-bearing entity (Kenya, the recipient, with the query's own "aid"
    token as cue so extract()'s cue grounding keeps the role for every
    wording this suite tries), and the bind stage always elects the sole
    candidate shape with no constraint bindings. Extraction, and everything
    derived from it, is byte-identical on every call. The only thing
    that can differ between two calls is the door's own reading of the raw
    query text (`extract.veto_words`, `raw_text._states_direction`), which
    this double never touches.
    """

    async def generate_structured(self, *, prompt: str, system: str, schema):
        if schema is Extraction:
            entity = ExtractedEntity(text="Kenya", span="Kenya", role="recipient", cue="aid")
            extraction = Extraction(
                kind="resolution",
                variables=["aid"],
                entities=[entity],
                dates=[],
                dropped_qualifiers=[],
            )
            return extraction, None
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert issubclass(schema, _UniversalBindOutput)
        return elected_reply(0), None


def _role_bearing_graph() -> FakeGraph:
    return FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _ROLE_SV_DCID: {
                "label": "Test aid flow",
                "arcs": {
                    "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
                    "measuredProperty": {"nodes": [{"dcid": "amount"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                    "observationProperties": {"nodes": [{"dcid": "creditor"}, {"dcid": "debtor"}]},
                },
            },
        },
        detect={"aid": {"svs": [_ROLE_SV_DCID], "cosine_scores": [0.9]}},
        resolve={"Kenya": _KEN},
        obs={},
    )


def _role_bearing_member_graph() -> FakeMintGraphReader:
    """The member walk the mint runs, over the one role-bearing StatVar."""
    return FakeMintGraphReader(
        arcs={
            _ROLE_SV_DCID: {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("EconomicActivity",),
                "measuredProperty": ("amount",),
                "statType": ("measuredValue",),
                "observationProperties": ("creditor", "debtor"),
            }
        }
    )


# The coverage row the destination-bound confirmation reads for both tests
# below: _ROLE_SV_DCID's own observationProperties, "creditor"/"debtor",
# mapped onto this deployment's fixed origin/destination axis labels.
_ROLE_NAMES = {_ROLE_SV_DCID: RoleNames(donor="creditor", recipient="debtor")}
_ROLE_COVERAGE_ROWS = [(_ROLE_SV_DCID, "f1", "country/FRA", _KEN, [2020])]


def test_role_direction_contradiction_is_not_masked_by_a_shared_extraction(monkeypatch):
    """The same extraction with opposing role words must not share a layer-2 result."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch, _ROLE_COVERAGE_ROWS, role_names=_ROLE_NAMES
    )
    llm = _CountingLLM(_FixedExtractionLLM())
    cache = ResponseCache()
    r1 = _run(
        make_request("aid to Kenya"),
        llm=llm,
        cache=cache,
        graph=_role_bearing_graph(),
        sdmx=FakeSdmxClient(observations={}),
        materialise_graph=_role_bearing_member_graph(),
    )
    assert r1.status == "definite", r1

    r2 = _run(
        make_request("aid from Kenya"),
        llm=llm,
        cache=cache,
        graph=_role_bearing_graph(),
        sdmx=FakeSdmxClient(observations={}),
        materialise_graph=_role_bearing_member_graph(),
    )
    # The role contradiction changes the layer-2 key, so bind runs again.
    assert llm.calls == [
        "Extraction",
        "Extraction",
        "QuantityOutput",
        "_UniversalBindOutput",
        "Extraction",
        "Extraction",
        "QuantityOutput",
        "_UniversalBindOutput",
    ]
    assert r2.status == "no_data", r2


def test_role_direction_agreement_keeps_distinct_query_sources_separate(monkeypatch):
    """A shared role verdict leaves each query's graph-value grounding independent."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch, _ROLE_COVERAGE_ROWS, role_names=_ROLE_NAMES
    )
    llm = _CountingLLM(_FixedExtractionLLM())
    cache = ResponseCache()
    r1 = _run(
        make_request("aid to Kenya"),
        llm=llm,
        cache=cache,
        graph=_role_bearing_graph(),
        sdmx=FakeSdmxClient(observations={}),
        materialise_graph=_role_bearing_member_graph(),
    )
    r2 = _run(
        make_request("please show the aid to Kenya"),
        llm=llm,
        cache=cache,
        graph=_role_bearing_graph(),
        sdmx=FakeSdmxClient(observations={}),
        materialise_graph=_role_bearing_member_graph(),
    )
    assert llm.calls == ["Extraction", "Extraction", "QuantityOutput", "_UniversalBindOutput"] * 2
    assert r1.status == "definite", r1
    assert r2.status == "definite", r2
    assert r1.rendered_sentence == r2.rendered_sentence


class _CuedRecipientLLM(_FixedExtractionLLM):
    """`_FixedExtractionLLM` reading two measures, aid with Kenya's cue "to" and GDP with
    Uganda. "aid to Kenya" holds the cue and so states Kenya's direction; "aid for Kenya"
    does not, so its role is read off the measure."""

    async def generate_structured(self, *, prompt: str, system: str, schema):
        if schema is Extraction:
            entities = [
                ExtractedEntity(
                    text="Kenya", span="Kenya", variables=[0], role="recipient", cue="to"
                ),
                ExtractedEntity(text="Uganda", span="Uganda", variables=[1]),
            ]
            return Extraction(kind="resolution", variables=["aid", "GDP"], entities=entities), None
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


_UGA = "country/UGA"
_PLAIN_ARCS = {
    "populationType": {"nodes": [{"dcid": "Person"}]},
    "measuredProperty": {"nodes": [{"dcid": "count"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
}


def _subject_graph() -> FakeGraph:
    """Two subject-only series, aid holding Kenya and GDP Uganda, so a stated direction is
    discarded."""
    facet = [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 6}]
    return FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _UGA: {"label": "Uganda", "type": "Country"},
            "Count_Aid": {"label": "Test aid count", "arcs": _PLAIN_ARCS},
            "Count_GDP": {"label": "Test GDP count", "arcs": _PLAIN_ARCS},
        },
        detect={
            "aid": {"svs": ["Count_Aid"], "cosine_scores": [0.9]},
            "GDP": {"svs": ["Count_GDP"], "cosine_scores": [0.9]},
        },
        resolve={"Kenya": _KEN, "Uganda": _UGA},
        obs={f"Count_Aid|{_KEN}": facet, f"Count_GDP|{_UGA}": facet},
    )


@pytest.mark.parametrize(
    "queries",
    [
        ("aid to Kenya and GDP of Uganda", "aid for Kenya and GDP of Uganda"),
        ("aid for Kenya and GDP of Uganda", "aid to Kenya and GDP of Uganda"),
    ],
)
def test_a_stated_direction_keeps_its_own_layer2_entry(queries):
    """One extraction whose cue one wording holds and the other does not shares no layer-2
    entry: the sentence discloses only the stated direction as discarded, so each warm
    answer reads as its cold one, in either order."""

    def run(query: str, cache: ResponseCache | None) -> str:
        response = _run(
            make_request(query), llm=_CuedRecipientLLM(), cache=cache, graph=_subject_graph()
        )
        return response.rendered_sentence

    cold = {query: run(query, None) for query in queries}
    assert "You named Kenya as a recipient" in cold["aid to Kenya and GDP of Uganda"], cold
    assert "You named" not in cold["aid for Kenya and GDP of Uganda"], cold
    cache = ResponseCache()
    assert {query: run(query, cache) for query in queries} == cold


class _ReadingLLM:
    """Reads each query as the extraction `readings` holds for it; the bind stage elects
    the sole candidate."""

    def __init__(self, readings: dict[str, Extraction]) -> None:
        self._readings = readings

    async def generate_structured(self, *, prompt: str, system: str, schema):
        if schema is Extraction:
            return self._readings[prompt], None
        if schema is QuantityOutput:
            return NO_QUANTITY
        return elected_reply(0), None


def _repeated_measure_reading(variables: list[str]) -> Extraction:
    """Kenya, the recipient, on the measure at position 1 of `variables`, and Uganda on the
    one at position 2."""
    return Extraction(
        kind="resolution",
        variables=variables,
        entities=[
            ExtractedEntity(text="Kenya", span="Kenya", variables=[1], cue="to", role="recipient"),
            ExtractedEntity(text="Uganda", span="Uganda", variables=[2]),
        ],
    )


@pytest.mark.parametrize(
    ("second", "same_key"),
    [
        pytest.param(["ODA", "ODA", "GDP"], True, id="same-scoping"),
        # Kenya scopes ODA in the first reading and GDP in the second.
        pytest.param(["ODA", "GDP", "GDP"], False, id="other-scoping"),
    ],
)
def test_the_layer2_key_states_the_parts_each_place_scopes(monkeypatch, second, same_key):
    """Two readings with the same distinct measures and the same entities, whose repeated
    measure moves Kenya onto another part, key apart; the same reading under the other
    wording shares its key."""
    keys: list[str] = []

    def recording_key(**kwargs):
        keys.append(layer2_key(**kwargs))
        return keys[-1]

    monkeypatch.setattr(response_cache_flow, "layer2_key", recording_key)
    first_query, second_query = "ODA to Kenya and GDP of Uganda", "ODA to Kenya, GDP of Uganda"
    llm = _ReadingLLM(
        {
            first_query: _repeated_measure_reading(["ODA", "ODA", "GDP"]),
            second_query: _repeated_measure_reading(second),
        }
    )
    cache = ResponseCache()
    for query in (first_query, second_query):
        _run(make_request(query), llm=llm, cache=cache)

    first, other = keys
    assert (first == other) is same_key


class _UnparsedDateLLM(_FixedExtractionLLM):
    """`_FixedExtractionLLM`, adding a relative date worded "lately" to a query holding the
    word. The time parser cannot read it, so extraction drops the date and warns: the
    resolved extraction matches the dateless query's, and only the warning differs."""

    async def generate_structured(self, *, prompt: str, system: str, schema):
        parsed, usage = await super().generate_structured(
            prompt=prompt, system=system, schema=schema
        )
        if schema is Extraction and "lately" in prompt:
            parsed = parsed.model_copy(
                update={"dates": [ExtractedDate(kind="range", relative="lately")]}
            )
        return parsed, usage


def _warning_codes(response) -> list[str]:
    return [warning.code for warning in response.diagnostics.warnings] + [
        warning.code for part in response.parts for warning in part.warnings
    ]


@pytest.mark.parametrize(
    "first, second",
    [("aid to Kenya", "aid to Kenya lately"), ("aid to Kenya lately", "aid to Kenya")],
    ids=["dateless-first", "unparsed-first"],
)
def test_an_unparsed_date_warning_keeps_its_own_layer2_entry(monkeypatch, first, second):
    """A dropped relative date leaves the extraction the dateless query has, so only its
    `time_text_not_parsed` warning tells the two apart. Each answer carries the warning
    exactly when its own query held the wording, in either order."""
    patch_coverage(monkeypatch, _ROLE_COVERAGE_ROWS, role_names=_ROLE_NAMES)
    llm = _CountingLLM(_UnparsedDateLLM())
    cache = ResponseCache()
    responses = [
        _run(
            make_request(query),
            llm=llm,
            cache=cache,
            graph=_role_bearing_graph(),
            sdmx=FakeSdmxClient(observations={}),
            materialise_graph=_role_bearing_member_graph(),
        )
        for query in (first, second)
    ]

    assert llm.calls == ["Extraction", "Extraction", "QuantityOutput", "_UniversalBindOutput"] * 2
    for query, response in zip((first, second), responses, strict=True):
        assert response.status == "definite", response
        assert (TIME_TEXT_NOT_PARSED in _warning_codes(response)) == ("lately" in query)


# ---------------------------------------------------------------------------
# graph generation: a `GenerationProvider`'s stamp keys and ages both layers
# ---------------------------------------------------------------------------


def test_same_query_under_the_same_token_still_hits():
    """Identical requests under one generation token reuse both cache layers."""
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    generation = FakeGeneration(token="gen-a")
    r1 = _run(make_request(_QUERY_A), llm=llm, cache=cache, generation=generation)
    r2 = _run(make_request(_QUERY_A), llm=llm, cache=cache, generation=generation)
    assert llm.calls == ["Extraction", "Extraction", "QuantityOutput", "_UniversalBindOutput"]
    assert r1.parts == r2.parts


def test_advancing_the_token_between_requests_misses_both_layers():
    """Advancing the generation token changes both keys and reruns the pipeline."""
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    generation = FakeGeneration(token="gen-a")
    r1 = _run(make_request(_QUERY_A), llm=llm, cache=cache, generation=generation)

    generation.advance("gen-b")
    r2 = _run(make_request(_QUERY_A), llm=llm, cache=cache, generation=generation)

    assert llm.calls == [
        "Extraction",
        "Extraction",
        "QuantityOutput",
        "_UniversalBindOutput",
        "Extraction",
        "Extraction",
        "QuantityOutput",
        "_UniversalBindOutput",
    ]
    # Per-run timing differs; compare stable spec IDs.
    assert [p.spec.spec_id for p in r1.parts] == [p.spec.spec_id for p in r2.parts]
    # Both requests retain separate layer-1 and layer-2 entries.
    assert cache.stats().entries == 4


def test_unknown_and_known_stamps_never_share_an_entry():
    """Unknown and known generation stamps produce separate cache entries."""
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    r1 = _run(make_request(_QUERY_A), llm=llm, cache=cache)  # generation defaults to unknown
    r2 = _run(
        make_request(_QUERY_A), llm=llm, cache=cache, generation=FakeGeneration(token="gen-a")
    )
    assert llm.calls == [
        "Extraction",
        "Extraction",
        "QuantityOutput",
        "_UniversalBindOutput",
        "Extraction",
        "Extraction",
        "QuantityOutput",
        "_UniversalBindOutput",
    ]
    # Per-run timing differs; compare stable spec IDs.
    assert [p.spec.spec_id for p in r1.parts] == [p.spec.spec_id for p in r2.parts]


def test_written_entry_lifetime_matches_the_known_generations_max_age():
    """A write made under a known token's stamp lives for
    `GRAPH_CACHE_MAX_AGE_SECONDS`, not the store's own constructor `ttl_s`
    default (unknown-generation, much shorter): `CacheScope.store` always
    passes `max_age_s=stamp.max_age_s`."""
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    generation = FakeGeneration(token="gen-a")
    before = time.monotonic()
    _run(make_request(_QUERY_A), llm=llm, cache=cache, generation=generation)

    expected_max_age = FakeGeneration(token="gen-a").stamp().max_age_s
    entries = list(cache._store.values())
    assert entries  # the request stored at least one entry
    for entry in entries:
        assert abs(entry.expires_at - (before + expected_max_age)) < 5.0


@pytest.fixture
def jev_routed(search_settings, monkeypatch):
    """Every request routes to Jev (`binder="jev"`), with a fresh circuit breaker. Jev has
    no key, so each Jev call falls back to Gemini."""
    search_settings(binder="jev")
    monkeypatch.setattr(jev_client, "_BREAKER", jev_client.CircuitBreaker())


def test_a_request_whose_jev_calls_fell_back_is_not_cached(jev_routed):
    """With no TypeSafe key every Jev call falls back to Gemini, so the answer is not the
    Jev binder's its key names: the repeat runs the whole pipeline again."""
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    r1 = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    r2 = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    assert llm.calls == ["Extraction", "Extraction", "QuantityOutput", "_UniversalBindOutput"] * 2
    assert r1.status == r2.status == "definite"


def test_a_repeated_query_under_jev_hits_layer1_and_never_a_gemini_entry(
    search_settings, monkeypatch
):
    """A Jev answer is stored under the Jev binder's key: its repeat is a layer-1 hit that
    asks Jev nothing, and an answer the Gemini binder stored for the same query is not
    served for it."""
    monkeypatch.setattr(jev_client, "_BREAKER", jev_client.CircuitBreaker())
    ask = FakeAsk()
    monkeypatch.setattr(binders, "JevBinder", lambda: JevBinder(ask=ask))
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    _run(make_request(_QUERY_A), llm=llm, cache=cache)
    search_settings(binder="jev")
    gemini_calls = len(llm.calls)

    first = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    jev_requests = len(ask.requests)
    second = _run(make_request(_QUERY_A), llm=llm, cache=cache)

    assert llm.calls[gemini_calls:] == ["Extraction", "Extraction"]
    assert jev_requests == 2
    assert len(ask.requests) == jev_requests
    assert set(second.diagnostics.timing_ms.by_step or {}) == {"coverage", "response_cache_layer1"}
    assert second.parts == first.parts
