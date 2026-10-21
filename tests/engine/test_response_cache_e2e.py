"""Pins both response-cache layers through the offline request path."""

from __future__ import annotations

import asyncio
import time
from datetime import date
from unittest.mock import patch

import pytest

from graph_generation import UNKNOWN_GENERATION, FakeGeneration, GenerationProvider
from search.engine import binders
from search.engine.bind import _UniversalBindOutput
from search.engine.binders import JevBinder, jev_client
from search.engine.core import resolve_async, resolve_events
from search.engine.degraded_reads import note_degraded_read
from search.engine.errors import GraphInfraError, LLMInfraError
from search.engine.extract import ExtractedDate, ExtractedEntity, Extraction
from search.engine.quantity_read import QuantityOutput
from search.engine.response_cache import ResponseCache
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
from tests.engine._harness import EMPTY_SDMX, make_request, member_walk_reader
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
                materialise_graph=(
                    materialise_graph
                    if materialise_graph is not None
                    else member_walk_reader(graph)
                ),
            )
        )


def test_identical_requests_share_one_warm_cache():
    """Two identical requests cost three LLM calls in total, not six: one extract, one
    quantity read and one bind on the first, none on the second."""
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    r1 = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    r2 = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    assert llm.calls == ["Extraction", "QuantityOutput", "_UniversalBindOutput"]
    assert r1.status == "definite"
    assert r2.status == "definite"
    assert r2.parts == r1.parts


def test_differently_worded_query_hits_layer2():
    """Identical extractions share the bind result, while each query is extracted."""
    llm = _CountingLLM(_AliasedLLM(alias_query=_QUERY_B, canonical_query=_QUERY_A))
    cache = ResponseCache()
    r1 = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    r2 = _run(make_request(_QUERY_B), llm=llm, cache=cache)
    assert llm.calls == ["Extraction", "QuantityOutput", "_UniversalBindOutput", "Extraction"]

    # Each response retains its request's wording.
    assert r2.query_echo.raw_query == _QUERY_B
    assert r1.query_echo.raw_query == _QUERY_A

    # Diagnostics report this request's trace and LLM usage.
    assert r2.diagnostics.trace_id != r1.diagnostics.trace_id
    assert r2.diagnostics.llm_usage != r1.diagnostics.llm_usage
    assert r2.diagnostics.llm_usage is not None and r1.diagnostics.llm_usage is not None
    assert r2.diagnostics.llm_usage["input_tokens"] < r1.diagnostics.llm_usage["input_tokens"]

    # Everything the query itself determines is shared between the two.
    assert r1.parts == r2.parts
    assert r1.rendered_sentence == r2.rendered_sentence


def test_buffered_include_observations_is_cached():
    """A buffered request carries its observations in the response it stores, so two
    identical requests cost one extract, one quantity read and one bind, and the hit equals
    the fresh answer except for its timing."""
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    request = ResolveRequest(
        input=RawTextInput(query=_QUERY_A),
        response_options=ResolveOptions(include_observations=True),
    )
    r1 = _run(request, llm=llm, cache=cache)
    r2 = _run(request, llm=llm, cache=cache)
    assert llm.calls == ["Extraction", "QuantityOutput", "_UniversalBindOutput"]
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
    assert llm.calls == ["Extraction", "QuantityOutput", "_UniversalBindOutput"]
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
        "QuantityOutput",
        "_UniversalBindOutput",
        "Extraction",
        "QuantityOutput",
        "_UniversalBindOutput",
    ]
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
        "QuantityOutput",
        "_UniversalBindOutput",
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
    role-bearing entity (Kenya, destination, anchored on the query's own
    "aid" token so extract()'s _validate_role_spans passes for every
    wording this suite tries), and the bind stage always elects the sole
    candidate shape with no constraint bindings. Extraction, and everything
    derived from it, is byte-identical on every call. The only thing
    that can differ between two calls is contradicts()' own scan of the
    raw query text, read inside `_phase_place_bindings`, which this double
    never touches.
    """

    async def generate_structured(self, *, prompt: str, system: str, schema):
        if schema is Extraction:
            entity = ExtractedEntity(
                text="Kenya", span="Kenya", flow_role="destination", role_span="aid"
            )
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
        "QuantityOutput",
        "_UniversalBindOutput",
        "Extraction",
        "QuantityOutput",
        "_UniversalBindOutput",
    ]
    assert r2.status == "no_data", r2


def test_role_direction_agreement_still_hits_layer2(monkeypatch):
    """Different wordings with the same role verdict share the layer-2 result."""
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
    assert llm.calls == ["Extraction", "QuantityOutput", "_UniversalBindOutput", "Extraction"]
    assert r1.status == "definite", r1
    assert r2.status == "definite", r2
    assert r1.parts == r2.parts


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

    assert llm.calls == ["Extraction", "QuantityOutput", "_UniversalBindOutput"] * 2
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
    assert llm.calls == ["Extraction", "QuantityOutput", "_UniversalBindOutput"]
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
        "QuantityOutput",
        "_UniversalBindOutput",
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
        "QuantityOutput",
        "_UniversalBindOutput",
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
    assert llm.calls == ["Extraction", "QuantityOutput", "_UniversalBindOutput"] * 2
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

    assert llm.calls[gemini_calls:] == ["Extraction"]
    assert jev_requests == 2
    assert len(ask.requests) == jev_requests
    assert set(second.diagnostics.timing_ms.by_step or {}) == {"coverage", "response_cache_layer1"}
    assert second.parts == first.parts
