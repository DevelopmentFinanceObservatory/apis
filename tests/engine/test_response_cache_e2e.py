"""Pins both response-cache layers through the offline request path."""

from __future__ import annotations

import asyncio
import time
from datetime import date
from unittest.mock import patch

from graph_generation import UNKNOWN_GENERATION, FakeGeneration, GenerationProvider
from search.engine.bind import _UniversalBindOutput
from search.engine.core import resolve_async
from search.engine.extract import ExtractedEntity, Extraction
from search.engine.response_cache import ResponseCache
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
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import EMPTY_SDMX, make_request, member_walk_reader
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
    """Two identical requests cost two LLM calls in total, not four: one
    extract and one bind on the first, zero on the second."""
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    r1 = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    r2 = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    assert llm.calls == ["Extraction", "_UniversalBindOutput"]
    assert r1.status == "definite"
    assert r2.status == "definite"
    assert r2.parts == r1.parts


def test_differently_worded_query_hits_layer2():
    """Identical extractions share the bind result, while each query is extracted."""
    llm = _CountingLLM(_AliasedLLM(alias_query=_QUERY_B, canonical_query=_QUERY_A))
    cache = ResponseCache()
    r1 = _run(make_request(_QUERY_A), llm=llm, cache=cache)
    r2 = _run(make_request(_QUERY_B), llm=llm, cache=cache)
    assert llm.calls == ["Extraction", "_UniversalBindOutput", "Extraction"]

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


def test_include_observations_is_not_cached():
    """A request setting include_observations is never cached, on either
    layer: two identical such requests still cost four LLM calls."""
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    request = ResolveRequest(
        input=RawTextInput(query=_QUERY_A),
        response_options=ResolveOptions(include_observations=True),
    )
    _run(request, llm=llm, cache=cache)
    _run(request, llm=llm, cache=cache)
    assert llm.calls == ["Extraction", "_UniversalBindOutput", "Extraction", "_UniversalBindOutput"]
    stats = cache.stats()
    assert stats.hits == 0
    assert stats.entries == 0


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
    assert llm.calls == ["Extraction", "_UniversalBindOutput", "Extraction", "_UniversalBindOutput"]
    stats = cache.stats()
    assert stats.hits == 0
    assert stats.misses == 0
    assert stats.entries == 0


def test_non_raw_text_kind_is_unaffected():
    """A parsed-kind request never touches the response cache: the shared
    cache's counters stay at zero across two requests."""
    cache = ResponseCache()
    request = ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
            ]
        )
    )
    r1 = _run(request, llm=FakeLLM(), cache=cache)
    r2 = _run(request, llm=FakeLLM(), cache=cache)
    assert r1.status == "definite"
    assert r2.status == "definite"
    stats = cache.stats()
    assert stats.hits == 0
    assert stats.misses == 0
    assert stats.entries == 0


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
        assert schema is _UniversalBindOutput
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


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
    assert llm.calls == ["Extraction", "_UniversalBindOutput", "Extraction", "_UniversalBindOutput"]
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
    assert llm.calls == ["Extraction", "_UniversalBindOutput", "Extraction"]
    assert r1.status == "definite", r1
    assert r2.status == "definite", r2
    assert r1.parts == r2.parts


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
    assert llm.calls == ["Extraction", "_UniversalBindOutput"]
    assert r1.parts == r2.parts


def test_advancing_the_token_between_requests_misses_both_layers():
    """Advancing the generation token changes both keys and reruns the pipeline."""
    llm = _CountingLLM(FakeLLM())
    cache = ResponseCache()
    generation = FakeGeneration(token="gen-a")
    r1 = _run(make_request(_QUERY_A), llm=llm, cache=cache, generation=generation)

    generation.advance("gen-b")
    r2 = _run(make_request(_QUERY_A), llm=llm, cache=cache, generation=generation)

    assert llm.calls == ["Extraction", "_UniversalBindOutput", "Extraction", "_UniversalBindOutput"]
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
    assert llm.calls == ["Extraction", "_UniversalBindOutput", "Extraction", "_UniversalBindOutput"]
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
