"""Entry handlers forward the request's rate cache to variable and resubmit resolution.

Tests call each handler directly because `core.py` passes no cache. The resolvers are patched
at their call sites.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from entity_resolution import Admission
from materialise.rate_tables import RateTableCache
from search.engine.binders import GeminiBinder
from search.engine.concurrency import scoped_concurrency
from search.engine.config import settings
from search.engine.entry import dataset_resubmit, parsed, raw_text
from search.engine.entry.params import EntryParams
from search.engine.extract import Extraction, ExtractionResult
from search.engine.region.result import RegionResult
from search.models import (
    DatasetResubmitInput,
    ParsedInput,
    ParsedPart,
    RawTextInput,
    ResubmitCandidate,
    ResubmitCandidatesEntry,
    ResubmitIdentityEntry,
    ResubmitParsedEntry,
)
from tests.engine._base_identity import GDP_SHAPE, make_identity, subject_entity
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_INDIA = "country/IND"
_GDP_SV = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"


def _stub_region(variable_text: str | None) -> RegionResult:
    """Builds an empty region for handler tests."""
    return RegionResult(
        variable_text=variable_text,
        status="no_data",
        specs=(),
        no_data_reason="variable_not_resolved",
        warnings=(),
        timing_by_step={},
    )


def _params() -> EntryParams:
    return EntryParams(
        deadline=time.monotonic() + 30,
        max_candidates=5,
        pac=True,
        stream_option_coverage=False,
        include_option_coverage=False,
        stream_observations=False,
        include_observations=False,
        include_skills=False,
        skills_index=None,
        coverage=empty_coverage_snapshot(),
        binder=GeminiBinder(),
    )


def _drain(agen) -> None:
    async def _run() -> None:
        async for _ in agen:
            pass

    asyncio.run(_run())


def _drain_scoped(agen) -> None:
    """Drain under the request-scoped concurrency bound used by dataset resubmits."""

    async def _run() -> None:
        async with scoped_concurrency():
            async for _ in agen:
                pass

    asyncio.run(_run())


def test_parsed_handle_forwards_rate_cache_to_resolve_variable():
    sentinel = RateTableCache()
    captured: list[object] = []

    async def _fake_resolve_variable(variable, **kwargs):
        captured.append(kwargs["options"].rate_cache)
        return _stub_region(variable)

    graph = FakeGraph()
    inp = ParsedInput(parts=[ParsedPart(variable_text="gdp")])
    with patch("search.engine.fanout.resolve_variable", _fake_resolve_variable):
        _drain(
            parsed.handle(
                inp,
                graph=graph,
                resolver=make_resolver(graph),
                admission=Admission(graph=graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                materialise_graph=None,
                materialise_sdmx=None,
                start_ms=0,
                params=_params(),
                rate_cache=sentinel,
            )
        )
    assert captured == [sentinel]


def test_parsed_handle_with_no_rate_cache_still_serves():
    captured: list[object] = []

    async def _fake_resolve_variable(variable, **kwargs):
        captured.append(kwargs["options"].rate_cache)
        return _stub_region(variable)

    graph = FakeGraph()
    inp = ParsedInput(parts=[ParsedPart(variable_text="gdp")])
    with patch("search.engine.fanout.resolve_variable", _fake_resolve_variable):
        _drain(
            parsed.handle(
                inp,
                graph=graph,
                resolver=make_resolver(graph),
                admission=Admission(graph=graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                materialise_graph=None,
                materialise_sdmx=None,
                start_ms=0,
                params=_params(),
            )
        )
    assert captured == [None]


def test_raw_text_handle_forwards_rate_cache_to_resolve_variable():
    sentinel = RateTableCache()
    captured: list[object] = []

    async def _fake_extract(query, *, llm):
        return ExtractionResult(
            extraction=Extraction(kind="resolution", variables=["gdp"]), qualifiers={}
        ), None

    async def _fake_resolve_variable(variable, **kwargs):
        captured.append(kwargs["options"].rate_cache)
        return _stub_region(variable)

    graph = FakeGraph()
    inp_query = "gdp"
    with (
        patch("search.engine.entry.raw_text.extract", _fake_extract),
        patch("search.engine.fanout.resolve_variable", _fake_resolve_variable),
    ):
        _drain(
            raw_text.handle(
                RawTextInput(query=inp_query),
                graph=graph,
                resolver=make_resolver(graph),
                admission=Admission(graph=graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                materialise_graph=None,
                materialise_sdmx=None,
                start_ms=0,
                params=_params(),
                rate_cache=sentinel,
            )
        )
    assert captured == [sentinel]


def _gdp_identity():
    return make_identity(GDP_SHAPE, entities=[subject_entity(_INDIA)])


def test_identity_entry_forwards_rate_cache_and_full_read_deadline_to_resolve_spec_resubmit():
    """Identity entries pass the request's cache to `resolve_spec_resubmit`, and the deadline
    a member walk's full read runs to: `resubmit_full_read_timeout_s` from request start,
    where the request's own deadline is `request_timeout_s` from it."""
    sentinel = RateTableCache()
    captured: list[object] = []
    full_read: list[float | None] = []
    identity = _gdp_identity()
    entry = ResubmitIdentityEntry(
        variable_text="GDP",
        identity=identity,
    )

    def _fake_resolve_spec_resubmit(**kwargs):
        captured.append(kwargs["options"].rate_cache)
        full_read.append(kwargs["full_read_deadline"])
        return _stub_region(None)

    graph = FakeGraph()
    deadline = time.monotonic() + 30
    with patch(
        "search.engine.entry.dataset_resubmit.resolve_spec_resubmit", _fake_resolve_spec_resubmit
    ):
        _drain_scoped(
            dataset_resubmit.handle(
                DatasetResubmitInput(entries=[entry]),
                graph=graph,
                resolver=make_resolver(graph),
                admission=Admission(graph=graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                materialise_graph=None,
                materialise_sdmx=None,
                start_ms=0,
                params=EntryParams(
                    deadline=deadline,
                    max_candidates=5,
                    pac=False,
                    stream_option_coverage=False,
                    include_option_coverage=False,
                    stream_observations=False,
                    include_observations=False,
                    include_skills=False,
                    skills_index=None,
                    coverage=empty_coverage_snapshot(),
                    binder=GeminiBinder(),
                ),
                rate_cache=sentinel,
            )
        )
    assert captured == [sentinel]
    extension = settings().resubmit_full_read_timeout_s - settings().request_timeout_s
    assert full_read == pytest.approx([deadline + extension], rel=0, abs=1e-9)


def test_dataset_resubmit_handle_forwards_rate_cache_to_every_entry_kind():
    """Identity, candidates, and parsed entries forward the cache through their resolution paths."""
    sentinel = RateTableCache()
    promote_calls: list[object] = []
    variable_calls: list[object] = []

    def _fake_resolve_spec_resubmit(**kwargs):
        promote_calls.append(kwargs["options"].rate_cache)
        return _stub_region(None)

    async def _fake_resolve_variable(variable, **kwargs):
        variable_calls.append(kwargs["options"].rate_cache)
        return _stub_region(variable)

    identity = _gdp_identity()
    identity_entry = ResubmitIdentityEntry(
        variable_text="GDP",
        identity=identity,
    )
    candidates_entry = ResubmitCandidatesEntry(
        variable_text="GDP candidates",
        ordering="broadest_first",
        truncated=False,
        candidates=[
            ResubmitCandidate(
                identity=make_identity(GDP_SHAPE, entities=[subject_entity(_INDIA)]),
            ),
            ResubmitCandidate(
                identity=make_identity(GDP_SHAPE, entities=[subject_entity("country/KEN")]),
            ),
        ],
    )
    parsed_entry = ResubmitParsedEntry(part=ParsedPart(variable_text="population"))
    inp = DatasetResubmitInput(entries=[identity_entry, candidates_entry, parsed_entry])

    graph = FakeGraph()
    with (
        patch(
            "search.engine.entry.dataset_resubmit.resolve_spec_resubmit",
            _fake_resolve_spec_resubmit,
        ),
        patch(
            "search.engine.entry.dataset_entries.resolve_spec_resubmit", _fake_resolve_spec_resubmit
        ),
        patch("search.engine.fanout.resolve_variable", _fake_resolve_variable),
    ):
        _drain_scoped(
            dataset_resubmit.handle(
                inp,
                graph=graph,
                resolver=make_resolver(graph),
                admission=Admission(graph=graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                materialise_graph=None,
                materialise_sdmx=None,
                start_ms=0,
                params=_params(),
                rate_cache=sentinel,
            )
        )
    assert promote_calls == [sentinel, sentinel, sentinel]  # identity entry + 2 candidate promotes
    assert variable_calls == [sentinel]  # the one parsed entry's leg
