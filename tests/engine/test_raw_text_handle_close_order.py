"""For N>=2, closing `raw_text.handle` runs the combiner's cleanup before `aclose()` returns."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

from entity_resolution import Admission
from search.engine.concurrency import scoped_concurrency
from search.engine.entry import raw_text
from search.engine.entry.params import EntryParams
from search.engine.extract import Extraction, ExtractionResult
from search.engine.region.result import RegionResult
from search.models import RawTextInput
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_MARKER = object()


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
    )


def _stub_region(variable_text: str | None) -> RegionResult:
    return RegionResult(
        variable_text=variable_text,
        status="no_data",
        specs=(),
        no_data_reason="variable_not_resolved",
        warnings=(),
        timing_by_step={},
    )


async def _fake_extract(query, *, llm):
    return (
        ExtractionResult(
            extraction=Extraction(kind="resolution", variables=["gdp", "population"]), qualifiers={}
        ),
        None,
    )


async def _fake_resolve_variable(variable, **kwargs):
    return _stub_region(variable)


def _fake_combined_regions(log: list[str]):
    async def _combiner(*args, **kwargs):
        try:
            yield _MARKER
            yield _MARKER
        finally:
            log.append("combiner-finally")

    return _combiner


def _drive(log: list[str]) -> None:
    graph = FakeGraph()

    async def _run() -> None:
        async with scoped_concurrency():
            gen = raw_text.handle(
                RawTextInput(query="gdp and population"),
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
            async for ev in gen:
                if ev is _MARKER:
                    break
            await gen.aclose()
            log.append("aclose returned")
            # Let any deferred async-generator finalizer run before checking order.
            await asyncio.sleep(0)
            await asyncio.sleep(0)

    asyncio.run(_run())


def test_handle_close_finishes_combiner_cleanup_before_aclose_returns():
    log: list[str] = []
    with (
        patch("search.engine.entry.raw_text.extract", _fake_extract),
        patch("search.engine.fanout.resolve_variable", _fake_resolve_variable),
        patch(
            "search.engine.entry.raw_text._stream_combined_regions",
            _fake_combined_regions(log),
        ),
    ):
        _drive(log)

    assert log == ["combiner-finally", "aclose returned"], log
