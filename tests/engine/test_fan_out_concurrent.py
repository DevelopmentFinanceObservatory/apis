"""Checks concurrent fan-out with a rendezvous that sequential dispatch cannot satisfy."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, patch

from entity_resolution import Resolution
from search.engine.core import resolve_async
from search.engine.extract import ExtractedEntity, Extraction, ExtractionResult
from search.engine.region import RegionResult
from tests.engine._harness import EMPTY_SDMX, make_request
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

N = 2  # number of variables in the fake extraction


class _LegRendezvous:
    """Holds each leg until all expected legs arrive. `peak` records overlap.

    `settle` bounds a run whose concurrency is below `expected`.
    """

    def __init__(self, expected: int, *, settle: float = 0.5) -> None:
        self._expected = expected
        self._settle = settle
        self._arrived = 0
        self._active = 0
        self.peak = 0
        self._held = asyncio.Event()

    def arrive(self) -> None:
        self._arrived += 1
        self._active += 1
        self.peak = max(self.peak, self._active)
        if self._arrived >= self._expected:
            self._held.set()

    def depart(self) -> None:
        self._active -= 1

    async def wait_for_peers(self) -> None:
        """Wait at most `settle` for all legs."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._held.wait(), timeout=self._settle)


def _leg_stub(legs: _LegRendezvous):
    """Return a `no_data` region after the rendezvous opens."""

    async def _resolve_variable(variable: str, **_kwargs: object) -> RegionResult:
        legs.arrive()
        try:
            await legs.wait_for_peers()
            return RegionResult(
                variable_text=variable,
                status="no_data",
                specs=(),
                no_data_reason="variable_not_resolved",
                warnings=(),
                timing_by_step={},
            )
        finally:
            legs.depart()

    return _resolve_variable


def _run(request, graph, llm, sdmx=EMPTY_SDMX, resolver=None):
    """Run `resolve_async` with injected fakes.

    Untyped parameters avoid ty errors on the fakes.
    """
    return asyncio.run(resolve_async(request, graph=graph, llm=llm, sdmx=sdmx, resolver=resolver))


def test_fan_out_runs_concurrently():
    """A rendezvous proves that both resolve legs overlap and return one part each."""
    legs = _LegRendezvous(N)
    fake_extraction = AsyncMock(
        return_value=(
            ExtractionResult(
                extraction=Extraction(
                    kind="resolution", variables=["v1", "v2"], entities=[], dates=[]
                ),
                qualifiers={},
            ),
            None,
        )
    )

    with (
        patch("search.engine.entry.raw_text.extract", fake_extraction),
        patch("search.engine.fanout.resolve_variable", new=_leg_stub(legs)),
    ):
        response = _run(
            make_request("test concurrent fan-out"),
            FakeGraph(nodes={}, obs={}, detect={}, resolve={}),
            FakeLLM(responses={}),
        )

    assert [part.variable_text for part in response.parts] == ["v1", "v2"]
    assert legs.peak == N, (
        f"peak co-active legs was {legs.peak}, expected {N}: the fan-out dispatch "
        f"is not concurrent -- a sequential implementation never lets leg 2 arrive "
        f"while leg 1 is still running."
    )


def test_semaphore_caps_max_concurrent_legs(search_settings):
    """Four legs with a cap of two should reach a peak concurrency of two.

    Fake extraction bypasses `max_variables`. The override also satisfies settings validation.
    """
    _N = 4  # number of variables (> cap so semaphore is tested)
    _CAP = 2  # semaphore value to enforce

    legs = _LegRendezvous(_N)

    fake_extraction = AsyncMock(
        return_value=(
            ExtractionResult(
                extraction=Extraction(
                    kind="resolution", variables=[f"v{i}" for i in range(_N)], entities=[], dates=[]
                ),
                qualifiers={},
            ),
            None,
        )
    )

    search_settings(max_variables=_CAP, max_variable_concurrency=_CAP)
    with (
        patch("search.engine.entry.raw_text.extract", fake_extraction),
        patch("search.engine.fanout.resolve_variable", new=_leg_stub(legs)),
    ):
        _run(
            make_request("test semaphore cap"),
            FakeGraph(nodes={}, obs={}, detect={}, resolve={}),
            FakeLLM(responses={}),
        )

    assert legs.peak <= _CAP, (
        f"peak co-active legs was {legs.peak}, exceeds cap of {_CAP}. "
        f"Semaphore is not being applied."
    )
    assert legs.peak == _CAP, (
        f"peak co-active legs was {legs.peak}, below cap of {_CAP}: the semaphore "
        f"admits fewer legs at once than SEARCH_MAX_VARIABLE_CONCURRENCY allows."
    )


def test_entity_pre_resolution_precedes_fan_out_with_deduplicated_entities():
    """Raw-text fan-out resolves deduplicated entities once before starting variable legs."""
    fake_graph = FakeGraph(nodes={}, obs={}, detect={}, resolve={})
    resolver = make_resolver(fake_graph)
    order: list[str] = []
    resolve_calls: list[list[str]] = []

    _original_resolve = resolver.resolve

    async def _tracked_resolve(texts: list[str]) -> list[Resolution]:
        resolve_calls.append(list(texts))
        order.append("resolve")
        return await _original_resolve(texts)

    resolver.resolve = _tracked_resolve  # ty: ignore[invalid-assignment]

    async def _ordering_stub_resolve_variable(variable: str, **_kwargs: object) -> RegionResult:
        order.append(f"leg:{variable}")
        return RegionResult(
            variable_text=variable,
            status="no_data",
            specs=(),
            no_data_reason="variable_not_resolved",
            warnings=(),
            timing_by_step={},
        )

    # "Kenya" repeats across the two variables -- pre-resolution dedupes it.
    fake_extraction = AsyncMock(
        return_value=(
            ExtractionResult(
                extraction=Extraction(
                    kind="resolution",
                    variables=["v0", "v1"],
                    entities=[ExtractedEntity(text=t) for t in ("Kenya", "Uganda", "Kenya")],
                    dates=[],
                ),
                qualifiers={},
            ),
            None,
        )
    )

    with (
        patch("search.engine.entry.raw_text.extract", fake_extraction),
        patch("search.engine.fanout.resolve_variable", new=_ordering_stub_resolve_variable),
    ):
        _run(
            make_request("test entity pre-resolution ordering"),
            fake_graph,
            FakeLLM(responses={}),
            resolver=resolver,
        )

    assert resolve_calls == [["Kenya", "Uganda"]]
    assert order == ["resolve", "leg:v0", "leg:v1"]
