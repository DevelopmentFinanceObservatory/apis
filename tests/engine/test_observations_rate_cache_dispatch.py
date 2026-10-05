"""Tests RateTableCache forwarding through observation dispatch and candidate resolution."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

import search.engine.region.candidate_collapse as candidate_collapse_module
import search.engine.region.observations.dispatch as region_dispatch
import search.engine.region.pipeline as region_pipeline
from materialise.errors import MaterialisationUsageError
from materialise.rate_tables import RateTableCache
from search.engine.concurrency import scoped_concurrency
from search.engine.region.candidates import resolve_candidates
from search.engine.region.codes import OBSERVATIONS_NOTHING_TO_PROBE
from search.engine.region.observations.dispatch import (
    _launch_observations_task,
    _observations_events,
    _observations_relaunch_closure,
    _resolve_spec_observations,
)
from search.engine.region.state import ResolveState
from search.models import Spec, omitted_observations
from tests.conftest import minimal_spec
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import sdmx_double
from tests.engine.test_observations_candidates_fanout import (
    _LONE_SURVIVOR_MATCHED_UNIT_OBS,
    _UNIT_CAND_QUERY,
    _run_candidate_unit,
)
from tests.fixtures import FakeGraph
from tests.materialise.conftest import FakeGraphReader

# ---------------------------------------------------------------------------
# region/observations/dispatch.py: every function that forwards
# materialise_graph into _build_observations, or into another dispatcher,
# forwards rate_cache the same way.
# ---------------------------------------------------------------------------


def _rate_cache_spy(captured: list[object]):
    def _spy(*args, **kwargs):
        captured.append(kwargs.get("rate_cache"))
        return (
            omitted_observations(OBSERVATIONS_NOTHING_TO_PROBE, "spy: nothing to fetch"),
            None,
        )

    return _spy


def test_observations_events_forwards_rate_cache_to_build_observations():
    captured: list[object] = []
    sentinel = RateTableCache()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(region_dispatch, "_build_observations", _rate_cache_spy(captured))
        _observations_events(
            "spec-1",
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(),
            materialise_graph=FakeGraphReader(),
            materialise_sdmx=sdmx_double(),
            rate_cache=sentinel,
            coverage=empty_coverage_snapshot(),
        )
    assert captured == [sentinel]


def test_observations_events_omits_rate_cache_by_default():
    """Omitting `rate_cache` forwards `None` to `_build_observations`."""
    captured: list[object] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(region_dispatch, "_build_observations", _rate_cache_spy(captured))
        _observations_events(
            "spec-1",
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(),
            materialise_graph=FakeGraphReader(),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )
    assert captured == [None]


def test_launch_observations_task_forwards_rate_cache():
    """_launch_observations_task -> _observations_events -> _build_observations."""
    captured: list[object] = []
    sentinel = RateTableCache()

    async def _run() -> None:
        async with scoped_concurrency():
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(region_dispatch, "_build_observations", _rate_cache_spy(captured))
                task = _launch_observations_task(
                    "spec-1",
                    ["Count_Person"],
                    {"observationAbout": ["country/IND"]},
                    graph=FakeGraph(),
                    materialise_graph=FakeGraphReader(),
                    materialise_sdmx=sdmx_double(),
                    rate_cache=sentinel,
                    coverage=empty_coverage_snapshot(),
                )
                await task

    asyncio.run(_run())
    assert captured == [sentinel]


def test_observations_relaunch_closure_forwards_rate_cache():
    """_observations_relaunch_closure -> _launch_observations_task ->
    _observations_events -> _build_observations."""
    captured: list[object] = []
    sentinel = RateTableCache()

    async def _run() -> None:
        async with scoped_concurrency():
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(region_dispatch, "_build_observations", _rate_cache_spy(captured))
                relaunch = _observations_relaunch_closure(
                    ["Count_Person"],
                    {"observationAbout": ["country/IND"]},
                    graph=FakeGraph(),
                    materialise_graph=FakeGraphReader(),
                    materialise_sdmx=sdmx_double(),
                    rate_cache=sentinel,
                    coverage=empty_coverage_snapshot(),
                )
                await relaunch("spec-1")

    asyncio.run(_run())
    assert captured == [sentinel]


def test_resolve_spec_observations_buffered_forwards_rate_cache():
    """The buffered (include_observations=True, stream_observations=False)
    branch forwards rate_cache straight into its own _build_observations
    call."""
    captured: list[object] = []
    sentinel = RateTableCache()
    spec = Spec.model_validate(minimal_spec())

    async def _run() -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(region_dispatch, "_build_observations", _rate_cache_spy(captured))
            await _resolve_spec_observations(
                spec,
                ["Count_Person"],
                {"observationAbout": ["country/IND"]},
                graph=FakeGraph(),
                materialise_graph=FakeGraphReader(),
                materialise_sdmx=sdmx_double(),
                include_observations=True,
                stream_observations=False,
                warnings=[],
                rate_cache=sentinel,
                coverage=empty_coverage_snapshot(),
            )

    asyncio.run(_run())
    assert captured == [sentinel]


def test_resolve_spec_observations_stream_forwards_rate_cache():
    """The deferred (stream_observations=True) branch forwards rate_cache
    through its own relaunch closure -- _observations_relaunch_closure ->
    _launch_observations_task -> _observations_events -> _build_observations."""
    captured: list[object] = []
    sentinel = RateTableCache()
    spec = Spec.model_validate(minimal_spec())

    async def _run() -> None:
        async with scoped_concurrency():
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(region_dispatch, "_build_observations", _rate_cache_spy(captured))
                _, task, _ = await _resolve_spec_observations(
                    spec,
                    ["Count_Person"],
                    {"observationAbout": ["country/IND"]},
                    graph=FakeGraph(),
                    materialise_graph=FakeGraphReader(),
                    materialise_sdmx=sdmx_double(),
                    include_observations=False,
                    stream_observations=True,
                    warnings=[],
                    rate_cache=sentinel,
                    coverage=empty_coverage_snapshot(),
                )
                assert task is not None
                await task

    asyncio.run(_run())
    assert captured == [sentinel]


# ---------------------------------------------------------------------------
# region/candidate_collapse.py: a streamed lone survivor's immediate preview under a
# bound unit (`_assemble_definite`).
# ---------------------------------------------------------------------------


class _PassEnded(Exception):
    """Ends the pass once the test body has run in place of `resolve_candidates`."""


def _resolve_candidates_in_pass(
    monkeypatch: pytest.MonkeyPatch, body: Callable[[ResolveState], Awaitable[None]]
) -> None:
    """Runs the lone-survivor bound-unit candidate fixture, set to stream its preview, and
    runs `body` on its state in place of `resolve_candidates`, inside the live pass. The
    pass cancels its queued member walks when it ends (`pipeline._cancel_queued_reads`), so
    `body` must await `resolve_candidates` before then."""

    async def _run_body_and_end(state: ResolveState) -> None:
        state.stream_observations = True
        await body(state)
        raise _PassEnded

    patched_phases = tuple(
        _run_body_and_end if phase is resolve_candidates else phase
        for phase in region_pipeline._PHASES
    )
    monkeypatch.setattr(region_pipeline, "_PHASES", patched_phases)
    with pytest.raises(_PassEnded):
        _run_candidate_unit(obs=_LONE_SURVIVOR_MATCHED_UNIT_OBS, unit_text="US dollars")


def test_resolve_candidates_forwards_rate_cache_to_build_observations(monkeypatch):
    captured: list[object] = []
    sentinel = RateTableCache()
    real_build_observations = candidate_collapse_module._build_observations

    def _spy(*args, **kwargs):
        captured.append(kwargs.get("rate_cache"))
        return real_build_observations(*args, **kwargs)

    monkeypatch.setattr(candidate_collapse_module, "_build_observations", _spy)

    async def _body(state: ResolveState) -> None:
        assert state.bound_unit_outcome is not None, (
            f"{_UNIT_CAND_QUERY!r} with unit_text='US dollars' must bind a unit "
            "outcome to reach the streamed preview this test targets"
        )
        async with scoped_concurrency():
            await resolve_candidates(state, rate_cache=sentinel)

    _resolve_candidates_in_pass(monkeypatch, _body)

    assert captured, "the survivor's preview never called _build_observations"
    assert all(c is sentinel for c in captured)


def test_resolve_candidates_omits_rate_cache_by_default(monkeypatch):
    """A direct call without a cache forwards `None` before unit conversion raises.

    The spy records the missing cache before the real builder checks it."""
    captured: list[object] = []
    real_build_observations = candidate_collapse_module._build_observations

    def _spy(*args, **kwargs):
        captured.append(kwargs.get("rate_cache"))
        return real_build_observations(*args, **kwargs)

    monkeypatch.setattr(candidate_collapse_module, "_build_observations", _spy)

    async def _body(state: ResolveState) -> None:
        with pytest.raises(MaterialisationUsageError):
            await resolve_candidates(state)

    _resolve_candidates_in_pass(monkeypatch, _body)

    assert captured, "the survivor's preview never called _build_observations"
    assert all(c is None for c in captured)
