"""Tests RateTableCache forwarding through observation dispatch and candidate resolution."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

import search.engine.region.observations.dispatch as region_dispatch
import search.engine.region.pipeline as region_pipeline
from materialise.errors import MaterialisationUsageError
from materialise.rate_tables import RateTableCache
from search.engine.concurrency import scoped_concurrency
from search.engine.region.candidates import resolve_candidates
from search.engine.region.codes import OBSERVATIONS_NOTHING_TO_PROBE
from search.engine.region.observations.dispatch import (
    PreviewRead,
    PreviewReader,
    _launch_observations_task,
    _observations_events,
)
from search.engine.region.phase_answer import _phase_outcome
from search.engine.region.state import ResolveState
from search.engine.region.un_system import NO_LABELS
from search.models import omitted_observations
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
            variable_labels=NO_LABELS,
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
            variable_labels=NO_LABELS,
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
                    variable_labels=NO_LABELS,
                )
                await task

    asyncio.run(_run())
    assert captured == [sentinel]


def _reader(rate_cache: RateTableCache) -> PreviewReader:
    return PreviewReader(
        read=PreviewRead(
            sv_dcids=("Count_Person",),
            obs_entities={"observationAbout": ["country/IND"]},
            plan=None,
            facet_filter=None,
            bulk_places=None,
            member_facts=None,
            unit_request=None,
            elected=None,
        ),
        graph=FakeGraph(),
        sdmx=None,
        materialise_graph=FakeGraphReader(),
        materialise_sdmx=sdmx_double(),
        rate_cache=rate_cache,
        deadline=None,
        coverage=empty_coverage_snapshot(),
    )


@pytest.mark.parametrize("streamed", [False, True])
def test_preview_reader_forwards_rate_cache(streamed: bool):
    """`PreviewReader.observations`, and `relaunch` through `_launch_observations_task` and
    `_observations_events`, reach `_build_observations` with the reader's cache."""
    captured: list[object] = []
    sentinel = RateTableCache()

    async def _run() -> None:
        async with scoped_concurrency():
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(region_dispatch, "_build_observations", _rate_cache_spy(captured))
                reader = _reader(sentinel)
                if streamed:
                    await reader.relaunch(NO_LABELS)("spec-1")
                else:
                    await asyncio.to_thread(reader.observations)

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
    runs `body` on its state in place of the outcome stage (`phase_answer._phase_outcome`),
    whose first step is `resolve_candidates`, inside the live pass. The
    pass cancels its queued member walks when it ends (`pipeline._cancel_queued_reads`), so
    `body` must await `resolve_candidates` before then."""

    async def _run_body_and_end(state: ResolveState) -> None:
        state.stream_observations = True
        await body(state)
        raise _PassEnded

    patched_phases = tuple(
        _run_body_and_end if phase is _phase_outcome else phase for phase in region_pipeline._PHASES
    )
    monkeypatch.setattr(region_pipeline, "_PHASES", patched_phases)
    with pytest.raises(_PassEnded):
        _run_candidate_unit(obs=_LONE_SURVIVOR_MATCHED_UNIT_OBS, unit_text="US dollars")


def test_resolve_candidates_forwards_rate_cache_to_build_observations(monkeypatch):
    captured: list[object] = []
    sentinel = RateTableCache()
    real_build_observations = region_dispatch._build_observations

    def _spy(*args, **kwargs):
        captured.append(kwargs.get("rate_cache"))
        return real_build_observations(*args, **kwargs)

    monkeypatch.setattr(region_dispatch, "_build_observations", _spy)

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
    real_build_observations = region_dispatch._build_observations

    def _spy(*args, **kwargs):
        captured.append(kwargs.get("rate_cache"))
        return real_build_observations(*args, **kwargs)

    monkeypatch.setattr(region_dispatch, "_build_observations", _spy)

    async def _body(state: ResolveState) -> None:
        with pytest.raises(MaterialisationUsageError):
            await resolve_candidates(state)

    _resolve_candidates_in_pass(monkeypatch, _body)

    assert captured, "the survivor's preview never called _build_observations"
    assert all(c is None for c in captured)
