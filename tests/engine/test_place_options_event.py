"""Test place-options events, pool admission, cancellation, and shutdown.

The offline fixture has Kenya and Uganda as Country peers, Coast as a Kenya
child, and Africa as an Earth child. Earth supports children but not peers.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
import time
from collections.abc import Coroutine

import pytest
from pydantic import TypeAdapter

from dfo_platform.build import build_commit
from graph_generation import FakeGeneration
from materialise.graph import Facet as MaterialiseFacet
from materialise.graph import SubjectPage
from origin_catalogue import current_catalogue
from search.engine.envelope import now_ms
from search.engine.errors import EngineInfraError
from search.engine.events import Event, PlaceOptionsEvent, serialize_sse
from search.engine.options.options_response_cache import (
    ComputedOptions,
    OptionsResponseCache,
    options_response_cache_key,
)
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.options.resolve import resolve_options_target
from search.engine.region import place_options
from search.engine.region.place_options import (
    PlaceOptionsPool,
    build_place_options_events,
)
from search.models import (
    BaseIdentity,
    GraphRef,
    IdentityWindowAll,
    OptionsRequest,
    SlotOption,
    SlotValue,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph
from tests.materialise.conftest import FakeGraphReader

# ---------------------------------------------------------------------------
# Population: single-member, subject-only (constraintProperties=()), so it
# needs no identity slot at all -- options_population's graph branch
# (statvar_members.enumerate_members) sources it deterministically.
# ---------------------------------------------------------------------------

_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "Person",
    "measured_property": "lifeExpectancy",
    "stat_type": "measuredValue",
}
_SV = "LifeExpectancy_Person"
_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    _SV: {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("lifeExpectancy",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": (),
        "observationProperties": (),
    },
}
_SUBJECTS = {("lifeExpectancy", "measuredProperty"): SubjectPage(dcids=(_SV,), truncated=False)}

# ---------------------------------------------------------------------------
# Places: Kenya/Uganda (Country, peers of each other under Earth), Kenya's
# own AdministrativeArea1 child (Coast), and Earth (Planet) with one
# Continent child (Africa) -- Earth's own type supports children but not
# peers (options.surface._PEER_TYPES/_CHILD_TYPES), the scenario the
# "one scope refused" test below relies on.
# ---------------------------------------------------------------------------

_KENYA = "country/KEN"
_UGANDA = "country/UGA"
_KENYA_COAST = "adminarea/KenyaCoast"
_EARTH = "Earth"
_AFRICA = "africa"

_PLACE_NODES = {
    _KENYA: {"label": "Kenya", "type": "Country"},
    _UGANDA: {"label": "Uganda", "type": "Country"},
    _KENYA_COAST: {"label": "Coast", "type": "AdministrativeArea1"},
    _EARTH: {"label": "Earth", "type": "Planet"},
    _AFRICA: {"label": "Africa", "type": "Continent"},
}
_PLACE_CHILDREN = {
    _EARTH: {"Country": [_KENYA, _UGANDA], "Continent": [_AFRICA]},
    _KENYA: {"AdministrativeArea1": [_KENYA_COAST]},
}
_DATA_PLACES = (_KENYA, _UGANDA, _KENYA_COAST, _AFRICA)
_PLACE_OBS = {
    f"{_SV}|{dcid}": [{"earliestDate": "1960", "latestDate": "2024", "obsCount": 10}]
    for dcid in _DATA_PLACES
}
_PLACE_FACETS = {
    (_SV, dcid): [
        MaterialiseFacet(
            facet_id="f1",
            provenance_id="dc/base/TestProvenance",
            measurement_method=None,
            unit=None,
            obs_count=10,
            earliest_date="1960",
            latest_date="2024",
        )
    ]
    for dcid in _DATA_PLACES
}


def _identity(entities: dict[str, list[str]]) -> BaseIdentity:
    return BaseIdentity(
        shape=_SHAPE, slots=[], entities=entities, requested_window=IdentityWindowAll()
    )


def _graph() -> FakeGraph:
    return FakeGraph(
        nodes=_PLACE_NODES,
        obs=_PLACE_OBS,
        detect={},
        resolve={},
        children=_PLACE_CHILDREN,
        enum_members={},
    )


def _materialise_graph() -> FakeGraphReader:
    return FakeGraphReader(arcs=_ARCS, subjects=_SUBJECTS, facets=_PLACE_FACETS)


def _build(identity: BaseIdentity, *, spec_id: str = "spec-1") -> list[PlaceOptionsEvent]:
    pool = PlaceOptionsPool.build(max_workers=2)
    try:
        return asyncio.run(
            build_place_options_events(
                spec_id,
                identity,
                graph=_graph(),
                sdmx=FakeSdmxClient(observations={}),
                materialise_graph=_materialise_graph(),
                pool=pool,
                coverage=empty_coverage_snapshot(),
                start_ms=now_ms(),
            )
        )
    finally:
        pool.shutdown()


class TestBuildPlaceOptionsEvents:
    def test_both_scopes_succeed_for_a_country_anchor(self):
        events = _build(_identity({"observationAbout": [_KENYA]}), spec_id="spec-kenya")

        # Order, not just membership: the two scope stages are submitted
        # together and awaited concurrently (asyncio.gather), which
        # preserves the order they were passed in (_SCOPES = ("peers",
        # "children")) regardless of which one's own executor job finishes
        # first.
        assert [ev.scope for ev in events] == ["peers", "children"]

        by_scope = {ev.scope: ev for ev in events}
        assert set(by_scope) == {"peers", "children"}
        for ev in events:
            assert ev.kind == "place_options"
            assert ev.spec_id == "spec-kenya"
            assert ev.role == "observationAbout"
            assert ev.anchor_ref == _KENYA
            assert ev.truncated is False

        # Peers: every Country under Earth with data -- the anchor's own
        # type. surface.py does not exclude the anchor itself from its own
        # peer list.
        peer_dcids = {opt.value.ref.ref for opt in by_scope["peers"].options}
        assert peer_dcids == {_KENYA, _UGANDA}

        # Children: Kenya's own AdministrativeArea1 descendants with data.
        child_dcids = {opt.value.ref.ref for opt in by_scope["children"].options}
        assert child_dcids == {_KENYA_COAST}

    def test_a_scope_the_anchor_type_does_not_support_is_dropped(self):
        # Earth (Planet) has a children mapping (-> Continent) but is not a
        # peers-scope-supported type (Country/Continent only) -- peers comes
        # back unsupported_peers_anchor_type and is dropped; children still
        # succeeds.
        events = _build(_identity({"observationAbout": [_EARTH]}), spec_id="spec-earth")

        assert len(events) == 1
        (event,) = events
        assert event.scope == "children"
        assert event.anchor_ref == _EARTH
        assert {opt.value.ref.ref for opt in event.options} == {_AFRICA}

    def test_no_observation_about_binding_returns_no_events(self):
        assert _build(_identity({})) == []


class TestPlaceOptionsPoolMechanics:
    """Test pool admission and cancellation directly, apart from launch-site wiring."""

    def test_a_saturated_pool_skips_the_whole_prefetch(self, monkeypatch):
        """A saturated pool skips resolution without touching the default executor."""
        pool = PlaceOptionsPool.build(max_workers=1)
        assert pool.admission.acquire(blocking=False)  # occupy the only slot

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("resolve_options_target must never run when the pool is saturated")

        monkeypatch.setattr(
            "search.engine.region.place_options.resolve_options_target", _fail_if_called
        )

        default_executor_touched = threading.Event()
        real_to_thread = asyncio.to_thread

        async def _guarded_to_thread(*args, **kwargs):
            default_executor_touched.set()
            return await real_to_thread(*args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", _guarded_to_thread)

        try:
            events = asyncio.run(
                build_place_options_events(
                    "spec-1",
                    _identity({"observationAbout": [_KENYA]}),
                    graph=_graph(),
                    sdmx=FakeSdmxClient(observations={}),
                    materialise_graph=_materialise_graph(),
                    pool=pool,
                    coverage=empty_coverage_snapshot(),
                    start_ms=now_ms(),
                )
            )
        finally:
            pool.shutdown()

        assert events == []
        assert not default_executor_touched.is_set()

    def test_cancelling_between_stages_prevents_the_scope_stage(self, monkeypatch):
        """Cancellation stops the scope stage, while the running executor job finishes."""
        resolve_started = threading.Event()
        may_return = threading.Event()
        resolve_finished = threading.Event()
        scope_called = threading.Event()

        def _fake_resolve_options_target(*args, **kwargs):
            resolve_started.set()
            assert may_return.wait(timeout=5), "test setup: cancel() never arrived"
            resolve_finished.set()
            return object()

        def _fake_compute_options(*args, **kwargs):
            scope_called.set()
            raise AssertionError("the scope stage must never run once cancelled")

        monkeypatch.setattr(
            "search.engine.region.place_options.resolve_options_target",
            _fake_resolve_options_target,
        )
        monkeypatch.setattr(
            "search.engine.region.place_options.compute_options",
            _fake_compute_options,
        )

        async def _body():
            pool = PlaceOptionsPool.build(max_workers=2)
            try:
                task = asyncio.create_task(
                    build_place_options_events(
                        "spec-1",
                        _identity({"observationAbout": [_KENYA]}),
                        graph=_graph(),
                        sdmx=FakeSdmxClient(observations={}),
                        materialise_graph=_materialise_graph(),
                        pool=pool,
                        coverage=empty_coverage_snapshot(),
                        start_ms=now_ms(),
                    )
                )
                for _ in range(500):
                    if resolve_started.is_set():
                        break
                    await asyncio.sleep(0.01)
                else:
                    raise AssertionError("the resolve stage never started")

                task.cancel()
                may_return.set()

                try:
                    await task
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError("expected the task to end up cancelled")

                # Give a wrongly-submitted scope stage a chance to run.
                await asyncio.sleep(0.1)
                assert resolve_finished.is_set()
                assert not scope_called.is_set()
            finally:
                pool.shutdown()

        asyncio.run(_body())

    def test_a_pool_shut_down_before_the_build_yields_no_events_without_raising(self):
        """A shutdown race yields no events and releases the acquired permit."""
        pool = PlaceOptionsPool.build(max_workers=2)
        pool.shutdown()

        events = asyncio.run(
            build_place_options_events(
                "spec-1",
                _identity({"observationAbout": [_KENYA]}),
                graph=_graph(),
                sdmx=FakeSdmxClient(observations={}),
                materialise_graph=_materialise_graph(),
                pool=pool,
                coverage=empty_coverage_snapshot(),
                start_ms=now_ms(),
            )
        )

        assert events == []
        acquired = 0
        while pool.admission.acquire(blocking=False):
            acquired += 1
        assert acquired == 2, "every permit must be back, none leaked by the submit race"

    def test_a_prefetch_for_an_inflight_population_key_drops_without_a_worker(self, monkeypatch):
        """An in-flight population key drops the prefetch without occupying a worker."""
        memo = OptionsPopulationMemo(enabled=True, max_entries=10)
        identity = _identity({"observationAbout": [_KENYA]})

        owner_started = threading.Event()
        owner_release = threading.Event()

        class _BlockingReader(FakeGraphReader):
            def node_subjects(self, *args, **kwargs):
                owner_started.set()
                assert owner_release.wait(timeout=5), "test deadlocked waiting for release"
                return super().node_subjects(*args, **kwargs)

        owner_reader = _BlockingReader(arcs=_ARCS, subjects=_SUBJECTS, facets=_PLACE_FACETS)
        owner_thread = threading.Thread(
            target=contextvars.copy_context().run,
            args=(lambda: memo.get_or_derive(identity, owner_reader),),
        )
        owner_thread.start()
        assert owner_started.wait(timeout=5), "owner never reached the fake reader"

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("compute_options must never run: the resolve stage dropped")

        monkeypatch.setattr("search.engine.region.place_options.compute_options", _fail_if_called)

        pool = PlaceOptionsPool.build(max_workers=2)
        try:
            events = asyncio.run(
                build_place_options_events(
                    "spec-1",
                    identity,
                    graph=_graph(),
                    sdmx=FakeSdmxClient(observations={}),
                    materialise_graph=_materialise_graph(),
                    pool=pool,
                    coverage=empty_coverage_snapshot(),
                    population_memo=memo,
                    start_ms=now_ms(),
                )
            )
            acquired = 0
            while pool.admission.acquire(blocking=False):
                acquired += 1
            assert acquired == 2, "the resolve stage's permit must be released, not leaked"
        finally:
            owner_release.set()
            owner_thread.join(timeout=5)
            pool.shutdown()

        assert events == []

    @pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
    def test_one_free_permit_at_the_scope_stage_still_runs_both_scopes_in_order(self, cached):
        """With one worker, peers and children run sequentially and retain their order."""
        pool = PlaceOptionsPool.build(
            max_workers=1,
            options_cache=OptionsResponseCache(enabled=True, max_entries=10) if cached else None,
        )
        try:
            events = asyncio.run(
                build_place_options_events(
                    "spec-1",
                    _identity({"observationAbout": [_KENYA]}),
                    graph=_graph(),
                    sdmx=FakeSdmxClient(observations={}),
                    materialise_graph=_materialise_graph(),
                    pool=pool,
                    coverage=empty_coverage_snapshot(),
                    start_ms=now_ms(),
                )
            )
        finally:
            pool.shutdown()

        assert [ev.scope for ev in events] == ["peers", "children"]
        by_scope = {ev.scope: ev for ev in events}
        peer_dcids = {opt.value.ref.ref for opt in by_scope["peers"].options}
        assert peer_dcids == {_KENYA, _UGANDA}
        child_dcids = {opt.value.ref.ref for opt in by_scope["children"].options}
        assert child_dcids == {_KENYA_COAST}


def _kenya_place_options(
    pool: PlaceOptionsPool, *, start_ms: int | None = None
) -> Coroutine[object, object, list[PlaceOptionsEvent]]:
    return build_place_options_events(
        "spec-1",
        _identity({"observationAbout": [_KENYA]}),
        graph=_graph(),
        sdmx=FakeSdmxClient(observations={}),
        materialise_graph=_materialise_graph(),
        pool=pool,
        coverage=_COVERAGE,
        start_ms=now_ms() if start_ms is None else start_ms,
    )


def _run(pool: PlaceOptionsPool, *, start_ms: int | None = None) -> list[PlaceOptionsEvent]:
    return asyncio.run(_kenya_place_options(pool, start_ms=start_ms))


def _kenya_peers_key() -> str:
    """The options route's cache key for Kenya's peers, which the place-options fill uses."""
    request = OptionsRequest.model_validate(
        {
            "identity": _identity({"observationAbout": [_KENYA]}),
            "for": {"role": "observationAbout", "scope": "peers"},
            "include_coverage": True,
        }
    )
    target = resolve_options_target(request, graph=_graph(), materialise_graph=_materialise_graph())
    return options_response_cache_key(
        request,
        target,
        build_commit=build_commit(),
        coverage=_COVERAGE,
        catalogue=current_catalogue(),
    )


_COVERAGE = empty_coverage_snapshot()


class _CountingBuild:
    """`compute_options`, counting its calls."""

    def __init__(self) -> None:
        self.calls = 0
        self._real = place_options.compute_options

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._real(*args, **kwargs)


class TestPlaceOptionsThroughTheOptionsCache:
    """A pool carrying the options route's cache reads each scope through it."""

    def test_a_miss_fills_the_route_entry_and_a_hit_computes_nothing(self, monkeypatch):
        build = _CountingBuild()
        monkeypatch.setattr(place_options, "compute_options", build)
        cache = OptionsResponseCache(enabled=True, max_entries=10)
        pool = PlaceOptionsPool.build(max_workers=2, options_cache=cache)
        try:
            first = _run(pool)
            assert build.calls == 2
            second = _run(pool)
        finally:
            pool.shutdown()
        assert build.calls == 2
        assert second == first
        assert [ev.scope for ev in first] == ["peers", "children"]

        # The options route keys the same request the same way and finds the peers entry.
        def _fail(_deadline: float):
            raise AssertionError("the route must read the stored entry")

        stored = asyncio.run(
            cache.get_or_compute(_kenya_peers_key(), _fail, deadline=time.monotonic() + 5)
        )
        assert stored.options.items == first[0].options

    def test_a_scope_job_outliving_its_stream_still_fills_the_cache(self, monkeypatch):
        """The stream's budget cancels the coroutine while its scope jobs run. Each job
        offers its own result, so the cache holds both scopes once the jobs finish.

        The cancel waits until both jobs run: a job still queued on the executor when its
        caller is cancelled is cancelled with it and computes nothing."""
        both_started = threading.Barrier(3)
        release = threading.Event()
        real = place_options.compute_options

        def held(*args, **kwargs):
            both_started.wait(timeout=5)
            assert release.wait(timeout=5), "test setup: the jobs were never released"
            return real(*args, **kwargs)

        monkeypatch.setattr(place_options, "compute_options", held)
        cache = OptionsResponseCache(enabled=True, max_entries=10)
        pool = PlaceOptionsPool.build(max_workers=2, options_cache=cache)

        async def scenario() -> None:
            task = asyncio.create_task(_kenya_place_options(pool))
            await asyncio.to_thread(both_started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()

        try:
            asyncio.run(scenario())
            waited_until = time.monotonic() + 5
            while cache.stats()["entries"] < 2 and time.monotonic() < waited_until:
                time.sleep(0.01)
        finally:
            pool.shutdown()
        assert cache.stats()["entries"] == 2

    def test_a_generation_change_during_the_scope_stage_stores_nothing(self, monkeypatch):
        """Option sets read before the generation moved are dropped, not filed under the
        new stamp."""
        generation = FakeGeneration(token="gen-a")
        real = place_options.compute_options

        def build_then_advance(*args, **kwargs):
            computed = real(*args, **kwargs)
            generation.advance("gen-b")
            return computed

        monkeypatch.setattr(place_options, "compute_options", build_then_advance)
        cache = OptionsResponseCache(enabled=True, max_entries=10, generation=generation)
        pool = PlaceOptionsPool.build(max_workers=2, options_cache=cache)
        try:
            events = _run(pool)
        finally:
            pool.shutdown()
        assert [ev.scope for ev in events] == ["peers", "children"]
        assert cache.stats()["entries"] == 0

    def test_a_failing_options_request_in_flight_neither_delays_nor_fails_the_stream(self):
        """The stream does not join an options POST computing the same key. That failure stays
        the POST's own, and the stream returns its own result while the POST still runs."""
        cache = OptionsResponseCache(enabled=True, max_entries=10)
        key = _kenya_peers_key()
        release = threading.Event()

        def failing(_deadline: float) -> ComputedOptions:
            assert release.wait(timeout=5), "test setup: the flight was never released"
            raise EngineInfraError("the options request's read failed")

        async def scenario() -> tuple[list[PlaceOptionsEvent], bool]:
            flight = asyncio.create_task(
                cache.get_or_compute(key, failing, deadline=time.monotonic() + 5)
            )
            await asyncio.sleep(0)  # the options request starts its flight
            pool = PlaceOptionsPool.build(max_workers=2, options_cache=cache)
            try:
                events = await _kenya_place_options(pool)
            finally:
                pool.shutdown()
            flight_still_running = not flight.done()
            release.set()
            with pytest.raises(EngineInfraError):
                await flight
            return events, flight_still_running

        events, flight_still_running = asyncio.run(scenario())
        assert flight_still_running, "the stream returned without waiting on the flight"
        assert [ev.scope for ev in events] == ["peers", "children"]

    @pytest.mark.parametrize("warm", [False, True], ids=["miss", "hit"])
    def test_a_busy_scope_stage_skips_misses_at_once_and_still_serves_hits(self, monkeypatch, warm):
        """Resolution takes the one permit, and another request takes it back before the scope
        stage. Misses are skipped without waiting and store nothing. Stored scopes need no
        permit."""
        build = _CountingBuild()
        monkeypatch.setattr(place_options, "compute_options", build)
        cache = OptionsResponseCache(enabled=True, max_entries=10)
        warmed = []
        if warm:
            pool = PlaceOptionsPool.build(max_workers=2, options_cache=cache)
            try:
                warmed = _run(pool)
            finally:
                pool.shutdown()
        calls = build.calls

        admission = _TakenBackAfterResolve()
        executor = PlaceOptionsPool.build(max_workers=1).executor
        pool = PlaceOptionsPool(executor=executor, admission=admission, options_cache=cache)
        began = time.monotonic()
        try:
            events = _run(pool)
        finally:
            pool.shutdown()
        assert admission.retaken
        assert time.monotonic() - began < 5, "a busy pool skips; it never waits for a permit"
        assert build.calls == calls
        assert events == warmed
        assert cache.stats()["entries"] == (2 if warm else 0)


class _TakenBackAfterResolve(threading.BoundedSemaphore):
    """One permit, retaken by another request as soon as the resolution stage frees it."""

    def __init__(self) -> None:
        super().__init__(1)
        self.retaken = False

    def release(self, n: int = 1) -> None:
        super().release(n)
        if not self.retaken:
            self.retaken = super().acquire(blocking=False)


class TestPlaceOptionsPool:
    def test_shutdown_stops_admitting_new_submissions(self):
        """Shutdown makes the executor reject later submissions."""
        pool = PlaceOptionsPool.build(max_workers=2)
        pool.shutdown()
        with pytest.raises(RuntimeError):
            pool.executor.submit(lambda: None)


def test_place_options_event_serialize_sse_roundtrip():
    ev = PlaceOptionsEvent(
        spec_id="abc123",
        role="observationAbout",
        scope="peers",
        anchor_ref=_KENYA,
        options=[
            SlotOption(
                value=SlotValue(ref=GraphRef(ref=_UGANDA, label="Uganda"), value_kind="entity")
            )
        ],
        truncated=False,
    )

    frame = serialize_sse(ev)
    lines = frame.strip("\n").split("\n")
    assert lines[0] == "event: place_options"
    assert lines[1].startswith("data: ")
    payload = json.loads(lines[1][len("data: ") :])
    assert payload == {
        "kind": "place_options",
        "spec_id": "abc123",
        "role": "observationAbout",
        "scope": "peers",
        "anchor_ref": _KENYA,
        "options": [
            {
                "value": {
                    "ref": {"ref": _UGANDA, "label": "Uganda"},
                    "value_kind": "entity",
                },
                # SlotOption's wrap serializer promises coverage=null (not
                # omitted) on the SSE surface -- see SlotOption._serialize.
                "coverage": None,
            }
        ],
        "truncated": False,
    }

    # Round-trips through the discriminated Event union, not just the concrete model.
    adapter = TypeAdapter(Event)
    parsed = adapter.validate_json(ev.model_dump_json())
    assert isinstance(parsed, PlaceOptionsEvent)
    assert parsed == ev
