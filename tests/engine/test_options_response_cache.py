"""Tests the options response cache through the route and its single-flight API.

Covers cache keys, hits, degraded results, generation changes, deadlines, cancellation, and failures."""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

import origin_catalogue.service as catalogue_service_module
from graph_generation import FakeGeneration, GenerationStamp
from materialise.facets import bound_variable_breadth
from materialise.graph import Facet as MaterialiseFacet
from materialise.graph import GraphReadError
from origin_catalogue import note_un_blocked_places, record_un_read_failure
from search.engine import config
from search.engine.degraded_reads import note_degraded_read
from search.engine.options import surface
from search.engine.options.options_response_cache import (
    ComputedOptions,
    OptionsResponseCache,
    options_response_cache_key,
)
from search.engine.options.population import OptionsPopulation
from search.engine.options.resolve import ResolvedOptionsTarget
from search.models import (
    BaseIdentity,
    IdentityWindowAll,
    OptionsFor,
    OptionsRequest,
    OptionsSet,
    Warning,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._options_fixtures import (
    _KENYA_DCID,
    _LIFE_EXPECTANCY_ARCS,
    _LIFE_EXPECTANCY_SHAPE,
    _LIFE_EXPECTANCY_SUBJECTS,
    _LIFE_EXPECTANCY_SV,
)
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver
from tests.materialise.conftest import FakeGraphReader
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, fake_catalogue_service

_GHANA_DCID = "country/GHA"
_KENYA_CHILD_DCID = "administrativearea1/KEN_X"
_GHANA_CHILD_DCID = "administrativearea1/GHA_Y"

_NODES = {
    _KENYA_DCID: {"label": "Kenya", "type": "Country"},
    _GHANA_DCID: {"label": "Ghana", "type": "Country"},
    _KENYA_CHILD_DCID: {"label": "Kenya County", "type": "AdministrativeArea1"},
    _GHANA_CHILD_DCID: {"label": "Ghana County", "type": "AdministrativeArea1"},
}
_CHILDREN = {
    "Earth": {"Country": [_KENYA_DCID, _GHANA_DCID]},
    _KENYA_DCID: {"AdministrativeArea1": [_KENYA_CHILD_DCID]},
    _GHANA_DCID: {"AdministrativeArea1": [_GHANA_CHILD_DCID]},
}
_OBS = {
    f"{_LIFE_EXPECTANCY_SV}|{_KENYA_DCID}": [
        {"earliestDate": "1960", "latestDate": "2024", "obsCount": 65}
    ],
    f"{_LIFE_EXPECTANCY_SV}|{_GHANA_DCID}": [
        {"earliestDate": "1960", "latestDate": "2024", "obsCount": 60}
    ],
    f"{_LIFE_EXPECTANCY_SV}|{_KENYA_CHILD_DCID}": [
        {"earliestDate": "1960", "latestDate": "2024", "obsCount": 10}
    ],
    f"{_LIFE_EXPECTANCY_SV}|{_GHANA_CHILD_DCID}": [
        {"earliestDate": "1960", "latestDate": "2024", "obsCount": 12}
    ],
}


class _CountingGraph(FakeGraph):
    """Counts graph reads used during `/options` resolution and response building."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.node_types_batch_calls = 0
        self.child_places_batch_calls = 0
        self.entities_with_data_calls = 0

    def node_types_batch(self, dcids):
        self.node_types_batch_calls += 1
        return super().node_types_batch(dcids)

    def child_places_batch(self, parent_dcids, child_type):
        self.child_places_batch_calls += 1
        return super().child_places_batch(parent_dcids, child_type)

    def entities_with_data(self, variable_dcids, entity_dcids, *, max_bytes=None):
        self.entities_with_data_calls += 1
        return super().entities_with_data(variable_dcids, entity_dcids, max_bytes=max_bytes)


def _counting_graph() -> _CountingGraph:
    return _CountingGraph(nodes=_NODES, obs=_OBS, detect={}, resolve={}, children=_CHILDREN)


def _materialise_graph() -> FakeGraphReader:
    return FakeGraphReader(arcs=_LIFE_EXPECTANCY_ARCS, subjects=_LIFE_EXPECTANCY_SUBJECTS)


class _FailingFacetsGraphReader(FakeGraphReader):
    """Raises `GraphReadError` on facet reads to test that warning-bearing responses are not cached."""

    def observation_facets_batch_many(self, *, stat_vars, entities, deadline=None):
        self._count("observation_facets_batch_many")
        raise GraphReadError("synthetic coverage-probe transport failure")


def _make_client(*, graph: FakeGraph, materialise_graph, cache: OptionsResponseCache) -> TestClient:
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}),
        materialise_graph=materialise_graph,
        options_response_cache=cache,
    )
    return TestClient(app, raise_server_exceptions=False)


def _identity(*, entities: dict[str, list[str]] | None = None) -> BaseIdentity:
    return BaseIdentity(
        shape=_LIFE_EXPECTANCY_SHAPE,
        slots=[],
        entities=entities or {},
        requested_window=IdentityWindowAll(),
    )


def _peers_request(dcid: str, *, include_coverage: bool = False) -> OptionsRequest:
    return OptionsRequest(
        identity=_identity(entities={"observationAbout": [dcid]}),
        **{"for": OptionsFor(role="observationAbout", scope="peers")},
        include_coverage=include_coverage,
    )


def _children_request(dcid: str) -> OptionsRequest:
    return OptionsRequest(
        identity=_identity(entities={"observationAbout": [dcid]}),
        **{"for": OptionsFor(role="observationAbout", scope="children")},
    )


def _body(req: OptionsRequest) -> dict:
    return req.model_dump(by_alias=True, mode="json")


# ---------------------------------------------------------------------------
# The coverage view's stamp participates in the key.
# ---------------------------------------------------------------------------


def _peers_target(*, identity: BaseIdentity | None = None) -> ResolvedOptionsTarget:
    return ResolvedOptionsTarget(
        identity=identity if identity is not None else _identity(),
        population=OptionsPopulation(
            stat_var_dcids=(_LIFE_EXPECTANCY_SV,),
            declared_roles=("observationAbout",),
            source="graph",
        ),
        anchors=(),
        facets=None,
        requested_window=None,
        role_names_by_axis={},
    )


class TestCacheKeyMovesWithTheCoverageStamp:
    def test_a_coverage_republish_changes_the_key(self):
        req = _peers_request(_KENYA_DCID)
        target = _peers_target()

        key_a = options_response_cache_key(
            req,
            target,
            build_commit="commit-x",
            coverage=empty_coverage_snapshot(stamp="stamp-a"),
            catalogue=FakeCatalogue(),
        )
        key_b = options_response_cache_key(
            req,
            target,
            build_commit="commit-x",
            coverage=empty_coverage_snapshot(stamp="stamp-b"),
            catalogue=FakeCatalogue(),
        )

        assert key_a != key_b

    def test_the_same_coverage_stamp_reproduces_the_same_key(self):
        req = _peers_request(_KENYA_DCID)
        target = _peers_target()
        coverage = empty_coverage_snapshot(stamp="stamp-a")

        key_first = options_response_cache_key(
            req, target, build_commit="commit-x", coverage=coverage, catalogue=FakeCatalogue()
        )
        key_second = options_response_cache_key(
            req, target, build_commit="commit-x", coverage=coverage, catalogue=FakeCatalogue()
        )

        assert key_first == key_second


class TestCacheKeyMovesWithTheOriginCatalogue:
    @pytest.mark.parametrize(
        "other",
        [
            FakeCatalogue(build_id="build-b"),
            FakeCatalogue(build_id="build-a", un_system_enabled=False),
        ],
        ids=["build", "un-system-setting"],
    )
    def test_a_catalogue_reload_changes_the_key(self, other: FakeCatalogue):
        """A reload moves the origin catalogue build and leaves the graph generation, so the
        build and the UN System setting it admits by are part of the key."""
        req = _peers_request(_KENYA_DCID)
        target = _peers_target()
        coverage = empty_coverage_snapshot()

        key_a, key_b = (
            options_response_cache_key(
                req, target, build_commit="commit-x", coverage=coverage, catalogue=catalogue
            )
            for catalogue in (FakeCatalogue(build_id="build-a"), other)
        )

        assert key_a != key_b


class TestCacheKeyMovesWithTheOpenedRole:
    def test_each_opened_role_keys_its_own_entry(self):
        """The key reads the identity the population was elected from, so one posted request
        keys apart under each flow axis the door opened."""
        req = _peers_request(_KENYA_DCID)
        coverage = empty_coverage_snapshot()
        keys = {
            options_response_cache_key(
                req,
                _peers_target(
                    identity=BaseIdentity(
                        shape=_LIFE_EXPECTANCY_SHAPE,
                        slots=[],
                        entities={},
                        open_roles=open_roles,
                        requested_window=IdentityWindowAll(),
                    )
                ),
                build_commit="commit-x",
                coverage=coverage,
                catalogue=FakeCatalogue(),
            )
            for open_roles in (None, ["origin"], ["destination"])
        }

        assert len(keys) == 3


# ---------------------------------------------------------------------------
# A repeat is served with no graph calls.
# ---------------------------------------------------------------------------


class TestRepeatRequestHitsTheCache:
    def test_repeat_peers_request_issues_no_further_graph_calls(self):
        graph = _counting_graph()
        client = _make_client(
            graph=graph, materialise_graph=_materialise_graph(), cache=OptionsResponseCache()
        )
        body = _body(_peers_request(_KENYA_DCID))

        resp1 = client.post("/api/search/options", json=body)
        assert resp1.status_code == 200
        after_first = (
            graph.child_places_batch_calls,
            graph.entities_with_data_calls,
        )
        assert after_first[0] >= 1 and after_first[1] >= 1  # the miss really read the graph

        resp2 = client.post("/api/search/options", json=body)
        assert resp2.status_code == 200
        # Target resolution still reads anchor types; a hit skips response-building graph reads.
        assert (
            graph.child_places_batch_calls,
            graph.entities_with_data_calls,
        ) == after_first
        assert resp1.json()["options"] == resp2.json()["options"]


# ---------------------------------------------------------------------------
# Kenya peers and Ghana peers share an entry; Kenya children and Ghana
# children don't.
# ---------------------------------------------------------------------------


class TestPeersVsChildrenSharing:
    def test_kenya_peers_and_ghana_peers_share_one_cache_entry(self):
        graph = _counting_graph()
        client = _make_client(
            graph=graph, materialise_graph=_materialise_graph(), cache=OptionsResponseCache()
        )

        resp_kenya = client.post("/api/search/options", json=_body(_peers_request(_KENYA_DCID)))
        assert resp_kenya.status_code == 200
        after_kenya = (graph.child_places_batch_calls, graph.entities_with_data_calls)

        resp_ghana = client.post("/api/search/options", json=_body(_peers_request(_GHANA_DCID)))
        assert resp_ghana.status_code == 200
        # Both anchors are Countries, so the peer candidate universe shares one cache entry.
        assert (graph.child_places_batch_calls, graph.entities_with_data_calls) == after_kenya
        assert resp_kenya.json()["options"] == resp_ghana.json()["options"]

    def test_kenya_children_and_ghana_children_do_not_share_a_cache_entry(self):
        graph = _counting_graph()
        client = _make_client(
            graph=graph, materialise_graph=_materialise_graph(), cache=OptionsResponseCache()
        )

        resp_kenya = client.post("/api/search/options", json=_body(_children_request(_KENYA_DCID)))
        assert resp_kenya.status_code == 200
        calls_after_kenya = graph.child_places_batch_calls

        resp_ghana = client.post("/api/search/options", json=_body(_children_request(_GHANA_DCID)))
        assert resp_ghana.status_code == 200
        # Child candidates depend on the anchor, so Ghana misses Kenya's entry.
        assert graph.child_places_batch_calls == calls_after_kenya + 1

        kenya_dcids = {i["value"]["ref"]["ref"] for i in resp_kenya.json()["options"]["items"]}
        ghana_dcids = {i["value"]["ref"]["ref"] for i in resp_ghana.json()["options"]["items"]}
        assert kenya_dcids == {_KENYA_CHILD_DCID}
        assert ghana_dcids == {_GHANA_CHILD_DCID}


# ---------------------------------------------------------------------------
# A degraded (coverage-probe-failed) result is never cached.
# ---------------------------------------------------------------------------


class TestDegradedResultsAreNeverCached:
    def test_a_coverage_probe_failure_is_not_cached(self):
        graph = _counting_graph()
        materialise_graph = _FailingFacetsGraphReader(
            arcs=_LIFE_EXPECTANCY_ARCS, subjects=_LIFE_EXPECTANCY_SUBJECTS
        )
        client = _make_client(
            graph=graph, materialise_graph=materialise_graph, cache=OptionsResponseCache()
        )
        body = _body(_peers_request(_KENYA_DCID, include_coverage=True))

        resp1 = client.post("/api/search/options", json=body)
        assert resp1.status_code == 200
        payload1 = resp1.json()
        assert payload1["diagnostics"]["warnings"], "the coverage probe failure must surface"
        probes_after_first = materialise_graph.calls["observation_facets_batch_many"]
        assert probes_after_first >= 1

        resp2 = client.post("/api/search/options", json=body)
        assert resp2.status_code == 200
        assert resp2.json()["diagnostics"]["warnings"]
        # A stored (non-warning) result would have skipped this probe
        # entirely on the second call; a degraded one re-probes every time.
        assert materialise_graph.calls["observation_facets_batch_many"] > probes_after_first

    @pytest.mark.parametrize(
        ("profiled", "computes"),
        [
            pytest.param({"sv": {"fUSD": 197, "fEUR": 25}}, 1, id="breadth-read-is-cached"),
            pytest.param({}, 2, id="fallback-election-is-not-cached"),
        ],
    )
    def test_a_place_tie_left_to_the_unit_is_not_cached(self, profiled, computes):
        """At France the euro and dollar facets tie on every key before whole-variable
        breadth. When the breadth is not read, Search leaves the tie to the unit's name and
        notes a degraded read, so the option's coverage is not stored for the generation."""
        eur = MaterialiseFacet(
            facet_id="fEUR",
            provenance_id="P",
            unit="CurrencyEUR",
            obs_count=1,
            earliest_date="2020",
            latest_date="2020",
        )
        usd = MaterialiseFacet(
            facet_id="fUSD",
            provenance_id="P",
            unit="CurrencyUSD",
            obs_count=1,
            earliest_date="2020",
            latest_date="2020",
        )
        calls: list[float] = []

        class _Labels:
            def node_labels_batch(self, dcids):
                return {dcid: dcid for dcid in dcids}

        def compute(deadline: float) -> ComputedOptions:
            calls.append(deadline)
            items = surface._build_observation_about_options(
                _Labels(),
                ["country/FRA"],
                include_coverage=True,
                facets_batch={"sv": {"country/FRA": [eur, usd]}},
                sv_dcids=["sv"],
                facet_bindings={},
                elected_units=None,
                date_request=None,
                deadline=deadline,
            )
            return ComputedOptions(options=OptionsSet(items=items, truncated=False), warnings=())

        async def scenario():
            cache = OptionsResponseCache(enabled=True, max_entries=10)
            deadline = time.monotonic() + 5
            for _ in range(2):
                with bound_variable_breadth(
                    lambda stat_vars, *, deadline: profiled,
                    refuse_unread=False,
                    on_fallback=lambda: note_degraded_read("whole-variable breadth"),
                ):
                    await cache.get_or_compute("k", compute, deadline=deadline)

        asyncio.run(scenario())
        assert len(calls) == computes


def _refused_by_the_un_front_door() -> None:
    note_un_blocked_places([_GHANA_DCID])


def _failed_un_read() -> None:
    assert record_un_read_failure(
        GraphReadError("synthetic UN read failure"),
        variables=["undata/sv"],
        places=[_GHANA_DCID],
        alongside_ours=True,
    )


class TestUnSystemWarnings:
    @pytest.mark.parametrize(
        ("unserved", "build_refuses", "stored"),
        [
            (_refused_by_the_un_front_door, frozenset({_GHANA_DCID}), True),
            (_refused_by_the_un_front_door, frozenset(), False),
            (_failed_un_read, frozenset({_GHANA_DCID}), False),
        ],
        ids=["refused-by-the-build", "refused-while-serving", "failed-read"],
    )
    def test_only_a_failed_un_read_keeps_a_result_out_of_the_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
        unserved: Callable[[], None],
        build_refuses: frozenset[str],
        stored: bool,
    ):
        """Places the origin catalogue build found refused are a fact of that build, so the
        result naming them is stored and a hit returns their warning. A place refused only
        while serving may be a stray refusal, and a UN read that failed during the compute
        is a failure; either leaves the result uncached."""
        real = surface._observation_about_options_set

        def unserving(*args, **kwargs):
            unserved()
            return real(*args, **kwargs)

        monkeypatch.setattr(surface, "_observation_about_options_set", unserving)
        graph = _counting_graph()
        service = fake_catalogue_service(FakeCatalogue(un_blocked_places=build_refuses))
        with catalogue_service_module._singleton.poisoned_for_test(service):
            client = _make_client(
                graph=graph, materialise_graph=_materialise_graph(), cache=OptionsResponseCache()
            )
            body = _body(_peers_request(_KENYA_DCID))

            resp1 = client.post("/api/search/options", json=body)
            after_first = graph.child_places_batch_calls
            resp2 = client.post("/api/search/options", json=body)

        assert resp1.status_code == resp2.status_code == 200
        codes = [[w["code"] for w in r.json()["diagnostics"]["warnings"]] for r in (resp1, resp2)]
        assert codes == [["un_system_unavailable"]] * 2
        assert (graph.child_places_batch_calls == after_first) is stored


class TestCatalogueReload:
    def test_a_stored_result_is_not_served_after_the_catalogue_build_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A reload changes the origin catalogue build and not the graph generation, so the
        entry stored under the old build must not answer a request bound to the new one."""
        graph = _counting_graph()
        service = fake_catalogue_service()
        served = [FakeCatalogue(build_id="build-a")]
        monkeypatch.setattr(service, "snapshot", lambda: served[0])
        with catalogue_service_module._singleton.poisoned_for_test(service):
            client = _make_client(
                graph=graph, materialise_graph=_materialise_graph(), cache=OptionsResponseCache()
            )
            body = _body(_peers_request(_KENYA_DCID))

            client.post("/api/search/options", json=body)
            after_first = graph.child_places_batch_calls
            client.post("/api/search/options", json=body)
            after_repeat = graph.child_places_batch_calls
            served[0] = FakeCatalogue(build_id="build-b")
            client.post("/api/search/options", json=body)

        assert after_repeat == after_first
        assert graph.child_places_batch_calls > after_repeat


# ---------------------------------------------------------------------------
# A refusal still fires on a warm cache.
# ---------------------------------------------------------------------------


class TestRefusalsStillFireOnAWarmCache:
    def test_an_unresolvable_anchor_still_refuses_after_the_cache_is_warm(self):
        graph = _counting_graph()
        client = _make_client(
            graph=graph, materialise_graph=_materialise_graph(), cache=OptionsResponseCache()
        )

        warm = client.post("/api/search/options", json=_body(_peers_request(_KENYA_DCID)))
        assert warm.status_code == 200

        bad_identity = _identity(entities={"observationAbout": ["country/DOES_NOT_EXIST"]})
        bad_req = OptionsRequest(
            identity=bad_identity, **{"for": OptionsFor(role="observationAbout", scope="peers")}
        )
        resp = client.post("/api/search/options", json=_body(bad_req))
        assert resp.status_code == 422
        assert resp.headers["content-type"].startswith("application/problem+json")
        assert resp.json()["code"] == "invalid_role_ref"


# ---------------------------------------------------------------------------
# get_or_compute itself: single flight through one cache-owned asyncio.Task,
# awaited via asyncio.shield -- never a thread per follower.
# ---------------------------------------------------------------------------

_EMPTY_OPTIONS = OptionsSet(items=[], truncated=False)
_OFFER_WARNING = Warning(code="option_coverage_probe_failed", severity="warn", message="m")


def _blocking_compute(
    *, loop: asyncio.AbstractEventLoop, started: asyncio.Event, release: threading.Event
) -> tuple[Callable[[float], ComputedOptions], list[float]]:
    """Returns a blocking compute double and the deadline each of its calls received."""
    calls: list[float] = []

    def compute(deadline: float) -> ComputedOptions:
        calls.append(deadline)
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5), "test deadlocked waiting for release"
        return ComputedOptions(options=_EMPTY_OPTIONS, warnings=())

    return compute, calls


def _count_to_thread(monkeypatch: pytest.MonkeyPatch, calls: list[object]) -> None:
    """Records `asyncio.to_thread` calls."""
    real = asyncio.to_thread

    async def _to_thread(func, /, *args, **kwargs):
        calls.append(func)
        return await real(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _to_thread)


class TestGetOrComputeSingleFlight:
    def test_compute_reads_by_the_flight_deadline_not_the_callers(self):
        """The caller that starts the compute gives it the flight's deadline,
        `request_timeout_s` from when the flight starts, so the caller's own
        shorter deadline never cuts a result other callers reuse."""

        async def scenario():
            cache = OptionsResponseCache(enabled=True, max_entries=10)
            compute, calls = _immediate_compute()
            before = time.monotonic()
            await cache.get_or_compute("k", compute, deadline=before + 1.0)
            return before, time.monotonic(), calls

        before, after, calls = asyncio.run(scenario())

        flight_s = config.settings().request_timeout_s
        (deadline,) = calls
        assert before + flight_s <= deadline <= after + flight_s

    def test_followers_never_use_a_worker_thread(self, monkeypatch):
        """Only the cache-owned task's own compute runs in a thread; every
        other concurrent caller for the same key is served by awaiting that
        SAME task through asyncio.shield, with no thread of its own."""
        to_thread_calls: list[object] = []
        _count_to_thread(monkeypatch, to_thread_calls)

        async def scenario():
            loop = asyncio.get_running_loop()
            started = asyncio.Event()
            release = threading.Event()
            compute, compute_calls = _blocking_compute(loop=loop, started=started, release=release)
            cache = OptionsResponseCache(enabled=True, max_entries=10)
            deadline = time.monotonic() + 5

            leader = asyncio.create_task(cache.get_or_compute("k", compute, deadline=deadline))
            await started.wait()
            follower = asyncio.create_task(cache.get_or_compute("k", compute, deadline=deadline))

            release.set()
            results = await asyncio.gather(leader, follower)
            return results, compute_calls

        results, compute_calls = asyncio.run(scenario())

        assert len(compute_calls) == 1, "two concurrent callers must share one compute"
        assert len(to_thread_calls) == 1, "only the cache-owned task's compute uses a thread"
        assert results[0] == results[1]

    def test_a_follower_past_its_deadline_raises_promptly_while_the_leader_keeps_running(self):
        """A follower's own deadline bounds only its own wait: it must not
        block until the shared compute finishes, and that compute -- untouched
        by another caller's timeout -- still lands for the leader."""

        async def scenario():
            loop = asyncio.get_running_loop()
            started = asyncio.Event()
            release = threading.Event()
            compute, compute_calls = _blocking_compute(loop=loop, started=started, release=release)
            cache = OptionsResponseCache(enabled=True, max_entries=10)

            leader = asyncio.create_task(
                cache.get_or_compute("k", compute, deadline=time.monotonic() + 5)
            )
            await started.wait()

            before = time.monotonic()
            with pytest.raises(asyncio.TimeoutError):
                await cache.get_or_compute("k", compute, deadline=time.monotonic() + 0.05)
            elapsed = time.monotonic() - before
            assert elapsed < 2, "the follower must not have waited for the leader to finish"

            release.set()
            leader_result = await leader
            return leader_result, compute_calls

        leader_result, compute_calls = asyncio.run(scenario())

        assert leader_result.options == _EMPTY_OPTIONS
        assert len(compute_calls) == 1, "the timed-out follower must not have retried the compute"

    def test_cancelling_a_follower_does_not_cancel_the_shared_compute(self):
        """asyncio.shield keeps a follower's own cancellation off the shared
        task: the leader's compute keeps running, publishes its result, and a
        later caller for the same key still gets it -- with no second
        compute."""

        async def scenario():
            loop = asyncio.get_running_loop()
            started = asyncio.Event()
            release = threading.Event()
            compute, compute_calls = _blocking_compute(loop=loop, started=started, release=release)
            cache = OptionsResponseCache(enabled=True, max_entries=10)
            deadline = time.monotonic() + 5

            leader = asyncio.create_task(cache.get_or_compute("k", compute, deadline=deadline))
            await started.wait()
            follower = asyncio.create_task(cache.get_or_compute("k", compute, deadline=deadline))
            await asyncio.sleep(0.05)  # give the follower a chance to park on the shielded task

            follower.cancel()
            with pytest.raises(asyncio.CancelledError):
                await follower

            release.set()
            leader_result = await leader
            later = await cache.get_or_compute("k", compute, deadline=time.monotonic() + 5)
            return leader_result, later, compute_calls

        leader_result, later, compute_calls = asyncio.run(scenario())

        assert leader_result.options == _EMPTY_OPTIONS
        assert later.options == _EMPTY_OPTIONS
        assert len(compute_calls) == 1, (
            "neither the cancelled follower nor the later hit may re-run compute"
        )


# ---------------------------------------------------------------------------
# A cache-owned task's own failure is always retrieved, even when every
# caller awaiting it has already timed out.
# ---------------------------------------------------------------------------


class TestFailedTaskExceptionIsAlwaysRetrieved:
    def test_a_failure_past_every_callers_own_deadline_is_never_logged_unretrieved(self, caplog):
        """The shared `SingleFlightCache`'s cache-owned task fails only after
        the one caller that started it has already given up at its own
        (short) deadline -- nobody is left awaiting the task by the time it
        raises. asyncio logs "Task exception was never retrieved" from
        `Task.__del__` only when a failed task is garbage-collected with its
        exception unretrieved; this asserts that never happens, forcing
        garbage collection after the task has had time to fail and be
        dropped from the cache's own in-flight bookkeeping.
        """

        def compute(_deadline: float) -> ComputedOptions:
            time.sleep(0.2)
            raise RuntimeError("synthetic failure, past the caller's own deadline")

        async def scenario() -> None:
            cache = OptionsResponseCache(enabled=True, max_entries=10)
            with pytest.raises(asyncio.TimeoutError):
                await cache.get_or_compute("k", compute, deadline=time.monotonic() + 0.05)
            # Let the cache-owned task actually run to its own failure --
            # well past the deadline above -- with nobody left awaiting it.
            await asyncio.sleep(0.3)
            gc.collect()
            # Give the loop a beat to run Task.__del__'s own
            # call_exception_handler, should it (wrongly) still fire.
            await asyncio.sleep(0)

        with caplog.at_level(logging.ERROR, logger="asyncio"):
            asyncio.run(scenario())
        gc.collect()

        assert not any("never retrieved" in record.message for record in caplog.records), (
            "the cache-owned task's own exception must be retrieved before it is collected"
        )


# ---------------------------------------------------------------------------
# Generation keying: every entry's own key and lifetime come from the
# `graph_generation.GenerationProvider` stamp captured for that call.
# ---------------------------------------------------------------------------


def _immediate_compute() -> tuple[Callable[[float], ComputedOptions], list[float]]:
    """A non-blocking `compute` double recording the deadline each call received."""
    calls: list[float] = []

    def compute(deadline: float) -> ComputedOptions:
        calls.append(deadline)
        return ComputedOptions(options=_EMPTY_OPTIONS, warnings=())

    return compute, calls


class TestGenerationKeying:
    def test_a_repeat_call_under_the_same_token_is_a_hit(self):
        async def scenario():
            generation = FakeGeneration(token="gen-a")
            cache = OptionsResponseCache(enabled=True, max_entries=10, generation=generation)
            compute, calls = _immediate_compute()
            deadline = time.monotonic() + 5

            await cache.get_or_compute("k", compute, deadline=deadline)
            await cache.get_or_compute("k", compute, deadline=deadline)
            return calls

        calls = asyncio.run(scenario())
        assert len(calls) == 1, "the same token must serve the second call from the store"

    def test_advancing_the_token_is_a_miss_for_an_identical_key(self):
        async def scenario():
            generation = FakeGeneration(token="gen-a")
            cache = OptionsResponseCache(enabled=True, max_entries=10, generation=generation)
            compute, calls = _immediate_compute()
            deadline = time.monotonic() + 5

            await cache.get_or_compute("k", compute, deadline=deadline)
            generation.advance("gen-b")
            await cache.get_or_compute("k", compute, deadline=deadline)
            return calls

        calls = asyncio.run(scenario())
        assert len(calls) == 2, "a token change must re-derive rather than reuse the old entry"

    def test_unknown_and_known_never_share_an_entry(self):
        async def scenario():
            generation = FakeGeneration()  # starts unknown
            cache = OptionsResponseCache(enabled=True, max_entries=10, generation=generation)
            compute, calls = _immediate_compute()
            deadline = time.monotonic() + 5

            await cache.get_or_compute("k", compute, deadline=deadline)
            generation.advance("gen-a")  # unknown -> known, not itself a token "change"
            await cache.get_or_compute("k", compute, deadline=deadline)
            return calls

        calls = asyncio.run(scenario())
        assert len(calls) == 2, "unknown and a known token must never share the identical key"

    def test_entry_lifetime_equals_the_stamps_own_max_age(self):
        async def scenario():
            clock = [0.0]
            generation = FakeGeneration(token="gen-a")
            cache = OptionsResponseCache(
                enabled=True, max_entries=10, clock=lambda: clock[0], generation=generation
            )
            compute, calls = _immediate_compute()
            deadline = time.monotonic() + 5
            max_age_s = GenerationStamp(token="gen-a").max_age_s

            await cache.get_or_compute("k", compute, deadline=deadline)
            clock[0] = max_age_s - 1
            await cache.get_or_compute("k", compute, deadline=deadline)
            assert len(calls) == 1, "still within the stamp's own lifetime: must stay a hit"

            clock[0] = max_age_s + 1
            await cache.get_or_compute("k", compute, deadline=deadline)
            return calls

        calls = asyncio.run(scenario())
        assert len(calls) == 2, "past the stamp's own lifetime: must re-derive"

    def test_a_post_flip_caller_never_joins_a_pre_flip_in_flight_computation(self):
        """A caller reading `generation` after a token change must never park
        on the still-running task a pre-flip caller started. The two
        computations run independently, keyed by their own stamp."""

        async def scenario():
            loop = asyncio.get_running_loop()
            started = asyncio.Event()
            release = threading.Event()
            leader_compute, leader_calls = _blocking_compute(
                loop=loop, started=started, release=release
            )
            generation = FakeGeneration(token="gen-a")
            cache = OptionsResponseCache(enabled=True, max_entries=10, generation=generation)
            deadline = time.monotonic() + 5

            leader = asyncio.create_task(
                cache.get_or_compute("k", leader_compute, deadline=deadline)
            )
            await started.wait()

            generation.advance("gen-b")
            follower_compute, follower_calls = _immediate_compute()
            follower_result = await cache.get_or_compute("k", follower_compute, deadline=deadline)

            release.set()
            leader_result = await leader
            return leader_result, follower_result, leader_calls, follower_calls

        leader_result, follower_result, leader_calls, follower_calls = asyncio.run(scenario())

        assert len(leader_calls) == 1
        assert len(follower_calls) == 1, "the post-flip caller must run its own compute"
        assert leader_result.options == _EMPTY_OPTIONS
        assert follower_result.options == _EMPTY_OPTIONS

    @pytest.mark.parametrize(
        ("advance", "transient", "occupied", "stored"),
        [
            (False, False, False, True),
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
        ],
        ids=["same-stamp", "stamp-moved", "transient", "live-entry"],
    )
    def test_an_offer_stores_only_under_the_stamp_it_was_computed_against(
        self, advance: bool, transient: bool, occupied: bool, stored: bool
    ):
        """`offer` files a value computed outside the cache only when the stamp read before
        the compute is still current, the value is cacheable and no live entry holds the
        key."""
        generation = FakeGeneration(token="gen-a")
        cache = OptionsResponseCache(enabled=True, max_entries=10, generation=generation)
        held = ComputedOptions(options=_EMPTY_OPTIONS, warnings=())
        if occupied:
            assert cache.offer("k", held, stamp_key=cache.stamp_key())
        stamp_key = cache.stamp_key()
        if advance:
            generation.advance("gen-b")
        offered = ComputedOptions(
            options=_EMPTY_OPTIONS, warnings=(_OFFER_WARNING,), transient=transient
        )

        assert cache.offer("k", offered, stamp_key=stamp_key) is stored
        assert cache.cached("k") is (offered if stored else held if occupied else None)

    def test_stats_counts_hits_misses_evictions_and_entries(self):
        async def scenario():
            cache = OptionsResponseCache(enabled=True, max_entries=1)
            compute, _calls = _immediate_compute()
            deadline = time.monotonic() + 5

            await cache.get_or_compute("k1", compute, deadline=deadline)  # miss
            await cache.get_or_compute("k1", compute, deadline=deadline)  # hit
            await cache.get_or_compute("k2", compute, deadline=deadline)  # miss, evicts k1
            return cache.stats()

        stats = asyncio.run(scenario())
        assert stats == {"hits": 1, "misses": 2, "evictions": 1, "entries": 1}

    def test_clear_drops_every_stored_entry(self):
        async def scenario():
            cache = OptionsResponseCache(enabled=True, max_entries=10)
            compute, calls = _immediate_compute()
            deadline = time.monotonic() + 5

            await cache.get_or_compute("k", compute, deadline=deadline)
            cache.clear()
            await cache.get_or_compute("k", compute, deadline=deadline)
            return calls

        calls = asyncio.run(scenario())
        assert len(calls) == 2, "a cleared cache must re-derive on the next call"


def test_max_entries_below_one_raises():
    with pytest.raises(ValueError, match="max_entries"):
        OptionsResponseCache(enabled=True, max_entries=0)
