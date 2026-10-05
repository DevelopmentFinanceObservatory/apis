"""Admission tests use a small `subClassOf` lattice and `rangeIncludes` roles."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

from cachetools import TLRUCache

from entity_resolution import Admission
from entity_resolution.admission import _PARENTS_CACHE_MAXSIZE, _RANGE_CACHE_MAXSIZE, _ttu
from graph_generation import FakeGeneration
from graph_generation.config import (
    GRAPH_CACHE_MAX_AGE_SECONDS,
    GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS,
)
from graph_generation.stamp import UNKNOWN_KEY

from .conftest import FakeGraphLookup

_NODES: dict[str, dict[str, tuple[str, ...]]] = {
    # Class chains converge on Place or Thing.
    "Country": {"subClassOf": ("AdministrativeArea",)},
    "State": {"subClassOf": ("AdministrativeArea",)},
    "AdministrativeArea": {"subClassOf": ("Place",)},
    "Continent": {"subClassOf": ("Landform",)},
    "Landform": {"subClassOf": ("Place",)},
    "UNGeoRegion": {"subClassOf": ("GeoRegion",)},
    "ContinentalUnion": {"subClassOf": ("GeoRegion",)},
    "GeoRegion": {"subClassOf": ("Place",)},
    "Organization": {"subClassOf": ("Thing",)},
    "DACRecipientUnallocatedByCountry": {"subClassOf": ("Thing",)},
    # Role ranges.
    "observationAbout": {"rangeIncludes": ("Place",)},
    "financingProvider": {
        "rangeIncludes": ("Country", "DACRecipientUnallocatedByCountry", "Organization")
    },
    "financingRecipient": {
        "rangeIncludes": ("Country", "DACRecipientUnallocatedByCountry", "Organization")
    },
}


def _graph() -> FakeGraphLookup:
    return FakeGraphLookup(nodes=_NODES)


def _run(coro):
    return asyncio.run(coro)


def test_state_admitted_on_observation_about():
    admission = Admission(_graph())

    assert _run(admission.admits("observationAbout", "State")) is True


def test_organization_refused_on_observation_about():
    admission = Admission(_graph())

    assert _run(admission.admits("observationAbout", "Organization")) is False


def test_organization_admitted_on_financing_provider():
    admission = Admission(_graph())

    assert _run(admission.admits("financingProvider", "Organization")) is True


def test_continent_refused_on_financing_recipient():
    admission = Admission(_graph())

    assert _run(admission.admits("financingRecipient", "Continent")) is False


def test_direct_range_hit_short_circuits_before_walking_closure():
    graph = _graph()
    admission = Admission(graph)

    assert _run(admission.admits("financingProvider", "Country")) is True
    assert graph.node_arcs_many_calls == [(("financingProvider",), ("rangeIncludes",))]


def test_missing_role_property_admits_nothing():
    admission = Admission(_graph())

    assert _run(admission.admits("noSuchProperty", "Country")) is False


def test_missing_class_node_has_no_parents():
    graph = FakeGraphLookup(nodes={"observationAbout": {"rangeIncludes": ("Place",)}})
    admission = Admission(graph)

    assert _run(admission.admits("observationAbout", "GhostType")) is False


def test_empty_graph_type_never_admits():
    admission = Admission(_graph())

    assert _run(admission.admits("observationAbout", "")) is False


def test_each_distinct_node_read_from_graph_once_across_repeated_calls():
    graph = _graph()
    admission = Admission(graph)

    # Both types share AdministrativeArea, whose graph read should be cached.
    assert _run(admission.admits("observationAbout", "State")) is True
    assert _run(admission.admits("observationAbout", "Country")) is True

    requested: list[str] = []
    for dcids, _properties in graph.node_arcs_many_calls:
        requested.extend(dcids)
    assert len(requested) == len(set(requested)), requested


def test_stale_write_race_lands_under_the_stamp_its_own_call_captured():
    """A blocked gen-a read writes under its captured key after another call advances to gen-b.

    Restoring gen-a then serves the same role lookup from cache without another graph read.
    """
    release_first_read = threading.Event()
    first_read_started = threading.Event()

    class BlockingGraph:
        def __init__(self) -> None:
            self.node_arcs_many_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

        def node_arcs_many(self, dcids, *, properties, deadline=None):
            self.node_arcs_many_calls.append((tuple(dcids), tuple(properties)))
            if dcids == ("rp",) and properties == ("rangeIncludes",):
                first_read_started.set()
                assert release_first_read.wait(timeout=5), "test deadlocked"
                return {"rp": {"rangeIncludes": ("TypeA",)}}
            return {dcid: {properties[0]: ()} for dcid in dcids}

    graph = BlockingGraph()
    generation = FakeGeneration(token="gen-a")
    admission = Admission(graph, generation=generation)

    async def task_a():
        return await admission.admits("rp", "TypeA")

    async def task_b():
        await asyncio.get_running_loop().run_in_executor(None, first_read_started.wait, 5)
        generation.advance("gen-b")
        await admission.admits("rp2", "TypeX")
        release_first_read.set()

    async def race():
        t_a = asyncio.create_task(task_a())
        t_b = asyncio.create_task(task_b())
        result_a = await t_a
        await t_b
        return result_a

    result_a = _run(race())

    assert result_a is True

    calls_before = len(graph.node_arcs_many_calls)
    generation.advance("gen-a")
    again = _run(admission.admits("rp", "TypeA"))
    assert again is True
    assert len(graph.node_arcs_many_calls) == calls_before


def test_generation_flip_mid_await_does_not_raise_keyerror_on_a_mixed_frontier():
    """A mixed BranchA/BranchB frontier keeps its cached value while BranchB is read.

    Advancing the generation during that read must not raise KeyError for BranchA.
    """
    release_second_read = threading.Event()
    second_read_started = threading.Event()
    nodes: dict[str, dict[str, tuple[str, ...]]] = {
        "Sub": {"subClassOf": ("BranchA", "BranchB")},
        "BranchA": {"subClassOf": ("Place",)},
        "BranchB": {"subClassOf": ("Place",)},
        "Country": {"subClassOf": ("Elsewhere",)},
        "observationAbout": {"rangeIncludes": ("Nothing",)},
        "otherRole": {"rangeIncludes": ("Nothing",)},
    }

    class BlockingGraph(FakeGraphLookup):
        def node_arcs_many(self, dcids, *, properties, deadline=None):
            # `uncached` is a list, so compare tuples to match this read.
            if tuple(dcids) == ("BranchB",) and tuple(properties) == ("subClassOf",):
                second_read_started.set()
                assert release_second_read.wait(timeout=5), "test deadlocked"
            return super().node_arcs_many(dcids, properties=properties, deadline=deadline)

    graph = BlockingGraph(nodes=nodes)
    generation = FakeGeneration(token="gen-a")
    admission = Admission(graph, generation=generation)

    # Cache BranchA and Place under gen-a before building the mixed frontier.
    assert _run(admission.admits("observationAbout", "BranchA")) is False

    async def walk_sub():
        return await admission.admits("observationAbout", "Sub")

    async def flip_generation():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, second_read_started.wait, 5)
        generation.advance("gen-b")
        await admission.admits("otherRole", "Country")
        release_second_read.set()

    async def race():
        t_walk = asyncio.create_task(walk_sub())
        t_flip = asyncio.create_task(flip_generation())
        result = await t_walk
        await t_flip
        return result

    result = _run(race())

    assert result is False


def test_hit_under_the_same_generation_token_avoids_a_second_graph_read():
    graph = _graph()
    generation = FakeGeneration(token="gen-a")
    admission = Admission(graph, generation=generation)

    assert _run(admission.admits("financingProvider", "Country")) is True
    assert _run(admission.admits("financingProvider", "Country")) is True

    assert graph.node_arcs_many_calls == [(("financingProvider",), ("rangeIncludes",))]


def test_advancing_the_generation_token_misses_a_previously_cached_range():
    graph = _graph()
    generation = FakeGeneration(token="gen-a")
    admission = Admission(graph, generation=generation)

    assert _run(admission.admits("financingProvider", "Country")) is True
    generation.advance("gen-b")
    assert _run(admission.admits("financingProvider", "Country")) is True

    assert graph.node_arcs_many_calls == [
        (("financingProvider",), ("rangeIncludes",)),
        (("financingProvider",), ("rangeIncludes",)),
    ]


def test_unknown_and_known_generations_never_share_a_cache_entry():
    graph = _graph()
    generation = FakeGeneration()  # starts unknown

    admission = Admission(graph, generation=generation)

    assert _run(admission.admits("financingProvider", "Country")) is True
    generation.advance("gen-a")
    assert _run(admission.admits("financingProvider", "Country")) is True

    assert graph.node_arcs_many_calls == [
        (("financingProvider",), ("rangeIncludes",)),
        (("financingProvider",), ("rangeIncludes",)),
    ]


def test_stats_reports_entries_and_clear_empties_both_caches():
    admission = Admission(_graph())

    assert admission.stats() == {"range_entries": 0, "parents_entries": 0}
    assert _run(admission.admits("observationAbout", "State")) is True
    assert admission.stats() == {"range_entries": 1, "parents_entries": 2}

    admission.clear()
    assert admission.stats() == {"range_entries": 0, "parents_entries": 0}


def test_ttu_uses_the_known_lifetime_for_a_real_token():
    assert _ttu(("gen-a", "Country"), None, 100.0) == 100.0 + GRAPH_CACHE_MAX_AGE_SECONDS


def test_ttu_uses_the_unknown_lifetime_for_the_unknown_key():
    assert _ttu((UNKNOWN_KEY, "Country"), None, 100.0) == (
        100.0 + GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS
    )


class _LockAssertingCache(TLRUCache):
    """A TLRUCache that asserts Admission holds `_cache_lock` for each cache operation."""

    def __init__(self, *, owner: Admission, maxsize: int, ttu: Callable[..., float]) -> None:
        super().__init__(maxsize=maxsize, ttu=ttu)
        self._owner = owner

    def _assert_locked(self) -> None:
        assert self._owner._cache_lock.locked(), "cache accessed without _cache_lock held"

    def get(self, key, default=None):
        self._assert_locked()
        return super().get(key, default)

    def __getitem__(self, key):
        self._assert_locked()
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        self._assert_locked()
        super().__setitem__(key, value)

    def __contains__(self, key) -> bool:
        self._assert_locked()
        return super().__contains__(key)

    def __len__(self) -> int:
        self._assert_locked()
        return super().__len__()

    def clear(self) -> None:
        self._assert_locked()
        super().clear()


def test_every_cache_access_holds_the_lock():
    """Miss, hit, stats, and clear all exercise cache operations while the lock assertion is active."""
    admission = Admission(_graph())
    admission._range_cache = _LockAssertingCache(
        owner=admission, maxsize=_RANGE_CACHE_MAXSIZE, ttu=_ttu
    )
    admission._parents_cache = _LockAssertingCache(
        owner=admission, maxsize=_PARENTS_CACHE_MAXSIZE, ttu=_ttu
    )

    assert _run(admission.admits("observationAbout", "State")) is True
    assert _run(admission.admits("observationAbout", "State")) is True
    admission.stats()
    admission.clear()


class _SelfClearingParentsCache(TLRUCache):
    """Clear the cache inside each write, before any later read can retrieve the entry."""

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        super().clear()


def test_parents_cache_cleared_from_inside_its_own_write_still_returns_the_right_answer():
    """`admits()` returns from fetched values when its cache clears inside `__setitem__`."""
    graph = _graph()
    admission = Admission(graph)
    admission._parents_cache = _SelfClearingParentsCache(maxsize=_PARENTS_CACHE_MAXSIZE, ttu=_ttu)

    assert _run(admission.admits("observationAbout", "State")) is True


def test_tiny_maxsize_that_evicts_its_own_batch_writes_still_returns_correct_parents():
    """A two-dcid frontier exceeds this one-entry cache; the answer uses fetched values."""
    nodes: dict[str, dict[str, tuple[str, ...]]] = {
        "Multi": {"subClassOf": ("BranchA", "BranchB")},
        "BranchA": {"subClassOf": ("Place",)},
        "BranchB": {"subClassOf": ("Place",)},
        "observationAbout": {"rangeIncludes": ("Place",)},
    }
    graph = FakeGraphLookup(nodes=nodes)
    admission = Admission(graph)
    admission._parents_cache = TLRUCache(maxsize=1, ttu=_ttu)

    assert _run(admission.admits("observationAbout", "Multi")) is True
