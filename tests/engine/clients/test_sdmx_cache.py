"""Test CachingSdmxClient role-series caching with offline SDMX doubles."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest

from graph_generation import FakeGeneration, GenerationStamp
from search.engine.clients.graph import Facet
from search.engine.clients.sdmx_cache import CachingSdmxClient
from search.engine.errors import GraphInfraError, RoleNotDeclaredError
from tests.engine._fake_sdmx import FakeSdmxClient

_RECIPIENT = (("recipient", "country/KEN"),)
_ENTITIES = {"recipient": ["country/KEN"]}


def _facet(*, facet_id: str = "f1", value: float = 1.0, date: str = "2020") -> Facet:
    return Facet(
        earliest_date=date,
        latest_date=date,
        obs_count=1,
        dates=(date,),
        values=(value,),
        provenance_id="dc/base/Fake",
        measurement_method="FakeMethod",
        unit="USDollar",
        scaling_factor=None,
        facet_id=facet_id,
    )


def _fake(observations: dict, **kwargs) -> FakeSdmxClient:
    return FakeSdmxClient(observations=observations, availability={}, **kwargs)


class _RogueFacetIdSdmxClient:
    """Return a fixed facet id to test the cache's response validation."""

    def __init__(self, facet: Facet) -> None:
        self._facet = facet
        self.series_calls: list[dict] = []

    def role_series(
        self,
        *,
        variables,
        entities,
        facet_ids,
        time_period=None,
        max_rows,
        max_bytes,
        deadline=None,
    ):
        self.series_calls.append({"variables": list(variables), "facet_ids": set(facet_ids)})
        return {variables[0]: {_RECIPIENT: [self._facet]}}


# Cache hit, miss, expiry, eviction, concurrency, and failure behaviour.


def test_hit_within_ttl_makes_no_second_call():
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})
    cache = CachingSdmxClient(fake, maxsize=10)

    r1 = cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    r2 = cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )

    assert r1 == r2
    assert len(fake.series_calls) == 1


def test_expired_entry_refetches():
    """Without a generation provider, entries use the unknown stamp's max_age_s."""
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})
    clock = [0.0]
    cache = CachingSdmxClient(fake, maxsize=10, clock=lambda: clock[0])
    max_age_s = GenerationStamp(token=None).max_age_s

    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 1

    clock[0] = max_age_s - 1  # one second before expiry
    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 1

    clock[0] = max_age_s + 1  # one second after expiry
    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 2


def test_subset_read_reuses_superset_entries_and_fetches_only_misses():
    fake = _fake(
        {
            "V1": {_RECIPIENT: [_facet(facet_id="f1")]},
            "V2": {_RECIPIENT: [_facet(facet_id="f1")]},
        }
    )
    cache = CachingSdmxClient(fake, maxsize=10)

    both = cache.role_series(
        variables=["V1", "V2"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 1

    v1_only = cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert v1_only["V1"] == both["V1"]
    assert len(fake.series_calls) == 1  # fully served from cache, no new call

    # A read naming a mix of a cached hit and a genuine miss fetches only the miss.
    fake2 = _fake(
        {
            "V1": {_RECIPIENT: [_facet(facet_id="f1")]},
            "V3": {_RECIPIENT: [_facet(facet_id="f1")]},
        }
    )
    cache2 = CachingSdmxClient(fake2, maxsize=10)
    cache2.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake2.series_calls) == 1
    cache2.role_series(
        variables=["V1", "V3"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake2.series_calls) == 2
    assert fake2.series_calls[1]["variables"] == ["V3"]  # only the miss reached the wire


def test_different_entities_or_time_period_are_different_keys():
    fake = _fake(
        {
            "V1": {
                _RECIPIENT: [_facet(facet_id="f1")],
                (("recipient", "country/USA"),): [_facet(facet_id="f1")],
            }
        }
    )
    cache = CachingSdmxClient(fake, maxsize=10)

    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/USA"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 2

    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        time_period=["2020"],
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 3


def test_two_concurrent_identical_reads_make_one_call():
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})

    class _SlowClient:
        def __init__(self, inner):
            self._inner = inner
            self.call_count = 0
            self._lock = threading.Lock()
            self.started = threading.Event()
            self.release = threading.Event()

        def role_series(self, **kwargs):
            with self._lock:
                self.call_count += 1
            self.started.set()
            assert self.release.wait(timeout=5), "test deadlocked waiting for release"
            return self._inner.role_series(**kwargs)

        def availability_values(self, **kwargs):
            return self._inner.availability_values(**kwargs)

        def subject_observations(self, **kwargs):
            return self._inner.subject_observations(**kwargs)

    slow = _SlowClient(fake)
    cache = CachingSdmxClient(slow, maxsize=10)
    results: list = []

    def worker():
        results.append(
            cache.role_series(
                variables=["V1"],
                entities={"recipient": ["country/KEN"]},
                facet_ids={"f1"},
                max_rows=1000,
                max_bytes=1000,
            )
        )

    t1 = threading.Thread(target=worker)
    t1.start()
    assert slow.started.wait(timeout=5), "first call never reached the fake"
    # The in-flight key is registered before fetch, so t2 parks as a waiter.
    t2 = threading.Thread(target=worker)
    t2.start()
    time.sleep(0.05)  # give t2 a chance to park on the in-flight slot
    slow.release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert slow.call_count == 1
    assert results[0] == results[1]


def test_exception_is_not_cached():
    def raise_first_call_only(call_index: int):
        return RoleNotDeclaredError("boom") if call_index == 0 else None

    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}}, raise_series=raise_first_call_only)
    cache = CachingSdmxClient(fake, maxsize=10)

    with pytest.raises(RoleNotDeclaredError):
        cache.role_series(
            variables=["V1"],
            entities={"recipient": ["country/KEN"]},
            facet_ids={"f1"},
            max_rows=1000,
            max_bytes=1000,
        )

    result = cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert result["V1"]
    assert len(fake.series_calls) == 2  # Both attempts reach the fake.


def test_waiter_on_failing_inflight_fetch_gets_the_same_failure():
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})

    class _FailingSlowClient:
        def __init__(self, inner):
            self._inner = inner
            self.started = threading.Event()
            self.release = threading.Event()

        def role_series(self, **kwargs):
            self.started.set()
            assert self.release.wait(timeout=5), "test deadlocked waiting for release"
            raise RoleNotDeclaredError("owner's own failure")

        def availability_values(self, **kwargs):
            return self._inner.availability_values(**kwargs)

        def subject_observations(self, **kwargs):
            return self._inner.subject_observations(**kwargs)

    slow = _FailingSlowClient(fake)
    cache = CachingSdmxClient(slow, maxsize=10)
    errors: list[BaseException] = []

    def worker():
        try:
            cache.role_series(
                variables=["V1"],
                entities={"recipient": ["country/KEN"]},
                facet_ids={"f1"},
                max_rows=1000,
                max_bytes=1000,
            )
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t1.start()
    assert slow.started.wait(timeout=5)
    t2 = threading.Thread(target=worker)
    t2.start()
    time.sleep(0.05)  # give t2 a chance to park on the in-flight slot
    slow.release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(errors) == 2
    assert all(isinstance(exc, RoleNotDeclaredError) for exc in errors)


def test_waiter_deadline_expires_before_leader_leader_result_still_cached():
    """An expired waiter times out while the leader fetch continues and caches its result."""
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})

    class _SlowClient:
        def __init__(self, inner):
            self._inner = inner
            self.call_count = 0
            self.started = threading.Event()
            self.release = threading.Event()

        def role_series(self, **kwargs):
            self.call_count += 1
            self.started.set()
            assert self.release.wait(timeout=5), "test deadlocked waiting for release"
            return self._inner.role_series(**kwargs)

        def availability_values(self, **kwargs):
            return self._inner.availability_values(**kwargs)

        def subject_observations(self, **kwargs):
            return self._inner.subject_observations(**kwargs)

    slow = _SlowClient(fake)
    cache = CachingSdmxClient(slow, maxsize=10)
    leader_result: list = []
    waiter_errors: list[BaseException] = []

    def leader():
        leader_result.append(
            cache.role_series(
                variables=["V1"],
                entities={"recipient": ["country/KEN"]},
                facet_ids={"f1"},
                max_rows=1000,
                max_bytes=1000,
            )
        )

    def waiter():
        # The expired deadline makes the waiter time out as soon as it parks.
        try:
            cache.role_series(
                variables=["V1"],
                entities={"recipient": ["country/KEN"]},
                facet_ids={"f1"},
                max_rows=1000,
                max_bytes=1000,
                deadline=time.monotonic() - 1.0,
            )
        except BaseException as exc:
            waiter_errors.append(exc)

    t1 = threading.Thread(target=leader)
    t1.start()
    assert slow.started.wait(timeout=5), "leader never reached the fake"

    t2 = threading.Thread(target=waiter)
    t2.start()
    t2.join(timeout=5)
    assert not t2.is_alive(), "waiter did not time out at its own deadline"
    assert len(waiter_errors) == 1
    assert isinstance(waiter_errors[0], TimeoutError)

    # Releasing the leader lets its independent fetch finish and cache.
    slow.release.set()
    t1.join(timeout=5)
    assert not t1.is_alive(), "leader never finished after being released"
    assert leader_result and leader_result[0]["V1"]

    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert slow.call_count == 1  # Only the leader fetched.


class _FakeCriticalError(BaseException):
    """Model a KeyboardInterrupt-like failure outside the Exception hierarchy."""


def test_leader_base_exception_wakes_waiters_and_a_later_call_fetches_normally():
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})

    class _FlakyClient:
        def __init__(self, inner):
            self._inner = inner
            self.started = threading.Event()
            self.release = threading.Event()
            self.calls = 0

        def role_series(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                assert self.release.wait(timeout=5), "test deadlocked waiting for release"
                raise _FakeCriticalError("simulated KeyboardInterrupt-like failure")
            return self._inner.role_series(**kwargs)

        def availability_values(self, **kwargs):
            return self._inner.availability_values(**kwargs)

        def subject_observations(self, **kwargs):
            return self._inner.subject_observations(**kwargs)

    flaky = _FlakyClient(fake)
    cache = CachingSdmxClient(flaky, maxsize=10)
    errors: list[BaseException] = []

    def worker():
        try:
            cache.role_series(
                variables=["V1"],
                entities={"recipient": ["country/KEN"]},
                facet_ids={"f1"},
                max_rows=1000,
                max_bytes=1000,
            )
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t1.start()
    assert flaky.started.wait(timeout=5), "leader never reached the fake"

    t2 = threading.Thread(target=worker)
    t2.start()
    # Give t2 time to reach the registered slot before releasing the leader.
    time.sleep(0.05)
    flaky.release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive(), "a waiter leaked a stale in-flight slot"

    assert len(errors) == 2
    assert all(isinstance(exc, _FakeCriticalError) for exc in errors)

    # A fresh call for the key starts a new fetch after the leader fails.
    t3 = threading.Thread(
        target=lambda: cache.role_series(
            variables=["V1"],
            entities={"recipient": ["country/KEN"]},
            facet_ids={"f1"},
            max_rows=1000,
            max_bytes=1000,
        )
    )
    t3.start()
    t3.join(timeout=5)
    assert not t3.is_alive(), "later call blocked on a leaked in-flight slot"
    assert flaky.calls == 2  # the failed attempt, plus this fresh, successful fetch


def test_eviction_at_max_entry_bound():
    fake = _fake(
        {
            "V1": {_RECIPIENT: [_facet(facet_id="f1")]},
            "V2": {_RECIPIENT: [_facet(facet_id="f1")]},
            "V3": {_RECIPIENT: [_facet(facet_id="f1")]},
        }
    )
    cache = CachingSdmxClient(fake, maxsize=2)

    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    cache.role_series(
        variables=["V2"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    cache.role_series(
        variables=["V3"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 3  # V1, V2, V3 each a miss so far

    # V1 is least recently used and is evicted when V3 is inserted.
    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 4  # V1 re-fetched

    # V3 is the most recently used cached entry.
    cache.role_series(
        variables=["V3"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 4  # no new call


def test_disabled_at_zero_never_caches():
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})
    cache = CachingSdmxClient(fake, maxsize=0)

    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )

    assert len(fake.series_calls) == 2


# Cache and live clients validate inputs before key construction (#850).


def test_role_series_empty_facet_ids_refused_the_same_cached_and_uncached():
    """#850 requires cached and live clients to reject empty facet_ids."""
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})
    cached = CachingSdmxClient(fake, maxsize=10)
    uncached = CachingSdmxClient(fake, maxsize=0)

    for client in (cached, uncached, fake):
        with pytest.raises(ValueError, match="facet_ids"):
            client.role_series(
                variables=["V1"],
                entities=_ENTITIES,
                facet_ids=set(),
                max_rows=1000,
                max_bytes=1000,
            )

    assert fake.series_calls == []


def test_role_series_empty_variables_refused_the_same_cached_and_uncached():
    """An empty variables list must raise consistently with caching enabled or disabled."""
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})
    cached = CachingSdmxClient(fake, maxsize=10)
    uncached = CachingSdmxClient(fake, maxsize=0)

    for client in (cached, uncached, fake):
        with pytest.raises(ValueError, match="at least one variable"):
            client.role_series(
                variables=[],
                entities=_ENTITIES,
                facet_ids={"f1"},
                max_rows=1000,
                max_bytes=1000,
            )

    assert fake.series_calls == []


def test_preconditions_raise_before_any_cache_entry_is_stored():
    """A refused call stores no entry, so a later valid read reaches the client."""
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})
    cache = CachingSdmxClient(fake, maxsize=10)

    with pytest.raises(ValueError):
        cache.role_series(
            variables=[], entities=_ENTITIES, facet_ids={"f1"}, max_rows=1000, max_bytes=1000
        )

    result = cache.role_series(
        variables=["V1"], entities=_ENTITIES, facet_ids={"f1"}, max_rows=1000, max_bytes=1000
    )
    assert result["V1"]
    assert len(fake.series_calls) == 1


def test_cache_refuses_exactly_what_the_live_client_refuses():
    """Live, cached, and uncached clients reject both empty inputs before HTTP."""
    import httpx

    from search.engine.clients.sdmx import LiveSdmxClient

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"a refused call reached the wire: {request.url}")

    live = LiveSdmxClient(base="http://x", transport=httpx.MockTransport(handler))
    cached = CachingSdmxClient(live, maxsize=10)
    uncached = CachingSdmxClient(live, maxsize=0)

    def _types(fn) -> list[type]:
        raised: list[type] = []
        for client in (live, cached, uncached):
            try:
                fn(client)
            except BaseException as exc:
                raised.append(type(exc))
        return raised

    empty_variables = _types(
        lambda c: c.role_series(
            variables=[],
            entities={"provider": ["country/USA"]},
            facet_ids={"f1"},
            max_rows=100,
            max_bytes=100_000,
        )
    )
    assert empty_variables == [ValueError, ValueError, ValueError]

    empty_facet_ids = _types(
        lambda c: c.role_series(
            variables=["Count_Person"],
            entities={"provider": ["country/USA"]},
            facet_ids=set(),
            max_rows=100,
            max_bytes=100_000,
        )
    )
    assert empty_facet_ids == [ValueError, ValueError, ValueError]


def test_cache_refuses_a_duplicate_variable_that_the_live_client_refuses():
    """#850 requires duplicate variables to raise before HTTP with either cache setting."""
    import httpx

    from search.engine.clients.sdmx import LiveSdmxClient

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"a refused call reached the wire: {request.url}")

    live = LiveSdmxClient(base="http://x", transport=httpx.MockTransport(handler))
    cached = CachingSdmxClient(live, maxsize=10)
    uncached = CachingSdmxClient(live, maxsize=0)

    def _types(fn) -> list[type]:
        raised: list[type] = []
        for client in (live, cached, uncached):
            try:
                fn(client)
            except BaseException as exc:
                raised.append(type(exc))
        return raised

    duplicate_variables = _types(
        lambda c: c.role_series(
            variables=["Count_Person", "Count_Person"],
            entities={"provider": ["country/USA"]},
            facet_ids={"f1"},
            max_rows=100,
            max_bytes=100_000,
        )
    )
    assert duplicate_variables == [ValueError, ValueError, ValueError]


def test_role_series_duplicate_variables_refused_the_same_cached_and_uncached():
    """Reject duplicate variables consistently with caching enabled or disabled."""
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})
    cached = CachingSdmxClient(fake, maxsize=10)
    uncached = CachingSdmxClient(fake, maxsize=0)

    for client in (cached, uncached, fake):
        with pytest.raises(ValueError, match=r"repeats 'V1' \(2 times\)"):
            client.role_series(
                variables=["V1", "V1"],
                entities=_ENTITIES,
                facet_ids={"f1"},
                max_rows=1000,
                max_bytes=1000,
            )

    assert fake.series_calls == []


# CachingSdmxClient.role_series


def test_role_series_hit_within_ttl_makes_no_second_call():
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})
    cache = CachingSdmxClient(fake, maxsize=10)

    r1 = cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    r2 = cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert r1 == r2
    assert len(fake.series_calls) == 1


def test_role_series_different_facet_id_is_a_different_key():
    fake = _fake(
        {
            "V1": {
                _RECIPIENT: [_facet(facet_id="f1"), _facet(facet_id="f2")],
            }
        }
    )
    cache = CachingSdmxClient(fake, maxsize=10)

    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f2"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 2


def test_role_series_facet_ids_set_only_fetches_missing_members():
    """A mixed cache hit and miss fetches only the missing facet and returns both."""
    fake = _fake(
        {
            "V1": {
                _RECIPIENT: [_facet(facet_id="f1"), _facet(facet_id="f2")],
            }
        }
    )
    cache = CachingSdmxClient(fake, maxsize=10)

    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 1

    result = cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1", "f2"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 2
    assert fake.series_calls[1]["facet_ids"] == {"f2"}
    assert {f.facet_id for f in result["V1"][_RECIPIENT]} == {"f1", "f2"}


def test_mutating_a_returned_series_list_does_not_change_a_later_hit():
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})
    cache = CachingSdmxClient(fake, maxsize=10)

    result = cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    result["V1"][_RECIPIENT].clear()

    result2 = cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert result2["V1"][_RECIPIENT] == [_facet(facet_id="f1")]


def test_role_series_cold_multi_facet_same_variables_is_one_inner_call():
    """Cold facets for one variable share a single inner call."""
    fake = _fake(
        {
            "V1": {
                _RECIPIENT: [_facet(facet_id="f1"), _facet(facet_id="f2"), _facet(facet_id="f3")],
            }
        }
    )
    cache = CachingSdmxClient(fake, maxsize=100)

    result = cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1", "f2", "f3"},
        max_rows=1000,
        max_bytes=1000,
    )

    assert len(fake.series_calls) == 1
    assert fake.series_calls[0]["facet_ids"] == {"f1", "f2", "f3"}
    assert fake.series_calls[0]["variables"] == ["V1"]
    assert {f.facet_id for f in result["V1"][_RECIPIENT]} == {"f1", "f2", "f3"}


def test_role_series_partial_hit_fetches_only_missing_cells():
    """Fetch only missing variable/facet cells and return the cold-request result."""

    def _make_fake() -> FakeSdmxClient:
        return _fake(
            {
                "V1": {_RECIPIENT: [_facet(facet_id="f1"), _facet(facet_id="f2")]},
                "V2": {_RECIPIENT: [_facet(facet_id="f1"), _facet(facet_id="f2")]},
            }
        )

    fake = _make_fake()
    cache = CachingSdmxClient(fake, maxsize=100)

    # Prime just the (V1, f1) cell.
    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 1

    result = cache.role_series(
        variables=["V1", "V2"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1", "f2"},
        max_rows=1000,
        max_bytes=1000,
    )

    # Every (variable, facet_id) cell reached the wire after priming except
    # the already-cached (V1, f1).
    requested_cells = {
        (variable, facet_id)
        for call in fake.series_calls[1:]
        for variable in call["variables"]
        for facet_id in call["facet_ids"]
    }
    assert requested_cells == {("V2", "f1"), ("V1", "f2"), ("V2", "f2")}

    cold_fake = _make_fake()
    cold_cache = CachingSdmxClient(cold_fake, maxsize=100)
    cold_result = cold_cache.role_series(
        variables=["V1", "V2"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1", "f2"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert result == cold_result


def test_role_series_facets_missing_different_variables_is_one_call_per_group():
    """Group facet misses by variable set, issuing one inner call per group."""
    fake = _fake(
        {
            "V1": {
                _RECIPIENT: [_facet(facet_id="f1"), _facet(facet_id="f2"), _facet(facet_id="f3")],
            },
            "V2": {
                _RECIPIENT: [_facet(facet_id="f1"), _facet(facet_id="f2"), _facet(facet_id="f3")],
            },
        }
    )
    cache = CachingSdmxClient(fake, maxsize=100)

    # Prime (V2, f3) so f3 misses only for V1, while f1/f2 miss for both.
    cache.role_series(
        variables=["V2"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f3"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 1

    cache.role_series(
        variables=["V1", "V2"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1", "f2", "f3"},
        max_rows=1000,
        max_bytes=1000,
    )

    # One priming call plus one call per distinct missing-variable group
    # f1/f2 share {V1, V2}. f3 misses for {V1}. The misses form two groups.
    assert len(fake.series_calls) == 3
    group_facet_id_sets = {frozenset(call["facet_ids"]) for call in fake.series_calls[1:]}
    assert group_facet_id_sets == {frozenset({"f1", "f2"}), frozenset({"f3"})}
    group_variable_sets = {tuple(call["variables"]) for call in fake.series_calls[1:]}
    assert group_variable_sets == {("V1", "V2"), ("V1",)}


def test_role_series_second_identical_multi_facet_request_makes_no_new_call():
    fake = _fake(
        {
            "V1": {_RECIPIENT: [_facet(facet_id="f1"), _facet(facet_id="f2")]},
        }
    )
    cache = CachingSdmxClient(fake, maxsize=100)

    r1 = cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1", "f2"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 1

    r2 = cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1", "f2"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert r1 == r2
    assert len(fake.series_calls) == 1


def test_role_series_raises_when_wire_returns_facet_id_outside_request():
    """A returned facet outside the requested group raises before cache insertion."""
    rogue = _RogueFacetIdSdmxClient(_facet(facet_id="rogue"))
    cache = CachingSdmxClient(rogue, maxsize=10)

    with pytest.raises(GraphInfraError):
        cache.role_series(
            variables=["V1"],
            entities={"recipient": ["country/KEN"]},
            facet_ids={"f1"},
            max_rows=1000,
            max_bytes=1000,
        )
    # The retry reaches the inner client, confirming the failure stored no entry.
    assert len(rogue.series_calls) == 1
    with pytest.raises(GraphInfraError):
        cache.role_series(
            variables=["V1"],
            entities={"recipient": ["country/KEN"]},
            facet_ids={"f1"},
            max_rows=1000,
            max_bytes=1000,
        )
    assert len(rogue.series_calls) == 2


def test_role_series_raises_when_wire_returns_facet_id_none():
    """A missing facet id raises before cache insertion."""
    unset = _RogueFacetIdSdmxClient(replace(_facet(facet_id="f1"), facet_id=None))
    cache = CachingSdmxClient(unset, maxsize=10)

    with pytest.raises(GraphInfraError):
        cache.role_series(
            variables=["V1"],
            entities={"recipient": ["country/KEN"]},
            facet_ids={"f1"},
            max_rows=1000,
            max_bytes=1000,
        )


def test_role_series_cached_entries_stay_attributed_to_their_own_facet_id():
    """A batched fetch stores each facet under its own key and returns no sibling rows on a hit."""
    fake = _fake(
        {
            "V1": {
                _RECIPIENT: [_facet(facet_id="f1"), _facet(facet_id="f2")],
            }
        }
    )
    cache = CachingSdmxClient(fake, maxsize=100)

    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1", "f2"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 1

    only_f1 = cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    only_f2 = cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f2"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == 1  # both served from cache, no new call
    assert {f.facet_id for f in only_f1["V1"][_RECIPIENT]} == {"f1"}
    assert {f.facet_id for f in only_f2["V1"][_RECIPIENT]} == {"f2"}


# Generation changes isolate in-flight reads and cached entries.


def test_a_post_flip_caller_never_joins_a_pre_flip_in_flight_fetch():
    """A post-change caller starts a separate fetch under the new generation stamp."""
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})

    class _SlowClient:
        def __init__(self, inner):
            self._inner = inner
            self.call_count = 0
            self._lock = threading.Lock()
            self.started = threading.Event()
            self.release = threading.Event()

        def role_series(self, **kwargs):
            with self._lock:
                self.call_count += 1
            self.started.set()
            assert self.release.wait(timeout=5), "test deadlocked waiting for release"
            return self._inner.role_series(**kwargs)

        def availability_values(self, **kwargs):
            return self._inner.availability_values(**kwargs)

        def subject_observations(self, **kwargs):
            return self._inner.subject_observations(**kwargs)

    slow = _SlowClient(fake)
    generation = FakeGeneration(token="gen-a")
    cache = CachingSdmxClient(slow, maxsize=10, generation=generation)
    results: list = []

    def worker():
        results.append(
            cache.role_series(
                variables=["V1"],
                entities={"recipient": ["country/KEN"]},
                facet_ids={"f1"},
                max_rows=1000,
                max_bytes=1000,
            )
        )

    leader = threading.Thread(target=worker)
    leader.start()
    assert slow.started.wait(timeout=5), "leader never reached the fake"

    generation.advance("gen-b")
    slow.started.clear()
    follower = threading.Thread(target=worker)
    follower.start()
    assert slow.started.wait(timeout=5), (
        "follower never reached the fake -- it must not have parked"
    )

    slow.release.set()
    leader.join(timeout=5)
    follower.join(timeout=5)

    assert slow.call_count == 2, (
        "the post-flip caller must run its own fetch, not join the pre-flip one"
    )
    assert results[0] == results[1]


def test_caching_sdmx_client_stats_reflects_the_series_cache():
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})
    cache = CachingSdmxClient(fake, maxsize=10)

    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )

    stats = cache.stats()
    assert stats["hits"] == 1  # the repeat call
    assert stats["misses"] == 1  # the first call
    assert stats["entries"] == 1


def test_caching_sdmx_client_clear_clears_the_series_cache():
    fake = _fake({"V1": {_RECIPIENT: [_facet(facet_id="f1")]}})
    cache = CachingSdmxClient(fake, maxsize=10)

    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    cache.clear()

    assert cache.stats()["entries"] == 0
    calls_before = len(fake.series_calls)
    cache.role_series(
        variables=["V1"],
        entities={"recipient": ["country/KEN"]},
        facet_ids={"f1"},
        max_rows=1000,
        max_bytes=1000,
    )
    assert len(fake.series_calls) == calls_before + 1, "a cleared cache must re-fetch"
