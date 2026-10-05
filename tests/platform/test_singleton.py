"""Tests `RefCountedSingleton` concurrency, refcounts, rebuilds, and failures."""

from __future__ import annotations

import threading

import pytest

from dfo_platform.singleton import RefCountedSingleton


def test_concurrent_first_acquires_build_exactly_once() -> None:
    build_calls = 0
    build_started = threading.Event()
    release_build = threading.Event()
    lock = threading.Lock()

    def build() -> object:
        nonlocal build_calls
        with lock:
            build_calls += 1
        build_started.set()
        release_build.wait(timeout=5)
        return object()

    singleton = RefCountedSingleton(build=build)
    results: list[object] = []
    results_lock = threading.Lock()

    def acquire() -> None:
        instance = singleton.acquire()
        with results_lock:
            results.append(instance)

    threads = [threading.Thread(target=acquire) for _ in range(8)]
    for thread in threads:
        thread.start()

    assert build_started.wait(timeout=5)
    release_build.set()
    for thread in threads:
        thread.join(timeout=5)

    assert build_calls == 1
    assert len(results) == 8
    assert len({id(r) for r in results}) == 1


def test_release_at_zero_closes_exactly_once() -> None:
    close_calls: list[object] = []
    singleton = RefCountedSingleton(build=object, close=close_calls.append)

    first = singleton.acquire()
    second = singleton.acquire()
    assert first is second

    singleton.release(second)
    assert close_calls == []  # one holder remains

    singleton.release(first)
    assert close_calls == [first]  # the last release closed it, exactly once


def test_release_clears_the_cache_before_a_raising_close() -> None:
    def raising_close(_: object) -> None:
        raise RuntimeError("boom on close")

    build_calls = 0

    def build() -> object:
        nonlocal build_calls
        build_calls += 1
        return object()

    singleton = RefCountedSingleton(build=build, close=raising_close)
    first = singleton.acquire()

    with pytest.raises(RuntimeError, match="boom on close"):
        singleton.release(first)

    second = singleton.acquire()
    assert second is not first
    assert build_calls == 2


def test_acquire_after_close_rebuilds() -> None:
    built: list[object] = []

    def build() -> object:
        instance = object()
        built.append(instance)
        return instance

    singleton = RefCountedSingleton(build=build)

    first = singleton.acquire()
    singleton.release(first)

    second = singleton.acquire()
    assert second is not first
    assert built == [first, second]


def test_a_build_that_raises_leaves_no_half_built_instance() -> None:
    attempts = 0

    def flaky_build() -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("boom on the first build")
        return object()

    singleton = RefCountedSingleton(build=flaky_build)

    with pytest.raises(RuntimeError, match="boom on the first build"):
        singleton.acquire()

    instance = singleton.acquire()
    assert attempts == 2
    assert instance is not None
    singleton.release(instance)


def test_release_refuses_an_instance_acquire_never_returned() -> None:
    singleton = RefCountedSingleton(build=object)
    held = singleton.acquire()
    try:
        with pytest.raises(RuntimeError):
            singleton.release(object())
    finally:
        singleton.release(held)


def test_release_refuses_once_the_refcount_is_already_zero() -> None:
    singleton = RefCountedSingleton(build=object)
    held = singleton.acquire()
    singleton.release(held)
    with pytest.raises(RuntimeError):
        singleton.release(held)


def test_begin_release_returns_the_instance_only_on_the_last_release() -> None:
    singleton = RefCountedSingleton(build=object)
    first = singleton.acquire()
    second = singleton.acquire()

    assert singleton.begin_release(second) is None
    assert singleton.begin_release(first) is first

    rebuilt = singleton.acquire()
    assert rebuilt is not first


def test_force_reset_drops_the_instance_regardless_of_refcount() -> None:
    singleton = RefCountedSingleton(build=object)
    first = singleton.acquire()
    singleton.acquire()  # a second holder, never released

    singleton.force_reset()

    second = singleton.acquire()
    assert second is not first


def test_poisoned_for_test_restores_the_previous_instance_on_exit() -> None:
    singleton = RefCountedSingleton(build=object)
    real = singleton.acquire()
    fake = object()

    with singleton.poisoned_for_test(fake):
        assert singleton.acquire() is fake

    assert singleton.acquire() is real
    singleton.release(real)
    singleton.release(real)


def test_isolated_restores_a_snapshot_held_by_another_caller() -> None:
    singleton = RefCountedSingleton(build=object)
    outer = singleton.acquire()  # e.g. a session-scoped fixture elsewhere

    with singleton.isolated():
        inner = singleton.acquire()
        assert inner is not outer
        singleton.release(inner)

    assert singleton.acquire() is outer
    singleton.release(outer)
    singleton.release(outer)
