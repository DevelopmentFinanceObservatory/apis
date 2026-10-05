"""Each test isolates the process-wide singleton while owning its acquire/release lifecycle."""

from __future__ import annotations

import threading
import time

import pytest

import entity_resolution.shared as shared_module
from entity_resolution.shared import acquire_shared, release_shared


@pytest.fixture(autouse=True)
def _isolated_singleton():
    with shared_module._singleton.isolated():
        yield


def test_acquire_shared_twice_returns_the_same_object(tmp_path):
    first = acquire_shared(cache_dir=tmp_path)
    second = acquire_shared(cache_dir=tmp_path)
    try:
        assert first is second
    finally:
        release_shared(second)
        release_shared(first)


def test_release_shared_twice_then_acquire_builds_anew(tmp_path):
    first = acquire_shared(cache_dir=tmp_path)
    second = acquire_shared(cache_dir=tmp_path)
    assert first is second

    release_shared(second)
    release_shared(first)

    third = acquire_shared(cache_dir=tmp_path)
    try:
        assert third is not first
    finally:
        release_shared(third)


def test_release_shared_refuses_an_object_it_never_returned(tmp_path):
    shared = acquire_shared(cache_dir=tmp_path)
    try:
        with pytest.raises(RuntimeError):
            release_shared(object())  # type: ignore[arg-type]
    finally:
        release_shared(shared)


def test_close_drains_the_worker_before_closing_the_resolver(tmp_path):
    shared = acquire_shared(cache_dir=tmp_path)
    started = threading.Event()
    finish = threading.Event()

    def blocking_job() -> None:
        started.set()
        finish.wait(timeout=5)

    shared.run(blocking_job)
    assert started.wait(timeout=5)

    closer = threading.Thread(target=shared.close)
    closer.start()
    try:
        # The blocked job keeps the resolver usable until close finishes.
        time.sleep(0.2)
        assert shared.resolver.resolve("France") is not None
    finally:
        finish.set()
        closer.join(timeout=5)

    assert shared.resolver._closed
