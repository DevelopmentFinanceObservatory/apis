"""Each test isolates the process-wide singleton while owning its acquire/release lifecycle."""

from __future__ import annotations

import threading
import time

import pytest

import entity_resolution.shared as shared_module
from entity_resolution.groups import build_group_registry
from entity_resolution.membership import Projection, build_policy
from entity_resolution.shared import (
    acquire_shared,
    release_shared,
    shared_membership_policy,
    shared_memberships,
)


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


def _refuse_to_build(*_args: object) -> None:
    raise AssertionError("a membership read built the shared resolver")


def test_the_policy_is_read_once_for_the_process_without_a_shared_resolver(monkeypatch):
    """dfo-apis#1281: a caller holding no resolver (`rerender_golden`, a surface built with an
    injected resolver) built and closed the whole resolver on every policy read, and raised
    without a cache directory. Reading the bundled geo modules takes about a second."""
    monkeypatch.delenv("RESOLVEKIT_CACHE_DIR", raising=False)
    monkeypatch.setattr(shared_module._singleton, "_build", _refuse_to_build)
    monkeypatch.setattr(shared_module, "_memberships", None)

    policy = shared_membership_policy()

    assert shared_membership_policy() is policy
    assert policy.kind_of("africa") == "continent"
    assert policy.kind_of("GroupOfSeven") == "group"


def test_the_process_memberships_match_the_shared_resolvers_stores(tmp_path):
    """The bundled geo modules alone give the policy the composed resolver's stores give, and
    every shared resolver carries the process's snapshot."""
    shared = acquire_shared(cache_dir=tmp_path)
    try:
        groups = shared.run(build_group_registry, shared.resolver).result()
        composed = shared.run(lambda: build_policy(shared.resolver, groups=groups)).result()
        held = shared_memberships()
        assert shared.policy is held.policy
        assert shared.groups is held.groups
    finally:
        release_shared(shared)

    def answers(policy):
        return {
            m.ref: (m.kind, m.label, m.entity_id, m.region_class, m.members(Projection.COUNTRIES))
            for m in policy
        }

    assert answers(held.policy) == answers(composed)
    assert held.policy.omitted_regions == composed.omitted_regions
