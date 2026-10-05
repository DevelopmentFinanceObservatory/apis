"""Checks borrowing and closing the presentation store's shared pool."""

from __future__ import annotations

import pytest

from dataset_store.pool import ConnectionPool
from presentation_store import provider


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    monkeypatch.setattr(provider, "_pool", None)
    monkeypatch.setattr(provider, "_connector", None)
    monkeypatch.setattr(provider, "_borrowers", 0)
    yield
    monkeypatch.setattr(provider, "_pool", None)
    monkeypatch.setattr(provider, "_connector", None)
    monkeypatch.setattr(provider, "_borrowers", 0)


@pytest.fixture(autouse=True)
def _runtime_env(monkeypatch):
    monkeypatch.delenv("PRESENTATION_STORE_INSTANCE_CONNECTION_NAME", raising=False)
    monkeypatch.setenv("PRESENTATION_STORE_RUNTIME_DB_USER", "runtime")
    monkeypatch.setenv("PRESENTATION_STORE_POOL_SIZE", "3")


def test_shared_pool_returns_the_same_object_across_calls():
    first = provider.shared_pool()
    second = provider.shared_pool()
    assert first is second
    assert isinstance(first, ConnectionPool)


def test_close_shared_pool_then_shared_pool_builds_a_fresh_pool():
    first = provider.shared_pool()
    provider.close_shared_pool()
    second = provider.shared_pool()
    assert first is not second


def test_two_borrows_require_two_closes_before_the_pool_actually_closes():
    first = provider.shared_pool()
    provider.shared_pool()  # add a second borrow

    provider.close_shared_pool()  # one borrow remains
    still_open = provider.shared_pool()
    assert still_open is first

    provider.close_shared_pool()
    provider.close_shared_pool()  # release the final borrow
    rebuilt = provider.shared_pool()
    assert rebuilt is not first
