"""Checks borrowing and closing the presentation store's shared pool."""

from __future__ import annotations

import pg8000.exceptions
import pytest

from assembler.ports import RevisionRecord
from dataset_store.pool import ConnectionPool
from dataset_store.store import LiveDatasetStore
from presentation_store import provider
from presentation_store.connection import PresentationStoreSettings
from presentation_store.store import LivePresentationStore
from presentation_store.store_errors import PresentationStoreUnavailable
from tests.dataset_store.conftest import RecordingConnection


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


def _create(store: LivePresentationStore) -> None:
    with store.begin_write(
        component_id=store.allocate_component_id(), is_create=True, composition_id=None
    ) as ctx:
        ctx.append(
            record=RevisionRecord(
                component_id=ctx.component_id,
                revision=1,
                stored_intent={},
                resolved_payload={},
                catalog_version="2026.8.0",
                schema_version=1,
                formatter_profile={},
            ),
            composition_entry=None,
        )


@pytest.mark.parametrize(
    "dead_from",
    [
        pytest.param("INSERT INTO presentation.component ", id="prepare_write"),
        pytest.param("INSERT INTO presentation.component_revision", id="append"),
    ],
)
def test_a_write_after_a_dead_connection_dials_a_new_one(monkeypatch, dead_from):
    """A dead connection raises `InterfaceError` and its `rollback()` returns quietly, as
    pg8000's does outside a transaction, so only the translated error can tell the pool.
    """
    dead = RecordingConnection()
    healthy = RecordingConnection()

    def _die(sql, _params):
        if sql.startswith(dead_from):
            raise pg8000.exceptions.InterfaceError("network error")

    dead.on_execute = _die
    dialled = iter([dead, healthy])
    monkeypatch.setattr(
        PresentationStoreSettings,
        "connect_callable",
        lambda self, *, connector: lambda: next(dialled),
    )
    store = LivePresentationStore(store=LiveDatasetStore(connections=provider.shared_pool()))

    with pytest.raises(PresentationStoreUnavailable):
        _create(store)
    _create(store)

    assert dead.closed
    assert healthy.commits == 1
