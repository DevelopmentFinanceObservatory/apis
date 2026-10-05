"""Test shared-pool caching, connector lifecycle, and borrower counting."""

from __future__ import annotations

import pytest

from dataset_store import provider
from dataset_store.pool import ConnectionPool


class _RecordingConnector:
    """Count calls to `close()` on a connector double."""

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


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
    monkeypatch.delenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", raising=False)
    monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
    monkeypatch.setenv("DATASET_STORE_POOL_SIZE", "3")


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


def test_close_shared_pool_is_a_no_op_when_never_built():
    provider.close_shared_pool()
    assert provider._pool is None


def test_shared_pool_composes_iam_auth_for_the_runtime_user(monkeypatch):
    captured = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("dataset_store.connection.connect", fake_connect)

    pool = provider.shared_pool()
    with pool.borrow():
        pass

    assert captured["user"] == "runtime"
    assert captured["enable_iam_auth"] is True
    assert captured["connector"] is None


def test_shared_pool_builds_a_connector_when_an_instance_is_configured(monkeypatch):
    monkeypatch.setenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", "proj:region:inst")
    sentinel_connector = object()
    monkeypatch.setattr(provider, "build_connector", lambda: sentinel_connector)

    captured = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("dataset_store.connection.connect", fake_connect)

    pool = provider.shared_pool()
    with pool.borrow():
        pass

    assert captured["connector"] is sentinel_connector
    assert captured["instance_connection_name"] == "proj:region:inst"
    assert captured["enable_iam_auth"] is True


def test_shared_pool_uses_the_configured_pool_size(monkeypatch):
    captured_sizes = []
    original_init = ConnectionPool.__init__

    def spy_init(self, **kwargs):
        captured_sizes.append(kwargs["size"])
        original_init(self, **kwargs)

    monkeypatch.setattr(ConnectionPool, "__init__", spy_init)

    provider.shared_pool()

    assert captured_sizes == [3]


def test_close_shared_pool_closes_the_connector_when_one_was_built(monkeypatch):
    monkeypatch.setenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", "proj:region:inst")
    fake_connector = _RecordingConnector()
    monkeypatch.setattr(provider, "build_connector", lambda: fake_connector)
    monkeypatch.setattr("dataset_store.connection.connect", lambda **kwargs: object())

    provider.shared_pool()
    provider.close_shared_pool()

    assert fake_connector.close_calls == 1
    assert provider._connector is None


def test_close_shared_pool_is_a_no_op_on_the_connector_for_plain_tcp(monkeypatch):
    monkeypatch.setattr("dataset_store.connection.connect", lambda **kwargs: object())

    provider.shared_pool()
    provider.close_shared_pool()

    assert provider._connector is None


def test_rebuild_after_close_constructs_a_fresh_connector(monkeypatch):
    monkeypatch.setenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", "proj:region:inst")
    connectors = [_RecordingConnector(), _RecordingConnector()]
    monkeypatch.setattr(provider, "build_connector", lambda: connectors.pop(0))
    monkeypatch.setattr("dataset_store.connection.connect", lambda **kwargs: object())

    provider.shared_pool()
    first_connector = provider._connector
    provider.close_shared_pool()
    provider.shared_pool()
    second_connector = provider._connector

    assert first_connector is not second_connector
    assert first_connector.close_calls == 1


def _spy_on_close_all(monkeypatch: pytest.MonkeyPatch) -> list[ConnectionPool]:
    """Record real `close_all()` calls to check exactly-once closure.

    Pool state alone cannot reveal extra no-op close calls.
    """
    calls: list[ConnectionPool] = []
    original_close_all = ConnectionPool.close_all

    def spy(self: ConnectionPool) -> None:
        calls.append(self)
        original_close_all(self)

    monkeypatch.setattr(ConnectionPool, "close_all", spy)
    return calls


def test_pool_stays_open_until_the_second_of_two_borrows_returns_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_calls = _spy_on_close_all(monkeypatch)

    first = provider.shared_pool()
    second = provider.shared_pool()
    assert first is second

    provider.close_shared_pool()  # Release one borrow; the other keeps the pool open.
    assert close_calls == []
    assert provider._pool is first

    provider.close_shared_pool()  # Releasing the last borrow closes the pool.
    assert close_calls == [first]
    assert provider._pool is None


def test_borrow_count_resets_after_a_real_close_so_reopen_needs_only_one_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_calls = _spy_on_close_all(monkeypatch)

    provider.shared_pool()
    provider.close_shared_pool()
    assert len(close_calls) == 1

    # A reopened pool starts with a fresh borrow count, so one close shuts it down.
    provider.shared_pool()
    provider.close_shared_pool()
    assert len(close_calls) == 2


def test_close_calls_past_zero_borrowers_are_safe_no_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    close_calls = _spy_on_close_all(monkeypatch)

    provider.shared_pool()
    provider.close_shared_pool()
    provider.close_shared_pool()
    provider.close_shared_pool()

    assert len(close_calls) == 1
    assert provider._pool is None
    assert provider._borrowers == 0


def test_close_shared_pool_warns_on_an_unpaired_close(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="dataset_store.provider"):
        provider.close_shared_pool()

    assert "unbalanced" in caplog.text


def test_close_shared_pool_does_not_warn_on_a_balanced_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider.shared_pool()

    with caplog.at_level("WARNING", logger="dataset_store.provider"):
        provider.close_shared_pool()

    assert "unbalanced" not in caplog.text
