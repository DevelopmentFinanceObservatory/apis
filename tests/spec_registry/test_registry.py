"""Tests spec definition writes, quarantine, and store errors."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Generator
from contextlib import contextmanager

import pytest

from dataset_store.errors import PoolTimeoutError
from spec_registry.errors import (
    IdentityCollision,
    IdentityQuarantined,
    SpecIdMismatch,
    SpecRegistryStoreError,
    SpecRegistryUnavailable,
)
from spec_registry.identity import recompute_identity
from spec_registry.registry import (
    IDENTITY_COLLISION_QUARANTINED_MARKER,
    LiveSpecRegistry,
)

from .conftest import FakeDatasetStore, FakeRegistryState, minimal_identity, shape_for


class _RaisingTransaction:
    """Raise `exc` from every query method."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def execute(self, sql: str, params: object = ()) -> None:
        raise self._exc

    def fetch_one(self, sql: str, params: object = ()) -> object:
        raise self._exc

    def fetch_all(self, sql: str, params: object = ()) -> object:
        raise self._exc


class _RaisingFakeDatasetStore:
    """Raise `exc` when a query runs inside `transaction()`."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    @contextmanager
    def transaction(self) -> Generator[_RaisingTransaction, None, None]:
        yield _RaisingTransaction(self._exc)


class _QuarantineInsertFailingTransaction:
    """Delegate ordinary queries. Raise `exc` on quarantine inserts."""

    def __init__(self, inner: object, exc: Exception) -> None:
        self._inner = inner
        self._exc = exc

    def execute(self, sql: str, params: object = ()) -> None:
        if "INSERT INTO datasets.identity_quarantine" in sql:
            raise self._exc
        self._inner.execute(sql, params)

    def fetch_one(self, sql: str, params: object = ()) -> object:
        return self._inner.fetch_one(sql, params)

    def fetch_all(self, sql: str, params: object = ()) -> object:
        return self._inner.fetch_all(sql, params)


class _QuarantineInsertFailingDatasetStore:
    """Wrap `FakeDatasetStore` and fail identity quarantine inserts with `exc`."""

    def __init__(self, *, state: FakeRegistryState, exc: Exception) -> None:
        self._inner = FakeDatasetStore(state=state)
        self._exc = exc

    @contextmanager
    def transaction(self) -> Generator[_QuarantineInsertFailingTransaction, None, None]:
        with self._inner.transaction() as tx:
            yield _QuarantineInsertFailingTransaction(tx, self._exc)


def _canonical_definition(*, identity_definition: object, schema_version: int) -> dict:
    """Build the canonical `spec_definition.definition` value persisted by `register_definition`."""
    recomputed = recompute_identity(identity_definition)
    return {
        "schema_version": schema_version,
        "spec_id": recomputed.spec_id,
        "identity": json.loads(recomputed.identity_bytes),
    }


def test_register_definition_inserts_new() -> None:
    registry = LiveSpecRegistry(store=FakeDatasetStore())
    identity_definition = minimal_identity(shape=shape_for("s1"))
    spec_id = recompute_identity(identity_definition).spec_id

    result = registry.register_definition(
        identity_definition=identity_definition,
        schema_version=1,
        supplied_spec_id=spec_id,
    )

    assert result.spec_id == spec_id
    assert result.schema_version == 1
    assert result.inserted is True


def test_register_definition_idempotent_repeat() -> None:
    store = FakeDatasetStore()
    registry = LiveSpecRegistry(store=store)
    identity_definition = minimal_identity(shape=shape_for("s2"))
    spec_id = recompute_identity(identity_definition).spec_id

    first = registry.register_definition(
        identity_definition=identity_definition,
        schema_version=1,
        supplied_spec_id=spec_id,
    )
    second = registry.register_definition(
        identity_definition=identity_definition,
        schema_version=1,
        supplied_spec_id=spec_id,
    )

    assert first.inserted is True
    assert second.inserted is False
    assert first.spec_id == second.spec_id
    assert len(store.state.spec_registry) == 1
    assert store.state.spec_definition[(spec_id, 1)]["definition"] == _canonical_definition(
        identity_definition=identity_definition, schema_version=1
    )


def test_register_definition_persists_the_canonical_projection_regardless_of_wire_variant() -> None:
    """Entity key order changes wire bytes but must preserve the spec ID and stored projection."""
    store = FakeDatasetStore()
    registry = LiveSpecRegistry(store=store)
    ordered = minimal_identity(shape=shape_for("s2b"), entities={"a": ["x1"], "b": ["y1"]})
    reordered = minimal_identity(shape=shape_for("s2b"), entities={"b": ["y1"], "a": ["x1"]})
    spec_id = recompute_identity(ordered).spec_id
    assert recompute_identity(reordered).spec_id == spec_id

    first = registry.register_definition(
        identity_definition=ordered, schema_version=1, supplied_spec_id=spec_id
    )
    second = registry.register_definition(
        identity_definition=reordered, schema_version=1, supplied_spec_id=spec_id
    )

    assert first.inserted is True
    assert second.inserted is False
    stored = store.state.spec_definition[(spec_id, 1)]["definition"]
    assert stored == _canonical_definition(identity_definition=ordered, schema_version=1)
    # The stored identity matches the bytes held in the registry row.
    row_identity_bytes = store.state.spec_registry[spec_id]["identity_bytes"]
    assert stored["identity"] == json.loads(row_identity_bytes)


def test_register_definition_spec_id_mismatch() -> None:
    registry = LiveSpecRegistry(store=FakeDatasetStore())
    identity_definition = minimal_identity(shape=shape_for("s3"))

    with pytest.raises(SpecIdMismatch) as excinfo:
        registry.register_definition(
            identity_definition=identity_definition,
            schema_version=1,
            supplied_spec_id="spec_wrong",
        )
    assert excinfo.value.supplied == "spec_wrong"
    assert excinfo.value.computed == recompute_identity(identity_definition).spec_id


def test_register_definition_byte_mismatch_quarantines(caplog: pytest.LogCaptureFixture) -> None:
    store = FakeDatasetStore()
    registry = LiveSpecRegistry(store=store)
    identity_definition = minimal_identity(shape=shape_for("s4"))
    recomputed = recompute_identity(identity_definition)

    # Seed a byte-different row for the same ID to model a hash collision or corrupted record.
    store.state.spec_registry[recomputed.spec_id] = {
        "identity_version": "spec1",
        "identity_bytes": b"different bytes entirely",
        "identity_digest": "0" * 64,
        "trace_id": None,
    }

    with caplog.at_level(logging.ERROR), pytest.raises(IdentityCollision) as excinfo:
        registry.register_definition(
            identity_definition=identity_definition,
            schema_version=1,
            supplied_spec_id=recomputed.spec_id,
            trace_id="trace-1",
        )

    assert excinfo.value.subject_id == recomputed.spec_id
    assert IDENTITY_COLLISION_QUARANTINED_MARKER in caplog.text

    assert len(store.state.identity_quarantine) == 1
    quarantined = store.state.identity_quarantine[0]
    assert quarantined["id_kind"] == "spec"
    assert quarantined["subject_id"] == recomputed.spec_id
    assert quarantined["held_bytes"] == b"different bytes entirely"
    assert quarantined["incoming_bytes"] == recomputed.identity_bytes
    assert quarantined["trace_id"] == "trace-1"

    assert (recomputed.spec_id, 1) not in store.state.spec_definition


def test_concurrent_identical_inserts_converge() -> None:
    shared_state = FakeRegistryState()
    identity_definition = minimal_identity(shape=shape_for("s5"))
    spec_id = recompute_identity(identity_definition).spec_id

    results: list = []
    errors: list = []

    def _register() -> None:
        registry = LiveSpecRegistry(store=FakeDatasetStore(state=shared_state))
        try:
            results.append(
                registry.register_definition(
                    identity_definition=identity_definition,
                    schema_version=1,
                    supplied_spec_id=spec_id,
                )
            )
        except Exception as exc:  # pragma: no cover - surfaced via `errors`
            errors.append(exc)

    threads = [threading.Thread(target=_register) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 2
    assert {r.spec_id for r in results} == {spec_id}
    assert sorted(r.inserted for r in results) == [False, True]
    assert len(shared_state.spec_registry) == 1


# Quarantined identifiers refuse both the held and colliding identities.


def test_register_definition_refuses_reposting_the_held_identity_once_quarantined() -> None:
    store = FakeDatasetStore()
    registry = LiveSpecRegistry(store=store)
    identity_definition = minimal_identity(shape=shape_for("s6"))
    recomputed = recompute_identity(identity_definition)

    # Seed a held row and its quarantine entry to model an existing collision.
    store.state.spec_registry[recomputed.spec_id] = {
        "identity_version": "spec1",
        "identity_bytes": recomputed.identity_bytes,
        "identity_digest": recomputed.identity_digest,
        "trace_id": None,
    }
    store.state.identity_quarantine.append(
        {
            "id_kind": "spec",
            "subject_id": recomputed.spec_id,
            "held_bytes": recomputed.identity_bytes,
            "incoming_bytes": b"some other colliding bytes",
            "trace_id": "trace-quarantine",
        }
    )

    with pytest.raises(IdentityQuarantined) as excinfo:
        registry.register_definition(
            identity_definition=identity_definition,
            schema_version=1,
            supplied_spec_id=recomputed.spec_id,
        )

    assert excinfo.value.subject_id == recomputed.spec_id
    assert excinfo.value.id_kind == "spec"
    # The held bytes match, and quarantine blocks registration.
    assert (recomputed.spec_id, 1) not in store.state.spec_definition


def test_register_definition_still_refuses_the_colliding_identity_once_quarantined() -> None:
    store = FakeDatasetStore()
    registry = LiveSpecRegistry(store=store)
    identity_definition = minimal_identity(shape=shape_for("s7"))
    recomputed = recompute_identity(identity_definition)

    store.state.spec_registry[recomputed.spec_id] = {
        "identity_version": "spec1",
        "identity_bytes": b"different bytes entirely",
        "identity_digest": "0" * 64,
        "trace_id": None,
    }
    store.state.identity_quarantine.append(
        {
            "id_kind": "spec",
            "subject_id": recomputed.spec_id,
            "held_bytes": b"different bytes entirely",
            "incoming_bytes": recomputed.identity_bytes,
            "trace_id": "trace-quarantine",
        }
    )

    with pytest.raises(IdentityCollision) as excinfo:
        registry.register_definition(
            identity_definition=identity_definition,
            schema_version=1,
            supplied_spec_id=recomputed.spec_id,
        )

    assert excinfo.value.subject_id == recomputed.spec_id


def test_register_definition_untouched_identifier_registers_despite_an_unrelated_quarantine() -> (
    None
):
    store = FakeDatasetStore()
    registry = LiveSpecRegistry(store=store)
    store.state.identity_quarantine.append(
        {
            "id_kind": "spec",
            "subject_id": "spec_unrelated",
            "held_bytes": b"held",
            "incoming_bytes": b"incoming",
            "trace_id": None,
        }
    )
    identity_definition = minimal_identity(shape=shape_for("s8"))
    spec_id = recompute_identity(identity_definition).spec_id

    first = registry.register_definition(
        identity_definition=identity_definition,
        schema_version=1,
        supplied_spec_id=spec_id,
    )
    second = registry.register_definition(
        identity_definition=identity_definition,
        schema_version=1,
        supplied_spec_id=spec_id,
    )

    assert first.inserted is True
    assert second.inserted is False


# Store failures expose retryability without driver details.


def test_register_definition_pool_timeout_raises_retryable_unavailable() -> None:
    pool_timeout = PoolTimeoutError("no connection freed up within 30.0s (pool size 2)")
    registry = LiveSpecRegistry(store=_RaisingFakeDatasetStore(pool_timeout))
    identity_definition = minimal_identity(shape=shape_for("s9"))
    spec_id = recompute_identity(identity_definition).spec_id

    with pytest.raises(SpecRegistryUnavailable) as excinfo:
        registry.register_definition(
            identity_definition=identity_definition,
            schema_version=1,
            supplied_spec_id=spec_id,
        )

    assert "PoolTimeoutError" in str(excinfo.value)
    assert "no connection freed up" not in str(excinfo.value)


def test_register_definition_generic_store_failure_raises_non_retryable_with_safe_class_name() -> (
    None
):
    driver_exc = RuntimeError("driver-secret: connection reset by peer on host db.internal")
    registry = LiveSpecRegistry(store=_RaisingFakeDatasetStore(driver_exc))
    identity_definition = minimal_identity(shape=shape_for("s10"))
    spec_id = recompute_identity(identity_definition).spec_id

    with pytest.raises(SpecRegistryStoreError) as excinfo:
        registry.register_definition(
            identity_definition=identity_definition,
            schema_version=1,
            supplied_spec_id=spec_id,
        )

    assert "RuntimeError" in str(excinfo.value)
    assert "driver-secret" not in str(excinfo.value)


def test_register_definition_sqlstate_surfaces_in_the_store_error_message() -> None:
    driver_exc = RuntimeError({"C": "42702", "M": "column reference is ambiguous"})
    registry = LiveSpecRegistry(store=_RaisingFakeDatasetStore(driver_exc))
    identity_definition = minimal_identity(shape=shape_for("s10b"))
    spec_id = recompute_identity(identity_definition).spec_id

    with pytest.raises(SpecRegistryStoreError) as excinfo:
        registry.register_definition(
            identity_definition=identity_definition,
            schema_version=1,
            supplied_spec_id=spec_id,
        )

    assert "sqlstate=42702" in str(excinfo.value)


def test_quarantine_insert_failure_raises_non_retryable_with_safe_class_name() -> None:
    state = FakeRegistryState()
    driver_exc = RuntimeError("driver-secret: quarantine insert connection reset by peer")
    registry = LiveSpecRegistry(
        store=_QuarantineInsertFailingDatasetStore(state=state, exc=driver_exc)
    )
    identity_definition = minimal_identity(shape=shape_for("s11"))
    recomputed = recompute_identity(identity_definition)

    # Seed a byte-different row to enter the collision path and test the quarantine write failure.
    state.spec_registry[recomputed.spec_id] = {
        "identity_version": "spec1",
        "identity_bytes": b"different bytes entirely",
        "identity_digest": "0" * 64,
        "trace_id": None,
    }

    with pytest.raises(SpecRegistryStoreError) as excinfo:
        registry.register_definition(
            identity_definition=identity_definition,
            schema_version=1,
            supplied_spec_id=recomputed.spec_id,
        )

    assert "RuntimeError" in str(excinfo.value)
    assert "driver-secret" not in str(excinfo.value)
