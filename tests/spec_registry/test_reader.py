"""Tests missing and quarantined reads plus batched lookup in `LiveSpecReader`."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import pytest

from dataset_store.errors import PoolTimeoutError
from spec_registry import LiveSpecRegistry
from spec_registry.errors import (
    IdentityQuarantined,
    SpecRegistryStoreError,
    SpecRegistryUnavailable,
)
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader

from .conftest import FakeDatasetStore, FakeRegistryState, minimal_identity, shape_for


def _registry(store: FakeDatasetStore) -> LiveSpecRegistry:
    return LiveSpecRegistry(store=store)


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


def test_read_definition_returns_none_for_an_unregistered_id():
    reader = LiveSpecReader(store=FakeDatasetStore())

    assert reader.read_definition("spec_does_not_exist") is None


def test_read_definition_returns_the_registered_definition():
    store = FakeDatasetStore()
    registry = _registry(store)
    identity = minimal_identity(shape=shape_for("shape_a"))
    recomputed = recompute_identity(identity)
    registry.register_definition(
        identity_definition=identity, schema_version=1, supplied_spec_id=recomputed.spec_id
    )
    reader = LiveSpecReader(store=store)

    stored = reader.read_definition(recomputed.spec_id)

    assert stored is not None
    assert stored.spec_id == recomputed.spec_id
    assert stored.schema_version == 1
    assert stored.identity_bytes == recomputed.identity_bytes
    assert stored.identity_digest == recomputed.identity_digest
    assert stored.identity["shape"] == shape_for("shape_a")


def test_read_definitions_batches_multiple_ids_in_one_call():
    store = FakeDatasetStore()
    registry = _registry(store)
    identities = [minimal_identity(shape=shape_for(f"shape_{i}")) for i in range(3)]
    spec_ids = []
    for identity in identities:
        recomputed = recompute_identity(identity)
        registry.register_definition(
            identity_definition=identity, schema_version=1, supplied_spec_id=recomputed.spec_id
        )
        spec_ids.append(recomputed.spec_id)

    reader = LiveSpecReader(store=store)
    results = reader.read_definitions([*spec_ids, "spec_unregistered"])

    assert set(results) == set(spec_ids)
    for spec_id in spec_ids:
        assert results[spec_id].spec_id == spec_id


def test_read_definitions_of_empty_list_returns_empty_dict():
    reader = LiveSpecReader(store=FakeDatasetStore())
    assert reader.read_definitions([]) == {}


def test_read_definition_raises_identity_quarantined_for_a_quarantined_spec_id():
    state = FakeRegistryState()
    store = FakeDatasetStore(state=state)
    registry = _registry(store)
    identity = minimal_identity(shape=shape_for("shape_a"))
    recomputed = recompute_identity(identity)
    registry.register_definition(
        identity_definition=identity, schema_version=1, supplied_spec_id=recomputed.spec_id
    )
    state.identity_quarantine.append(
        {
            "id_kind": "spec",
            "subject_id": recomputed.spec_id,
            "held_bytes": recomputed.identity_bytes,
            "incoming_bytes": b"{}",
            "trace_id": None,
        }
    )
    reader = LiveSpecReader(store=store)

    with pytest.raises(IdentityQuarantined):
        reader.read_definition(recomputed.spec_id)


def test_read_definitions_pool_timeout_raises_retryable_unavailable() -> None:
    pool_timeout = PoolTimeoutError("no connection freed up within 30.0s (pool size 2)")
    reader = LiveSpecReader(store=_RaisingFakeDatasetStore(pool_timeout))

    with pytest.raises(SpecRegistryUnavailable) as excinfo:
        reader.read_definitions(["spec_x"])

    assert "PoolTimeoutError" in str(excinfo.value)
    assert "no connection freed up" not in str(excinfo.value)


def test_read_definitions_generic_store_failure_raises_non_retryable_with_safe_class_name() -> None:
    driver_exc = RuntimeError("driver-secret: connection reset by peer on host db.internal")
    reader = LiveSpecReader(store=_RaisingFakeDatasetStore(driver_exc))

    with pytest.raises(SpecRegistryStoreError) as excinfo:
        reader.read_definitions(["spec_x"])

    assert "RuntimeError" in str(excinfo.value)
    assert "driver-secret" not in str(excinfo.value)


def test_read_definitions_sqlstate_surfaces_in_the_store_error_message() -> None:
    driver_exc = RuntimeError({"C": "42702", "M": "column reference is ambiguous"})
    reader = LiveSpecReader(store=_RaisingFakeDatasetStore(driver_exc))

    with pytest.raises(SpecRegistryStoreError) as excinfo:
        reader.read_definitions(["spec_x"])

    assert "sqlstate=42702" in str(excinfo.value)


def _registered(registry: LiveSpecRegistry, name: str):
    recomputed = recompute_identity(minimal_identity(shape=shape_for(name)))
    registry.register_definition(
        identity_definition=recomputed.identity,
        schema_version=1,
        supplied_spec_id=recomputed.spec_id,
    )
    return recomputed


def test_read_served_spec_ids_returns_the_bound_ids_newest_first():
    store = FakeDatasetStore()
    registry = _registry(store)
    posted = _registered(registry, "posted")
    served = [recompute_identity(minimal_identity(shape=shape_for(f"s{i}"))) for i in range(2)]
    for identity in served:
        registry.register_served_definition(
            served=identity, schema_version=1, posted_spec_id=posted.spec_id
        )
    reader = LiveSpecReader(store=store)

    assert reader.read_served_spec_ids(posted.spec_id) == (served[1].spec_id, served[0].spec_id)
    assert reader.read_served_spec_ids(served[0].spec_id) == ()
    assert reader.read_served_spec_ids("spec_unregistered") is None


@pytest.mark.parametrize("quarantined", ["posted", "served"])
def test_every_read_raises_for_a_quarantined_posted_or_bound_served_id(quarantined):
    store = FakeDatasetStore()
    registry = _registry(store)
    posted = _registered(registry, "posted")
    served = recompute_identity(minimal_identity(shape=shape_for("served")))
    registry.register_served_definition(
        served=served, schema_version=1, posted_spec_id=posted.spec_id
    )
    subject_id = posted.spec_id if quarantined == "posted" else served.spec_id
    store.state.identity_quarantine.append(
        {"id_kind": "spec", "subject_id": subject_id, "held_bytes": b"a", "incoming_bytes": b"b"}
    )

    reader = LiveSpecReader(store=store)
    reads = [
        reader.read_served_spec_ids,
        reader.read_definition,
        lambda spec_id: reader.read_definitions([spec_id]),
    ]

    for read in reads:
        with pytest.raises(IdentityQuarantined) as excinfo:
            read(posted.spec_id)
        assert excinfo.value.subject_id == subject_id


@pytest.mark.parametrize(
    ("exc", "raised"),
    [
        (PoolTimeoutError("no connection freed up"), SpecRegistryUnavailable),
        (RuntimeError("driver fault"), SpecRegistryStoreError),
    ],
    ids=["pool-timeout", "store-error"],
)
def test_read_served_spec_ids_maps_store_faults(exc: Exception, raised: type):
    reader = LiveSpecReader(store=_RaisingFakeDatasetStore(exc))

    with pytest.raises(raised):
        reader.read_served_spec_ids("spec_x")
