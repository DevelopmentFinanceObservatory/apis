"""Pin inventory row uniqueness, idempotency, and write ordering over SQLite."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from google.api_core.exceptions import ServiceUnavailable

from artifact_store.errors import (
    ArtifactInventoryError,
    ArtifactRecordConflictError,
    ArtifactWriteError,
)
from artifact_store.inventory import ArtifactLocation, LiveArtifactInventory
from artifact_store.store import LiveArtifactStore, Tier

from .conftest import FakeArtifactInventoryDatasetStore, FakeGCSClient

_BUCKETS = {
    Tier.working: "dev-fin-observatory-datasets-working",
    Tier.pinned: "dev-fin-observatory-datasets-pinned",
}


@pytest.fixture
def dataset_store():
    store = FakeArtifactInventoryDatasetStore()
    try:
        yield store
    finally:
        store.close()


def test_record_writes_one_inventory_row_and_one_location_row(dataset_store):
    inventory = LiveArtifactInventory(store=dataset_store)

    inventory.record(
        digest="sha256:abc123",
        tier=Tier.working,
        bucket="dev-fin-observatory-datasets-working",
        object_name="sha256:abc123",
        size_bytes=42,
        uploaded=True,
    )

    with dataset_store.transaction() as tx:
        inventory_row = tx.fetch_one(
            "SELECT * FROM datasets.artifact_inventory WHERE artifact_digest = %s",
            ("sha256:abc123",),
        )
        location_row = tx.fetch_one(
            "SELECT * FROM datasets.artifact_location WHERE artifact_digest = %s AND tier = %s",
            ("sha256:abc123", "working"),
        )
    assert inventory_row["size_bytes"] == 42
    assert location_row["bucket"] == "dev-fin-observatory-datasets-working"
    assert location_row["object_name"] == "sha256:abc123"


def test_record_is_idempotent_on_repeat(dataset_store):
    inventory = LiveArtifactInventory(store=dataset_store)
    kwargs = {
        "digest": "sha256:repeat",
        "tier": Tier.working,
        "bucket": "dev-fin-observatory-datasets-working",
        "object_name": "sha256:repeat",
        "size_bytes": 7,
        "uploaded": True,
    }

    inventory.record(**kwargs)
    inventory.record(**kwargs)

    with dataset_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT * FROM datasets.artifact_inventory WHERE artifact_digest = %s",
            ("sha256:repeat",),
        )
    assert len(rows) == 1


def test_one_location_row_per_tier_for_the_same_digest(dataset_store):
    inventory = LiveArtifactInventory(store=dataset_store)

    inventory.record(
        digest="sha256:shared",
        tier=Tier.working,
        bucket="dev-fin-observatory-datasets-working",
        object_name="sha256:shared",
        size_bytes=3,
        uploaded=True,
    )
    inventory.record(
        digest="sha256:shared",
        tier=Tier.pinned,
        bucket="dev-fin-observatory-datasets-pinned",
        object_name="sha256:shared",
        size_bytes=3,
        uploaded=True,
    )

    with dataset_store.transaction() as tx:
        locations = tx.fetch_all(
            "SELECT tier FROM datasets.artifact_location WHERE artifact_digest = %s ORDER BY tier",
            ("sha256:shared",),
        )
        inventory_rows = tx.fetch_all(
            "SELECT * FROM datasets.artifact_inventory WHERE artifact_digest = %s",
            ("sha256:shared",),
        )
    assert [row["tier"] for row in locations] == ["pinned", "working"]
    assert len(inventory_rows) == 1


def test_record_failure_is_wrapped_without_the_driver_message(dataset_store, monkeypatch):
    inventory = LiveArtifactInventory(store=dataset_store)

    def _boom(self, *args, **kwargs):
        raise RuntimeError("driver said something secret")

    monkeypatch.setattr(type(dataset_store), "transaction", _boom)

    with pytest.raises(ArtifactInventoryError) as excinfo:
        inventory.record(
            digest="sha256:fails",
            tier=Tier.working,
            bucket="dev-fin-observatory-datasets-working",
            object_name="sha256:fails",
            size_bytes=1,
            uploaded=True,
        )
    assert "driver said something secret" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_store_put_never_calls_record_when_the_write_fails(dataset_store):
    # Inject a failed GCS write through the complete store path.
    client = FakeGCSClient()
    inventory = LiveArtifactInventory(store=dataset_store)
    store = LiveArtifactStore(buckets=_BUCKETS, inventory=inventory, client=client)

    bucket_name = _BUCKETS[Tier.working]
    client.bucket(bucket_name).raise_on_next_upload = ServiceUnavailable("simulated outage")

    with pytest.raises(ArtifactWriteError):
        store.put(b"never lands", tier=Tier.working)

    with dataset_store.transaction() as tx:
        rows = tx.fetch_all("SELECT * FROM datasets.artifact_inventory")
    assert rows == []


def test_get_location_returns_the_stored_bucket_and_object_name_once_recorded(dataset_store):
    inventory = LiveArtifactInventory(store=dataset_store)
    inventory.record(
        digest="sha256:present",
        tier=Tier.working,
        bucket="dev-fin-observatory-datasets-working",
        object_name="sha256:present",
        size_bytes=5,
        uploaded=True,
    )

    location = inventory.get_location("sha256:present", tier=Tier.working)
    assert location == ArtifactLocation(
        bucket="dev-fin-observatory-datasets-working", object_name="sha256:present"
    )


@pytest.mark.parametrize(
    ("recorded_tier", "requested_tier"),
    [
        pytest.param(None, Tier.working, id="never-recorded"),
        pytest.param(Tier.working, Tier.pinned, id="recorded-in-another-tier"),
    ],
)
def test_get_location_is_none_when_digest_or_tier_is_unrecorded(
    dataset_store, recorded_tier, requested_tier
):
    inventory = LiveArtifactInventory(store=dataset_store)
    if recorded_tier is not None:
        inventory.record(
            digest="sha256:queried",
            tier=recorded_tier,
            bucket=_BUCKETS[recorded_tier],
            object_name="sha256:queried",
            size_bytes=5,
            uploaded=True,
        )

    assert inventory.get_location("sha256:queried", tier=requested_tier) is None


def test_get_location_failure_is_wrapped_without_the_driver_message(dataset_store, monkeypatch):
    inventory = LiveArtifactInventory(store=dataset_store)

    def _boom(self, *args, **kwargs):
        raise RuntimeError("driver said something secret")

    monkeypatch.setattr(type(dataset_store), "transaction", _boom)

    with pytest.raises(ArtifactInventoryError) as excinfo:
        inventory.get_location("sha256:fails", tier=Tier.working)
    assert "driver said something secret" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_record_raises_on_a_conflicting_bucket_for_the_same_digest_and_tier(dataset_store):
    inventory = LiveArtifactInventory(store=dataset_store)
    inventory.record(
        digest="sha256:conflict",
        tier=Tier.working,
        bucket="dev-fin-observatory-datasets-working",
        object_name="sha256:conflict",
        size_bytes=9,
        uploaded=True,
    )

    with pytest.raises(ArtifactRecordConflictError) as excinfo:
        inventory.record(
            digest="sha256:conflict",
            tier=Tier.working,
            bucket="a-different-bucket",
            object_name="sha256:conflict",
            size_bytes=9,
            uploaded=True,
        )
    assert excinfo.value.digest == "sha256:conflict"
    assert excinfo.value.tier == Tier.working
    assert excinfo.value.stored_bucket == "dev-fin-observatory-datasets-working"
    assert excinfo.value.stored_object_name == "sha256:conflict"
    assert excinfo.value.stored_size_bytes == 9
    assert excinfo.value.incoming_bucket == "a-different-bucket"
    assert excinfo.value.incoming_object_name == "sha256:conflict"
    assert excinfo.value.incoming_size_bytes == 9

    # A conflict leaves the stored location unchanged.
    location = inventory.get_location("sha256:conflict", tier=Tier.working)
    assert location == ArtifactLocation(
        bucket="dev-fin-observatory-datasets-working", object_name="sha256:conflict"
    )


def test_record_raises_on_a_conflicting_size_for_the_same_digest(dataset_store):
    inventory = LiveArtifactInventory(store=dataset_store)
    inventory.record(
        digest="sha256:sizeconflict",
        tier=Tier.working,
        bucket="dev-fin-observatory-datasets-working",
        object_name="sha256:sizeconflict",
        size_bytes=9,
        uploaded=True,
    )

    with pytest.raises(ArtifactRecordConflictError) as excinfo:
        inventory.record(
            digest="sha256:sizeconflict",
            tier=Tier.pinned,
            bucket="dev-fin-observatory-datasets-pinned",
            object_name="sha256:sizeconflict",
            size_bytes=99,
            uploaded=True,
        )
    assert excinfo.value.stored_size_bytes == 9
    assert excinfo.value.incoming_size_bytes == 99


def test_store_put_raises_artifact_inventory_error_when_the_bytes_landed_but_recording_failed(
    dataset_store, monkeypatch
):
    client = FakeGCSClient()
    inventory = LiveArtifactInventory(store=dataset_store)
    store = LiveArtifactStore(buckets=_BUCKETS, inventory=inventory, client=client)

    def _boom(self, *args, **kwargs):
        raise RuntimeError("driver said something secret")

    monkeypatch.setattr(type(dataset_store), "transaction", _boom)

    with pytest.raises(ArtifactInventoryError) as excinfo:
        store.put(b"lands but never gets a row", tier=Tier.working)
    assert "driver said something secret" not in str(excinfo.value)

    # Retry after inventory recovery records the landed bytes without re-uploading.
    monkeypatch.undo()
    bucket_name = _BUCKETS[Tier.working]
    upload_count_before_retry = len(client.buckets[bucket_name].upload_log)

    result = store.put(b"lands but never gets a row", tier=Tier.working)

    assert result.deduplicated is True
    assert len(client.buckets[bucket_name].upload_log) == upload_count_before_retry
    with dataset_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT * FROM datasets.artifact_inventory WHERE artifact_digest = %s",
            (result.digest,),
        )
    assert row is not None


def test_store_put_records_inventory_only_after_the_write_succeeds(dataset_store):
    client = FakeGCSClient()
    inventory = LiveArtifactInventory(store=dataset_store)
    store = LiveArtifactStore(buckets=_BUCKETS, inventory=inventory, client=client)

    result = store.put(b"lands fine", tier=Tier.working)

    with dataset_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT * FROM datasets.artifact_inventory WHERE artifact_digest = %s",
            (result.digest,),
        )
    assert row is not None
    assert row["size_bytes"] == len(b"lands fine")


# `stored_at` refreshes only after a new upload to the working tier.

_FIRST_LANDING = datetime(2026, 1, 1, tzinfo=UTC)
_LATER_LANDING = datetime(2026, 2, 1, tzinfo=UTC)


def _stored_at(dataset_store, *, digest: str, tier: Tier) -> datetime:
    with dataset_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT stored_at FROM datasets.artifact_location "
            "WHERE artifact_digest = %s AND tier = %s",
            (digest, tier.value),
        )
    assert row is not None
    # SQLite stores this `timestamptz` as text; decode it as Postgres does.
    stored_at = row["stored_at"]
    return stored_at if isinstance(stored_at, datetime) else datetime.fromisoformat(stored_at)


@pytest.mark.parametrize(
    ("digest", "uploaded", "expected_stored_at"),
    [
        pytest.param("sha256:refresh", True, _LATER_LANDING, id="uploaded_true_refreshes"),
        pytest.param("sha256:dedup", False, _FIRST_LANDING, id="uploaded_false_dedup_preserves"),
    ],
)
def test_repeat_record_updates_stored_at_only_for_uploaded_artifacts(
    dataset_store, digest: str, uploaded: bool, expected_stored_at: datetime
):
    LiveArtifactInventory(store=dataset_store, now=lambda: _FIRST_LANDING).record(
        digest=digest,
        tier=Tier.working,
        bucket="dev-fin-observatory-datasets-working",
        object_name=digest,
        size_bytes=5,
        uploaded=True,
    )

    LiveArtifactInventory(store=dataset_store, now=lambda: _LATER_LANDING).record(
        digest=digest,
        tier=Tier.working,
        bucket="dev-fin-observatory-datasets-working",
        object_name=digest,
        size_bytes=5,
        uploaded=uploaded,
    )

    assert _stored_at(dataset_store, digest=digest, tier=Tier.working) == expected_stored_at


def test_refreshing_the_working_tier_row_leaves_the_pinned_tier_rows_stored_at_alone(dataset_store):
    LiveArtifactInventory(store=dataset_store, now=lambda: _FIRST_LANDING).record(
        digest="sha256:bothtiers",
        tier=Tier.working,
        bucket="dev-fin-observatory-datasets-working",
        object_name="sha256:bothtiers",
        size_bytes=5,
        uploaded=True,
    )
    LiveArtifactInventory(store=dataset_store, now=lambda: _FIRST_LANDING).record(
        digest="sha256:bothtiers",
        tier=Tier.pinned,
        bucket="dev-fin-observatory-datasets-pinned",
        object_name="sha256:bothtiers",
        size_bytes=5,
        uploaded=True,
    )

    LiveArtifactInventory(store=dataset_store, now=lambda: _LATER_LANDING).record(
        digest="sha256:bothtiers",
        tier=Tier.working,
        bucket="dev-fin-observatory-datasets-working",
        object_name="sha256:bothtiers",
        size_bytes=5,
        uploaded=True,
    )

    assert _stored_at(dataset_store, digest="sha256:bothtiers", tier=Tier.working) == _LATER_LANDING
    assert _stored_at(dataset_store, digest="sha256:bothtiers", tier=Tier.pinned) == _FIRST_LANDING


def test_store_puts_own_dedup_hit_does_not_extend_the_working_tier_clock(dataset_store):
    # End-to-end through LiveArtifactStore: a repeat mint of identical bytes
    # dedups on put()'s own if-generation-match precondition, computing
    # uploaded=False -- the working tier's own retention clock must not move.
    client = FakeGCSClient()
    first_inventory = LiveArtifactInventory(store=dataset_store, now=lambda: _FIRST_LANDING)
    store_first = LiveArtifactStore(buckets=_BUCKETS, inventory=first_inventory, client=client)
    first = store_first.put(b"repeat mint bytes", tier=Tier.working)
    assert first.deduplicated is False

    later_inventory = LiveArtifactInventory(store=dataset_store, now=lambda: _LATER_LANDING)
    store_later = LiveArtifactStore(buckets=_BUCKETS, inventory=later_inventory, client=client)
    second = store_later.put(b"repeat mint bytes", tier=Tier.working)

    assert second.deduplicated is True
    assert _stored_at(dataset_store, digest=first.digest, tier=Tier.working) == _FIRST_LANDING


def test_a_conflicting_bucket_never_moves_stored_at_even_with_uploaded_true(dataset_store):
    # The refused call below still carries uploaded=True -- proving the
    # conflict check wins over the refresh, not merely that a conflicting
    # call happens to also pass uploaded=False in the test above.
    LiveArtifactInventory(store=dataset_store, now=lambda: _FIRST_LANDING).record(
        digest="sha256:conflictclock",
        tier=Tier.working,
        bucket="dev-fin-observatory-datasets-working",
        object_name="sha256:conflictclock",
        size_bytes=9,
        uploaded=True,
    )

    with pytest.raises(ArtifactRecordConflictError):
        LiveArtifactInventory(store=dataset_store, now=lambda: _LATER_LANDING).record(
            digest="sha256:conflictclock",
            tier=Tier.working,
            bucket="a-different-bucket",
            object_name="sha256:conflictclock",
            size_bytes=9,
            uploaded=True,
        )

    assert (
        _stored_at(dataset_store, digest="sha256:conflictclock", tier=Tier.working)
        == _FIRST_LANDING
    )
