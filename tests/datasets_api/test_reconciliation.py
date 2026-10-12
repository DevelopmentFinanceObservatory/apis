"""Reconciliation reports missing objects, storage orphans, and unreferenced pinned digests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from artifact_store import Tier
from dataset_store.snapshot import WORKING_TIER_RETENTION
from datasets_api.reconciliation import sweep, sweep_tier

from ._artifact_doubles import FakeArtifactStore
from ._version_record_store import FakeVersionRecordDatasetStore

_NOW = datetime(2026, 6, 1, tzinfo=UTC)


def _seed_location(
    store, *, digest: str, tier: str, bucket: str = "fake-bucket", stored_at: datetime = _NOW
) -> None:
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_location "
            "(artifact_digest, tier, bucket, object_name, stored_at) VALUES (%s, %s, %s, %s, %s)",
            (digest, tier, bucket, digest, stored_at),
        )


def _seed_reference(
    store,
    *,
    digest: str,
    principal: str = "first-party-key",
    subject_id: str = "spec_a",
    content_version: str = "cv1",
    removed_at: datetime | None = None,
) -> None:
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_reference "
            "(principal, artifact_digest, subject_id, content_version, reference_kind, "
            "created_at, removed_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (principal, digest, subject_id, content_version, "pin", _NOW, removed_at),
        )


def test_sweep_tier_finds_a_missing_object_and_an_orphaned_one():
    store = FakeVersionRecordDatasetStore()
    artifact_store = FakeArtifactStore()
    try:
        in_sync = artifact_store.put(b"in-sync-bytes", tier=Tier.working)
        _seed_location(store, digest=in_sync.digest, tier="working")

        # Simulates deletion outside the API's retention path.
        _seed_location(store, digest="sha256:deleted-outside-the-api", tier="working")

        # Simulates bytes uploaded before the inventory write failed.
        orphan = artifact_store.put(b"orphan-bytes", tier=Tier.working)
        with store.transaction() as tx:
            tx.execute(
                "DELETE FROM datasets.artifact_location WHERE artifact_digest = %s",
                (orphan.digest,),
            )

        report = sweep_tier(store=store, artifact_store=artifact_store, tier=Tier.working, now=_NOW)

        assert report.tier == Tier.working
        assert report.missing_objects == {"sha256:deleted-outside-the-api"}
        assert report.orphaned_objects == {orphan.digest}
    finally:
        store.close()


def test_sweep_tier_is_clean_when_bucket_and_inventory_agree():
    store = FakeVersionRecordDatasetStore()
    artifact_store = FakeArtifactStore()
    try:
        stored = artifact_store.put(b"tidy-bytes", tier=Tier.pinned)
        _seed_location(store, digest=stored.digest, tier="pinned")

        report = sweep_tier(store=store, artifact_store=artifact_store, tier=Tier.pinned, now=_NOW)

        assert report.missing_objects == frozenset()
        assert report.orphaned_objects == frozenset()
    finally:
        store.close()


def test_sweep_covers_both_tiers_in_one_call():
    store = FakeVersionRecordDatasetStore()
    artifact_store = FakeArtifactStore()
    try:
        working = artifact_store.put(b"working-bytes", tier=Tier.working)
        _seed_location(store, digest=working.digest, tier="working")
        pinned = artifact_store.put(b"pinned-bytes", tier=Tier.pinned)
        _seed_location(store, digest=pinned.digest, tier="pinned")

        reports = sweep(store=store, artifact_store=artifact_store, now=_NOW)

        assert {r.tier for r in reports} == {Tier.working, Tier.pinned}
        assert all(
            r.missing_objects == frozenset() and r.orphaned_objects == frozenset() for r in reports
        )
    finally:
        store.close()


def test_a_pinned_object_with_no_reference_row_at_all_is_flagged_unreferenced():
    store = FakeVersionRecordDatasetStore()
    artifact_store = FakeArtifactStore()
    try:
        orphaned_pin = artifact_store.put(b"pinned-with-no-reference", tier=Tier.pinned)
        _seed_location(store, digest=orphaned_pin.digest, tier="pinned")

        report = sweep_tier(store=store, artifact_store=artifact_store, tier=Tier.pinned, now=_NOW)

        assert report.unreferenced_pinned_digests == {orphaned_pin.digest}
    finally:
        store.close()


def test_a_pinned_object_with_only_a_soft_deleted_reference_is_not_flagged():
    """Any reference row, including a soft-deleted one, keeps a pinned object out of the
    unreferenced set.
    """
    store = FakeVersionRecordDatasetStore()
    artifact_store = FakeArtifactStore()
    try:
        unpinned = artifact_store.put(b"pinned-then-unpinned", tier=Tier.pinned)
        _seed_location(store, digest=unpinned.digest, tier="pinned")
        _seed_reference(store, digest=unpinned.digest, removed_at=_NOW)

        report = sweep_tier(store=store, artifact_store=artifact_store, tier=Tier.pinned, now=_NOW)

        assert report.unreferenced_pinned_digests == frozenset()
    finally:
        store.close()


def test_a_pinned_object_with_a_live_reference_is_not_flagged():
    store = FakeVersionRecordDatasetStore()
    artifact_store = FakeArtifactStore()
    try:
        referenced = artifact_store.put(b"pinned-and-referenced", tier=Tier.pinned)
        _seed_location(store, digest=referenced.digest, tier="pinned")
        _seed_reference(store, digest=referenced.digest, removed_at=None)

        report = sweep_tier(store=store, artifact_store=artifact_store, tier=Tier.pinned, now=_NOW)

        assert report.unreferenced_pinned_digests == frozenset()
    finally:
        store.close()


def test_the_unreferenced_leg_is_always_empty_for_the_working_tier():
    store = FakeVersionRecordDatasetStore()
    artifact_store = FakeArtifactStore()
    try:
        working = artifact_store.put(b"working-tier-object", tier=Tier.working)
        _seed_location(store, digest=working.digest, tier="working")
        # Working-tier objects are not checked for references.

        report = sweep_tier(store=store, artifact_store=artifact_store, tier=Tier.working, now=_NOW)

        assert report.unreferenced_pinned_digests == frozenset()
    finally:
        store.close()


def test_an_expired_working_row_counts_as_expired_and_not_missing():
    """The lifecycle rule deletes an expired working row's object by design, so only an unexpired
    row whose object is absent is missing. An expired row's object still in the bucket is not
    orphaned."""
    store = FakeVersionRecordDatasetStore()
    artifact_store = FakeArtifactStore()
    expired_at = _NOW - WORKING_TIER_RETENTION - timedelta(seconds=1)
    try:
        _seed_location(store, digest="sha256:expired-gone", tier="working", stored_at=expired_at)
        _seed_location(
            store,
            digest="sha256:fresh-gone",
            tier="working",
            stored_at=_NOW - WORKING_TIER_RETENTION,
        )
        lingering = artifact_store.put(b"expired-but-not-yet-deleted", tier=Tier.working)
        _seed_location(store, digest=lingering.digest, tier="working", stored_at=expired_at)
        _seed_location(store, digest="sha256:old-pin-gone", tier="pinned", stored_at=expired_at)

        working, pinned = sweep(store=store, artifact_store=artifact_store, now=_NOW)

        assert working.missing_objects == {"sha256:fresh-gone"}
        assert working.expired_working_locations == {"sha256:expired-gone", lingering.digest}
        assert working.orphaned_objects == frozenset()
        assert pinned.missing_objects == {"sha256:old-pin-gone"}
        assert pinned.expired_working_locations == frozenset()
    finally:
        store.close()
