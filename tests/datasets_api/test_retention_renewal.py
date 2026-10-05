"""Tests that working-tier expiry follows `artifact_location.stored_at` and refreshes only
when bytes land again, through the snapshot and pin doors with real readers and writers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from artifact_store import Tier
from artifact_store.inventory import LiveArtifactInventory
from dataset_store.snapshot import WORKING_TIER_RETENTION, LiveSnapshotReader
from datasets_api.doors.pin import build_pin_door
from datasets_api.doors.snapshot import build_snapshot_door
from datasets_api.reference_store import LiveArtifactReferenceWriter
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord, VersionRecord

from ._artifact_doubles import FakeArtifactStore
from ._snapshot_doubles import FakeSpecReader, _version, make_client_for_door
from ._version_record_store import FakeVersionRecordDatasetStore

_PRINCIPAL = "first-party-key"
_MINTED_AT = datetime(2026, 1, 1, tzinfo=UTC)
_PAST_THE_WINDOW = _MINTED_AT + WORKING_TIER_RETENTION + timedelta(days=1)


@pytest.fixture
def version_store():
    store = FakeVersionRecordDatasetStore()
    try:
        yield store
    finally:
        store.close()


def _seed(
    version_store: FakeVersionRecordDatasetStore, *, artifact_store: FakeArtifactStore
) -> str:
    """Stores working-tier bytes and a version, both dated `_MINTED_AT`."""
    stored = artifact_store.put(b"renewal-fixture-bytes", tier=Tier.working)
    LiveArtifactInventory(store=version_store, now=lambda: _MINTED_AT).record(
        digest=stored.digest,
        tier=Tier.working,
        bucket=stored.bucket,
        object_name=stored.object_name,
        size_bytes=stored.size_bytes,
        uploaded=True,
    )
    LiveVersionRecordWriter(store=version_store, now=lambda: _MINTED_AT).record_version(
        VersionRecord(
            subject_id="spec_a",
            subject_kind="spec",
            content_version="cv1",
            schema_version=1,
            immutable_body=_version().immutable_body,
            artifacts=(
                ArtifactRecord(
                    artifact_digest=stored.digest,
                    rendering={"format": "csv"},
                    media_type="text/csv",
                ),
            ),
        )
    )
    return stored.digest


def _snapshot_client(version_store: FakeVersionRecordDatasetStore, *, artifact_store, now):
    door = build_snapshot_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        artifact_store=artifact_store,
        now=now,
    )
    return make_client_for_door(door)


def _pin_client(version_store: FakeVersionRecordDatasetStore, *, artifact_store, now):
    door = build_pin_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        artifact_store=artifact_store,
        reference_writer=LiveArtifactReferenceWriter(store=version_store),
        now=now,
        allowed_principals=frozenset({_PRINCIPAL}),
    )
    return make_client_for_door(door)


def test_renewing_with_uploaded_true_lets_an_aged_out_artifact_serve_and_pin_again(version_store):
    artifact_store = FakeArtifactStore()
    digest = _seed(version_store, artifact_store=artifact_store)

    # Sanity: past the window from the original mint, the artifact is gone.
    expired_client = _snapshot_client(
        version_store, artifact_store=artifact_store, now=lambda: _PAST_THE_WINDOW
    )
    assert expired_client.get("/api/datasets/spec_a/cv1").status_code == 410

    # Bytes genuinely land again (e.g. a fresh render_and_store call for the
    # same digest) -- uploaded=True refreshes stored_at to this instant.
    LiveArtifactInventory(store=version_store, now=lambda: _PAST_THE_WINDOW).record(
        digest=digest,
        tier=Tier.working,
        bucket="fake-bucket",
        object_name=digest,
        size_bytes=len(b"renewal-fixture-bytes"),
        uploaded=True,
    )
    read_shortly_after_renewal = _PAST_THE_WINDOW + timedelta(days=1)

    snapshot_resp = _snapshot_client(
        version_store, artifact_store=artifact_store, now=lambda: read_shortly_after_renewal
    ).get("/api/datasets/spec_a/cv1")
    assert snapshot_resp.status_code == 200, snapshot_resp.text
    assert snapshot_resp.json()["artifacts"][0]["artifact_digest"] == digest

    pin_resp = _pin_client(
        version_store, artifact_store=artifact_store, now=lambda: read_shortly_after_renewal
    ).post(
        "/api/datasets/spec_a/cv1/pin",
        json={"schema_version": 1, "artifact_digest": digest},
        headers={"x-dfo-principal": _PRINCIPAL},
    )
    assert pin_resp.status_code == 200, pin_resp.text


def test_renewing_with_uploaded_false_leaves_an_aged_out_artifact_expired(version_store):
    artifact_store = FakeArtifactStore()
    digest = _seed(version_store, artifact_store=artifact_store)

    # A dedup touch (e.g. a mint that lands on the same digest but whose
    # bytes were already there) -- uploaded=False leaves stored_at at
    # _MINTED_AT, so the artifact stays exactly as expired as before.
    LiveArtifactInventory(store=version_store, now=lambda: _PAST_THE_WINDOW).record(
        digest=digest,
        tier=Tier.working,
        bucket="fake-bucket",
        object_name=digest,
        size_bytes=len(b"renewal-fixture-bytes"),
        uploaded=False,
    )
    read_shortly_after_the_dedup_touch = _PAST_THE_WINDOW + timedelta(days=1)

    snapshot_resp = _snapshot_client(
        version_store, artifact_store=artifact_store, now=lambda: read_shortly_after_the_dedup_touch
    ).get("/api/datasets/spec_a/cv1")
    assert snapshot_resp.status_code == 410, snapshot_resp.text
    assert snapshot_resp.json()["code"] == "artifact_expired"

    pin_resp = _pin_client(
        version_store, artifact_store=artifact_store, now=lambda: read_shortly_after_the_dedup_touch
    ).post(
        "/api/datasets/spec_a/cv1/pin",
        json={"schema_version": 1, "artifact_digest": digest},
        headers={"x-dfo-principal": _PRINCIPAL},
    )
    assert pin_resp.status_code == 410, pin_resp.text
    assert pin_resp.json()["code"] == "artifact_expired"


def test_promoting_to_pinned_leaves_the_wire_created_at_identical(version_store):
    """Wire `created_at` comes from the digest's first inventory sighting.
    Pinned promotion adds a separate location row and preserves that timestamp.

    The fake store leaves inventory updates to the test, which records the pinned row.
    """
    artifact_store = FakeArtifactStore()
    digest = _seed(version_store, artifact_store=artifact_store)
    just_inside_the_window = _MINTED_AT + WORKING_TIER_RETENTION - timedelta(days=1)

    before = _snapshot_client(
        version_store, artifact_store=artifact_store, now=lambda: just_inside_the_window
    ).get("/api/datasets/spec_a/cv1")
    assert before.status_code == 200, before.text
    created_at_before = before.json()["artifacts"][0]["created_at"]

    artifact_store.put(b"renewal-fixture-bytes", tier=Tier.pinned)
    LiveArtifactInventory(store=version_store, now=lambda: just_inside_the_window).record(
        digest=digest,
        tier=Tier.pinned,
        bucket="fake-bucket",
        object_name=digest,
        size_bytes=len(b"renewal-fixture-bytes"),
        uploaded=True,
    )

    after = _snapshot_client(
        version_store, artifact_store=artifact_store, now=lambda: just_inside_the_window
    ).get("/api/datasets/spec_a/cv1")
    assert after.status_code == 200, after.text
    assert after.json()["artifacts"][0]["tier"] == "pinned"

    assert after.json()["artifacts"][0]["created_at"] == created_at_before


def test_promoting_to_pinned_leaves_the_working_tier_rows_stored_at_alone(version_store):
    # Pinned promotion writes a separate row without changing working-tier `stored_at`.
    artifact_store = FakeArtifactStore()
    digest = _seed(version_store, artifact_store=artifact_store)
    just_inside_the_window = _MINTED_AT + WORKING_TIER_RETENTION - timedelta(days=1)

    pin_resp = _pin_client(
        version_store, artifact_store=artifact_store, now=lambda: just_inside_the_window
    ).post(
        "/api/datasets/spec_a/cv1/pin",
        json={"schema_version": 1, "artifact_digest": digest},
        headers={"x-dfo-principal": _PRINCIPAL},
    )
    assert pin_resp.status_code == 200, pin_resp.text

    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT stored_at FROM datasets.artifact_location "
            "WHERE artifact_digest = %s AND tier = 'working'",
            (digest,),
        )
    assert row is not None
    assert row["stored_at"] == _MINTED_AT
