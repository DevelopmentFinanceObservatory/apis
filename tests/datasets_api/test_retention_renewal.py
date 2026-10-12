"""Tests that working-tier expiry follows `artifact_location.stored_at` and refreshes only
when bytes land again, through the snapshot and pin doors with real readers and writers, and
that `LiveArtifactStore` renews an expired working location under a new object name.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from artifact_store import LiveArtifactStore, Tier
from artifact_store.errors import ArtifactIntegrityError
from artifact_store.inventory import (
    ArtifactLocation,
    InventoryBatch,
    LiveArtifactInventory,
    write_entry,
)
from artifact_store.store import SPILL_THRESHOLD_BYTES
from dataset_store.snapshot import WORKING_TIER_RETENTION, LiveSnapshotReader
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.pin import build_pin_door
from datasets_api.doors.snapshot import build_snapshot_door
from datasets_api.reference_store import LiveArtifactReferenceWriter
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord, VersionRecord
from tests.artifact_store.conftest import FakeGCSClient

from ._artifact_doubles import FakeArtifactStore
from ._deps_doubles import fake_deps_cell
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
            facts=_version().facts,
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
        deps=fake_deps_cell(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=LiveSnapshotReader(store=version_store),
            artifact_store=artifact_store,
            now=now,
        )
    )
    return make_client_for_door(door)


def _pin_client(version_store: FakeVersionRecordDatasetStore, *, artifact_store, now):
    door = build_pin_door(
        deps=fake_deps_cell(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=LiveSnapshotReader(store=version_store),
            artifact_store=artifact_store,
            reference_writer=LiveArtifactReferenceWriter(store=version_store),
            now=now,
        ),
        settings=DatasetsBuildSettings.from_env(),
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


_BUCKETS = {Tier.working: "renewal-working", Tier.pinned: "renewal-pinned"}
_SMALL = b"renewal-fixture-bytes"
# Crosses the spill threshold, so the write stages and then copies to its object name.
_STAGED = b"s" * SPILL_THRESHOLD_BYTES + b"past the spill threshold"


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _live_store(
    version_store: FakeVersionRecordDatasetStore,
    client: FakeGCSClient,
    *,
    at: datetime,
    inventory: LiveArtifactInventory | None = None,
) -> LiveArtifactStore:
    """A live store over fake GCS whose inventory and retention clock both read `at`."""
    return LiveArtifactStore(
        buckets=_BUCKETS,
        inventory=inventory or LiveArtifactInventory(store=version_store, now=lambda: at),
        client=client,
        now=lambda: at,
    )


def _location_row(
    version_store: FakeVersionRecordDatasetStore, digest: str, *, tier: Tier = Tier.working
) -> dict[str, object]:
    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT object_name, stored_at, renewal FROM datasets.artifact_location "
            "WHERE artifact_digest = %s AND tier = %s",
            (digest, tier.value),
        )
    assert row is not None
    return dict(row)


@pytest.mark.parametrize("data", [_SMALL, _STAGED], ids=["buffered", "staged"])
def test_a_put_past_retention_writes_a_new_object_name_and_moves_the_location(version_store, data):
    """I7. The lifecycle rule has not yet deleted the expired object, so its name is taken."""
    client = FakeGCSClient()
    digest = _digest(data)
    _live_store(version_store, client, at=_MINTED_AT).put(data, tier=Tier.working)

    renewed = _live_store(version_store, client, at=_PAST_THE_WINDOW).put(data, tier=Tier.working)

    assert renewed.object_name == f"{digest}#r1"
    assert renewed.deduplicated is False
    bucket = client.buckets[_BUCKETS[Tier.working]]
    assert bucket.objects[f"{digest}#r1"] == data
    assert _location_row(version_store, digest) == {
        "object_name": f"{digest}#r1",
        "stored_at": _PAST_THE_WINDOW,
        "renewal": 1,
    }


def test_a_renewed_location_serves_its_digest_after_the_old_object_is_deleted(version_store):
    """I3. `get` reads the object the location names and verifies the bytes hash to the digest."""
    client = FakeGCSClient()
    digest = _digest(_SMALL)
    _live_store(version_store, client, at=_MINTED_AT).put(_SMALL, tier=Tier.working)
    store = _live_store(version_store, client, at=_PAST_THE_WINDOW)
    store.put(_SMALL, tier=Tier.working)
    # The lifecycle rule deletes the expired object at its own age.
    del client.buckets[_BUCKETS[Tier.working]].objects[digest]

    data = store.get(artifact_digest=digest, tier=Tier.working)

    assert _digest(data) == digest


def test_get_refuses_a_renewed_object_whose_bytes_do_not_hash_to_the_digest(version_store):
    """I3. A location's object name carries no authority over the digest."""
    client = FakeGCSClient()
    digest = _digest(_SMALL)
    _live_store(version_store, client, at=_MINTED_AT).put(_SMALL, tier=Tier.working)
    store = _live_store(version_store, client, at=_PAST_THE_WINDOW)
    store.put(_SMALL, tier=Tier.working)
    client.buckets[_BUCKETS[Tier.working]].objects[f"{digest}#r1"] = b"other bytes"

    with pytest.raises(ArtifactIntegrityError) as excinfo:
        store.get(artifact_digest=digest, tier=Tier.working)

    assert excinfo.value.computed_digest == _digest(b"other bytes")


def test_a_taken_renewal_name_past_retention_is_skipped_and_never_recorded(version_store):
    """I7. An object already under the next renewal name whose creation time is past retention,
    for example one a renewal uploaded before its location move failed, is never recorded as
    servable. The store writes the next name instead."""
    client = FakeGCSClient()
    digest = _digest(_SMALL)
    _live_store(version_store, client, at=_MINTED_AT).put(_SMALL, tier=Tier.working)
    bucket = client.buckets[_BUCKETS[Tier.working]]
    bucket.objects[f"{digest}#r1"] = _SMALL
    bucket.time_created[f"{digest}#r1"] = _MINTED_AT

    renewed = _live_store(version_store, client, at=_PAST_THE_WINDOW).put(_SMALL, tier=Tier.working)

    assert renewed.object_name == f"{digest}#r2"
    assert bucket.upload_log[-1] == f"{digest}#r2"
    assert _location_row(version_store, digest) == {
        "object_name": f"{digest}#r2",
        "stored_at": _PAST_THE_WINDOW,
        "renewal": 2,
    }


def test_a_dedupe_inside_the_window_keeps_the_location_and_its_clock(version_store):
    client = FakeGCSClient()
    digest = _digest(_SMALL)
    _live_store(version_store, client, at=_MINTED_AT).put(_SMALL, tier=Tier.working)
    inside = _MINTED_AT + WORKING_TIER_RETENTION - timedelta(days=1)

    again = _live_store(version_store, client, at=inside).put(_SMALL, tier=Tier.working)

    assert again.deduplicated is True
    assert again.object_name == digest
    assert _location_row(version_store, digest) == {
        "object_name": digest,
        "stored_at": _MINTED_AT,
        "renewal": 0,
    }


def test_a_pinned_location_never_renews(version_store):
    client = FakeGCSClient()
    digest = _digest(_SMALL)
    _live_store(version_store, client, at=_MINTED_AT).put(_SMALL, tier=Tier.pinned)

    again = _live_store(version_store, client, at=_PAST_THE_WINDOW).put(_SMALL, tier=Tier.pinned)

    assert again.deduplicated is True
    assert again.object_name == digest
    assert _location_row(version_store, digest, tier=Tier.pinned)["renewal"] == 0


class _LifecycleDeletesAfterRead(LiveArtifactInventory):
    """Deletes the located object right after the store reads its location, as the bucket's
    lifecycle rule can between that read and the upload."""

    def __init__(self, *, client: FakeGCSClient, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = client

    def get_location(self, digest: str, *, tier: Tier) -> ArtifactLocation | None:
        location = super().get_location(digest, tier=tier)
        if location is not None:
            self._client.bucket(location.bucket).objects.pop(location.object_name, None)
        return location


def test_an_object_deleted_between_the_inventory_read_and_the_upload_lands_afresh(version_store):
    client = FakeGCSClient()
    digest = _digest(_SMALL)
    _live_store(version_store, client, at=_MINTED_AT).put(_SMALL, tier=Tier.working)
    inventory = _LifecycleDeletesAfterRead(
        client=client, store=version_store, now=lambda: _PAST_THE_WINDOW
    )
    store = _live_store(version_store, client, at=_PAST_THE_WINDOW, inventory=inventory)

    landed = store.put(_SMALL, tier=Tier.working)

    assert landed.deduplicated is False
    assert client.buckets[_BUCKETS[Tier.working]].objects[landed.object_name] == _SMALL
    row = _location_row(version_store, digest)
    assert row["object_name"] == landed.object_name
    assert row["stored_at"] == _PAST_THE_WINDOW


def _plant_orphan(client: FakeGCSClient, data: bytes, *, created: datetime) -> None:
    """An object under the bare digest with no location row, as a failed inventory write leaves."""
    bucket = client.bucket(_BUCKETS[Tier.working])
    bucket.objects[_digest(data)] = data
    bucket.time_created[_digest(data)] = created


def test_a_dedupe_against_an_unrecorded_object_inside_the_window_dates_it_from_its_creation(
    version_store,
):
    """I7. The orphan's lifecycle clock started at its creation, so the location's does too."""
    client = FakeGCSClient()
    digest = _digest(_SMALL)
    _plant_orphan(client, _SMALL, created=_MINTED_AT)
    inside = _MINTED_AT + WORKING_TIER_RETENTION - timedelta(days=1)

    again = _live_store(version_store, client, at=inside).put(_SMALL, tier=Tier.working)

    assert again.deduplicated is True
    assert again.object_name == digest
    assert _location_row(version_store, digest) == {
        "object_name": digest,
        "stored_at": _MINTED_AT,
        "renewal": 0,
    }


@pytest.mark.parametrize("data", [_SMALL, _STAGED], ids=["buffered", "staged"])
def test_an_unrecorded_object_past_retention_renews_under_a_new_object_name(version_store, data):
    """I7. The lifecycle rule may delete the orphan at any moment, so the bytes land again."""
    client = FakeGCSClient()
    digest = _digest(data)
    _plant_orphan(client, data, created=_MINTED_AT)

    renewed = _live_store(version_store, client, at=_PAST_THE_WINDOW).put(data, tier=Tier.working)

    assert renewed.object_name == f"{digest}#r1"
    assert renewed.deduplicated is False
    assert client.buckets[_BUCKETS[Tier.working]].objects[f"{digest}#r1"] == data
    assert _location_row(version_store, digest) == {
        "object_name": f"{digest}#r1",
        "stored_at": _PAST_THE_WINDOW,
        "renewal": 1,
    }


def test_a_fresh_upload_under_the_bare_digest_moves_a_renewed_location_back_to_it(
    version_store,
):
    """The location was renewed to `#r1` and the lifecycle rule has since deleted the bare
    object, so a new write lands under the bare digest with no location read. Those bytes are
    the newest copy: the location moves to them with a fresh `stored_at` and keeps its renewal
    count, so a later renewal takes the next free name."""
    client = FakeGCSClient()
    digest = _digest(_SMALL)
    _live_store(version_store, client, at=_MINTED_AT).put(_SMALL, tier=Tier.working)
    renewed = _live_store(version_store, client, at=_PAST_THE_WINDOW).put(_SMALL, tier=Tier.working)
    assert renewed.object_name == f"{digest}#r1"
    client.bucket(_BUCKETS[Tier.working]).objects.pop(digest)
    later = _PAST_THE_WINDOW + timedelta(days=2)

    stored = _live_store(version_store, client, at=later).put(_SMALL, tier=Tier.working)

    assert (stored.object_name, stored.deduplicated) == (digest, False)
    assert _location_row(version_store, digest) == {
        "object_name": digest,
        "stored_at": later,
        "renewal": 1,
    }


@pytest.mark.parametrize("bare_first", [True, False], ids=["bare-writer-first", "dedupe-first"])
def test_two_deferred_writers_of_one_digest_against_a_renewed_location_both_commit(
    version_store, bare_first
):
    """The location names `#r1` and the bare object is gone. Writer X lands the bytes under the
    bare digest; writer Y, before X commits, finds the bare name taken and deduplicates against
    `#r1`. Both rows commit in either order and the location ends on X's fresh bare object."""
    client = FakeGCSClient()
    digest = _digest(_SMALL)
    _live_store(version_store, client, at=_MINTED_AT).put(_SMALL, tier=Tier.working)
    _live_store(version_store, client, at=_PAST_THE_WINDOW).put(_SMALL, tier=Tier.working)
    client.bucket(_BUCKETS[Tier.working]).objects.pop(digest)
    later = _PAST_THE_WINDOW + timedelta(days=2)
    store = _live_store(version_store, client, at=later)
    batch_x, batch_y = InventoryBatch(), InventoryBatch()

    stored_x = store.deferred(batch_x).put(_SMALL, tier=Tier.working)
    stored_y = store.deferred(batch_y).put(_SMALL, tier=Tier.working)

    assert (stored_x.object_name, stored_x.deduplicated) == (digest, False)
    assert (stored_y.object_name, stored_y.deduplicated) == (f"{digest}#r1", True)
    for batch in (batch_x, batch_y) if bare_first else (batch_y, batch_x):
        with version_store.transaction() as tx:
            for entry in batch.entries:
                write_entry(tx, entry, now=later)
    assert _location_row(version_store, digest) == {
        "object_name": digest,
        "stored_at": later,
        "renewal": 1,
    }
