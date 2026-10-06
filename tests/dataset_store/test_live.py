"""Live smoke tests against a real reachable Postgres, either CI's
ephemeral service container or a developer's local/proxied instance.
Skipped by default. Set DFO_DATASET_STORE_LIVE_DSN
(postgresql://user:pass@host:port/db) to run these.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import cast
from urllib.parse import urlsplit
from uuid import uuid4

import pg8000.exceptions
import pytest

from artifact_store.inventory import LiveArtifactInventory
from artifact_store.store import Tier
from dataset_store import ConnectionPool, LiveDatasetStore, connect
from dataset_store.migrate import MIGRATIONS_DIR, apply_pending
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.codes_snapshot import ArtifactNotFound
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.definition import DefinitionDeps
from datasets_api.doors.pin import pin_artifact
from datasets_api.operation_store import (
    MAX_REDISPATCHES,
    REDISPATCH_GRACE_SECONDS,
    LiveOperationStore,
    OperationBacklogFull,
    lease_stale_after_seconds,
    never_claimed_stale_after_seconds,
)
from datasets_api.operations import worker as worker_module
from datasets_api.operations.dispatch import dispatch_if_due
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.operations.worker import OperationSuperseded, run_operation
from datasets_api.reference_store import PIN_REFERENCE_KIND, LiveArtifactReferenceWriter
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord, MemberSignal, SubjectHead, VersionRecord
from series_coverage.index import Estimate
from tests.datasets_api._legacy_conversion_trace import (
    LEGACY_CONVERSION,
    LEGACY_DEFLATED_CONVERSION,
    MIGRATED_CONVERSION,
    MIGRATED_DEFLATED_CONVERSION,
)
from tests.datasets_api._snapshot_doubles import FakeSpecReader

_DSN = os.environ.get("DFO_DATASET_STORE_LIVE_DSN")

# The safe-default settings every `LiveOperationStore` below resolves for
# itself when built with no explicit `settings=`.
_DEFAULT_SETTINGS = DatasetsBuildSettings()


def _lease_stale_after_seconds() -> float:
    return lease_stale_after_seconds(
        max_retry_duration_seconds=_DEFAULT_SETTINGS.operation_max_retry_duration_seconds
    )


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _DSN, reason="set DFO_DATASET_STORE_LIVE_DSN to run live dataset_store tests"
    ),
]


def _pool(*, size: int = 2) -> ConnectionPool:
    parts = urlsplit(_DSN)
    return ConnectionPool(
        connect=lambda: connect(
            connector=None,
            instance_connection_name=None,
            host=parts.hostname,
            port=parts.port or 5432,
            database=parts.path.lstrip("/"),
            user=parts.username,
            password=parts.password,
        ),
        size=size,
    )


@pytest.fixture
def live_store():
    pool = _pool()
    try:
        yield LiveDatasetStore(connections=pool)
    finally:
        pool.close_all()


def test_transaction_commits_and_reads_back(live_store):
    with live_store.transaction() as tx:
        tx.execute("CREATE TABLE IF NOT EXISTS dataset_store_live_probe (id INTEGER PRIMARY KEY)")
        tx.execute(
            "INSERT INTO dataset_store_live_probe (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
            (1,),
        )
    with live_store.transaction() as tx:
        row = tx.fetch_one("SELECT id FROM dataset_store_live_probe WHERE id = %s", (1,))
    assert row == {"id": 1}


def test_rollback_on_exception(live_store):
    # Table created and committed separately. Postgres DDL is transactional,
    # so creating it inside the rolled-back transaction would erase it too.
    with live_store.transaction() as tx:
        tx.execute(
            "CREATE TABLE IF NOT EXISTS dataset_store_live_rollback_probe (id INTEGER PRIMARY KEY)"
        )

    with pytest.raises(ValueError), live_store.transaction() as tx:
        tx.execute("INSERT INTO dataset_store_live_rollback_probe (id) VALUES (2)")
        raise ValueError("deliberate")

    with live_store.transaction() as tx:
        row = tx.fetch_one("SELECT id FROM dataset_store_live_rollback_probe WHERE id = %s", (2,))
    assert row is None


def test_ddl_is_transactional_unlike_the_sqlite_fake(live_store):
    # Postgres DDL is transactional (unlike the sqlite3 fake in conftest.py).
    # A table created then rolled back doesn't exist afterward.
    with pytest.raises(ValueError), live_store.transaction() as tx:
        tx.execute("CREATE TABLE dataset_store_live_ddl_probe (id INTEGER PRIMARY KEY)")
        raise ValueError("deliberate")
    with live_store.transaction() as tx:
        row = tx.fetch_one("SELECT to_regclass('dataset_store_live_ddl_probe') AS reg")
    assert row == {"reg": None}


def test_artifact_inventory_tables_exist_after_migration(live_store):
    # migrations.yml applies every pending migration (including
    # 0003_artifact_inventory.sql) before running this file, so this is a
    # live-gated DDL assertion rather than something test_migrate.py (a
    # synthetic-only file) could check.
    with live_store.transaction() as tx:
        inventory_reg = tx.fetch_one("SELECT to_regclass('datasets.artifact_inventory') AS reg")
        location_reg = tx.fetch_one("SELECT to_regclass('datasets.artifact_location') AS reg")
    assert inventory_reg == {"reg": "datasets.artifact_inventory"}
    assert location_reg == {"reg": "datasets.artifact_location"}


def test_artifact_inventory_and_location_round_trip(live_store):
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes) "
            "VALUES (%s, %s) ON CONFLICT (artifact_digest) DO NOTHING",
            ("sha256:test-live-probe", 5),
        )
        tx.execute(
            "INSERT INTO datasets.artifact_location "
            "(artifact_digest, tier, bucket, object_name) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (artifact_digest, tier) DO NOTHING",
            ("sha256:test-live-probe", "working", "test-bucket", "sha256:test-live-probe"),
        )
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT bucket, object_name FROM datasets.artifact_location "
            "WHERE artifact_digest = %s AND tier = %s",
            ("sha256:test-live-probe", "working"),
        )
    assert row == {"bucket": "test-bucket", "object_name": "sha256:test-live-probe"}


def test_artifact_location_upsert_refreshes_stored_at_only_on_a_matching_upload(live_store):
    # LiveArtifactInventory.record()'s stored_at refresh against real
    # Postgres. A dedup hit (uploaded=False) leaves the existing row's
    # stored_at untouched, and a fresh landing (uploaded=True) with a
    # matching bucket/object_name advances it.
    digest = "sha256:test-live-upsert-refresh"
    bucket = "test-live-upsert-bucket"
    first_landing = datetime(2020, 1, 1, tzinfo=UTC)
    dedup_hit = datetime(2020, 2, 1, tzinfo=UTC)
    fresh_landing = datetime(2020, 3, 1, tzinfo=UTC)

    def _stored_at() -> datetime:
        with live_store.transaction() as tx:
            row = tx.fetch_one(
                "SELECT stored_at FROM datasets.artifact_location "
                "WHERE artifact_digest = %s AND tier = %s",
                (digest, "working"),
            )
        assert row is not None
        return row["stored_at"]

    LiveArtifactInventory(store=live_store, now=lambda: first_landing).record(
        digest=digest,
        tier=Tier.working,
        bucket=bucket,
        object_name=digest,
        size_bytes=5,
        uploaded=True,
    )
    assert _stored_at() == first_landing

    LiveArtifactInventory(store=live_store, now=lambda: dedup_hit).record(
        digest=digest,
        tier=Tier.working,
        bucket=bucket,
        object_name=digest,
        size_bytes=5,
        uploaded=False,
    )
    assert _stored_at() == first_landing

    LiveArtifactInventory(store=live_store, now=lambda: fresh_landing).record(
        digest=digest,
        tier=Tier.working,
        bucket=bucket,
        object_name=digest,
        size_bytes=5,
        uploaded=True,
    )
    assert _stored_at() == fresh_landing


def test_artifact_location_tier_check_constraint_rejects_an_invalid_tier(live_store):
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes) "
            "VALUES (%s, %s) ON CONFLICT (artifact_digest) DO NOTHING",
            ("sha256:test-live-tier-check", 1),
        )
    with pytest.raises(pg8000.exceptions.Error), live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_location "
            "(artifact_digest, tier, bucket, object_name) VALUES (%s, %s, %s, %s)",
            ("sha256:test-live-tier-check", "bogus-tier", "b", "sha256:test-live-tier-check"),
        )


def test_artifact_location_fk_rejects_an_unknown_digest(live_store):
    with pytest.raises(pg8000.exceptions.Error), live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_location "
            "(artifact_digest, tier, bucket, object_name) VALUES (%s, %s, %s, %s)",
            (
                "sha256:does-not-exist-in-inventory",
                "working",
                "b",
                "sha256:does-not-exist-in-inventory",
            ),
        )


def test_artifact_location_primary_key_rejects_a_duplicate_digest_tier_pair(live_store):
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes) "
            "VALUES (%s, %s) ON CONFLICT (artifact_digest) DO NOTHING",
            ("sha256:test-live-pk-check", 1),
        )
        tx.execute(
            "INSERT INTO datasets.artifact_location "
            "(artifact_digest, tier, bucket, object_name) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (artifact_digest, tier) DO NOTHING",
            ("sha256:test-live-pk-check", "working", "b", "sha256:test-live-pk-check"),
        )
    with pytest.raises(pg8000.exceptions.Error), live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_location "
            "(artifact_digest, tier, bucket, object_name) VALUES (%s, %s, %s, %s)",
            (
                "sha256:test-live-pk-check",
                "working",
                "different-bucket",
                "sha256:test-live-pk-check",
            ),
        )


def test_percent_escaping_only_matters_outside_a_quoted_literal():
    # With no params, pg8000 sends SQL verbatim (no %-processing). With
    # params, it scans for a bare % outside quoted literals only.
    pool = _pool(size=1)
    store = LiveDatasetStore(connections=pool)
    try:
        with store.transaction() as tx:
            row = tx.fetch_one("SELECT 'a%' AS v")
        assert row == {"v": "a%"}

        with store.transaction() as tx:
            row = tx.fetch_one("SELECT 'a%' AS v WHERE %s = %s", ("x", "x"))
        assert row == {"v": "a%"}

        with pytest.raises(pg8000.exceptions.Error), store.transaction() as tx:
            tx.fetch_one("SELECT %s AS v WHERE 1 = 1 % 2", ("x",))

        with store.transaction() as tx:
            row = tx.fetch_one("SELECT %s AS v WHERE 10 %% 3 = %s", ("x", 1))
        assert row == {"v": "x"}
    finally:
        pool.close_all()


def test_version_record_tables_exist_after_migration(live_store):
    # migrations.yml applies every pending migration (including
    # 0004_version_records.sql) before running this file, mirroring
    # test_artifact_inventory_tables_exist_after_migration's own approach.
    with live_store.transaction() as tx:
        record_reg = tx.fetch_one("SELECT to_regclass('datasets.version_record') AS reg")
        member_reg = tx.fetch_one("SELECT to_regclass('datasets.version_member') AS reg")
        artifact_reg = tx.fetch_one("SELECT to_regclass('datasets.version_artifact') AS reg")
    assert record_reg == {"reg": "datasets.version_record"}
    # 0021_drop_collections.sql drops the collection member table.
    assert member_reg == {"reg": None}
    assert artifact_reg == {"reg": "datasets.version_artifact"}


def test_version_record_write_once_and_a_repeat_is_a_no_op(live_store):
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at) VALUES (%s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (subject_id, content_version) DO NOTHING",
            ("spec:test-live-version", "spec", "cv1", 1, {"revision": "first"}),
        )
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at) VALUES (%s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (subject_id, content_version) DO NOTHING",
            ("spec:test-live-version", "spec", "cv1", 1, {"revision": "second"}),
        )
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT immutable_body FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            ("spec:test-live-version", "cv1"),
        )
    assert row == {"immutable_body": {"revision": "first"}}


def test_version_artifact_links_are_enumerable(live_store):
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes) "
            "VALUES (%s, %s) ON CONFLICT (artifact_digest) DO NOTHING",
            ("sha256:test-live-version-artifact", 7),
        )
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at) VALUES (%s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (subject_id, content_version) DO NOTHING",
            ("spec:test-live-artifact-link", "spec", "cv1", 1, {}),
        )
        tx.execute(
            "INSERT INTO datasets.version_artifact "
            "(subject_id, content_version, artifact_digest, rendering_key, rendering, "
            "media_type) VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (subject_id, content_version, rendering_key) DO NOTHING",
            (
                "spec:test-live-artifact-link",
                "cv1",
                "sha256:test-live-version-artifact",
                "rk-test-live-artifact-link",
                {"format": "csv"},
                "text/csv",
            ),
        )
    with live_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT artifact_digest FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s",
            ("spec:test-live-artifact-link", "cv1"),
        )
    assert [row["artifact_digest"] for row in rows] == ["sha256:test-live-version-artifact"]


@pytest.mark.parametrize(
    ("tier", "age", "linked"),
    [
        pytest.param("working", timedelta(days=30), "old", id="servable-working"),
        pytest.param("working", timedelta(days=31), "new", id="expired-working"),
        pytest.param("pinned", timedelta(days=400), "old", id="pinned"),
    ],
)
def test_version_artifact_link_is_replaced_only_once_its_digest_is_not_servable(
    live_store, tier, age, linked
):
    """`_insert_artifact`'s conflict update keeps a link whose digest `select_servable_locations`
    would serve at the writer's `now`, and replaces one it would not."""
    now = datetime.now(UTC)
    suffix = uuid4().hex
    subject_id = f"spec:test-live-relink-{suffix}"
    digests = {
        "old": f"sha256:test-live-relink-old-{suffix}",
        "new": f"sha256:test-live-relink-new-{suffix}",
    }
    with live_store.transaction() as tx:
        for digest in digests.values():
            tx.execute(
                "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes) "
                "VALUES (%s, %s)",
                (digest, 7),
            )
        tx.execute(
            "INSERT INTO datasets.artifact_location "
            "(artifact_digest, tier, bucket, object_name, stored_at) VALUES (%s, %s, %s, %s, %s)",
            (digests["old"], tier, "bucket", digests["old"], now - age),
        )
    writer = LiveVersionRecordWriter(store=live_store, now=lambda: now)
    rendering = {"format": "csv", "bundle": True}
    writer.record_version(
        VersionRecord(
            subject_id=subject_id,
            subject_kind="spec",
            content_version="cv1",
            schema_version=1,
            immutable_body={},
            artifacts=(
                ArtifactRecord(
                    artifact_digest=digests["old"],
                    rendering=rendering,
                    media_type="application/zip",
                ),
            ),
        )
    )

    writer.record_artifact(
        subject_id=subject_id,
        content_version="cv1",
        artifact=ArtifactRecord(
            artifact_digest=digests["new"], rendering=rendering, media_type="application/zip"
        ),
    )

    with live_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT artifact_digest FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s",
            (subject_id, "cv1"),
        )
    assert [row["artifact_digest"] for row in rows] == [digests[linked]]


_RELINK_RENDERING = {"format": "csv", "bundle": True}


def _seed_linked_artifact(
    live_store, *, subject_id: str, digests: tuple[str, str], location: tuple[str, datetime]
) -> None:
    """Record `subject_id` at cv1 linking `digests[0]`, stored at `location` (tier, stored_at).
    `digests[1]` is in the inventory, ready for a re-render to link."""
    with live_store.transaction() as tx:
        for digest in digests:
            tx.execute(
                "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes) "
                "VALUES (%s, %s)",
                (digest, 7),
            )
        tx.execute(
            "INSERT INTO datasets.artifact_location "
            "(artifact_digest, tier, bucket, object_name, stored_at) VALUES (%s, %s, %s, %s, %s)",
            (digests[0], location[0], "bucket", digests[0], location[1]),
        )
    LiveVersionRecordWriter(store=live_store, now=lambda: location[1]).record_version(
        VersionRecord(
            subject_id=subject_id,
            subject_kind="spec",
            content_version="cv1",
            schema_version=1,
            immutable_body={},
            artifacts=(
                ArtifactRecord(
                    artifact_digest=digests[0],
                    rendering=_RELINK_RENDERING,
                    media_type="application/zip",
                ),
            ),
        )
    )


def _linked_digests(live_store, subject_id: str) -> list[str]:
    with live_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT artifact_digest FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s",
            (subject_id, "cv1"),
        )
    return [cast(str, row["artifact_digest"]) for row in rows]


def test_a_pin_refuses_when_a_rerender_replaces_its_link_during_the_copy(live_store):
    """The working copy is servable when the pin reads it and expired when a re-render links
    new bytes, which happens before the copy records the pinned location. The pin refuses,
    because snapshot reads cannot list a pin the version does not link."""
    pinned_at = datetime.now(UTC)
    suffix = uuid4().hex
    subject_id = f"spec:test-live-pin-relink-{suffix}"
    old = f"sha256:test-live-pin-relink-old-{suffix}"
    new = f"sha256:test-live-pin-relink-new-{suffix}"
    _seed_linked_artifact(
        live_store,
        subject_id=subject_id,
        digests=(old, new),
        location=("working", pinned_at - timedelta(days=29)),
    )
    inventory = LiveArtifactInventory(store=live_store)
    rerender = LiveVersionRecordWriter(store=live_store, now=lambda: pinned_at + timedelta(days=2))

    class _RerenderDuringCopy:
        def copy(self, digest, *, from_tier, to_tier):
            rerender.record_artifact(
                subject_id=subject_id,
                content_version="cv1",
                artifact=ArtifactRecord(
                    artifact_digest=new, rendering=_RELINK_RENDERING, media_type="application/zip"
                ),
            )
            inventory.record(
                digest=digest,
                tier=to_tier,
                bucket="pinned-bucket",
                object_name=digest,
                size_bytes=7,
                uploaded=True,
            )

    with pytest.raises(ArtifactNotFound):
        pin_artifact(
            id=subject_id,
            content_version="cv1",
            artifact_digest=old,
            principal="first-party",
            spec_reader=FakeSpecReader(spec_ids=[subject_id]),
            snapshot_reader=LiveSnapshotReader(store=live_store),
            artifact_store=_RerenderDuringCopy(),
            reference_writer=LiveArtifactReferenceWriter(store=live_store),
            daily_max=1_000_000,
            now=lambda: pinned_at,
        )

    assert _linked_digests(live_store, subject_id) == [new]
    with live_store.transaction() as tx:
        references = tx.fetch_all(
            "SELECT 1 FROM datasets.artifact_reference WHERE subject_id = %s", (subject_id,)
        )
    assert references == []


def _an_advisory_lock_is_blocked(live_store) -> bool:
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT count(*) AS n FROM pg_stat_activity "
            "WHERE wait_event_type = 'Lock' AND wait_event = 'advisory' "
            "AND query LIKE 'SELECT pg_advisory_xact_lock%'"
        )
    return row is not None and row["n"] > 0


def test_a_pin_blocked_behind_an_uncommitted_rerender_refuses_once_it_commits(live_store):
    """The race the link lock closes. A re-render's upsert sees an expired working copy and no
    pinned location, replaces the link, and holds its transaction open. The pin then commits
    its pinned location and starts its reference write, which still reads the old link unless
    it waits for the re-render's lock. Once the re-render commits, the pin must refuse rather
    than succeed for a digest the version no longer links."""
    now = datetime.now(UTC)
    suffix = uuid4().hex
    subject_id = f"spec:test-live-pin-race-{suffix}"
    old = f"sha256:test-live-pin-race-old-{suffix}"
    new = f"sha256:test-live-pin-race-new-{suffix}"
    _seed_linked_artifact(
        live_store,
        subject_id=subject_id,
        digests=(old, new),
        location=("working", now - timedelta(days=29)),
    )
    rerender_pool = _pool(size=1)
    pin_pool = _pool(size=1)
    holder = _HoldBeforeCommitStore(
        LiveDatasetStore(connections=rerender_pool),
        hold_when=lambda sql: sql.upper().startswith("INSERT INTO DATASETS.VERSION_ARTIFACT"),
    )
    try:
        rerender = _Call(
            lambda: LiveVersionRecordWriter(
                store=holder, now=lambda: now + timedelta(days=2)
            ).record_artifact(
                subject_id=subject_id,
                content_version="cv1",
                artifact=ArtifactRecord(
                    artifact_digest=new, rendering=_RELINK_RENDERING, media_type="application/zip"
                ),
            )
        )
        assert holder.held.wait(timeout=5), "the re-render never reached its commit"
        LiveArtifactInventory(store=live_store).record(
            digest=old,
            tier=Tier.pinned,
            bucket="pinned-bucket",
            object_name=old,
            size_bytes=7,
            uploaded=True,
        )
        pin = _Call(
            lambda: LiveArtifactReferenceWriter(
                store=LiveDatasetStore(connections=pin_pool)
            ).record_reference_unless_revoked(
                principal="first-party",
                artifact_digest=old,
                subject_id=subject_id,
                content_version="cv1",
                reference_kind=PIN_REFERENCE_KIND,
                now=now,
            )
        )
        blocked = False
        deadline = time.monotonic() + 5
        while not blocked and not pin.done() and time.monotonic() < deadline:
            blocked = _an_advisory_lock_is_blocked(live_store)
            time.sleep(0.02)
        holder.release.set()
        rerender.join()

        with pytest.raises(ArtifactNotFound):
            pin.join()
    finally:
        holder.release.set()
        rerender_pool.close_all()
        pin_pool.close_all()

    assert blocked, "the reference write never waited for the re-render's link lock"
    assert _linked_digests(live_store, subject_id) == [new]
    with live_store.transaction() as tx:
        references = tx.fetch_all(
            "SELECT 1 FROM datasets.artifact_reference WHERE subject_id = %s", (subject_id,)
        )
    assert references == []


def test_version_artifact_fk_rejects_an_unknown_digest(live_store):
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at) VALUES (%s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (subject_id, content_version) DO NOTHING",
            ("spec:test-live-artifact-fk", "spec", "cv1", 1, {}),
        )
    with pytest.raises(pg8000.exceptions.Error), live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_artifact "
            "(subject_id, content_version, artifact_digest, rendering_key, rendering, "
            "media_type) VALUES (%s, %s, %s, %s, %s, %s)",
            (
                "spec:test-live-artifact-fk",
                "cv1",
                "sha256:does-not-exist-in-inventory",
                "rk-test-live-artifact-fk",
                {"format": "csv"},
                "text/csv",
            ),
        )


def test_version_revocation_and_artifact_reference_tables_exist_after_migration(live_store):
    with live_store.transaction() as tx:
        revocation_reg = tx.fetch_one("SELECT to_regclass('datasets.version_revocation') AS reg")
        reference_reg = tx.fetch_one("SELECT to_regclass('datasets.artifact_reference') AS reg")
    assert revocation_reg == {"reg": "datasets.version_revocation"}
    assert reference_reg == {"reg": "datasets.artifact_reference"}


def test_version_revocation_round_trip(live_store):
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at) VALUES (%s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (subject_id, content_version) DO NOTHING",
            ("spec:test-live-revocation", "spec", "cv1", 1, {}),
        )
        tx.execute(
            "INSERT INTO datasets.version_revocation "
            "(subject_id, content_version, reason, revoked_at) VALUES (%s, %s, %s, now())",
            ("spec:test-live-revocation", "cv1", "takedown"),
        )
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT reason FROM datasets.version_revocation "
            "WHERE subject_id = %s AND content_version = %s",
            ("spec:test-live-revocation", "cv1"),
        )
    assert row == {"reason": "takedown"}


def test_version_revocation_fk_rejects_an_unknown_version(live_store):
    with pytest.raises(pg8000.exceptions.Error), live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_revocation "
            "(subject_id, content_version, revoked_at) VALUES (%s, %s, now())",
            ("spec:does-not-exist", "cv1"),
        )


def test_subject_head_table_exists_after_migration(live_store):
    # migrations.yml applies every pending migration (including
    # 0009_subject_head.sql) before running this file, mirroring
    # test_version_record_tables_exist_after_migration's own approach.
    with live_store.transaction() as tx:
        head_reg = tx.fetch_one("SELECT to_regclass('datasets.subject_head') AS reg")
    assert head_reg == {"reg": "datasets.subject_head"}


def test_subject_head_fk_rejects_an_unknown_version(live_store):
    with pytest.raises(pg8000.exceptions.Error), live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.subject_head "
            "(subject_id, content_version, confirmed_at, materialised_at, signal) "
            "VALUES (%s, %s, now(), now(), %s)",
            ("spec:does-not-exist", "cv1", {"members": []}),
        )


def _insert_version_record(live_store, *, subject_id: str, content_version: str) -> None:
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at) VALUES (%s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (subject_id, content_version) DO NOTHING",
            (subject_id, "spec", content_version, 1, {}),
        )


def test_live_head_store_a_write_whose_expected_version_has_moved_on_is_a_no_op(live_store):
    """After `cv2` advances the head from `cv0`, a writer expecting `cv0` cannot write `cv1`."""
    subject_id = "spec:test-live-head-forward"
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv0")
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv1")
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv2")

    store = LiveHeadStore(store=live_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    store.advance_head(
        subject_id=subject_id,
        content_version="cv0",
        signal=[MemberSignal("facet0", 0, None, None)],
        expected_content_version=None,
    )

    won = store.advance_head(
        subject_id=subject_id,
        content_version="cv2",
        signal=[MemberSignal("facet2", 20, "2020", "2021")],
        expected_content_version="cv0",
    )
    lost = store.advance_head(
        subject_id=subject_id,
        content_version="cv1",
        signal=[MemberSignal("facet1", 10, "2010", "2011")],
        expected_content_version="cv0",
    )

    assert won is not None
    assert lost is None
    head = store.read_head(subject_id)
    assert head is not None
    assert head.content_version == "cv2"
    assert head.signal == (MemberSignal("facet2", 20, "2020", "2021"),)


def test_live_head_store_read_head_returns_none_for_an_unknown_subject(live_store):
    store = LiveHeadStore(store=live_store)
    assert store.read_head("spec:does-not-exist-at-all") is None


def test_live_head_store_two_racing_writers_agree_on_exactly_one_winner(live_store):
    """Two writers share `cv-seed`; exactly one advances the head.

    A barrier releases both threads together so their writes overlap. The stored head matches the
    winner's version and signal.
    """
    subject_id = "spec:test-live-head-race"
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv-seed")
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv-a")
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv-b")
    LiveHeadStore(store=live_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC)).advance_head(
        subject_id=subject_id,
        content_version="cv-seed",
        signal=[MemberSignal("facet-seed", 0, None, None)],
        expected_content_version=None,
    )

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    results: dict[str, SubjectHead | None] = {}

    def _write(*, content_version: str) -> None:
        try:
            barrier.wait(timeout=5)
            results[content_version] = LiveHeadStore(
                store=live_store, now=lambda: datetime(2026, 6, 1, tzinfo=UTC)
            ).advance_head(
                subject_id=subject_id,
                content_version=content_version,
                signal=[MemberSignal(f"facet-{content_version}", 1, None, None)],
                expected_content_version="cv-seed",
            )
        # surfaced on the main thread below
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=_write, kwargs={"content_version": "cv-a"}),
        threading.Thread(target=_write, kwargs={"content_version": "cv-b"}),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors, errors
    winners = [content_version for content_version, result in results.items() if result is not None]
    losers = [content_version for content_version, result in results.items() if result is None]
    assert len(winners) == 1, results
    assert len(losers) == 1, results
    head = LiveHeadStore(store=live_store).read_head(subject_id)
    assert head is not None
    assert head.content_version == winners[0]
    # The None-dates round trip (`MemberSignal.earliest_date`/`latest_date`
    # both `None`) through the real jsonb column here. This proves
    # `decode_signal` accepts the shape a member with no observations yet
    # writes. `test_head_store.py`'s
    # `test_decode_signal_accepts_none_dates_and_none_dispatch_facet_id`
    # separately pins the same shape against the fake SQLite store.
    assert head.signal == (MemberSignal(f"facet-{winners[0]}", 1, None, None),)


class _PausingTransaction:
    """Pauses a transaction before its advisory lock, after the unlocked existence read."""

    def __init__(self, inner, *, paused_at_lock: threading.Event, resume: threading.Event) -> None:
        self._inner = inner
        self._paused_at_lock = paused_at_lock
        self._resume = resume

    def execute(self, sql: str, params=()) -> None:
        if sql.strip().upper().startswith("SELECT PG_ADVISORY_XACT_LOCK"):
            self._paused_at_lock.set()
            if not self._resume.wait(timeout=10):
                raise TimeoutError("test never signalled resume")
        self._inner.execute(sql, params)

    def fetch_one(self, sql: str, params=()):
        return self._inner.fetch_one(sql, params)

    def fetch_all(self, sql: str, params=()):
        return self._inner.fetch_all(sql, params)


class _PausingDatasetStore:
    """Wraps each real transaction in `_PausingTransaction`."""

    def __init__(self, inner, *, paused_at_lock: threading.Event, resume: threading.Event) -> None:
        self._inner = inner
        self._paused_at_lock = paused_at_lock
        self._resume = resume

    @contextmanager
    def transaction(self):
        with self._inner.transaction() as tx:
            yield _PausingTransaction(tx, paused_at_lock=self._paused_at_lock, resume=self._resume)


def test_operation_store_create_or_get_reset_survives_a_concurrent_resubmit_race(live_store):
    """A second connection resets a failed operation while the first waits for its backlog lock.

    The paused call must read the committed `pending` row and avoid a false full-backlog result.
    """
    now = datetime(2026, 1, 1, tzinfo=UTC)
    run_id = uuid4().hex
    identity = {"input": {"spec_id": f"spec-test-live-operation-race-{run_id}"}}
    options = {"format": "parquet"}
    stamp = f"cov-test-live-operation-race-{run_id}"
    estimate = Estimate(rows=1, requests=1)

    with live_store.transaction() as tx:
        baseline_row = tx.fetch_one(
            "SELECT count(*) AS n FROM datasets.operation WHERE state IN ('pending', 'running')"
        )
    baseline = baseline_row["n"] if baseline_row is not None else 0
    backlog_max = baseline + 5

    setup_store = LiveOperationStore(store=live_store, now=lambda: now, backlog_max=backlog_max)
    # 4 filler pending rows plus the target, 5 total once the target is
    # reset, lands the in-flight count exactly at backlog_max, the
    # margin the reviewed race needs to turn into a spurious raise.
    for i in range(4):
        setup_store.create_or_get(
            request_identity={
                "input": {"spec_id": f"spec-test-live-operation-race-{run_id}-filler-{i}"}
            },
            options=options,
            coverage_stamp=stamp,
            origin_catalogue_build_id="fake-catalogue",
            estimate=estimate,
        )
    target = setup_store.create_or_get(
        request_identity=identity,
        options=options,
        coverage_stamp=stamp,
        origin_catalogue_build_id="fake-catalogue",
        estimate=estimate,
    )
    setup_store.claim_lease(target.id, submission_count=0, lease_seconds=60)
    setup_store.fail(target.id, attempt=1, refusal_code="internal_error", retryable=True)

    paused_at_lock = threading.Event()
    resume = threading.Event()
    loser_pool = _pool(size=1)
    try:
        loser_store = LiveOperationStore(
            store=_PausingDatasetStore(
                LiveDatasetStore(connections=loser_pool),
                paused_at_lock=paused_at_lock,
                resume=resume,
            ),
            now=lambda: now,
            backlog_max=backlog_max,
        )
        errors: list[BaseException] = []
        loser_op: dict[str, object] = {}

        def _loser_call() -> None:
            try:
                loser_op["result"] = loser_store.create_or_get(
                    request_identity=identity,
                    options=options,
                    coverage_stamp=stamp,
                    origin_catalogue_build_id="fake-catalogue",
                    estimate=estimate,
                )
            # surfaced on the main thread below
            except BaseException as exc:
                errors.append(exc)

        loser_thread = threading.Thread(target=_loser_call)
        loser_thread.start()
        assert paused_at_lock.wait(timeout=5), "loser never reached its own advisory-lock statement"

        winner_pool = _pool(size=1)
        try:
            winner_store = LiveOperationStore(
                store=LiveDatasetStore(connections=winner_pool),
                now=lambda: now,
                backlog_max=backlog_max,
            )
            winner_op = winner_store.create_or_get(
                request_identity=identity,
                options=options,
                coverage_stamp=stamp,
                origin_catalogue_build_id="fake-catalogue",
                estimate=estimate,
            )
        finally:
            winner_pool.close_all()

        assert winner_op.id == target.id
        assert winner_op.state == "pending"
        assert winner_op.submission_count == 1

        resume.set()
        loser_thread.join(timeout=10)

        assert not any(isinstance(exc, OperationBacklogFull) for exc in errors), errors
        assert not errors, errors
        result = loser_op["result"]
        assert result.id == target.id
        assert result.state == "pending"
        # The winner's own reset, seen and returned unchanged, not a
        # second increment stacked on top of it.
        assert result.submission_count == 1
    finally:
        loser_pool.close_all()


def test_operation_store_dead_running_rows_do_not_block_a_new_submit(live_store):
    """After 20 leases expire, `create_or_get` sweeps their rows before counting the backlog."""
    run_id = uuid4().hex
    options = {"format": "parquet"}
    stamp = f"cov-test-live-operation-dead-running-{run_id}"
    estimate = Estimate(rows=1, requests=1)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    with live_store.transaction() as tx:
        baseline_row = tx.fetch_one(
            "SELECT count(*) AS n FROM datasets.operation WHERE state IN ('pending', 'running')"
        )
    baseline = baseline_row["n"] if baseline_row is not None else 0
    backlog_max = baseline + 20

    setup_store = LiveOperationStore(store=live_store, now=lambda: now, backlog_max=backlog_max)
    dead_ids = []
    for i in range(20):
        op = setup_store.create_or_get(
            request_identity={"input": {"spec_id": f"spec-test-live-operation-dead-{run_id}-{i}"}},
            options=options,
            coverage_stamp=stamp,
            origin_catalogue_build_id="fake-catalogue",
            estimate=estimate,
        )
        claimed = setup_store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        assert claimed is not None
        dead_ids.append(op.id)

    # Right at the backlog max, with every one of the 20 leases still
    # live, this refuses.
    with pytest.raises(OperationBacklogFull):
        setup_store.create_or_get(
            request_identity={
                "input": {"spec_id": f"spec-test-live-operation-dead-{run_id}-too-soon"}
            },
            options=options,
            coverage_stamp=stamp,
            origin_catalogue_build_id="fake-catalogue",
            estimate=estimate,
        )

    # Advance the clock past this store's own lease_stale_after_seconds,
    # measured from each lease's own expiry (60s after `now` above):
    # Cloud Tasks has certainly exhausted every retry by then, with
    # nothing since to renew any of the 20 leases.
    stale_now = now + timedelta(seconds=60 + _lease_stale_after_seconds() + 1)
    stale_store = LiveOperationStore(
        store=live_store, now=lambda: stale_now, backlog_max=backlog_max
    )
    new_op = stale_store.create_or_get(
        request_identity={"input": {"spec_id": f"spec-test-live-operation-dead-{run_id}-new"}},
        options=options,
        coverage_stamp=stamp,
        origin_catalogue_build_id="fake-catalogue",
        estimate=estimate,
    )
    assert new_op.state == "pending"

    for dead_id in dead_ids:
        swept = stale_store.get(dead_id)
        assert swept is not None
        assert swept.state == "failed"
        assert swept.refusal_code == "internal_error"
        assert swept.retryable is True


def test_operation_insert_predating_submitted_at_still_succeeds(live_store):
    """An insert without `submitted_at` receives the database's `now()` default.

    An older worker image can run this insert during a rollout or rollback.
    """
    run_id = uuid4().hex
    op_id = f"op_test-live-legacy-insert-{run_id}"
    synthetic_created_at = datetime(2026, 1, 1, tzinfo=UTC)

    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.operation "
            "(id, request_identity, options, coverage_stamp, state, estimate_rows, "
            "estimate_requests, rows_written, attempt_count, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, 'pending', %s, %s, 0, 0, %s, %s) "
            "ON CONFLICT (id) DO NOTHING",
            (
                op_id,
                {"input": {"spec_id": f"spec-test-live-legacy-insert-{run_id}"}},
                {"format": "csv"},
                f"cov-test-live-legacy-insert-{run_id}",
                1,
                1,
                synthetic_created_at,
                synthetic_created_at,
            ),
        )

    with live_store.transaction() as tx:
        row = tx.fetch_one("SELECT submitted_at FROM datasets.operation WHERE id = %s", (op_id,))
    assert row is not None
    # DEFAULT now() stamps the real insert time, not the synthetic
    # created_at this test wrote, and not NULL (which SET NOT NULL would
    # have refused outright).
    assert row["submitted_at"] is not None
    assert row["submitted_at"] > synthetic_created_at


class _HoldingTransaction:
    """Forwards to a real transaction, noting whether any statement matched
    `_HoldBeforeCommitStore.hold_when`."""

    def __init__(self, inner, *, hold_when) -> None:
        self._inner = inner
        self._hold_when = hold_when
        self.matched = False

    def _note(self, sql: str) -> None:
        if self._hold_when(sql):
            self.matched = True

    def execute(self, sql: str, params=()) -> None:
        self._note(sql)
        self._inner.execute(sql, params)

    def fetch_one(self, sql: str, params=()):
        self._note(sql)
        return self._inner.fetch_one(sql, params)

    def fetch_all(self, sql: str, params=()):
        self._note(sql)
        return self._inner.fetch_all(sql, params)


class _HoldBeforeCommitStore:
    """Holds matching real transactions before commit so a second connection waits on their row locks."""

    def __init__(self, inner, *, hold_when=lambda sql: sql.startswith("UPDATE")) -> None:
        self._inner = inner
        self._hold_when = hold_when
        self.held = threading.Event()
        self.release = threading.Event()

    @contextmanager
    def transaction(self):
        with self._inner.transaction() as tx:
            holding = _HoldingTransaction(tx, hold_when=self._hold_when)
            yield holding
            if holding.matched:
                self.held.set()
                if not self.release.wait(timeout=10):
                    raise TimeoutError("test never released the held transaction")


class _Call:
    """Runs a callable on a thread and re-raises its exception after join."""

    def __init__(self, fn) -> None:
        self.result: object = None
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(fn,))
        self._thread.start()

    def _run(self, fn) -> None:
        try:
            self.result = fn()
        # surfaced on the main thread by join()
        except BaseException as exc:
            self.error = exc

    def done(self) -> bool:
        return not self._thread.is_alive()

    def join(self):
        self._thread.join(timeout=10)
        assert not self._thread.is_alive(), "call never finished"
        if self.error is not None:
            raise self.error
        return self.result


def _wait_until_an_operation_update_is_blocked(live_store) -> None:
    """Wait until another session blocks on an operation-row update."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with live_store.transaction() as tx:
            row = tx.fetch_one(
                "SELECT count(*) AS n FROM pg_stat_activity "
                "WHERE wait_event_type = 'Lock' AND query LIKE 'UPDATE datasets.operation%'"
            )
        if row is not None and row["n"] > 0:
            return
        time.sleep(0.02)
    raise AssertionError("no UPDATE datasets.operation ever blocked on the held row")


def _new_operation(store: LiveOperationStore, run_id: str, label: str):
    return store.create_or_get(
        request_identity={"input": {"spec_id": f"spec-test-live-fence-{run_id}-{label}"}},
        options={"format": "parquet"},
        coverage_stamp=f"cov-test-live-fence-{run_id}",
        origin_catalogue_build_id="fake-catalogue",
        estimate=Estimate(rows=1, requests=1),
    )


# Wide enough that rows other live tests leave behind never fill it.
_RACE_BACKLOG_MAX = 100_000


def test_operation_store_stale_attempt_complete_is_refused_after_a_reclaim(live_store):
    """Attempt 1's lease lapsed; attempt 2's reclaim holds the row lock,
    uncommitted, when attempt 1's worker tries to complete. The complete
    blocks on the lock, re-evaluates its `attempt_count` guard against the
    committed reclaim, and matches nothing: attempt 2 keeps the row.
    """
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    setup = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(setup, run_id, "reclaim")
    first = setup.claim_lease(op.id, submission_count=0, lease_seconds=30)
    assert first is not None

    t1 = t0 + timedelta(seconds=31)
    reclaim_pool = _pool(size=1)
    stale_pool = _pool(size=1)
    try:
        holder = _HoldBeforeCommitStore(LiveDatasetStore(connections=reclaim_pool))
        reclaimer = LiveOperationStore(store=holder, now=lambda: t1)
        stale_worker = LiveOperationStore(
            store=LiveDatasetStore(connections=stale_pool), now=lambda: t1
        )

        reclaim = _Call(
            lambda: reclaimer.claim_lease(op.id, submission_count=0, lease_seconds=1_700)
        )
        assert holder.held.wait(timeout=5), "the reclaim never reached its commit"
        complete = _Call(
            lambda: stale_worker.complete(
                op.id,
                attempt=first.attempt_count,
                subject_id="spec-test-live-fence",
                content_version="cv-never-written",
            )
        )
        _wait_until_an_operation_update_is_blocked(live_store)
        holder.release.set()

        reclaimed = reclaim.join()
        assert complete.join() is None
    finally:
        reclaim_pool.close_all()
        stale_pool.close_all()

    assert reclaimed is not None
    assert reclaimed.attempt_count == first.attempt_count + 1
    current = setup.get(op.id)
    assert current is not None
    assert current.state == "running"
    assert current.attempt_count == first.attempt_count + 1
    assert current.content_version is None


def test_operation_store_heartbeat_wins_over_a_concurrent_expiry(live_store):
    """A poll's expiry and a heartbeat race on a row whose lease looks dead
    to both. The heartbeat's renewal holds the row lock; the expiry blocks,
    re-evaluates the dead-row predicate against the renewed lease, and
    leaves the operation running. A second row, identical but never
    renewed, is the control: the same expiry fails it.
    """
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    setup = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(setup, run_id, "heartbeat")
    control = _new_operation(setup, run_id, "control")
    claimed = setup.claim_lease(op.id, submission_count=0, lease_seconds=60)
    assert claimed is not None
    assert setup.claim_lease(control.id, submission_count=0, lease_seconds=60) is not None

    stale_now = t0 + timedelta(seconds=60 + _lease_stale_after_seconds() + 1)
    heartbeat_pool = _pool(size=1)
    poll_pool = _pool(size=1)
    try:
        holder = _HoldBeforeCommitStore(LiveDatasetStore(connections=heartbeat_pool))
        worker = LiveOperationStore(store=holder, now=lambda: stale_now)
        poller = LiveOperationStore(
            store=LiveDatasetStore(connections=poll_pool), now=lambda: stale_now
        )

        expired_control = poller.expire_if_dead(control.id)
        assert expired_control is not None and expired_control.state == "failed"

        heartbeat = _Call(
            lambda: worker.progress(
                op.id, attempt=claimed.attempt_count, rows_written=5, lease_seconds=60
            )
        )
        assert holder.held.wait(timeout=5), "the heartbeat never reached its commit"
        expiry = _Call(lambda: poller.expire_if_dead(op.id))
        _wait_until_an_operation_update_is_blocked(live_store)
        holder.release.set()

        assert heartbeat.join() is not None
        assert expiry.join() is None
    finally:
        heartbeat_pool.close_all()
        poll_pool.close_all()

    current = setup.get(op.id)
    assert current is not None
    assert current.state == "running"
    assert current.rows_written == 5
    assert current.lease_until == stale_now + timedelta(seconds=60)


def test_operation_store_sweep_skips_a_row_a_heartbeat_renews_mid_sweep(live_store):
    """The create_or_get sweep is one set-based UPDATE. A heartbeat holding
    one of its candidate rows makes it wait on that row and re-evaluate it
    against the renewed lease, so the sweep leaves the live row alone."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    setup = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(setup, run_id, "sweep")
    claimed = setup.claim_lease(op.id, submission_count=0, lease_seconds=60)
    assert claimed is not None

    stale_now = t0 + timedelta(seconds=60 + _lease_stale_after_seconds() + 1)
    heartbeat_pool = _pool(size=1)
    submit_pool = _pool(size=1)
    try:
        holder = _HoldBeforeCommitStore(LiveDatasetStore(connections=heartbeat_pool))
        worker = LiveOperationStore(store=holder, now=lambda: stale_now)
        submitter = LiveOperationStore(
            store=LiveDatasetStore(connections=submit_pool),
            now=lambda: stale_now,
            backlog_max=_RACE_BACKLOG_MAX,
        )

        heartbeat = _Call(
            lambda: worker.progress(
                op.id, attempt=claimed.attempt_count, rows_written=7, lease_seconds=60
            )
        )
        assert holder.held.wait(timeout=5), "the heartbeat never reached its commit"
        submit = _Call(lambda: _new_operation(submitter, run_id, "trigger"))
        _wait_until_an_operation_update_is_blocked(live_store)
        holder.release.set()

        assert heartbeat.join() is not None
        assert submit.join().state == "pending"
    finally:
        heartbeat_pool.close_all()
        submit_pool.close_all()

    current = setup.get(op.id)
    assert current is not None
    assert current.state == "running"
    assert current.rows_written == 7


def test_operation_store_claim_with_an_old_submission_count_is_refused(live_store):
    """A resubmit reset the row to submission 1. A delivery of submission
    0's task, on its own connection, cannot claim it; submission 1's task
    can."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    setup = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(setup, run_id, "submission")
    first = setup.claim_lease(op.id, submission_count=0, lease_seconds=60)
    assert first is not None
    setup.fail(op.id, attempt=first.attempt_count, refusal_code="internal_error", retryable=True)
    reset = _new_operation(setup, run_id, "submission")
    assert reset.id == op.id
    assert reset.submission_count == 1

    old_pool = _pool(size=1)
    new_pool = _pool(size=1)
    try:
        old_delivery = LiveOperationStore(
            store=LiveDatasetStore(connections=old_pool), now=lambda: t0
        )
        new_delivery = LiveOperationStore(
            store=LiveDatasetStore(connections=new_pool), now=lambda: t0
        )

        assert old_delivery.claim_lease(op.id, submission_count=0, lease_seconds=60) is None
        claimed = new_delivery.claim_lease(op.id, submission_count=1, lease_seconds=60)
    finally:
        old_pool.close_all()
        new_pool.close_all()

    assert claimed is not None
    assert claimed.submission_count == 1
    assert claimed.attempt_count == first.attempt_count + 1


def test_operation_store_two_concurrent_dispatch_claims_have_one_winner(live_store):
    """Two concurrent claims yield one dispatch winner."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    setup = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(setup, run_id, "dispatch")
    assert op.dispatch_count == 0

    first_pool = _pool(size=1)
    second_pool = _pool(size=1)
    try:
        holder = _HoldBeforeCommitStore(LiveDatasetStore(connections=first_pool))
        first = LiveOperationStore(store=holder, now=lambda: t0)
        second = LiveOperationStore(store=LiveDatasetStore(connections=second_pool), now=lambda: t0)

        first_claim = _Call(lambda: first.claim_dispatch(op.id))
        assert holder.held.wait(timeout=5), "the first dispatch claim never reached its commit"
        second_claim = _Call(lambda: second.claim_dispatch(op.id))
        _wait_until_an_operation_update_is_blocked(live_store)
        holder.release.set()

        won = first_claim.join()
        assert second_claim.join() is None
    finally:
        first_pool.close_all()
        second_pool.close_all()

    assert won is not None
    assert won.dispatch_count == 1
    assert setup.get(op.id).dispatch_count == 1


def test_operation_store_two_pollers_racing_a_redispatch_enqueue_once(live_store):
    """Concurrent polls of one due redispatch enqueue and confirm one task."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    setup = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(setup, run_id, "pollers")
    assert setup.claim_dispatch(op.id) is not None  # its enqueue was lost

    t1 = t0 + timedelta(seconds=REDISPATCH_GRACE_SECONDS + 1)
    task_client = FakeTaskClient()
    first_pool = _pool(size=1)
    second_pool = _pool(size=1)
    try:
        holder = _HoldBeforeCommitStore(
            LiveDatasetStore(connections=first_pool),
            hold_when=lambda sql: "dispatch_count = dispatch_count + 1" in sql,
        )
        first = LiveOperationStore(store=holder, now=lambda: t1)
        second = LiveOperationStore(store=LiveDatasetStore(connections=second_pool), now=lambda: t1)
        pending = setup.get(op.id)
        assert pending is not None

        first_poll = _Call(
            lambda: dispatch_if_due(pending, operation_store=first, task_client=task_client)
        )
        assert holder.held.wait(timeout=5), "the first poll's claim never reached its commit"
        second_poll = _Call(
            lambda: dispatch_if_due(pending, operation_store=second, task_client=task_client)
        )
        _wait_until_an_operation_update_is_blocked(live_store)
        holder.release.set()

        first_poll.join()
        second_poll.join()
    finally:
        first_pool.close_all()
        second_pool.close_all()

    assert task_client.enqueued == [(op.id, 0)]
    current = setup.get(op.id)
    assert current is not None
    assert current.dispatch_count == 2
    assert current.dispatch_confirmed == 2


def test_operation_store_a_confirmed_dispatch_is_never_reclaimed(live_store):
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    setup = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(setup, run_id, "confirmed")
    claimed = setup.claim_dispatch(op.id)
    assert claimed is not None
    setup.confirm_dispatch(op.id, submission_count=0, dispatch_count=claimed.dispatch_count)

    later = t0 + timedelta(hours=2)
    poller_pool = _pool(size=1)
    try:
        poller = LiveOperationStore(
            store=LiveDatasetStore(connections=poller_pool), now=lambda: later
        )
        assert poller.claim_dispatch(op.id) is None
        assert poller.expire_if_dead(op.id) is None
    finally:
        poller_pool.close_all()

    current = setup.get(op.id)
    assert current is not None
    assert current.state == "pending"
    assert current.dispatch_count == 1


# The test window covers one dispatch deadline plus the retry duration.
_WINDOW_DISPATCH_CONCURRENCY = _RACE_BACKLOG_MAX
_NEVER_CLAIMED_WINDOW = timedelta(
    seconds=never_claimed_stale_after_seconds(
        backlog_max=_RACE_BACKLOG_MAX,
        dispatch_concurrency=_WINDOW_DISPATCH_CONCURRENCY,
        max_retry_duration_seconds=_DEFAULT_SETTINGS.operation_max_retry_duration_seconds,
    )
)


def _windowed_store(live_store, now_box) -> LiveOperationStore:
    return LiveOperationStore(
        store=live_store,
        now=lambda: now_box["t"],
        backlog_max=_RACE_BACKLOG_MAX,
        dispatch_concurrency=_WINDOW_DISPATCH_CONCURRENCY,
    )


def test_operation_store_the_redispatch_cap_holds_and_then_fails_the_operation(live_store):
    """Past the cap the last task may still be queued, so the row stays
    pending until the never-claimed window has run from the last dispatch."""
    run_id = uuid4().hex
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    store = _windowed_store(live_store, now_box)
    op = _new_operation(store, run_id, "cap")

    won = []
    for _ in range(MAX_REDISPATCHES + 3):
        claimed = store.claim_dispatch(op.id)
        if claimed is not None:
            won.append(claimed.dispatch_count)
        now_box["t"] = now_box["t"] + timedelta(seconds=REDISPATCH_GRACE_SECONDS + 1)

    assert won == list(range(1, MAX_REDISPATCHES + 2))
    exhausted = store.get(op.id)
    assert exhausted is not None and exhausted.dispatched_at is not None
    last_dispatched_at = exhausted.dispatched_at
    now_box["t"] = last_dispatched_at + _NEVER_CLAIMED_WINDOW
    assert store.expire_if_dead(op.id) is None
    pending = store.get(op.id)
    assert pending is not None
    assert pending.state == "pending"

    now_box["t"] = last_dispatched_at + _NEVER_CLAIMED_WINDOW + timedelta(seconds=1)
    expired = store.expire_if_dead(op.id)
    assert expired is not None
    assert expired.state == "failed"
    assert expired.refusal_code == "internal_error"
    assert expired.retryable is True


def test_operation_store_a_resubmit_of_an_old_operation_survives_inside_its_new_window(
    live_store,
):
    """Resubmission resets `submitted_at`, so expiry uses the new dispatch window."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    now_box = {"t": t0}
    store = _windowed_store(live_store, now_box)
    first = _new_operation(store, run_id, "resubmit")
    claimed = store.claim_lease(first.id, submission_count=0, lease_seconds=60)
    assert claimed is not None
    assert store.fail(
        first.id, attempt=claimed.attempt_count, refusal_code="internal_error", retryable=True
    )

    now_box["t"] = t0 + _NEVER_CLAIMED_WINDOW + timedelta(hours=1)
    reset = _new_operation(store, run_id, "resubmit")
    assert reset.state == "pending"
    assert reset.submission_count == 1
    assert reset.created_at == first.created_at
    assert reset.submitted_at == now_box["t"]
    dispatched = store.claim_dispatch(reset.id)
    assert dispatched is not None and dispatched.dispatched_at is not None
    store.confirm_dispatch(reset.id, submission_count=1, dispatch_count=dispatched.dispatch_count)

    now_box["t"] = now_box["t"] + timedelta(seconds=2)
    assert store.expire_if_dead(reset.id) is None
    _new_operation(store, run_id, "resubmit-sweeper")
    current = store.get(reset.id)
    assert current is not None
    assert current.state == "pending"

    now_box["t"] = dispatched.dispatched_at + _NEVER_CLAIMED_WINDOW + timedelta(seconds=1)
    expired = store.expire_if_dead(reset.id)
    assert expired is not None
    assert expired.state == "failed"
    assert expired.retryable is True


class _StubCoverageSnapshot:
    stamp = "cov-test-live-execution"
    index = object()


class _StubCoverage:
    def snapshot_current(self):
        return _StubCoverageSnapshot()


def _worker_deps() -> DefinitionDeps:
    """Build minimal worker dependencies for the patched resolve and estimate calls."""
    return DefinitionDeps(
        spec_reader=None,  # type: ignore[arg-type]
        graph=None,  # type: ignore[arg-type]
        sdmx=None,  # type: ignore[arg-type]
        revision_reader=None,  # type: ignore[arg-type]
        artifact_store=None,  # type: ignore[arg-type]
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=1_000_000,
        member_concurrency=4,
        enumerator=None,  # type: ignore[arg-type]
        enrichment=None,  # type: ignore[arg-type]
        code_index=None,  # type: ignore[arg-type]
        rate_cache=None,  # type: ignore[arg-type]
        coverage=_StubCoverage(),  # type: ignore[arg-type]
        head_recheck_interval=timedelta(seconds=_DEFAULT_SETTINGS.head_recheck_seconds),
        head_max_age=timedelta(seconds=_DEFAULT_SETTINGS.head_max_age_seconds),
    )


def _subject_operation(store: LiveOperationStore, run_id: str):
    return store.create_or_get(
        request_identity={"subject_id": f"spec_test_live_exec_{run_id}"},
        options={"format": "csv"},
        coverage_stamp="cov-test-live-admission",
        origin_catalogue_build_id="fake-catalogue",
        estimate=Estimate(rows=200_000, requests=5),
    )


def _must_not_serve(**_):
    raise AssertionError("an over-ceiling or superseded run must not materialise")


def test_worker_fails_over_ceiling_when_the_execution_estimate_breaches_it(live_store, monkeypatch):
    """An execution estimate above the row limit fails without serving data."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    store = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _subject_operation(store, run_id)
    claimed = store.claim_lease(op.id, submission_count=0, lease_seconds=600)
    assert claimed is not None
    over = Estimate(rows=DatasetsBuildSettings().async_max_rows + 1, requests=1)
    monkeypatch.setattr(worker_module, "_resolve_spec_requests", lambda **_: (object(), []))
    monkeypatch.setattr(
        worker_module, "sync_estimate", lambda requests, *, index, member_concurrency: over
    )
    monkeypatch.setattr(worker_module, "serve_definition", _must_not_serve)

    run_operation(
        operation=claimed,
        deps=_worker_deps(),
        operation_store=store,
        lease_seconds=_DEFAULT_SETTINGS.operation_lease_seconds,
    )

    stored = store.get(op.id)
    assert stored is not None
    assert stored.state == "failed"
    assert stored.refusal_code == "over_ceiling"
    assert stored.retryable is False
    # The jsonb column returns the classified refusal the worker stored.
    assert stored.refusal is not None
    assert (stored.refusal.status, stored.refusal.code) == (422, "over_ceiling")
    assert stored.refusal.ceiling_kind == "operation_rows"
    assert stored.coverage_stamp == "cov-test-live-admission"
    assert stored.execution_coverage_stamp == _StubCoverageSnapshot.stamp
    assert stored.estimate == over


def test_worker_a_stale_attempt_cannot_fail_the_reclaimed_row_over_ceiling(live_store, monkeypatch):
    """A reclaimed attempt cannot record an over-ceiling failure against the new lease."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    setup = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _subject_operation(setup, run_id)
    first = setup.claim_lease(op.id, submission_count=0, lease_seconds=30)
    assert first is not None
    over = Estimate(rows=DatasetsBuildSettings().async_max_rows + 1, requests=1)
    monkeypatch.setattr(worker_module, "_resolve_spec_requests", lambda **_: (object(), []))
    monkeypatch.setattr(
        worker_module, "sync_estimate", lambda requests, *, index, member_concurrency: over
    )
    monkeypatch.setattr(worker_module, "serve_definition", _must_not_serve)

    t1 = t0 + timedelta(seconds=31)
    reclaim_pool = _pool(size=1)
    stale_pool = _pool(size=1)
    try:
        holder = _HoldBeforeCommitStore(LiveDatasetStore(connections=reclaim_pool))
        reclaimer = LiveOperationStore(store=holder, now=lambda: t1)
        stale_store = LiveOperationStore(
            store=LiveDatasetStore(connections=stale_pool), now=lambda: t1
        )

        reclaim = _Call(
            lambda: reclaimer.claim_lease(op.id, submission_count=0, lease_seconds=1_700)
        )
        assert holder.held.wait(timeout=5), "the reclaim never reached its commit"
        stale_run = _Call(
            lambda: run_operation(
                operation=first,
                deps=_worker_deps(),
                operation_store=stale_store,
                lease_seconds=_DEFAULT_SETTINGS.operation_lease_seconds,
            )
        )
        _wait_until_an_operation_update_is_blocked(live_store)
        holder.release.set()

        reclaimed = reclaim.join()
        with pytest.raises(OperationSuperseded):
            stale_run.join()
    finally:
        reclaim_pool.close_all()
        stale_pool.close_all()

    assert reclaimed is not None
    current = setup.get(op.id)
    assert current is not None
    assert current.state == "running"
    assert current.attempt_count == first.attempt_count + 1
    assert current.refusal_code is None
    assert current.execution_coverage_stamp is None


def test_artifact_reference_write_once_per_principal_and_the_pk_matches_decision_8(live_store):
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes) "
            "VALUES (%s, %s) ON CONFLICT (artifact_digest) DO NOTHING",
            ("sha256:test-live-reference", 5),
        )
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at) VALUES (%s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (subject_id, content_version) DO NOTHING",
            ("spec:test-live-reference", "spec", "cv1", 1, {}),
        )
        tx.execute(
            "INSERT INTO datasets.artifact_reference "
            "(principal, artifact_digest, subject_id, content_version, reference_kind, "
            "created_at) VALUES (%s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (principal, artifact_digest, subject_id, content_version, "
            "reference_kind) DO NOTHING",
            ("first-party", "sha256:test-live-reference", "spec:test-live-reference", "cv1", "pin"),
        )
    with live_store.transaction() as tx:
        # Repeating the same pin key leaves one row.
        tx.execute(
            "INSERT INTO datasets.artifact_reference "
            "(principal, artifact_digest, subject_id, content_version, reference_kind, "
            "created_at) VALUES (%s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (principal, artifact_digest, subject_id, content_version, "
            "reference_kind) DO NOTHING",
            ("first-party", "sha256:test-live-reference", "spec:test-live-reference", "cv1", "pin"),
        )
    with live_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT removed_at FROM datasets.artifact_reference WHERE principal = %s "
            "AND artifact_digest = %s AND subject_id = %s AND content_version = %s "
            "AND reference_kind = %s",
            ("first-party", "sha256:test-live-reference", "spec:test-live-reference", "cv1", "pin"),
        )
    assert len(rows) == 1
    assert rows[0]["removed_at"] is None


def _seed_reference_fk_targets(live_store, *, digest: str, subject_id: str) -> None:
    """Seed the rows a reference names, and the link the guarded writer requires."""
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes) "
            "VALUES (%s, %s) ON CONFLICT (artifact_digest) DO NOTHING",
            (digest, 5),
        )
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at) VALUES (%s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (subject_id, content_version) DO NOTHING",
            (subject_id, "spec", "cv1", 1, {}),
        )
        tx.execute(
            "INSERT INTO datasets.version_artifact "
            "(subject_id, content_version, artifact_digest, rendering_key, rendering, "
            "media_type) VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (subject_id, content_version, rendering_key) DO NOTHING",
            (subject_id, "cv1", digest, f"rk-{digest}", {"format": "csv"}, "text/csv"),
        )


@pytest.mark.parametrize("writer_method", ["record_reference", "record_reference_unless_revoked"])
def test_artifact_reference_upsert_inserts_a_fresh_pin(live_store, writer_method):
    # Postgres parses the full upsert on insert, so this exercises SQLSTATE 42702 on both writers.
    # The unseeded revocation table also checks that the guarded writer inserts an active version.
    digest = f"sha256:test-live-reference-upsert-fresh-{writer_method}"
    subject_id = f"spec:test-live-reference-upsert-fresh-{writer_method}"
    _seed_reference_fk_targets(live_store, digest=digest, subject_id=subject_id)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    writer = LiveArtifactReferenceWriter(store=live_store)
    write = getattr(writer, writer_method)
    returned = write(
        principal="first-party",
        artifact_digest=digest,
        subject_id=subject_id,
        content_version="cv1",
        reference_kind=PIN_REFERENCE_KIND,
        now=now,
    )

    assert returned == now
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT created_at, removed_at FROM datasets.artifact_reference "
            "WHERE principal = %s AND artifact_digest = %s AND subject_id = %s "
            "AND content_version = %s AND reference_kind = %s",
            ("first-party", digest, subject_id, "cv1", PIN_REFERENCE_KIND),
        )
    assert row is not None
    assert row["created_at"] == now
    assert row["removed_at"] is None


@pytest.mark.parametrize("writer_method", ["record_reference", "record_reference_unless_revoked"])
def test_artifact_reference_upsert_repeat_pin_of_a_live_row_keeps_the_original_created_at(
    live_store, writer_method
):
    # Both writers must preserve `created_at` when pinning a live row again.
    digest = f"sha256:test-live-reference-upsert-repeat-{writer_method}"
    subject_id = f"spec:test-live-reference-upsert-repeat-{writer_method}"
    _seed_reference_fk_targets(live_store, digest=digest, subject_id=subject_id)
    first_pin = datetime(2026, 1, 1, tzinfo=UTC)
    later = datetime(2026, 2, 1, tzinfo=UTC)

    writer = LiveArtifactReferenceWriter(store=live_store)
    write = getattr(writer, writer_method)
    kwargs = {
        "principal": "first-party",
        "artifact_digest": digest,
        "subject_id": subject_id,
        "content_version": "cv1",
        "reference_kind": PIN_REFERENCE_KIND,
    }
    first = write(now=first_pin, **kwargs)
    second = write(now=later, **kwargs)

    assert first == second == first_pin
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT created_at, removed_at FROM datasets.artifact_reference "
            "WHERE principal = %s AND artifact_digest = %s AND subject_id = %s "
            "AND content_version = %s AND reference_kind = %s",
            ("first-party", digest, subject_id, "cv1", PIN_REFERENCE_KIND),
        )
    assert row is not None
    assert row["created_at"] == first_pin  # never bumped by the repeat pin
    assert row["removed_at"] is None


def test_pinned_copy_count_counts_each_pin_door_promotion_once(live_store):
    """The pin budget reads this count. Two `pin` references to one pinned copy add one, and so
    does a copy whose only `pin` reference was removed, since unpinning leaves the copy stored. A
    working copy, a pinned copy older than the window, and a frame-store pin add nothing."""
    now = datetime.now(UTC)
    since = now - timedelta(hours=24)
    writer = LiveArtifactReferenceWriter(store=live_store)

    def count() -> int:
        return writer.count_pinned_copies_since(reference_kind=PIN_REFERENCE_KIND, since=since)

    baseline = count()
    suffix = uuid4().hex
    seeds = {
        "shared": ("pinned", now - timedelta(hours=1), PIN_REFERENCE_KIND),
        "removed": ("pinned", now - timedelta(hours=1), PIN_REFERENCE_KIND),
        "working": ("working", now, PIN_REFERENCE_KIND),
        "old": ("pinned", now - timedelta(hours=25), PIN_REFERENCE_KIND),
        "frame": ("pinned", now - timedelta(hours=1), "presentation_draft"),
    }
    subjects = (f"spec:test-live-pinned-count-{suffix}", f"col_test_live_count_{suffix}")
    for name, (tier, stored_at, kind) in seeds.items():
        digest = f"sha256:test-live-pinned-count-{name}-{suffix}"
        for subject_id in subjects if name == "shared" else subjects[:1]:
            _seed_reference_fk_targets(live_store, digest=digest, subject_id=subject_id)
            writer.record_reference(
                principal="first-party",
                artifact_digest=digest,
                subject_id=subject_id,
                content_version="cv1",
                reference_kind=kind,
                now=now,
            )
            if name == "removed":
                writer.remove_reference(
                    principal="first-party",
                    artifact_digest=digest,
                    subject_id=subject_id,
                    content_version="cv1",
                    reference_kind=kind,
                    now=now,
                )
        with live_store.transaction() as tx:
            tx.execute(
                "INSERT INTO datasets.artifact_location "
                "(artifact_digest, tier, bucket, object_name, stored_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (digest, tier, "bucket", digest, stored_at),
            )

    assert count() == baseline + 2


def test_artifact_reference_upsert_reactivates_a_soft_deleted_row(live_store):
    digest = "sha256:test-live-reference-upsert-reactivate"
    subject_id = "spec:test-live-reference-upsert-reactivate"
    _seed_reference_fk_targets(live_store, digest=digest, subject_id=subject_id)
    first_pin = datetime(2026, 1, 1, tzinfo=UTC)
    removed_at = datetime(2026, 2, 1, tzinfo=UTC)
    reactivated_at = datetime(2026, 3, 1, tzinfo=UTC)

    writer = LiveArtifactReferenceWriter(store=live_store)
    writer.record_reference(
        principal="first-party",
        artifact_digest=digest,
        subject_id=subject_id,
        content_version="cv1",
        reference_kind=PIN_REFERENCE_KIND,
        now=first_pin,
    )
    # Soft-delete directly because this test covers the store write path.
    with live_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_reference SET removed_at = %s "
            "WHERE principal = %s AND artifact_digest = %s AND subject_id = %s "
            "AND content_version = %s AND reference_kind = %s",
            (removed_at, "first-party", digest, subject_id, "cv1", PIN_REFERENCE_KIND),
        )

    returned = writer.record_reference(
        principal="first-party",
        artifact_digest=digest,
        subject_id=subject_id,
        content_version="cv1",
        reference_kind=PIN_REFERENCE_KIND,
        now=reactivated_at,
    )

    assert returned == reactivated_at
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT created_at, removed_at FROM datasets.artifact_reference "
            "WHERE principal = %s AND artifact_digest = %s AND subject_id = %s "
            "AND content_version = %s AND reference_kind = %s",
            ("first-party", digest, subject_id, "cv1", PIN_REFERENCE_KIND),
        )
    assert row is not None
    assert row["created_at"] == reactivated_at  # a repin is a genuinely new reference
    assert row["removed_at"] is None


def test_record_reference_unless_revoked_reactivates_a_soft_deleted_row(live_store):
    # Exercise this writer's upsert independently from `record_reference`.
    digest = "sha256:test-live-reference-unless-revoked-reactivate"
    subject_id = "spec:test-live-reference-unless-revoked-reactivate"
    _seed_reference_fk_targets(live_store, digest=digest, subject_id=subject_id)
    first_pin = datetime(2026, 1, 1, tzinfo=UTC)
    removed_at = datetime(2026, 2, 1, tzinfo=UTC)
    reactivated_at = datetime(2026, 3, 1, tzinfo=UTC)

    writer = LiveArtifactReferenceWriter(store=live_store)
    writer.record_reference_unless_revoked(
        principal="first-party",
        artifact_digest=digest,
        subject_id=subject_id,
        content_version="cv1",
        reference_kind=PIN_REFERENCE_KIND,
        now=first_pin,
    )
    with live_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_reference SET removed_at = %s "
            "WHERE principal = %s AND artifact_digest = %s AND subject_id = %s "
            "AND content_version = %s AND reference_kind = %s",
            (removed_at, "first-party", digest, subject_id, "cv1", PIN_REFERENCE_KIND),
        )

    returned = writer.record_reference_unless_revoked(
        principal="first-party",
        artifact_digest=digest,
        subject_id=subject_id,
        content_version="cv1",
        reference_kind=PIN_REFERENCE_KIND,
        now=reactivated_at,
    )

    assert returned == reactivated_at
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT created_at, removed_at FROM datasets.artifact_reference "
            "WHERE principal = %s AND artifact_digest = %s AND subject_id = %s "
            "AND content_version = %s AND reference_kind = %s",
            ("first-party", digest, subject_id, "cv1", PIN_REFERENCE_KIND),
        )
    assert row is not None
    assert row["created_at"] == reactivated_at
    assert row["removed_at"] is None


def test_0002_spec_registry_migration_creates_tables_and_constraints():
    # The schema-qualified PostgreSQL DDL uses types and foreign keys that SQLite cannot exercise.
    parts = urlsplit(_DSN)
    runtime_user = parts.username

    def _connect():
        return connect(
            connector=None,
            instance_connection_name=None,
            host=parts.hostname,
            port=parts.port or 5432,
            database=parts.path.lstrip("/"),
            user=parts.username,
            password=parts.password,
        )

    apply_pending(
        connect=_connect,
        migrations_dir=MIGRATIONS_DIR,
        substitutions={"RUNTIME_DB_USER": runtime_user},
    )

    pool = _pool()
    try:
        store = LiveDatasetStore(connections=pool)

        expected_tables = {"spec_registry", "spec_definition", "identity_quarantine"}
        # 0021_drop_collections.sql drops the two collection tables 0002 creates.
        dropped_tables = {"collection_registry", "collection_member"}
        with store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'datasets' AND table_name = ANY(%s)",
                (list(expected_tables | dropped_tables),),
            )
        assert {row["table_name"] for row in rows} == expected_tables

        with store.transaction() as tx:
            check_row = tx.fetch_one(
                "SELECT pg_get_constraintdef(oid) AS def FROM pg_constraint "
                "WHERE conrelid = 'datasets.identity_quarantine'::regclass AND contype = 'c'"
            )
        assert check_row is not None
        assert "id_kind" in check_row["def"]

        with store.transaction() as tx:
            fk_rows = tx.fetch_all(
                "SELECT conrelid::regclass::text AS table_name "
                "FROM pg_constraint "
                "WHERE contype = 'f' "
                "AND conrelid = 'datasets.spec_definition'::regclass"
            )
        assert {row["table_name"] for row in fk_rows} == {"datasets.spec_definition"}
    finally:
        pool.close_all()


def _rerun_migration(live_store, version: str) -> list[str]:
    """Drops `version`'s bookkeeping row and applies pending migrations, which reruns only it."""
    with live_store.transaction() as tx:
        tx.execute(
            "DELETE FROM datasets.dataset_store_schema_migrations WHERE version = %s",
            (version,),
        )
    parts = urlsplit(_DSN)
    return apply_pending(
        connect=lambda: connect(
            connector=None,
            instance_connection_name=None,
            host=parts.hostname,
            port=parts.port or 5432,
            database=parts.path.lstrip("/"),
            user=parts.username,
            password=parts.password,
        ),
        migrations_dir=MIGRATIONS_DIR,
        substitutions={},
    )


_CONVERSION_FALLBACK = {"count": 3, "examples": [["KEN", "2019"]], "truncated": False}
_CONVERSION_PROJECTED = {"count": 0, "examples": [], "truncated": False}

# Stored `trace.conversion` values and what each migration leaves in their place. An object
# loses `native_facets` and gains `members`, sorted by `stat_var` under COLLATE "C"
# (code-point order, so upper case and `_` sort before lower case); every other key is kept,
# present or absent. A list, a JSON null and a trace with no conversion are left alone.
_STORED_CONVERSIONS: dict[str, tuple[object, object]] = {
    "legacy": (LEGACY_CONVERSION, MIGRATED_CONVERSION),
    "deflated": (LEGACY_DEFLATED_CONVERSION, MIGRATED_DEFLATED_CONVERSION),
    "code-point-order": (
        {
            "recipe": "ConversionRecipeDAC",
            "rate_table_version": "r1",
            "deflator_table_version": "d1",
            "native_facets": {
                "oda_b": "CurrencyGBP",
                "Oda_c": "CurrencyEUR",
                "ODA_a": "CurrencyUSD",
                "oda_A": "CurrencyJPY",
            },
            "fallback": _CONVERSION_FALLBACK,
            "projected": _CONVERSION_PROJECTED,
        },
        [
            {
                "recipe": "ConversionRecipeDAC",
                "rate_table_version": "r1",
                "deflator_table_version": "d1",
                "members": [
                    {"stat_var": "ODA_a", "native_unit": "CurrencyUSD"},
                    {"stat_var": "Oda_c", "native_unit": "CurrencyEUR"},
                    {"stat_var": "oda_A", "native_unit": "CurrencyJPY"},
                    {"stat_var": "oda_b", "native_unit": "CurrencyGBP"},
                ],
                "fallback": _CONVERSION_FALLBACK,
                "projected": _CONVERSION_PROJECTED,
            }
        ],
    ),
    "no-fallback-or-projected": (
        {
            "recipe": "ConversionRecipeDAC",
            "rate_table_version": "r1",
            "native_facets": {"b": "CurrencyEUR", "a": "CurrencyUSD"},
        },
        [
            {
                "recipe": "ConversionRecipeDAC",
                "rate_table_version": "r1",
                "members": [
                    {"stat_var": "a", "native_unit": "CurrencyUSD"},
                    {"stat_var": "b", "native_unit": "CurrencyEUR"},
                ],
            }
        ],
    ),
    "no-native-facets": (
        {"recipe": "ConversionRecipeDAC", "rate_table_version": "r1"},
        [{"recipe": "ConversionRecipeDAC", "rate_table_version": "r1", "members": []}],
    ),
    "list": (MIGRATED_CONVERSION, MIGRATED_CONVERSION),
    "null": (None, None),
}


def _conversion_body(spec_id: str, trace: dict[str, object] | None) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": {}},
        "content_version": "cv1",
        "prepared_at": "2026-01-01T00:00:00Z",
        "coverage": {"rows": 1, "dimensions": []},
        "citation": {"text": "", "sources": []},
        "warnings": [],
    }
    if trace is not None:
        body["trace"] = trace
    return body


@pytest.mark.parametrize(
    "migration",
    ["0017_trace_conversion_entries.sql", "0020_trace_conversion_entries_after_deploy.sql"],
)
def test_trace_conversion_migrations_rewrite_object_shapes_into_one_entry_lists(
    live_store, migration
):
    """Object-shaped `trace.conversion` bodies become one-entry lists with sorted `members`
    and decode as manifests; every other key in the body and trace is kept, a list, null or
    absent conversion and a body with no trace are left alone, and a rerun changes nothing."""
    from datasets_api.manifest import decode_version_body
    from datasets_api.manifest_models import Manifest

    suffix = uuid4().hex
    trace: dict[str, object] = {
        "graph_revision": "rev",
        "source_revisions": [],
        "facets": [{"axis": "unit"}],
    }
    seeded: dict[str, dict[str, object]] = {}
    expected: dict[str, dict[str, object]] = {}
    for case, (stored, migrated) in _STORED_CONVERSIONS.items():
        spec_id = f"spec_conv_{case}_{suffix}"
        seeded[spec_id] = _conversion_body(spec_id, {**trace, "conversion": stored})
        expected[spec_id] = _conversion_body(spec_id, {**trace, "conversion": migrated})
    for case, case_trace in (("no-conversion", trace), ("no-trace", None)):
        spec_id = f"spec_conv_{case}_{suffix}"
        seeded[spec_id] = expected[spec_id] = _conversion_body(spec_id, case_trace)

    def _bodies() -> dict[str, object]:
        with live_store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT subject_id, immutable_body FROM datasets.version_record "
                "WHERE subject_id = ANY(%s)",
                (list(seeded),),
            )
        return {cast(str, row["subject_id"]): row["immutable_body"] for row in rows}

    with live_store.transaction() as tx:
        for subject_id, body in seeded.items():
            tx.execute(
                "INSERT INTO datasets.version_record "
                "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
                "recorded_at) VALUES (%s, 'spec', 'cv1', 1, %s, now())",
                (subject_id, body),
            )
    try:
        assert _rerun_migration(live_store, migration) == [migration]
        assert _bodies() == expected

        assert _rerun_migration(live_store, migration) == [migration]
        bodies = _bodies()
        assert bodies == expected

        for body in bodies.values():
            body = cast(dict[str, object], body)
            if "trace" not in body:
                continue
            fields, _profile = decode_version_body(body)
            manifest = Manifest.model_validate({**fields, "artifacts": []})
            assert manifest.trace == body["trace"]
    finally:
        with live_store.transaction() as tx:
            tx.execute(
                "DELETE FROM datasets.version_record WHERE subject_id = ANY(%s)", (list(seeded),)
            )


_BACKFILL_MIGRATION = "0018_backfill_prepared_at.sql"


def _old_manifest_body(*, spec_id: str, content_version: str) -> dict[str, object]:
    """A definition body as stored before `prepared_at` and the citation's licence and
    documentation fields existed."""
    return {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": {}},
        "content_version": content_version,
        "coverage": {"rows": 2, "dimensions": []},
        "citation": {
            "text": "OECD DAC2a",
            "sources": [
                {
                    "ref": "OECD_DAC2A",
                    "label": "OECD DAC2a",
                    "processing_level": None,
                    "license": "Creative Commons Attribution License",
                    "url": None,
                    "source": None,
                    "data_commons": None,
                }
            ],
        },
        "trace": {"graph_revision": "rev"},
        "warnings": [],
    }


def test_0018_backfills_prepared_at_and_keeps_every_link(live_store):
    """Old bodies gain `prepared_at` from `recorded_at`, a collapsed record its base's, and
    collection members their own record's `prepared_at` and cited refs, a first member with no
    record the collection trace's provenance facet; a definition body decodes with the current models,
    every artifact link stays, `ops_` subjects are untouched, and a rerun changes nothing."""
    from datasets_api.manifest import decode_version_body
    from datasets_api.manifest_models import Manifest
    from datasets_api.version_store import rendering_key

    suffix = uuid4().hex
    definition = f"spec_backfill_def_{suffix}"
    member_a = f"spec_backfill_a_{suffix}"
    member_b = f"spec_backfill_b_{suffix}"
    collection = f"col_backfill_{suffix}"
    first_unrecorded = f"col_backfill_first_{suffix}"
    derived = f"ops_backfill_{suffix}"
    subjects = [definition, member_a, collection, first_unrecorded, derived]
    first_mint = datetime(2026, 9, 1, 8, 30, 15, 987654, tzinfo=UTC)
    recorded = {
        member_a: first_mint,
        definition: first_mint + timedelta(days=1),
        collection: first_mint + timedelta(days=2),
        first_unrecorded: first_mint + timedelta(days=2),
        derived: first_mint + timedelta(days=3),
    }
    # A head-hit groupby on the definition, recorded days after its base.
    collapsed_recorded = first_mint + timedelta(days=5)
    provenance_facets = [
        {"axis": "stat_var", "ref": "dcid:x"},
        {"axis": "provenance", "ref": "OECD_DAC2A"},
    ]

    def _collection_body(collection_id: str, members: tuple[str, ...]) -> dict[str, object]:
        return {
            **_old_manifest_body(spec_id=members[0], content_version="cv_col"),
            "definition": None,
            "collection": {
                "collection_id": collection_id,
                "members": [
                    {
                        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": {}},
                        "content_version": f"cv_{spec_id}",
                        "coverage": {"rows": 1, "dimensions": []},
                        "series_count": 1,
                        "output_profile": None,
                    }
                    for spec_id in members
                ],
            },
            "trace": {"graph_revision": "rev", "facets": provenance_facets},
        }

    collection_body = _collection_body(collection, (member_a, member_b))
    derived_body = {"spec_id": derived, "input_content_versions": ["cv_x"]}
    member_a_body = _old_manifest_body(spec_id=member_a, content_version=f"cv_{member_a}")
    member_a_citation = cast(dict, member_a_body["citation"])
    (oecd_source,) = member_a_citation["sources"]
    imf_source = {**oecd_source, "ref": "dc/base/IMF", "label": "IMF"}
    # A repeated ref (here under a second processing level) counts once, at its first place.
    member_a_citation["sources"] = [
        imf_source,
        oecd_source,
        {**imf_source, "processing_level": "minor"},
    ]
    rows = [
        (
            definition,
            "spec",
            "cv_def",
            _old_manifest_body(spec_id=definition, content_version="cv_def"),
        ),
        (member_a, "spec", f"cv_{member_a}", member_a_body),
        (collection, "collection", "cv_col", collection_body),
        (first_unrecorded, "collection", "cv_col", _collection_body(first_unrecorded, (member_b,))),
        (derived, "spec", "cv_ops", derived_body),
    ]
    csv = {"format": "csv", "bundle": False}
    bundle = {"format": "csv", "bundle": True}
    pinned_bundle = {"format": "xlsx", "bundle": True}
    links = [
        (definition, "cv_def", f"sha256:def-csv-{suffix}", csv),
        (definition, "cv_def", f"sha256:def-bundle-{suffix}", bundle),
        (definition, "cv_def", f"sha256:def-pinned-{suffix}", pinned_bundle),
        (collection, "cv_col", f"sha256:col-bundle-{suffix}", bundle),
        (derived, "cv_ops", f"sha256:ops-bundle-{suffix}", bundle),
    ]
    digests = [digest for _subject, _cv, digest, _rendering in links]

    def _links() -> set[tuple[str, str]]:
        with live_store.transaction() as tx:
            found = tx.fetch_all(
                "SELECT subject_id, artifact_digest FROM datasets.version_artifact "
                "WHERE subject_id = ANY(%s)",
                (subjects,),
            )
        return {(cast(str, row["subject_id"]), cast(str, row["artifact_digest"])) for row in found}

    def _bodies() -> dict[str, dict]:
        with live_store.transaction() as tx:
            found = tx.fetch_all(
                "SELECT subject_id, immutable_body FROM datasets.version_record "
                "WHERE subject_id = ANY(%s)",
                (subjects,),
            )
        return {cast(str, row["subject_id"]): cast(dict, row["immutable_body"]) for row in found}

    with live_store.transaction() as tx:
        for subject_id, kind, content_version, body in rows:
            tx.execute(
                "INSERT INTO datasets.version_record "
                "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
                "recorded_at) VALUES (%s, %s, %s, 1, %s, %s)",
                (subject_id, kind, content_version, body, recorded[subject_id]),
            )
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at, base_content_version) VALUES (%s, 'spec', 'cv_collapsed', 1, %s, %s, "
            "'cv_def')",
            (
                definition,
                _old_manifest_body(spec_id=definition, content_version="cv_collapsed"),
                collapsed_recorded,
            ),
        )
        for subject_id, content_version, digest, rendering in links:
            tx.execute(
                "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes) "
                "VALUES (%s, 10)",
                (digest,),
            )
            tx.execute(
                "INSERT INTO datasets.version_artifact (subject_id, content_version, "
                "artifact_digest, rendering_key, rendering, media_type) "
                "VALUES (%s, %s, %s, %s, %s, 'application/zip')",
                (subject_id, content_version, digest, rendering_key(rendering), rendering),
            )
        tx.execute(
            "INSERT INTO datasets.artifact_reference (principal, artifact_digest, subject_id, "
            "content_version, reference_kind, created_at) VALUES (%s, %s, %s, %s, %s, now())",
            ("site", f"sha256:def-pinned-{suffix}", definition, "cv_def", PIN_REFERENCE_KIND),
        )
    try:
        assert _rerun_migration(live_store, _BACKFILL_MIGRATION) == [_BACKFILL_MIGRATION]
        bodies = _bodies()
        remaining = _links()
        assert _rerun_migration(live_store, _BACKFILL_MIGRATION) == [_BACKFILL_MIGRATION]

        assert _bodies() == bodies
        assert _links() == remaining == {(subject, digest) for subject, _cv, digest, _r in links}
        fields, _profile = decode_version_body(bodies[definition])
        manifest = Manifest.model_validate({**fields, "artifacts": []})
        assert manifest.prepared_at == datetime(2026, 9, 2, 8, 30, 15, tzinfo=UTC)
        assert bodies[definition]["prepared_at"] == "2026-09-02T08:30:15Z"
        with live_store.transaction() as tx:
            collapsed = tx.fetch_one(
                "SELECT immutable_body FROM datasets.version_record "
                "WHERE subject_id = %s AND content_version = 'cv_collapsed'",
                (definition,),
            )
        assert collapsed is not None
        assert cast(dict, collapsed["immutable_body"])["prepared_at"] == "2026-09-02T08:30:15Z"
        assert "documentation" not in bodies[definition]["citation"]["sources"][0]

        # Collection bodies stay stored after Datasets stopped serving them, so their raw
        # JSON is checked.
        assert bodies[collection]["prepared_at"] == "2026-09-03T08:30:15Z"
        members = bodies[collection]["collection"]["members"]
        assert [member["prepared_at"] for member in members] == [
            "2026-09-01T08:30:15Z",
            "2026-09-03T08:30:15Z",
        ]
        assert [member["provenance_refs"] for member in members] == [
            ["dc/base/IMF", "OECD_DAC2A"],
            [],
        ]
        members = bodies[first_unrecorded]["collection"]["members"]
        assert [member["provenance_refs"] for member in members] == [["OECD_DAC2A"]]
        assert bodies[derived] == derived_body
    finally:
        with live_store.transaction() as tx:
            tx.execute(
                "DELETE FROM datasets.artifact_reference WHERE subject_id = ANY(%s)", (subjects,)
            )
            tx.execute(
                "DELETE FROM datasets.version_artifact WHERE subject_id = ANY(%s)", (subjects,)
            )
            tx.execute(
                "DELETE FROM datasets.version_record WHERE subject_id = ANY(%s)", (subjects,)
            )
            tx.execute(
                "DELETE FROM datasets.artifact_inventory WHERE artifact_digest = ANY(%s)",
                (digests,),
            )


_REPEAT_BACKFILL_MIGRATION = "0019_backfill_prepared_at_after_deploy.sql"


def test_0019_fills_rows_the_previous_revision_wrote_after_0018(live_store):
    """A body and a collection member written without the new keys once 0018 had run take the
    same values 0018 gives, a body that carries its keys keeps them, and a rerun changes
    nothing."""
    from datasets_api.manifest import decode_version_body
    from datasets_api.manifest_models import Manifest

    suffix = uuid4().hex
    definition = f"spec_window_def_{suffix}"
    unrecorded = f"spec_window_none_{suffix}"
    collection = f"col_window_{suffix}"
    current = f"spec_window_current_{suffix}"
    subjects = [definition, collection, current]
    minted = datetime(2026, 10, 1, 7, 45, 30, 123456, tzinfo=UTC)
    definition_body = _old_manifest_body(spec_id=definition, content_version="cv_def")
    collection_body = {
        **_old_manifest_body(spec_id=definition, content_version="cv_col"),
        "definition": None,
        "collection": {
            "collection_id": collection,
            "members": [
                {
                    "definition": {"schema_version": 1, "spec_id": spec_id, "identity": {}},
                    "content_version": content_version,
                    "coverage": {"rows": 1, "dimensions": []},
                    "series_count": 1,
                    "output_profile": None,
                }
                for spec_id, content_version in ((definition, "cv_def"), (unrecorded, "cv_none"))
            ],
        },
    }
    current_body = {
        **_old_manifest_body(spec_id=current, content_version="cv_current"),
        "prepared_at": "2020-01-02T03:04:05Z",
    }
    rows = [
        (definition, "spec", "cv_def", definition_body, minted),
        (collection, "collection", "cv_col", collection_body, minted + timedelta(days=1)),
        (current, "spec", "cv_current", current_body, minted + timedelta(days=2)),
    ]

    def _bodies() -> dict[str, dict]:
        with live_store.transaction() as tx:
            found = tx.fetch_all(
                "SELECT subject_id, immutable_body FROM datasets.version_record "
                "WHERE subject_id = ANY(%s)",
                (subjects,),
            )
        return {cast(str, row["subject_id"]): cast(dict, row["immutable_body"]) for row in found}

    with live_store.transaction() as tx:
        for subject_id, kind, content_version, body, recorded_at in rows:
            tx.execute(
                "INSERT INTO datasets.version_record "
                "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
                "recorded_at) VALUES (%s, %s, %s, 1, %s, %s)",
                (subject_id, kind, content_version, body, recorded_at),
            )
    try:
        assert _rerun_migration(live_store, _REPEAT_BACKFILL_MIGRATION) == [
            _REPEAT_BACKFILL_MIGRATION
        ]
        bodies = _bodies()
        assert _rerun_migration(live_store, _REPEAT_BACKFILL_MIGRATION) == [
            _REPEAT_BACKFILL_MIGRATION
        ]
        assert _bodies() == bodies

        fields, _profile = decode_version_body(bodies[definition])
        manifest = Manifest.model_validate({**fields, "artifacts": []})
        assert manifest.prepared_at == datetime(2026, 10, 1, 7, 45, 30, tzinfo=UTC)
        assert bodies[current] == current_body

        assert bodies[collection]["prepared_at"] == "2026-10-02T07:45:30Z"
        members = bodies[collection]["collection"]["members"]
        assert [member["prepared_at"] for member in members] == [
            "2026-10-01T07:45:30Z",
            "2026-10-02T07:45:30Z",
        ]
        assert [member["provenance_refs"] for member in members] == [["OECD_DAC2A"], []]
    finally:
        with live_store.transaction() as tx:
            tx.execute(
                "DELETE FROM datasets.version_record WHERE subject_id = ANY(%s)", (subjects,)
            )


_NULL_COLLECTION_MIGRATION = "0022_drop_null_collection.sql"


def _null_collection_body(spec_id: str) -> dict[str, object]:
    """A definition body in the shape an earlier `encode_version_body` wrote, with `collection`
    stored as null and every optional field and nested field set. The identity is the
    projection of dfo-spec-kit's `shape-statvar-schema-all-five-members` identity vector."""
    return {
        "schema_version": 1,
        "definition": {
            "schema_version": 1,
            "spec_id": spec_id,
            "identity": {
                "identity_version": "spec1",
                "shape": {
                    "kind": "statvar_schema",
                    "population_type": "DevelopmentFinance",
                    "measured_property": "developmentFinanceFlow",
                    "stat_type": "measuredValue",
                    "measurement_qualifier": "Nominal",
                    "measurement_denominator": {"ref": "Count_Person"},
                },
                "slots": [
                    {
                        "property": "DevelopmentFinancePurpose",
                        "binding": {"state": "value", "ref": "Malaria"},
                    }
                ],
                "entities": {"origin": ["country/FRA"], "destination": ["country/TGO"]},
                "requested_window": {"kind": "range", "start_year": 2015, "end_year": 2024},
                "facets": {
                    "provenance": {"state": "value", "ref": "OECD_DAC2A"},
                    "unit": {
                        "state": "value",
                        "request": {"currency": "EUR", "prices": "constant", "base_year": 2023},
                    },
                },
            },
        },
        "collection": None,
        "content_version": "cv_dev",
        "prepared_at": "2026-09-30T12:00:00Z",
        "coverage": {
            "rows": 2,
            "dimensions": [{"label": "recipient", "count": 1, "axis": "destination"}],
            "window": {"kind": "range", "start_year": 2015, "end_year": 2016, "origin": "query"},
        },
        "citation": {
            "text": "OECD DAC2a",
            "sources": [
                {
                    "ref": "OECD_DAC2A",
                    "label": "OECD DAC2a",
                    "processing_level": "minor",
                    "license": "Creative Commons Attribution License",
                    "license_type": "CCBY4",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "url": "https://data-explorer.oecd.org",
                    "source": {"ref": "dc/base/OECD", "label": "OECD"},
                    "data_commons": {"id": "dfo", "label": "DFO Data Commons"},
                    "documentation": {
                        "publisher": "OECD",
                        "description": "Aid disbursements by donor and recipient.",
                        "source_data_url": "https://data-explorer.oecd.org/dac2a",
                        "attribution": "OECD (2026), DAC2a.",
                        "update_cadence": "Quarterly",
                        "last_update": "2026-09-01",
                        "years_covered": "1960-2024",
                        "granularity": "Annual, by provider and recipient",
                        "currencies": ["USD"],
                        "transformation_log": [
                            {"transformation": "Pivot", "description": "Long to wide."}
                        ],
                        "version_history": [
                            {
                                "date": "2026-09-01",
                                "title": "September refresh",
                                "changes": [{"kind": "added", "note": "2024 data."}],
                            }
                        ],
                    },
                }
            ],
        },
        "trace": {
            "graph_revision": "rev",
            "facets": [{"axis": "provenance", "ref": "OECD_DAC2A"}],
        },
        "warnings": [
            {
                "code": "stat_vars_summed",
                "severity": "info",
                "message": "Summed 2 StatVars.",
                "contributing_member_count": 2,
                "stat_var_ref": f"dcid:{spec_id}",
                "stat_var_label": "Gross ODA",
            }
        ],
        "series_count": 1,
        "series_coverage_stamp": "sha256:coverage",
        "output_profile": {
            "schema_version": 1,
            "rows": 2,
            "columns": [
                {"name": "recipient", "kind": "role", "droppable": False, "distinct": 1},
                {"name": "value", "kind": "value", "droppable": False, "distinct": None},
            ],
            "pivots": [
                {
                    "index": "recipient",
                    "columns": "time_period",
                    "duplicate_keys": True,
                    "homogeneous": False,
                    "unique": False,
                    "width": 3,
                    "cells": 3,
                    "varying": [["unit", 2]],
                }
            ],
            "enrichments": [
                {
                    "column": "recipient",
                    "field": "iso3",
                    "entities": 2,
                    "covered": 1,
                    "missing": [["country/XKX", "Kosovo"]],
                    "truncated": False,
                }
            ],
        },
        "series_conversions": [
            {
                "stat_var": "Oda_a",
                "roles": [["recipient", "country/KEN"]],
                "recipe_id": "ConversionRecipeDAC",
                "table_provenance_id": "dc/base/IMF",
                "exchange_rate_table_version": "r1",
                "deflator_table_version": "d1",
                "fallback_count": 1,
                "fallback_examples": [["country/KEN", "2015"]],
                "projected_count": 1,
                "projected_examples": ["2016"],
            }
        ],
        "default_scope_plural": True,
    }


def test_0022_drops_a_null_collection_so_the_body_decodes(live_store):
    """A body that stores `"collection": null` fails `Manifest` validation on that key alone and
    decodes once 0022 drops it, with every other key kept. A non-null `collection` and a body
    without the key stay as stored, and a rerun changes nothing."""
    from pydantic import ValidationError

    from datasets_api.manifest import decode_version_body
    from datasets_api.manifest_models import Manifest

    suffix = uuid4().hex
    definition = f"spec_null_collection_{suffix}"
    collection = f"col_null_collection_{suffix}"
    derived = f"ops_null_collection_{suffix}"
    subjects = [definition, collection, derived]
    definition_body = _null_collection_body(definition)
    collection_body = {
        **_old_manifest_body(spec_id=definition, content_version="cv_col"),
        "definition": None,
        "collection": {"collection_id": collection, "members": []},
    }
    derived_body = {"spec_id": derived, "input_content_versions": ["cv_x"]}
    rows = [
        (definition, "spec", definition_body),
        (collection, "collection", collection_body),
        (derived, "spec", derived_body),
    ]

    def _validate(body: dict[str, object]) -> Manifest:
        fields, _profile = decode_version_body(body)
        return Manifest.model_validate({**fields, "artifacts": []})

    def _bodies() -> dict[str, dict]:
        with live_store.transaction() as tx:
            found = tx.fetch_all(
                "SELECT subject_id, immutable_body FROM datasets.version_record "
                "WHERE subject_id = ANY(%s)",
                (subjects,),
            )
        return {cast(str, row["subject_id"]): cast(dict, row["immutable_body"]) for row in found}

    with pytest.raises(ValidationError) as refused:
        _validate(definition_body)
    assert [(error["type"], error["loc"]) for error in refused.value.errors()] == [
        ("extra_forbidden", ("collection",))
    ]

    with live_store.transaction() as tx:
        for subject_id, kind, body in rows:
            tx.execute(
                "INSERT INTO datasets.version_record "
                "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
                "recorded_at) VALUES (%s, %s, 'cv1', 1, %s, now())",
                (subject_id, kind, body),
            )
    try:
        assert _rerun_migration(live_store, _NULL_COLLECTION_MIGRATION) == [
            _NULL_COLLECTION_MIGRATION
        ]
        bodies = _bodies()
        assert _rerun_migration(live_store, _NULL_COLLECTION_MIGRATION) == [
            _NULL_COLLECTION_MIGRATION
        ]
        assert _bodies() == bodies

        expected = {key: value for key, value in definition_body.items() if key != "collection"}
        assert bodies == {
            definition: expected,
            collection: collection_body,
            derived: derived_body,
        }
        manifest = _validate(bodies[definition])
        assert manifest.definition.spec_id == definition
        assert manifest.series_coverage_stamp == "sha256:coverage"
    finally:
        with live_store.transaction() as tx:
            tx.execute(
                "DELETE FROM datasets.version_record WHERE subject_id = ANY(%s)", (subjects,)
            )
