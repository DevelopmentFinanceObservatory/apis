"""Live smoke tests against a real reachable Postgres, either CI's
ephemeral service container or a developer's local/proxied instance.
Skipped by default. Set DFO_DATASET_STORE_LIVE_DSN
(postgresql://user:pass@host:port/db) to run these.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from urllib.parse import urlsplit
from uuid import uuid4

import pg8000.exceptions
import pytest

from artifact_store import ArtifactNotFoundError
from artifact_store.errors import ArtifactWriteError
from artifact_store.inventory import LiveArtifactInventory
from artifact_store.store import StoredArtifact, Tier
from dataset_store import ConnectionPool, LiveDatasetStore, connect
from dataset_store.migrate import MIGRATIONS_DIR, apply_pending
from dataset_store.snapshot import (
    VERSION_FACT_COLUMNS,
    ArtifactNotFound,
    FrameLineage,
    LiveSnapshotReader,
    SnapshotRevoked,
    VersionFactsMissing,
)
from datasets_api.allowance import (
    AllowanceExhausted,
    AllowanceKind,
    ClaimState,
    LiveAllowanceLedger,
    Spend,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.definition import DefinitionDeps
from datasets_api.doors.pin import pin_artifact
from datasets_api.operation_store import (
    _LIVE,
    _UNTRACKED_LEASE,
    LiveOperationStore,
    OperationBacklogFull,
)
from datasets_api.operations import worker as worker_module
from datasets_api.operations.lease import LEASE_SECONDS
from datasets_api.operations.worker import OperationSuperseded, run_operation
from datasets_api.principals import Principal
from datasets_api.recipe_store import (
    AnswerRecord,
    LiveRecipeStore,
    RecipeRecord,
    ReplayedFrame,
    ReplayKeyHeld,
)
from datasets_api.reference_store import PIN_REFERENCE_KIND, LiveArtifactReferenceWriter
from datasets_api.retention_admission import ExportClaimUnsettled, PreparingCopy
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from datasets_api.versions import (
    ArtifactRecord,
    HeadFence,
    HeadFenceLost,
    SubjectHead,
    VersionRecord,
)
from dfo_platform.caller import Caller
from series_coverage.index import Estimate
from spec_registry import (
    IdentityQuarantined,
    LiveSpecReader,
    LiveSpecRegistry,
    SpecRegistryStoreError,
    recompute_identity,
)
from tests.datasets_api._app_factories import offline_allowances
from tests.datasets_api._legacy_conversion_trace import (
    LEGACY_CONVERSION,
    LEGACY_DEFLATED_CONVERSION,
    MIGRATED_CONVERSION,
    MIGRATED_DEFLATED_CONVERSION,
)
from tests.datasets_api._snapshot_doubles import FakeSpecReader
from tests.datasets_api._version_record_store import materialisation_record
from tests.spec_registry.conftest import minimal_identity, shape_for

_CONFIRMED_AT = datetime(2026, 1, 1, tzinfo=UTC)
_DSN = os.environ.get("DFO_DATASET_STORE_LIVE_DSN")

# The safe-default settings every `LiveOperationStore` below resolves for
# itself when built with no explicit `settings=`.
_DEFAULT_SETTINGS = DatasetsBuildSettings()
_OPERATION_DEADLINE = timedelta(seconds=_DEFAULT_SETTINGS.operation_deadline_seconds)


def _live_params(now: datetime) -> tuple[datetime, datetime, datetime, datetime]:
    """The parameters of `_LIVE` at `now` under the default operation deadline."""
    return (now, now - _UNTRACKED_LEASE, now - _OPERATION_DEADLINE, now - _OPERATION_DEADLINE)


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
            facts=FrameLineage(input_content_versions=[]),
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
            facts=FrameLineage(input_content_versions=[]),
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
            return StoredArtifact(
                digest=digest,
                tier=to_tier,
                bucket="pinned-bucket",
                object_name=digest,
                size_bytes=7,
                deduplicated=False,
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
            allowances=offline_allowances(),
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


@pytest.mark.parametrize(
    ("revoked", "revoked_under_other"),
    [
        ("cv_base", False),
        (None, False),
        ("cv_collapsed", True),
        ("cv_base", True),
    ],
    ids=["base-revoked", "live", "own-revoked-under-other-id", "base-revoked-under-other-id"],
)
def test_read_version_state_reads_the_base_revocation_and_links_in_one_transaction(
    live_store, revoked, revoked_under_other
):
    """A collapsed version reads as revoked when its base version is, or when either is
    revoked under another id the read names (`revoked_under`), beside its linked locations,
    from one `read_version_state` call."""
    subject_id = f"spec:test-live-version-state-{uuid4().hex}"
    other_id = f"spec:test-live-version-state-other-{uuid4().hex}"
    digest = f"sha256:test-live-version-state-{uuid4().hex}"
    with live_store.transaction() as tx:
        for recorded_under in (subject_id, other_id):
            for content_version, base in (("cv_base", None), ("cv_collapsed", "cv_base")):
                tx.execute(
                    "INSERT INTO datasets.version_record "
                    "(subject_id, subject_kind, content_version, schema_version, "
                    "immutable_body, recorded_at, base_content_version) "
                    "VALUES (%s, %s, %s, %s, %s, now(), %s)",
                    (recorded_under, "spec", content_version, 1, _LINEAGE_BODY, base),
                )
        tx.execute(
            "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes) VALUES (%s, %s)",
            (digest, 7),
        )
        tx.execute(
            "INSERT INTO datasets.artifact_location "
            "(artifact_digest, tier, bucket, object_name, stored_at) "
            "VALUES (%s, %s, %s, %s, now())",
            (digest, "working", "b", digest),
        )
        tx.execute(
            "INSERT INTO datasets.version_artifact "
            "(subject_id, content_version, artifact_digest, rendering_key, rendering, "
            "media_type) VALUES (%s, %s, %s, %s, %s, %s)",
            (subject_id, "cv_collapsed", digest, "rk", {"format": "csv"}, "text/csv"),
        )
        if revoked is not None:
            tx.execute(
                "INSERT INTO datasets.version_revocation "
                "(subject_id, content_version, reason, revoked_at) VALUES (%s, %s, %s, now())",
                (other_id if revoked_under_other else subject_id, revoked, "takedown"),
            )
    reader = LiveSnapshotReader(store=live_store)

    state = reader.read_version_state(subject_id, "cv_collapsed", revoked_under=[other_id])

    assert state is not None
    assert state.version.base_content_version == "cv_base"
    assert (state.revocation.reason if state.revocation else None) == (
        "takedown" if revoked else None
    )
    assert [location.artifact_digest for location in state.locations] == [digest]
    assert reader.read_version_state(subject_id, "cv_x") is None
    unnamed = reader.read_version_state(subject_id, "cv_collapsed")
    assert unnamed is not None
    assert (unnamed.revocation is not None) is (revoked is not None and not revoked_under_other)
    assert reader.artifact_inventoried(digest)
    assert not reader.artifact_inventoried(f"sha256:never-stored-{uuid4().hex}")


def test_version_revocation_fk_rejects_an_unknown_version(live_store):
    with pytest.raises(pg8000.exceptions.Error), live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_revocation "
            "(subject_id, content_version, revoked_at) VALUES (%s, %s, now())",
            ("spec:does-not-exist", "cv1"),
        )


def test_an_analysis_answer_is_replaced_under_its_key_and_names_a_stored_version(live_store):
    subject_id = f"ops_{uuid4().hex}"
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at) VALUES (%s, %s, %s, %s, %s, now())",
            (subject_id, "spec", "ocv1", 1, {"prepared_at": "2026-01-01T00:00:00Z"}),
        )
    answers = LiveRecipeStore(store=live_store)
    key = uuid4().hex
    answers.write_answer(
        AnswerRecord(request_key=key, spec_id=subject_id, content_version="ocv1", body={"rows": 1})
    )
    answers.write_answer(
        AnswerRecord(request_key=key, spec_id=subject_id, content_version="ocv1", body={"rows": 2})
    )

    assert answers.read_answer(request_key=key) == AnswerRecord(
        request_key=key, spec_id=subject_id, content_version="ocv1", body={"rows": 2}
    )
    assert answers.read_answer(request_key=uuid4().hex) is None
    with pytest.raises(DatasetsStoreUnavailable):
        answers.write_answer(
            AnswerRecord(
                request_key=uuid4().hex,
                spec_id="ops_does-not-exist",
                content_version="ocv1",
                body={},
            )
        )


def test_a_superseded_recipe_gives_its_replay_key_to_another_pair(live_store):
    """The replay index holds the key once among recipes not superseded, so a second pair
    takes it only after the first gives it up, and the first keeps its recipe."""
    old, new = f"ops_{uuid4().hex}", f"ops_{uuid4().hex}"
    with live_store.transaction() as tx:
        for subject_id in (old, new):
            tx.execute(
                "INSERT INTO datasets.version_record "
                "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
                "recorded_at) VALUES (%s, %s, %s, %s, %s, now())",
                (subject_id, "spec", "ocv1", 1, {"prepared_at": "2026-01-01T00:00:00Z"}),
            )
    recipes = LiveRecipeStore(store=live_store)
    replay_key = f"rpk_{uuid4().hex}"

    def record(subject_id: str) -> RecipeRecord:
        return RecipeRecord(
            spec_id=subject_id,
            content_version="ocv1",
            normal_form_hash="rnf_" + "0" * 32,
            replay_key=replay_key,
            body={"replay_key": replay_key},
        )

    assert recipes.write_recipe(record(old))
    with pytest.raises(ReplayKeyHeld):
        recipes.write_recipe(record(new))
    recipes.supersede_replay(replay_key=replay_key, spec_id=old, content_version="ocv1")

    assert recipes.write_recipe(record(new))
    assert not recipes.write_recipe(record(old))
    assert recipes.read_replay(replay_key=replay_key) == ReplayedFrame(
        spec_id=new, content_version="ocv1"
    )
    assert recipes.read_recipe(spec_id=old, content_version="ocv1") == record(old)


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
    """After `cv2` advances the head from the `cv0` run, a writer expecting that run cannot
    write `cv1`."""
    subject_id = "spec:test-live-head-forward"
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv0")
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv1")
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv2")

    store = LiveHeadStore(store=live_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    observed = store.advance_head(
        materialisation=materialisation_record(subject_id=subject_id, content_version="cv0"),
        expected_materialisation_id=None,
    )
    assert observed is not None

    won = store.advance_head(
        materialisation=materialisation_record(subject_id=subject_id, content_version="cv2"),
        expected_materialisation_id=observed.materialisation_id,
    )
    lost = store.advance_head(
        materialisation=materialisation_record(subject_id=subject_id, content_version="cv1"),
        expected_materialisation_id=observed.materialisation_id,
    )

    assert won is not None
    assert lost is None
    head = store.read_head(subject_id)
    assert head is not None
    assert head.content_version == "cv2"
    assert (
        head.dependencies
        == materialisation_record(subject_id=subject_id, content_version="cv2").dependencies
    )


def test_live_head_store_advance_inserts_the_run_the_head_names(live_store):
    subject_id = "spec:test-live-head-materialisation"
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv1")

    head = LiveHeadStore(store=live_store).advance_head(
        materialisation=materialisation_record(
            subject_id=subject_id, content_version="cv1", trigger="worker"
        ),
        expected_materialisation_id=None,
    )

    assert head is not None
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            'SELECT subject_id, content_version, "trigger", dependencies, materialised_at '
            "FROM datasets.materialisation WHERE id = %s",
            (head.materialisation_id,),
        )
        signal = tx.fetch_one(
            "SELECT signal FROM datasets.subject_head WHERE subject_id = %s", (subject_id,)
        )
    assert row == {
        "subject_id": subject_id,
        "content_version": "cv1",
        "trigger": "worker",
        "dependencies": head.dependencies,
        "materialised_at": head.materialised_at,
    }
    # The store writes no signal, so the column keeps its default.
    assert signal == {"signal": {"members": []}}


def test_a_head_written_before_runs_were_recorded_reads_with_no_dependencies(live_store):
    subject_id = "spec:test-live-head-no-signal"
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv1")
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.subject_head "
            "(subject_id, content_version, confirmed_at, materialised_at) "
            "VALUES (%s, %s, now(), now())",
            (subject_id, "cv1"),
        )

    head = LiveHeadStore(store=live_store).read_head(subject_id)

    assert head is not None
    assert head.materialisation_id is None
    assert head.dependencies is None


def test_a_head_whose_run_made_another_version_reads_with_no_dependencies(live_store):
    """A writer that moves `content_version` alone leaves the head naming a run of another
    version, so the head read drops that run's fingerprint."""
    subject_id = "spec:test-live-head-foreign-run"
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv1")
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv2")
    store = LiveHeadStore(store=live_store)
    written = store.advance_head(
        materialisation=materialisation_record(subject_id=subject_id, content_version="cv1"),
        expected_materialisation_id=None,
    )
    assert written is not None and written.dependencies is not None
    with live_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.subject_head SET content_version = %s WHERE subject_id = %s",
            ("cv2", subject_id),
        )

    head = store.read_head(subject_id)

    assert head is not None
    assert (head.content_version, head.materialisation_id) == ("cv2", written.materialisation_id)
    assert head.dependencies is None


def test_live_head_store_read_head_returns_none_for_an_unknown_subject(live_store):
    store = LiveHeadStore(store=live_store)
    assert store.read_head("spec:does-not-exist-at-all") is None


def test_live_head_store_two_racing_writers_agree_on_exactly_one_winner(live_store):
    """Two writers share `cv-seed`; exactly one advances the head.

    A barrier releases both threads together so their writes overlap. The stored head matches the
    winner's version and dependencies.
    """
    subject_id = "spec:test-live-head-race"
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv-seed")
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv-a")
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv-b")
    seed = LiveHeadStore(
        store=live_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    ).advance_head(
        materialisation=materialisation_record(subject_id=subject_id, content_version="cv-seed"),
        expected_materialisation_id=None,
    )
    assert seed is not None

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    results: dict[str, SubjectHead | None] = {}

    def _write(*, content_version: str) -> None:
        try:
            barrier.wait(timeout=5)
            results[content_version] = LiveHeadStore(
                store=live_store, now=lambda: datetime(2026, 6, 1, tzinfo=UTC)
            ).advance_head(
                materialisation=materialisation_record(
                    subject_id=subject_id, content_version=content_version
                ),
                expected_materialisation_id=seed.materialisation_id,
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
    assert (
        head.dependencies
        == materialisation_record(subject_id=subject_id, content_version=winners[0]).dependencies
    )


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
            f"SELECT count(*) AS n FROM datasets.operation WHERE {_LIVE}",
            _live_params(now),
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
    setup_store.claim(target.id, submission_count=0, lease_seconds=60, final_delivery=False)
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


class _InterleavingTransaction:
    """Commits `before` and `after` on another connection around the first reset update."""

    def __init__(self, inner, store: _InterleavingDatasetStore) -> None:
        self._inner = inner
        self._store = store

    def _around_reset(self, sql: str, run):
        hooks = self._store.pending_hooks
        if hooks is None or not sql.startswith("UPDATE datasets.operation SET state = 'pending'"):
            return run()
        self._store.pending_hooks = None
        before, after = hooks
        before()
        result = run()
        after()
        return result

    def execute(self, sql: str, params=()) -> None:
        self._around_reset(sql, lambda: self._inner.execute(sql, params))

    def fetch_one(self, sql: str, params=()):
        return self._around_reset(sql, lambda: self._inner.fetch_one(sql, params))

    def fetch_all(self, sql: str, params=()):
        return self._around_reset(sql, lambda: self._inner.fetch_all(sql, params))


class _InterleavingDatasetStore:
    """Wraps each real transaction in `_InterleavingTransaction`."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.pending_hooks = None

    @contextmanager
    def transaction(self):
        with self._inner.transaction() as tx:
            yield _InterleavingTransaction(tx, self)


def test_operation_store_create_or_get_decides_again_when_its_reset_matches_nothing(live_store):
    """A redelivery claims the abandoned row after the locked read and commits, so the reset
    re-checks the new version and matches nothing. The run completes before the call returns.
    Under READ COMMITTED a final read would see `done`. The call resets the done row instead."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    run_id = uuid4().hex
    subject_id = f"spec-test-live-operation-redecide-{run_id}"
    identity = {"input": {"spec_id": subject_id}}
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv1")
    interleaving = _InterleavingDatasetStore(live_store)
    store = LiveOperationStore(store=interleaving, now=lambda: now_box["t"], backlog_max=10_000)
    # The redelivery commits on its own connection while the submission's transaction is open.
    worker_pool = _pool(size=1)
    worker = LiveOperationStore(
        store=LiveDatasetStore(connections=worker_pool), now=lambda: now_box["t"]
    )

    try:

        def _submit():
            return store.create_or_get(
                request_identity=identity,
                options={"format": "parquet"},
                coverage_stamp=f"cov-test-live-operation-redecide-{run_id}",
                origin_catalogue_build_id="fake-catalogue",
                estimate=Estimate(rows=1, requests=1),
            )

        first = _submit()
        assert worker.claim(first.id, submission_count=0, lease_seconds=60, final_delivery=True)
        now_box["t"] += timedelta(seconds=61)

        def _redelivery_claims() -> None:
            assert worker.claim(first.id, submission_count=0, lease_seconds=60, final_delivery=True)

        def _redelivery_completes() -> None:
            assert worker.complete(
                first.id,
                attempt=2,
                subject_id=subject_id,
                content_version="cv1",
                confirmed_at=now_box["t"],
            )

        interleaving.pending_hooks = (_redelivery_claims, _redelivery_completes)

        resubmitted = _submit()

        assert interleaving.pending_hooks is None
        assert (resubmitted.state, resubmitted.submission_count) == ("pending", 1)
        assert worker.get(first.id) == resubmitted
    finally:
        worker_pool.close_all()


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


def _wait_until_an_operation_update_is_blocked(
    live_store, *, statement: str = "UPDATE datasets.operation"
) -> None:
    """Wait until another session blocks on a lock in a statement starting with `statement`."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with live_store.transaction() as tx:
            row = tx.fetch_one(
                "SELECT count(*) AS n FROM pg_stat_activity "
                "WHERE wait_event_type = 'Lock' AND starts_with(query, %s)",
                (statement,),
            )
        if row is not None and row["n"] > 0:
            return
        time.sleep(0.02)
    raise AssertionError(f"no {statement} ever blocked on the held row")


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
    first = setup.claim(op.id, submission_count=0, lease_seconds=30, final_delivery=False)
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
            lambda: reclaimer.claim(
                op.id, submission_count=0, lease_seconds=1_500, final_delivery=False
            )
        )
        assert holder.held.wait(timeout=5), "the reclaim never reached its commit"
        complete = _Call(
            lambda: stale_worker.complete(
                op.id,
                attempt=first.attempt_count,
                subject_id="spec-test-live-fence",
                content_version="cv-never-written",
                confirmed_at=_CONFIRMED_AT,
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


def test_operation_store_complete_keeps_the_runs_confirmed_instant(live_store):
    """`complete` stores the instant the run confirmed the subject head, apart from the
    completion's own `updated_at`, and `get` reads it back as that instant."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    operations = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(operations, run_id, "confirmed")
    claimed = operations.claim(op.id, submission_count=0, lease_seconds=30, final_delivery=False)
    assert claimed is not None and claimed.confirmed_at is None
    confirmed_at = t0 - timedelta(seconds=7, microseconds=250)
    subject_id = f"spec-test-live-confirmed-{run_id}"
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv1")

    done = operations.complete(
        op.id,
        attempt=claimed.attempt_count,
        subject_id=subject_id,
        content_version="cv1",
        confirmed_at=confirmed_at,
    )

    assert done is not None
    assert (done.confirmed_at, done.updated_at) == (confirmed_at, t0)
    stored = operations.get(op.id)
    assert stored is not None and stored.confirmed_at == confirmed_at


def test_operation_store_create_or_get_resets_a_done_row(live_store):
    """The reset `UPDATE`'s `_RESETTABLE` admits a done row on PostgreSQL, so it returns to
    pending under a new submission and its next claim takes a new attempt."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    operations = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(operations, run_id, "done-reset")
    claimed = operations.claim(op.id, submission_count=0, lease_seconds=30, final_delivery=False)
    assert claimed is not None
    subject_id = f"spec-test-live-done-reset-{run_id}"
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv1")
    assert operations.complete(
        op.id,
        attempt=claimed.attempt_count,
        subject_id=subject_id,
        content_version="cv1",
        confirmed_at=t0,
    )

    resubmitted = operations.create_or_get(
        request_identity=op.request_identity,
        options=op.options,
        coverage_stamp=op.coverage_stamp,
        origin_catalogue_build_id="fake-catalogue",
        estimate=op.estimate,
    )

    assert (resubmitted.id, resubmitted.state, resubmitted.submission_count) == (
        op.id,
        "pending",
        1,
    )
    reclaimed = operations.claim(op.id, submission_count=1, lease_seconds=30, final_delivery=False)
    assert reclaimed is not None
    assert reclaimed.attempt_count == claimed.attempt_count + 1


def _materialisation_count(live_store, subject_id: str) -> int:
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT count(*) AS n FROM datasets.materialisation WHERE subject_id = %s",
            (subject_id,),
        )
    assert row is not None
    return cast(int, row["n"])


def test_live_head_store_a_superseded_attempts_fenced_advance_writes_nothing(live_store):
    """Attempt 1 advances the head while it owns the operation. Once attempt 2 reclaims the
    row, attempt 1's fenced advance refuses and leaves the head and the materialisation table
    unchanged, and attempt 2's advances the head."""
    run_id = uuid4().hex
    subject_id = f"spec:test-live-head-fence-{run_id}"
    for content_version in ("cv0", "cv-stale", "cv-new"):
        _insert_version_record(live_store, subject_id=subject_id, content_version=content_version)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    operations = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(operations, run_id, "head")
    first = operations.claim(op.id, submission_count=0, lease_seconds=30, final_delivery=False)
    assert first is not None
    first_fence = HeadFence(operation_id=op.id, attempt=first.attempt_count)
    heads = LiveHeadStore(store=live_store)
    seeded = heads.advance_head(
        materialisation=materialisation_record(subject_id=subject_id, content_version="cv0"),
        expected_materialisation_id=None,
        fence=first_fence,
    )
    assert seeded is not None
    reclaimed = LiveOperationStore(store=live_store, now=lambda: t0 + timedelta(seconds=31)).claim(
        op.id, submission_count=0, lease_seconds=1_500, final_delivery=False
    )
    assert reclaimed is not None

    with pytest.raises(HeadFenceLost):
        heads.advance_head(
            materialisation=materialisation_record(
                subject_id=subject_id, content_version="cv-stale"
            ),
            expected_materialisation_id=seeded.materialisation_id,
            fence=first_fence,
        )

    assert heads.read_head(subject_id) == seeded
    assert _materialisation_count(live_store, subject_id) == 1
    advanced = heads.advance_head(
        materialisation=materialisation_record(subject_id=subject_id, content_version="cv-new"),
        expected_materialisation_id=seeded.materialisation_id,
        fence=HeadFence(operation_id=op.id, attempt=reclaimed.attempt_count),
    )
    assert advanced is not None
    assert heads.read_head(subject_id) == advanced
    assert advanced.content_version == "cv-new"
    assert _materialisation_count(live_store, subject_id) == 2


def test_live_head_store_a_fenced_advance_waits_on_an_uncommitted_reclaim_and_refuses(
    live_store,
):
    """Attempt 2's reclaim holds the operation row, uncommitted, when attempt 1 advances the
    head. The advance's share lock waits, re-checks the committed reclaim, and refuses."""
    run_id = uuid4().hex
    subject_id = f"spec:test-live-head-fence-wait-{run_id}"
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv-stale")
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    setup = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(setup, run_id, "head-wait")
    first = setup.claim(op.id, submission_count=0, lease_seconds=30, final_delivery=False)
    assert first is not None

    t1 = t0 + timedelta(seconds=31)
    reclaim_pool = _pool(size=1)
    head_pool = _pool(size=1)
    try:
        holder = _HoldBeforeCommitStore(LiveDatasetStore(connections=reclaim_pool))
        reclaimer = LiveOperationStore(store=holder, now=lambda: t1)
        heads = LiveHeadStore(store=LiveDatasetStore(connections=head_pool))

        reclaim = _Call(
            lambda: reclaimer.claim(
                op.id, submission_count=0, lease_seconds=1_500, final_delivery=False
            )
        )
        assert holder.held.wait(timeout=5), "the reclaim never reached its commit"
        advance = _Call(
            lambda: heads.advance_head(
                materialisation=materialisation_record(
                    subject_id=subject_id, content_version="cv-stale"
                ),
                expected_materialisation_id=None,
                fence=HeadFence(operation_id=op.id, attempt=first.attempt_count),
            )
        )
        _wait_until_an_operation_update_is_blocked(
            live_store, statement="SELECT 1 AS held FROM datasets.operation"
        )
        holder.release.set()

        reclaimed = reclaim.join()
        with pytest.raises(HeadFenceLost):
            advance.join()
    finally:
        reclaim_pool.close_all()
        head_pool.close_all()

    assert reclaimed is not None
    assert reclaimed.attempt_count == first.attempt_count + 1
    assert LiveHeadStore(store=live_store).read_head(subject_id) is None
    assert _materialisation_count(live_store, subject_id) == 0


def test_live_head_store_a_reclaim_waits_on_an_uncommitted_fenced_advance(live_store):
    """Attempt 1's fenced advance holds its share lock, uncommitted, when attempt 2 reclaims
    the row. The reclaim waits until the advance commits, so the head names attempt 1's run
    only while attempt 1 owned the row."""
    run_id = uuid4().hex
    subject_id = f"spec:test-live-head-fence-claim-{run_id}"
    _insert_version_record(live_store, subject_id=subject_id, content_version="cv1")
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    setup = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(setup, run_id, "head-claim")
    first = setup.claim(op.id, submission_count=0, lease_seconds=30, final_delivery=False)
    assert first is not None

    t1 = t0 + timedelta(seconds=31)
    head_pool = _pool(size=1)
    reclaim_pool = _pool(size=1)
    try:
        holder = _HoldBeforeCommitStore(
            LiveDatasetStore(connections=head_pool),
            hold_when=lambda sql: sql.startswith("INSERT INTO datasets.subject_head"),
        )
        heads = LiveHeadStore(store=holder)
        reclaimer = LiveOperationStore(
            store=LiveDatasetStore(connections=reclaim_pool), now=lambda: t1
        )

        advance = _Call(
            lambda: heads.advance_head(
                materialisation=materialisation_record(
                    subject_id=subject_id, content_version="cv1"
                ),
                expected_materialisation_id=None,
                fence=HeadFence(operation_id=op.id, attempt=first.attempt_count),
            )
        )
        assert holder.held.wait(timeout=5), "the advance never reached its commit"
        reclaim = _Call(
            lambda: reclaimer.claim(
                op.id, submission_count=0, lease_seconds=1_500, final_delivery=False
            )
        )
        _wait_until_an_operation_update_is_blocked(live_store)
        assert not reclaim.done()
        holder.release.set()

        advanced = advance.join()
        reclaimed = reclaim.join()
    finally:
        head_pool.close_all()
        reclaim_pool.close_all()

    assert advanced is not None
    assert LiveHeadStore(store=live_store).read_head(subject_id) == advanced
    assert reclaimed is not None
    assert reclaimed.attempt_count == first.attempt_count + 1


def test_operation_store_claim_with_an_old_submission_count_is_refused(live_store):
    """A resubmit reset the row to submission 1. A delivery of submission
    0's task, on its own connection, cannot claim it; submission 1's task
    can."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    setup = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(setup, run_id, "submission")
    first = setup.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
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

        assert (
            old_delivery.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False)
            is None
        )
        claimed = new_delivery.claim(
            op.id, submission_count=1, lease_seconds=60, final_delivery=False
        )
    finally:
        old_pool.close_all()
        new_pool.close_all()

    assert claimed is not None
    assert claimed.submission_count == 1
    assert claimed.attempt_count == first.attempt_count + 1


def _counts_in_the_backlog(live_store, operation_id: str, at: datetime) -> bool:
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            f"SELECT count(*) AS n FROM datasets.operation WHERE id = %s AND {_LIVE}",
            (operation_id, *_live_params(at)),
        )
    return row is not None and row["n"] == 1


@pytest.mark.parametrize("final_delivery", [True, False], ids=["final", "delivery-due"])
def test_operation_store_a_run_past_its_lease_leaves_the_backlog_once_no_delivery_is_due(
    live_store, final_delivery: bool
):
    """The backlog counts a running row inside `claimed_at + lease_seconds`, the interval
    arithmetic `make_interval` does in Postgres. Past the lease it counts the row only while a
    later delivery is due, and `abandon` leaves that row until the operation deadline. Past the
    deadline `claim` refuses it."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    now_box = {"t": t0}
    store = LiveOperationStore(
        store=live_store, now=lambda: now_box["t"], backlog_max=_RACE_BACKLOG_MAX
    )
    op = _new_operation(store, run_id, "backlog")
    claimed = store.claim(
        op.id, submission_count=0, lease_seconds=60, final_delivery=final_delivery
    )
    assert claimed is not None and claimed.final_delivery is final_delivery

    assert _counts_in_the_backlog(live_store, op.id, t0 + timedelta(seconds=60))
    past_lease = t0 + timedelta(seconds=61)
    assert _counts_in_the_backlog(live_store, op.id, past_lease) is not final_delivery

    now_box["t"] = past_lease
    abandoned = store.abandon(op.id, attempt=1)
    if final_delivery:
        assert abandoned is not None and abandoned.state == "failed"
        return
    assert abandoned is None
    at_deadline = t0 + _OPERATION_DEADLINE
    assert _counts_in_the_backlog(live_store, op.id, at_deadline)
    now_box["t"] = at_deadline + timedelta(seconds=1)
    assert not _counts_in_the_backlog(live_store, op.id, now_box["t"])
    assert store.claim(op.id, submission_count=0, lease_seconds=60, final_delivery=False) is None
    abandoned = store.abandon(op.id, attempt=1)
    assert abandoned is not None and abandoned.state == "failed"


def test_operation_store_two_racing_deliveries_have_one_claim(live_store):
    """A second delivery blocks on the first claim's row lock, re-evaluates against the
    committed claim, and finds a live lease."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    setup = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(setup, run_id, "deliveries")

    first_pool = _pool(size=1)
    second_pool = _pool(size=1)
    try:
        holder = _HoldBeforeCommitStore(LiveDatasetStore(connections=first_pool))
        first = LiveOperationStore(store=holder, now=lambda: t0)
        second = LiveOperationStore(store=LiveDatasetStore(connections=second_pool), now=lambda: t0)

        first_claim = _Call(
            lambda: first.claim(
                op.id, submission_count=0, lease_seconds=1_500, final_delivery=False
            )
        )
        assert holder.held.wait(timeout=5), "the first claim never reached its commit"
        second_claim = _Call(
            lambda: second.claim(
                op.id, submission_count=0, lease_seconds=1_500, final_delivery=False
            )
        )
        _wait_until_an_operation_update_is_blocked(live_store)
        holder.release.set()

        won = first_claim.join()
        assert second_claim.join() is None
    finally:
        first_pool.close_all()
        second_pool.close_all()

    assert won is not None and won.attempt_count == 1
    assert setup.get(op.id).attempt_count == 1


def test_operation_store_a_poll_abandon_racing_a_reclaim_leaves_the_reclaim(live_store):
    """A delivery reclaims an ended lease while a poll abandons the same attempt. The abandon
    blocks on the reclaim's row lock, re-evaluates its attempt guard, and matches nothing."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    setup = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(setup, run_id, "abandon")
    # A final delivery's ended lease is abandoned at once, so the poll's update matches the row
    # it read and waits on the reclaim's lock.
    first = setup.claim(op.id, submission_count=0, lease_seconds=30, final_delivery=True)
    assert first is not None

    t1 = t0 + timedelta(seconds=31)
    reclaim_pool = _pool(size=1)
    poll_pool = _pool(size=1)
    try:
        holder = _HoldBeforeCommitStore(LiveDatasetStore(connections=reclaim_pool))
        reclaimer = LiveOperationStore(store=holder, now=lambda: t1)
        poller = LiveOperationStore(store=LiveDatasetStore(connections=poll_pool), now=lambda: t1)

        reclaim = _Call(
            lambda: reclaimer.claim(
                op.id, submission_count=0, lease_seconds=1_500, final_delivery=False
            )
        )
        assert holder.held.wait(timeout=5), "the reclaim never reached its commit"
        abandon = _Call(lambda: poller.abandon(op.id, attempt=first.attempt_count))
        _wait_until_an_operation_update_is_blocked(live_store)
        holder.release.set()

        reclaimed = reclaim.join()
        assert abandon.join() is None
    finally:
        reclaim_pool.close_all()
        poll_pool.close_all()

    assert reclaimed is not None
    current = setup.get(op.id)
    assert current is not None
    assert (current.state, current.attempt_count, current.claimed_at) == ("running", 2, t1)


def test_operation_store_records_one_dispatch_per_submission(live_store):
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    store = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _new_operation(store, run_id, "dispatch")
    assert op.dispatched_at is None

    marked = store.mark_dispatched(op.id, submission_count=0)

    assert marked is not None and marked.dispatched_at == t0
    assert store.mark_dispatched(op.id, submission_count=0) is None
    assert store.mark_dispatched(op.id, submission_count=1) is None


class _StubCoverageSnapshot:
    stamp = "cov-test-live-execution"
    index = object()


class _StubCoverage:
    def snapshot(self):
        return _StubCoverageSnapshot()


def _worker_deps() -> DefinitionDeps:
    """Build minimal worker dependencies for the patched resolve and estimate calls."""
    return DefinitionDeps(
        registry=None,  # type: ignore[arg-type]
        spec_reader=None,  # type: ignore[arg-type]
        graph=None,  # type: ignore[arg-type]
        sdmx=None,  # type: ignore[arg-type]
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


_PLAIN_DEFINITION = SimpleNamespace(identity={})
"""A stored definition whose identity binds no composite, as the worker reads one."""


def _must_not_serve(*_, **__):
    raise AssertionError("an over-ceiling or superseded run must not materialise")


def test_worker_fails_over_ceiling_when_the_execution_estimate_breaches_it(live_store, monkeypatch):
    """An execution estimate above the row limit fails without serving data."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    store = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _subject_operation(store, run_id)
    claimed = store.claim(op.id, submission_count=0, lease_seconds=600, final_delivery=False)
    assert claimed is not None
    over = Estimate(rows=DatasetsBuildSettings().async_max_rows + 1, requests=1)
    monkeypatch.setattr(
        worker_module, "_resolve_spec_requests", lambda **_: (_PLAIN_DEFINITION, [])
    )
    monkeypatch.setattr(
        worker_module,
        "sync_estimate",
        lambda requests, *, index, member_concurrency, composite: over,
    )
    monkeypatch.setattr(worker_module, "serve_current", _must_not_serve)

    run_operation(
        operation=claimed,
        deps=_worker_deps(),
        operation_store=store,
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
    assert stored.estimate == over


def test_worker_a_stale_attempt_cannot_fail_the_reclaimed_row_over_ceiling(live_store, monkeypatch):
    """A reclaimed attempt cannot record an over-ceiling failure against the new lease."""
    run_id = uuid4().hex
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    setup = LiveOperationStore(store=live_store, now=lambda: t0, backlog_max=_RACE_BACKLOG_MAX)
    op = _subject_operation(setup, run_id)
    first = setup.claim(op.id, submission_count=0, lease_seconds=30, final_delivery=False)
    assert first is not None
    over = Estimate(rows=DatasetsBuildSettings().async_max_rows + 1, requests=1)
    monkeypatch.setattr(
        worker_module, "_resolve_spec_requests", lambda **_: (_PLAIN_DEFINITION, [])
    )
    monkeypatch.setattr(
        worker_module,
        "sync_estimate",
        lambda requests, *, index, member_concurrency, composite: over,
    )
    monkeypatch.setattr(worker_module, "serve_current", _must_not_serve)

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
            lambda: reclaimer.claim(
                op.id, submission_count=0, lease_seconds=1_500, final_delivery=False
            )
        )
        assert holder.held.wait(timeout=5), "the reclaim never reached its commit"
        stale_run = _Call(
            lambda: run_operation(
                operation=first,
                deps=_worker_deps(),
                operation_store=stale_store,
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
    assert current.estimate == op.estimate


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


# The body of a seeded version nothing reads back. It holds a lineage, so the row stores facts.
_LINEAGE_BODY = {"input_content_versions": ["cv_input"]}


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
            (subject_id, "spec", "cv1", 1, _LINEAGE_BODY),
        )
        tx.execute(
            "INSERT INTO datasets.version_artifact "
            "(subject_id, content_version, artifact_digest, rendering_key, rendering, "
            "media_type) VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (subject_id, content_version, rendering_key) DO NOTHING",
            (subject_id, "cv1", digest, f"rk-{digest}", {"format": "csv"}, "text/csv"),
        )


def test_a_replacement_artifact_takes_every_reference_of_the_digest_it_replaces(live_store):
    """`record_replacement_artifact` links the new digest and copies each reference row the
    version holds on the replaced digest, live or removed, in one Postgres statement. A rerun
    changes nothing."""
    suffix = uuid4().hex
    subject_id = f"spec:test-live-replacement-{suffix}"
    replaced = f"sha256:test-live-replaced-{suffix}"
    replacement = f"sha256:test-live-replacement-{suffix}"
    _seed_reference_fk_targets(live_store, digest=replaced, subject_id=subject_id)
    earlier = datetime(2026, 1, 1, tzinfo=UTC)
    later = earlier + timedelta(days=2)
    seeded = [
        (f"secret-sha256:{suffix[:16]}", PIN_REFERENCE_KIND, earlier, None),
        (f"presentation:chart-{suffix}", "presentation", earlier, later),
    ]
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes) VALUES (%s, %s)",
            (replacement, 7),
        )
        for principal, kind, created_at, removed_at in seeded:
            tx.execute(
                "INSERT INTO datasets.artifact_reference (principal, artifact_digest, "
                "subject_id, content_version, reference_kind, created_at, removed_at) "
                "VALUES (%s, %s, %s, 'cv1', %s, %s, %s)",
                (principal, replaced, subject_id, kind, created_at, removed_at),
            )

    def _references(digest: str) -> list[tuple[object, ...]]:
        with live_store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT principal, reference_kind, created_at, removed_at "
                "FROM datasets.artifact_reference WHERE artifact_digest = %s "
                "AND subject_id = %s AND content_version = 'cv1' ORDER BY principal",
                (digest, subject_id),
            )
        return [
            (row["principal"], row["reference_kind"], row["created_at"], row["removed_at"])
            for row in rows
        ]

    writer = LiveVersionRecordWriter(store=live_store)
    try:
        for _ in range(2):
            writer.record_replacement_artifact(
                subject_id=subject_id,
                content_version="cv1",
                artifact=ArtifactRecord(
                    artifact_digest=replacement,
                    rendering={"format": "frame"},
                    media_type="application/vnd.apache.parquet",
                ),
                replaces=replaced,
            )
            assert _references(replacement) == sorted(seeded)
            assert _references(replaced) == sorted(seeded)
        with live_store.transaction() as tx:
            links = tx.fetch_all(
                "SELECT artifact_digest FROM datasets.version_artifact "
                "WHERE subject_id = %s ORDER BY artifact_digest",
                (subject_id,),
            )
        assert sorted(row["artifact_digest"] for row in links) == sorted([replaced, replacement])
    finally:
        with live_store.transaction() as tx:
            tx.execute(
                "DELETE FROM datasets.artifact_reference WHERE subject_id = %s", (subject_id,)
            )
            tx.execute("DELETE FROM datasets.version_artifact WHERE subject_id = %s", (subject_id,))
            tx.execute("DELETE FROM datasets.version_record WHERE subject_id = %s", (subject_id,))


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


@pytest.mark.parametrize("together", [False, True], ids=["allowance", "budget-and-allowance"])
def test_concurrent_allowance_spends_stop_at_the_limit(live_store, together: bool):
    """Spends racing on one caller's day pass the limit by none, under Postgres row locking. A
    pin's budget and allowance spent together stop at the tighter limit, and a refused pin
    spends neither."""
    caller = Caller(kind="address", key="ip:" + uuid4().hex)
    budget = Spend(f"test-budget-{uuid4().hex}", AllowanceKind.open_pin_budget)
    ledger = LiveAllowanceLedger(
        store=live_store, limits={AllowanceKind.pin: 3, AllowanceKind.open_pin_budget: 3}
    )
    now = datetime.now(UTC)
    outcomes: list[str] = []

    def spend() -> None:
        try:
            if together:
                ledger.spend_together([budget, Spend(caller.key, AllowanceKind.pin)], now)
            else:
                ledger.spend(caller, AllowanceKind.pin, now)
            outcomes.append("spent")
        except AllowanceExhausted:
            outcomes.append("refused")

    threads = [threading.Thread(target=spend) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert sorted(outcomes) == ["refused"] * 3 + ["spent"] * 3
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT day, count FROM datasets.caller_allowance WHERE caller_key = %s",
            (caller.key,),
        )
    assert row == {"day": now.date(), "count": 3}
    if together:
        with live_store.transaction() as tx:
            spent = tx.fetch_one(
                "SELECT count FROM datasets.caller_allowance WHERE caller_key = %s",
                (budget.key,),
            )
        assert spent == {"count": 3}


def test_a_card_unit_refund_takes_one_back_and_deletes_a_day_left_at_zero(live_store):
    caller = Caller(kind="address", key="ip:" + uuid4().hex)
    ledger = LiveAllowanceLedger(store=live_store, limits={AllowanceKind.card: 2})
    now = datetime.now(UTC)

    def held() -> object:
        with live_store.transaction() as tx:
            return tx.fetch_one(
                "SELECT count FROM datasets.caller_allowance WHERE caller_key = %s AND kind = %s",
                (caller.key, AllowanceKind.card.value),
            )

    ledger.spend(caller, AllowanceKind.card, now)
    ledger.spend(caller, AllowanceKind.card, now)
    ledger.refund(caller, AllowanceKind.card, now)
    after_one = held()
    ledger.refund(caller, AllowanceKind.card, now)
    ledger.refund(caller, AllowanceKind.card, now)

    assert after_one == {"count": 1}
    assert held() is None


_CLAIMED_AT = datetime(2026, 6, 1, 12, 0, 0, 123456, tzinfo=UTC)
_CLAIM_LEASE = _CLAIMED_AT + timedelta(minutes=2)
_AFTER_LEASE = _CLAIM_LEASE + timedelta(seconds=1)


def _claim_rig(live_store, *, limit: int = 2):
    """A ledger, an export id and one unit of a caller of its own, so runs do not meet."""
    suffix = uuid4().hex
    ledger = LiveAllowanceLedger(store=live_store, limits={AllowanceKind.pin: limit})
    return ledger, f"exp_{suffix}", Spend(f"ip:{suffix}", AllowanceKind.pin)


def _held_claim(ledger, export, spend):
    claim = ledger.claim(export, now=_CLAIMED_AT, expires_at=_CLAIM_LEASE)
    return ledger.hold_claim(claim, [spend], _CLAIMED_AT)


def _unit_count(live_store, spend) -> int:
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT count FROM datasets.caller_allowance WHERE caller_key = %s AND kind = %s",
            (spend.key, spend.kind.value),
        )
    return 0 if row is None else row["count"]


def test_an_export_claim_holds_units_and_refunds_them_once_under_real_postgres(live_store):
    """`spends` round-trips as jsonb, and a limit reads it with `->>` in its own transaction."""
    ledger, export, spend = _claim_rig(live_store, limit=1)

    held = _held_claim(ledger, export, spend)

    assert held.state == ClaimState.held
    assert held.spends == (spend,)
    assert ledger.claim(export, now=_CLAIMED_AT, expires_at=_CLAIM_LEASE) is None
    with pytest.raises(AllowanceExhausted) as refused:
        ledger.spend_together([spend], _CLAIMED_AT)
    assert refused.value.held is True
    ledger.close_claim(held)
    ledger.close_claim(held)
    assert ledger.read_claim(export) is None
    assert _unit_count(live_store, spend) == 0


def test_a_sweep_behind_a_hold_changes_nothing_under_real_postgres(live_store):
    ledger, export, spend = _claim_rig(live_store)
    opened = ledger.claim(export, now=_CLAIMED_AT, expires_at=_CLAIM_LEASE)
    swept = next(c for c in ledger.expired_claims(_AFTER_LEASE) if c.export_id == export)

    held = ledger.hold_claim(opened, [spend], _AFTER_LEASE)
    ledger.close_claim(swept)

    assert ledger.read_claim(export) == held
    assert _unit_count(live_store, spend) == 1


@pytest.mark.parametrize("landed", [False, True], ids=["copy-running", "copy-landed"])
def test_a_sweep_behind_a_copy_refunds_nothing_under_real_postgres(live_store, landed: bool):
    """The sweep read the claim before the copy began, so its bucket read says nothing."""
    ledger, export, spend = _claim_rig(live_store)
    held = _held_claim(ledger, export, spend)
    swept = next(c for c in ledger.expired_claims(_AFTER_LEASE) if c.export_id == export)

    current = ledger.begin_copy(held, "sha256:a", now=_CLAIMED_AT, expires_at=_CLAIM_LEASE)
    if landed:
        current = ledger.settle_copy(current, landed=True)
    ledger.close_claim(swept)

    assert ledger.read_claim(export) == current
    assert _unit_count(live_store, spend) == 1


def test_a_sweep_refunds_a_copying_claim_once_against_a_settling_holder_under_real_postgres(
    live_store,
):
    ledger, export, spend = _claim_rig(live_store)
    copying = ledger.begin_copy(
        _held_claim(ledger, export, spend), "sha256:a", now=_CLAIMED_AT, expires_at=_CLAIM_LEASE
    )
    swept = next(c for c in ledger.expired_claims(_AFTER_LEASE) if c.export_id == export)

    ledger.settle_copy(copying, landed=False)
    ledger.close_claim(swept)
    ledger.settle_copy(copying, landed=False)

    assert ledger.read_claim(export).state == ClaimState.open
    assert _unit_count(live_store, spend) == 0


def test_a_takeover_by_another_caller_refunds_and_fences_the_first_under_real_postgres(
    live_store,
):
    ledger, export, spend = _claim_rig(live_store)
    other = Spend(f"ip:{uuid4().hex}", AllowanceKind.pin)
    first = _held_claim(ledger, export, spend)

    taken = ledger.claim(export, now=_AFTER_LEASE, expires_at=_AFTER_LEASE + timedelta(minutes=2))
    second = ledger.hold_claim(taken, [other], _AFTER_LEASE)

    assert _unit_count(live_store, spend) == 0
    assert ledger.begin_copy(first, "sha256:a", now=_CLAIMED_AT, expires_at=_CLAIM_LEASE) is None
    ledger.close_claim(first)
    ledger.free_claim(first, _CLAIMED_AT)
    assert ledger.read_claim(export) == second
    assert _unit_count(live_store, other) == 1


def test_a_copy_after_the_row_has_gone_copies_nothing_under_real_postgres(live_store):
    ledger, export, spend = _claim_rig(live_store)
    held = _held_claim(ledger, export, spend)
    ledger.close_claim(held)

    class _Forbidden:
        def copy(self, digest, *, from_tier, to_tier):
            raise AssertionError("a prepare that lost its claim must not copy")

    preparing = PreparingCopy(
        held, allowances=ledger, artifacts=_Forbidden(), caller=None, clock=lambda: _AFTER_LEASE
    )
    with pytest.raises(ExportClaimUnsettled):
        preparing.copy("sha256:a")
    assert _unit_count(live_store, spend) == 0


class _PaidBucket:
    """A bucket whose working bytes have gone, whose pinned object `pinned` says stands or raises
    when it is an error."""

    def __init__(self, pinned: bool | Exception) -> None:
        self._pinned = pinned

    def copy(self, digest, *, from_tier, to_tier):
        raise ArtifactNotFoundError(digest=digest, tier=from_tier)

    def exists(self, digest, *, tier):
        if isinstance(self._pinned, Exception):
            raise self._pinned
        return self._pinned


def _preparing(ledger, claim, bucket) -> PreparingCopy:
    return PreparingCopy(
        claim, allowances=ledger, artifacts=bucket, caller=None, clock=lambda: _CLAIMED_AT
    )


def test_a_paid_claim_stays_paid_when_a_retry_finds_no_working_bytes_under_real_postgres(
    live_store,
):
    ledger, export, spend = _claim_rig(live_store)
    copying = ledger.begin_copy(
        _held_claim(ledger, export, spend), "sha256:a", now=_CLAIMED_AT, expires_at=_CLAIM_LEASE
    )
    ledger.settle_copy(copying, landed=True)
    paid = ledger.claim(export, now=_AFTER_LEASE, expires_at=_AFTER_LEASE + timedelta(minutes=3))

    with pytest.raises(ArtifactNotFoundError):
        _preparing(ledger, paid, _PaidBucket(True)).copy("sha256:a")

    assert ledger.read_claim(export).state == ClaimState.landed
    assert _unit_count(live_store, spend) == 1


def test_a_takeover_of_a_copy_it_cannot_judge_refuses_before_any_render_under_real_postgres(
    live_store,
):
    ledger, export, spend = _claim_rig(live_store)
    ledger.begin_copy(
        _held_claim(ledger, export, spend), "sha256:a", now=_CLAIMED_AT, expires_at=_CLAIM_LEASE
    )
    taken = ledger.claim(export, now=_AFTER_LEASE, expires_at=_AFTER_LEASE + timedelta(minutes=3))
    unreadable = _PaidBucket(ArtifactWriteError(object_name="sha256:a", digest="sha256:a"))

    with pytest.raises(ExportClaimUnsettled), _preparing(ledger, taken, unreadable) as preparing:
        preparing.settle_earlier()

    assert ledger.read_claim(export).state == ClaimState.copying
    assert ledger.read_claim(export).expires_at == _CLAIMED_AT
    assert _unit_count(live_store, spend) == 1


def test_a_freed_claim_keeps_its_units_for_the_retry_that_takes_it_under_real_postgres(
    live_store,
):
    ledger, export, spend = _claim_rig(live_store)
    copying = ledger.begin_copy(
        _held_claim(ledger, export, spend), "sha256:a", now=_CLAIMED_AT, expires_at=_CLAIM_LEASE
    )

    ledger.free_claim(copying, _CLAIMED_AT)
    taken = ledger.claim(export, now=_CLAIMED_AT, expires_at=_CLAIM_LEASE)

    assert (taken.state, taken.spends, taken.digest) == (
        ClaimState.copying,
        (spend,),
        "sha256:a",
    )
    assert taken.generation != copying.generation
    assert _unit_count(live_store, spend) == 1
    ledger.settle_copy(copying, landed=False)
    assert _unit_count(live_store, spend) == 1


def test_a_stale_claims_unit_reads_as_held_until_the_sweep_settles_it_under_real_postgres(
    live_store,
):
    ledger, export, spend = _claim_rig(live_store, limit=1)
    _held_claim(ledger, export, spend)

    with pytest.raises(AllowanceExhausted) as refused:
        ledger.spend_together([spend], _AFTER_LEASE)
    assert refused.value.held is True

    swept = next(c for c in ledger.expired_claims(_AFTER_LEASE) if c.export_id == export)
    ledger.close_claim(swept)
    ledger.spend_together([spend], _AFTER_LEASE)
    assert _unit_count(live_store, spend) == 1


def test_a_copy_past_its_lease_cannot_land_after_a_sweep_refund_under_real_postgres(live_store):
    ledger, export, spend = _claim_rig(live_store)
    held = _held_claim(ledger, export, spend)

    assert (
        ledger.begin_copy(
            held, "sha256:a", now=_AFTER_LEASE, expires_at=_AFTER_LEASE + timedelta(minutes=3)
        )
        is None
    )
    assert ledger.read_claim(export) == held
    assert _unit_count(live_store, spend) == 1


def test_a_refund_racing_a_spend_takes_its_unit_back(live_store):
    """A refund that reads the day's last unit while a spend holds the row uncommitted waits for
    the spend, then takes one from the count the spend leaves."""
    caller = Caller(kind="address", key="ip:" + uuid4().hex)
    limits = {AllowanceKind.card: 5}
    now = datetime.now(UTC)
    LiveAllowanceLedger(store=live_store, limits=limits).spend(caller, AllowanceKind.card, now)
    spend_pool = _pool(size=1)
    refund_pool = _pool(size=1)
    try:
        holder = _HoldBeforeCommitStore(
            LiveDatasetStore(connections=spend_pool),
            hold_when=lambda sql: sql.startswith("INSERT INTO datasets.caller_allowance"),
        )
        spender = LiveAllowanceLedger(store=holder, limits=limits)
        refunder = LiveAllowanceLedger(
            store=LiveDatasetStore(connections=refund_pool), limits=limits
        )

        spend = _Call(lambda: spender.spend(caller, AllowanceKind.card, now))
        assert holder.held.wait(timeout=5), "the spend never reached its commit"
        refund = _Call(lambda: refunder.refund(caller, AllowanceKind.card, now))
        deadline = time.monotonic() + 5
        while not _waits_on_a_lock(live_store, table="datasets.caller_allowance"):
            assert time.monotonic() < deadline, "the refund never waited on the spend's row"
            time.sleep(0.02)
        holder.release.set()
        spend.join()
        refund.join()
    finally:
        spend_pool.close_all()
        refund_pool.close_all()

    with live_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT count FROM datasets.caller_allowance WHERE caller_key = %s AND kind = %s",
            (caller.key, AllowanceKind.card.value),
        )
    assert row == {"count": 1}


def _waits_on_a_lock(store, *, table: str) -> bool:
    """Whether another session waits on a lock in a statement naming `table`."""
    with store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT count(*) AS n FROM pg_stat_activity "
            "WHERE wait_event_type = 'Lock' AND strpos(query, %s) > 0 "
            "AND pid <> pg_backend_pid()",
            (table,),
        )
    return row is not None and row["n"] > 0


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


@pytest.mark.parametrize(
    ("revoked_version", "revoked_under_other"),
    [("cv1", False), ("cv_base", False), ("cv1", True), ("cv_base", True)],
    ids=["own", "base", "own-under-other-id", "base-under-other-id"],
)
def test_record_reference_unless_revoked_refuses_a_revoked_version_or_base(
    live_store, revoked_version, revoked_under_other
):
    """A revocation of the version or its base version, under the subject id or another id
    the write names (`revoked_under`), refuses the write."""
    case = f"{revoked_version}-{revoked_under_other}-{uuid4().hex}"
    digest = f"sha256:test-live-reference-unless-revoked-{case}"
    subject_id = f"spec:test-live-reference-unless-revoked-{case}"
    other_id = f"spec:test-live-reference-unless-revoked-other-{case}"
    _seed_reference_fk_targets(live_store, digest=digest, subject_id=subject_id)
    with live_store.transaction() as tx:
        for recorded_under, content_version in (
            (subject_id, "cv_base"),
            (other_id, "cv1"),
            (other_id, "cv_base"),
        ):
            tx.execute(
                "INSERT INTO datasets.version_record "
                "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
                "recorded_at) VALUES (%s, %s, %s, %s, %s, now())",
                (recorded_under, "spec", content_version, 1, _LINEAGE_BODY),
            )
        tx.execute(
            "UPDATE datasets.version_record SET base_content_version = %s "
            "WHERE subject_id = %s AND content_version = %s",
            ("cv_base", subject_id, "cv1"),
        )
        tx.execute(
            "INSERT INTO datasets.version_revocation "
            "(subject_id, content_version, reason, revoked_at) VALUES (%s, %s, %s, now())",
            (other_id if revoked_under_other else subject_id, revoked_version, "takedown"),
        )

    with pytest.raises(SnapshotRevoked, match="takedown"):
        LiveArtifactReferenceWriter(store=live_store).record_reference_unless_revoked(
            principal="first-party",
            artifact_digest=digest,
            subject_id=subject_id,
            content_version="cv1",
            reference_kind=PIN_REFERENCE_KIND,
            now=datetime(2026, 1, 1, tzinfo=UTC),
            revoked_under=[other_id],
        )

    with live_store.transaction() as tx:
        references = tx.fetch_all(
            "SELECT 1 FROM datasets.artifact_reference WHERE subject_id = %s", (subject_id,)
        )
    assert references == []


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


def _scratch_connect(database: str):
    parts = urlsplit(_DSN)
    return lambda: connect(
        connector=None,
        instance_connection_name=None,
        host=parts.hostname,
        port=parts.port or 5432,
        database=database,
        user=parts.username,
        password=parts.password,
    )


@contextmanager
def _scratch_database():
    """A database of its own, dropped on exit, for a migration test that must start from a
    schema older than the shared database's. Yields its connect callable."""
    name = f"dataset_store_scratch_{uuid4().hex[:12]}"
    admin = _scratch_connect(urlsplit(_DSN).path.lstrip("/"))()
    admin.autocommit = True
    try:
        admin.run(f"CREATE DATABASE {name}")
        for role in ("dataset-store-scratch-runtime", "dataset-store-scratch-worker"):
            admin.run(
                f'DO $$ BEGIN CREATE ROLE "{role}" NOLOGIN; '
                "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
            )
        try:
            yield _scratch_connect(name)
        finally:
            admin.run(f"DROP DATABASE {name} WITH (FORCE)")
    finally:
        admin.close()


def test_0024_leaves_an_in_flight_old_worker_run_to_its_heartbeat():
    """A run the image before 0024 claimed keeps renewing `updated_at` after the migration. The
    new store neither abandons nor reclaims it while that heartbeat is fresh, and does both
    once `updated_at` is `LEASE_SECONDS` old."""
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    substitutions = {
        "RUNTIME_DB_USER": "dataset-store-scratch-runtime",
        "WORKER_DB_USER": "dataset-store-scratch-worker",
    }
    with _scratch_database() as scratch_connect:
        with tempfile.TemporaryDirectory() as before_0024:
            copied = 0
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                if path.name < "0024":
                    shutil.copy(path, before_0024)
                    copied += 1
            assert copied == 22
            apply_pending(
                connect=scratch_connect,
                migrations_dir=Path(before_0024),
                substitutions=substitutions,
            )
        pool = ConnectionPool(connect=scratch_connect, size=2)
        try:
            store = LiveDatasetStore(connections=pool)
            with store.transaction() as tx:
                tx.execute(
                    "INSERT INTO datasets.operation (id, request_identity, options, "
                    "coverage_stamp, state, estimate_rows, estimate_requests, attempt_count, "
                    "lease_until, created_at, updated_at, submitted_at, dispatch_count, "
                    "dispatch_confirmed, dispatched_at) VALUES "
                    "(%s, %s, %s, 's', 'running', 1, 1, 1, %s, %s, %s, %s, 1, 1, %s)",
                    ("op_in_flight", {}, {}, t0 + timedelta(seconds=60), t0, t0, t0, t0),
                )
            assert "0024_operation_attempt_fencing.sql" in apply_pending(
                connect=scratch_connect, substitutions=substitutions
            )
            # One heartbeat of the old worker, which renews `lease_until` and `updated_at`.
            heartbeat = t0 + timedelta(seconds=20)
            with store.transaction() as tx:
                tx.execute(
                    "UPDATE datasets.operation SET lease_until = %s, updated_at = %s "
                    "WHERE id = 'op_in_flight'",
                    (heartbeat + timedelta(seconds=60), heartbeat),
                )
            now_box = {"t": heartbeat + timedelta(seconds=120)}
            operations = LiveOperationStore(store=store, now=lambda: now_box["t"])

            fresh = operations.get("op_in_flight")
            assert fresh is not None and fresh.claimed_at is None
            assert fresh.final_delivery is None
            assert not fresh.abandoned(now_box["t"], deadline=_OPERATION_DEADLINE)
            assert operations.abandon("op_in_flight", attempt=1) is None
            assert (
                operations.claim(
                    "op_in_flight", submission_count=0, lease_seconds=60, final_delivery=False
                )
                is None
            )

            now_box["t"] = heartbeat + timedelta(seconds=LEASE_SECONDS + 1)
            stale = operations.get("op_in_flight")
            assert stale is not None
            assert stale.abandoned(now_box["t"], deadline=_OPERATION_DEADLINE)
            reclaimed = operations.claim(
                "op_in_flight", submission_count=0, lease_seconds=60, final_delivery=False
            )
            assert reclaimed is not None and reclaimed.attempt_count == 2
        finally:
            pool.close_all()


def _manifest_of(body: dict[str, object]):
    """The manifest a migrated body's facts assemble once 0031 backfills them."""
    from dataset_store.snapshot import StoredVersion
    from datasets_api.manifest import stored_manifest
    from tests.datasets_api._version_facts import facts_from_body

    return stored_manifest(
        StoredVersion(
            subject_id="spec_migrated",
            subject_kind="spec",
            content_version=cast(str, body["content_version"]),
            schema_version=1,
            facts=facts_from_body(body),
        )
    )


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
            assert _manifest_of(body).trace == body["trace"]
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
        manifest = _manifest_of(bodies[definition])
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

        manifest = _manifest_of(bodies[definition])
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
    projection of the `spec-kit` repository's `shape-statvar-schema-all-five-members`
    identity vector."""
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
    """0022 drops a body's `"collection": null`, with every other key kept, so the body's
    facts assemble a manifest. A non-null `collection` and a body without the key stay as
    stored, and a rerun changes nothing."""

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

    def _bodies() -> dict[str, dict]:
        with live_store.transaction() as tx:
            found = tx.fetch_all(
                "SELECT subject_id, immutable_body FROM datasets.version_record "
                "WHERE subject_id = ANY(%s)",
                (subjects,),
            )
        return {cast(str, row["subject_id"]): cast(dict, row["immutable_body"]) for row in found}

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
        manifest = _manifest_of(bodies[definition])
        assert manifest.definition.spec_id == definition
        assert manifest.series_coverage_stamp == "sha256:coverage"
    finally:
        with live_store.transaction() as tx:
            tx.execute(
                "DELETE FROM datasets.version_record WHERE subject_id = ANY(%s)", (subjects,)
            )


_PRINCIPAL_LABEL_MIGRATION = MIGRATIONS_DIR / "0026_artifact_reference_principal_label.sql"


def _principal_label(secret: str) -> str:
    label = Principal(presented=secret, grants=frozenset()).label
    assert label is not None
    return label


def test_raw_pin_principals_become_their_labels_and_fold_into_a_labelled_twin():
    pool = _pool()
    store = LiveDatasetStore(connections=pool)
    suffix = uuid4().hex
    subject_id = f"spec:test-live-principal-label-{suffix}"
    digest = f"sha256:test-live-principal-label-{suffix}"
    _seed_reference_fk_targets(store, digest=digest, subject_id=subject_id)
    lone, twinned = f"lone-secret-{suffix}", f"twinned-secret-é-{suffix}"
    earlier = datetime(2026, 1, 1, tzinfo=UTC)
    later = earlier + timedelta(days=3)
    rows = [
        (lone, "pin", earlier, None),
        (twinned, "pin", earlier, later),
        (_principal_label(twinned), "pin", later, None),
        (f"presentation:component-{suffix}", "presentation_draft", earlier, None),
    ]
    try:
        with store.transaction() as tx:
            for principal, kind, created_at, removed_at in rows:
                tx.execute(
                    "INSERT INTO datasets.artifact_reference (principal, artifact_digest, "
                    "subject_id, content_version, reference_kind, created_at, removed_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (principal, digest, subject_id, "cv1", kind, created_at, removed_at),
                )
        with pool.borrow() as conn:
            cursor = conn.cursor()
            cursor.execute(_PRINCIPAL_LABEL_MIGRATION.read_text())
            cursor.close()
            conn.commit()
        with store.transaction() as tx:
            stored = tx.fetch_all(
                "SELECT principal, created_at, removed_at FROM datasets.artifact_reference "
                "WHERE subject_id = %s ORDER BY principal",
                (subject_id,),
            )
    finally:
        pool.close_all()

    assert {row["principal"]: (row["created_at"], row["removed_at"]) for row in stored} == {
        _principal_label(lone): (earlier, None),
        _principal_label(twinned): (earlier, None),
        f"presentation:component-{suffix}": (earlier, None),
    }


@pytest.mark.parametrize("own_under_other", [True, False], ids=["own-other", "own-subject"])
def test_a_version_revocation_wins_over_a_base_revocation_under_another_id(
    live_store, own_under_other
):
    """A collapsed version revoked under one id while its base version is revoked under the
    other reads and refuses with the version's own revocation, in `read_version_state` and in
    the guarded reference write."""
    case = uuid4().hex
    digest = f"sha256:test-live-revocation-precedence-{case}"
    subject_id = f"spec:test-live-revocation-precedence-{case}"
    other_id = f"spec:test-live-revocation-precedence-other-{case}"
    _seed_reference_fk_targets(live_store, digest=digest, subject_id=subject_id)
    own, base = (other_id, subject_id) if own_under_other else (subject_id, other_id)
    with live_store.transaction() as tx:
        for recorded_under, content_version in (
            (subject_id, "cv_base"),
            (other_id, "cv1"),
            (other_id, "cv_base"),
        ):
            tx.execute(
                "INSERT INTO datasets.version_record "
                "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
                "recorded_at) VALUES (%s, %s, %s, %s, %s, now())",
                (recorded_under, "spec", content_version, 1, _LINEAGE_BODY),
            )
        tx.execute(
            "UPDATE datasets.version_record "
            "SET base_content_version = %s, input_content_versions = %s "
            "WHERE subject_id = %s AND content_version = %s",
            ("cv_base", json.dumps(["cv_input"]), subject_id, "cv1"),
        )
        for revoked_under, content_version, reason in (
            (own, "cv1", "own takedown"),
            (base, "cv_base", "base takedown"),
        ):
            tx.execute(
                "INSERT INTO datasets.version_revocation "
                "(subject_id, content_version, reason, revoked_at) VALUES (%s, %s, %s, now())",
                (revoked_under, content_version, reason),
            )

    state = LiveSnapshotReader(store=live_store).read_version_state(
        subject_id, "cv1", revoked_under=[other_id]
    )
    with pytest.raises(SnapshotRevoked) as refused:
        LiveArtifactReferenceWriter(store=live_store).record_reference_unless_revoked(
            principal="first-party",
            artifact_digest=digest,
            subject_id=subject_id,
            content_version="cv1",
            reference_kind=PIN_REFERENCE_KIND,
            now=datetime(2026, 1, 1, tzinfo=UTC),
            revoked_under=[other_id],
        )

    assert state is not None and state.revocation is not None
    assert state.revocation.reason == "own takedown"
    assert refused.value.reason == "own takedown"


def test_0028_spec_alias_binds_a_posted_id_to_each_served_id_once():
    """`register_served_definition` and `read_served_spec_ids` against the real table: one
    row per pair, newest binding first, both ids held to the registry by foreign key, and a
    quarantined served id refusing the read and a later resolution under the posted id."""
    substitutions = {
        "RUNTIME_DB_USER": "dataset-store-scratch-runtime",
        "WORKER_DB_USER": "dataset-store-scratch-worker",
    }
    with _scratch_database() as scratch_connect:
        apply_pending(
            connect=scratch_connect, migrations_dir=MIGRATIONS_DIR, substitutions=substitutions
        )
        pool = ConnectionPool(connect=scratch_connect, size=1)
        try:
            store = LiveDatasetStore(connections=pool)
            registry, reader = LiveSpecRegistry(store=store), LiveSpecReader(store=store)
            posted, earlier, later = (
                recompute_identity(minimal_identity(shape=shape_for(name)))
                for name in ("alias_posted", "alias_earlier", "alias_later")
            )
            registry.register_definition(
                identity_definition=posted.identity,
                schema_version=1,
                supplied_spec_id=posted.spec_id,
            )
            assert reader.read_served_spec_ids(posted.spec_id) == ()
            assert reader.read_served_spec_ids(earlier.spec_id) is None

            for served in (earlier, earlier, later):
                stored = registry.register_served_definition(
                    served=served, schema_version=1, posted_spec_id=posted.spec_id
                )
                assert stored == reader.read_definition(served.spec_id)

            with store.transaction() as tx:
                rows = tx.fetch_all(
                    "SELECT served_spec_id FROM datasets.spec_alias WHERE posted_spec_id = %s",
                    (posted.spec_id,),
                )
            assert sorted(row["served_spec_id"] for row in rows) == sorted(
                [earlier.spec_id, later.spec_id]
            )
            assert reader.read_served_spec_ids(posted.spec_id) == (
                later.spec_id,
                earlier.spec_id,
            )

            with pytest.raises(SpecRegistryStoreError):
                registry.register_served_definition(
                    served=later, schema_version=1, posted_spec_id="spec_never_registered"
                )

            with store.transaction() as tx:
                tx.execute(
                    "INSERT INTO datasets.identity_quarantine "
                    "(id_kind, subject_id, held_bytes, incoming_bytes) VALUES (%s, %s, %s, %s)",
                    ("spec", earlier.spec_id, b"held", b"incoming"),
                )
            for read in (reader.read_served_spec_ids, reader.read_definition):
                with pytest.raises(IdentityQuarantined) as excinfo:
                    read(posted.spec_id)
                assert excinfo.value.subject_id == earlier.spec_id
            assert reader.read_definition(later.spec_id) is not None
            with pytest.raises(IdentityQuarantined) as excinfo:
                registry.register_served_definition(
                    served=later, schema_version=1, posted_spec_id=posted.spec_id
                )
            assert excinfo.value.subject_id == earlier.spec_id
        finally:
            pool.close_all()


_VERSION_FACTS_MIGRATION = "0031_version_record_facts_backfill.sql"
_STORED_FACT_COLUMNS = (*VERSION_FACT_COLUMNS, "frame_grammar_version")


def _insert_as_the_previous_image(
    live_store, *, subject_id: str, content_version: str, body: object, base: str | None = None
) -> None:
    """Insert a version row naming only the columns the previous image's writer names."""
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, "
            "immutable_body, trace_id, recorded_at, base_content_version) "
            "VALUES (%s, 'spec', %s, 1, %s, NULL, now(), %s) "
            "ON CONFLICT (subject_id, content_version) DO NOTHING "
            "RETURNING subject_id",
            (subject_id, content_version, body, base),
        )


def _clear_stored_facts(live_store, subjects: list[str]) -> None:
    """Leave `subjects`' rows as rows inserted before migration 0030 are: a body, no fact and no
    grammar version."""
    assignments = ", ".join(f"{column} = NULL" for column in _STORED_FACT_COLUMNS)
    with live_store.transaction() as tx:
        tx.execute(
            f"UPDATE datasets.version_record SET {assignments} WHERE subject_id = ANY(%s)",
            (subjects,),
        )


def _stored_fact_columns(
    live_store, subject_id: str, content_version: str = "cv1"
) -> dict[str, object]:
    with live_store.transaction() as tx:
        row = tx.fetch_one(
            f"SELECT {', '.join(_STORED_FACT_COLUMNS)} FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (subject_id, content_version),
        )
    assert row is not None
    return dict(row)


def _delete_versions(live_store, subjects: list[str]) -> None:
    with live_store.transaction() as tx:
        tx.execute("DELETE FROM datasets.version_record WHERE subject_id = ANY(%s)", (subjects,))


def test_0031_backfills_frame_grammar_version_from_the_stored_trace(live_store):
    """A body whose trace names its grammar version gains the column, a body with no trace (a
    derived `ops_` frame) keeps NULL, and a rerun changes nothing. A version written through
    `LiveVersionRecordWriter` stores its own grammar version."""
    suffix = uuid4().hex
    traced = f"spec_fgv_traced_{suffix}"
    derived = f"ops_fgv_{suffix}"
    written = f"spec_fgv_written_{suffix}"
    seeded = {
        traced: {
            "prepared_at": "2026-09-02T08:30:15Z",
            "trace": {"graph_revision": "rev", "frame_grammar_version": "1.0.0"},
        },
        derived: {"op_spec_hash": derived, "input_content_versions": ["cv_x"]},
    }

    def _columns() -> dict[str, tuple[object, object]]:
        with live_store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT subject_id, content_version, frame_grammar_version "
                "FROM datasets.version_record WHERE subject_id = ANY(%s)",
                ([traced, derived, written],),
            )
        return {
            cast(str, row["subject_id"]): (row["content_version"], row["frame_grammar_version"])
            for row in rows
        }

    for subject_id, body in seeded.items():
        _insert_as_the_previous_image(
            live_store,
            subject_id=subject_id,
            content_version="cv_0123456789abcdef0123456789abcdef",
            body=body,
        )
    _clear_stored_facts(live_store, list(seeded))
    try:
        assert _rerun_migration(live_store, _VERSION_FACTS_MIGRATION) == [_VERSION_FACTS_MIGRATION]
        expected = {
            traced: ("cv_0123456789abcdef0123456789abcdef", "1.0.0"),
            derived: ("cv_0123456789abcdef0123456789abcdef", None),
        }
        assert _columns() == expected
        assert _rerun_migration(live_store, _VERSION_FACTS_MIGRATION) == [_VERSION_FACTS_MIGRATION]
        assert _columns() == expected

        LiveVersionRecordWriter(store=live_store).record_version(
            VersionRecord(
                subject_id=written,
                subject_kind="spec",
                content_version="cv_fedcba9876543210fedcba9876543210",
                schema_version=1,
                facts=FrameLineage(input_content_versions=[]),
                frame_grammar_version="1.1.0",
                frame_sha256="f" * 64,
            )
        )
        assert _columns()[written] == ("cv_fedcba9876543210fedcba9876543210", "1.1.0")
        stored = LiveSnapshotReader(store=live_store).read_version(
            written, "cv_fedcba9876543210fedcba9876543210"
        )
        assert stored is not None and stored.frame_sha256 == "f" * 64
    finally:
        _delete_versions(live_store, [traced, derived, written])


def test_0031_backfills_each_fact_from_the_stored_body(live_store):
    """Every golden body (`tests/datasets_api/version_facts_golden.json`) inserted before 0030
    gains the facts it holds, which assemble the manifest the body rendered, and a rerun changes
    nothing. A version written through `LiveVersionRecordWriter` stores the body its facts
    encode in `immutable_body` and reads back its facts."""
    from dataset_store.snapshot import ManifestFacts
    from datasets_api.manifest import stored_manifest
    from tests.datasets_api._version_facts import facts_from_body, golden_cases

    suffix = uuid4().hex
    cases = golden_cases()
    subjects = {name: f"{case['subject_id']}_{suffix}" for name, case in cases.items()}
    written = f"spec_written_{suffix}"
    reader = LiveSnapshotReader(store=live_store)

    def _read(name: str):
        stored = reader.read_version(subjects[name], cast(str, cases[name]["content_version"]))
        assert stored is not None
        return stored

    def _facts() -> dict[str, object]:
        return {name: _read(name).facts for name in cases}

    for name, case in cases.items():
        _insert_as_the_previous_image(
            live_store,
            subject_id=subjects[name],
            content_version=case["content_version"],
            body=case["body"],
            base=case["base_content_version"],
        )
    _clear_stored_facts(live_store, list(subjects.values()))
    try:
        with pytest.raises(VersionFactsMissing):
            _read("definition")

        assert _rerun_migration(live_store, _VERSION_FACTS_MIGRATION) == [_VERSION_FACTS_MIGRATION]
        facts = _facts()
        assert facts == {name: facts_from_body(case["body"]) for name, case in cases.items()}
        assert _rerun_migration(live_store, _VERSION_FACTS_MIGRATION) == [_VERSION_FACTS_MIGRATION]
        assert _facts() == facts

        for name, case in cases.items():
            if "manifest" not in case["expected"]:
                continue
            assert (
                stored_manifest(_read(name)).model_dump(mode="json") == case["expected"]["manifest"]
            )

        definition = cases["definition"]
        stored_facts = facts_from_body(definition["body"])
        assert isinstance(stored_facts, ManifestFacts)
        LiveVersionRecordWriter(store=live_store).record_version(
            VersionRecord(
                subject_id=written,
                subject_kind="spec",
                content_version=cast(str, definition["content_version"]),
                schema_version=1,
                facts=stored_facts,
                frame_grammar_version="1.0.0",
            )
        )
        with live_store.transaction() as tx:
            row = tx.fetch_one(
                "SELECT immutable_body FROM datasets.version_record WHERE subject_id = %s",
                (written,),
            )
        assert row == {"immutable_body": definition["body"]}
        read_back = reader.read_version(written, cast(str, definition["content_version"]))
        assert read_back is not None
        assert read_back.facts == stored_facts
    finally:
        _delete_versions(live_store, [*subjects.values(), written])


def test_0031_fails_and_fills_nothing_while_a_body_holds_no_facts(live_store):
    """A row whose body has neither `prepared_at` nor `input_content_versions` fails 0031 with
    the raise of its closing check (SQLSTATE P0001): the migration is not recorded as applied
    and the row it would have filled keeps no facts. Without that row it applies."""
    from tests.datasets_api._version_facts import golden_cases

    suffix = uuid4().hex
    definition = golden_cases()["definition"]
    fillable = f"spec_0031_fillable_{suffix}"
    factless = f"spec_0031_factless_{suffix}"
    empty = dict.fromkeys(_STORED_FACT_COLUMNS)

    def _applied() -> bool:
        with live_store.transaction() as tx:
            row = tx.fetch_one(
                "SELECT 1 AS applied FROM datasets.dataset_store_schema_migrations "
                "WHERE version = %s",
                (_VERSION_FACTS_MIGRATION,),
            )
        return row is not None

    _insert_as_the_previous_image(
        live_store,
        subject_id=fillable,
        content_version=definition["content_version"],
        body=definition["body"],
    )
    _insert_as_the_previous_image(
        live_store, subject_id=factless, content_version="cv1", body={"revision": "first"}
    )
    _clear_stored_facts(live_store, [fillable])
    try:
        assert _stored_fact_columns(live_store, factless) == empty
        with pytest.raises(pg8000.exceptions.DatabaseError) as refused:
            _rerun_migration(live_store, _VERSION_FACTS_MIGRATION)
        assert cast(dict, refused.value.args[0])["C"] == "P0001"
        assert not _applied()
        assert _stored_fact_columns(live_store, fillable, definition["content_version"]) == empty
    finally:
        _delete_versions(live_store, [factless])
        try:
            assert _rerun_migration(live_store, _VERSION_FACTS_MIGRATION) in (
                [_VERSION_FACTS_MIGRATION],
                [],
            )
            assert (
                _stored_fact_columns(live_store, fillable, definition["content_version"])[
                    "prepared_at"
                ]
                is not None
            )
        finally:
            _delete_versions(live_store, [fillable])


def test_a_row_the_previous_image_inserts_stores_the_facts_of_its_body(live_store):
    """With every migration applied, a row inserted with only `immutable_body`, as the previous
    image inserts it, reads through `LiveSnapshotReader.read_version` with the facts its body
    holds and stores its body trace's `frame_grammar_version`."""
    from tests.datasets_api._version_facts import facts_from_body, golden_cases

    suffix = uuid4().hex
    cases = golden_cases()
    subjects = {name: f"{case['subject_id']}_{suffix}" for name, case in cases.items()}
    reader = LiveSnapshotReader(store=live_store)
    try:
        for name, case in cases.items():
            _insert_as_the_previous_image(
                live_store,
                subject_id=subjects[name],
                content_version=case["content_version"],
                body=case["body"],
                base=case["base_content_version"],
            )
        for name, case in cases.items():
            stored = reader.read_version(subjects[name], case["content_version"])
            assert stored is not None
            assert stored.facts == facts_from_body(case["body"]), name
            assert stored.frame_sha256 is None, name
            assert stored.frame_summed is None, name
            stored_columns = _stored_fact_columns(
                live_store, subjects[name], case["content_version"]
            )
            assert stored_columns["frame_grammar_version"] == (
                case["body"].get("trace", {}).get("frame_grammar_version")
            ), name
    finally:
        _delete_versions(live_store, list(subjects.values()))


def test_a_row_inserted_with_facts_is_stored_as_written(live_store):
    """The insert trigger leaves a row that arrives with facts alone: a dataset version and a
    published frame whose bodies hold other values, and a grammar version the body's trace
    contradicts, keep every column as inserted."""
    suffix = uuid4().hex
    prepared_at = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
    decoy = {
        "prepared_at": "1999-01-01T00:00:00Z",
        "definition": {"from": "body"},
        "coverage": {"from": "body"},
        "citation": {"from": "body"},
        "trace": {"frame_grammar_version": "9.9.9"},
        "warnings": [{"from": "body"}],
        "series_count": 99,
        "series_coverage_stamp": "body",
        "output_profile": {"from": "body"},
        "series_conversions": [{"from": "body"}],
        "default_scope_plural": True,
        "input_content_versions": ["cv_body"],
        "composed_stat_vars": [{"from": "body"}],
    }
    written: dict[str, dict[str, object]] = {
        f"spec_trigger_manifest_{suffix}": {
            **dict.fromkeys(_STORED_FACT_COLUMNS),
            "prepared_at": prepared_at,
            "definition": {"from": "column"},
            "frame_grammar_version": "1.1.0",
        },
        f"ops_trigger_lineage_{suffix}": {
            **dict.fromkeys(_STORED_FACT_COLUMNS),
            "input_content_versions": [],
        },
    }
    try:
        with live_store.transaction() as tx:
            for subject_id, columns in written.items():
                tx.execute(
                    "INSERT INTO datasets.version_record "
                    "(subject_id, subject_kind, content_version, schema_version, "
                    f"immutable_body, recorded_at, {', '.join(columns)}) "
                    f"VALUES (%s, 'spec', 'cv1', 1, %s, now(), {', '.join(['%s'] * len(columns))})",
                    (
                        subject_id,
                        decoy,
                        *(
                            json.dumps(value) if isinstance(value, dict | list) else value
                            for value in columns.values()
                        ),
                    ),
                )
        for subject_id, columns in written.items():
            assert _stored_fact_columns(live_store, subject_id) == columns
    finally:
        _delete_versions(live_store, list(written))


def test_settle_counts_factless_versions_and_labels_raw_pin_principals(live_store):
    """`datasets-maintenance settle` turns a pin reference holding a raw secret (the previous
    image's write after 0026) into the secret's label and returns 0, and a second run changes
    nothing. A version row holding no facts and one whose profile names the enrichment `kind`
    each add one to its count and stay as they are."""
    from datasets_api.maintenance import run_settle

    suffix = uuid4().hex
    factless = f"spec_settle_factless_{suffix}"
    legacy = f"spec_settle_legacy_party_kind_{suffix}"
    pinned = f"spec_settle_pin_{suffix}"
    digest = f"sha256:test-live-settle-{suffix}"
    secret = f"settle-secret-{suffix}"
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    empty = dict.fromkeys(_STORED_FACT_COLUMNS)
    _seed_reference_fk_targets(live_store, digest=digest, subject_id=pinned)
    with live_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_reference (principal, artifact_digest, subject_id, "
            "content_version, reference_kind, created_at) VALUES (%s, %s, %s, 'cv1', %s, %s)",
            (secret, digest, pinned, PIN_REFERENCE_KIND, created_at),
        )

    def _principals() -> list[object]:
        with live_store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT principal FROM datasets.artifact_reference WHERE subject_id = %s",
                (pinned,),
            )
        return [row["principal"] for row in rows]

    try:
        assert _principals() == [secret]
        for _ in range(2):
            assert run_settle(store=live_store) == 0
            assert _principals() == [_principal_label(secret)]

        _insert_as_the_previous_image(
            live_store, subject_id=factless, content_version="cv1", body={"revision": "first"}
        )
        assert run_settle(store=live_store) == 1
        assert _stored_fact_columns(live_store, factless) == empty

        _insert_as_the_previous_image(
            live_store,
            subject_id=legacy,
            content_version="cv1",
            body={"prepared_at": "2026-10-01T00:00:00Z", "output_profile": _profile("kind")},
        )
        _store_profile_past_the_trigger(live_store, legacy, _profile("kind"))
        assert run_settle(store=live_store) == 2
        assert _stored_profiles(live_store, [legacy]) == {
            legacy: (_profile("kind"), _profile("kind"))
        }
    finally:
        with live_store.transaction() as tx:
            tx.execute("DELETE FROM datasets.artifact_reference WHERE subject_id = %s", (pinned,))
            tx.execute("DELETE FROM datasets.version_artifact WHERE subject_id = %s", (pinned,))
        _delete_versions(live_store, [factless, legacy, pinned])


def test_settle_leaves_labels_and_caller_keys_as_they_are(live_store):
    """Only a raw secret is rewritten. An open caller's `ip:` key and a principal's label stay
    as stored over two runs, so a caller's repeat pin still matches its own reference."""
    from datasets_api.maintenance import run_settle

    suffix = uuid4().hex
    pinned = f"spec_settle_keys_{suffix}"
    digest = f"sha256:test-live-settle-keys-{suffix}"
    caller_key = "ip:" + suffix
    label = _principal_label(f"settle-keys-secret-{suffix}")
    _seed_reference_fk_targets(live_store, digest=digest, subject_id=pinned)
    with live_store.transaction() as tx:
        for principal in (caller_key, label):
            tx.execute(
                "INSERT INTO datasets.artifact_reference (principal, artifact_digest, "
                "subject_id, content_version, reference_kind, created_at) "
                "VALUES (%s, %s, %s, 'cv1', %s, now())",
                (principal, digest, pinned, PIN_REFERENCE_KIND),
            )

    try:
        for _ in range(2):
            assert run_settle(store=live_store) == 0
            with live_store.transaction() as tx:
                rows = tx.fetch_all(
                    "SELECT principal FROM datasets.artifact_reference WHERE subject_id = %s",
                    (pinned,),
                )
            assert sorted(row["principal"] for row in rows) == sorted([caller_key, label])
    finally:
        with live_store.transaction() as tx:
            tx.execute("DELETE FROM datasets.artifact_reference WHERE subject_id = %s", (pinned,))
            tx.execute("DELETE FROM datasets.version_artifact WHERE subject_id = %s", (pinned,))
        _delete_versions(live_store, [pinned])


def test_settle_waits_for_a_pin_in_flight_and_folds_the_raw_row_into_it(live_store):
    """A pin under the label of a raw row, uncommitted when settle starts, would collide with
    settle's rewrite of that row. Settle queues behind it, then folds the raw row into the
    pin's and counts nothing left."""
    from datasets_api.maintenance import run_settle

    suffix = uuid4().hex
    pinned = f"spec_settle_race_{suffix}"
    digest = f"sha256:test-live-settle-race-{suffix}"
    secret = f"settle-race-secret-{suffix}"
    _seed_reference_fk_targets(live_store, digest=digest, subject_id=pinned)
    insert = (
        "INSERT INTO datasets.artifact_reference (principal, artifact_digest, subject_id, "
        "content_version, reference_kind, created_at) VALUES (%s, %s, %s, 'cv1', %s, now())"
    )
    with live_store.transaction() as tx:
        tx.execute(insert, (secret, digest, pinned, PIN_REFERENCE_KIND))

    observer_pool = _pool(size=1)
    observer = LiveDatasetStore(connections=observer_pool)
    settled: list[int | BaseException] = []

    def _settle() -> None:
        try:
            settled.append(run_settle(store=live_store))
        except BaseException as exc:  # The test reads it below.
            settled.append(exc)

    try:
        with live_store.transaction() as pin:
            pin.execute(insert, (_principal_label(secret), digest, pinned, PIN_REFERENCE_KIND))
            settling = threading.Thread(target=_settle)
            settling.start()
            deadline = time.monotonic() + 10
            while not _settle_is_waiting_on_a_lock(observer):
                assert time.monotonic() < deadline, "settle never waited on the pin"
                time.sleep(0.05)
        settling.join(timeout=10)
        with live_store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT principal FROM datasets.artifact_reference WHERE subject_id = %s",
                (pinned,),
            )
    finally:
        observer_pool.close_all()
        with live_store.transaction() as tx:
            tx.execute("DELETE FROM datasets.artifact_reference WHERE subject_id = %s", (pinned,))
            tx.execute("DELETE FROM datasets.version_artifact WHERE subject_id = %s", (pinned,))
        _delete_versions(live_store, [pinned])

    assert settled == [0]
    assert [row["principal"] for row in rows] == [_principal_label(secret)]


def _settle_is_waiting_on_a_lock(store) -> bool:
    with store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT count(*) AS n FROM pg_stat_activity "
            "WHERE wait_event_type = 'Lock' AND query LIKE '%datasets.artifact_reference%' "
            "AND pid <> pg_backend_pid()"
        )
    return row is not None and row["n"] > 0


_STRIP_DOCUMENTATION_MIGRATION = "0035_strip_backfilled_documentation.sql"


def test_0035_strips_documentation_from_bodies_prepared_before_the_cutoff(live_store):
    """The migration drops `documentation` from each cited source of a body prepared before
    2026-10-04T09:22:14Z, keeps every other member and the `citation` column equal to the
    body, and leaves a body prepared at that instant and a body with no citation as stored. A
    rerun changes nothing."""

    suffix = uuid4().hex
    before = f"spec_0035_before_{suffix}"
    at_cutoff = f"spec_0035_at_cutoff_{suffix}"
    no_citation = f"spec_0035_no_citation_{suffix}"
    subjects = [before, at_cutoff, no_citation]
    documented = _null_collection_body(before)
    undocumented = {"ref": "OECD_CRS", "label": "OECD CRS", "processing_level": "minor"}
    citation = cast(dict, documented["citation"])
    (source,) = citation["sources"]
    documented["citation"] = {**citation, "sources": [source, undocumented]}
    documented["prepared_at"] = "2026-10-04T09:22:13Z"
    cutoff_body = {**documented, "prepared_at": "2026-10-04T09:22:14Z"}
    no_citation_body = {key: value for key, value in documented.items() if key != "citation"}
    stripped = {
        **documented,
        "citation": {
            **citation,
            "sources": [
                {key: value for key, value in source.items() if key != "documentation"},
                undocumented,
            ],
        },
    }
    stored = {before: documented, at_cutoff: cutoff_body, no_citation: no_citation_body}
    expected = {before: stripped, at_cutoff: cutoff_body, no_citation: no_citation_body}

    def _rows() -> dict[str, tuple[dict, dict | None]]:
        with live_store.transaction() as tx:
            found = tx.fetch_all(
                "SELECT subject_id, immutable_body, citation FROM datasets.version_record "
                "WHERE subject_id = ANY(%s)",
                (subjects,),
            )
        return {
            cast(str, row["subject_id"]): (
                cast(dict, row["immutable_body"]),
                cast("dict | None", row["citation"]),
            )
            for row in found
        }

    for subject_id, body in stored.items():
        _insert_as_the_previous_image(
            live_store, subject_id=subject_id, content_version="cv1", body=body
        )
    try:
        assert _rerun_migration(live_store, _STRIP_DOCUMENTATION_MIGRATION) == [
            _STRIP_DOCUMENTATION_MIGRATION
        ]
        rows = _rows()
        assert {subject_id: body for subject_id, (body, _) in rows.items()} == expected
        assert {subject_id: citation for subject_id, (_, citation) in rows.items()} == {
            subject_id: body.get("citation") for subject_id, body in expected.items()
        }

        assert _rerun_migration(live_store, _STRIP_DOCUMENTATION_MIGRATION) == [
            _STRIP_DOCUMENTATION_MIGRATION
        ]
        assert _rows() == rows
    finally:
        _delete_versions(live_store, subjects)


_PARTY_KIND_MIGRATION = "0033_output_profile_party_kind.sql"


def _profile(*fields: str) -> dict[str, object]:
    """An output profile with one role column and one enrichment entry per field, in order."""
    return {
        "schema_version": 1,
        "rows": 2,
        "columns": [{"name": "recipient", "kind": "role", "droppable": True, "distinct": 2}],
        "pivots": [],
        "enrichments": [
            {
                "column": "recipient",
                "field": field,
                "entities": 2,
                "covered": 2,
                "missing": [],
                "truncated": False,
            }
            for field in fields
        ],
    }


def _store_profile_past_the_trigger(live_store, subject_id: str, profile: object) -> None:
    """Set `subject_id`'s `output_profile` column and body member to `profile` by UPDATE, which
    0033's insert trigger does not see, as a row stored before 0033 holds it."""
    with live_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.version_record SET output_profile = %s::jsonb, "
            "immutable_body = jsonb_set(immutable_body, '{output_profile}', %s::jsonb) "
            "WHERE subject_id = %s",
            (json.dumps(profile), json.dumps(profile), subject_id),
        )


def _stored_profiles(live_store, subjects: list[str]) -> dict[str, tuple[object, object]]:
    """Each subject's `output_profile` column and body member."""
    with live_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT subject_id, output_profile, immutable_body -> 'output_profile' AS body "
            "FROM datasets.version_record WHERE subject_id = ANY(%s)",
            (subjects,),
        )
    return {cast(str, row["subject_id"]): (row["output_profile"], row["body"]) for row in rows}


def test_0033_renames_the_kind_enrichment_in_the_column_and_the_body(live_store):
    """A profile stored with a `kind` entry before 0033 has it renamed `party_kind` in the
    `output_profile` column and the body, in place, with its other entries and the column's
    own `kind` kept. A profile with no `kind` entry and a version with no profile stay as
    stored, and a rerun changes nothing."""
    suffix = uuid4().hex
    seeded = f"spec_party_kind_seeded_{suffix}"
    renamed = f"spec_party_kind_renamed_{suffix}"
    bare = f"spec_party_kind_bare_{suffix}"
    bodies = {
        seeded: {"prepared_at": "2026-10-01T00:00:00Z", "output_profile": None},
        renamed: {"prepared_at": "2026-10-01T00:00:00Z", "output_profile": _profile("party_kind")},
        bare: {"prepared_at": "2026-10-01T00:00:00Z", "output_profile": None},
    }

    for subject_id, body in bodies.items():
        _insert_as_the_previous_image(
            live_store, subject_id=subject_id, content_version="cv1", body=body
        )
    legacy = _profile("label", "kind", "iso3")
    _store_profile_past_the_trigger(live_store, seeded, legacy)
    try:
        assert _stored_profiles(live_store, [seeded]) == {seeded: (legacy, legacy)}
        assert _rerun_migration(live_store, _PARTY_KIND_MIGRATION) == [_PARTY_KIND_MIGRATION]
        stored = _stored_profiles(live_store, list(bodies))
        assert _rerun_migration(live_store, _PARTY_KIND_MIGRATION) == [_PARTY_KIND_MIGRATION]
        assert _stored_profiles(live_store, list(bodies)) == stored

        expected = _profile("label", "party_kind", "iso3")
        assert stored == {
            seeded: (expected, expected),
            renamed: (_profile("party_kind"), _profile("party_kind")),
            bare: (None, None),
        }
    finally:
        _delete_versions(live_store, list(bodies))


def test_a_kind_enrichment_inserted_after_0033_is_stored_as_party_kind(live_store):
    """The previous image keeps writing `kind` after 0033 runs. A row it inserts with only
    `immutable_body`, whose profile column 0030's trigger copies from the body, and a row it
    inserts with its facts both store `party_kind` in the column and the body."""
    suffix = uuid4().hex
    body_only = f"spec_party_kind_body_only_{suffix}"
    with_facts = f"spec_party_kind_with_facts_{suffix}"
    legacy = _profile("label", "kind", "iso3")
    body = {"prepared_at": "2026-10-01T00:00:00Z", "output_profile": legacy}
    try:
        _insert_as_the_previous_image(
            live_store, subject_id=body_only, content_version="cv1", body=body
        )
        with live_store.transaction() as tx:
            tx.execute(
                "INSERT INTO datasets.version_record "
                "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
                "recorded_at, prepared_at, output_profile) "
                "VALUES (%s, 'spec', 'cv1', 1, %s::jsonb, now(), %s, %s::jsonb)",
                (
                    with_facts,
                    json.dumps(body),
                    datetime(2026, 10, 1, tzinfo=UTC),
                    json.dumps(legacy),
                ),
            )

        expected = _profile("label", "party_kind", "iso3")
        assert _stored_profiles(live_store, [body_only, with_facts]) == {
            body_only: (expected, expected),
            with_facts: (expected, expected),
        }
    finally:
        _delete_versions(live_store, [body_only, with_facts])
