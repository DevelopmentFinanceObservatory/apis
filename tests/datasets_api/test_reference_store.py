"""Exercises `LiveArtifactReferenceWriter` SQL against the SQLite-backed fake store."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime

import pytest

from dataset_store.snapshot import ArtifactNotFound, SnapshotRevoked
from datasets_api.reference_store import (
    PIN_REFERENCE_KIND,
    LiveArtifactReferenceWriter,
    list_live_reference_principals,
)
from datasets_api.store_errors import DatasetsStoreUnavailable

from ._store_fault_doubles import DriverExplodingFakeDatasetStore, SqlstateExplodingFakeDatasetStore
from ._version_record_store import FakeVersionRecordDatasetStore

_PRINCIPAL = "first-party-key"
_DIGEST = "sha256:aaa"
_SUBJECT_ID = "spec_a"
_CONTENT_VERSION = "cv1"
_FIRST_PIN = datetime(2026, 1, 1, tzinfo=UTC)
_LATER = datetime(2026, 2, 1, tzinfo=UTC)


def _link(store: FakeVersionRecordDatasetStore, *, digest: str) -> None:
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_artifact "
            "(subject_id, content_version, artifact_digest, rendering_key, rendering, "
            "media_type) VALUES (%s, %s, %s, %s, %s, %s)",
            (_SUBJECT_ID, _CONTENT_VERSION, digest, f"rk-{digest}", {"format": "csv"}, "text/csv"),
        )


@pytest.fixture
def store():
    """A store whose version links `_DIGEST`, which the guarded reference writes require."""
    dataset_store = FakeVersionRecordDatasetStore()
    _link(dataset_store, digest=_DIGEST)
    try:
        yield dataset_store
    finally:
        dataset_store.close()


def _record(store: FakeVersionRecordDatasetStore, *, now: datetime) -> datetime:
    writer = LiveArtifactReferenceWriter(store=store)
    return writer.record_reference(
        principal=_PRINCIPAL,
        artifact_digest=_DIGEST,
        subject_id=_SUBJECT_ID,
        content_version=_CONTENT_VERSION,
        reference_kind=PIN_REFERENCE_KIND,
        now=now,
    )


def _row(store: FakeVersionRecordDatasetStore) -> dict[str, object]:
    with store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT created_at, removed_at FROM datasets.artifact_reference "
            "WHERE principal = %s AND artifact_digest = %s AND subject_id = %s "
            "AND content_version = %s AND reference_kind = %s",
            (_PRINCIPAL, _DIGEST, _SUBJECT_ID, _CONTENT_VERSION, PIN_REFERENCE_KIND),
        )
    assert row is not None
    return row


def test_a_fresh_pin_returns_its_own_created_at(store):
    returned = _record(store, now=_FIRST_PIN)

    assert returned == _FIRST_PIN
    row = _row(store)
    assert row["created_at"] == _FIRST_PIN
    assert row["removed_at"] is None


def test_a_repeat_pin_of_a_still_live_row_returns_the_original_created_at(store):
    first = _record(store, now=_FIRST_PIN)

    second = _record(store, now=_LATER)

    assert second == first == _FIRST_PIN
    row = _row(store)
    assert row["created_at"] == _FIRST_PIN  # A repeat pin preserves the first timestamp.
    assert row["removed_at"] is None


def test_repinning_a_soft_deleted_row_reactivates_it_with_a_fresh_created_at(store):
    _record(store, now=_FIRST_PIN)
    # No route on this door sets removed_at -- unpinning is a first-party
    # operation on another surface -- so this test reaches into the store
    # directly to soft-delete the row the same way that operation would.
    with store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_reference SET removed_at = %s "
            "WHERE principal = %s AND artifact_digest = %s AND subject_id = %s "
            "AND content_version = %s AND reference_kind = %s",
            (_LATER, _PRINCIPAL, _DIGEST, _SUBJECT_ID, _CONTENT_VERSION, PIN_REFERENCE_KIND),
        )

    reactivated_at = datetime(2026, 3, 1, tzinfo=UTC)
    returned = _record(store, now=reactivated_at)

    assert returned == reactivated_at  # Reactivation sets a new timestamp.
    row = _row(store)
    assert row["created_at"] == reactivated_at
    assert row["removed_at"] is None  # live again


def test_record_reference_unless_revoked_writes_normally_when_nothing_is_revoked(store):
    writer = LiveArtifactReferenceWriter(store=store)

    returned = writer.record_reference_unless_revoked(
        principal=_PRINCIPAL,
        artifact_digest=_DIGEST,
        subject_id=_SUBJECT_ID,
        content_version=_CONTENT_VERSION,
        reference_kind=PIN_REFERENCE_KIND,
        now=_FIRST_PIN,
    )

    assert returned == _FIRST_PIN
    row = _row(store)
    assert row["created_at"] == _FIRST_PIN
    assert row["removed_at"] is None


def test_claim_reference_reports_whether_it_changes_the_row(store):
    writer = LiveArtifactReferenceWriter(store=store)

    first = writer.claim_reference_unless_revoked(
        principal=_PRINCIPAL,
        artifact_digest=_DIGEST,
        subject_id=_SUBJECT_ID,
        content_version=_CONTENT_VERSION,
        reference_kind=PIN_REFERENCE_KIND,
        now=_FIRST_PIN,
    )
    second = writer.claim_reference_unless_revoked(
        principal=_PRINCIPAL,
        artifact_digest=_DIGEST,
        subject_id=_SUBJECT_ID,
        content_version=_CONTENT_VERSION,
        reference_kind=PIN_REFERENCE_KIND,
        now=_LATER,
    )

    assert first.created_at == second.created_at == _FIRST_PIN
    assert first.changed is True
    assert second.changed is False


@pytest.mark.parametrize(
    "revoked_version", [_CONTENT_VERSION, "cv_base"], ids=["the-version", "its-base-version"]
)
def test_record_reference_unless_revoked_refuses_and_writes_zero_rows_when_already_revoked(
    store, revoked_version
):
    """A revocation of the version, or of the base version it collapses from, that lands after
    the pin door's read refuses the write."""
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at, base_content_version) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (_SUBJECT_ID, "spec", _CONTENT_VERSION, 1, {}, _FIRST_PIN, "cv_base"),
        )
        tx.execute(
            "INSERT INTO datasets.version_revocation "
            "(subject_id, content_version, reason, revoked_at) VALUES (%s, %s, %s, %s)",
            (_SUBJECT_ID, revoked_version, "takedown", _FIRST_PIN),
        )
    writer = LiveArtifactReferenceWriter(store=store)

    with pytest.raises(SnapshotRevoked, match="takedown"):
        writer.record_reference_unless_revoked(
            principal=_PRINCIPAL,
            artifact_digest=_DIGEST,
            subject_id=_SUBJECT_ID,
            content_version=_CONTENT_VERSION,
            reference_kind=PIN_REFERENCE_KIND,
            now=_FIRST_PIN,
        )

    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT * FROM datasets.artifact_reference "
            "WHERE subject_id = %s AND content_version = %s",
            (_SUBJECT_ID, _CONTENT_VERSION),
        )
    assert rows == []


def test_record_reference_unless_revoked_refuses_a_digest_the_version_no_longer_links(store):
    """A re-render that replaced the link before the reference write leaves nothing to pin."""
    with store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.version_artifact SET artifact_digest = %s WHERE artifact_digest = %s",
            ("sha256:rerendered", _DIGEST),
        )
    writer = LiveArtifactReferenceWriter(store=store)

    with pytest.raises(ArtifactNotFound):
        writer.record_reference_unless_revoked(
            principal=_PRINCIPAL,
            artifact_digest=_DIGEST,
            subject_id=_SUBJECT_ID,
            content_version=_CONTENT_VERSION,
            reference_kind=PIN_REFERENCE_KIND,
            now=_FIRST_PIN,
        )

    with store.transaction() as tx:
        rows = tx.fetch_all("SELECT * FROM datasets.artifact_reference")
    assert rows == []


def test_a_driver_failure_raises_datasets_store_unavailable_with_no_leaked_context():
    """A driver failure becomes `DatasetsStoreUnavailable` without exception context or text."""
    distinctive_driver_text = "distinctive_driver_boom_reference_write_9c2e"
    store = DriverExplodingFakeDatasetStore(message=distinctive_driver_text)
    writer = LiveArtifactReferenceWriter(store=store)

    with pytest.raises(DatasetsStoreUnavailable) as excinfo:
        writer.record_reference(
            principal=_PRINCIPAL,
            artifact_digest=_DIGEST,
            subject_id=_SUBJECT_ID,
            content_version=_CONTENT_VERSION,
            reference_kind=PIN_REFERENCE_KIND,
            now=_FIRST_PIN,
        )

    exc = excinfo.value
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert distinctive_driver_text not in str(exc)


def test_record_reference_driver_failure_carrying_a_sqlstate_attaches_it():
    """A pg8000-shaped driver exception's own SQLSTATE reaches
    `DatasetsStoreUnavailable.sqlstate` -- see `datasets_api.store_errors
    .extract_sqlstate`.
    """
    store = SqlstateExplodingFakeDatasetStore(sqlstate="23503")
    writer = LiveArtifactReferenceWriter(store=store)

    with pytest.raises(DatasetsStoreUnavailable) as excinfo:
        writer.record_reference(
            principal=_PRINCIPAL,
            artifact_digest=_DIGEST,
            subject_id=_SUBJECT_ID,
            content_version=_CONTENT_VERSION,
            reference_kind=PIN_REFERENCE_KIND,
            now=_FIRST_PIN,
        )

    assert excinfo.value.sqlstate == "23503"


def test_record_reference_unless_revoked_driver_failure_carrying_a_sqlstate_attaches_it():
    store = SqlstateExplodingFakeDatasetStore(sqlstate="23503")
    writer = LiveArtifactReferenceWriter(store=store)

    with pytest.raises(DatasetsStoreUnavailable) as excinfo:
        writer.record_reference_unless_revoked(
            principal=_PRINCIPAL,
            artifact_digest=_DIGEST,
            subject_id=_SUBJECT_ID,
            content_version=_CONTENT_VERSION,
            reference_kind=PIN_REFERENCE_KIND,
            now=_FIRST_PIN,
        )

    assert excinfo.value.sqlstate == "23503"


class _NoRowTransaction:
    def execute(self, sql: str, params: object = ()) -> None:
        pass

    def fetch_one(self, sql: str, params: object = ()) -> object:
        return None

    def fetch_all(self, sql: str, params: object = ()) -> object:
        return []


class _NoRowFakeDatasetStore:
    """A `DatasetStore` whose every query succeeds but returns no row --
    reproduces the anomaly of an `INSERT ... RETURNING` that raises nothing
    yet yields an empty result, without a real driver defect to trigger it.
    """

    @contextmanager
    def transaction(self):
        yield _NoRowTransaction()


def test_an_empty_returning_raises_datasets_store_unavailable_instead_of_asserting():
    writer = LiveArtifactReferenceWriter(store=_NoRowFakeDatasetStore())

    with pytest.raises(DatasetsStoreUnavailable):
        writer.record_reference(
            principal=_PRINCIPAL,
            artifact_digest=_DIGEST,
            subject_id=_SUBJECT_ID,
            content_version=_CONTENT_VERSION,
            reference_kind=PIN_REFERENCE_KIND,
            now=_FIRST_PIN,
        )


# -- remove_reference -----------------------------------------------------


def test_remove_reference_soft_deletes_a_live_row(store):
    _record(store, now=_FIRST_PIN)
    writer = LiveArtifactReferenceWriter(store=store)

    writer.remove_reference(
        principal=_PRINCIPAL,
        artifact_digest=_DIGEST,
        subject_id=_SUBJECT_ID,
        content_version=_CONTENT_VERSION,
        reference_kind=PIN_REFERENCE_KIND,
        now=_LATER,
    )

    row = _row(store)
    assert row["removed_at"] == _LATER
    assert row["created_at"] == _FIRST_PIN  # Soft deletion preserves the creation time.


def test_holds_reference_counts_a_removed_row_and_only_its_own_version(store):
    writer = LiveArtifactReferenceWriter(store=store)

    def held() -> tuple[bool, bool]:
        return (
            writer.holds_reference(subject_id=_SUBJECT_ID, content_version=_CONTENT_VERSION),
            writer.holds_reference(subject_id=_SUBJECT_ID, content_version="cv2"),
        )

    before = held()
    _record(store, now=_FIRST_PIN)
    recorded = held()
    writer.remove_reference(
        principal=_PRINCIPAL,
        artifact_digest=_DIGEST,
        subject_id=_SUBJECT_ID,
        content_version=_CONTENT_VERSION,
        reference_kind=PIN_REFERENCE_KIND,
        now=_LATER,
    )

    assert (before, recorded, held()) == ((False, False), (True, False), (True, False))


def test_remove_reference_is_a_no_op_when_no_row_matches(store):
    # Removing an absent reference is idempotent.
    writer = LiveArtifactReferenceWriter(store=store)

    writer.remove_reference(
        principal=_PRINCIPAL,
        artifact_digest=_DIGEST,
        subject_id=_SUBJECT_ID,
        content_version=_CONTENT_VERSION,
        reference_kind=PIN_REFERENCE_KIND,
        now=_LATER,
    )

    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT * FROM datasets.artifact_reference "
            "WHERE subject_id = %s AND content_version = %s",
            (_SUBJECT_ID, _CONTENT_VERSION),
        )
    assert rows == []


def test_remove_reference_does_not_touch_an_already_removed_row(store):
    _record(store, now=_FIRST_PIN)
    writer = LiveArtifactReferenceWriter(store=store)
    writer.remove_reference(
        principal=_PRINCIPAL,
        artifact_digest=_DIGEST,
        subject_id=_SUBJECT_ID,
        content_version=_CONTENT_VERSION,
        reference_kind=PIN_REFERENCE_KIND,
        now=_LATER,
    )

    # A repeated removal preserves the original removal time.
    even_later = datetime(2026, 3, 1, tzinfo=UTC)
    writer.remove_reference(
        principal=_PRINCIPAL,
        artifact_digest=_DIGEST,
        subject_id=_SUBJECT_ID,
        content_version=_CONTENT_VERSION,
        reference_kind=PIN_REFERENCE_KIND,
        now=even_later,
    )

    row = _row(store)
    assert row["removed_at"] == _LATER


def test_remove_reference_driver_failure_raises_datasets_store_unavailable_with_no_leaked_context():
    distinctive_driver_text = "distinctive_driver_boom_reference_remove_4a1f"
    store = DriverExplodingFakeDatasetStore(message=distinctive_driver_text)
    writer = LiveArtifactReferenceWriter(store=store)

    with pytest.raises(DatasetsStoreUnavailable) as excinfo:
        writer.remove_reference(
            principal=_PRINCIPAL,
            artifact_digest=_DIGEST,
            subject_id=_SUBJECT_ID,
            content_version=_CONTENT_VERSION,
            reference_kind=PIN_REFERENCE_KIND,
            now=_FIRST_PIN,
        )

    exc = excinfo.value
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert distinctive_driver_text not in str(exc)


def test_remove_reference_driver_failure_carrying_a_sqlstate_attaches_it():
    store = SqlstateExplodingFakeDatasetStore(sqlstate="23503")
    writer = LiveArtifactReferenceWriter(store=store)

    with pytest.raises(DatasetsStoreUnavailable) as excinfo:
        writer.remove_reference(
            principal=_PRINCIPAL,
            artifact_digest=_DIGEST,
            subject_id=_SUBJECT_ID,
            content_version=_CONTENT_VERSION,
            reference_kind=PIN_REFERENCE_KIND,
            now=_FIRST_PIN,
        )

    assert excinfo.value.sqlstate == "23503"


# -- list_live_reference_principals ----------------------------------------


def test_list_live_reference_principals_returns_only_live_rows_under_the_kind(store):
    writer = LiveArtifactReferenceWriter(store=store)
    writer.record_reference(
        principal="presentation:cmp_live",
        artifact_digest=_DIGEST,
        subject_id=_SUBJECT_ID,
        content_version=_CONTENT_VERSION,
        reference_kind="presentation_draft",
        now=_FIRST_PIN,
    )
    writer.record_reference(
        principal="presentation:cmp_removed",
        artifact_digest=_DIGEST,
        subject_id=_SUBJECT_ID,
        content_version=_CONTENT_VERSION,
        reference_kind="presentation_draft",
        now=_FIRST_PIN,
    )
    writer.remove_reference(
        principal="presentation:cmp_removed",
        artifact_digest=_DIGEST,
        subject_id=_SUBJECT_ID,
        content_version=_CONTENT_VERSION,
        reference_kind="presentation_draft",
        now=_LATER,
    )
    # A live row of another kind is excluded.
    writer.record_reference(
        principal=_PRINCIPAL,
        artifact_digest=_DIGEST,
        subject_id=_SUBJECT_ID,
        content_version=_CONTENT_VERSION,
        reference_kind=PIN_REFERENCE_KIND,
        now=_FIRST_PIN,
    )

    principals = list_live_reference_principals(store=store, reference_kind="presentation_draft")

    assert principals == ("presentation:cmp_live",)


def test_list_live_reference_principals_empty_for_an_unused_kind(store):
    assert list_live_reference_principals(store=store, reference_kind="presentation_draft") == ()
