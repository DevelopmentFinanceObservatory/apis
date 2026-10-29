"""Implements `ArtifactReferenceWriter` on `datasets.artifact_reference`.

References are unique by principal, digest, subject, version, and kind. A repeat pin leaves a live
row's `created_at` unchanged. A repin reactivates a removed row with a fresh timestamp. Database
failures and empty `RETURNING` results raise `DatasetsStoreUnavailable`.

`record_reference_unless_revoked` checks the revocation of the version and of its base version,
under the subject id and every id in `revoked_under`, in the upsert statement, and under
`lock_version_links` checks that the version still links the digest, so a pin completes only
for a digest the version keeps. A pin naming a served id honours revocations under it, and a
pin naming a posted id passes both ids. `remove_reference` soft-deletes a live row.
`list_live_reference_principals` is a read-only diagnostic query.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from dataset_store import DatasetStore
from dataset_store.snapshot import (
    ArtifactNotFound,
    SnapshotRevoked,
    revoking_subjects_sql,
)
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import lock_version_links
from dfo_platform.store_errors import extract_sqlstate, safe_diagnostics

# The pin door writes `pin` to the free-text kind column.
PIN_REFERENCE_KIND = "pin"

# Alerts use this marker when a pinned copy succeeds but its reference write fails. A retry writes
# the reference without copying again. Keep the marker stable for alert policies.
ARTIFACT_REFERENCE_WRITE_FAILED_MARKER = "artifact_reference_write_failed"


def _revocations(
    subject_id: str, content_version: str, revoked_under: Sequence[str]
) -> tuple[str, tuple[str, ...]]:
    """The `FROM ... WHERE` clause selecting the revocations of a version and of the base
    version it collapses from, under `subject_id` and every id in `revoked_under`, and its
    parameters."""
    subjects = tuple(dict.fromkeys((subject_id, *revoked_under)))
    sql = (
        "FROM datasets.version_revocation "
        f"WHERE {revoking_subjects_sql(len(subjects))} AND (content_version = %s "
        "OR content_version = (SELECT base_content_version FROM datasets.version_record "
        "WHERE subject_id = %s AND content_version = %s))"
    )
    return sql, (*subjects, content_version, subject_id, content_version)


@dataclass(frozen=True)
class ReferenceWrite:
    created_at: datetime
    changed: bool  # True when this call inserted or reactivated the reference.


class LiveArtifactReferenceWriter:
    """`dataset_store.snapshot.ArtifactReferenceWriter` over a `DatasetStore`."""

    def __init__(self, *, store: DatasetStore) -> None:
        self._store = store

    def record_reference(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> datetime:
        failure: str | None = None
        sqlstate: str | None = None
        row = None
        try:
            with self._store.transaction() as tx:
                # Live rows keep their timestamp. Reactivated rows get this call's timestamp.
                # Qualify target columns because Postgres also exposes `excluded` in the update.
                row = tx.fetch_one(
                    "INSERT INTO datasets.artifact_reference "
                    "(principal, artifact_digest, subject_id, content_version, reference_kind, "
                    "created_at) VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (principal, artifact_digest, subject_id, content_version, "
                    "reference_kind) DO UPDATE SET "
                    "created_at = CASE WHEN datasets.artifact_reference.removed_at IS NULL "
                    "THEN datasets.artifact_reference.created_at "
                    "ELSE excluded.created_at END, "
                    "removed_at = NULL "
                    "RETURNING created_at",
                    (principal, artifact_digest, subject_id, content_version, reference_kind, now),
                )
        except Exception as exc:
            # Raise after leaving the handler so the wrapped error has no `__context__`.
            failure = (
                f"recording artifact_reference for principal {principal!r} artifact_digest "
                f"{artifact_digest!r} subject_id {subject_id!r} content_version "
                f"{content_version!r} failed: {safe_diagnostics(exc)}"
            )
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        if row is None:
            # An empty `RETURNING` result indicates a store anomaly. Report it as a typed error.
            raise DatasetsStoreUnavailable(
                f"recording artifact_reference for principal {principal!r} artifact_digest "
                f"{artifact_digest!r} subject_id {subject_id!r} content_version "
                f"{content_version!r} returned no row"
            )
        return cast(datetime, row["created_at"])

    def record_reference_unless_revoked(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
        revoked_under: Sequence[str] = (),
    ) -> datetime:
        return self.claim_reference_unless_revoked(
            principal=principal,
            artifact_digest=artifact_digest,
            subject_id=subject_id,
            content_version=content_version,
            reference_kind=reference_kind,
            now=now,
            revoked_under=revoked_under,
        ).created_at

    def claim_reference_unless_revoked(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
        revoked_under: Sequence[str] = (),
    ) -> ReferenceWrite:
        failure: str | None = None
        revocations_sql, revocations_params = _revocations(
            subject_id, content_version, revoked_under
        )
        sqlstate: str | None = None
        row = None
        changed = False
        try:
            with self._store.transaction() as tx:
                # Under the lock, a re-render either replaced the link already, and this refuses,
                # or sees the pinned location this reference's caller wrote, and keeps the link.
                lock_version_links(tx, subject_id=subject_id, content_version=content_version)
                linked = tx.fetch_one(
                    "SELECT 1 AS linked FROM datasets.version_artifact "
                    "WHERE subject_id = %s AND content_version = %s AND artifact_digest = %s "
                    "LIMIT 1",
                    (subject_id, content_version, artifact_digest),
                )
                if linked is None:
                    raise ArtifactNotFound(
                        id=subject_id,
                        content_version=content_version,
                        artifact_digest=artifact_digest,
                    )
                # Revocation check and upsert share a statement snapshot, and the check covers
                # the version and its base version. Removed rows reactivate. Live rows remain
                # unchanged and return no row.
                row = tx.fetch_one(
                    "INSERT INTO datasets.artifact_reference "
                    "(principal, artifact_digest, subject_id, content_version, reference_kind, "
                    "created_at) "
                    "SELECT %s, %s, %s, %s, %s, %s "
                    f"WHERE NOT EXISTS (SELECT 1 {revocations_sql}) "
                    "ON CONFLICT (principal, artifact_digest, subject_id, content_version, "
                    "reference_kind) DO UPDATE SET "
                    "created_at = excluded.created_at, removed_at = NULL "
                    "WHERE datasets.artifact_reference.removed_at IS NOT NULL "
                    "RETURNING created_at",
                    (
                        principal,
                        artifact_digest,
                        subject_id,
                        content_version,
                        reference_kind,
                        now,
                        *revocations_params,
                    ),
                )
                if row is None:
                    # The version's own revocation wins over its base version's, and one under
                    # `subject_id` over one under another id.
                    revoked = tx.fetch_one(
                        f"SELECT reason {revocations_sql} "
                        "ORDER BY CASE WHEN content_version = %s THEN 0 ELSE 1 END, "
                        "CASE WHEN subject_id = %s THEN 0 ELSE 1 END LIMIT 1",
                        (*revocations_params, content_version, subject_id),
                    )
                    if revoked is not None:
                        raise SnapshotRevoked(
                            id=subject_id,
                            content_version=content_version,
                            reason=cast("str | None", revoked["reason"]),
                        )
                    row = tx.fetch_one(
                        "SELECT created_at FROM datasets.artifact_reference "
                        "WHERE principal = %s AND artifact_digest = %s AND subject_id = %s "
                        "AND content_version = %s AND reference_kind = %s "
                        "AND removed_at IS NULL",
                        (
                            principal,
                            artifact_digest,
                            subject_id,
                            content_version,
                            reference_kind,
                        ),
                    )
                else:
                    changed = True
        except (SnapshotRevoked, ArtifactNotFound):
            # Raised inside the `with` block above before any write, or after the insert
            # statement's WHERE NOT EXISTS suppressed it, so the rollback discards nothing.
            raise
        except Exception as exc:
            # Raise after leaving the handler so the wrapped error has no `__context__`.
            failure = (
                f"recording artifact_reference for principal {principal!r} artifact_digest "
                f"{artifact_digest!r} subject_id {subject_id!r} content_version "
                f"{content_version!r} failed: {safe_diagnostics(exc)}"
            )
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        if row is None:
            # Non-revoked writes should return a row. This typed error catches store anomalies.
            raise DatasetsStoreUnavailable(
                f"recording artifact_reference for principal {principal!r} artifact_digest "
                f"{artifact_digest!r} subject_id {subject_id!r} content_version "
                f"{content_version!r} returned no row"
            )
        return ReferenceWrite(created_at=cast(datetime, row["created_at"]), changed=changed)

    def remove_reference(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> None:
        failure: str | None = None
        sqlstate: str | None = None
        try:
            with self._store.transaction() as tx:
                # A zero-row update is a no-op. `record_reference` requires `RETURNING`.
                tx.execute(
                    "UPDATE datasets.artifact_reference SET removed_at = %s "
                    "WHERE principal = %s AND artifact_digest = %s AND subject_id = %s "
                    "AND content_version = %s AND reference_kind = %s AND removed_at IS NULL",
                    (now, principal, artifact_digest, subject_id, content_version, reference_kind),
                )
        except Exception as exc:
            failure = (
                f"removing artifact_reference for principal {principal!r} artifact_digest "
                f"{artifact_digest!r} subject_id {subject_id!r} content_version "
                f"{content_version!r} failed: {safe_diagnostics(exc)}"
            )
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)


def list_live_reference_principals(*, store: DatasetStore, reference_kind: str) -> tuple[str, ...]:
    """Returns principals with live references of this kind.

    Combine with `list_orphaned_presentation_references` to find drafts with absent components.
    The databases are queried separately."""
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT DISTINCT principal FROM datasets.artifact_reference "
            "WHERE reference_kind = %s AND removed_at IS NULL",
            (reference_kind,),
        )
    return tuple(str(row["principal"]) for row in rows)
