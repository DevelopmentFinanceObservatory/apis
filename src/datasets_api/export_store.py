"""The export store keeps one row per prepared export in `datasets.export`, keyed by its
`export_id` (`exports.export_id`).

A row names the version the export renders under the id its record is keyed by. It also holds
the pinned copy of the file by digest, bucket and object name, which the export link signs
without another read, and `body`, the facts the prepare answered (`exports.ExportResponse`),
sources and licences included, as they stood when the file was prepared.

`write_export` stores a row once. A write under an id that already holds a row changes nothing
and returns the stored row, so two concurrent prepares of one request answer one record.

Store errors become `DatasetsStoreUnavailable` after the transaction exits.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from dataset_store import DatasetStore
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import Clock, utc_now
from dfo_platform.store_errors import extract_sqlstate, safe_diagnostics


@dataclass(frozen=True, kw_only=True)
class ExportRecord:
    """One prepared export: the version it renders, the pinned file's location and the prepare's
    facts. `subject_id` is the id the version is recorded under, the served id for a dataset
    the request named by a posted id, while `body` names the frame as the request did."""

    export_id: str
    subject_id: str
    content_version: str
    artifact_digest: str
    bucket: str
    object_name: str
    body: Mapping[str, object]


class ExportStore(Protocol):
    def read_export(self, export_id: str) -> ExportRecord | None:
        """Return the export recorded under `export_id`, or `None`."""
        ...

    def write_export(self, record: ExportRecord) -> ExportRecord:
        """Store `record` once under its id and return the row the id holds: `record`, or the
        one an earlier write stored."""
        ...


_COLUMNS = "export_id, subject_id, content_version, artifact_digest, bucket, object_name, body"


class LiveExportStore:
    def __init__(self, *, store: DatasetStore, now: Clock = utc_now) -> None:
        self._store = store
        self._now = now

    def read_export(self, export_id: str) -> ExportRecord | None:
        failure: str | None = None
        sqlstate: str | None = None
        row = None
        try:
            with self._store.transaction() as tx:
                row = tx.fetch_one(
                    f"SELECT {_COLUMNS} FROM datasets.export WHERE export_id = %s", (export_id,)
                )
        except Exception as exc:
            failure = f"reading export {export_id!r} failed: {safe_diagnostics(exc)}"
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        return None if row is None else _record(row)

    def write_export(self, record: ExportRecord) -> ExportRecord:
        failure: str | None = None
        sqlstate: str | None = None
        row = None
        try:
            with self._store.transaction() as tx:
                # `ON CONFLICT DO NOTHING` waits for a concurrent inserter to commit, and the read
                # after it, its own statement under READ COMMITTED, sees the row that won.
                tx.execute(
                    f"INSERT INTO datasets.export ({_COLUMNS}, recorded_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (
                        record.export_id,
                        record.subject_id,
                        record.content_version,
                        record.artifact_digest,
                        record.bucket,
                        record.object_name,
                        dict(record.body),
                        self._now(),
                    ),
                )
                row = tx.fetch_one(
                    f"SELECT {_COLUMNS} FROM datasets.export WHERE export_id = %s",
                    (record.export_id,),
                )
        except Exception as exc:
            failure = f"writing export {record.export_id!r} failed: {safe_diagnostics(exc)}"
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        # Export rows are never deleted, so the id holds a row after the insert.
        assert row is not None
        return _record(row)


def _record(row: Mapping[str, object]) -> ExportRecord:
    return ExportRecord(
        export_id=cast(str, row["export_id"]),
        subject_id=cast(str, row["subject_id"]),
        content_version=cast(str, row["content_version"]),
        artifact_digest=cast(str, row["artifact_digest"]),
        bucket=cast(str, row["bucket"]),
        object_name=cast(str, row["object_name"]),
        body=cast(Mapping[str, object], row["body"]),
    )
