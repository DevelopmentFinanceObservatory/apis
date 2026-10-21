"""Fakes for the BigQuery reader and the GCS client that `artifact_publication` jobs read
and write through. It imports no pytest, so a release published outside pytest
(`tests.graph_release.support`) can use them; the `gs_bucket` fixture lives in
`tests.artifact_publication.fixtures`."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from google.api_core.exceptions import NotFound, PreconditionFailed

BUCKET = "test-artifact-bucket"


class FakeBigQuery:
    """`artifact_publication.jobs.BigQueryReader` fake: `responses` maps a distinctive SQL
    substring to the rows that query should return, checked in insertion
    order (first match wins). A call whose SQL matches nothing raises
    `AssertionError` -- a test wiring up too few canned responses fails
    loudly rather than silently returning nothing.
    """

    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self._responses = responses
        self.queries: list[str] = []

    def query(self, sql: str) -> list[dict[str, Any]]:
        self.queries.append(sql)
        for marker, rows in self._responses.items():
            if marker in sql:
                return rows
        raise AssertionError(f"FakeBigQuery: no canned response matches SQL:\n{sql}")

    def stream(self, sql: str) -> Iterator[dict[str, Any]]:
        return iter(self.query(sql))


_CREATED_AT = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)


def history_row(
    execution: str = "wf-1",
    *,
    status: str = "SUCCESS",
    failed: bool = False,
    created_at: datetime = _CREATED_AT,
) -> dict[str, Any]:
    """An `IngestionHistory` row as BigQuery returns it, completed ten minutes
    after it was created."""
    return {
        "WorkflowExecutionID": execution,
        "Status": status,
        "IngestionFailure": failed,
        "CreationTimestamp": created_at,
        "CompletionTimestamp": created_at + timedelta(minutes=10),
    }


def import_key_responses(
    execution: str = "wf-1",
    *,
    lock_owner: str | None = None,
    newest: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """`FakeBigQuery` responses for the import key: `execution` is the newest
    successful import, and `newest`, when given, a newer row of another
    status. The success marker precedes the history marker it contains."""
    success = history_row(execution)
    return {
        "FROM IngestionLock": [{"LockOwner": lock_owner}],
        "Status = 'SUCCESS'": [success],
        "FROM IngestionHistory": [success if newest is None else newest],
    }


@dataclass
class StoredObject:
    data: bytes
    generation: int
    time_created: datetime


class FakeBlob:
    """Stands in for `google.cloud.storage.Blob`. `_objects` is the
    bucket-level dict `_FakeBucket`/`FakeStorageClient` all share, so a
    blob handle always reads and writes the same live state a fresh
    `.blob()` call for the same name would. `generation` is what the real
    client exposes: `None` on a fresh handle, set by a download from that
    response's generation header, and set on a listed blob from the
    listing.
    """

    def __init__(
        self, objects: dict[str, StoredObject], name: str, generation: int | None = None
    ) -> None:
        self._objects = objects
        self.name = name
        self.generation = generation

    @property
    def time_created(self) -> datetime:
        return self._objects[self.name].time_created

    def download_as_bytes(self) -> bytes:
        try:
            stored = self._objects[self.name]
        except KeyError:
            raise NotFound(f"gs://{BUCKET}/{self.name} does not exist") from None
        self.generation = stored.generation
        return stored.data

    def download_to_filename(self, filename: str) -> None:
        Path(filename).write_bytes(self.download_as_bytes())

    def upload_from_string(self, data: bytes, if_generation_match: int | None = None) -> None:
        current = self._objects.get(self.name)
        if if_generation_match is not None:
            current_generation = current.generation if current is not None else 0
            if current_generation != if_generation_match:
                raise PreconditionFailed(f"generation mismatch for {self.name}")
        next_generation = (current.generation if current is not None else 0) + 1
        self._objects[self.name] = StoredObject(
            data=data, generation=next_generation, time_created=datetime.now(UTC)
        )

    def delete(self) -> None:
        try:
            del self._objects[self.name]
        except KeyError:
            raise NotFound(f"gs://{BUCKET}/{self.name} does not exist") from None


class _FakeBucket:
    def __init__(self, objects: dict[str, StoredObject]) -> None:
        self._objects = objects

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self._objects, name)


class FakeStorageClient:
    def __init__(self, buckets: dict[str, dict[str, StoredObject]]) -> None:
        self._buckets = buckets

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self._buckets.setdefault(name, {}))

    def list_blobs(self, bucket_name: str, *, prefix: str = "") -> list[FakeBlob]:
        objects = self._buckets.setdefault(bucket_name, {})
        return [
            FakeBlob(objects, name, objects[name].generation)
            for name in objects
            if name.startswith(prefix)
        ]
