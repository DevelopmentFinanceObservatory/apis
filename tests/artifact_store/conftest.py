"""Test doubles for the artifact store, GCS, inventory, and inventory SQL."""

from __future__ import annotations

import hashlib
import io
import re
import sqlite3
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from artifact_store import signing
from artifact_store.errors import ArtifactNotFoundError, ArtifactRecordConflictError
from artifact_store.inventory import ArtifactLocation, InventoryBatch
from artifact_store.signing import SIGNED_URL_TTL_CEILING, SIGNED_URL_TTL_FLOOR
from artifact_store.store import SPILL_THRESHOLD_BYTES, ArtifactSink, StoredArtifact, Tier


@pytest.fixture(autouse=True)
def _fresh_process_credentials(monkeypatch):
    """Give each test its own ADC cache, so a test's stubbed `google.auth.default` is the one
    signing discovers."""
    monkeypatch.setattr(signing, "_process_credentials", signing._ProcessCredentials())


class _BufferedSink(io.BytesIO):
    """A plain in-memory `ArtifactSink` stand-in: `io.BytesIO` already
    satisfies `write`/`tell`/`flush`/`readable`/`writable`/`seekable`, so
    only `result()` needs adding, set once by the fake's own `open_put`
    once its `with` block exits without raising.
    """

    def __init__(self) -> None:
        super().__init__()
        self._result: StoredArtifact | None = None

    def _set_result(self, result: StoredArtifact) -> None:
        self._result = result

    def result(self) -> StoredArtifact:
        if self._result is None:
            raise RuntimeError(
                "ArtifactSink.result() called before its owning open_put() block committed"
            )
        return self._result


class FakeArtifactStore:
    """In-memory store keyed by `(tier, digest)`; records calls in order.

    Repeated puts deduplicate within a tier. Reads return stored bytes or
    raise `ArtifactNotFoundError`; this fake cannot model digest corruption.
    Signed URLs are deterministic, enforce TTL bounds, and record response
    headers. Bucket and object names are derived, so use `LiveArtifactStore`
    with `RecordingArtifactInventory` to test location drift or conflicts.
    """

    def __init__(self) -> None:
        self.objects: dict[tuple[Tier, str], bytes] = {}
        self.calls: list[tuple[str, Tier, str]] = []
        self.signed_headers: dict[tuple[Tier, str], tuple[str, str]] = {}

    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        self.calls.append(("put", tier, digest))
        key = (tier, digest)
        deduplicated = key in self.objects
        if not deduplicated:
            self.objects[key] = data
        return StoredArtifact(
            digest=digest,
            tier=tier,
            bucket=f"fake-{tier.value}-bucket",
            object_name=digest,
            size_bytes=len(data),
            deduplicated=deduplicated,
        )

    @contextmanager
    def open_put(self, *, tier: Tier) -> Generator[ArtifactSink, None, None]:
        """Buffer writes and call `put()` on clean exit; no staging path is modelled."""
        sink = _BufferedSink()
        yield sink
        sink._set_result(self.put(sink.getvalue(), tier=tier))

    def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
        self.calls.append(("get", tier, artifact_digest))
        key = (tier, artifact_digest)
        if key not in self.objects:
            raise ArtifactNotFoundError(digest=artifact_digest, tier=tier)
        return self.objects[key]

    def signed_location_url(
        self,
        digest: str,
        *,
        tier: Tier,
        bucket: str,
        object_name: str,
        ttl: timedelta,
        response_type: str,
        response_disposition: str,
    ) -> str:
        self.calls.append(("signed_location_url", tier, digest))
        if ttl < SIGNED_URL_TTL_FLOOR or ttl > SIGNED_URL_TTL_CEILING:
            raise ValueError(
                f"ttl {ttl!r} is out of bounds "
                f"[{SIGNED_URL_TTL_FLOOR!r}, {SIGNED_URL_TTL_CEILING!r}]"
            )
        self.signed_headers[(tier, digest)] = (response_type, response_disposition)
        expires = int(ttl.total_seconds())
        return (
            f"https://storage.googleapis.com/fake-{tier.value}-bucket/{digest}"
            f"?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Expires={expires}"
            "&X-Goog-Signature=fake0000signature"
        )

    def deferred(self, batch: InventoryBatch) -> FakeArtifactStore:
        """This store: it records no inventory, so a write owes no row."""
        return self


class RecordingArtifactInventory:
    """Records accepted inventory calls and returns locations from them.

    Conflicting size, bucket, or object names raise
    `ArtifactRecordConflictError`; identical repeats remain separate calls.
    `uploaded` is recorded but no row state is modelled: every location reads as stored
    at construction time, so it is never past retention, and a `renew` call replaces it.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.renewals: list[dict[str, object]] = []
        self._stored_at = datetime.now(UTC)

    def record(
        self,
        *,
        digest: str,
        tier: Tier,
        bucket: str,
        object_name: str,
        size_bytes: int,
        uploaded: bool = True,
        stored_at: datetime | None = None,
    ) -> None:
        stored_for_digest = next((c for c in self.calls if c["digest"] == digest), None)
        stored_size = stored_for_digest["size_bytes"] if stored_for_digest is not None else None
        stored_for_tier = next(
            (c for c in self.calls if c["digest"] == digest and c["tier"] == tier), None
        )
        stored_bucket = stored_for_tier["bucket"] if stored_for_tier is not None else None
        stored_object_name = stored_for_tier["object_name"] if stored_for_tier is not None else None
        conflicts = (stored_size is not None and stored_size != size_bytes) or (
            stored_for_tier is not None
            and (stored_bucket != bucket or stored_object_name != object_name)
        )
        if conflicts:
            raise ArtifactRecordConflictError(
                digest=digest,
                tier=tier,
                stored_bucket=stored_bucket if stored_bucket is not None else bucket,
                stored_object_name=(
                    stored_object_name if stored_object_name is not None else object_name
                ),
                stored_size_bytes=stored_size if stored_size is not None else size_bytes,
                incoming_bucket=bucket,
                incoming_object_name=object_name,
                incoming_size_bytes=size_bytes,
            )
        # Either a fresh (digest, tier) or an identical repeat of one
        # already recorded, not a conflict; either way this call is
        # appended, so `calls` keeps tracking every accepted invocation,
        # not distinct rows (a test counting concurrent-writer calls
        # relies on that).
        self.calls.append(
            {
                "digest": digest,
                "tier": tier,
                "bucket": bucket,
                "object_name": object_name,
                "size_bytes": size_bytes,
                "uploaded": uploaded,
            }
        )

    def renew(
        self,
        *,
        digest: str,
        tier: Tier,
        bucket: str,
        object_name: str,
        size_bytes: int,
        renewal: int,
        stored_at: datetime | None = None,
    ) -> None:
        self.renewals.append(
            {
                "digest": digest,
                "tier": tier,
                "bucket": bucket,
                "object_name": object_name,
                "renewal": renewal,
            }
        )

    def get_location(self, digest: str, *, tier: Tier) -> ArtifactLocation | None:
        for renewal in reversed(self.renewals):
            if renewal["digest"] == digest and renewal["tier"] == tier:
                return ArtifactLocation(
                    bucket=renewal["bucket"],
                    object_name=renewal["object_name"],
                    stored_at=self._stored_at,
                    renewal=renewal["renewal"],
                )
        for call in self.calls:
            if call["digest"] == digest and call["tier"] == tier:
                return ArtifactLocation(
                    bucket=call["bucket"],
                    object_name=call["object_name"],
                    stored_at=self._stored_at,
                    renewal=0,
                )
        return None


class FakeGCSBlob:
    def __init__(self, bucket: FakeGCSBucket, name: str) -> None:
        self._bucket = bucket
        self.name = name
        # A landed upload stores this value; new blobs read the stored value back.
        self.cache_control: str | None = bucket.cache_control.get(name)

    def upload_from_string(
        self,
        data: bytes,
        content_type: str | None = None,
        if_generation_match: int | None = None,
    ) -> None:
        from google.api_core.exceptions import PreconditionFailed

        if self._bucket.raise_on_next_upload is not None:
            exc = self._bucket.raise_on_next_upload
            self._bucket.raise_on_next_upload = None
            raise exc
        exists = self.name in self._bucket.objects
        if if_generation_match == 0 and exists:
            raise PreconditionFailed(f"object {self.name!r} already exists")
        self._bucket.objects[self.name] = data
        self._bucket.time_created[self.name] = datetime.now(UTC)
        self._bucket.upload_log.append(self.name)
        self._bucket.cache_control[self.name] = self.cache_control

    @property
    def time_created(self) -> datetime | None:
        """Return when the object landed, or `None` when it is absent."""
        return self._bucket.time_created.get(self.name)

    @property
    def size(self) -> int | None:
        """Return the stored byte length, or `None` when the object is absent."""
        data = self._bucket.objects.get(self.name)
        return len(data) if data is not None else None

    def download_as_bytes(self) -> bytes:
        from google.api_core.exceptions import NotFound

        data = self._bucket.objects.get(self.name)
        if data is None:
            raise NotFound(f"object {self.name!r} not found in bucket {self._bucket.name!r}")
        return data

    def open(
        self,
        mode: str,
        *,
        chunk_size: int | None = None,
        ignore_flush: bool = False,
        content_type: str | None = None,
        if_generation_match: int | None = None,
    ) -> FakeGCSBlobWriter:
        """Open the write mode used by `_HashingSink._spill`."""
        if mode != "wb":
            raise NotImplementedError(f"FakeGCSBlob.open only supports mode='wb', got {mode!r}")
        writer = FakeGCSBlobWriter(self, if_generation_match=if_generation_match)
        self._bucket.last_writer = writer
        return writer


class FakeGCSBlobWriter:
    """Buffer writes and commit on clean close, enforcing `if_generation_match=0`.

    An exception terminates the upload without committing, matching GCS
    resumable-upload semantics.
    """

    def __init__(self, blob: FakeGCSBlob, *, if_generation_match: int | None) -> None:
        self._blob = blob
        self._if_generation_match = if_generation_match
        self._buffer = bytearray()
        self.terminated = False
        self._closed = False

    def write(self, data: bytes) -> int:
        if self._closed:
            raise ValueError("write to a closed FakeGCSBlobWriter")
        self._buffer += data
        return len(data)

    def tell(self) -> int:
        return len(self._buffer)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        bucket = self._blob._bucket
        if bucket.raise_on_next_upload is not None:
            exc = bucket.raise_on_next_upload
            bucket.raise_on_next_upload = None
            raise exc
        from google.api_core.exceptions import PreconditionFailed

        exists = self._blob.name in bucket.objects
        if self._if_generation_match == 0 and exists:
            raise PreconditionFailed(f"object {self._blob.name!r} already exists")
        bucket.objects[self._blob.name] = bytes(self._buffer)
        bucket.time_created[self._blob.name] = datetime.now(UTC)
        bucket.upload_log.append(self._blob.name)
        bucket.cache_control[self._blob.name] = self._blob.cache_control

    def terminate(self) -> None:
        self.terminated = True
        self._closed = True

    def __enter__(self) -> FakeGCSBlobWriter:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if exc_type is not None:
            self.terminate()
        else:
            self.close()

    @property
    def closed(self) -> bool:
        return self._closed


class DiscardingFakeGCSBlobWriter(FakeGCSBlobWriter):
    """Discard each full chunk while recording call sizes and peak residency.

    `landed` preserves the final bytes for assertions; it is test bookkeeping,
    not part of the residency measurement.
    """

    def __init__(self, blob: FakeGCSBlob, *, if_generation_match: int | None) -> None:
        super().__init__(blob, if_generation_match=if_generation_match)
        self.chunk_size = SPILL_THRESHOLD_BYTES
        self.call_sizes: list[int] = []
        self.peak_resident_bytes = 0
        self.landed = bytearray()

    def write(self, data: bytes) -> int:
        if self._closed:
            raise ValueError("write to a closed DiscardingFakeGCSBlobWriter")
        n = len(data)
        self.call_sizes.append(n)
        self._buffer += data
        self.peak_resident_bytes = max(self.peak_resident_bytes, len(self._buffer))
        while len(self._buffer) >= self.chunk_size:
            self.landed += self._buffer[: self.chunk_size]
            del self._buffer[: self.chunk_size]
        return n

    def close(self) -> None:
        if self._closed:
            return
        self.landed += self._buffer
        self._buffer.clear()
        self._closed = True
        bucket = self._blob._bucket
        if bucket.raise_on_next_upload is not None:
            exc = bucket.raise_on_next_upload
            bucket.raise_on_next_upload = None
            raise exc
        from google.api_core.exceptions import PreconditionFailed

        exists = self._blob.name in bucket.objects
        if self._if_generation_match == 0 and exists:
            raise PreconditionFailed(f"object {self._blob.name!r} already exists")
        bucket.objects[self._blob.name] = bytes(self.landed)
        bucket.time_created[self._blob.name] = datetime.now(UTC)
        bucket.upload_log.append(self._blob.name)
        bucket.cache_control[self._blob.name] = self._blob.cache_control

    def tell(self) -> int:
        return len(self.landed) + len(self._buffer)


class FakeGCSBucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self.objects: dict[str, bytes] = {}
        self.upload_log: list[str] = []
        # Set when an object lands; a test planting an object sets its own.
        self.time_created: dict[str, datetime] = {}
        # Populated only when an upload lands.
        self.cache_control: dict[str, str | None] = {}
        # Fail the next upload in this bucket when set.
        self.raise_on_next_upload: Exception | None = None
        # Fail the next copy in this bucket when set.
        self.raise_on_next_copy: Exception | None = None
        # Most recent staging writer opened in this bucket.
        self.last_writer: FakeGCSBlobWriter | None = None
        # GCS answers 404 to a copy into a bucket that does not exist.
        self.missing = False

    def blob(self, name: str) -> FakeGCSBlob:
        return FakeGCSBlob(self, name)

    def get_blob(self, name: str) -> FakeGCSBlob | None:
        return FakeGCSBlob(self, name) if name in self.objects else None

    def copy_blob(
        self,
        blob: FakeGCSBlob,
        destination_bucket: FakeGCSBucket,
        new_name: str | None = None,
        if_generation_match: int | None = None,
    ) -> FakeGCSBlob:
        from google.api_core.exceptions import NotFound, PreconditionFailed

        if self.raise_on_next_copy is not None:
            exc = self.raise_on_next_copy
            self.raise_on_next_copy = None
            raise exc
        if blob.name not in self.objects:
            raise NotFound(f"source object {blob.name!r} not found in bucket {self.name!r}")
        if destination_bucket.missing:
            raise NotFound(f"bucket {destination_bucket.name!r} not found")
        dest_name = new_name if new_name is not None else blob.name
        exists = dest_name in destination_bucket.objects
        if if_generation_match == 0 and exists:
            raise PreconditionFailed(
                f"object {dest_name!r} already exists in bucket {destination_bucket.name!r}"
            )
        destination_bucket.objects[dest_name] = self.objects[blob.name]
        destination_bucket.time_created[dest_name] = datetime.now(UTC)
        destination_bucket.upload_log.append(dest_name)
        return FakeGCSBlob(destination_bucket, dest_name)


class FakeGCSClient:
    """In-memory GCS client for the methods called by `LiveArtifactStore`."""

    def __init__(self) -> None:
        self.buckets: dict[str, FakeGCSBucket] = {}

    def bucket(self, name: str) -> FakeGCSBucket:
        return self.buckets.setdefault(name, FakeGCSBucket(name))

    def list_blobs(
        self, bucket_or_name: FakeGCSBucket | str, *, match_glob: str | None = None
    ) -> list[FakeGCSBlob]:
        name = bucket_or_name if isinstance(bucket_or_name, str) else bucket_or_name.name
        bucket = self.buckets.setdefault(name, FakeGCSBucket(name))
        matcher = _glob_matcher(match_glob) if match_glob is not None else None
        return [
            FakeGCSBlob(bucket, object_name)
            for object_name in bucket.objects
            if matcher is None or matcher(object_name)
        ]


def _glob_matcher(pattern: str) -> Callable[[str], bool]:
    """Match GCS glob patterns used here: `*` and `?` exclude `/`, while `**` spans it.

    Other characters match literally. See the GCS list-object glob syntax.
    """
    segments: list[str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if pattern[i : i + 2] == "**":
            segments.append(".*")
            i += 2
        elif char == "*":
            segments.append("[^/]*")
            i += 1
        elif char == "?":
            segments.append("[^/]")
            i += 1
        else:
            segments.append(re.escape(char))
            i += 1
    compiled = re.compile("^" + "".join(segments) + "$")
    return lambda name: compiled.match(name) is not None


# Attach an in-memory database as `datasets` so inventory SQL runs unchanged.
# Translate unquoted `%s` placeholders to SQLite's `?` paramstyle.
_QUOTED_LITERAL_OR_PERCENT = re.compile(r"('(?:[^']|'')*')|%(.)")


def _to_sqlite_paramstyle(sql: str) -> str:
    def _sub(match: re.Match[str]) -> str:
        quoted = match.group(1)
        if quoted is not None:
            return quoted
        following = match.group(2)
        if following == "s":
            return "?"
        if following == "%":
            return "%"
        raise ValueError("Only %s and %% are supported in the query.")

    return _QUOTED_LITERAL_OR_PERCENT.sub(_sub, sql)


_ARTIFACT_INVENTORY_SCHEMA = """
CREATE TABLE datasets.artifact_inventory (
    artifact_digest TEXT PRIMARY KEY,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE datasets.artifact_location (
    artifact_digest TEXT NOT NULL,
    tier TEXT NOT NULL,
    bucket TEXT NOT NULL,
    object_name TEXT NOT NULL,
    stored_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    renewal INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (artifact_digest, tier)
);
"""


class _FakeTransaction:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        params = tuple(p.value if isinstance(p, Tier) else p for p in params)
        self._conn.execute(_to_sqlite_paramstyle(sql), params)

    def fetch_one(self, sql: str, params: Sequence[object] = ()) -> Mapping[str, object] | None:
        rows = self.fetch_all(sql, params)
        if not rows:
            return None
        if len(rows) > 1:
            raise ValueError(f"fetch_one got {len(rows)} rows for {sql!r}")
        return rows[0]

    def fetch_all(self, sql: str, params: Sequence[object] = ()) -> list[Mapping[str, object]]:
        params = tuple(p.value if isinstance(p, Tier) else p for p in params)
        cursor = self._conn.execute(_to_sqlite_paramstyle(sql), params)
        return [_decode_timestamps(dict(row)) for row in cursor.fetchall()]


def _decode_timestamps(row: dict[str, object]) -> dict[str, object]:
    """SQLite stores `timestamptz` columns as text; decode them as Postgres does."""
    for key in ("stored_at", "created_at"):
        if isinstance(row.get(key), str):
            row[key] = datetime.fromisoformat(cast(str, row[key]))
    return row


class FakeArtifactInventoryDatasetStore:
    """SQLite `DatasetStore` double for `LiveArtifactInventory` SQL.

    One connection shares writes across transactions; create its schema up
    front because SQLite DDL survives rollback.
    """

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("ATTACH DATABASE ':memory:' AS datasets")
        self._conn.executescript(_ARTIFACT_INVENTORY_SCHEMA)

    @contextmanager
    def transaction(self):
        try:
            yield _FakeTransaction(self._conn)
        except BaseException:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
