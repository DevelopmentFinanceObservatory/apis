"""Content-addressed GCS storage for artifacts.

The digest is the identity, and the inventory location names the object. An
object's name is its digest until the working tier renews it as
`<digest>#r<renewal>`. Writes use `if_generation_match=0`, so a 412 means the
name already holds the same bytes. A write tries the bare digest first and
reads the inventory location only when that name is taken, so storing new
bytes reads no inventory. `get()` rehashes bytes before returning them. Async
callers use `asyncio.to_thread()`.

A store records each write's inventory row in its own transaction. A view from
`deferred(batch)` adds the row to an `InventoryBatch` instead, for the caller to
write in the transaction that links the artifact. A signing that misses the URL
cache adds one to the call's `sign_blob_calls` count (`dfo_platform.timings`).
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import sys
import threading
import time
import uuid
from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage
from google.cloud.storage import fileio
from google.cloud.storage.retry import DEFAULT_RETRY

from artifact_store.errors import (
    ArtifactIntegrityError,
    ArtifactLocationDriftError,
    ArtifactNotFoundError,
    ArtifactWriteError,
)
from artifact_store.inventory import ArtifactLocation, InventoryBatch, InventoryEntry
from artifact_store.signing import SIGNED_URL_REUSE_WINDOW, emulator_host, sign_get_url
from dataset_store.clock import Clock, utc_now
from dataset_store.snapshot import working_tier_expired
from dfo_platform.timings import count

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

    from artifact_store.inventory import ArtifactInventory

_CACHE_CONTROL = "private, max-age=31536000, immutable"
"""The one cache-control value this store ever sets, on any object it
uploads under its own name -- content-addressed, so the bytes behind a name
never change, renewal names included. Set on a staging blob too, before its `open()` call, since a
same-bucket `copy_blob` carries metadata across without re-setting it."""

SPILL_THRESHOLD_BYTES = 16 * 1024 * 1024
"""`ArtifactSink`'s in-memory buffer limit, and the chunk size its staged
resumable upload uses once crossed (a multiple of GCS's required 256 KiB,
per `google.cloud.storage.fileio.BlobWriter`'s own `_chunk_size` check).
Almost every write stays under this -- the synchronous mint budget is
100,000 rows -- so most `open_put` calls stay on the single-request
`upload_from_string` path and never open a staging upload at all."""

_RENEWAL_ATTEMPTS = 3
"""Successive renewal names one write tries before `ArtifactWriteError`. A
taken renewal name holds a concurrent renewer's bytes or those of a renewal
whose location move failed. The write adopts it when its creation time is
within working-tier retention and moves on to the next name when it is not,
so only renewal names past retention use up attempts."""

_STAGING_PREFIX = "staging/"
"""Every staged object's name starts with this. `LiveArtifactStore.list`
excludes it (`match_glob="sha256:*"` never crosses the `/`), so a
reconciliation sweep over `list()` never sees an in-flight or
soon-to-expire staging object as an orphan. The working bucket's own
1-day age-based lifecycle rule on this prefix is what actually deletes a
staged object afterward -- neither this module nor the runtime identity it
executes under holds a delete grant on either bucket."""


class Tier(StrEnum):
    """The two buckets this deployment writes artifacts to.
    `dev-fin-observatory-datasets-working` (`working`) carries a 30-day
    age-based delete rule; `dev-fin-observatory-datasets-pinned` (`pinned`)
    carries a 90-day soft delete and no expiry. Both have uniform bucket-level
    access -- no per-object ACL is ever set here.
    """

    working = "working"
    pinned = "pinned"


@dataclass(frozen=True)
class StoredArtifact:
    """Result of `put()`. `digest` is the SHA-256 of the bytes.

    `deduplicated` marks bytes another write already stored. `bucket`,
    `object_name` and `size_bytes` describe the stored object, whichever
    writer landed it.
    """

    digest: str
    tier: Tier
    bucket: str
    object_name: str
    size_bytes: int
    deduplicated: bool


class ArtifactSink(Protocol):
    """Write target yielded by `open_put()`.

    `write()` accepts chunks. `tell()` counts accepted bytes, regardless of upload progress.
    `flush()` does nothing. `closed` becomes true at context exit.
    TextIOWrapper, ParquetWriter, and ZipFile query `readable()`, `writable()`,
    and `seekable()` on wrapped buffers. Call `result()` after successful exit.
    Earlier calls raise `RuntimeError`.
    """

    def write(self, data: bytes, /) -> int: ...

    def tell(self) -> int: ...

    def flush(self) -> None: ...

    def readable(self) -> bool: ...

    def writable(self) -> bool: ...

    def seekable(self) -> bool: ...

    @property
    def closed(self) -> bool: ...

    def result(self) -> StoredArtifact: ...


class ExistsRead(StrEnum):
    """How patiently `ArtifactStore.exists` reads: `short` for a caller holding a work permit,
    `long` for the sweep, which holds none."""

    short = "short"
    long = "long"


_EXISTS_READS: Mapping[ExistsRead, tuple[float, float]] = {
    # 5 s attempts retried within 5 s, so one 503 does not leave a copy unsettled and a bucket
    # that is down answers in about 10 s.
    ExistsRead.short: (5.0, 5.0),
    # The library's own: 60 s attempts retried within 120 s.
    ExistsRead.long: (60.0, 120.0),
}
"""Each read's attempt timeout and retry deadline, in seconds."""


@dataclass(frozen=True)
class CopyBounds:
    """`copy_blob`'s `timeout=(connect, read)` and retry deadline, in seconds. The read timeout
    bounds each gap between bytes on the socket, not the whole request. Both tiers' buckets sit in
    one region in the default storage class, so a copy between them rewrites metadata only and
    answers in one short response whatever its size, `materialise.formats.PARQUET_MAX_BYTES`
    (256 MiB) included."""

    connect: float = 10.0
    read: float = 30.0
    retry_deadline: float = 30.0

    @property
    def lands_within(self) -> timedelta:
        """How long `copy_blob` takes to land its object in the usual case: its retries, then a
        last attempt whose connection and reply each come within their timeouts. A server that
        trickles its reply can run past it, and a copy that lands later costs one free pinned
        object for its digest. What follows the landing, a size read and the inventory write,
        changes nothing an existence read says."""
        return timedelta(seconds=self.retry_deadline + self.connect + self.read)


DEFAULT_COPY_BOUNDS = CopyBounds()


class ArtifactStore(Protocol):
    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        """Write under the location's object name, or the digest when the tier has none.

        An identical write returns `deduplicated=True`. A working location past
        retention renews: the bytes upload again under `<digest>#r<renewal>`
        and the location moves to that name with a fresh `stored_at`, or, when
        the name is taken by bytes within retention, adopts it at their creation
        time and returns `deduplicated=True`.
        Raises `ArtifactWriteError` on write failure, `ArtifactInventoryError`
        if the inventory read or record fails, or `ArtifactRecordConflictError`
        if the stored bucket, object name, or size differs.
        """
        ...

    def open_put(self, *, tier: Tier) -> AbstractContextManager[ArtifactSink]:
        """Stream into an `ArtifactSink`. The digest is known at context exit.

        Writes below `SPILL_THRESHOLD_BYTES` use one request. Larger writes
        use a staging upload, then a server-side copy to the object name.
        Deduplication, renewal and errors match `put()`. Call `result()` after a
        successful exit. A failing block cancels any active staging session and propagates its exception.
        """
        ...

    def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
        """Fetch the object the inventory location names and verify the bytes hash to the digest.

        Raises `ArtifactNotFoundError` if the location or object is missing,
        `ArtifactLocationDriftError` if the location's bucket is not the
        tier's, or `ArtifactIntegrityError` if the bytes hash to another digest.
        """
        ...

    def location(self, digest: str, *, tier: Tier) -> ArtifactLocation | None:
        """The location the inventory records for `digest` at `tier`, or `None` when it records
        none. Tier retention can delete an object while its location row persists.

        Raises `ArtifactInventoryError` if the read fails.
        """
        ...

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
        """Sign a GET URL for a location the caller already read, with the response headers.

        Raises `ArtifactLocationDriftError` when `bucket` is not the tier's,
        `ArtifactSigningError` on signing failure, or `ValueError` outside
        the signing TTL bounds. Tier retention can expire an object while its
        location row persists.
        """
        ...

    @property
    def copy_lands_within(self) -> timedelta:
        """How long `copy` takes to land its object in the usual case (`CopyBounds`)."""
        ...

    def exists(self, digest: str, *, tier: Tier, read: ExistsRead = ExistsRead.short) -> bool:
        """Whether the tier's bucket holds the object of `digest`, whatever the inventory says,
        read as `read` says."""
        ...

    def copy(
        self,
        digest: str,
        *,
        from_tier: Tier,
        to_tier: Tier,
        source: tuple[str, str] | None = None,
    ) -> StoredArtifact:
        """Copy to the destination bucket. An existing digest is a deduplication. `source` is
        the bucket and object name of the source a caller holds, else the copy reads them from
        the inventory.

        Raises `ArtifactNotFoundError` if the source location or its object is absent, or
        `ArtifactWriteError` for another copy failure.
        """
        ...

    def list(self, *, tier: Tier) -> frozenset[str]:
        """Return the bucket's object names, digests and renewal names, excluding staging objects.

        This is a raw GCS bucket listing, independent of inventory rows.
        """
        ...

    def deferred(self, batch: InventoryBatch) -> ArtifactStore:
        """A view of this store whose `put`, `open_put` and `copy` add the inventory row each
        write owes to `batch` in place of recording it. Reads, signing and the URL cache are
        this store's. The caller writes `batch.entries` in its own transaction
        (`inventory.write_entry`); until then the bytes are stored but not inventoried.
        """
        ...


class _HashingSink(io.RawIOBase):
    """`open_put()` sink that hashes writes and spills at `SPILL_THRESHOLD_BYTES`.

    Smaller writes upload once at commit. Larger writes use a resumable
    upload under `_STAGING_PREFIX` in the working bucket, then copy to the
    target tier's object name. Staging stays in working storage to avoid
    the pinned tier's 90-day soft delete. The working bucket's 1-day lifecycle
    rule removes staged objects. Runtime grants omit delete access.
    `_land()` chooses the object name and decides between deduplication and
    renewal, and owes each landing's inventory row to `inventory` or, when set, to
    `batch`. `_abort()` cancels an open upload when the caller's block raises.
    """

    def __init__(
        self,
        *,
        client: storage.Client,
        staging_bucket_name: str,
        target_bucket_name: str,
        tier: Tier,
        inventory: ArtifactInventory,
        batch: InventoryBatch | None,
        now: Clock,
    ) -> None:
        super().__init__()
        self._client = client
        self._staging_bucket_name = staging_bucket_name
        self._target_bucket_name = target_bucket_name
        self._tier = tier
        self._inventory = inventory
        self._batch = batch
        self._now = now
        self._hasher = hashlib.sha256()
        self._buffer = bytearray()
        self._bytes_written = 0
        self._staging_object_name: str | None = None
        self._staging_writer: fileio.BlobWriter | None = None
        self._result: StoredArtifact | None = None

    def writable(self) -> bool:
        return True

    def write(self, data: ReadableBuffer, /) -> int:
        self._checkClosed()
        # A memoryview supports hashing and upload, and supplies len() absent from ReadableBuffer.
        view = memoryview(data)
        self._hasher.update(view)
        length = len(view)
        self._bytes_written += length
        if self._staging_writer is None:
            self._buffer += view
            if len(self._buffer) >= SPILL_THRESHOLD_BYTES:
                self._spill()
        else:
            self._staging_writer.write(view)
        return length

    def tell(self) -> int:
        return self._bytes_written

    def flush(self) -> None:
        # BlobWriter.flush() raises without ignore_flush=True.
        pass

    def result(self) -> StoredArtifact:
        if self._result is None:
            raise RuntimeError(
                "ArtifactSink.result() called before its owning open_put() "
                "block committed -- call it only after that `with` block "
                "has exited without raising"
            )
        return self._result

    def _spill(self) -> None:
        self._staging_object_name = _STAGING_PREFIX + str(uuid.uuid4())
        blob = self._client.bucket(self._staging_bucket_name).blob(self._staging_object_name)
        # Set before open(). The copy carries staging metadata to the final object.
        blob.cache_control = _CACHE_CONTROL
        self._staging_writer = blob.open(
            "wb",
            chunk_size=SPILL_THRESHOLD_BYTES,
            ignore_flush=True,
            content_type="application/octet-stream",
            if_generation_match=0,
        )
        buffered = bytes(self._buffer)
        self._buffer.clear()
        if buffered:
            self._staging_writer.write(buffered)

    def _abort(self) -> None:
        if self._staging_writer is not None:
            # Cleanup failures leave the caller's exception as the raised error.
            with contextlib.suppress(Exception):
                self._staging_writer.__exit__(*sys.exc_info())
        self.close()

    def _commit(self) -> StoredArtifact:
        try:
            digest = "sha256:" + self._hasher.hexdigest()
            result = (
                self._commit_staged(digest)
                if self._staging_writer is not None
                else self._commit_buffered(digest)
            )
            self._result = result
            return result
        finally:
            self.close()

    def _commit_buffered(self, digest: str) -> StoredArtifact:
        data = bytes(self._buffer)
        target_bucket = self._client.bucket(self._target_bucket_name)

        def upload(object_name: str) -> None:
            blob = target_bucket.blob(object_name)
            blob.cache_control = _CACHE_CONTROL
            blob.upload_from_string(
                data, content_type="application/octet-stream", if_generation_match=0
            )

        return self._land(digest, upload)

    def _commit_staged(self, digest: str) -> StoredArtifact:
        assert self._staging_writer is not None
        assert self._staging_object_name is not None
        close_failed = False
        try:
            self._staging_writer.close()
        except Exception:
            close_failed = True
        if close_failed:
            raise ArtifactWriteError(object_name=digest, digest=digest)

        staging_bucket = self._client.bucket(self._staging_bucket_name)
        staging_blob = staging_bucket.blob(self._staging_object_name)
        target_bucket = self._client.bucket(self._target_bucket_name)

        def copy(object_name: str) -> None:
            staging_bucket.copy_blob(
                staging_blob, target_bucket, new_name=object_name, if_generation_match=0
            )

        return self._land(digest, copy)

    def _land(self, digest: str, write: Callable[[str], None]) -> StoredArtifact:
        """Write under the bare digest, or, when that name is taken, under the location's object
        name, and record the landing.

        Only a taken bare name reads the inventory location, so new bytes land with no read.
        A precondition failure under the location's name deduplicates, unless the bytes are
        past working-tier retention: the location's `stored_at`, or, with no working location
        row, the existing object's creation time, since such an orphan of a failed inventory
        write ages under the lifecycle rule from its creation. A deduplication records that time
        as `stored_at`. A pinned deduplication without a row reads no object metadata, since
        pinned bytes never expire. Bytes past retention upload again under the next renewal
        name, and the location moves to it only after that upload returns. A renewal name that
        is already taken is adopted by the same creation-time rule: within retention, the
        location moves to it with that time as `stored_at` and the write deduplicates; past
        retention, the write tries the next name. Only this store writes `<digest>#r<n>`, after
        hashing the bytes, so a taken renewal name holds this digest's bytes, and `get()`
        rehashes them on read. A deduplication against expired bytes is never recorded, so it
        never becomes servable.
        """
        if self._write(write, digest, digest=digest):
            return self._record(digest, digest, uploaded=True)
        location = self._inventory.get_location(digest, tier=self._tier)
        object_name = digest if location is None else location.object_name
        if object_name != digest and self._write(write, object_name, digest=digest):
            return self._record(digest, object_name, uploaded=True)
        if location is None and self._tier != Tier.working:
            return self._record(digest, object_name, uploaded=False)
        if location is None:
            stored_at, renewed = self._created_at(object_name, digest=digest), 0
        else:
            stored_at, renewed = location.stored_at, location.renewal
        if not self._past_retention(stored_at):
            return self._record(digest, object_name, uploaded=False, stored_at=stored_at)
        for renewal in range(renewed + 1, renewed + 1 + _RENEWAL_ATTEMPTS):
            object_name = f"{digest}#r{renewal}"
            if self._write(write, object_name, digest=digest):
                return self._renew(digest, object_name, renewal=renewal, stored_at=None)
            stored_at = self._created_at(object_name, digest=digest)
            if not self._past_retention(stored_at):
                return self._renew(digest, object_name, renewal=renewal, stored_at=stored_at)
        raise ArtifactWriteError(object_name=object_name, digest=digest)

    def _created_at(self, object_name: str, *, digest: str) -> datetime:
        """The creation time of the object a precondition failure found, by a metadata read.

        An object deleted since that failure raises `ArtifactWriteError`, as does a failed read.
        """
        created: datetime | None = None
        try:
            blob = self._client.bucket(self._target_bucket_name).get_blob(object_name)
            created = None if blob is None else blob.time_created
        except Exception:
            created = None
        if created is None:
            raise ArtifactWriteError(object_name=object_name, digest=digest)
        return created

    def _past_retention(self, stored_at: datetime) -> bool:
        return self._tier == Tier.working and working_tier_expired(stored_at, now=self._now())

    @staticmethod
    def _write(write: Callable[[str], None], object_name: str, *, digest: str) -> bool:
        """Whether `write` landed the bytes. False when the name already holds an object."""
        landed = False
        write_failed = False
        try:
            write(object_name)
            landed = True
        except PreconditionFailed:
            pass
        except Exception:
            write_failed = True
        if write_failed:
            raise ArtifactWriteError(object_name=object_name, digest=digest)
        return landed

    def _record(
        self,
        digest: str,
        object_name: str,
        *,
        uploaded: bool,
        stored_at: datetime | None = None,
    ) -> StoredArtifact:
        self._owe(
            InventoryEntry(
                digest=digest,
                tier=self._tier,
                bucket=self._target_bucket_name,
                object_name=object_name,
                size_bytes=self._bytes_written,
                uploaded=uploaded,
                stored_at=stored_at,
            )
        )
        return self._stored(object_name, digest=digest, deduplicated=not uploaded)

    def _renew(
        self, digest: str, object_name: str, *, renewal: int, stored_at: datetime | None
    ) -> StoredArtifact:
        """Owe the location's move to a renewal name this write uploaded, or adopted at
        `stored_at`."""
        self._owe(
            InventoryEntry(
                digest=digest,
                tier=self._tier,
                bucket=self._target_bucket_name,
                object_name=object_name,
                size_bytes=self._bytes_written,
                uploaded=stored_at is None,
                stored_at=stored_at,
                renewal=renewal,
            )
        )
        return self._stored(object_name, digest=digest, deduplicated=stored_at is not None)

    def _owe(self, entry: InventoryEntry) -> None:
        """Record `entry` now, or add it to the batch the caller writes."""
        if self._batch is not None:
            self._batch.add(entry)
        elif entry.renewal is not None:
            self._inventory.renew(
                digest=entry.digest,
                tier=entry.tier,
                bucket=entry.bucket,
                object_name=entry.object_name,
                size_bytes=entry.size_bytes,
                renewal=entry.renewal,
                stored_at=entry.stored_at,
            )
        else:
            self._inventory.record(
                digest=entry.digest,
                tier=entry.tier,
                bucket=entry.bucket,
                object_name=entry.object_name,
                size_bytes=entry.size_bytes,
                uploaded=entry.uploaded,
                stored_at=entry.stored_at,
            )

    def _stored(self, object_name: str, *, digest: str, deduplicated: bool) -> StoredArtifact:
        return StoredArtifact(
            digest=digest,
            tier=self._tier,
            bucket=self._target_bucket_name,
            object_name=object_name,
            size_bytes=self._bytes_written,
            deduplicated=deduplicated,
        )


class LiveArtifactStore:
    """GCS-backed `ArtifactStore`. `buckets` must map both tiers.

    `inventory.record()` runs after a fresh upload or confirmed duplicate, and
    `inventory.renew()` after a renewal upload, unless the store is a `deferred`
    view, which adds those rows to its batch. `now` is the wall clock the
    working tier's retention is judged against.
    Signed URLs are cached per instance for `SIGNED_URL_REUSE_WINDOW`, keyed
    by tier, digest, bucket, object name, TTL, response type, and disposition,
    so subjects sharing bytes cannot share the wrong headers and a renewed
    location never reuses a URL signed for its old object name. Cache age
    starts before signing. Cached URLs keep their original expiry.
    """

    def __init__(
        self,
        *,
        buckets: Mapping[Tier, str],
        inventory: ArtifactInventory,
        client: storage.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
        now: Clock = utc_now,
        copy_bounds: CopyBounds = DEFAULT_COPY_BOUNDS,
    ) -> None:
        missing_tiers = [tier for tier in Tier if tier not in buckets]
        if missing_tiers:
            raise ValueError(
                f"buckets is missing an entry for tier(s): "
                f"{[tier.value for tier in missing_tiers]!r}"
            )
        # Refuse an emulator outside the local stack at start-up, before any link goes out.
        emulator_host()
        self._buckets = dict(buckets)
        self._inventory = inventory
        self._client = client if client is not None else storage.Client()
        self._clock = clock
        self._now = now
        self._copy_bounds = copy_bounds
        self._copy_retry = DEFAULT_RETRY.with_timeout(copy_bounds.retry_deadline)
        self._copy_timeout = (copy_bounds.connect, copy_bounds.read)
        self._url_cache: dict[_UrlCacheKey, _CachedUrl] = {}
        self._url_cache_lock = threading.Lock()
        self._batch: InventoryBatch | None = None

    def deferred(self, batch: InventoryBatch) -> LiveArtifactStore:
        """A view sharing this store's client, buckets and URL cache whose writes add their
        inventory rows to `batch`."""
        view = copy.copy(self)
        view._batch = batch
        return view

    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        """Write under the content digest. Data below `SPILL_THRESHOLD_BYTES` uses one request.

        Inherits `open_put()`'s deduplication and error behaviour.
        """
        with self.open_put(tier=tier) as sink:
            sink.write(data)
        return sink.result()

    @contextmanager
    def open_put(self, *, tier: Tier) -> Generator[ArtifactSink, None, None]:
        """Yield a hashing sink. Commit on clean exit and abort if the block raises."""
        sink = _HashingSink(
            client=self._client,
            staging_bucket_name=self._buckets[Tier.working],
            target_bucket_name=self._buckets[tier],
            tier=tier,
            inventory=self._inventory,
            batch=self._batch,
            now=self._now,
        )
        try:
            yield sink
        except BaseException:
            sink._abort()
            raise
        else:
            sink._commit()

    def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
        """Read the object the location names and verify the returned bytes' hash."""
        location = self._inventory.get_location(artifact_digest, tier=tier)
        if location is None:
            raise ArtifactNotFoundError(digest=artifact_digest, tier=tier)
        self._check_bucket(
            location.bucket, object_name=location.object_name, digest=artifact_digest, tier=tier
        )
        blob = self._client.bucket(location.bucket).blob(location.object_name)
        try:
            data = blob.download_as_bytes()
        except NotFound:
            data = None
        if data is None:
            raise ArtifactNotFoundError(digest=artifact_digest, tier=tier)
        computed_digest = _digest_of(data)
        if computed_digest != artifact_digest:
            raise ArtifactIntegrityError(
                digest=artifact_digest, tier=tier, computed_digest=computed_digest
            )
        return data

    def location(self, digest: str, *, tier: Tier) -> ArtifactLocation | None:
        return self._inventory.get_location(digest, tier=tier)

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
        """Return a valid cache hit for this location and these headers, or sign them."""
        cache_key = (
            tier,
            digest,
            bucket,
            object_name,
            ttl.total_seconds(),
            response_type,
            response_disposition,
        )
        cached = self._cached_url(cache_key)
        if cached is not None:
            return cached.url
        # Start cache age before signing. V4 expiry starts at sign time, so the stored age is
        # conservative.
        signed_at = self._clock()
        self._check_bucket(bucket, object_name=object_name, digest=digest, tier=tier)
        count("sign_blob_calls")
        url = sign_get_url(
            client=self._client,
            bucket=bucket,
            object_name=object_name,
            digest=digest,
            ttl=ttl,
            response_type=response_type,
            response_disposition=response_disposition,
        )
        with self._url_cache_lock:
            self._url_cache[cache_key] = _CachedUrl(signed_at=signed_at, url=url)
            stale_before = signed_at - SIGNED_URL_REUSE_WINDOW.total_seconds()
            for key, entry in list(self._url_cache.items()):
                if entry.signed_at <= stale_before:
                    del self._url_cache[key]
        return url

    def _cached_url(self, cache_key: _UrlCacheKey) -> _CachedUrl | None:
        with self._url_cache_lock:
            cached = self._url_cache.get(cache_key)
        if cached is None:
            return None
        if self._clock() - cached.signed_at >= SIGNED_URL_REUSE_WINDOW.total_seconds():
            return None
        return cached

    def _check_bucket(self, bucket: str, *, object_name: str, digest: str, tier: Tier) -> None:
        """Refuse a location outside the tier's configured bucket, which signals drift or
        corruption."""
        if bucket != self._buckets[tier]:
            raise ArtifactLocationDriftError(
                digest=digest, tier=tier, stored_bucket=bucket, stored_object_name=object_name
            )

    @property
    def copy_lands_within(self) -> timedelta:
        return self._copy_bounds.lands_within

    def exists(self, digest: str, *, tier: Tier, read: ExistsRead = ExistsRead.short) -> bool:
        timeout, deadline = _EXISTS_READS[read]
        bucket = self._client.bucket(self._buckets[tier])
        blob = bucket.get_blob(digest, timeout=timeout, retry=DEFAULT_RETRY.with_timeout(deadline))
        return blob is not None

    def copy(
        self,
        digest: str,
        *,
        from_tier: Tier,
        to_tier: Tier,
        source: tuple[str, str] | None = None,
    ) -> StoredArtifact:
        """Copy from the recorded source location. Record the destination after success. A
        source object the lifecycle rule deleted raises `ArtifactNotFoundError`. GCS also
        answers 404 for a missing destination bucket, so a `NotFound` raises
        `ArtifactNotFoundError` only when the source object is absent and
        `ArtifactWriteError` otherwise."""
        if source is None:
            location = self._inventory.get_location(digest, tier=from_tier)
            if location is None:
                raise ArtifactNotFoundError(digest=digest, tier=from_tier)
            source = (location.bucket, location.object_name)
        source_bucket_name, source_object_name = source
        source_bucket = self._client.bucket(source_bucket_name)
        source_blob = source_bucket.blob(source_object_name)
        dest_bucket_name = self._buckets[to_tier]
        dest_bucket = self._client.bucket(dest_bucket_name)

        deduplicated = False
        source_missing = False
        copy_failed = False
        size_bytes: int | None = None
        try:
            dest_blob = source_bucket.copy_blob(
                source_blob,
                dest_bucket,
                new_name=digest,
                if_generation_match=0,
                timeout=self._copy_timeout,
                retry=self._copy_retry,
            )
            size_bytes = dest_blob.size
        except PreconditionFailed:
            deduplicated = True
        except NotFound:
            try:
                source_missing = (
                    source_bucket.get_blob(
                        source_object_name,
                        timeout=self._copy_timeout,
                        retry=self._copy_retry,
                    )
                    is None
                )
            except Exception:
                source_missing = False
            copy_failed = not source_missing
        except Exception:
            copy_failed = True
        if source_missing:
            raise ArtifactNotFoundError(digest=digest, tier=from_tier)
        if copy_failed:
            raise ArtifactWriteError(object_name=digest, digest=digest)
        if size_bytes is None:
            existing = dest_bucket.get_blob(
                digest, timeout=self._copy_timeout, retry=self._copy_retry
            )
            if existing is None:
                raise ArtifactWriteError(object_name=digest, digest=digest)
            size_bytes = existing.size

        entry = InventoryEntry(
            digest=digest,
            tier=to_tier,
            bucket=dest_bucket_name,
            object_name=digest,
            size_bytes=size_bytes,
            uploaded=not deduplicated,
        )
        if self._batch is not None:
            self._batch.add(entry)
        else:
            self._inventory.record(
                digest=digest,
                tier=to_tier,
                bucket=dest_bucket_name,
                object_name=digest,
                size_bytes=size_bytes,
                uploaded=not deduplicated,
            )
        return StoredArtifact(
            digest=digest,
            tier=to_tier,
            bucket=dest_bucket_name,
            object_name=digest,
            size_bytes=size_bytes,
            deduplicated=deduplicated,
        )

    def list(self, *, tier: Tier) -> frozenset[str]:
        """List object names. GCS's `sha256:*` glob excludes staged objects containing `/`."""
        return frozenset(
            blob.name
            for blob in self._client.list_blobs(self._buckets[tier], match_glob="sha256:*")
        )


@dataclass(frozen=True)
class _CachedUrl:
    signed_at: float
    url: str


# Tier, digest, bucket, object name, TTL seconds, response type, response disposition.
type _UrlCacheKey = tuple[Tier, str, str, str, float, str, str]


def _digest_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()
