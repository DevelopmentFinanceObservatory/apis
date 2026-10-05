"""Content-addressed GCS storage for artifacts.

Objects use their digest as the name. `put()` uses `if_generation_match=0`,
so a concurrent 412 means another writer stored the same object. `get()`
rehashes bytes before returning them. Async callers use `asyncio.to_thread()`.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import sys
import threading
import time
import uuid
from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage
from google.cloud.storage import fileio

from artifact_store.errors import (
    ArtifactIntegrityError,
    ArtifactLocationDriftError,
    ArtifactNotFoundError,
    ArtifactWriteError,
)
from artifact_store.signing import SIGNED_URL_REUSE_WINDOW, sign_get_url

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

    from artifact_store.inventory import ArtifactInventory

_CACHE_CONTROL = "private, max-age=31536000, immutable"
"""The one cache-control value this store ever sets, on any object it
uploads under its own name -- content-addressed, so the bytes behind a name
never change. Set on a staging blob too, before its `open()` call, since a
same-bucket `copy_blob` carries metadata across without re-setting it."""

SPILL_THRESHOLD_BYTES = 16 * 1024 * 1024
"""`ArtifactSink`'s in-memory buffer limit, and the chunk size its staged
resumable upload uses once crossed (a multiple of GCS's required 256 KiB,
per `google.cloud.storage.fileio.BlobWriter`'s own `_chunk_size` check).
Almost every write stays under this -- the synchronous mint budget is
100,000 rows -- so most `open_put` calls stay on the single-request
`upload_from_string` path and never open a staging upload at all."""

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
    """Result of `put()`. `digest` is the SHA-256 name.

    `deduplicated` marks a concurrent write. `bucket` and `size_bytes`
    describe the stored object, whichever writer landed it.
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


class ArtifactStore(Protocol):
    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        """Write under the content digest. An identical write returns `deduplicated=True`.

        Raises `ArtifactWriteError` on write failure, `ArtifactInventoryError`
        if recording fails after upload, or `ArtifactRecordConflictError` if
        the stored bucket, object name, or size differs.
        """
        ...

    def open_put(self, *, tier: Tier) -> AbstractContextManager[ArtifactSink]:
        """Stream into an `ArtifactSink`. The digest is known at context exit.

        Writes below `SPILL_THRESHOLD_BYTES` use one request. Larger writes
        use a staging upload, then a server-side copy to the digest name.
        Deduplication and errors match `put()`. Call `result()` after a
        successful exit. A failing block cancels any active staging session and propagates its exception.
        """
        ...

    def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
        """Fetch by digest from the configured tier bucket and verify the bytes' hash.

        Raises `ArtifactNotFoundError` if missing or `ArtifactIntegrityError`
        if the object name and content digest differ.
        """
        ...

    def signed_get_url(
        self,
        digest: str,
        *,
        tier: Tier,
        ttl: timedelta,
        response_type: str,
        response_disposition: str,
    ) -> str:
        """Sign a GET URL for the inventory-recorded location and response headers.

        Raises `ArtifactNotFoundError` when no location exists,
        `ArtifactLocationDriftError` for a bucket or object-name mismatch,
        `ArtifactSigningError` on signing failure, or `ValueError` outside
        the signing TTL bounds. Tier retention can expire an object while its
        location row persists.
        """
        ...

    def copy(self, digest: str, *, from_tier: Tier, to_tier: Tier) -> StoredArtifact:
        """Copy to the destination bucket. An existing digest is a deduplication.

        Raises `ArtifactNotFoundError` if the source location is absent or
        `ArtifactWriteError` for another copy failure.
        """
        ...

    def list(self, *, tier: Tier) -> frozenset[str]:
        """Return digest-named objects from the bucket, excluding staging objects.

        This is a raw GCS bucket listing, independent of inventory rows.
        """
        ...


class _HashingSink(io.RawIOBase):
    """`open_put()` sink that hashes writes and spills at `SPILL_THRESHOLD_BYTES`.

    Smaller writes upload once at commit. Larger writes use a resumable
    upload under `_STAGING_PREFIX` in the working bucket, then copy to the
    target tier under the digest. Staging stays in working storage to avoid
    the pinned tier's 90-day soft delete. The working bucket's 1-day lifecycle
    rule removes staged objects. Runtime grants omit delete access.
    A destination precondition failure counts as deduplication. `_abort()`
    cancels an open upload when the caller's block raises.
    """

    def __init__(
        self,
        *,
        client: storage.Client,
        staging_bucket_name: str,
        target_bucket_name: str,
        tier: Tier,
        inventory: ArtifactInventory,
    ) -> None:
        super().__init__()
        self._client = client
        self._staging_bucket_name = staging_bucket_name
        self._target_bucket_name = target_bucket_name
        self._tier = tier
        self._inventory = inventory
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
        blob = self._client.bucket(self._target_bucket_name).blob(digest)
        deduplicated = False
        write_failed = False
        try:
            blob.cache_control = _CACHE_CONTROL
            blob.upload_from_string(
                data, content_type="application/octet-stream", if_generation_match=0
            )
        except PreconditionFailed:
            deduplicated = True
        except Exception:
            write_failed = True
        if write_failed:
            raise ArtifactWriteError(object_name=digest, digest=digest)
        self._inventory.record(
            digest=digest,
            tier=self._tier,
            bucket=self._target_bucket_name,
            object_name=digest,
            size_bytes=len(data),
            uploaded=not deduplicated,
        )
        return StoredArtifact(
            digest=digest,
            tier=self._tier,
            bucket=self._target_bucket_name,
            object_name=digest,
            size_bytes=len(data),
            deduplicated=deduplicated,
        )

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
        deduplicated = False
        copy_failed = False
        size_bytes: int | None = None
        try:
            dest_blob = staging_bucket.copy_blob(
                staging_blob, target_bucket, new_name=digest, if_generation_match=0
            )
            size_bytes = dest_blob.size
        except PreconditionFailed:
            deduplicated = True
        except Exception:
            copy_failed = True
        if copy_failed:
            raise ArtifactWriteError(object_name=digest, digest=digest)
        if size_bytes is None:
            # The failed copy returned no size. Read it from the destination.
            existing = target_bucket.get_blob(digest)
            if existing is None:
                raise ArtifactWriteError(object_name=digest, digest=digest)
            size_bytes = existing.size

        self._inventory.record(
            digest=digest,
            tier=self._tier,
            bucket=self._target_bucket_name,
            object_name=digest,
            size_bytes=size_bytes,
            uploaded=not deduplicated,
        )
        return StoredArtifact(
            digest=digest,
            tier=self._tier,
            bucket=self._target_bucket_name,
            object_name=digest,
            size_bytes=size_bytes,
            deduplicated=deduplicated,
        )


class LiveArtifactStore:
    """GCS-backed `ArtifactStore`. `buckets` must map both tiers.

    `inventory.record()` runs after a fresh upload or confirmed duplicate.
    Signed URLs are cached per instance for `SIGNED_URL_REUSE_WINDOW`, keyed
    by tier, digest, TTL, response type, and disposition so subjects sharing
    bytes cannot share the wrong headers. A cache hit skips inventory lookup
    and signing. After a row changes or disappears, its URL can remain cached
    until expiry. Cache age starts before signing. Cached URLs keep their original expiry.
    """

    def __init__(
        self,
        *,
        buckets: Mapping[Tier, str],
        inventory: ArtifactInventory,
        client: storage.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        missing_tiers = [tier for tier in Tier if tier not in buckets]
        if missing_tiers:
            raise ValueError(
                f"buckets is missing an entry for tier(s): "
                f"{[tier.value for tier in missing_tiers]!r}"
            )
        self._buckets = dict(buckets)
        self._inventory = inventory
        self._client = client if client is not None else storage.Client()
        self._clock = clock
        self._url_cache: dict[tuple[Tier, str, float, str, str], tuple[float, str]] = {}
        self._url_cache_lock = threading.Lock()

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
        )
        try:
            yield sink
        except BaseException:
            sink._abort()
            raise
        else:
            sink._commit()

    def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
        """Read the configured tier bucket by digest and verify the returned bytes' hash."""
        blob = self._client.bucket(self._buckets[tier]).blob(artifact_digest)
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

    def signed_get_url(
        self,
        digest: str,
        *,
        tier: Tier,
        ttl: timedelta,
        response_type: str,
        response_disposition: str,
    ) -> str:
        """Return a valid cache hit or read, validate, and sign the recorded location.

        Misses sign without holding the cache lock, so concurrent misses may
        issue duplicate signing calls.
        """
        cache_key = (tier, digest, ttl.total_seconds(), response_type, response_disposition)
        with self._url_cache_lock:
            cached = self._url_cache.get(cache_key)
        if cached is not None:
            signed_at, url = cached
            if self._clock() - signed_at < SIGNED_URL_REUSE_WINDOW.total_seconds():
                return url

        # Start cache age before signing. V4 expiry starts at sign time.
        # The stored age is therefore conservative.
        signed_at = self._clock()

        location = self._inventory.get_location(digest, tier=tier)
        if location is None:
            raise ArtifactNotFoundError(digest=digest, tier=tier)
        # `put()` records this bucket and digest name. Mismatches indicate drift or corruption.
        if location.bucket != self._buckets[tier] or location.object_name != digest:
            raise ArtifactLocationDriftError(
                digest=digest,
                tier=tier,
                stored_bucket=location.bucket,
                stored_object_name=location.object_name,
            )
        url = sign_get_url(
            client=self._client,
            bucket=location.bucket,
            object_name=location.object_name,
            ttl=ttl,
            response_type=response_type,
            response_disposition=response_disposition,
        )

        with self._url_cache_lock:
            self._url_cache[cache_key] = (signed_at, url)
            stale_before = signed_at - SIGNED_URL_REUSE_WINDOW.total_seconds()
            for key, (entry_signed_at, _) in list(self._url_cache.items()):
                if entry_signed_at <= stale_before:
                    del self._url_cache[key]
        return url

    def copy(self, digest: str, *, from_tier: Tier, to_tier: Tier) -> StoredArtifact:
        """Copy from the recorded source location. Record the destination after success."""
        location = self._inventory.get_location(digest, tier=from_tier)
        if location is None:
            raise ArtifactNotFoundError(digest=digest, tier=from_tier)
        source_bucket = self._client.bucket(location.bucket)
        source_blob = source_bucket.blob(location.object_name)
        dest_bucket_name = self._buckets[to_tier]
        dest_bucket = self._client.bucket(dest_bucket_name)

        deduplicated = False
        copy_failed = False
        size_bytes: int | None = None
        try:
            dest_blob = source_bucket.copy_blob(
                source_blob, dest_bucket, new_name=digest, if_generation_match=0
            )
            size_bytes = dest_blob.size
        except PreconditionFailed:
            deduplicated = True
        except Exception:
            copy_failed = True
        if copy_failed:
            raise ArtifactWriteError(object_name=digest, digest=digest)
        if size_bytes is None:
            existing = dest_bucket.get_blob(digest)
            if existing is None:
                raise ArtifactWriteError(object_name=digest, digest=digest)
            size_bytes = existing.size

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
        """List digest names. GCS's `sha256:*` glob excludes staged objects containing `/`."""
        return frozenset(
            blob.name
            for blob in self._client.list_blobs(self._buckets[tier], match_glob="sha256:*")
        )


def _digest_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()
