"""Inventory repository for GCS artifacts, backed by `dataset_store`.

`record()` runs after bytes land. It keeps one inventory row per digest and
one location per (digest, tier). A fresh upload refreshes `stored_at`, which
controls the working tier's retention clock, and moves the location to the
object it uploaded, since those bytes are the newest copy. A new location row
takes the caller's `stored_at`, the existing object's creation time for a
deduplication that found no row, or now. A deduplication against another object
of the same digest in the same bucket keeps the location, since those bytes are
already inventoried. A different bucket or size raises
`ArtifactRecordConflictError`.

A location's `object_name` is the bare digest until the working tier renews
it. Runtime identities cannot overwrite an object, so bytes whose working
location is past retention upload again under `<digest>#r<renewal>`, and
`renew()` moves the location to that name with a fresh `stored_at`, or the
creation time of a taken renewal name the write adopts, creating the row when
none exists.

An `InventoryEntry` is the row one write owes. `write_entry` applies it inside
a caller's transaction, which is how a serve records every artifact it wrote
in the transaction that links them (`InventoryBatch`,
`LiveArtifactStore.deferred`). `record()` and `renew()` apply one entry in
their own transaction.

Synchronous interface. Async callers use `asyncio.to_thread()`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast

from artifact_store.errors import ArtifactInventoryError, ArtifactRecordConflictError
from dataset_store.store import DatasetStore, Transaction

if TYPE_CHECKING:
    from artifact_store.store import Tier

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ArtifactLocation:
    """The recorded bucket and object name of one (digest, tier), which may differ from tier
    configuration. `stored_at` starts the tier's retention clock, and `renewal` is the highest
    renewal name `<digest>#r<renewal>` taken for it (0 when none), so the next renewal tries the
    name after it. A fresh upload under the bare digest keeps `renewal` and moves `object_name`
    back to the digest."""

    bucket: str
    object_name: str
    stored_at: datetime
    renewal: int


@dataclass(frozen=True)
class InventoryEntry:
    """The inventory row one artifact write owes. It holds `record()`'s arguments, or `renew()`'s
    when `renewal` is set. A renewal's `stored_at` is `None` after a fresh upload and the object's
    creation time when the write adopted a renewal name it found taken."""

    digest: str
    tier: Tier
    bucket: str
    object_name: str
    size_bytes: int
    uploaded: bool
    stored_at: datetime | None = None
    renewal: int | None = None


class InventoryBatch:
    """The entries a group of artifact writes owes, held for one caller transaction.

    `LiveArtifactStore.deferred` adds to it in place of recording, from any thread, and the
    caller applies `entries` with `write_entry` in the transaction that links the artifacts.
    """

    def __init__(self) -> None:
        self._entries: list[InventoryEntry] = []
        self._lock = threading.Lock()

    def add(self, entry: InventoryEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    @property
    def entries(self) -> tuple[InventoryEntry, ...]:
        with self._lock:
            return tuple(self._entries)


def write_entry(tx: Transaction, entry: InventoryEntry, *, now: datetime) -> None:
    """Apply `entry` inside `tx` as `renew()`'s write when `entry.renewal` is set, else
    `record()`'s, with `now` standing in for an absent `stored_at`.

    A deduplication whose object differs from the location's, as when a concurrent writer's
    fresh upload moved the location first, keeps the location. Both objects hold the digest's
    bytes in one bucket. Raises `ArtifactRecordConflictError` when a recorded row differs in
    bucket or size. The caller's transaction then rolls back. Store failures propagate as the
    driver raises them.
    """
    tx.execute(
        "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes) "
        "VALUES (%s, %s) ON CONFLICT (artifact_digest) DO NOTHING",
        (entry.digest, entry.size_bytes),
    )
    stored_at = now if entry.stored_at is None else entry.stored_at
    if entry.renewal is not None:
        tx.execute(
            "INSERT INTO datasets.artifact_location "
            "(artifact_digest, tier, bucket, object_name, stored_at, renewal) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (artifact_digest, tier) DO UPDATE SET "
            "bucket = excluded.bucket, object_name = excluded.object_name, "
            "stored_at = excluded.stored_at, renewal = excluded.renewal "
            "WHERE datasets.artifact_location.renewal < excluded.renewal",
            (
                entry.digest,
                entry.tier.value,
                entry.bucket,
                entry.object_name,
                stored_at,
                entry.renewal,
            ),
        )
        return
    # Qualified columns name the stored row, not `excluded`. A deduplication keeps the stored object
    # and its retention timestamp. A fresh upload in the same bucket moves the location to the
    # object it wrote and restarts the clock.
    tx.execute(
        "INSERT INTO datasets.artifact_location "
        "(artifact_digest, tier, bucket, object_name, stored_at) "
        "VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (artifact_digest, tier) DO UPDATE SET "
        "object_name = CASE WHEN %s "
        "AND datasets.artifact_location.bucket = excluded.bucket "
        "THEN excluded.object_name ELSE datasets.artifact_location.object_name END, "
        "stored_at = CASE WHEN %s "
        "AND datasets.artifact_location.bucket = excluded.bucket "
        "THEN excluded.stored_at ELSE datasets.artifact_location.stored_at END",
        (
            entry.digest,
            entry.tier.value,
            entry.bucket,
            entry.object_name,
            stored_at,
            entry.uploaded,
            entry.uploaded,
        ),
    )
    stored_inventory = tx.fetch_one(
        "SELECT size_bytes FROM datasets.artifact_inventory WHERE artifact_digest = %s",
        (entry.digest,),
    )
    stored_location = tx.fetch_one(
        "SELECT bucket, object_name FROM datasets.artifact_location "
        "WHERE artifact_digest = %s AND tier = %s",
        (entry.digest, entry.tier.value),
    )
    if stored_inventory is None or stored_location is None:
        return
    conflicts = (
        stored_inventory["size_bytes"] != entry.size_bytes
        or stored_location["bucket"] != entry.bucket
    )
    if conflicts:
        raise ArtifactRecordConflictError(
            digest=entry.digest,
            tier=entry.tier,
            stored_bucket=cast(str, stored_location["bucket"]),
            stored_object_name=cast(str, stored_location["object_name"]),
            stored_size_bytes=cast(int, stored_inventory["size_bytes"]),
            incoming_bucket=entry.bucket,
            incoming_object_name=entry.object_name,
            incoming_size_bytes=entry.size_bytes,
        )


class ArtifactInventory(Protocol):
    def record(
        self,
        *,
        digest: str,
        tier: Tier,
        bucket: str,
        object_name: str,
        size_bytes: int,
        uploaded: bool,
        stored_at: datetime | None = None,
    ) -> None:
        """Record one row per digest and one location per tier, idempotently.

        `stored_at` is when the bytes at `object_name` landed, now when `None`. A new location
        row takes it, and a fresh upload refreshes an existing row's to it. Dedup calls preserve
        an existing row's.
        Raises `ArtifactInventoryError` on failure or
        `ArtifactRecordConflictError` for a different bucket or size.
        """
        ...

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
        """Point the location at the renewal object `object_name`, with `stored_at` and
        `renewal` recorded, creating the inventory and location rows when absent.

        `stored_at` is when the bytes at `object_name` landed: now when `None`, as after a fresh
        upload, or the object's creation time when a write adopts a renewal name it found taken.
        A location already at `renewal` or later stays, so a slower concurrent renewer never
        moves it back, and an uploader and an adopter of one name converge on one location.
        Raises `ArtifactInventoryError` on failure.
        """
        ...

    def get_location(self, digest: str, *, tier: Tier) -> ArtifactLocation | None:
        """Return the stored location, or `None` if no row exists.

        Tier retention can delete an object while its location row persists.
        Raises `ArtifactInventoryError` if the lookup fails.
        """
        ...


class LiveArtifactInventory:
    """DatasetStore-backed inventory.

    `now` supplies the UTC timestamp for `stored_at`.
    """

    def __init__(self, *, store: DatasetStore, now: Clock = utc_now) -> None:
        self._store = store
        self._now = now

    def record(
        self,
        *,
        digest: str,
        tier: Tier,
        bucket: str,
        object_name: str,
        size_bytes: int,
        uploaded: bool,
        stored_at: datetime | None = None,
    ) -> None:
        self.write(
            InventoryEntry(
                digest=digest,
                tier=tier,
                bucket=bucket,
                object_name=object_name,
                size_bytes=size_bytes,
                uploaded=uploaded,
                stored_at=stored_at,
            )
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
        self.write(
            InventoryEntry(
                digest=digest,
                tier=tier,
                bucket=bucket,
                object_name=object_name,
                size_bytes=size_bytes,
                uploaded=stored_at is None,
                stored_at=stored_at,
                renewal=renewal,
            )
        )

    def write(self, entry: InventoryEntry) -> None:
        """Apply `entry` in its own transaction (`write_entry`). Raises
        `ArtifactRecordConflictError` for a conflicting row and `ArtifactInventoryError` for a
        store failure."""
        write_failed = False
        try:
            with self._store.transaction() as tx:
                write_entry(tx, entry, now=self._now())
        except ArtifactRecordConflictError:
            raise
        except Exception:
            write_failed = True
        if write_failed:
            raise ArtifactInventoryError(
                digest=entry.digest,
                tier=entry.tier,
                bucket=entry.bucket,
                object_name=entry.object_name,
            )

    def get_location(self, digest: str, *, tier: Tier) -> ArtifactLocation | None:
        lookup_failed = False
        row = None
        try:
            with self._store.transaction() as tx:
                row = tx.fetch_one(
                    "SELECT bucket, object_name, stored_at, renewal FROM datasets.artifact_location "
                    "WHERE artifact_digest = %s AND tier = %s",
                    (digest, tier.value),
                )
        except Exception:
            lookup_failed = True
        if lookup_failed:
            raise ArtifactInventoryError(digest=digest, tier=tier)
        if row is None:
            return None
        return ArtifactLocation(
            bucket=cast(str, row["bucket"]),
            object_name=cast(str, row["object_name"]),
            stored_at=cast(datetime, row["stored_at"]),
            renewal=cast(int, row["renewal"]),
        )
