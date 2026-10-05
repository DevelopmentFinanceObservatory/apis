"""Inventory repository for GCS artifacts, backed by `dataset_store`.

`record()` runs after bytes land. It keeps one inventory row per digest and
one location per (digest, tier). Only a fresh upload refreshes `stored_at`,
which controls the working tier's retention clock. Conflicting locations or
sizes raise `ArtifactRecordConflictError`.

Between `stored_at` + 30 days and the working bucket's lifecycle delete (age
30, applied within about a day), a re-render of the same bytes dedups against
the old object and leaves its location expired. The version link stays
unservable, so each request in that window renders again and a pin refuses.
The first render after the lifecycle delete uploads afresh, resets
`stored_at`, and the link serves again. An overwrite would close the window
but needs `storage.objects.delete` on the working bucket, which the runtime
identities do not hold.

Synchronous interface. Async callers use `asyncio.to_thread()`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast

from artifact_store.errors import ArtifactInventoryError, ArtifactRecordConflictError
from artifact_store.store import Tier
from dataset_store.store import DatasetStore

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ArtifactLocation:
    """The bucket and object name recorded for signing, which may differ from tier configuration."""

    bucket: str
    object_name: str


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
    ) -> None:
        """Record one row per digest and one location per tier, idempotently.

        Fresh uploads refresh `stored_at`. Dedup calls preserve it.
        Raises `ArtifactInventoryError` on failure or
        `ArtifactRecordConflictError` for a different bucket, object, or size.
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
    ) -> None:
        stored_at = self._now()
        record_failed = False
        stored_inventory = None
        stored_location = None
        try:
            with self._store.transaction() as tx:
                tx.execute(
                    "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes) "
                    "VALUES (%s, %s) ON CONFLICT (artifact_digest) DO NOTHING",
                    (digest, size_bytes),
                )
                # Qualify these columns to select the stored row, not `excluded`.
                # Dedup hits and conflicting locations must preserve its retention timestamp.
                tx.execute(
                    "INSERT INTO datasets.artifact_location "
                    "(artifact_digest, tier, bucket, object_name, stored_at) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (artifact_digest, tier) DO UPDATE SET "
                    "stored_at = CASE "
                    "WHEN %s "
                    "AND datasets.artifact_location.bucket = excluded.bucket "
                    "AND datasets.artifact_location.object_name = excluded.object_name "
                    "THEN %s ELSE datasets.artifact_location.stored_at END",
                    (digest, tier.value, bucket, object_name, stored_at, uploaded, stored_at),
                )
                stored_inventory = tx.fetch_one(
                    "SELECT size_bytes FROM datasets.artifact_inventory WHERE artifact_digest = %s",
                    (digest,),
                )
                stored_location = tx.fetch_one(
                    "SELECT bucket, object_name FROM datasets.artifact_location "
                    "WHERE artifact_digest = %s AND tier = %s",
                    (digest, tier.value),
                )
        except Exception:
            record_failed = True
        if record_failed:
            raise ArtifactInventoryError(
                digest=digest, tier=tier, bucket=bucket, object_name=object_name
            )
        if stored_inventory is not None and stored_location is not None:
            conflicts = (
                stored_inventory["size_bytes"] != size_bytes
                or stored_location["bucket"] != bucket
                or stored_location["object_name"] != object_name
            )
            if conflicts:
                raise ArtifactRecordConflictError(
                    digest=digest,
                    tier=tier,
                    stored_bucket=cast(str, stored_location["bucket"]),
                    stored_object_name=cast(str, stored_location["object_name"]),
                    stored_size_bytes=cast(int, stored_inventory["size_bytes"]),
                    incoming_bucket=bucket,
                    incoming_object_name=object_name,
                    incoming_size_bytes=size_bytes,
                )

    def get_location(self, digest: str, *, tier: Tier) -> ArtifactLocation | None:
        lookup_failed = False
        row = None
        try:
            with self._store.transaction() as tx:
                row = tx.fetch_one(
                    "SELECT bucket, object_name FROM datasets.artifact_location "
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
            bucket=cast(str, row["bucket"]), object_name=cast(str, row["object_name"])
        )
