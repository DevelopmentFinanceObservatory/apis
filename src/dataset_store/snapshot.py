"""Snapshot version, artifact-location and revocation records for dataset retrieval and pinning.

Working-tier artifacts expire 30 days after `stored_at`. Pinned artifacts are permanent. Artifact
links retain the rendering and media type recorded for each version.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, cast

from dataset_store.store import DatasetStore

# The 30-day working-tier retention clock ignores downloads.
WORKING_TIER_RETENTION = timedelta(days=30)


def working_tier_expired(stored_at: datetime, *, now: datetime) -> bool:
    """Whether a working-tier artifact is more than 30 days old. Pinned locations are permanent."""
    return now - stored_at > WORKING_TIER_RETENTION


class SnapshotRevoked(Exception):
    """A taken-down version. The API layer that catches this maps it to
    `snapshot_revoked`, 410, on both snapshot retrieval and pin.
    """

    def __init__(self, *, id: str, content_version: str, reason: str | None) -> None:
        self.id = id
        self.content_version = content_version
        self.reason = reason
        detail = f"{id!r} at content_version {content_version!r} was taken down"
        if reason is not None:
            detail += f": {reason}"
        super().__init__(detail)


@dataclass(frozen=True)
class StoredVersion:
    subject_id: str
    subject_kind: str
    content_version: str
    schema_version: int
    immutable_body: Mapping[str, object]
    # Collapsed versions record the raw materialisation's content version. Other versions use None.
    base_content_version: str | None = None


@dataclass(frozen=True)
class StoredArtifactLocation:
    """One version-linked artifact location, keyed by digest, rendering and tier. A digest can have
    multiple renderings or both working and pinned locations. `created_at` records the digest's
    first sighting. Promotion leaves it unchanged. `stored_at` starts this tier's retention clock.
    """

    artifact_digest: str
    rendering_key: str
    rendering: Mapping[str, object]
    media_type: str
    size_bytes: int
    created_at: datetime
    stored_at: datetime
    tier: str
    bucket: str
    object_name: str


@dataclass(frozen=True)
class Revocation:
    reason: str | None
    revoked_at: datetime


class SnapshotReader(Protocol):
    def read_version(self, subject_id: str, content_version: str) -> StoredVersion | None:
        """Return `None` when the subject/version pair has no record."""
        ...

    def read_artifact_locations(
        self, subject_id: str, content_version: str
    ) -> list[StoredArtifactLocation]:
        """Return all linked locations, unfiltered by expiry. Data renderings precede
        `series-statistics/1`. Each group is ordered by `(artifact_digest, rendering_key, tier)`.
        """
        ...

    def read_revocation(self, subject_id: str, content_version: str) -> Revocation | None:
        """Return the revocation record, or `None` when the version is live."""
        ...


class LiveSnapshotReader:
    """`SnapshotReader` backed by one read-only store transaction per public call."""

    def __init__(self, *, store: DatasetStore) -> None:
        self._store = store

    def read_version(self, subject_id: str, content_version: str) -> StoredVersion | None:
        with self._store.transaction() as tx:
            row = tx.fetch_one(
                "SELECT subject_kind, schema_version, immutable_body, base_content_version "
                "FROM datasets.version_record WHERE subject_id = %s AND content_version = %s",
                (subject_id, content_version),
            )
        if row is None:
            return None
        return StoredVersion(
            subject_id=subject_id,
            subject_kind=cast(str, row["subject_kind"]),
            content_version=content_version,
            schema_version=cast(int, row["schema_version"]),
            immutable_body=cast(Mapping[str, object], row["immutable_body"]),
            base_content_version=cast("str | None", row["base_content_version"]),
        )

    def read_artifact_locations(
        self, subject_id: str, content_version: str
    ) -> list[StoredArtifactLocation]:
        with self._store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT va.artifact_digest AS artifact_digest, "
                "va.rendering_key AS rendering_key, va.rendering AS rendering, "
                "va.media_type AS media_type, ai.size_bytes AS size_bytes, "
                "ai.created_at AS created_at, al.stored_at AS stored_at, al.tier AS tier, "
                "al.bucket AS bucket, al.object_name AS object_name "
                "FROM datasets.version_artifact va "
                "JOIN datasets.artifact_inventory ai ON ai.artifact_digest = va.artifact_digest "
                "JOIN datasets.artifact_location al ON al.artifact_digest = va.artifact_digest "
                "WHERE va.subject_id = %s AND va.content_version = %s "
                # Put data renderings before the auxiliary statistics file, even when `statistics`
                # is absent. Remaining keys make multiple renderings and tiers deterministic.
                "ORDER BY COALESCE((va.rendering ->> 'statistics') = 'series', false), "
                "va.artifact_digest, va.rendering_key, al.tier",
                (subject_id, content_version),
            )
        return [
            StoredArtifactLocation(
                artifact_digest=cast(str, row["artifact_digest"]),
                rendering_key=cast(str, row["rendering_key"]),
                rendering=cast(Mapping[str, object], row["rendering"]),
                media_type=cast(str, row["media_type"]),
                size_bytes=cast(int, row["size_bytes"]),
                created_at=cast(datetime, row["created_at"]),
                stored_at=cast(datetime, row["stored_at"]),
                tier=cast(str, row["tier"]),
                bucket=cast(str, row["bucket"]),
                object_name=cast(str, row["object_name"]),
            )
            for row in rows
        ]

    def read_revocation(self, subject_id: str, content_version: str) -> Revocation | None:
        with self._store.transaction() as tx:
            row = tx.fetch_one(
                "SELECT reason, revoked_at FROM datasets.version_revocation "
                "WHERE subject_id = %s AND content_version = %s",
                (subject_id, content_version),
            )
        if row is None:
            return None
        return Revocation(
            reason=cast("str | None", row["reason"]), revoked_at=cast(datetime, row["revoked_at"])
        )


def select_servable_locations(
    locations: Sequence[StoredArtifactLocation], *, now: datetime
) -> list[StoredArtifactLocation]:
    """Choose one servable location per `(artifact_digest, rendering_key)`. Prefer `pinned`. Return
    an unexpired `working` location as fallback. Return pairs with a servable location. Group by
    rendering as well as digest because distinct renderings can share stored bytes.
    """
    by_rendering: dict[tuple[str, str], list[StoredArtifactLocation]] = {}
    for location in locations:
        by_rendering.setdefault((location.artifact_digest, location.rendering_key), []).append(
            location
        )

    servable: list[StoredArtifactLocation] = []
    for rendering_locations in by_rendering.values():
        pinned = next((loc for loc in rendering_locations if loc.tier == "pinned"), None)
        if pinned is not None:
            servable.append(pinned)
            continue
        working = next((loc for loc in rendering_locations if loc.tier == "working"), None)
        if working is not None and not working_tier_expired(working.stored_at, now=now):
            servable.append(working)
    return servable


def servable_location_of_media_type(
    locations: Sequence[StoredArtifactLocation], *, media_type: str, now: datetime
) -> StoredArtifactLocation | None:
    """Return the first servable location carrying `media_type`, or `None` when the version has
    no such rendering or every location of it has aged out unpinned. Servability follows
    `select_servable_locations`, so a pinned location wins over a working one.

    Callers locate a version's canonical frame by passing `materialise.formats.FRAME1_MEDIA_TYPE`.
    The constant stays in `materialise` because importing that module here would pull pyarrow
    into `dataset_store`.
    """
    matching = [location for location in locations if location.media_type == media_type]
    return next(iter(select_servable_locations(matching, now=now)), None)


class ArtifactReferenceWriter(Protocol):
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
        """Insert or reactivate the reference at `now`. A live row keeps its `created_at` timestamp.
        Return the resulting timestamp.
        """
        ...

    def record_reference_unless_revoked(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> datetime:
        """Same write as `record_reference`, made in one statement that also checks
        `datasets.version_revocation`. A revocation visible in the statement's starting snapshot
        raises `SnapshotRevoked` and writes nothing. A revocation committed later goes unseen,
        and nothing here locks against a concurrent revocation write.

        Before the write it takes the version's link lock, the one every artifact link write
        takes, and requires a `datasets.version_artifact` row linking `artifact_digest` to the
        version. A digest no longer linked, because a re-render replaced its expired link, raises
        `datasets_api.codes_snapshot.ArtifactNotFound` and writes nothing. A link write after
        this one commits sees the pinned location its caller recorded first, and keeps the link.
        """
        ...

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
        """Set `removed_at` on a matching live row. Missing or removed rows are a no-op."""
        ...
