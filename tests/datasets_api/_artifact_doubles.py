"""Artifact-store doubles for datasets API tests.
`FakeArtifactStore` stores objects in memory; `RecordingArtifactStore` captures live-store inputs.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta

from artifact_store import (
    ArtifactInventory,
    ArtifactNotFoundError,
    ArtifactSink,
    ArtifactStore,
    CopyBounds,
    ExistsRead,
    StoredArtifact,
    Tier,
)
from artifact_store.inventory import ArtifactLocation, InventoryBatch


class _BufferedSink(io.BytesIO):
    """Buffer `open_put` bytes and attach the `StoredArtifact` after commit.

    `BytesIO` supplies the sink methods; this class adds result storage.
    """

    def __init__(self) -> None:
        super().__init__()
        self._result: StoredArtifact | None = None

    def _set_result(self, result: StoredArtifact) -> None:
        self._result = result

    def result(self) -> StoredArtifact:
        if self._result is None:
            raise RuntimeError("ArtifactSink.result() called before open_put committed")
        return self._result


class FakeArtifactStore:
    """Store bytes by digest and tier, with fake URLs resolved by `fetch`. `gets` records reads.

    Signed response type and disposition are recorded per tier and digest, and `signings` records
    each signing call's digest, tier, object name and ttl. No GCS signing occurs.
    """

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}
        self.signed_headers: dict[tuple[Tier, str], tuple[str, str]] = {}
        self.signings: list[tuple[str, Tier, str, timedelta]] = []
        self.gets: list[tuple[str, Tier]] = []
        self.locations: list[tuple[str, Tier]] = []

    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        key = (digest, tier.value)
        deduplicated = key in self._objects
        self._objects[key] = data
        return StoredArtifact(
            digest=digest,
            tier=tier,
            bucket="fake-bucket",
            object_name=digest,
            size_bytes=len(data),
            deduplicated=deduplicated,
        )

    @contextmanager
    def open_put(self, *, tier: Tier) -> Generator[ArtifactSink, None, None]:
        sink = _BufferedSink()
        yield sink
        sink._set_result(self.put(sink.getvalue(), tier=tier))

    def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
        """The stored bytes; each read is recorded in `gets`."""
        self.gets.append((artifact_digest, tier))
        key = (artifact_digest, tier.value)
        if key not in self._objects:
            raise ArtifactNotFoundError(digest=artifact_digest, tier=tier)
        return self._objects[key]

    copy_lands_within = CopyBounds().lands_within

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
        key = (digest, tier.value)
        if key not in self._objects:
            raise ArtifactNotFoundError(digest=digest, tier=tier)
        self.signed_headers[(tier, digest)] = (response_type, response_disposition)
        self.signings.append((digest, tier, object_name, ttl))
        return f"fake://artifact/{tier.value}/{object_name}"

    def exists(self, digest: str, *, tier: Tier, read: ExistsRead = ExistsRead.short) -> bool:
        return (digest, tier.value) in self._objects

    def location(self, digest: str, *, tier: Tier) -> ArtifactLocation | None:
        """The fake location; each read is recorded in `locations`."""
        self.locations.append((digest, tier))
        if (digest, tier.value) not in self._objects:
            return None
        return ArtifactLocation(
            bucket="fake-bucket", object_name=digest, stored_at=datetime.min, renewal=0
        )

    def copy(
        self,
        digest: str,
        *,
        from_tier: Tier,
        to_tier: Tier,
        source: tuple[str, str] | None = None,
    ) -> StoredArtifact:
        source_key = (digest, from_tier.value)
        if source_key not in self._objects:
            raise ArtifactNotFoundError(digest=digest, tier=from_tier)
        dest_key = (digest, to_tier.value)
        deduplicated = dest_key in self._objects
        data = self._objects[source_key]
        self._objects[dest_key] = data
        return StoredArtifact(
            digest=digest,
            tier=to_tier,
            bucket="fake-bucket",
            object_name=digest,
            size_bytes=len(data),
            deduplicated=deduplicated,
        )

    def list(self, *, tier: Tier) -> frozenset[str]:
        return frozenset(digest for digest, tier_value in self._objects if tier_value == tier.value)

    def deferred(self, batch: InventoryBatch) -> ArtifactStore:
        """This store: it records no inventory, so a write owes no row."""
        return self

    def fetch(self, url: str) -> bytes:
        """The bytes behind a signed fake URL, or behind a manifest's stable download link
        (`/api/datasets/{id}/{content_version}/files/{digest}`), read from the pinned tier when
        it holds them and the working tier otherwise."""
        prefix = "fake://artifact/"
        if not url.startswith(prefix):
            assert "/files/" in url, f"not a fake artifact URL or download link: {url!r}"
            digest = url.rsplit("/", 1)[1]
            pinned = self._objects.get((digest, Tier.pinned.value))
            return pinned if pinned is not None else self._objects[(digest, Tier.working.value)]
        tier_value, object_name = url[len(prefix) :].split("/", 1)
        return self._objects[(object_name.split("#", 1)[0], tier_value)]


class RecordingArtifactStore:
    """Capture `LiveArtifactStore` bucket and inventory arguments without creating a GCS client."""

    def __init__(self, *, buckets: Mapping[Tier, str], inventory: ArtifactInventory) -> None:
        self.buckets = buckets
        self.inventory = inventory
