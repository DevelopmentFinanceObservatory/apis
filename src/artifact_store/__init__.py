"""Content-addressed GCS storage, inventory records, and V4 signed URLs for the Datasets API.

Client and credential creation is deferred to `LiveArtifactStore` methods.
"""

from __future__ import annotations

from artifact_store.errors import (
    ARTIFACT_INVENTORY_WRITE_FAILED_MARKER,
    ArtifactInventoryError,
    ArtifactLocationDriftError,
    ArtifactNotFoundError,
    ArtifactRecordConflictError,
    ArtifactSigningError,
    ArtifactStoreError,
    ArtifactWriteError,
    StorageEmulatorRefused,
)
from artifact_store.inventory import ArtifactInventory, ArtifactLocation, LiveArtifactInventory
from artifact_store.signing import SIGNED_URL_TTL_CEILING, SIGNED_URL_TTL_FLOOR, V4_MAX_TTL
from artifact_store.store import (
    SPILL_THRESHOLD_BYTES,
    ArtifactSink,
    ArtifactStore,
    LiveArtifactStore,
    StoredArtifact,
    Tier,
)

__all__ = [
    "ARTIFACT_INVENTORY_WRITE_FAILED_MARKER",
    "SIGNED_URL_TTL_CEILING",
    "SIGNED_URL_TTL_FLOOR",
    "SPILL_THRESHOLD_BYTES",
    "V4_MAX_TTL",
    "ArtifactInventory",
    "ArtifactInventoryError",
    "ArtifactLocation",
    "ArtifactLocationDriftError",
    "ArtifactNotFoundError",
    "ArtifactRecordConflictError",
    "ArtifactSigningError",
    "ArtifactSink",
    "ArtifactStore",
    "ArtifactStoreError",
    "ArtifactWriteError",
    "LiveArtifactInventory",
    "LiveArtifactStore",
    "StorageEmulatorRefused",
    "StoredArtifact",
    "Tier",
]
