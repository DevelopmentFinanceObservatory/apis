"""Artifact errors carry only the object name and digest, never driver messages, URLs, or query
strings.

Driver errors can include signed URLs and credentials, and the platform logs
exception chains. Wrap sites raise typed errors after leaving `except`
blocks. `raise ... from None` leaves `__context__` set. Starlette middleware
can restore it as a cause.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from artifact_store.store import Tier


# Alerts and reconciliation use this marker for writes whose bytes landed but
# whose inventory row failed. Keep the marker stable.
ARTIFACT_INVENTORY_WRITE_FAILED_MARKER = "artifact_inventory_write_failed"


class ArtifactStoreError(Exception):
    """Base for every artifact_store exception."""


class StorageEmulatorRefused(ArtifactStoreError):
    """`STORAGE_EMULATOR_HOST` is set outside the local stack, where it would hand out unsigned
    links. The local stack sets `DFO_LOCAL_STACK=1` beside it."""

    def __init__(self) -> None:
        super().__init__(
            "STORAGE_EMULATOR_HOST is set but DFO_LOCAL_STACK is not 1: an emulator serves "
            "unsigned links, so only the local stack (scripts/local-stack/local.env) may set it; "
            "unset STORAGE_EMULATOR_HOST to sign links through IAM"
        )


class ArtifactWriteError(ArtifactStoreError):
    """A `put()` failure other than an if-generation-match deduplication race."""

    def __init__(self, *, object_name: str, digest: str) -> None:
        self.object_name = object_name
        self.digest = digest
        super().__init__(f"failed to write artifact object {object_name!r} (digest {digest!r})")


class ArtifactSigningError(ArtifactStoreError):
    """V4 URL signing failed.

    Stores only the underlying exception's class and, for `google.api_core`
    errors, HTTP status. Driver messages can expose the string-to-sign or URL.
    """

    def __init__(
        self,
        *,
        bucket: str,
        object_name: str,
        digest: str,
        error_class_name: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.bucket = bucket
        self.object_name = object_name
        self.digest = digest
        self.error_class_name = error_class_name
        self.status_code = status_code
        detail = (
            f"failed to sign a GET url for artifact object {object_name!r} "
            f"in bucket {bucket!r} (digest {digest!r})"
        )
        if error_class_name is not None:
            detail += f"; underlying failure class {error_class_name!r}"
        if status_code is not None:
            detail += f", status {status_code!r}"
        super().__init__(detail)


class ArtifactInventoryError(ArtifactStoreError):
    """Inventory recording or lookup failed.

    Location fields are `None` when unknown.
    """

    def __init__(
        self,
        *,
        digest: str,
        tier: Tier | None = None,
        bucket: str | None = None,
        object_name: str | None = None,
    ) -> None:
        self.digest = digest
        self.tier = tier
        self.bucket = bucket
        self.object_name = object_name
        detail = f"failed to access inventory for artifact digest {digest!r}"
        if tier is not None:
            detail += f" tier {tier.value!r}"
        if bucket is not None:
            detail += f" bucket {bucket!r}"
        if object_name is not None:
            detail += f" object {object_name!r}"
        super().__init__(detail)


class ArtifactRecordConflictError(ArtifactStoreError):
    """A record's bucket, object name, or size differs from the stored row.

    Identical retries succeed. A bucket change can leave the fresh object
    outside the recorded location. `stored_*` and `incoming_*` identify both.
    Callers may attach `subject_id` and `content_version`. Both default to `None`.
    """

    subject_id: str | None = None
    content_version: str | None = None

    def __init__(
        self,
        *,
        digest: str,
        tier: Tier,
        stored_bucket: str,
        stored_object_name: str,
        stored_size_bytes: int,
        incoming_bucket: str,
        incoming_object_name: str,
        incoming_size_bytes: int,
    ) -> None:
        self.digest = digest
        self.tier = tier
        self.stored_bucket = stored_bucket
        self.stored_object_name = stored_object_name
        self.stored_size_bytes = stored_size_bytes
        self.incoming_bucket = incoming_bucket
        self.incoming_object_name = incoming_object_name
        self.incoming_size_bytes = incoming_size_bytes
        super().__init__(
            f"digest {digest!r} tier {tier.value!r} already has a stored bucket "
            f"{stored_bucket!r} object {stored_object_name!r} size {stored_size_bytes!r}; "
            f"the incoming bucket {incoming_bucket!r} object {incoming_object_name!r} "
            f"size {incoming_size_bytes!r} does not match it"
        )


class ArtifactLocationDriftError(ArtifactStoreError):
    """The recorded bucket differs from tier configuration.

    Signing or reading a location in another bucket could reach the wrong object.
    """

    def __init__(
        self, *, digest: str, tier: Tier, stored_bucket: str, stored_object_name: str
    ) -> None:
        self.digest = digest
        self.tier = tier
        self.stored_bucket = stored_bucket
        self.stored_object_name = stored_object_name
        super().__init__(
            f"artifact digest {digest!r} tier {tier.value!r} is recorded as bucket "
            f"{stored_bucket!r} object {stored_object_name!r}, which is not this store's "
            f"configured bucket for that tier"
        )


class ArtifactNotFoundError(ArtifactStoreError):
    """Missing inventory location for (`digest`, `tier`), or a bucket 404.

    Tier retention can delete an object while its location row persists.
    """

    def __init__(self, *, digest: str, tier: Tier) -> None:
        self.digest = digest
        self.tier = tier
        super().__init__(f"no artifact stored for digest {digest!r} in tier {tier.value!r}")


class ArtifactIntegrityError(ArtifactStoreError):
    """Fetched bytes hash to a different digest, indicating corruption or overwrite.

    `computed_digest` is the observed SHA-256. `ArtifactNotFoundError`
    covers missing objects.
    """

    def __init__(self, *, digest: str, tier: Tier, computed_digest: str) -> None:
        self.digest = digest
        self.tier = tier
        self.computed_digest = computed_digest
        super().__init__(
            f"artifact object for digest {digest!r} in tier {tier.value!r} does not hash to "
            f"that digest -- computed {computed_digest!r}"
        )
