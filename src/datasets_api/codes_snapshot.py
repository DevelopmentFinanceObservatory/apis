"""Snapshot, observations, pin and download refusal codes. The doors reuse `codes_core`'s
`spec_not_found`. `snapshot_not_found`, `snapshot_revoked` and `artifact_not_found` raise from
`dataset_store.snapshot.read_servable_version`, so their classes live there."""

from __future__ import annotations

from dfo_platform.problem import Refusal

SNAPSHOT_DOOR_CODE_CATALOGUE: dict[str, tuple[str, str]] = {
    "snapshot_not_found": (
        "Snapshot Not Found",
        "This id is known, but no version was ever materialised at the given content_version.",
    ),
    "artifact_expired": (
        "Artifact Expired",
        "This version has no servable artifact left; the working-tier copy aged out unpinned.",
    ),
    "snapshot_revoked": (
        "Snapshot Revoked",
        "This snapshot was taken down and stays revoked.",
    ),
    "observations_unavailable": (
        "Observations Unavailable",
        "The observations read cannot build from this version: it holds no stored frame, a "
        "converted series lacks its per-series conversion record, the version does not store "
        "its default place scope, or it is a collapsed version.",
    ),
    "artifact_not_found": (
        "Artifact Not Found",
        "The named artifact_digest is not among this version's artifacts.",
    ),
    "pin_not_permitted": (
        "Pin Not Permitted",
        "The authenticated caller lacks the pin permission, or programmatic pinning "
        "is not live on this deployment.",
    ),
}


class SpecNotFound(Refusal):
    """The path id is unregistered. It maps to shared `spec_not_found`, 404."""

    code = "spec_not_found"

    def __init__(self, *, id: str) -> None:
        self.id = id
        super().__init__(f"{id!r} is not registered")


class ArtifactExpired(Refusal):
    """The version's working-tier artifacts are expired and its pinned tier has no copy.

    Snapshot retrieval maps this to `artifact_expired`, 410, directing callers to definition
    retrieval. Pin requests report the named artifact's expiry.
    """

    code = "artifact_expired"

    def __init__(self, *, id: str, content_version: str) -> None:
        self.id = id
        self.content_version = content_version
        super().__init__(
            f"{id!r} at content_version {content_version!r} has no servable artifact left; "
            "re-request through definition retrieval"
        )


class ObservationsUnavailable(Refusal):
    """The version is derived (a groupby collapse or a party-totals cut), holds no servable
    stored frame where its preview reads rows, holds a converted series without its per-series
    conversion record, or does not say whether its default place scope was plural. It maps to `observations_unavailable`, 404."""

    code = "observations_unavailable"

    def __init__(self, *, id: str, content_version: str, reason: str) -> None:
        self.id = id
        self.content_version = content_version
        super().__init__(f"{id!r} at content_version {content_version!r} {reason}")


class SnapshotSigningFailed(Exception):
    """A signed URL for a stored artifact failed.

    The raise site copies safe error fields outside the caught exception so `__context__` stays
    unset. `error_class_name` and `status_code` distinguish permanent causes from outages.
    """

    def __init__(
        self,
        *,
        bucket: str,
        object_name: str,
        digest: str,
        error_class_name: str | None,
        status_code: int | None,
    ) -> None:
        self.bucket = bucket
        self.object_name = object_name
        self.digest = digest
        self.error_class_name = error_class_name
        self.status_code = status_code
        detail = (
            f"failed to sign a GET url for a stored artifact object {object_name!r} "
            f"in bucket {bucket!r} (digest {digest!r})"
        )
        if error_class_name is not None:
            detail += f"; underlying failure class {error_class_name!r}"
        if status_code is not None:
            detail += f", status {status_code!r}"
        super().__init__(detail)
