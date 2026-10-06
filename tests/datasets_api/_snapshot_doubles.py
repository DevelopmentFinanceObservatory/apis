"""Snapshot readers and artifact rows for datasets API door tests.
Includes reader doubles, row builders, and a helper for mounting one door on a platform app.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from artifact_store import Tier
from dataset_store.snapshot import Revocation, StoredArtifactLocation, StoredVersion
from datasets_api.codes import DATASETS_CODE_CATALOGUE
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from dfo_platform.app import Surface, create_app
from dfo_platform.doors import DoorContribution

from ._artifact_doubles import FakeArtifactStore

_NOW = datetime(2026, 6, 1, tzinfo=UTC)


class FakeSpecReader:
    """Return a sentinel for known IDs; callers check presence by testing for `None`."""

    def __init__(self, *, spec_ids: Sequence[str] = ()) -> None:
        self._spec_ids = set(spec_ids)

    def read_definition(self, spec_id: str, *, schema_version: int | None = None):
        return object() if spec_id in self._spec_ids else None

    def read_definitions(self, spec_ids):
        return {}


class BlockingFakeSpecReader(FakeSpecReader):
    """Block `read_definition` on `release` so backpressure tests can occupy an executor slot.

    Other reads use `FakeSpecReader` behavior.
    """

    def __init__(self, *, spec_ids: Sequence[str] = ()) -> None:
        super().__init__(spec_ids=spec_ids)
        self.entered = threading.Event()
        self.release = threading.Event()

    def read_definition(self, spec_id: str, *, schema_version: int | None = None):
        self.entered.set()
        self.release.wait(timeout=5.0)
        return super().read_definition(spec_id, schema_version=schema_version)


@dataclass
class FakeSnapshotReader:
    versions: dict[tuple[str, str], StoredVersion] = field(default_factory=dict)
    locations: dict[tuple[str, str], list[StoredArtifactLocation]] = field(default_factory=dict)
    revocations: dict[tuple[str, str], Revocation] = field(default_factory=dict)

    def read_version(self, subject_id: str, content_version: str) -> StoredVersion | None:
        return self.versions.get((subject_id, content_version))

    def read_artifact_locations(
        self, subject_id: str, content_version: str
    ) -> list[StoredArtifactLocation]:
        return list(self.locations.get((subject_id, content_version), []))

    def read_revocation(self, subject_id: str, content_version: str) -> Revocation | None:
        return self.revocations.get((subject_id, content_version))


@dataclass
class StoreBackedSnapshotReader(FakeSnapshotReader):
    """Lists a pinned location for every linked digest the store holds in the pinned tier, as
    the live join over `datasets.artifact_location` does once `copy` records the destination.
    """

    store: FakeArtifactStore = field(default_factory=FakeArtifactStore)

    def read_artifact_locations(
        self, subject_id: str, content_version: str
    ) -> list[StoredArtifactLocation]:
        linked = super().read_artifact_locations(subject_id, content_version)
        listed = {(location.artifact_digest, location.tier) for location in linked}
        promoted = [
            replace(location, tier=Tier.pinned.value)
            for location in linked
            if (location.artifact_digest, Tier.pinned.value) in self.store._objects
            and (location.artifact_digest, Tier.pinned.value) not in listed
        ]
        return linked + promoted


def _version(
    *,
    immutable_body: Mapping[str, object] | None = None,
    content_version: str = "cv1",
    base_content_version: str | None = None,
) -> StoredVersion:
    return StoredVersion(
        subject_id="spec_a",
        subject_kind="spec",
        content_version=content_version,
        schema_version=1,
        immutable_body=immutable_body
        if immutable_body is not None
        else {
            "schema_version": 1,
            "definition": {
                "schema_version": 1,
                "spec_id": "spec_a",
                "identity": {"identity_version": "spec1"},
            },
            "content_version": content_version,
            "prepared_at": "2026-01-01T00:00:00Z",
            "coverage": {"rows": 1, "dimensions": []},
            "citation": {"text": "", "sources": []},
            "trace": {},
            "warnings": [],
        },
        base_content_version=base_content_version,
    )


def _location(
    *,
    digest: str = "sha256:aaa",
    tier: str = "working",
    stored_at: datetime = _NOW,
    created_at: datetime = _NOW,
    rendering_key: str | None = None,
    rendering: Mapping[str, object] | None = None,
    media_type: str = "text/csv",
) -> StoredArtifactLocation:
    return StoredArtifactLocation(
        artifact_digest=digest,
        # Default keys are per digest. Tests with multiple renderings for one digest pass a key.
        rendering_key=rendering_key if rendering_key is not None else f"rk:{digest}",
        rendering=dict(rendering) if rendering is not None else {"format": "csv", "bundle": False},
        media_type=media_type,
        size_bytes=10,
        created_at=created_at,
        stored_at=stored_at,
        tier=tier,
        bucket="fake-bucket",
        object_name=digest,
    )


def _store_and_locate(
    store: FakeArtifactStore,
    data: bytes,
    *,
    tier: Tier = Tier.working,
    stored_at: datetime = _NOW,
    created_at: datetime = _NOW,
    rendering_key: str | None = None,
    rendering: Mapping[str, object] | None = None,
    media_type: str = "text/csv",
) -> StoredArtifactLocation:
    """Store `data` and return a location with its actual digest."""
    stored = store.put(data, tier=tier)
    return _location(
        digest=stored.digest,
        tier=tier.value,
        stored_at=stored_at,
        created_at=created_at,
        rendering_key=rendering_key,
        rendering=rendering,
        media_type=media_type,
    )


def make_client_for_door(door: DoorContribution) -> TestClient:
    """Mount one door on a bare platform app and preserve its middleware and exception handlers.

    Use this when the full datasets surface is unnecessary.
    """
    surface = Surface(
        name="datasets",
        prefix="/api/datasets",
        router=door.router,
        capability_version=DATASETS_CAPABILITY_VERSION,
        codes=DATASETS_CODE_CATALOGUE,
        classify=door.classify,
        exception_types=door.exception_types,
        exception_exemplars=door.exception_exemplars,
        middleware=door.middleware,
    )
    return TestClient(create_app([surface]), raise_server_exceptions=False)
