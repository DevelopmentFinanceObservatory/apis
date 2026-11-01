"""Snapshot readers and artifact rows for datasets API door tests.
Includes reader doubles, row builders, and a helper for mounting one door on a platform app.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import cast

from fastapi.testclient import TestClient

from artifact_store import Tier
from code_index.shared import get_index
from dataset_store.clock import Clock, utc_now
from dataset_store.snapshot import (
    Revocation,
    StoredArtifactLocation,
    StoredVersion,
    VersionFacts,
    VersionState,
)
from datasets_api.codes import DATASETS_CODE_CATALOGUE
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from datasets_api.version_store import enriched_rendering_key
from datasets_api.versions import (
    NULL_VERSION_RECORD_WRITER,
    ArtifactRecord,
    NullVersionRecordWriter,
    VersionRecord,
    VersionRecordWriter,
    VersionWrite,
    WrittenVersion,
    unlinked_state,
)
from dfo_platform.app import Surface, create_app
from dfo_platform.doors import DoorContribution, endpoint_exemplars
from spec_registry.reader import StoredDefinition

from ._artifact_doubles import FakeArtifactStore
from ._version_facts import facts_from_body

_NOW = datetime(2026, 6, 1, tzinfo=UTC)


class FakeSpecReader:
    """Return a pinned-arm definition for known IDs, whose identity is its own canonical form
    (`code_index.members.canonical_identity`), so each id is its own served id and no served
    id is bound to it."""

    def __init__(self, *, spec_ids: Sequence[str] = ()) -> None:
        self._spec_ids = set(spec_ids)

    def read_definition(self, spec_id: str, *, schema_version: int | None = None):
        if spec_id not in self._spec_ids:
            return None
        return StoredDefinition(
            spec_id=spec_id,
            schema_version=1,
            identity={"shape": {"kind": "stat_var"}},
            identity_bytes=b"",
            identity_digest="",
        )

    def read_definitions(self, spec_ids):
        return {}

    def read_served_spec_ids(self, spec_id: str) -> tuple[str, ...] | None:
        return () if spec_id in self._spec_ids else None


class BlockingFakeSpecReader(FakeSpecReader):
    """Block `read_definition` and `read_served_spec_ids` on `release` so backpressure tests
    can occupy an executor slot.

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

    def read_served_spec_ids(self, spec_id: str) -> tuple[str, ...] | None:
        self.entered.set()
        self.release.wait(timeout=5.0)
        return super().read_served_spec_ids(spec_id)


@dataclass
class FakeSnapshotReader:
    versions: dict[tuple[str, str], StoredVersion] = field(default_factory=dict)
    locations: dict[tuple[str, str], list[StoredArtifactLocation]] = field(default_factory=dict)
    revocations: dict[tuple[str, str], Revocation] = field(default_factory=dict)
    # Digests the inventory holds beyond those `locations` link.
    inventoried: set[str] = field(default_factory=set)

    def read_version(self, subject_id: str, content_version: str) -> StoredVersion | None:
        return self.versions.get((subject_id, content_version))

    def read_artifact_locations(
        self, subject_id: str, content_version: str
    ) -> list[StoredArtifactLocation]:
        return list(self.locations.get((subject_id, content_version), []))

    def read_revocation(self, subject_id: str, content_version: str) -> Revocation | None:
        return self.revocations.get((subject_id, content_version))

    def artifact_inventoried(self, artifact_digest: str) -> bool:
        return (
            any(
                location.artifact_digest == artifact_digest
                for locations in self.locations.values()
                for location in locations
            )
            or artifact_digest in self.inventoried
        )

    def read_version_state(
        self, subject_id: str, content_version: str, *, revoked_under: Sequence[str] = ()
    ) -> VersionState | None:
        version = self.read_version(subject_id, content_version)
        if version is None:
            return None
        subjects = dict.fromkeys((subject_id, *revoked_under))
        revocation = next(
            (
                revoked
                for cv in (content_version, version.base_content_version)
                if cv is not None
                for subject in subjects
                if (revoked := self.read_revocation(subject, cv)) is not None
            ),
            None,
        )
        return VersionState(
            version=version,
            revocation=revocation,
            locations=self.read_artifact_locations(subject_id, content_version),
        )


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


@dataclass
class RecordingVersionWriter:
    """`inner`, whose writes this writer also records into `reader`. `make_app` and `fake_deps`
    wire it beside a fake reader.

    Every call answers as `inner` does, the null writer by default, so the serve path runs as
    under it. `reader` gains each written version, keeping one it already holds, and a working
    location stored at `now()` for each artifact a write links, so a door reading the version
    it just served (`datasets_api.retention`) finds it, as it does against the live store.
    """

    reader: FakeSnapshotReader
    inner: VersionRecordWriter = NULL_VERSION_RECORD_WRITER
    now: Clock = utc_now

    def _record(self, record: VersionRecord) -> None:
        self.reader.versions.setdefault(
            (record.subject_id, record.content_version), unlinked_state(record).version
        )
        for artifact in record.artifacts:
            self._link(record.subject_id, record.content_version, artifact)

    def _link(self, subject_id: str, content_version: str, artifact: ArtifactRecord) -> None:
        key = enriched_rendering_key(artifact.rendering, artifact.enrichment_digest)
        linked = self.reader.locations.setdefault((subject_id, content_version), [])
        linked[:] = [location for location in linked if location.rendering_key != key]
        stored_at = self.now()
        linked.append(
            _location(
                digest=artifact.artifact_digest,
                stored_at=stored_at,
                created_at=stored_at,
                rendering_key=key,
                rendering=artifact.rendering,
                media_type=artifact.media_type,
            )
        )

    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> bool:
        self._record(record)
        return self.inner.record_version(record, trace_id=trace_id)

    def record_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: ArtifactRecord,
        trace_id: str | None = None,
    ) -> str:
        self._link(subject_id, content_version, artifact)
        return self.inner.record_artifact(
            subject_id=subject_id,
            content_version=content_version,
            artifact=artifact,
            trace_id=trace_id,
        )

    def write_version(self, write: VersionWrite, *, trace_id: str | None = None) -> WrittenVersion:
        self._record(write.record)
        for artifact in write.rendered:
            self._link(write.record.subject_id, write.record.content_version, artifact)
        if write.collapsed is not None:
            self._record(write.collapsed)
        return self.inner.write_version(write, trace_id=trace_id)


def offline_version_pair(*, version_writer: object, snapshot_reader: object) -> dict[str, object]:
    """The `version_writer` and `snapshot_reader` of an offline app or `Deps`. An omitted reader
    is an empty `FakeSnapshotReader` and an omitted writer the null writer. A fake reader sees
    what the writer writes (`RecordingVersionWriter`), as the live reader sees the live
    writer's rows."""
    reader = snapshot_reader if snapshot_reader is not None else FakeSnapshotReader()
    writer = version_writer if version_writer is not None else NullVersionRecordWriter()
    if isinstance(reader, FakeSnapshotReader):
        writer = RecordingVersionWriter(reader=reader, inner=cast(VersionRecordWriter, writer))
    return {"version_writer": writer, "snapshot_reader": reader}


def version_body(content_version: str = "cv1") -> dict[str, object]:
    """A minimal stored dataset version body, projected onto facts by `facts_from_body`."""
    return {
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
    }


def _version(
    *,
    facts: VersionFacts | None = None,
    content_version: str = "cv1",
    base_content_version: str | None = None,
) -> StoredVersion:
    return StoredVersion(
        subject_id="spec_a",
        subject_kind="spec",
        content_version=content_version,
        schema_version=1,
        facts=facts if facts is not None else facts_from_body(version_body(content_version)),
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
    rendering = dict(rendering) if rendering is not None else {"format": "csv", "bundle": False}
    return StoredArtifactLocation(
        artifact_digest=digest,
        # The default key is the one the rendering is linked under for the packaged code index,
        # which the fake dependencies serve, so a snapshot read lists the location. Tests that
        # narrow by key, or hold more than one location of a rendering, pass a key.
        rendering_key=(
            rendering_key
            if rendering_key is not None
            else enriched_rendering_key(rendering, get_index().enrichment_digest)
        ),
        rendering=rendering,
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
        exception_exemplars=endpoint_exemplars((door,)),
        middleware=door.middleware,
    )
    return TestClient(create_app([surface]), raise_server_exceptions=False)
