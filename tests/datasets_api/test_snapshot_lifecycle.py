"""Tests snapshot-door startup, backpressure, and HTTP reads with a real `LiveSnapshotReader`
over the SQLite-backed version store.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta

import pytest

from artifact_store import (
    ArtifactSink,
    ArtifactStore,
    LiveArtifactInventory,
    StoredArtifact,
    Tier,
)
from artifact_store.inventory import InventoryBatch, InventoryEntry
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api import door_executor
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.snapshot import build_snapshot_door
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord, VersionRecord
from dc_sdmx import SdmxClient
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._deps_doubles import fake_deps_cell
from ._executor_doubles import BrokenSubmitExecutor
from ._frame_doubles import (
    _BASE_URL,
    _body,
    _canonical_frame_row,
    _coverage_service,
    _graph,
    _sdmx_transport,
    _statistics_row,
    _version_artifact_rows,
)
from ._snapshot_doubles import (
    _NOW,
    BlockingFakeSpecReader,
    FakeSnapshotReader,
    FakeSpecReader,
    _store_and_locate,
    _version,
    make_client_for_door,
    version_body,
)
from ._version_facts import facts_from_body
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

# --- Backpressure on the read executor ---


def test_backpressure_refuses_503_retryable_when_executor_is_already_at_capacity() -> None:
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"backpressure-data")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    blocking_spec_reader = BlockingFakeSpecReader(spec_ids=["spec_a"])
    door = build_snapshot_door(
        deps=fake_deps_cell(
            settings=DatasetsBuildSettings.from_env().model_copy(
                update={"read_max_concurrency": 1}
            ),
            spec_reader=blocking_spec_reader,
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
        )
    )
    client = make_client_for_door(door)

    first_response: list = []
    # The client shares one portal loop and leaves the door lifespan inactive.
    with client:
        first_thread = threading.Thread(
            target=lambda: first_response.append(client.get("/api/datasets/spec_a/cv1"))
        )
        first_thread.start()
        # The first request holds the sole executor slot and backpressure permit.
        assert blocking_spec_reader.entered.wait(timeout=5.0), (
            "first request never reached the blocking spec reader"
        )

        resp = client.get("/api/datasets/spec_a/cv1")

        assert resp.status_code == 503
        assert resp.headers["retry-after"] == "1"
        body = resp.json()
        assert body["code"] == "service_unavailable"
        assert body["retryable"] is True

        blocking_spec_reader.release.set()
        first_thread.join(timeout=5.0)
        assert first_response and first_response[0].status_code == 200


def test_broken_thread_pool_surfaces_as_internal_error_not_backpressure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(door_executor, "ContextCarryingExecutor", BrokenSubmitExecutor)
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"broken-pool-data")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    door = build_snapshot_door(
        deps=fake_deps_cell(
            settings=DatasetsBuildSettings.from_env().model_copy(
                update={"read_max_concurrency": 1}
            ),
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
        )
    )
    client = make_client_for_door(door)

    with caplog.at_level(logging.ERROR):
        resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert body["code"] == "internal_error"
    assert body["retryable"] is False
    assert "permanently broken" in caplog.text


# --- End-to-end: the real door, a real LiveSnapshotReader, real writer ---


# A version minted before `content_version` hashed the frame alone: a `cv1`-family token, a trace
# carrying the lineage that token hashed, and the `revision_unavailable` warning manifests carried.
_PRE_FRAME_HASH_TOKEN = "cv_ea9dc1d4c4be9cf2ad62a810aa2f4db7"
_PRE_FRAME_HASH_BODY = {
    **version_body(_PRE_FRAME_HASH_TOKEN),
    "trace": {
        "graph_revision": "unavailable:dfo-dcp-revision-endpoint-absent",
        "source_revisions": [],
        "facets": [],
        "mint_facets": [{"axis": "provenance", "ref": "OECD_DAC2A", "basis": "bound"}],
        "normalisations": [],
        "frame_grammar_version": "1.0.0",
    },
    "warnings": [{"code": "revision_unavailable", "severity": "info", "message": "m"}],
}


@pytest.mark.parametrize(
    ("content_version", "body"),
    [("cv1", version_body()), (_PRE_FRAME_HASH_TOKEN, _PRE_FRAME_HASH_BODY)],
    ids=["current", "minted-before-frame-hash"],
)
def test_snapshot_door_serves_a_manifest_written_by_the_real_writer(
    version_store, content_version, body
):
    """A version minted before tokens hashed the frame alone still resolves by its old token,
    with its stored trace and warnings."""
    artifact_store = FakeArtifactStore()
    stored = artifact_store.put(b"hello", tier=Tier.working)
    # Seed inventory tables because `FakeArtifactStore.put()` leaves them to the test.
    with version_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.artifact_inventory (artifact_digest, size_bytes, created_at) "
            "VALUES (%s, %s, %s)",
            (stored.digest, stored.size_bytes, _NOW),
        )
        tx.execute(
            "INSERT INTO datasets.artifact_location "
            "(artifact_digest, tier, bucket, object_name, stored_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (stored.digest, "working", stored.bucket, stored.object_name, _NOW),
        )
    writer = LiveVersionRecordWriter(store=version_store, now=lambda: _NOW)
    writer.record_version(
        VersionRecord(
            subject_id="spec_a",
            subject_kind="spec",
            content_version=content_version,
            schema_version=1,
            facts=facts_from_body(body),
            artifacts=(
                ArtifactRecord(
                    artifact_digest=stored.digest,
                    rendering={"format": "csv"},
                    media_type="text/csv",
                ),
            ),
        )
    )

    door = build_snapshot_door(
        deps=fake_deps_cell(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=LiveSnapshotReader(store=version_store),
            artifact_store=artifact_store,
            now=lambda: _NOW,
        )
    )
    client = make_client_for_door(door)

    resp = client.get(f"/api/datasets/spec_a/{content_version}")

    assert resp.status_code == 200, resp.text
    served = resp.json()
    assert served["content_version"] == content_version
    assert served["trace"] == body["trace"]
    assert served["warnings"] == body["warnings"]
    assert len(served["artifacts"]) == 1
    assert served["artifacts"][0]["artifact_digest"] == stored.digest


# The batch a `deferred` view of `_InventoryRecordingArtifactStore` is writing for, if any.
_deferred_batch: contextvars.ContextVar[InventoryBatch | None] = contextvars.ContextVar(
    "deferred_batch", default=None
)


class _InventoryRecordingArtifactStore(FakeArtifactStore):
    """Records fake uploads and copies in the inventory tables that `LiveSnapshotReader` joins,
    as the GCS store records each destination it writes. A `deferred` view adds those rows to
    its batch, as `LiveArtifactStore.deferred` does, and shares this store's objects and
    counters."""

    def __init__(self, *, inventory: LiveArtifactInventory) -> None:
        super().__init__()
        self._inventory = inventory

    def _record(self, stored: StoredArtifact) -> StoredArtifact:
        entry = InventoryEntry(
            digest=stored.digest,
            tier=stored.tier,
            bucket=stored.bucket,
            object_name=stored.object_name,
            size_bytes=stored.size_bytes,
            uploaded=not stored.deduplicated,
        )
        batch = _deferred_batch.get()
        if batch is not None:
            batch.add(entry)
        else:
            self._inventory.write(entry)
        return stored

    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        return self._record(super().put(data, tier=tier))

    def copy(self, digest: str, *, from_tier: Tier, to_tier: Tier) -> StoredArtifact:
        return self._record(super().copy(digest, from_tier=from_tier, to_tier=to_tier))

    def deferred(self, batch: InventoryBatch) -> ArtifactStore:
        return _DeferredRecordingView(self, batch=batch)


class _DeferredRecordingView:
    """`store`'s `ArtifactStore` methods, with `put` and `open_put` run under `batch` so
    `_record` adds to it."""

    def __init__(self, store: _InventoryRecordingArtifactStore, *, batch: InventoryBatch) -> None:
        self._store = store
        self._batch = batch

    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        token = _deferred_batch.set(self._batch)
        try:
            return self._store.put(data, tier=tier)
        finally:
            _deferred_batch.reset(token)

    @contextmanager
    def open_put(self, *, tier: Tier) -> Iterator[ArtifactSink]:
        token = _deferred_batch.set(self._batch)
        try:
            with self._store.open_put(tier=tier) as sink:
                yield sink
        finally:
            _deferred_batch.reset(token)

    def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
        return self._store.get(artifact_digest=artifact_digest, tier=tier)

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
        return self._store.signed_location_url(
            digest,
            tier=tier,
            bucket=bucket,
            object_name=object_name,
            ttl=ttl,
            response_type=response_type,
            response_disposition=response_disposition,
        )

    def copy(self, digest: str, *, from_tier: Tier, to_tier: Tier) -> StoredArtifact:
        return self._store.copy(digest, from_tier=from_tier, to_tier=to_tier)

    def list(self, *, tier: Tier) -> frozenset[str]:
        return self._store.list(tier=tier)

    def deferred(self, batch: InventoryBatch) -> ArtifactStore:
        return _DeferredRecordingView(self._store, batch=batch)


def test_a_mint_through_the_real_mint_door_serves_a_snapshot_listing_the_canonical_frame(
    version_store,
):
    """A mint and snapshot GET share real version storage. The mint records a canonical frame
    row, which the snapshot lists under `{"format": "frame"}` and its stable download link.
    """
    dataset_store = FakeDatasetStore()
    artifact_store = _InventoryRecordingArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )
    client = make_client(
        registry=LiveSpecRegistry(store=dataset_store),
        spec_reader=LiveSpecReader(store=dataset_store),
        graph=_graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        artifact_store=artifact_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )

    mint_resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))
    assert mint_resp.status_code == 200, mint_resp.text
    manifest = mint_resp.json()
    spec_id = manifest["definition"]["spec_id"]
    content_version = manifest["content_version"]

    rows = _version_artifact_rows(
        version_store, subject_id=spec_id, content_version=content_version
    )
    assert len(rows) == 3
    frame_row = _canonical_frame_row(rows)
    _statistics_row(rows)

    snapshot_resp = client.get(f"/api/datasets/{spec_id}/{content_version}")

    assert snapshot_resp.status_code == 200, snapshot_resp.text
    snapshot_body = snapshot_resp.json()
    # A plain snapshot GET (no narrowing) lists every currently servable
    # listable rendering: the requested csv, then the canonical frame and the
    # statistics artifact.
    assert [a["rendering"].get("format") for a in snapshot_body["artifacts"]] == [
        "csv",
        "frame",
        None,
    ]
    frame = snapshot_body["artifacts"][1]
    assert frame["artifact_digest"] == frame_row["artifact_digest"]
    assert frame["media_type"] == "application/vnd.apache.parquet"
    assert frame["url"] == (
        f"/api/datasets/{spec_id}/{content_version}/files/{frame_row['artifact_digest']}"
    )
