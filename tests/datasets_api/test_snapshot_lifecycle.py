"""Tests snapshot-door startup, backpressure, and HTTP reads with a real `LiveSnapshotReader`
over the SQLite-backed version store.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from artifact_store import LiveArtifactInventory, StoredArtifact, Tier
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api import door_executor
from datasets_api.config import DatasetsConfigError
from datasets_api.doors import snapshot as snapshot_door
from datasets_api.doors.snapshot import build_snapshot_door
from datasets_api.revision import SentinelRevisionReader
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord, VersionRecord
from dc_sdmx import SdmxClient
from materialise.formats import FRAME1_MEDIA_TYPE
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._executor_doubles import BrokenSubmitExecutor
from ._frame1_doubles import (
    _BASE_URL,
    _body,
    _coverage_service,
    _frame1_row,
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
)
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

# --- Lifespan construction failure and the door's own backpressure gate ---


def test_construction_failure_returns_the_pool_borrow_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When settings construction fails after the pool opens, the door returns the connection."""
    open_calls = 0
    close_calls = 0

    def _counting_shared_pool() -> object:
        nonlocal open_calls
        open_calls += 1
        return object()

    def _counting_close_shared_pool() -> None:
        nonlocal close_calls
        close_calls += 1

    monkeypatch.setattr(snapshot_door, "shared_pool", _counting_shared_pool)
    monkeypatch.setattr(snapshot_door, "close_shared_pool", _counting_close_shared_pool)
    monkeypatch.setattr(snapshot_door, "LiveDatasetStore", lambda *, connections: object())

    door = build_snapshot_door()  # spec_reader, snapshot_reader, artifact_store all uninjected

    async def _enter() -> None:
        async with door.lifespan(None):
            pass  # pragma: no cover -- construction fails before this is reached

    with pytest.raises(DatasetsConfigError):
        asyncio.run(_enter())

    assert open_calls == 1
    assert close_calls == 1


@pytest.mark.parametrize(
    "uninjected_deps",
    [frozenset({"snapshot_reader"})],
    ids=lambda deps: "uninjected_" + "_".join(sorted(deps)),
)
def test_build_refuses_to_start_with_only_some_dependencies_injected(
    uninjected_deps: frozenset[str],
) -> None:
    """Injecting `spec_reader` without `snapshot_reader` fails during door construction."""
    with pytest.raises(DatasetsConfigError) as excinfo:
        build_snapshot_door(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=None if "snapshot_reader" in uninjected_deps else FakeSnapshotReader(),
            artifact_store=FakeArtifactStore(),
        )

    for dep in uninjected_deps:
        assert dep in str(excinfo.value)


def test_backpressure_refuses_503_retryable_when_executor_is_already_at_capacity() -> None:
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"backpressure-data")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    blocking_spec_reader = BlockingFakeSpecReader(spec_ids=["spec_a"])
    door = build_snapshot_door(
        spec_reader=blocking_spec_reader,
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
        pool_size=1,
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
        body = resp.json()
        assert body["code"] == "service_unavailable"
        assert body["retryable"] is True

        blocking_spec_reader.release.set()
        first_thread.join(timeout=5.0)
        assert first_response and first_response[0].status_code == 200


def test_broken_thread_pool_surfaces_as_internal_error_not_backpressure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(door_executor, "ThreadPoolExecutor", BrokenSubmitExecutor)
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"broken-pool-data")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    door = build_snapshot_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
        pool_size=1,
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


def test_snapshot_door_serves_a_manifest_written_by_the_real_writer(version_store):
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
            content_version="cv1",
            schema_version=1,
            immutable_body=_version().immutable_body,
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
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        artifact_store=artifact_store,
        now=lambda: _NOW,
    )
    client = make_client_for_door(door)

    resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["content_version"] == "cv1"
    assert len(body["artifacts"]) == 1
    assert body["artifacts"][0]["artifact_digest"] == stored.digest


class _InventoryRecordingArtifactStore(FakeArtifactStore):
    """Records fake uploads and copies in the inventory tables that `LiveSnapshotReader` joins,
    as the GCS store records each destination it writes."""

    def __init__(self, *, inventory: LiveArtifactInventory) -> None:
        super().__init__()
        self._inventory = inventory

    def _record(self, stored: StoredArtifact) -> StoredArtifact:
        self._inventory.record(
            digest=stored.digest,
            tier=stored.tier,
            bucket=stored.bucket,
            object_name=stored.object_name,
            size_bytes=stored.size_bytes,
            uploaded=not stored.deduplicated,
        )
        return stored

    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        return self._record(super().put(data, tier=tier))

    def copy(self, digest: str, *, from_tier: Tier, to_tier: Tier) -> StoredArtifact:
        return self._record(super().copy(digest, from_tier=from_tier, to_tier=to_tier))


def test_a_mint_through_the_real_mint_door_serves_a_snapshot_excluding_frame1(
    version_store,
):
    """A mint and snapshot GET share real version storage. The mint records a frame1 row, which
    the snapshot read must exclude from listable artifacts.
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
        revision_reader=SentinelRevisionReader(),
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
    _frame1_row(rows)  # the record-only frame1 row is present in the database
    _statistics_row(rows)

    snapshot_resp = client.get(f"/api/datasets/{spec_id}/{content_version}")

    assert snapshot_resp.status_code == 200, snapshot_resp.text
    snapshot_body = snapshot_resp.json()
    # A plain snapshot GET (no narrowing) lists every currently servable
    # listable rendering: the requested csv, plus the statistics artifact.
    # Frame1 stays internal.
    assert len(snapshot_body["artifacts"]) == 2
    requested = next(
        a for a in snapshot_body["artifacts"] if a["rendering"] != {"statistics": "series"}
    )
    assert requested["rendering"]["format"] == "csv"
    assert all(a["media_type"] != FRAME1_MEDIA_TYPE for a in snapshot_body["artifacts"])
