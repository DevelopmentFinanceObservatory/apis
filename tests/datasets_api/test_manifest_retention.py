"""`Manifest.retention`: every Manifest door reports how long the version it serves can still be
fetched, from the frame its files render from (`datasets_api.retention`), and no stored body
holds it."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from artifact_store import LiveArtifactInventory, Tier
from dataset_store.snapshot import WORKING_TIER_RETENTION, LiveSnapshotReader
from datasets_api import manifest as manifest_module
from datasets_api.doors.pin import pin_artifact
from datasets_api.manifest import version_document
from datasets_api.manifest_models import RetentionModel
from datasets_api.reference_store import PIN_REFERENCE_KIND
from datasets_api.retention import RetentionUnreadable, version_retention
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._frame_doubles import _BASE_URL, _body, _coverage_service, _graph, _sdmx_transport
from ._snapshot_doubles import (
    _NOW,
    FakeSnapshotReader,
    FakeSpecReader,
    StoreBackedSnapshotReader,
    _location,
    _store_and_locate,
    _version,
)
from .conftest import FakeDatasetStore
from .test_meta_doors import (
    _PARQUET_BUNDLE,
    _SITE_SETTINGS,
    _FixedOperationStore,
    _operation,
    _stored_rendering,
)
from .test_pin_door import _PRINCIPAL, FakeArtifactReferenceWriter
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

_FRAME = {"format": "frame"}
_PARQUET = "application/vnd.apache.parquet"
_LONG_AGO = _NOW - WORKING_TIER_RETENTION - timedelta(days=5)


def _frame(*, digest: str, tier: str = "working", stored_at: datetime = _NOW):
    return _location(
        digest=digest, tier=tier, stored_at=stored_at, rendering=_FRAME, media_type=_PARQUET
    )


def _held_wire(stored_at: datetime) -> dict:
    """`_held` as the wire writes the instant (`Z` for UTC)."""
    held = stored_at + WORKING_TIER_RETENTION
    return {"pinned": False, "held_until": held.isoformat().replace("+00:00", "Z")}


# --- The frame rule ---


@pytest.mark.parametrize(
    ("own", "expected"),
    [
        pytest.param(
            [_frame(digest="sha256:frame"), _location(digest="sha256:csv", stored_at=_LONG_AGO)],
            {"pinned": False, "held_until": "2026-07-01T00:00:00Z"},
            id="working-frame-held-30-days-from-stored-at",
        ),
        pytest.param(
            [
                _frame(digest="sha256:frame", tier="working", stored_at=_LONG_AGO),
                _frame(digest="sha256:frame", tier="pinned", stored_at=_LONG_AGO),
            ],
            {"pinned": True},
            id="a-pinned-copy-wins-and-omits-held-until",
        ),
        pytest.param(
            [_frame(digest="sha256:frame", stored_at=_LONG_AGO)],
            {"pinned": False, "held_until": "2026-05-27T00:00:00Z"},
            id="an-aged-frame-reports-a-past-instant",
        ),
        pytest.param(
            [
                _location(digest="sha256:old", stored_at=_LONG_AGO),
                _location(digest="sha256:new", stored_at=_NOW),
                _location(
                    digest="sha256:stats",
                    stored_at=_NOW + timedelta(days=9),
                    rendering={"statistics": "series"},
                    media_type=_PARQUET,
                ),
            ],
            {"pinned": False, "held_until": "2026-07-01T00:00:00Z"},
            id="no-frame-reports-the-latest-data-artifact",
        ),
        pytest.param(
            [
                _location(digest="sha256:working", stored_at=_NOW),
                _location(digest="sha256:pinned", tier="pinned", stored_at=_LONG_AGO),
            ],
            {"pinned": True},
            id="no-frame-and-a-pinned-data-artifact",
        ),
    ],
)
def test_a_version_reports_the_retention_of_the_frame_its_files_render_from(own, expected):
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()}, locations={("spec_a", "cv1"): own}
    )

    retention = version_retention(reader, subject_id="spec_a", content_version="cv1", now=_NOW)

    assert retention.model_dump(mode="json", exclude_none=True) == expected


def test_a_derived_version_reports_its_base_frame_and_ignores_its_own():
    reader = FakeSnapshotReader(
        versions={
            ("spec_a", "cv1"): _version(),
            ("spec_a", "cv2"): _version(content_version="cv2", base_content_version="cv1"),
        },
        locations={
            ("spec_a", "cv1"): [_frame(digest="sha256:base", tier="pinned", stored_at=_LONG_AGO)],
            ("spec_a", "cv2"): [_frame(digest="sha256:own", stored_at=_NOW)],
        },
    )

    retention = version_retention(reader, subject_id="spec_a", content_version="cv2", now=_NOW)

    assert retention.model_dump(mode="json", exclude_none=True) == {"pinned": True}


@pytest.mark.parametrize(
    "locations",
    [
        {},
        {
            ("spec_a", "cv1"): [
                _location(
                    digest="sha256:stats", media_type=_PARQUET, rendering={"statistics": "series"}
                )
            ]
        },
    ],
    ids=["no-version-record", "only-statistics"],
)
def test_a_version_with_no_record_or_nothing_to_fetch_refuses_loudly(locations):
    versions = {("spec_a", "cv1"): _version()} if locations else {}
    reader = FakeSnapshotReader(versions=versions, locations=locations)

    with pytest.raises(RetentionUnreadable):
        version_retention(reader, subject_id="spec_a", content_version="cv1", now=_NOW)


# --- The four doors ---


def test_the_mint_definition_and_snapshot_doors_report_the_frame_retention_and_store_none(
    version_store,
):
    """The mint rebuilds, the first GET hits the head, and a GET for another rendering rebuilds
    on the work executor; each reports the canonical frame's 30 days, as does the snapshot.
    The stored body and the version document carry no retention."""
    artifact_store = _InventoryRecordingArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )
    reader = LiveSnapshotReader(store=version_store)
    dataset_store = FakeDatasetStore()
    client = make_client(
        registry=LiveSpecRegistry(store=dataset_store),
        spec_reader=LiveSpecReader(store=dataset_store),
        graph=_graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        artifact_store=artifact_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        head_writer=LiveHeadStore(store=version_store),
        snapshot_reader=reader,
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )

    minted = client.post("/api/datasets", json=_body(format="csv", bundle=False))
    assert minted.status_code == 200, minted.text
    spec_id, content_version = (
        minted.json()["definition"]["spec_id"],
        minted.json()["content_version"],
    )
    head_hit = client.get(f"/api/datasets/{spec_id}?format=csv&bundle=false")
    rebuilt = client.get(f"/api/datasets/{spec_id}?format=parquet&bundle=false")
    snapshot = client.get(f"/api/datasets/{spec_id}/{content_version}")

    frame = next(
        location
        for location in reader.read_artifact_locations(spec_id, content_version)
        if location.rendering == _FRAME
    )
    assert head_hit.json()["trace"]["served_from"] == "head"
    for resp in (minted, head_hit, rebuilt, snapshot):
        assert resp.status_code == 200, resp.text
        assert resp.json()["content_version"] == content_version
        assert resp.json()["retention"] == _held_wire(frame.stored_at)
    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT immutable_body FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (spec_id, content_version),
        )
    assert row is not None
    assert "retention" not in str(row["immutable_body"])
    stored = reader.read_version(spec_id, content_version)
    assert stored is not None
    assert "retention" not in version_document(stored)


def test_a_done_operation_poll_reports_retention():
    """The version links no frame here, so its pinned rendering decides."""
    artifact_store = FakeArtifactStore()
    rendering = _stored_rendering(artifact_store, _PARQUET_BUNDLE, data=b"a,b\n1,2\n")
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        settings=_SITE_SETTINGS,
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=FakeSnapshotReader(
            versions={("spec_a", "cv1"): _version()},
            locations={("spec_a", "cv1"): [rendering]},
        ),
        artifact_store=artifact_store,
        reference_writer=FakeArtifactReferenceWriter(),
        operation_store=_FixedOperationStore(
            {"op_1": _operation(state="done", subject_id="spec_a", content_version="cv1")}
        ),
    )

    resp = client.get("/api/datasets/operations/op_1")

    assert resp.status_code == 200, resp.text
    assert resp.json()["manifest"]["retention"] == {"pinned": True}


# --- Pinning a derived version ---


def _derived_world(*, parquet_tier: Tier = Tier.working):
    """`cv2` is a derived version of `cv1`. Each links its own working canonical frame, stored
    at different instants, and `cv2` links a parquet rendering."""
    store = FakeArtifactStore()
    base_frame = _store_and_locate(
        store, b"PAR1-base-frame", rendering=_FRAME, media_type=_PARQUET, stored_at=_NOW
    )
    own_frame = _store_and_locate(
        store,
        b"PAR1-own-frame",
        rendering=_FRAME,
        media_type=_PARQUET,
        stored_at=_NOW + timedelta(days=3),
    )
    parquet = _store_and_locate(
        store,
        b"PAR1-derived-rendering",
        tier=parquet_tier,
        rendering={"format": "parquet", "bundle": False},
        media_type=_PARQUET,
    )
    reader = StoreBackedSnapshotReader(
        versions={
            ("spec_a", "cv1"): _version(),
            ("spec_a", "cv2"): _version(content_version="cv2", base_content_version="cv1"),
        },
        locations={("spec_a", "cv1"): [base_frame], ("spec_a", "cv2"): [parquet, own_frame]},
        store=store,
    )
    return store, reader, base_frame, parquet


def _pin(store, reader, writer, digest: str, *, daily_max: int = 500) -> datetime:
    return pin_artifact(
        id="spec_a",
        content_version="cv2",
        artifact_digest=digest,
        principal=_PRINCIPAL,
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=writer,
        daily_max=daily_max,
        now=lambda: _NOW,
    )


def test_pinning_a_derived_version_promotes_its_base_frame_and_reports_it_pinned():
    store, reader, base_frame, parquet = _derived_world()
    writer = FakeArtifactReferenceWriter()
    before = version_retention(reader, subject_id="spec_a", content_version="cv2", now=_NOW)
    assert before.model_dump(mode="json") == {
        "pinned": False,
        "held_until": (base_frame.stored_at + WORKING_TIER_RETENTION)
        .isoformat()
        .replace("+00:00", "Z"),
    }

    _pin(store, reader, writer, parquet.artifact_digest)

    assert (base_frame.artifact_digest, Tier.pinned.value) in store._objects
    base_key = (_PRINCIPAL, base_frame.artifact_digest, "spec_a", "cv1", PIN_REFERENCE_KIND)
    assert writer.records[base_key] == _NOW
    after = version_retention(reader, subject_id="spec_a", content_version="cv2", now=_NOW)
    assert after.model_dump(mode="json", exclude_none=True) == {"pinned": True}


def test_the_base_frame_copy_skips_an_exhausted_budget_and_counts_toward_later_calls():
    """The named rendering is already pinned, so this call waits on no budget, and the frames
    are copied though the trailing count sits at the budget. The base frame's copy holds a
    `pin` reference, which is what the trailing count reads, so it counts toward later calls."""
    store, reader, base_frame, parquet = _derived_world(parquet_tier=Tier.pinned)
    writer = FakeArtifactReferenceWriter(pinned_copies={"sha256:seed": _NOW})
    writer.records[(_PRINCIPAL, "sha256:seed", "spec_seed", "cv1", PIN_REFERENCE_KIND)] = _NOW
    assert writer.count_pinned_copies_since(reference_kind=PIN_REFERENCE_KIND, since=_NOW) == 1

    _pin(store, reader, writer, parquet.artifact_digest, daily_max=1)

    assert (base_frame.artifact_digest, Tier.pinned.value) in store._objects
    assert (_PRINCIPAL, base_frame.artifact_digest, "spec_a", "cv1", PIN_REFERENCE_KIND) in (
        writer.records
    )


def test_the_version_document_leaves_out_a_retention_its_manifest_holds(
    monkeypatch: pytest.MonkeyPatch,
):
    """Stored-body readers are `extra="forbid"`, so the document drops the key even from a
    manifest that holds it."""
    stored = _version()
    assembled = manifest_module.stored_manifest(stored)
    monkeypatch.setattr(
        manifest_module,
        "stored_manifest",
        lambda version: assembled.model_copy(
            update={"retention": RetentionModel(pinned=False, held_until=_NOW)}
        ),
    )

    assert "retention" not in version_document(stored)
