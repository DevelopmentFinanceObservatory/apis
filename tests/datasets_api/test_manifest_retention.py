"""`Manifest.retention`: every Manifest door reports how long the version it serves can still be
fetched, from the frames that serve it (`datasets_api.retention`), and no stored body holds it.
The pin door promotes the first of those frames whose bytes survive."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from artifact_store import LiveArtifactInventory, StoredArtifact, Tier
from dataset_store.snapshot import WORKING_TIER_RETENTION, LiveSnapshotReader, StoredVersion
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
    """The `retention` of a working frame stored at `stored_at`, with the instant as the wire
    writes it (`Z` for UTC)."""
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


_LEFT_OUT = {"party_totals_left_out": [{"role": "recipient", "entities": ["X"]}]}
# A party-totals cut whose file records a groupby, minted before `frame_summed`: the record
# cannot say what its own frame holds, so its base frame alone serves it.
_CUT_AND_SUMMED = {
    "format": "parquet",
    "bundle": False,
    "groupby": {"by": ["recipient"], "op": "sum"},
    "without_party_totals": True,
}


def _derived(*, trace: dict | None = None) -> StoredVersion:
    version = _version(content_version="cv2", base_content_version="cv1")
    if trace is None:
        return version
    return replace(version, facts=replace(version.facts, trace=trace))


@pytest.mark.parametrize(
    ("derived", "own", "base", "expected"),
    [
        pytest.param(
            _derived(),
            [_frame(digest="sha256:own", tier="pinned", stored_at=_NOW)],
            [_frame(digest="sha256:base", stored_at=_LONG_AGO)],
            {"pinned": True},
            id="own-frame-pinned-base-aged",
        ),
        pytest.param(
            _derived(),
            [_frame(digest="sha256:own", stored_at=_NOW)],
            [_frame(digest="sha256:base", tier="pinned", stored_at=_LONG_AGO)],
            {"pinned": True},
            id="base-frame-pinned-own-working",
        ),
        pytest.param(
            _derived(),
            [_frame(digest="sha256:own", stored_at=_NOW)],
            [_frame(digest="sha256:base", stored_at=_LONG_AGO)],
            _held_wire(_NOW),
            id="the-latest-held-until-among-serving-frames",
        ),
        pytest.param(
            _derived(),
            [_frame(digest="sha256:own", stored_at=_NOW)],
            [],
            _held_wire(_NOW),
            id="own-frame-alone",
        ),
        pytest.param(
            _derived(trace=_LEFT_OUT),
            [
                _frame(digest="sha256:own", tier="pinned", stored_at=_NOW),
                _location(digest="sha256:file", rendering=_CUT_AND_SUMMED, media_type=_PARQUET),
            ],
            [_frame(digest="sha256:base", stored_at=_LONG_AGO)],
            _held_wire(_LONG_AGO),
            id="base-only-version-ignores-its-own-frame",
        ),
    ],
)
def test_a_derived_version_reports_the_frames_that_serve_it(derived, own, base, expected):
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version(), ("spec_a", "cv2"): derived},
        locations={("spec_a", "cv1"): base, ("spec_a", "cv2"): own},
    )

    retention = version_retention(reader, subject_id="spec_a", content_version="cv2", now=_NOW)

    assert retention.model_dump(mode="json", exclude_none=True) == expected


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
    on the work executor. Each reports the canonical frame's 30 days, as does the snapshot.
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


class _CopyRecordingArtifactStore(FakeArtifactStore):
    """Records each digest a copy promotes, in call order."""

    def __init__(self) -> None:
        super().__init__()
        self.copied: list[str] = []

    def copy(self, digest: str, *, from_tier: Tier, to_tier: Tier) -> StoredArtifact:
        stored = super().copy(digest, from_tier=from_tier, to_tier=to_tier)
        self.copied.append(digest)
        return stored


def _derived_world(
    *,
    parquet_tier: Tier = Tier.working,
    own_tier: Tier = Tier.working,
    base_stored_at: datetime = _NOW,
    trace: dict | None = None,
    parquet_rendering: Mapping[str, object] | None = None,
):
    """`cv2` is a derived version of `cv1`, served by its own frame then its base frame unless
    `trace` and `parquet_rendering` say otherwise. Each links its own canonical frame, stored at
    different instants, the base one at `base_stored_at`, and `cv2` links a parquet rendering."""
    store = _CopyRecordingArtifactStore()
    base_frame = _store_and_locate(
        store, b"PAR1-base-frame", rendering=_FRAME, media_type=_PARQUET, stored_at=base_stored_at
    )
    own_frame = _store_and_locate(
        store,
        b"PAR1-own-frame",
        tier=own_tier,
        rendering=_FRAME,
        media_type=_PARQUET,
        stored_at=_NOW + timedelta(days=3),
    )
    parquet = _store_and_locate(
        store,
        b"PAR1-derived-rendering",
        tier=parquet_tier,
        rendering=parquet_rendering or {"format": "parquet", "bundle": False},
        media_type=_PARQUET,
    )
    reader = StoreBackedSnapshotReader(
        versions={("spec_a", "cv1"): _version(), ("spec_a", "cv2"): _derived(trace=trace)},
        locations={("spec_a", "cv1"): [base_frame], ("spec_a", "cv2"): [parquet, own_frame]},
        store=store,
    )
    return store, reader, base_frame, own_frame, parquet


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


def _pin_key(frame, content_version: str) -> tuple[str, str, str, str, str]:
    return (_PRINCIPAL, frame.artifact_digest, "spec_a", content_version, PIN_REFERENCE_KIND)


@pytest.mark.parametrize(
    "base_stored_at",
    [
        pytest.param(_NOW, id="servable-base-frame"),
        pytest.param(_LONG_AGO, id="aged-base-frame-whose-bytes-are-gone"),
    ],
)
def test_pinning_a_derived_version_promotes_its_own_frame_and_stops(base_stored_at: datetime):
    """The own frame serves the version, so the base frame is neither copied nor referenced,
    and the version reports pinned whether or not the base bytes survive."""
    store, reader, base_frame, own_frame, parquet = _derived_world(base_stored_at=base_stored_at)
    if base_stored_at == _LONG_AGO:
        del store._objects[(base_frame.artifact_digest, Tier.working.value)]
    writer = FakeArtifactReferenceWriter()

    _pin(store, reader, writer, parquet.artifact_digest)

    assert store.copied == [parquet.artifact_digest, own_frame.artifact_digest]
    assert writer.records[_pin_key(own_frame, "cv2")] == _NOW
    assert _pin_key(base_frame, "cv1") not in writer.records
    after = version_retention(reader, subject_id="spec_a", content_version="cv2", now=_NOW)
    assert after.model_dump(mode="json", exclude_none=True) == {"pinned": True}


def test_a_pin_of_a_derived_version_whose_own_frame_is_pinned_copies_no_frame():
    """A pinned own frame already serves the version: the pin records a reference to it and
    leaves the base frame alone."""
    store, reader, base_frame, own_frame, parquet = _derived_world(own_tier=Tier.pinned)
    writer = FakeArtifactReferenceWriter()

    _pin(store, reader, writer, parquet.artifact_digest)

    assert store.copied == [parquet.artifact_digest]
    assert writer.records[_pin_key(own_frame, "cv2")] == _NOW
    assert _pin_key(base_frame, "cv1") not in writer.records


@pytest.mark.parametrize(
    ("trace", "parquet_rendering"),
    [
        pytest.param(None, None, id="own-frame-bytes-gone"),
        pytest.param(_LEFT_OUT, _CUT_AND_SUMMED, id="base-only-version"),
    ],
)
def test_a_pin_promotes_the_base_frame_when_the_own_frame_cannot_serve(trace, parquet_rendering):
    """An own frame whose bytes are gone is skipped, and a version its own frame cannot serve
    never promotes it. Either way the base frame is promoted and the version reports pinned."""
    store, reader, base_frame, own_frame, parquet = _derived_world(
        trace=trace, parquet_rendering=parquet_rendering
    )
    if trace is None:
        del store._objects[(own_frame.artifact_digest, Tier.working.value)]
    writer = FakeArtifactReferenceWriter()

    _pin(store, reader, writer, parquet.artifact_digest)

    assert store.copied == [parquet.artifact_digest, base_frame.artifact_digest]
    assert writer.records[_pin_key(base_frame, "cv1")] == _NOW
    assert _pin_key(own_frame, "cv2") not in writer.records
    after = version_retention(reader, subject_id="spec_a", content_version="cv2", now=_NOW)
    assert after.model_dump(mode="json", exclude_none=True) == {"pinned": True}


def test_a_pin_whose_serving_frames_bytes_are_gone_pins_the_named_file_alone():
    """Neither frame's bytes survive, so nothing is left to pin. The named parquet still pins
    and the version keeps the latest `held_until` of its frames."""
    store, reader, base_frame, own_frame, parquet = _derived_world(base_stored_at=_LONG_AGO)
    del store._objects[(own_frame.artifact_digest, Tier.working.value)]
    del store._objects[(base_frame.artifact_digest, Tier.working.value)]
    writer = FakeArtifactReferenceWriter()

    pinned_at = _pin(store, reader, writer, parquet.artifact_digest)

    assert pinned_at == _NOW
    assert store.copied == [parquet.artifact_digest]
    assert set(writer.records) == {_pin_key(parquet, "cv2")}
    after = version_retention(reader, subject_id="spec_a", content_version="cv2", now=_NOW)
    assert after.model_dump(mode="json") == _held_wire(own_frame.stored_at)


def test_the_frame_copy_skips_an_exhausted_budget_and_counts_toward_later_calls():
    """The named rendering is already pinned, so this call waits on no budget, and the frame is
    copied though the trailing count sits at the budget. The frame's copy holds a `pin`
    reference, which is what the trailing count reads, so it counts toward later calls."""
    store, reader, _, own_frame, parquet = _derived_world(parquet_tier=Tier.pinned)
    writer = FakeArtifactReferenceWriter(pinned_copies={"sha256:seed": _NOW})
    writer.records[(_PRINCIPAL, "sha256:seed", "spec_seed", "cv1", PIN_REFERENCE_KIND)] = _NOW
    assert writer.count_pinned_copies_since(reference_kind=PIN_REFERENCE_KIND, since=_NOW) == 1

    _pin(store, reader, writer, parquet.artifact_digest, daily_max=1)

    assert store.copied == [own_frame.artifact_digest]
    assert _pin_key(own_frame, "cv2") in writer.records


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
