"""Download resolution (I16) and manifests that sign nothing (I14's signing half).

A manifest's artifact `url` is the stable link `/api/datasets/{id}/{content_version}/files/
{digest}`. The link checks the version at the click: a revoked version refuses
`snapshot_revoked` and signs nothing, a servable artifact answers 302 to a URL signed for at
most five minutes, and a working artifact past retention renews the stored frame when it aged
out too, re-renders from it, renews under a new object name, and then redirects.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from artifact_store import LiveArtifactInventory, LiveArtifactStore, StoredArtifact, Tier
from dataset_store.snapshot import (
    ArtifactNotFound,
    LiveSnapshotReader,
    Revocation,
    read_servable_version,
    working_tier_expired,
)
from datasets_api import current as current_module
from datasets_api import deps as deps_module
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors import files as files_module
from datasets_api.doors.files import build_files_door
from datasets_api.render import (
    artifact_rendering,
    incomplete_rendering,
    rendering_options,
    rendition_token,
)
from datasets_api.version_store import LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.artifact_store.conftest import FakeGCSClient, RecordingArtifactInventory

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._deps_doubles import fake_deps_cell
from ._frame_doubles import (
    _BASE_URL,
    _body,
    _coverage_service,
    _graph,
    _sdmx_transport,
    canonical_frame_digest,
    link_frame1_only,
)
from ._snapshot_doubles import (
    _NOW,
    FakeSnapshotReader,
    FakeSpecReader,
    _location,
    _store_and_locate,
    _version,
    make_client_for_door,
)
from .conftest import FakeDatasetStore
from .test_materialise_semaphore import _CountedPermits, _FrameTrackingPermits, _PermitLog
from .test_pin_door import FakeArtifactReferenceWriter
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

_FIVE_MINUTES = timedelta(minutes=5)


def _files_client(reader: FakeSnapshotReader, store: FakeArtifactStore):
    door = build_files_door(
        deps=fake_deps_cell(
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
        )
    )
    return make_client_for_door(door)


@pytest.mark.parametrize(
    ("revoked_version", "base_content_version"),
    [
        pytest.param("cv1", None, id="the-version"),
        pytest.param("cv_base", "cv_base", id="its-base-version"),
    ],
)
def test_a_link_to_a_revoked_version_refuses_snapshot_revoked_and_signs_nothing(
    revoked_version: str, base_content_version: str | None
):
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"csv-bytes")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version(base_content_version=base_content_version)},
        locations={("spec_a", "cv1"): [location]},
        revocations={("spec_a", revoked_version): Revocation(reason="takedown", revoked_at=_NOW)},
    )

    resp = _files_client(reader, store).get(
        f"/api/datasets/spec_a/cv1/files/{location.artifact_digest}", follow_redirects=False
    )

    assert resp.status_code == 410, resp.text
    assert resp.json()["code"] == "snapshot_revoked"
    assert store.signings == []


def test_a_link_to_a_servable_artifact_redirects_to_a_url_signed_for_at_most_five_minutes():
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"csv-bytes")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )

    resp = _files_client(reader, store).get(
        f"/api/datasets/spec_a/cv1/files/{location.artifact_digest}", follow_redirects=False
    )

    assert resp.status_code == 302, resp.text
    assert resp.headers["location"] == f"fake://artifact/working/{location.artifact_digest}"
    assert "no-store" in resp.headers["cache-control"]
    [(digest, tier, object_name, ttl)] = store.signings
    assert (digest, tier, object_name) == (location.artifact_digest, Tier.working, digest)
    assert ttl <= _FIVE_MINUTES


def test_a_link_naming_a_digest_the_version_does_not_link_refuses_artifact_not_found():
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"csv-bytes")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )

    with pytest.raises(ArtifactNotFound):
        read_servable_version(
            reader,
            subject_id="spec_a",
            content_version="cv1",
            artifact_digest="sha256:unlinked",
            now=_NOW,
        )
    resp = _files_client(reader, store).get(
        "/api/datasets/spec_a/cv1/files/sha256:unlinked", follow_redirects=False
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["code"] == "artifact_not_found"
    assert store.signings == []


class _RenewingArtifactStore(_InventoryRecordingArtifactStore):
    """Renews a working location past retention on `put` as the GCS store does: the bytes land
    under `<digest>#r<n>` and the location moves there with a fresh `stored_at`."""

    def __init__(self, *, inventory: LiveArtifactInventory) -> None:
        super().__init__(inventory=inventory)
        self.renewals: list[str] = []

    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        stored = FakeArtifactStore.put(self, data, tier=tier)
        location = self._inventory.get_location(stored.digest, tier=tier)
        if (
            tier is Tier.working
            and location is not None
            and working_tier_expired(location.stored_at, now=datetime.now(UTC))
        ):
            renewal = location.renewal + 1
            object_name = f"{stored.digest}#r{renewal}"
            self._inventory.renew(
                digest=stored.digest,
                tier=tier,
                bucket=stored.bucket,
                object_name=object_name,
                size_bytes=stored.size_bytes,
                renewal=renewal,
            )
            self.renewals.append(object_name)
            return StoredArtifact(
                digest=stored.digest,
                tier=tier,
                bucket=stored.bucket,
                object_name=object_name,
                size_bytes=stored.size_bytes,
                deduplicated=False,
            )
        return self._record(stored)


def _surface_client(version_store, artifact_store, *, coverage=None):
    dataset_store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=dataset_store),
        spec_reader=LiveSpecReader(store=dataset_store),
        graph=_graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        artifact_store=artifact_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=coverage if coverage is not None else _coverage_service(),
    )


def test_a_mint_and_a_snapshot_list_stable_links_and_sign_nothing(version_store):
    artifact_store = _InventoryRecordingArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )
    client = _surface_client(version_store, artifact_store)

    minted = client.post("/api/datasets", json=_body(format="csv", bundle=False))
    assert minted.status_code == 200, minted.text
    manifest = minted.json()
    spec_id, content_version = manifest["definition"]["spec_id"], manifest["content_version"]
    snapshot = client.get(f"/api/datasets/{spec_id}/{content_version}")
    assert snapshot.status_code == 200, snapshot.text

    for body in (manifest, snapshot.json()):
        assert body["artifacts"]
        for artifact in body["artifacts"]:
            assert artifact["url"] == (
                f"/api/datasets/{spec_id}/{content_version}/files/{artifact['artifact_digest']}"
            )
            assert "url_expires_at" not in artifact
    assert artifact_store.signings == []


def _mint_and_age_every_location(
    client, version_store, *, frame1_in: FakeArtifactStore | None = None
) -> tuple[dict[str, object], str]:
    """Mint a CSV, then age every location the version links past working retention, the
    stored frame's too, as one mint writes them all on the same day. Returns the manifest and
    the stored frame's digest. With `frame1_in`, the artifact store, the version links a
    frame1 document and no canonical frame."""
    minted = client.post("/api/datasets", json=_body(format="csv", bundle=False))
    assert minted.status_code == 200, minted.text
    manifest = minted.json()
    spec_id, content_version = manifest["definition"]["spec_id"], manifest["content_version"]
    if frame1_in is not None:
        frame_digest = link_frame1_only(
            version_store, frame1_in, subject_id=spec_id, content_version=content_version
        )
    else:
        frame_digest = canonical_frame_digest(
            version_store, subject_id=spec_id, content_version=content_version
        )
    with version_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_location SET stored_at = %s",
            (datetime.now(UTC) - timedelta(days=31),),
        )
    return manifest, frame_digest


def _csv(manifest: dict[str, object]) -> dict[str, object]:
    return next(a for a in manifest["artifacts"] if a["media_type"] == "text/csv")


@pytest.mark.parametrize("stored_frame", ["frame1", "canonical"])
def test_a_link_past_working_retention_renews_the_aged_frame_re_renders_and_redirects(
    version_store, stored_frame: str
):
    """The rendering's object is gone and the stored frame, a frame1 document or a canonical
    frame alone, aged out with it, its object still present: the link renews the frame, renders
    from its bytes, renews the rendering, and redirects."""
    artifact_store = _RenewingArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _surface_client(version_store, artifact_store)
    manifest, frame_digest = _mint_and_age_every_location(
        client,
        version_store,
        frame1_in=artifact_store if stored_frame == "frame1" else None,
    )
    csv = _csv(manifest)
    csv_bytes = artifact_store.fetch(csv["url"])
    # The lifecycle rule deleted the rendering's object.
    del artifact_store._objects[(csv["artifact_digest"], Tier.working.value)]

    resp = client.get(csv["url"], follow_redirects=False)

    assert resp.status_code == 302, resp.text
    renewed = f"{csv['artifact_digest']}#r1"
    assert artifact_store.renewals == [f"{frame_digest}#r1", renewed]
    assert resp.headers["location"] == f"fake://artifact/working/{renewed}"
    [(digest, _tier, object_name, ttl)] = artifact_store.signings
    assert (digest, object_name) == (csv["artifact_digest"], renewed)
    assert ttl <= _FIVE_MINUTES
    assert artifact_store.fetch(resp.headers["location"]) == csv_bytes
    # The render takes the renewed frame's bytes, so the frame is read once.
    assert [read for read in artifact_store.gets if read[0] == frame_digest] == [
        (frame_digest, Tier.working)
    ]


@pytest.mark.parametrize(
    ("link", "extension"), [("statistics", "parquet"), ("csv", "csv"), ("frame", "parquet")]
)
def test_a_link_past_working_retention_whose_object_is_present_renews_it_and_redirects(
    version_store, link: str, extension: str, monkeypatch: pytest.MonkeyPatch
):
    """A link whose object the lifecycle rule has not yet deleted renews that object under a
    new name, whatever its rendering, and redirects with no render to a URL naming the file by
    its media type, the stored frame's too. The renewal hands back whether it renewed and none
    of the bytes it read under its permit."""
    renewed: list[object] = []
    real_renew = files_module._renew

    def renew(location, *, deps):
        renewed.append(real_renew(location, deps=deps))
        return renewed[-1]

    monkeypatch.setattr(files_module, "_renew", renew)
    artifact_store = _RenewingArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _surface_client(version_store, artifact_store)
    manifest, frame_digest = _mint_and_age_every_location(client, version_store)
    if link == "frame":
        definition = manifest["definition"]
        digest = frame_digest
        url = f"/api/datasets/{definition['spec_id']}/{manifest['content_version']}/files/{digest}"
    else:
        [artifact] = [
            a
            for a in manifest["artifacts"]
            if (a["rendering"] == {"statistics": "series"}) is (link == "statistics")
        ]
        digest, url = artifact["artifact_digest"], artifact["url"]

    resp = client.get(url, follow_redirects=False)

    assert resp.status_code == 302, resp.text
    assert renewed == [True]
    assert artifact_store.renewals == [f"{digest}#r1"]
    assert resp.headers["location"] == f"fake://artifact/working/{digest}#r1"
    _media_type, disposition = artifact_store.signed_headers[(Tier.working, digest)]
    assert disposition.endswith(f'.{extension}"')


def test_a_re_render_from_an_aged_frame_holds_one_permit_from_the_frame_read_to_the_render(
    version_store, monkeypatch: pytest.MonkeyPatch
):
    """With one materialise permit, the re-render of a link whose frame aged out takes it
    once: the frame's read and renewal, the coverage snapshot, the decode and the render all
    happen while it is held, and the call does not wait on itself. The clicked object's own
    renewal, which finds the object gone, takes and returns the permit before that."""
    log = _PermitLog()
    monkeypatch.setattr(
        deps_module, "MaterialisePermits", lambda _permits: _CountedPermits(1, log=log)
    )
    frame: dict[str, str] = {}

    class _LoggedStore(_RenewingArtifactStore):
        def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
            log.record("frame_read" if artifact_digest == frame.get("digest") else "read")
            return super().get(artifact_digest=artifact_digest, tier=tier)

        def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
            log.record("write")
            return super().put(data, tier=tier)

    for name, kind in (("read_stored_frame", "decode"), ("render_and_store", "render")):
        real = getattr(current_module, name)

        def logged(*args, _real=real, _kind=kind, **kwargs):
            log.record(_kind)
            return _real(*args, **kwargs)

        monkeypatch.setattr(current_module, name, logged)
    coverage = _coverage_service()
    take_snapshot = coverage.snapshot

    def logged_snapshot():
        log.record("snapshot")
        return take_snapshot()

    monkeypatch.setattr(coverage, "snapshot", logged_snapshot)
    artifact_store = _LoggedStore(inventory=LiveArtifactInventory(store=version_store))
    client = _surface_client(version_store, artifact_store, coverage=coverage)
    manifest, frame["digest"] = _mint_and_age_every_location(client, version_store)
    csv = _csv(manifest)
    del artifact_store._objects[(csv["artifact_digest"], Tier.working.value)]
    log.events.clear()

    resp = client.get(csv["url"], follow_redirects=False)

    assert resp.status_code == 302, resp.text
    kinds = log.kinds()
    assert kinds[:3] == ["acquire", "read", "release"]
    re_render = kinds[3:]
    assert re_render[0] == "acquire" and re_render[-1] == "release"
    assert re_render.count("acquire") == re_render.count("release") == 1
    assert [kind for kind in re_render if kind != "write"][1:-1] == [
        "frame_read",
        "snapshot",
        "decode",
        "render",
    ]
    # No frame bytes are read, renewed, decoded or rendered outside the permit.
    assert log.held_at("frame_read") == log.held_at("decode") == log.held_at("render") == {1}
    assert log.held_at("write") == {1}
    # The render's coverage snapshot is taken under the permit, where no reload replaces it.
    assert log.held_at("snapshot") == {1}


def test_a_re_render_frees_the_aged_frame_before_its_permit_is_released(
    version_store, monkeypatch: pytest.MonkeyPatch
):
    """A link renders again from a renewed aged frame. When the re-render gives its permit
    back, nothing references the frame's bytes."""
    frame: dict[str, str] = {}
    permits = _FrameTrackingPermits(1)
    monkeypatch.setattr(deps_module, "MaterialisePermits", lambda _permits: permits)

    class _TrackingStore(_RenewingArtifactStore):
        def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
            data = super().get(artifact_digest=artifact_digest, tier=tier)
            if artifact_digest != frame.get("digest"):
                return data
            return permits.track(data)

        def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
            return super().put(bytes(data), tier=tier)

    artifact_store = _TrackingStore(inventory=LiveArtifactInventory(store=version_store))
    client = _surface_client(version_store, artifact_store)
    manifest, frame["digest"] = _mint_and_age_every_location(client, version_store)
    csv = _csv(manifest)
    del artifact_store._objects[(csv["artifact_digest"], Tier.working.value)]

    resp = client.get(csv["url"], follow_redirects=False)

    assert resp.status_code == 302, resp.text
    assert len(permits.read) == 1
    assert permits.freed_at_release == [True]


class _ChangingRenderArtifactStore(_RenewingArtifactStore):
    """Stores every artifact but the frame with a trailing byte once `change` is set, as a
    re-render whose live citation read changed would."""

    change = False
    frame_digest = ""

    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        if self.change and f"sha256:{hashlib.sha256(data).hexdigest()}" != self.frame_digest:
            data += b"\n"
        return super().put(data, tier=tier)


def test_a_link_whose_re_render_produces_other_bytes_refuses_artifact_expired(version_store):
    """The re-render links its new bytes in the clicked digest's place, so the clicked link
    names an expired artifact (410), never an unknown one (404), on this click and every later
    one."""
    artifact_store = _ChangingRenderArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )
    client = _surface_client(version_store, artifact_store)
    manifest, frame_digest = _mint_and_age_every_location(client, version_store)
    csv = _csv(manifest)
    del artifact_store._objects[(csv["artifact_digest"], Tier.working.value)]
    artifact_store.change, artifact_store.frame_digest = True, frame_digest

    resp = client.get(csv["url"], follow_redirects=False)

    assert resp.status_code == 410, resp.text
    assert resp.json()["code"] == "artifact_expired"
    assert artifact_store.renewals == [f"{frame_digest}#r1"]
    assert artifact_store.signings == []
    again = client.get(csv["url"], follow_redirects=False)
    assert again.status_code == 410, again.text
    assert again.json()["code"] == "artifact_expired"


def test_a_link_past_working_retention_whose_objects_are_gone_refuses_artifact_expired(
    version_store,
):
    artifact_store = _RenewingArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _surface_client(version_store, artifact_store)
    manifest, frame_digest = _mint_and_age_every_location(client, version_store)
    csv = _csv(manifest)
    # The lifecycle rule deleted the rendering's object and the frame's.
    del artifact_store._objects[(csv["artifact_digest"], Tier.working.value)]
    del artifact_store._objects[(frame_digest, Tier.working.value)]

    resp = client.get(csv["url"], follow_redirects=False)

    assert resp.status_code == 410, resp.text
    assert resp.json()["code"] == "artifact_expired"
    assert artifact_store.renewals == []
    assert artifact_store.signings == []


_SITE = DatasetsBuildSettings().site


@pytest.mark.parametrize(
    ("rendering", "renders"),
    [
        pytest.param({"format": "csv", "bundle": False, "layout": 3}, True, id="bare-long"),
        pytest.param(
            {"format": "csv", "bundle": False, "layout": 2}, False, id="superseded-bare-long"
        ),
        pytest.param(
            {"format": "parquet", "bundle": False, "layout": "frame"}, True, id="bare-frame"
        ),
        pytest.param(
            {"format": "csv", "bundle": True, "rendition": rendition_token(_SITE)},
            True,
            id="current-bundle",
        ),
        pytest.param(
            {"format": "csv", "bundle": True, "rendition": "v1-000000000000"},
            False,
            id="superseded-bundle",
        ),
        pytest.param({"format": "csv", "bundle": False}, False, id="old-bare-layout"),
        pytest.param(
            incomplete_rendering(
                {"format": "csv", "bundle": True, "rendition": rendition_token(_SITE)}
            ),
            False,
            id="incomplete-bundle",
        ),
        pytest.param({"format": "frame"}, False, id="canonical-frame"),
        pytest.param({"format": "frame1"}, False, id="frame1"),
        pytest.param({"statistics": "series"}, False, id="statistics"),
    ],
)
def test_only_a_rendering_this_build_renders_has_options_to_render_it_again(
    rendering: dict[str, object], renders: bool
):
    options = rendering_options(rendering, site=_SITE)

    assert (options is not None) is renders
    if options is not None:
        assert artifact_rendering(options, site=_SITE) == rendering


def test_a_renewal_inside_the_reuse_window_signs_the_renewed_object_name(monkeypatch):
    """A put renews expired bytes under `#r1` between two clicks. The second click signs the
    location it read, never a URL cached for the old object name."""
    inventory = RecordingArtifactInventory()
    store = LiveArtifactStore(
        buckets={Tier.working: "fake-bucket", Tier.pinned: "fake-pinned-bucket"},
        inventory=inventory,
        client=FakeGCSClient(),
    )
    monkeypatch.setattr(
        "artifact_store.store.sign_get_url",
        lambda **kwargs: f"https://signed.example/{kwargs['object_name']}",
    )
    digest = "sha256:aaa"
    inventory.record(
        digest=digest, tier=Tier.working, bucket="fake-bucket", object_name=digest, size_bytes=10
    )
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [_location(digest=digest)]},
    )
    client = _files_client(reader, store)
    link = f"/api/datasets/spec_a/cv1/files/{digest}"

    first = client.get(link, follow_redirects=False)
    inventory.renew(
        digest=digest,
        tier=Tier.working,
        bucket="fake-bucket",
        object_name=f"{digest}#r1",
        size_bytes=10,
        renewal=1,
    )
    reader.locations[("spec_a", "cv1")] = [
        replace(_location(digest=digest), object_name=f"{digest}#r1")
    ]
    second = client.get(link, follow_redirects=False)

    assert (first.status_code, second.status_code) == (302, 302), second.text
    assert first.headers["location"] == f"https://signed.example/{digest}"
    assert second.headers["location"] == f"https://signed.example/{digest}#r1"


class _SlowArtifactStore(FakeArtifactStore):
    """Takes a second over every read, as a stalled GCS read would."""

    def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
        time.sleep(1.0)
        return super().get(artifact_digest=artifact_digest, tier=tier)


def test_a_renewal_slower_than_the_request_deadline_refuses_timeout():
    """A renewal runs under the request deadline as every door's work does, so a stalled read
    answers 504 at the deadline and signs nothing."""
    store = _SlowArtifactStore()
    location = replace(_store_and_locate(store, b"csv-bytes"), stored_at=_NOW - timedelta(days=31))
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()}, locations={("spec_a", "cv1"): [location]}
    )
    door = build_files_door(
        deps=fake_deps_cell(
            settings=DatasetsBuildSettings(definition_timeout_seconds=0.2),
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
        )
    )

    started = time.monotonic()
    resp = make_client_for_door(door).get(
        f"/api/datasets/spec_a/cv1/files/{location.artifact_digest}", follow_redirects=False
    )

    assert resp.status_code == 504, resp.text
    assert resp.json()["code"] == "timeout"
    assert time.monotonic() - started < 1.0
    assert store.signings == []
