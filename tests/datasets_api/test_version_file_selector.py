"""`GET /api/datasets/{id}/{content_version}/file?format=&bundle=&layout=` answers 302 to one
file of a stored version. A file the version lists under the requested rendering is signed in
hand. A missing one renders from the version's stored frame through the function a link's
renewal also calls (`doors.files.render_version_from_frame`), and a derived version renders as
itself, under its own recorded minting options.
"""

from __future__ import annotations

import contextvars
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from artifact_store import LiveArtifactInventory, Tier
from dataset_store.snapshot import LiveSnapshotReader, Revocation
from datasets_api import deps as deps_module
from datasets_api.admission import MaterialisePermits
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors import files as files_module
from datasets_api.doors.files import build_files_door
from datasets_api.mint_models import MintOptions
from datasets_api.render import artifact_rendering
from datasets_api.retention import version_retention
from materialise.formats import EXCEL_MAX_ROWS
from spec_registry import IdentityQuarantined

from ._artifact_doubles import FakeArtifactStore
from ._deps_doubles import fake_deps_cell
from ._snapshot_doubles import (
    _NOW,
    BlockingFakeSpecReader,
    FakeSnapshotReader,
    FakeSpecReader,
    _store_and_locate,
    _version,
    make_client_for_door,
)
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door import _client
from .test_definition_door_currency import _FakeHeadStore
from .test_definition_door_groupby_collapse import (
    _GROUPBY_OPTIONS,
    _record_excel_overflowing_rows,
    _writer_row_detail,
)
from .test_download_links import (
    _mint_and_age_every_location,
    _RenewingArtifactStore,
    _surface_client,
)
from .test_serve_groupby_collapse import (
    _code_index,
    _coverage_service,
    _graph,
    _mint,
    _sdmx_transport,
    _spec_id_and_sidecar,
)

_SITE = DatasetsBuildSettings().site
_PARQUET = "format=parquet&bundle=false"


class _RenderSpy:
    """Records the options of each call to `render_version_from_frame` and lets it run."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.options: list[MintOptions] = []
        real = files_module.render_version_from_frame

        def spy(**kwargs: Any):
            self.options.append(kwargs["options"])
            return real(**kwargs)

        monkeypatch.setattr(files_module, "render_version_from_frame", spy)


class _ClosablePermits(MaterialisePermits):
    """Materialise permits that grant none once `closed` is set, as a semaphore every frame
    holds does."""

    closed = False

    def acquire(self, *, timeout: float, keep_waiting=None) -> bool:
        if self.closed:
            return False
        return super().acquire(timeout=timeout, keep_waiting=keep_waiting)


def _rig(**client_kwargs: Any):
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _RenewingArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        head_writer=_FakeHeadStore(),
        artifact_store=artifact_store,
        code_index=_code_index(),
        coverage=_coverage_service(),
        **client_kwargs,
    )
    return client, version_store, artifact_store


def _selector(manifest: dict[str, Any], query: str) -> str:
    spec_id, content_version = manifest["definition"]["spec_id"], manifest["content_version"]
    return f"/api/datasets/{spec_id}/{content_version}/file?{query}"


def _signed_digest(location: str) -> str:
    return location.removeprefix("fake://artifact/working/").split("#", 1)[0]


def _linked(version_store, *, subject_id: str, content_version: str) -> list[dict[str, Any]]:
    with version_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT artifact_digest, rendering FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s",
            (subject_id, content_version),
        )
    return [dict(row) for row in rows]


def test_a_rendering_the_version_lists_redirects_to_its_file_with_no_render(monkeypatch):
    spy = _RenderSpy(monkeypatch)
    client, _, artifact_store = _rig()
    spec_id, sidecar = _spec_id_and_sidecar()
    minted = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    [csv] = [a for a in minted["artifacts"] if a["media_type"] == "text/csv"]

    resp = client.get(_selector(minted, "format=csv&bundle=false"), follow_redirects=False)

    assert resp.status_code == 302, resp.text
    assert resp.headers["location"] == f"fake://artifact/working/{csv['artifact_digest']}"
    assert "no-store" in resp.headers["cache-control"]
    assert spy.options == []
    [(digest, _tier, _object_name, ttl)] = artifact_store.signings
    assert digest == csv["artifact_digest"]
    assert ttl <= timedelta(minutes=5)


def test_a_format_the_version_lacks_renders_from_its_frame_and_redirects_to_it(monkeypatch):
    spy = _RenderSpy(monkeypatch)
    client, version_store, artifact_store = _rig()
    spec_id, sidecar = _spec_id_and_sidecar()
    minted = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    parquet = MintOptions(format="parquet", bundle=False)

    resp = client.get(_selector(minted, _PARQUET), follow_redirects=False)

    assert resp.status_code == 302, resp.text
    assert spy.options == [parquet]
    rendering = artifact_rendering(parquet, site=_SITE)
    [linked] = [
        link
        for link in _linked(
            version_store, subject_id=spec_id, content_version=minted["content_version"]
        )
        if link["rendering"] == rendering
    ]
    assert _signed_digest(resp.headers["location"]) == linked["artifact_digest"]
    # The rendering is linked now, so the next call is a hit.
    again = client.get(_selector(minted, _PARQUET), follow_redirects=False)
    assert again.status_code == 302, again.text
    assert again.headers["location"] == resp.headers["location"]
    assert len(spy.options) == 1
    assert [s[0] for s in artifact_store.signings] == [linked["artifact_digest"]] * 2


def test_a_derived_version_renders_as_itself_from_its_base_frame(monkeypatch):
    """A collapse renders a missing format under its own recorded groupby, from its base
    version's frame, and links the file under itself. Its minted format is a hit."""
    spy = _RenderSpy(monkeypatch)
    client, version_store, _ = _rig()
    spec_id, sidecar = _spec_id_and_sidecar()
    derived = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_GROUPBY_OPTIONS)
    [csv] = [a for a in derived["artifacts"] if a["media_type"] == "text/csv"]

    hit = client.get(_selector(derived, "format=csv&bundle=false"), follow_redirects=False)
    resp = client.get(_selector(derived, _PARQUET), follow_redirects=False)

    assert hit.status_code == 302, hit.text
    assert _signed_digest(hit.headers["location"]) == csv["artifact_digest"]
    assert resp.status_code == 302, resp.text
    grouped = MintOptions.model_validate({**_GROUPBY_OPTIONS, "format": "parquet"})
    assert spy.options == [grouped]
    rendering = artifact_rendering(grouped, site=_SITE)
    [linked] = [
        link
        for link in _linked(
            version_store, subject_id=spec_id, content_version=derived["content_version"]
        )
        if link["rendering"] == rendering
    ]
    assert _signed_digest(resp.headers["location"]) == linked["artifact_digest"]


def test_an_aged_file_renews_through_the_function_the_selector_renders_with(
    version_store, monkeypatch
):
    """A link whose object is gone renders its rendering again through
    `render_version_from_frame`, the selector's render."""
    spy = _RenderSpy(monkeypatch)
    artifact_store = _RenewingArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _surface_client(version_store, artifact_store)
    manifest, _frame_digest = _mint_and_age_every_location(client, version_store)
    [csv] = [a for a in manifest["artifacts"] if a["media_type"] == "text/csv"]
    del artifact_store._objects[(csv["artifact_digest"], Tier.working.value)]

    resp = client.get(csv["url"], follow_redirects=False)

    assert resp.status_code == 302, resp.text
    assert [artifact_rendering(o, site=_SITE) for o in spy.options] == [csv["rendering"]]


def test_a_selector_render_renews_a_frame_past_working_retention(version_store):
    artifact_store = _RenewingArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _surface_client(version_store, artifact_store)
    manifest, frame_digest = _mint_and_age_every_location(client, version_store)

    resp = client.get(_selector(manifest, _PARQUET), follow_redirects=False)

    assert resp.status_code == 302, resp.text
    assert artifact_store.renewals[0] == f"{frame_digest}#r1"


def test_a_selector_render_whose_frame_bytes_are_gone_refuses_artifact_expired(version_store):
    artifact_store = _RenewingArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _surface_client(version_store, artifact_store)
    manifest, frame_digest = _mint_and_age_every_location(client, version_store)
    del artifact_store._objects[(frame_digest, Tier.working.value)]

    resp = client.get(_selector(manifest, _PARQUET), follow_redirects=False)

    assert resp.status_code == 410, resp.text
    assert resp.json()["code"] == "artifact_expired"
    assert artifact_store.signings == []


def test_a_selector_serves_an_aged_file_whose_bytes_survive_though_the_frame_bytes_are_gone(
    version_store, monkeypatch
):
    """The aged CSV renews as its link would renew it, with no render."""
    spy = _RenderSpy(monkeypatch)
    artifact_store = _RenewingArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _surface_client(version_store, artifact_store)
    manifest, frame_digest = _mint_and_age_every_location(client, version_store)
    [csv] = [a for a in manifest["artifacts"] if a["media_type"] == "text/csv"]
    del artifact_store._objects[(frame_digest, Tier.working.value)]

    resp = client.get(_selector(manifest, "format=csv&bundle=false"), follow_redirects=False)

    assert resp.status_code == 302, resp.text
    assert _signed_digest(resp.headers["location"]) == csv["artifact_digest"]
    assert artifact_store.renewals == [f"{csv['artifact_digest']}#r1"]
    assert spy.options == []


@pytest.mark.parametrize(
    ("download", "frame_renews"),
    [
        pytest.param("frame-link", True, id="the-frame-link"),
        pytest.param("layout-frame", True, id="the-selector-with-layout-frame"),
        pytest.param("csv-link", False, id="another-file-link"),
    ],
)
def test_only_a_download_of_the_frame_moves_held_until(
    version_store, download: str, frame_renews: bool
):
    """A download renews only the file it names, and the frame is itself a listed file."""
    artifact_store = _RenewingArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _surface_client(version_store, artifact_store)
    manifest, frame_digest = _mint_and_age_every_location(client, version_store)
    [csv] = [a for a in manifest["artifacts"] if a["media_type"] == "text/csv"]
    spec_id, content_version = manifest["definition"]["spec_id"], manifest["content_version"]
    reader = LiveSnapshotReader(store=version_store)

    def held_until():
        retention = version_retention(
            reader, subject_id=spec_id, content_version=content_version, now=datetime.now(UTC)
        )
        return retention.held_until

    before = held_until()
    path = {
        "frame-link": f"/api/datasets/{spec_id}/{content_version}/files/{frame_digest}",
        "layout-frame": _selector(manifest, f"{_PARQUET}&layout=frame"),
        "csv-link": csv["url"],
    }[download]

    resp = client.get(path, follow_redirects=False)

    assert resp.status_code == 302, resp.text
    after = held_until()
    assert before is not None and after is not None
    assert (after > before + timedelta(days=30)) is frame_renews
    assert (after == before) is not frame_renews


def test_an_xlsx_over_the_sheet_limit_refuses_excel_limit():
    client, version_store, _ = _rig()
    spec_id, sidecar = _spec_id_and_sidecar()
    minted = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    _record_excel_overflowing_rows(
        version_store, subject_id=spec_id, content_version=minted["content_version"]
    )

    resp = client.get(_selector(minted, "format=xlsx&bundle=false"), follow_redirects=False)

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "excel_limit"
    assert resp.json()["detail"] == _writer_row_detail(EXCEL_MAX_ROWS + 1)


def test_a_full_materialise_semaphore_refuses_a_render_retryable_503(monkeypatch):
    permits: list[_ClosablePermits] = []

    def build(count: int) -> _ClosablePermits:
        permits.append(_ClosablePermits(count))
        return permits[-1]

    monkeypatch.setattr(deps_module, "MaterialisePermits", build)
    client, _, _ = _rig(
        settings=DatasetsBuildSettings.from_env().model_copy(
            update={"materialise_wait_seconds": 0.1}
        )
    )
    spec_id, sidecar = _spec_id_and_sidecar()
    minted = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    for held in permits:
        held.closed = True

    resp = client.get(_selector(minted, _PARQUET), follow_redirects=False)

    assert resp.status_code == 503, resp.text
    assert resp.json()["code"] == "service_unavailable"
    assert resp.json()["retryable"] is True


@pytest.mark.parametrize(
    ("query", "pointer"),
    [
        pytest.param("groupby.by=recipient&groupby.op=sum", "/options/groupby/by", id="groupby"),
        pytest.param("without_party_totals=true", "/options/without_party_totals", id="cut"),
        pytest.param("format=csv&format=xlsx", "/options/format", id="repeated"),
        pytest.param("format=csv&bundle=false&layout=frame", "/options/layout", id="layout-csv"),
        pytest.param("layout=3", "/options/layout", id="layout-integer"),
    ],
)
def test_a_query_the_selector_does_not_read_refuses_request_invalid(query: str, pointer: str):
    client, _, artifact_store = _rig()
    spec_id, sidecar = _spec_id_and_sidecar()
    minted = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})

    resp = client.get(_selector(minted, query), follow_redirects=False)

    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "request_invalid"
    assert resp.json()["pointer"] == pointer
    assert artifact_store.signings == []


class _QuarantiningSpecReader(FakeSpecReader):
    def read_served_spec_ids(self, spec_id: str) -> tuple[str, ...] | None:
        raise IdentityQuarantined(subject_id=spec_id, id_kind="spec")


@pytest.mark.parametrize(
    ("spec_reader", "reader", "status", "code"),
    [
        pytest.param(FakeSpecReader(), FakeSnapshotReader(), 404, "spec_not_found", id="spec"),
        pytest.param(
            _QuarantiningSpecReader(spec_ids=["spec_a"]),
            FakeSnapshotReader(versions={("spec_a", "cv1"): _version()}),
            500,
            "identity_quarantined",
            id="quarantined",
        ),
        pytest.param(
            FakeSpecReader(spec_ids=["spec_a"]),
            FakeSnapshotReader(),
            404,
            "snapshot_not_found",
            id="version",
        ),
        pytest.param(
            FakeSpecReader(spec_ids=["spec_a"]),
            FakeSnapshotReader(
                versions={("spec_a", "cv1"): _version()},
                revocations={("spec_a", "cv1"): Revocation(reason="takedown", revoked_at=_NOW)},
            ),
            410,
            "snapshot_revoked",
            id="revoked",
        ),
    ],
)
def test_the_selector_refuses_a_version_the_snapshot_door_refuses(
    spec_reader, reader, status: int, code: str
):
    store = FakeArtifactStore()
    door = build_files_door(
        deps=fake_deps_cell(
            spec_reader=spec_reader, snapshot_reader=reader, artifact_store=store, now=lambda: _NOW
        )
    )

    resp = make_client_for_door(door).get(
        "/api/datasets/spec_a/cv1/file?format=csv", follow_redirects=False
    )

    assert resp.status_code == status, resp.text
    assert resp.json()["code"] == code
    assert store.signings == []


def test_a_full_read_executor_refuses_a_selector_retryable_503():
    store = FakeArtifactStore()
    options = MintOptions(format="csv", bundle=False)
    location = _store_and_locate(
        store,
        b"csv-bytes",
        rendering=artifact_rendering(options, site=_SITE),
    )
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()}, locations={("spec_a", "cv1"): [location]}
    )
    blocking = BlockingFakeSpecReader(spec_ids=["spec_a"])
    door = build_files_door(
        deps=fake_deps_cell(
            settings=DatasetsBuildSettings.from_env().model_copy(
                update={"read_max_concurrency": 1}
            ),
            spec_reader=blocking,
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
        )
    )
    client = make_client_for_door(door)
    path = "/api/datasets/spec_a/cv1/file?format=csv&bundle=false"
    first: list = []

    with client:
        # The thread reads under this context's bound graph release, as a request does.
        context = contextvars.copy_context()
        thread = threading.Thread(
            target=lambda: first.append(context.run(client.get, path, follow_redirects=False))
        )
        thread.start()
        assert blocking.entered.wait(timeout=5.0)

        resp = client.get(path, follow_redirects=False)

        assert resp.status_code == 503, resp.text
        assert resp.json()["code"] == "service_unavailable"
        assert resp.json()["retryable"] is True
        blocking.release.set()
        thread.join(timeout=5.0)
    assert first and first[0].status_code == 302, first[0].text
