"""`GET /api/datasets/{id}/{content_version}/file?format=&bundle=&layout=` answers 302 to one
file of a stored version. A file the version lists under the requested rendering is signed in
hand. A missing one renders from the version's stored frame through the function a link's
renewal also calls (`doors.files.render_version_from_frame`), and a derived version renders as
itself, under its own recorded minting options.
"""

from __future__ import annotations

import contextvars
import io
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pyarrow.parquet as pq
import pytest

from artifact_store import LiveArtifactInventory, Tier
from code_index import CodeIndex, CodeRow, CodeSystemDeclaration
from dataset_store.snapshot import LiveSnapshotReader, Revocation
from datasets_api import deps as deps_module
from datasets_api.admission import MaterialisePermits
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors import files as files_module
from datasets_api.doors.files import build_files_door
from datasets_api.mint_models import MintOptions
from datasets_api.render import artifact_rendering
from datasets_api.retention import version_retention
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord
from materialise.formats import (
    EXCEL_MAX_ROWS,
    FRAME1_MEDIA_TYPE,
    read_canonical_frame,
    write_frame1,
)
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import IdentityQuarantined
from spec_registry.identity import recompute_identity
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from . import test_serve_party_totals as party_totals
from ._artifact_doubles import FakeArtifactStore
from ._deps_doubles import fake_deps_cell
from ._frame_doubles import canonical_frame_digest
from ._graph_doubles import FakeMintGraphReader
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
from .conftest import FakeDatasetStore
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


def test_a_derived_version_renders_as_itself_from_its_own_frame(monkeypatch):
    """A collapse renders a missing format under its own recorded groupby, from the frame it
    stored, and links the file under itself. Its minted format is a hit."""
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


def _base_content_version(version_store, *, subject_id: str, content_version: str) -> str:
    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT base_content_version FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (subject_id, content_version),
        )
    assert row is not None and row["base_content_version"] is not None
    return str(row["base_content_version"])


def _age_every_location(version_store) -> None:
    with version_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_location SET stored_at = %s",
            (datetime.now(UTC) - timedelta(days=31),),
        )


class _PartyRig:
    """One spec, version, head and artifact store shared by clients that read the party-totals
    fixture under different code indexes."""

    def __init__(self) -> None:
        self.dataset_store = FakeDatasetStore()
        self.version_store = FakeVersionRecordDatasetStore()
        self.head_store = _FakeHeadStore()
        self.artifact_store = _RenewingArtifactStore(
            inventory=LiveArtifactInventory(store=self.version_store)
        )

    def client(self, code_index: CodeIndex):
        return _client(
            graph=party_totals._graph(),
            sdmx_transport=party_totals._sdmx_transport(),
            store=self.dataset_store,
            version_store=self.version_store,
            head_writer=self.head_store,
            artifact_store=self.artifact_store,
            code_index=code_index,
            coverage=party_totals._coverage_service(),
        )


def _dac_total_read_as_a_party() -> CodeIndex:
    """The party-totals fixture index with the DAC total classified as a party, so a cut leaves
    nothing out."""
    index = party_totals._code_index()
    return CodeIndex(
        stamp=index.stamp,
        declarations=(),
        codes=(),
        statvars=index.statvars,
        parties=[
            replace(row, kind="party") if row.dcid == party_totals._DAC_TOTAL else row
            for row in index.parties.values()
        ],
    )


# The renderings a stored frame renders: each format, a bundle, and the frame layout.
_OUTPUTS = pytest.mark.parametrize(
    "output",
    [
        {"format": "csv", "bundle": False},
        {"format": "csv", "bundle": True},
        {"format": "xlsx", "bundle": False},
        {"format": "parquet", "bundle": False},
        {"format": "parquet", "bundle": False, "layout": "frame"},
    ],
    ids=["csv", "bundle", "xlsx", "parquet", "frame"],
)


def _query(output: dict[str, Any]) -> str:
    return "&".join(
        f"{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in output.items()
    )


def _derived_rig(derivation: str):
    """A client, its version and artifact stores, and the spec and options of a derived mint:
    a summed groupby, a groupby that sums `flowAxis` and keeps the code axis `testAxis`, or a
    party-totals cut."""
    if derivation == "groupby":
        client, version_store, artifact_store = _rig()
        return client, version_store, artifact_store, *_spec_id_and_sidecar(), _GROUPBY_OPTIONS
    if derivation == "groupby-keeps-a-code-axis":
        version_store = FakeVersionRecordDatasetStore()
        artifact_store = _RenewingArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        )
        spec_id = recompute_identity(_CUT_FAMILY_IDENTITY).spec_id
        return (
            _cut_family_client(version_store, artifact_store, flow_summable=True),
            version_store,
            artifact_store,
            spec_id,
            {"schema_version": 1, "spec_id": spec_id, "identity": _CUT_FAMILY_IDENTITY},
            {"groupby": {"by": ["recipient", "testAxis"], "op": "sum"}},
        )
    rig = _PartyRig()
    client = rig.client(party_totals._code_index())
    return (
        client,
        rig.version_store,
        rig.artifact_store,
        *party_totals._spec_id_and_sidecar(),
        party_totals._CUT,
    )


@pytest.mark.parametrize("via", ["selector", "link"])
@_OUTPUTS
@pytest.mark.parametrize("derivation", ["groupby", "groupby-keeps-a-code-axis", "cut"])
def test_a_derived_version_renders_from_its_own_frame_once_its_base_frame_is_gone(
    derivation: str, output: dict[str, Any], via: str
):
    """Every file aged out, the minted file's object and the base version's frame object gone:
    the selector, or a renewal of the minted file's link, renders the derived version from the
    frame it stored, and the file it serves is the one its mint wrote."""
    client, version_store, artifact_store, spec_id, sidecar, options = _derived_rig(derivation)
    options = {**options, **output}
    derived = _mint(client, spec_id=spec_id, sidecar=sidecar, options=options)
    rendering = artifact_rendering(MintOptions.model_validate(options), site=_SITE)
    [minted] = [a for a in derived["artifacts"] if a["rendering"] == rendering]
    base_frame = canonical_frame_digest(
        version_store,
        subject_id=spec_id,
        content_version=_base_content_version(
            version_store, subject_id=spec_id, content_version=derived["content_version"]
        ),
    )
    _age_every_location(version_store)
    del artifact_store._objects[(minted["artifact_digest"], Tier.working.value)]
    del artifact_store._objects[(base_frame, Tier.working.value)]

    path = minted["url"] if via == "link" else _selector(derived, _query(output))
    resp = client.get(path, follow_redirects=False)

    assert resp.status_code == 302, resp.text
    assert _signed_digest(resp.headers["location"]) == minted["artifact_digest"]


def test_a_cut_version_serves_its_files_after_a_party_classification_changes():
    """The cut left the DAC total out. Under an index that reads it as a party, a cut of the
    base frame leaves nothing out, yet the cut version still serves a file of its own rows."""
    rig = _PartyRig()
    cut = _mint(
        rig.client(party_totals._code_index()),
        spec_id=(spec := party_totals._spec_id_and_sidecar())[0],
        sidecar=spec[1],
        options=party_totals._CUT,
    )
    reclassified = rig.client(_dac_total_read_as_a_party())

    resp = reclassified.get(_selector(cut, _PARQUET), follow_redirects=False)

    assert resp.status_code == 302, resp.text
    rendering = artifact_rendering(
        MintOptions.model_validate({**party_totals._CUT, "format": "parquet"}), site=_SITE
    )
    [linked] = [
        link
        for link in _linked(
            rig.version_store, subject_id=spec[0], content_version=cut["content_version"]
        )
        if link["rendering"] == rendering
    ]
    assert _signed_digest(resp.headers["location"]) == linked["artifact_digest"]
    frame = pq.read_table(
        io.BytesIO(rig.artifact_store._objects[(linked["artifact_digest"], "working")])
    )
    assert frame.num_rows == 1


@pytest.mark.parametrize(
    ("code_index", "status"),
    [
        pytest.param(party_totals._code_index, 302, id="same-classification"),
        pytest.param(_dac_total_read_as_a_party, 410, id="reclassified"),
    ],
)
def test_a_cut_version_whose_own_frame_is_gone_serves_a_re_cut_only_when_it_reproduces(
    code_index, status: int
):
    """With the cut version's own frame object gone, the selector cuts its base frame again
    under the serving index, and serves the file only when that cut gives the same version."""
    rig = _PartyRig()
    spec_id, sidecar = party_totals._spec_id_and_sidecar()
    cut = _mint(
        rig.client(party_totals._code_index()),
        spec_id=spec_id,
        sidecar=sidecar,
        options=party_totals._CUT,
    )
    [csv] = [a for a in cut["artifacts"] if a["media_type"] == "text/csv"]
    cut_frame = canonical_frame_digest(
        rig.version_store, subject_id=spec_id, content_version=cut["content_version"]
    )
    _age_every_location(rig.version_store)
    del rig.artifact_store._objects[(csv["artifact_digest"], Tier.working.value)]
    del rig.artifact_store._objects[(cut_frame, Tier.working.value)]

    resp = rig.client(code_index()).get(
        _selector(cut, "format=csv&bundle=false"), follow_redirects=False
    )

    assert resp.status_code == status, resp.text
    if status == 302:
        assert _signed_digest(resp.headers["location"]) == csv["artifact_digest"]
    else:
        assert resp.json()["code"] == "artifact_expired"


def test_a_dead_aged_canonical_frame_gives_way_to_a_servable_pinned_frame(version_store):
    """The version links its canonical frame, aged out with its object gone, and a frame1
    document of the same rows on the pinned tier. The selector renders from the pinned one."""
    artifact_store = _RenewingArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _surface_client(version_store, artifact_store)
    manifest, frame_digest = _mint_and_age_every_location(client, version_store)
    spec_id, content_version = manifest["definition"]["spec_id"], manifest["content_version"]
    frame = read_canonical_frame(
        artifact_store.get(artifact_digest=frame_digest, tier=Tier.working)
    )
    document = io.BytesIO()
    write_frame1((frame,), sink=document)
    pinned = artifact_store.put(document.getvalue(), tier=Tier.pinned)
    LiveVersionRecordWriter(store=version_store).record_artifact(
        subject_id=spec_id,
        content_version=content_version,
        artifact=ArtifactRecord(
            artifact_digest=pinned.digest,
            rendering={"format": "frame1"},
            media_type=FRAME1_MEDIA_TYPE,
        ),
    )
    del artifact_store._objects[(frame_digest, Tier.working.value)]

    resp = client.get(_selector(manifest, _PARQUET), follow_redirects=False)

    assert resp.status_code == 302, resp.text
    assert artifact_store.renewals == []


# A flow family of four members over a summable `testAxis` and a `flowAxis` that is not
# summable, each sent by France and the DAC total to Kenya.
_CUT_FAMILY = {
    f"CutMember{axis}{flow}": (f"TestAxis{axis}", f"Flow{flow}")
    for axis in ("A", "B")
    for flow in ("X", "Y")
}
_CUT_FAMILY_CSV = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,provider,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
) + "".join(
    f"dataflow,DC:DF_OBS(1.0.0),I,{member},{provider},country/KEN,USD,GrantEquivalent,"
    f"NotApplicable,OECD_DAC2A,2020,{value + offset},,facet1\r\n"
    for offset, member in enumerate(_CUT_FAMILY)
    for provider, value in ((party_totals._DAC_TOTAL, 900), (party_totals._FRANCE, 100))
)
_CUT_FAMILY_IDENTITY = {
    **party_totals._IDENTITY,
    "slots": [
        {"property": "testAxis", "binding": {"state": "set", "refs": ["TestAxisA", "TestAxisB"]}},
        {"property": "flowAxis", "binding": {"state": "set", "refs": ["FlowX", "FlowY"]}},
    ],
}


def _cut_family_code_index(*, flow_summable: bool) -> CodeIndex:
    """The party-totals fixture index over `_CUT_FAMILY`'s two code axes, `testAxis` summable
    and `flowAxis` summable with `flow_summable`."""
    index = party_totals._code_index()
    return CodeIndex(
        stamp=index.stamp,
        declarations=tuple(
            CodeSystemDeclaration(
                code_system=system,
                hierarchical=False,
                disjoint_from_children=False,
                summable=summable,
                absent_reading="not_declared",
                plain_name=system,
            )
            for system, summable in (("testAxis", True), ("flowAxis", flow_summable))
        ),
        codes=tuple(
            CodeRow(
                dcid=dcid,
                code_system=system,
                code=dcid[-1],
                label=f"Value {dcid[-1]}",
                synonyms=(),
                parent=None,
            )
            for system, dcids in (
                ("testAxis", ("TestAxisA", "TestAxisB")),
                ("flowAxis", ("FlowX", "FlowY")),
            )
            for dcid in dcids
        ),
        statvars=tuple(
            replace(
                index.statvars[0],
                dcid=member,
                constraint_properties={"testAxis": axis, "flowAxis": flow},
            )
            for member, (axis, flow) in _CUT_FAMILY.items()
        ),
        parties=list(index.parties.values()),
    )


def _cut_family_client(
    version_store,
    artifact_store,
    *,
    flow_summable: bool = False,
    store: FakeDatasetStore | None = None,
):
    """A client over `_CUT_FAMILY`, registering specs in `store`. With `flow_summable`, the
    index declares `flowAxis` summable too, so a groupby that keeps `testAxis` sums `flowAxis`
    into a frame of its own."""
    graph = FakeMintGraphReader(
        arcs={
            member: {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("DevelopmentFinance",),
                "measuredProperty": ("amount",),
                "observationProperties": ("provider", "recipient"),
                "constraintProperties": ("testAxis", "flowAxis"),
                "testAxis": (axis,),
                "flowAxis": (flow,),
            }
            for member, (axis, flow) in _CUT_FAMILY.items()
        },
        labels={"OECD_DAC2A": "OECD DAC2a", "country/KEN": "Kenya", "country/FRA": "France"},
    )
    coverage = FakeCoverageService(
        build_coverage_index(
            [
                (member, "facet1", provider, "country/KEN", [2020])
                for member in _CUT_FAMILY
                for provider in (party_totals._DAC_TOTAL, party_totals._FRANCE)
            ],
            role_names={
                member: RoleNames(donor="provider", recipient="recipient") for member in _CUT_FAMILY
            },
            facet_metadata={
                "facet1": FacetMetadata(
                    provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
                )
            },
        )
    )
    return _client(
        graph=graph,
        sdmx_transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text=_CUT_FAMILY_CSV,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )
        ),
        store=store,
        version_store=version_store,
        head_writer=_FakeHeadStore(),
        artifact_store=artifact_store,
        code_index=_cut_family_code_index(flow_summable=flow_summable),
        coverage=coverage,
    )


def _cut_with_groupby(
    by: list[str], *, flow_summable: bool, output: dict[str, Any]
) -> tuple[Any, FakeDatasetStore, Any, _RenewingArtifactStore, dict[str, Any], dict[str, Any]]:
    """A client over `_CUT_FAMILY`, its spec, version and artifact stores, a cut of it minted
    with a groupby by `by` in `output`, and the minted file of `output`'s rendering."""
    store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _RenewingArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _cut_family_client(
        version_store, artifact_store, flow_summable=flow_summable, store=store
    )
    spec_id = recompute_identity(_CUT_FAMILY_IDENTITY).spec_id
    options = {**output, "without_party_totals": True, "groupby": {"by": by, "op": "sum"}}
    cut = _mint(
        client,
        spec_id=spec_id,
        sidecar={"schema_version": 1, "spec_id": spec_id, "identity": _CUT_FAMILY_IDENTITY},
        options=options,
    )
    assert cut["trace"]["party_totals_left_out"] == party_totals._LEFT_OUT
    rendering = artifact_rendering(MintOptions.model_validate(options), site=_SITE)
    [minted] = [a for a in cut["artifacts"] if a["rendering"] == rendering]
    return client, store, version_store, artifact_store, cut, minted


def _frame_summed(version_store, *, subject_id: str, content_version: str) -> bool | None:
    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT frame_summed FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (subject_id, content_version),
        )
    assert row is not None
    return row["frame_summed"]


def _drop_base_frame(version_store, artifact_store, *, cut: dict[str, Any]) -> None:
    """Age every location and delete the bytes of `cut`'s base version's frame."""
    spec_id = cut["definition"]["spec_id"]
    base_frame = canonical_frame_digest(
        version_store,
        subject_id=spec_id,
        content_version=_base_content_version(
            version_store, subject_id=spec_id, content_version=cut["content_version"]
        ),
    )
    _age_every_location(version_store)
    del artifact_store._objects[(base_frame, Tier.working.value)]


# A cut minted with a groupby by `by`, and whether its stored frame is the collapse.
_CUT_GROUPBYS = pytest.mark.parametrize(
    ("by", "flow_summable", "summed"),
    [
        pytest.param(
            ["recipient", "testAxis"], False, False, id="groupby-leaves-the-cut-unchanged"
        ),
        pytest.param(["recipient"], False, False, id="groupby-keeps-an-unsummed-axis"),
        pytest.param(
            ["recipient", "testAxis"], True, True, id="groupby-sums-and-keeps-a-code-axis"
        ),
    ],
)


@pytest.mark.parametrize("aged", ["every-location", "the-minted-file"])
@_OUTPUTS
@_CUT_GROUPBYS
def test_a_cut_minted_with_a_groupby_serves_the_file_its_mint_wrote(
    by: list[str], flow_summable: bool, summed: bool, output: dict[str, Any], aged: str
):
    """The groupby leaves the cut frame unchanged, or keeps both values of the unsummed
    `flowAxis` and sums `testAxis` into rows that are no frame of their own, so the version is
    the cut frame (`render._derived_version`), or sums a summable `flowAxis` and keeps the
    code axis `testAxis`, so the version is the collapse. With its minted file aged out and
    the file's bytes gone, the selector serves the same file."""
    client, _, version_store, artifact_store, cut, minted = _cut_with_groupby(
        by, flow_summable=flow_summable, output=output
    )
    spec_id = cut["definition"]["spec_id"]
    assert (
        _frame_summed(version_store, subject_id=spec_id, content_version=cut["content_version"])
        is summed
    )
    if aged == "every-location":
        _age_every_location(version_store)
    else:
        with version_store.transaction() as tx:
            tx.execute(
                "UPDATE datasets.artifact_location SET stored_at = %s WHERE artifact_digest = %s",
                (datetime.now(UTC) - timedelta(days=31), minted["artifact_digest"]),
            )
    del artifact_store._objects[(minted["artifact_digest"], Tier.working.value)]
    resp = client.get(_selector(cut, _query(output)), follow_redirects=False)

    assert resp.status_code == 302, resp.text
    assert _signed_digest(resp.headers["location"]) == minted["artifact_digest"]


@pytest.mark.parametrize("via", ["selector", "link"])
@_OUTPUTS
@_CUT_GROUPBYS
def test_a_cut_minted_with_a_groupby_renders_from_its_own_frame_once_its_base_frame_is_gone(
    by: list[str], flow_summable: bool, summed: bool, output: dict[str, Any], via: str
):
    """Every file aged out, the minted file's bytes and the base version's frame bytes gone:
    the cut renders from its own frame as its recorded `frame_summed` says, and the file it
    serves is the one its mint wrote."""
    client, _, version_store, artifact_store, cut, minted = _cut_with_groupby(
        by, flow_summable=flow_summable, output=output
    )
    _drop_base_frame(version_store, artifact_store, cut=cut)
    del artifact_store._objects[(minted["artifact_digest"], Tier.working.value)]

    path = minted["url"] if via == "link" else _selector(cut, _query(output))
    resp = client.get(path, follow_redirects=False)

    assert resp.status_code == 302, resp.text
    assert _signed_digest(resp.headers["location"]) == minted["artifact_digest"]


@pytest.mark.parametrize(
    ("base_frame", "status"),
    [pytest.param(True, 302, id="base-frame-kept"), pytest.param(False, 410, id="base-frame-gone")],
)
@pytest.mark.parametrize("via", ["selector", "link"])
def test_a_cut_with_a_groupby_recorded_before_frame_summed_renders_from_its_base_alone(
    via: str, base_frame: bool, status: int
):
    """A cut minted with a groupby whose record holds no `frame_summed`, and whose files all
    record the groupby, cannot say whether its own frame is the collapse. It renders from its
    base version's frame alone, so once that frame's bytes are gone it serves nothing."""
    output = {"format": "parquet", "bundle": False}
    client, _, version_store, artifact_store, cut, minted = _cut_with_groupby(
        ["recipient"], flow_summable=False, output=output
    )
    with version_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.version_record SET frame_summed = NULL WHERE content_version = %s",
            (cut["content_version"],),
        )
    if base_frame:
        _age_every_location(version_store)
    else:
        _drop_base_frame(version_store, artifact_store, cut=cut)
    del artifact_store._objects[(minted["artifact_digest"], Tier.working.value)]

    path = minted["url"] if via == "link" else _selector(cut, _query(output))
    resp = client.get(path, follow_redirects=False)

    assert resp.status_code == status, resp.text
    if status == 302:
        assert _signed_digest(resp.headers["location"]) == minted["artifact_digest"]
    else:
        assert resp.json()["code"] == "artifact_expired"


@pytest.mark.parametrize("via", ["selector", "link"])
def test_a_cut_frame_whose_groupby_now_collapses_falls_back_to_the_base_frame(via: str):
    """The groupby kept the unsummed `flowAxis`, so the cut's frame is the cut frame. Under an
    index that declares `flowAxis` summable, the same groupby of that frame gives a collapse
    of its own. That render records no version derived from the cut, and the cut renders from
    its base version's frame instead, which records its collapse beside the cut."""
    output = {"format": "parquet", "bundle": False}
    _, store, version_store, artifact_store, cut, minted = _cut_with_groupby(
        ["recipient"], flow_summable=False, output=output
    )
    spec_id = cut["definition"]["spec_id"]
    base = _base_content_version(
        version_store, subject_id=spec_id, content_version=cut["content_version"]
    )
    _age_every_location(version_store)
    del artifact_store._objects[(minted["artifact_digest"], Tier.working.value)]
    summable = _cut_family_client(version_store, artifact_store, flow_summable=True, store=store)

    path = minted["url"] if via == "link" else _selector(cut, _query(output))
    resp = summable.get(path, follow_redirects=False)

    assert resp.status_code == 410, resp.text
    with version_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT content_version, base_content_version FROM datasets.version_record "
            "WHERE subject_id = %s AND base_content_version IS NOT NULL",
            (spec_id,),
        )
    derived = {row["content_version"]: row["base_content_version"] for row in rows}
    assert set(derived.values()) == {base}
    assert len(derived) == 2
