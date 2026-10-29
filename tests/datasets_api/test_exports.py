"""`POST /api/datasets/exports` prepares one exact frame version as a pinned file and records it;
`GET /api/datasets/exports/{export_id}` answers 302 to that file and never renders. Both run
through the app factory over live store SQL: a minted dataset version for a fetched frame, and a
frame the analysis runner stored for a derived one.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

import datasets_api.doors.analysis as analysis_door
import datasets_api.doors.exports as exports_door
import datasets_api.exports as exports_module
from artifact_store import LiveArtifactInventory, Tier
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.allowance import BUDGET_KEY, LiveAllowanceLedger
from datasets_api.config import DatasetsBuildSettings
from datasets_api.export_store import LiveExportStore
from datasets_api.version_store import LiveVersionLister, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from dfo_platform.caller import address_key
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader

from ._analysis_doubles import Stores
from ._app_factories import make_client
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_definition_door import _BASE_URL
from .test_definition_door_currency import _FakeHeadStore
from .test_derived import _roots
from .test_derived import _stored as _stored_derived
from .test_pin_door import FakeArtifactReferenceWriter, _ledger, _spent
from .test_serve_groupby_collapse import (
    _code_index,
    _coverage_service,
    _graph,
    _mint,
    _sdmx_transport,
    _spec_id_and_sidecar,
)
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

_HMAC_KEY = "test-caller-key"
_CLIENT_ADDRESS = "203.0.113.7"
_CALLER_HEADERS = {"x-forwarded-for": f"198.51.100.1, {_CLIENT_ADDRESS}"}
_FORMATS = [(fmt, bundle) for fmt in ("csv", "xlsx", "parquet") for bundle in (False, True)]


@pytest.fixture(autouse=True)
def caller_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure the caller digest key, and return the key of a caller at `_CLIENT_ADDRESS`."""
    monkeypatch.setenv("DFO_CALLER_HMAC_KEY", _HMAC_KEY)
    return address_key(_CLIENT_ADDRESS, hmac_key=_HMAC_KEY)


class _Rig:
    """A client over one SQLite-backed store, with a frame to export: a minted dataset version
    (`fetched`) or a frame the runner stored (`derived`)."""

    def __init__(self, kind: str, monkeypatch: pytest.MonkeyPatch, *, ledger=None) -> None:
        self.ledger: LiveAllowanceLedger = ledger if ledger is not None else _ledger()
        if kind == "derived":
            stores = Stores()
            member = _stored_derived(stores)
            monkeypatch.setattr(analysis_door, "LiveRootSource", lambda **_: _roots(stores))
            self.store = stores.store
            self.artifact_store = stores.artifact_store
            # The fake GCS client cannot sign, so the derived rig records signings as the fake
            # artifact store does.
            self.signings: list[tuple[str, Tier, str, Any]] = []
            monkeypatch.setattr(stores.artifact_store, "signed_location_url", self._sign)
            self.client = make_client(
                registry=LiveSpecRegistry(store=FakeDatasetStore()),
                settings=DatasetsBuildSettings.from_env(),
                artifact_store=stores.artifact_store,
                snapshot_reader=stores.snapshot_reader,
                version_writer=stores.version_writer,
                recipe_store=stores.recipe_store,
                export_store=LiveExportStore(store=stores.store),
                version_lister=LiveVersionLister(store=stores.store),
                allowances=self.ledger,
                now=lambda: stores.now,
            )
            self.frame = {
                "spec_id": member["spec"]["spec_id"],
                "content_version": member["content_version"],
            }
            return
        self.store = FakeVersionRecordDatasetStore()
        self.artifact_store = _InventoryRecordingArtifactStore(
            inventory=LiveArtifactInventory(store=self.store)
        )
        self.signings = self.artifact_store.signings
        spec_store = FakeDatasetStore()
        self.client = make_client(
            registry=LiveSpecRegistry(store=spec_store),
            spec_reader=LiveSpecReader(store=spec_store),
            snapshot_reader=LiveSnapshotReader(store=self.store),
            version_writer=LiveVersionRecordWriter(store=self.store),
            head_writer=_FakeHeadStore(),
            graph=_graph(),
            sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
            artifact_store=self.artifact_store,
            reference_writer=FakeArtifactReferenceWriter(),
            export_store=LiveExportStore(store=self.store),
            code_index=_code_index(),
            coverage=_coverage_service(),
            allowances=self.ledger,
        )
        spec_id, sidecar = _spec_id_and_sidecar()
        minted = _mint(self.client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
        self.frame = {"spec_id": spec_id, "content_version": minted["content_version"]}
        self.coverage_rows = minted["coverage"]["rows"]

    def _sign(self, digest: str, *, tier: Tier, object_name: str, ttl: Any, **_: object) -> str:
        self.signings.append((digest, tier, object_name, ttl))
        return f"fake://artifact/{tier.value}/{object_name}"

    def prepare(self, fmt: str = "csv", *, bundle: bool = False, frame=None):
        return self.client.post(
            "/api/datasets/exports",
            json={"frame": frame or self.frame, "format": fmt, "bundle": bundle},
            headers=_CALLER_HEADERS,
        )

    def link(self, export_id: str):
        return self.client.get(f"/api/datasets/exports/{export_id}", follow_redirects=False)

    def pinned_bytes(self, sha256: str) -> bytes:
        return self.artifact_store.get(artifact_digest=f"sha256:{sha256}", tier=Tier.pinned)

    def forget_exports(self) -> None:
        with self.store.transaction() as tx:
            tx.execute("DELETE FROM datasets.export")


@pytest.fixture(params=["fetched", "derived"])
def rig(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> _Rig:
    return _Rig(request.param, monkeypatch)


@pytest.mark.parametrize(("fmt", "bundle"), _FORMATS)
def test_a_prepare_pins_the_file_and_answers_facts_that_match_its_bytes(
    rig: _Rig, fmt: str, bundle: bool
):
    resp = rig.prepare(fmt, bundle=bundle)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    data = rig.pinned_bytes(body["sha256"])
    assert hashlib.sha256(data).hexdigest() == body["sha256"]
    assert body["bytes"] == len(data)
    assert body["export_id"].startswith("exp_") and len(body["export_id"]) == 36
    assert body["url_path"] == f"/api/datasets/exports/{body['export_id']}"
    assert body["frame"] == rig.frame
    assert (body["format"], body["bundle"], body["pinned"]) == (fmt, bundle, True)
    assert body["file_name"].endswith(".zip" if bundle else f".{fmt}")
    assert (body["media_type"] == "application/zip") is bundle
    assert body["rows"] > 0 and body["columns"] > 0
    # The prepare answers no signed URL; only the link signs.
    assert "fake://" not in resp.text and "pinned-bucket" not in resp.text
    assert "no-store" in resp.headers["cache-control"]


def test_every_format_of_one_frame_states_the_same_rows_and_columns(rig: _Rig):
    shapes = set()
    for fmt, bundle in _FORMATS:
        resp = rig.prepare(fmt, bundle=bundle)
        assert resp.status_code == 200, resp.text
        shapes.add((resp.json()["rows"], resp.json()["columns"]))
    assert len(shapes) == 1, shapes
    if hasattr(rig, "coverage_rows"):
        assert shapes == {(rig.coverage_rows, next(iter(shapes))[1])}


def test_the_link_redirects_to_the_pinned_file_without_rendering(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
):
    body = rig.prepare("xlsx", bundle=True).json()
    monkeypatch.setattr(exports_module, "derived_file", _forbidden)
    monkeypatch.setattr(exports_module, "render_file_location", _forbidden)

    resp = rig.link(body["export_id"])

    assert resp.status_code == 302, resp.text
    assert resp.headers["location"].startswith("fake://artifact/pinned/")
    assert "no-store" in resp.headers["cache-control"]
    [(digest, tier, _object_name, ttl)] = rig.signings
    assert (digest, tier) == (f"sha256:{body['sha256']}", Tier.pinned)
    assert ttl.total_seconds() <= 300


def test_a_repeat_prepare_answers_the_record_and_renders_and_spends_nothing(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
):
    first = rig.prepare("csv", bundle=True)
    assert first.status_code == 200, first.text
    spent = _spent(rig.ledger)
    monkeypatch.setattr(exports_door, "prepare_export", _forbidden)

    again = rig.prepare("csv", bundle=True)

    assert again.status_code == 200, again.text
    assert again.json() == first.json()
    assert _spent(rig.ledger) == spent


def test_a_new_pinned_copy_spends_one_unit_of_the_callers_pin_allowance(rig: _Rig, caller_key: str):
    assert rig.prepare("parquet", bundle=True).status_code == 200

    assert _spent(rig.ledger) == {
        (caller_key, "pin"): 1,
        (BUDGET_KEY, "pin_budget"): 1,
        (BUDGET_KEY, "open_pin_budget"): 1,
    }


@pytest.mark.parametrize("kind", ["fetched", "derived"])
def test_an_open_caller_past_its_pin_allowance_refuses_429_until_utc_midnight(
    kind: str, monkeypatch: pytest.MonkeyPatch
):
    rig = _Rig(kind, monkeypatch, ledger=_ledger(pin=0))

    resp = rig.prepare("csv", bundle=True)

    assert resp.status_code == 429, resp.text
    assert resp.json()["code"] == "allowance_exhausted"
    assert int(resp.headers["retry-after"]) > 0
    assert rig.link(_export_id(rig.frame, "csv", bundle=True)).status_code == 404


@pytest.mark.parametrize("kind", ["fetched", "derived"])
def test_a_re_render_of_a_forgotten_export_gives_the_same_id_and_bytes_and_spends_nothing(
    kind: str, monkeypatch: pytest.MonkeyPatch
):
    """Every render is deterministic, bundles included, so a second prepare stores the same
    bytes, which the store holds once, and its pinned copy finds them pinned already."""
    rig = _Rig(kind, monkeypatch)
    first = {(fmt, bundle): rig.prepare(fmt, bundle=bundle).json() for fmt, bundle in _FORMATS}
    spent = _spent(rig.ledger)
    rig.forget_exports()

    again = {(fmt, bundle): rig.prepare(fmt, bundle=bundle).json() for fmt, bundle in _FORMATS}

    assert again == first
    assert _spent(rig.ledger) == spent


def test_two_concurrent_prepares_of_one_request_answer_one_export(rig: _Rig, caller_key: str):
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: rig.prepare("xlsx", bundle=True), range(2)))

    assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
    assert responses[0].json() == responses[1].json()
    assert _spent(rig.ledger)[(caller_key, "pin")] == 1


def test_an_export_id_is_the_hash_of_the_request_alone(rig: _Rig):
    body = rig.prepare("parquet").json()

    assert body["export_id"] == _export_id(rig.frame, "parquet", bundle=False)


def test_a_csv_or_xlsx_over_the_row_ceiling_refuses_over_ceiling_and_parquet_serves(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
):
    # The derived frame holds one row, so a ceiling of none refuses both frames.
    monkeypatch.setattr(exports_module, "EXPORT_MAX_ROWS", 0)

    for fmt in ("csv", "xlsx"):
        resp = rig.prepare(fmt)
        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert (problem["code"], problem["ceiling_kind"], problem["cap"]) == (
            "over_ceiling",
            "rows",
            0,
        )
        assert problem["count"] > 0
    assert rig.prepare("parquet").status_code == 200


def test_a_derived_frame_without_a_recipe_refuses_provenance_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    rig = _Rig("derived", monkeypatch)
    with rig.store.transaction() as tx:
        tx.execute("DELETE FROM datasets.frame_recipe")

    resp = rig.prepare("csv")

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "provenance_unavailable"
    assert _spent(rig.ledger) == {}


@pytest.mark.parametrize("kind", ["fetched", "derived"])
def test_an_unknown_version_refuses_snapshot_not_found(kind: str, monkeypatch):
    rig = _Rig(kind, monkeypatch)
    prefix = "ocv_" if kind == "derived" else "cv_"
    frame = {**rig.frame, "content_version": prefix + "0" * 32}

    resp = rig.prepare("csv", frame=frame)

    assert resp.status_code == 404, resp.text
    assert resp.json()["code"] == "snapshot_not_found"


def test_a_source_on_the_licence_deny_list_refuses_the_prepare_and_the_link(
    monkeypatch: pytest.MonkeyPatch,
):
    rig = _Rig("fetched", monkeypatch)
    prepared = rig.prepare("csv").json()
    [licence, *_] = prepared["licences"]
    monkeypatch.setattr(
        exports_module, "restricted_sources", lambda: {licence["source"]: "terms forbid it"}
    )

    fresh = rig.prepare("parquet")
    repeat = rig.prepare("csv")
    link = rig.link(prepared["export_id"])

    for resp in (fresh, repeat, link):
        assert resp.status_code == 403, resp.text
        assert resp.json()["code"] == "licence_not_redistributable"
        assert licence["source"] in resp.json()["detail"]
    assert rig.link(_export_id(rig.frame, "parquet", bundle=False)).status_code == 404


def test_the_committed_licence_deny_list_is_empty():
    exports_module.restricted_sources.cache_clear()
    assert exports_module.restricted_sources() == {}


def test_an_unknown_export_id_refuses_not_found(rig: _Rig):
    resp = rig.link("exp_" + "0" * 32)

    assert resp.status_code == 404, resp.text
    assert resp.json()["code"] == "not_found"


@pytest.mark.parametrize(
    "body",
    [
        {"frame": {"spec_id": "x_1", "content_version": "cv_1"}},
        {"frame": {"spec_id": "spec_1", "content_version": "cv_1"}, "format": "json"},
        {"frame": {"spec_id": "spec_1", "content_version": "cv_1"}, "labels": True},
    ],
    ids=["unknown-id-kind", "unknown-format", "unknown-key"],
)
def test_a_malformed_request_refuses_request_invalid(body: dict[str, Any], monkeypatch):
    rig = _Rig("fetched", monkeypatch)

    resp = rig.client.post("/api/datasets/exports", json=body, headers=_CALLER_HEADERS)

    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "request_invalid"


def _export_id(frame: dict[str, str], fmt: str, *, bundle: bool) -> str:
    return exports_module.export_id(
        exports_module.ExportRequest.model_validate(
            {"frame": frame, "format": fmt, "bundle": bundle}
        )
    )


def _forbidden(**_: object) -> object:
    raise AssertionError("a recorded export renders nothing")
