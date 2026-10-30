"""`POST /api/datasets/exports` prepares one exact frame version as a pinned file and records it;
`GET /api/datasets/exports/{export_id}` answers 302 to that file and renders nothing. Both run
through the app factory over live store SQL: a minted dataset version for a fetched frame, and a
frame the analysis runner stored for a derived one.
"""

from __future__ import annotations

import csv
import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import httpx
import openpyxl
import pyarrow.parquet as pq
import pytest

import datasets_api.doors.analysis as analysis_door
import datasets_api.doors.exports as exports_door
import datasets_api.exports as exports_module
from artifact_store import LiveArtifactInventory, Tier
from code_index import CodeIndex, IndexStamp
from code_index.types import PartyRow, StatVarRow
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.allowance import BUDGET_KEY, LiveAllowanceLedger
from datasets_api.analysis import AnalysisPending
from datasets_api.codes_materialise import SpecNotExportable
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.definition import SpecNotFound
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.doors.recipe import RecipeNotFound
from datasets_api.doors.refresh import RefreshRebuildFailed
from datasets_api.export_store import LiveExportStore
from datasets_api.version_store import LiveVersionLister, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from dfo_platform.caller import address_key
from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import IdentityQuarantined, LiveSpecRegistry
from spec_registry.errors import IdentityCollision
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._analysis_doubles import Stores
from ._app_factories import make_client
from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_definition_door import _BASE_URL
from .test_definition_door_currency import _FakeHeadStore
from .test_derived import _roots
from .test_derived import _stored as _stored_derived
from .test_pin_door import FakeArtifactReferenceWriter, _ledger, _spent
from .test_serve_groupby_collapse import _mint
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

# A flow dataset as DAC2a serves one: France and Germany give to Kenya.
_MEMBER = "FlowMember"
_FRANCE = "country/FRA"
_GERMANY = "country/DEU"
_KENYA = "country/KEN"
_FACET_ID = "facet1"
_PROVENANCE = "OECD_DAC2A"
_ROLE_COLUMNS = ("financingProvider", "financingRecipient")
_FLOW_IDENTITY = {
    "identity_version": "spec1",
    "shape": {
        "kind": "statvar_schema",
        "population_type": "DevelopmentFinance",
        "measured_property": "amount",
    },
    "slots": [],
    "entities": {"destination": [_KENYA]},
    "open_roles": ["origin"],
    "requested_window": {"kind": "all"},
}
_SDMX_BODY = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,financingProvider,financingRecipient,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
) + "".join(
    f"dataflow,DC:DF_OBS(1.0.0),I,{_MEMBER},{provider},{_KENYA},USD,GrantEquivalent,"
    f"NotApplicable,{_PROVENANCE},2020,{value},,{_FACET_ID}\r\n"
    for provider, value in ((_GERMANY, 900), (_FRANCE, 100))
)


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _MEMBER: {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("DevelopmentFinance",),
                "measuredProperty": ("amount",),
                "observationProperties": _ROLE_COLUMNS,
            },
            **{place: {"typeOf": ("Country",)} for place in (_FRANCE, _GERMANY, _KENYA)},
        },
        labels={_PROVENANCE: "OECD DAC2a", _KENYA: "Kenya", _FRANCE: "France", _GERMANY: "Germany"},
    )


def _sdmx_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_SDMX_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


def _coverage_service() -> FakeCoverageService:
    return FakeCoverageService(
        build_coverage_index(
            [(_MEMBER, _FACET_ID, provider, _KENYA, [2020]) for provider in (_GERMANY, _FRANCE)],
            role_names={_MEMBER: RoleNames(donor=_ROLE_COLUMNS[0], recipient=_ROLE_COLUMNS[1])},
            facet_metadata={
                _FACET_ID: FacetMetadata(
                    provenance_id=_PROVENANCE, measurement_method="GrantEquivalent", unit="USD"
                )
            },
        )
    )


def _code_index() -> CodeIndex:
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="https://fixture.example.org",
            exported_at="2026-09-15T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="sha256:fixture",
        ),
        declarations=(),
        codes=(),
        statvars=(
            StatVarRow(
                dcid=_MEMBER,
                population_type="DevelopmentFinance",
                measured_property="amount",
                stat_type=None,
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=_ROLE_COLUMNS,
                constraint_properties={},
                provenances=(_PROVENANCE,),
            ),
        ),
        parties=(
            PartyRow(dcid=_GERMANY, kind="party", datasets=(_PROVENANCE,), axes=("origin",)),
            PartyRow(dcid=_FRANCE, kind="party", datasets=(_PROVENANCE,), axes=("origin",)),
            PartyRow(dcid=_KENYA, kind="party", datasets=(_PROVENANCE,), axes=("destination",)),
        ),
    )


# A place dataset as the World Bank serves one: Kenya's life expectancy.
_PLACE_STAT_VAR = "LifeExpectancy_Person"
_PLACE_SDMX_BODY = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
) + "".join(
    f"dataflow,DC:DF_OBS(1.0.0),I,dcid:{_PLACE_STAT_VAR},{_KENYA},Years,dc-aggregate,"
    f"NotApplicable,WorldBank,{year},{value},,facet1\r\n"
    for year, value in (("2015", "61.4"), ("2016", "61.8"))
)


class _Dataset:
    """A dataset the fetched rig mints: the stores its app reads it through, and its mint."""

    def __init__(self, *, stores: Any, mint: Any) -> None:
        self.stores = stores
        self.mint = mint


def _flow_stores() -> dict[str, Any]:
    return {
        "graph": _graph(),
        "sdmx": SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        "code_index": _code_index(),
        "coverage": _coverage_service(),
    }


def _flow_mint(identity: dict[str, Any]) -> Any:
    def mint(client: Any) -> dict[str, Any]:
        spec_id = recompute_identity(identity).spec_id
        sidecar = {"schema_version": 1, "spec_id": spec_id, "identity": identity}
        return _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})

    return mint


def _place_stores() -> dict[str, Any]:
    graph = FakeMintGraphReader(
        arcs={
            _PLACE_STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "measuredProperty": ("lifeExpectancy",),
                "populationType": ("Person",),
            },
            _KENYA: {"typeOf": ("Country",)},
        },
        facets={
            (_PLACE_STAT_VAR, _KENYA): [
                Facet(
                    facet_id="facet1",
                    provenance_id="WorldBank",
                    measurement_method="dc-aggregate",
                    unit="Years",
                    obs_count=2,
                    earliest_date="2015",
                    latest_date="2016",
                )
            ]
        },
        labels={"WorldBank": "World Bank", _KENYA: "Kenya"},
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_PLACE_SDMX_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )
    return {"graph": graph, "sdmx": SdmxClient(base_url=_BASE_URL, transport=transport)}


def _place_mint(client: Any) -> dict[str, Any]:
    resp = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [f"dcid:{_PLACE_STAT_VAR}"],
                "scope": {
                    "entities": {"observationAbout": [_KENYA]},
                    "facets": {
                        "provenance": {"ref": "WorldBank"},
                        "measurementMethod": {"ref": "dc-aggregate"},
                        "unit": {"ref": "Years"},
                    },
                },
            },
            "options": {"bundle": False},
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


_DATASETS = {
    "flow": _Dataset(stores=_flow_stores, mint=_flow_mint(_FLOW_IDENTITY)),
    # A total over each role's members: the identity binds and opens no role.
    "flow-binding-no-role": _Dataset(
        stores=_flow_stores,
        mint=_flow_mint(
            {k: v for k, v in _FLOW_IDENTITY.items() if k != "open_roles"} | {"entities": {}}
        ),
    ),
    "place": _Dataset(stores=_place_stores, mint=_place_mint),
}

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

    def __init__(
        self, kind: str, monkeypatch: pytest.MonkeyPatch, *, ledger=None, dataset: str = "flow"
    ) -> None:
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
        self.spec_store = spec_store = FakeDatasetStore()
        self.client = make_client(
            registry=LiveSpecRegistry(store=spec_store),
            spec_reader=LiveSpecReader(store=spec_store),
            snapshot_reader=LiveSnapshotReader(store=self.store),
            version_writer=LiveVersionRecordWriter(store=self.store),
            head_writer=_FakeHeadStore(),
            artifact_store=self.artifact_store,
            reference_writer=FakeArtifactReferenceWriter(),
            export_store=LiveExportStore(store=self.store),
            allowances=self.ledger,
            **_DATASETS[dataset].stores(),
        )
        minted = _DATASETS[dataset].mint(self.client)
        self.frame = {
            "spec_id": minted["definition"]["spec_id"],
            "content_version": minted["content_version"],
        }
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

    def bind_posted_id(self) -> str:
        """Register a posted id the registry binds to the minted served id, as a resolution of
        a posted definition does, and return it."""
        posted = "spec_posted"
        state = self.spec_store.state
        [definition] = [
            row for key, row in state.spec_definition.items() if key[0] == self.frame["spec_id"]
        ]
        state.spec_definition[(posted, 1)] = dict(definition)
        state.spec_alias.append((posted, self.frame["spec_id"]))
        return posted

    def revoke(self, subject_id: str) -> None:
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO datasets.version_revocation "
                "(subject_id, content_version, reason, trace_id, revoked_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (subject_id, self.frame["content_version"], "taken down", None, "2026-10-30"),
            )

    def recorded_subject(self, export_id: str) -> str:
        with self.store.transaction() as tx:
            row = tx.fetch_one(
                "SELECT subject_id FROM datasets.export WHERE export_id = %s", (export_id,)
            )
        assert row is not None
        return str(row["subject_id"])

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
    # Only the link signs a URL, so the prepare answer carries none.
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


def test_a_posted_id_records_the_served_id_and_answers_the_frame_it_named(
    monkeypatch: pytest.MonkeyPatch,
):
    """The record's version is keyed by the id `version_record` holds, the served one, which the
    row's foreign key needs. The facts and the export id keep the id the request named."""
    rig = _Rig("fetched", monkeypatch)
    served = rig.frame["spec_id"]
    posted = rig.bind_posted_id()
    frame = {**rig.frame, "spec_id": posted}

    resp = rig.prepare("csv", frame=frame)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["frame"] == frame
    assert body["export_id"] == _export_id(frame, "csv", bundle=False)
    assert rig.recorded_subject(body["export_id"]) == served
    assert rig.link(body["export_id"]).status_code == 302


@pytest.mark.parametrize("revoked", ["served", "posted"])
def test_a_version_revoked_after_its_export_refuses_the_link_and_signs_nothing(
    revoked: str, monkeypatch: pytest.MonkeyPatch
):
    rig = _Rig("fetched", monkeypatch)
    served = rig.frame["spec_id"]
    posted = rig.bind_posted_id()
    body = rig.prepare("csv", frame={**rig.frame, "spec_id": posted}).json()
    rig.revoke(served if revoked == "served" else posted)

    resp = rig.link(body["export_id"])

    assert resp.status_code == 410, resp.text
    assert resp.json()["code"] == "snapshot_revoked"
    assert rig.signings == []


def test_a_revoked_derived_frame_refuses_its_export_link(monkeypatch: pytest.MonkeyPatch):
    rig = _Rig("derived", monkeypatch)
    body = rig.prepare("csv").json()
    rig.revoke(rig.frame["spec_id"])

    resp = rig.link(body["export_id"])

    assert resp.status_code == 410, resp.text
    assert resp.json()["code"] == "snapshot_revoked"
    assert rig.signings == []


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
        {"frame": {"spec_id": "ops_" + "d" * 32}},
        {"frame": {"spec_id": "spec_1"}},
    ],
    ids=["unknown-id-kind", "unknown-format", "unknown-key", "bare-derived", "bare-short"],
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


def _resolving(monkeypatch: pytest.MonkeyPatch, answer: Any) -> list[str]:
    """Answer every bare id's resolve with `answer`, a pair or an exception to raise, and
    return the ids asked about."""
    asked: list[str] = []

    def newest(spec_id: str, **_: object) -> Any:
        asked.append(spec_id)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(**answer)

    monkeypatch.setattr(exports_module, "newest_spec_version", newest)
    return asked


def test_a_bare_spec_id_exports_the_newest_pair_as_a_request_naming_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One export, id, file and sha256, whichever names the version."""
    rig = _Rig("fetched", monkeypatch)
    asked = _resolving(monkeypatch, rig.frame)

    bare = rig.prepare("xlsx", frame={"spec_id": rig.frame["spec_id"]})
    monkeypatch.setattr(exports_module, "prepare_export", _forbidden)
    pair = rig.prepare("xlsx")

    assert bare.status_code == 200, bare.text
    assert bare.json() == pair.json()
    assert bare.json()["frame"] == rig.frame
    assert bare.json()["export_id"] == _export_id(rig.frame, "xlsx", bundle=False)
    assert asked == [rig.frame["spec_id"]]


_BARE = "spec_" + "c" * 32


@pytest.mark.parametrize(
    ("answer", "status", "code", "retryable"),
    [
        pytest.param(
            AnalysisPending(operations=("op_1",), retry_after_s=2.0),
            503,
            "service_unavailable",
            True,
            id="building",
        ),
        pytest.param(MintBackpressure(), 503, "service_unavailable", True, id="backpressure"),
        pytest.param(
            IdentityQuarantined(subject_id=_BARE, id_kind="spec"),
            None,
            "identity_quarantined",
            False,
            id="quarantined",
        ),
        pytest.param(SpecNotFound(id=_BARE), 404, "spec_not_found", False, id="unregistered"),
        pytest.param(
            RefreshRebuildFailed(operation_id="op_1", refusal_code="over_ceiling"),
            500,
            "internal_error",
            False,
            id="rebuild-failed",
        ),
        pytest.param(
            SpecNotExportable(spec_id=_BARE, detail="exemplar"),
            409,
            "spec_not_exportable",
            False,
            id="registered-rebuild-4xx",
        ),
        pytest.param(
            IdentityCollision(subject_id=_BARE),
            500,
            "identity_collision",
            False,
            id="registered-rebuild-5xx",
        ),
        pytest.param(
            RecipeNotFound(spec_id=_BARE, content_version="cv_" + "d" * 32),
            404,
            "snapshot_not_found",
            False,
            id="code-exports-does-not-register",
        ),
    ],
)
def test_a_bare_spec_id_that_cannot_be_resolved_refuses_with_a_code_exports_registers(
    monkeypatch: pytest.MonkeyPatch,
    answer: Exception,
    status: int | None,
    code: str,
    retryable: bool,
) -> None:
    rig = _Rig("fetched", monkeypatch)
    _resolving(monkeypatch, answer)
    monkeypatch.setattr(exports_module, "prepare_export", _forbidden)

    resp = rig.prepare(frame={"spec_id": _BARE})

    if status is not None:
        assert resp.status_code == status, resp.text
    assert resp.json()["code"] == code, resp.text
    assert resp.json()["retryable"] is retryable


def _table(data: bytes, fmt: str) -> list[dict[str, Any]]:
    """The rows of an unbundled export file, each as its header's cells."""
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(data.decode())))
    if fmt == "xlsx":
        sheet = openpyxl.load_workbook(io.BytesIO(data), read_only=True).worksheets[0]
        header, *rows = sheet.iter_rows(values_only=True)
        return [dict(zip(header, row, strict=True)) for row in rows]
    return pq.read_table(io.BytesIO(data)).to_pylist()


@pytest.mark.parametrize("fmt", ["csv", "xlsx", "parquet"])
@pytest.mark.parametrize(
    ("dataset", "roles", "kinds"),
    [
        pytest.param("flow", _ROLE_COLUMNS, True, id="flow"),
        pytest.param("flow-binding-no-role", _ROLE_COLUMNS, True, id="flow-binding-no-role"),
        pytest.param("place", ("observationAbout",), False, id="place"),
    ],
)
def test_a_fetched_frame_exports_the_enrichment_a_client_frame_holds(
    fmt: str, dataset: str, roles: tuple[str, ...], kinds: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each role's `_type` cells, which the Spec root's default enrichment of the roles the
    version holds adds, beside the long layout's own `_label` cells and a party's `_kind`."""
    rig = _Rig("fetched", monkeypatch, dataset=dataset)

    resp = rig.prepare(fmt)

    assert resp.status_code == 200, resp.text
    rows = _table(rig.pinned_bytes(resp.json()["sha256"]), fmt)
    assert len(rows) == resp.json()["rows"] == 2
    for role in roles:
        assert {f"{role}_label", f"{role}_type"} <= rows[0].keys()
        assert (f"{role}_kind" in rows[0]) is kinds
        assert {row[f"{role}_type"] for row in rows} == {"Country"}
