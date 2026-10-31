"""`POST /api/datasets/exports` prepares one exact frame version as a pinned file and records it;
`GET /api/datasets/exports/{export_id}` answers 302 to that file and renders nothing. Both run
through the app factory over live store SQL: a minted dataset version for a fetched frame, and a
frame the analysis runner stored for a derived one.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import threading
import time
import zipfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import datasets_api.analysis_roots as analysis_roots
import datasets_api.current as current_module
import datasets_api.doors.analysis as analysis_door
import datasets_api.doors.derived as derived_door
import datasets_api.doors.exports as exports_door
import datasets_api.exports as exports_module
from artifact_store import ArtifactNotFoundError, LiveArtifactInventory, Tier
from artifact_store.errors import ArtifactInventoryError, ArtifactWriteError
from code_index import CodeIndex, IndexStamp
from code_index.types import PartyRow, StatVarRow
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.allowance import (
    BUDGET_KEY,
    SWEEP_INTERVAL_SECONDS,
    AllowanceExhausted,
    AllowanceKind,
    CopyClaim,
    LiveAllowanceLedger,
    Spend,
)
from datasets_api.analysis import AnalysisPending
from datasets_api.codes_exports import (
    ExportOverCeiling,
    LicenceNotRedistributable,
)
from datasets_api.codes_materialise import SpecNotExportable
from datasets_api.codes_snapshot import ArtifactExpired
from datasets_api.config import DatasetsBuildSettings
from datasets_api.current import StoredTraceMissingResolvedStatVars
from datasets_api.doors.definition import SpecNotFound
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.doors.recipe import RecipeNotFound
from datasets_api.doors.refresh import RefreshRebuildFailed
from datasets_api.export_store import LiveExportStore
from datasets_api.exports import BareExportFrame
from datasets_api.reconciliation import sweep_tier
from datasets_api.reference_store import LiveArtifactReferenceWriter
from datasets_api.retention_admission import PinUnitHeld, PreparingCopy
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import LiveVersionLister, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from dfo_platform.caller import CallerUnresolved, address_key
from materialise.formats import (
    ExcelIllegalCharacterError,
    ExcelLimitError,
    ParquetSizeBoundError,
)
from materialise.graph import Facet
from materialise.reshape import RollupNotSummableError
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import IdentityQuarantined, LiveSpecRegistry
from spec_registry.errors import IdentityCollision
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._analysis_doubles import AID, AID_V1, POPULATION, POPULATION_V1, Stores
from ._app_factories import make_client
from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_definition_door import _BASE_URL
from .test_definition_door_currency import _FakeHeadStore
from .test_derived import _roots
from .test_derived import _stored as _stored_derived
from .test_pin_door import _ledger, _spent
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


_BARE_FILE: dict[str, Any] = {"bundle": False}


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
    def mint(client: Any, options: dict[str, Any] = _BARE_FILE) -> dict[str, Any]:
        spec_id = recompute_identity(identity).spec_id
        sidecar = {"schema_version": 1, "spec_id": spec_id, "identity": identity}
        return _mint(client, spec_id=spec_id, sidecar=sidecar, options=options)

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


def _place_mint(client: Any, options: dict[str, Any] = _BARE_FILE) -> dict[str, Any]:
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
            "options": options,
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

_PIN_PRINCIPAL = "pin-principal"
_HMAC_KEY = "test-caller-key"
_CLIENT_ADDRESS = "203.0.113.7"
_CALLER_HEADERS = {"x-forwarded-for": f"198.51.100.1, {_CLIENT_ADDRESS}"}
_FORMATS = [(fmt, bundle) for fmt in ("csv", "xlsx", "parquet") for bundle in (False, True)]
_RENDERS = {
    "fetched": (exports_module, "render_file_location"),
    "derived": (derived_door, "render_derived_file"),
}


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
            self.spec_store = spec_store = FakeDatasetStore()
            for spec_id, content_version in _DERIVED_ROOTS:
                _store_root(self.store, spec_id, content_version)
            # The fake GCS client cannot sign, so the derived rig records signings as the fake
            # artifact store does.
            self.signings: list[tuple[str, Tier, str, Any]] = []
            monkeypatch.setattr(stores.artifact_store, "signed_location_url", self._sign)
            self.client = make_client(
                registry=LiveSpecRegistry(store=spec_store),
                spec_reader=LiveSpecReader(store=spec_store),
                settings=DatasetsBuildSettings.from_env(),
                artifact_store=stores.artifact_store,
                snapshot_reader=stores.snapshot_reader,
                version_writer=stores.version_writer,
                reference_writer=LiveArtifactReferenceWriter(store=stores.store),
                recipe_store=stores.recipe_store,
                export_store=LiveExportStore(store=stores.store),
                version_lister=LiveVersionLister(store=stores.store),
                allowances=self.ledger,
                pin_allowed_principals=frozenset({_PIN_PRINCIPAL}),
                now=lambda: stores.now,
            )
            self.spent_at = stores.now
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
            reference_writer=LiveArtifactReferenceWriter(store=self.store),
            export_store=LiveExportStore(store=self.store),
            allowances=self.ledger,
            pin_allowed_principals=frozenset({_PIN_PRINCIPAL}),
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

    def bind_posted_id(self, posted: str = "spec_posted") -> str:
        """Register a posted id the registry binds to the minted served id, as a resolution of
        a posted definition does, and return it."""
        state = self.spec_store.state
        [definition] = [
            row for key, row in state.spec_definition.items() if key[0] == self.frame["spec_id"]
        ]
        state.spec_definition[(posted, 1)] = dict(definition)
        state.spec_registry[posted] = dict(state.spec_registry[self.frame["spec_id"]])
        state.spec_alias.append((posted, self.frame["spec_id"]))
        return posted

    def revoke(self, subject_id: str, content_version: str | None = None) -> None:
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO datasets.version_revocation "
                "(subject_id, content_version, reason, trace_id, revoked_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    subject_id,
                    content_version or self.frame["content_version"],
                    "taken down",
                    None,
                    "2026-10-30",
                ),
            )

    def references(self) -> list[tuple[str, str, str]]:
        with self.store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT principal, artifact_digest, reference_kind "
                "FROM datasets.artifact_reference",
                (),
            )
        return [
            (str(row["principal"]), str(row["artifact_digest"]), str(row["reference_kind"]))
            for row in rows
        ]

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


_DERIVED_ROOTS = ((AID, AID_V1), (POPULATION, POPULATION_V1))


def _store_root(store: Any, spec_id: str, content_version: str) -> None:
    """Record the root dataset version a derived frame's recipe names, with a citation."""
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record (subject_id, subject_kind, content_version, "
            "schema_version, immutable_body, recorded_at, prepared_at, definition, coverage, "
            "citation, trace, warnings) "
            "VALUES (%s, 'spec', %s, 1, '{}', %s, %s, '{}', '{}', %s, '{}', '[]')",
            (
                spec_id,
                content_version,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                {"text": f"{spec_id}.", "sources": [{"ref": f"dcid:{spec_id}", "label": spec_id}]},
            ),
        )


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


def test_a_prepare_copies_from_the_location_it_holds_without_reading_it_again(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
):
    location = rig.artifact_store.location
    read: list[Tier] = []

    def recorded(digest: str, *, tier: Tier) -> Any:
        read.append(tier)
        return location(digest, tier=tier)

    monkeypatch.setattr(rig.artifact_store, "location", recorded)

    assert rig.prepare("csv").status_code == 200
    assert Tier.working not in read


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
    monkeypatch.setattr(*_RENDERS["derived"], _forbidden)
    monkeypatch.setattr(*_RENDERS["fetched"], _forbidden)

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
@pytest.mark.parametrize(
    ("spent", "status", "code", "retryable"),
    [
        ({"pin": 0}, 429, "allowance_exhausted", False),
        ({"pin": 0, "pin_budget": 0}, 503, "service_unavailable", True),
        ({"pin": 0, "open_pin_budget": 0}, 503, "service_unavailable", True),
        ({"pin_budget": 0}, 503, "service_unavailable", True),
        ({"open_pin_budget": 0}, 503, "service_unavailable", True),
    ],
    ids=[
        "allowance",
        "allowance-and-pin-budget",
        "allowance-and-open-budget",
        "pin-budget",
        "open-budget",
    ],
)
def test_a_caller_with_a_unit_spent_refuses_as_the_copy_would_before_any_render(
    kind: str,
    spent: dict[str, int],
    status: int,
    code: str,
    retryable: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """Each refusal holds until UTC midnight, and is the one spending the units would raise."""
    rig = _Rig(kind, monkeypatch, ledger=_ledger(**spent))
    monkeypatch.setattr(*_RENDERS[kind], _forbidden)

    resp = rig.prepare("csv", bundle=True)

    assert resp.status_code == status, resp.text
    assert (resp.json()["code"], resp.json()["retryable"]) == (code, retryable)
    assert int(resp.headers["retry-after"]) > 0
    assert _spent(rig.ledger) == {}
    assert rig.link(_export_id(rig.frame, "csv", bundle=True)).status_code == 404


@pytest.mark.parametrize("kind", ["fetched", "derived"])
def test_a_spent_caller_refuses_a_file_that_must_render_even_when_its_bytes_are_pinned(
    kind: str, monkeypatch: pytest.MonkeyPatch
):
    """Only a render names the digest, and a spent caller gets none, so a file another export or
    a pin holds refuses here as any new file does."""
    rig = _Rig(kind, monkeypatch, ledger=_ledger(pin=1))
    assert rig.prepare("csv").status_code == 200
    rig.forget_exports()
    monkeypatch.setattr(exports_module, "listed_file_location", lambda *_, **__: None)
    monkeypatch.setattr(*_RENDERS[kind], _forbidden)

    resp = rig.prepare("csv")

    assert (resp.status_code, resp.json()["code"]) == (429, "allowance_exhausted"), resp.text


def test_a_file_pinned_already_asks_for_no_copy(monkeypatch: pytest.MonkeyPatch):
    """The first prepare spends the caller's last unit and lists the file. The second reads it
    from the listing, finds the pinned copy first, and so spends and copies nothing."""
    rig = _Rig("fetched", monkeypatch, ledger=_ledger(pin=1))
    first = rig.prepare("csv")
    assert first.status_code == 200, first.text
    spent = _spent(rig.ledger)
    rig.forget_exports()
    monkeypatch.setattr(PreparingCopy, "copy", _forbidden)
    monkeypatch.setattr(*_RENDERS["fetched"], _forbidden)

    again = rig.prepare("csv")

    assert again.status_code == 200, again.text
    assert again.json() == first.json()
    assert _spent(rig.ledger) == spent


def test_an_invalid_request_writes_no_claim_and_a_second_does_not_wait_for_it(
    monkeypatch: pytest.MonkeyPatch,
):
    rig = _Rig("fetched", monkeypatch)
    monkeypatch.setattr(LiveAllowanceLedger, "claim", _forbidden)

    missing = rig.prepare("csv", frame={**rig.frame, "content_version": "missing"})

    assert missing.status_code == 404, missing.text


def test_a_first_prepare_sheds_at_a_full_pool_at_once(monkeypatch: pytest.MonkeyPatch):
    """Only a call that waited on a prepare queues for a permit, so overload still sheds."""
    monkeypatch.setenv("DATASETS_WORK_MAX_CONCURRENCY", "1")
    rig = _Rig("derived", monkeypatch)
    render = _HeldRender("derived", monkeypatch, results=[None])

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(rig.prepare, "csv")
        assert render.entered.wait(10)
        try:
            shed = pool.submit(rig.prepare, "xlsx").result(timeout=5)
        finally:
            render.release.set()
        assert leader.result().status_code == 200

    assert (shed.status_code, shed.headers["retry-after"]) == (503, "1"), shed.text


@pytest.mark.parametrize("waited", [True, False])
def test_only_a_call_that_waited_reads_the_claim_before_the_checks(waited: bool):
    """A waiter that a live claim still stands before returns at once. A first call checks first,
    so an invalid request writes no claim."""
    frame = {"spec_id": "spec_a", "content_version": "cv1"}
    request = exports_module.ExportRequest.model_validate({"frame": frame})
    now = datetime(2026, 6, 1, 12, tzinfo=UTC)
    ledger = _ledger()
    ledger.claim(_export_id(frame, "csv", bundle=False), now=now, expires_at=now + timedelta(1))
    checked: list[str] = []

    def check(*_: object, **__: object) -> Any:
        checked.append("checked")
        raise ExportOverCeiling(format="csv", count=2, cap=1)

    current = SimpleNamespace(allowances=ledger, now=lambda: now)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(exports_module, "_prepare_dataset", check)
        if waited:
            answer = exports_module.prepare_export(
                request=request, caller=None, waited=True, current=current, deps=None
            )
            assert answer is None
        else:
            with pytest.raises(ExportOverCeiling):
                exports_module.prepare_export(
                    request=request, caller=None, current=current, deps=None
                )

    assert checked == ([] if waited else ["checked"])


def test_a_call_that_lost_the_claim_waits_on_it_and_answers_the_record(
    monkeypatch: pytest.MonkeyPatch,
):
    rig = _Rig("fetched", monkeypatch)
    first = rig.prepare("csv")
    rig.forget_exports()
    real = exports_door.prepare_export
    calls: list[bool] = []

    def lost_once(**kwargs: Any) -> Any:
        calls.append(kwargs["waited"])
        if len(calls) == 1:
            return None
        return real(**kwargs)

    monkeypatch.setattr(exports_door, "prepare_export", lost_once)

    again = rig.prepare("csv")

    assert again.status_code == 200, again.text
    assert again.json() == first.json()
    assert calls == [False, True]


def test_a_repeat_prepare_answers_its_record_with_one_read(monkeypatch: pytest.MonkeyPatch):
    rig = _Rig("fetched", monkeypatch)
    first = rig.prepare("csv")
    monkeypatch.setattr(LiveAllowanceLedger, "read_claim", _forbidden)

    again = rig.prepare("csv")

    assert again.status_code == 200, again.text
    assert again.json() == first.json()


@pytest.mark.parametrize("kind", ["fetched", "derived"])
def test_a_holder_of_the_pin_grant_has_no_allowance_to_spend(
    kind: str, monkeypatch: pytest.MonkeyPatch
):
    rig = _Rig(kind, monkeypatch, ledger=_ledger(pin=0))

    resp = rig.client.post(
        "/api/datasets/exports",
        json={"frame": rig.frame, "format": "csv", "bundle": False},
        headers={"x-dfo-principal": _PIN_PRINCIPAL},
    )

    assert resp.status_code == 200, resp.text
    assert _spent(rig.ledger) == {(BUDGET_KEY, "pin_budget"): 1}


@pytest.mark.parametrize("kind", ["fetched", "derived"])
def test_a_holder_of_the_pin_grant_refuses_a_spent_budget_before_any_render(
    kind: str, monkeypatch: pytest.MonkeyPatch
):
    rig = _Rig(kind, monkeypatch, ledger=_ledger(pin_budget=0))
    monkeypatch.setattr(*_RENDERS[kind], _forbidden)

    resp = rig.client.post(
        "/api/datasets/exports",
        json={"frame": rig.frame, "format": "csv", "bundle": True},
        headers={"x-dfo-principal": _PIN_PRINCIPAL},
    )

    assert resp.status_code == 503, resp.text
    assert int(resp.headers["retry-after"]) > 0


def _failing_once(monkeypatch: pytest.MonkeyPatch, owner: Any, name: str, error: Exception) -> None:
    """Make `owner.name` raise `error` on its first call and run as usual after."""
    real = getattr(owner, name)
    left = [error]

    def flaky(*args: Any, **kwargs: Any) -> Any:
        if left:
            raise left.pop()
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, flaky)


@pytest.mark.parametrize("kind", ["fetched", "derived"])
@pytest.mark.parametrize(
    "units", [{"pin": 1}, {"pin_budget": 1}, {"open_pin_budget": 1}], ids=list("abc")
)
@pytest.mark.parametrize(
    ("owner", "name"),
    [
        (LiveArtifactReferenceWriter, "record_reference_unless_revoked"),
        (LiveExportStore, "write_export"),
    ],
    ids=["reference", "record"],
)
def test_a_retry_finds_the_copy_its_last_unit_paid_for(
    kind: str, units: dict[str, int], owner: Any, name: str, monkeypatch: pytest.MonkeyPatch
):
    """The copy stands and a write after it fails. The claim keeps the units and the digest, so
    the retry, with no unit left, answers the export and spends nothing more."""
    rig = _Rig(kind, monkeypatch, ledger=_ledger(**units))
    _failing_once(monkeypatch, owner, name, DatasetsStoreUnavailable("write failed"))

    first = rig.prepare("csv")
    spent = _spent(rig.ledger)
    retry = rig.prepare("csv")

    assert first.status_code == 503, first.text
    assert retry.status_code == 200, retry.text
    assert _spent(rig.ledger) == spent
    assert rig.pinned_bytes(retry.json()["sha256"])
    assert rig.ledger.read_claim(retry.json()["export_id"]) is None


@pytest.mark.parametrize("kind", ["fetched", "derived"])
def test_a_copy_that_landed_and_raised_ends_with_one_set_of_units(
    kind: str, monkeypatch: pytest.MonkeyPatch, caller_key: str
):
    """Only gone working bytes prove nothing landed, so the first attempt keeps its units, and
    its retry spends none. The derived rig's inventory goes without the row, as a live one does."""
    rig = _Rig(kind, monkeypatch)
    copy = rig.artifact_store.copy
    left = [ArtifactWriteError(object_name="o", digest="d")]

    def landed_and_raised(digest: str, **kwargs: Any) -> Any:
        copied = copy(digest, **kwargs)
        if left:
            raise left.pop()
        return copied

    monkeypatch.setattr(rig.artifact_store, "copy", landed_and_raised)

    assert rig.prepare("csv").status_code >= 500
    retry = rig.prepare("csv")

    assert retry.status_code == 200, retry.text
    assert _spent(rig.ledger) == {
        (caller_key, "pin"): 1,
        (BUDGET_KEY, "pin_budget"): 1,
        (BUDGET_KEY, "open_pin_budget"): 1,
    }


def test_a_copy_that_landed_with_no_inventory_row_ends_with_one_set_of_units(
    monkeypatch: pytest.MonkeyPatch, caller_key: str
):
    rig = _Rig("derived", monkeypatch)
    record = LiveArtifactInventory.record
    left = [1]

    def flaky(self: Any, *, digest: str, tier: Tier, **kwargs: Any) -> Any:
        if tier == Tier.pinned and left:
            left.pop()
            raise ArtifactInventoryError(digest=digest, tier=tier)
        return record(self, digest=digest, tier=tier, **kwargs)

    monkeypatch.setattr(LiveArtifactInventory, "record", flaky)

    assert rig.prepare("csv").status_code >= 500
    retry = rig.prepare("csv")

    assert retry.status_code == 200, retry.text
    assert _spent(rig.ledger)[(caller_key, "pin")] == 1


def test_the_loser_of_the_last_unit_answers_the_copy_the_winner_landed(
    monkeypatch: pytest.MonkeyPatch,
):
    """A listed dataset file spends at its copy, so only it can lose a spend there. A prepare
    whose working bytes were gone leaves the file listed and unpinned, with its units back."""
    rig = _Rig("fetched", monkeypatch)
    _failing_once(
        monkeypatch, PreparingCopy, "copy", ArtifactNotFoundError(digest="d", tier=Tier.working)
    )
    assert rig.prepare("csv").status_code == 410
    assert _spent(rig.ledger) == {}
    now = datetime.now(UTC)

    def lost_the_spend(copy: PreparingCopy) -> None:
        for digest in rig.artifact_store.list(tier=Tier.working):
            rig.artifact_store.copy(digest, from_tier=Tier.working, to_tier=Tier.pinned)
        raise AllowanceExhausted(kind=AllowanceKind.pin, limit=1, now=now)

    monkeypatch.setattr(PreparingCopy, "hold", lost_the_spend)

    resp = rig.prepare("csv")

    assert resp.status_code == 200, resp.text
    assert rig.pinned_bytes(resp.json()["sha256"])


@pytest.mark.parametrize("kind", ["fetched", "derived"])
def test_every_export_holds_a_reference_the_sweep_counts(
    kind: str, monkeypatch: pytest.MonkeyPatch
):
    """A derived frame's file is linked to no version, and a reference needs no link."""
    rig = _Rig(kind, monkeypatch)
    bodies = [rig.prepare(fmt, bundle=bundle).json() for fmt, bundle in _FORMATS]

    assert sorted(rig.references()) == sorted(
        (f"export:{body['export_id']}", f"sha256:{body['sha256']}", "export") for body in bodies
    )
    report = sweep_tier(
        store=rig.store, artifact_store=rig.artifact_store, tier=Tier.pinned, now=datetime.now(UTC)
    )
    assert report.unreferenced_pinned_digests == frozenset()


def test_a_revocation_during_the_copy_refuses_the_export_and_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    rig = _Rig("fetched", monkeypatch)
    copy = PreparingCopy.copy

    def revoked_meanwhile(*args: Any, **kwargs: Any) -> Any:
        rig.revoke(rig.frame["spec_id"])
        return copy(*args, **kwargs)

    monkeypatch.setattr(PreparingCopy, "copy", revoked_meanwhile)

    resp = rig.prepare("csv")

    assert resp.status_code == 410, resp.text
    assert resp.json()["code"] == "snapshot_revoked"
    assert rig.references() == []
    assert rig.link(_export_id(rig.frame, "csv", bundle=False)).status_code == 404


def test_the_pinned_file_is_read_and_measured_inside_one_slot_and_stays_there(
    monkeypatch: pytest.MonkeyPatch,
):
    rig = _Rig("fetched", monkeypatch)
    read = rig.artifact_store.get
    held: set[int] = set()
    held_at_release: list[int] = []

    class _Bytes(bytes):
        def __del__(self) -> None:
            held.discard(id(self))

    def get(*, artifact_digest: str, tier: Tier) -> bytes:
        data = read(artifact_digest=artifact_digest, tier=tier)
        if tier != Tier.pinned:
            return data
        tracked = _Bytes(data)
        held.add(id(tracked))
        return tracked

    slot = exports_module.bounded_materialise_slot

    @contextmanager
    def watched(*args: Any, **kwargs: Any) -> Iterator[None]:
        with slot(*args, **kwargs):
            yield
            held_at_release.append(len(held))

    monkeypatch.setattr(rig.artifact_store, "get", get)
    monkeypatch.setattr(exports_module, "bounded_materialise_slot", watched)

    assert rig.prepare("csv").status_code == 200

    assert held_at_release == [0]


def test_a_render_linked_incomplete_refuses_retryably_and_pins_and_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    """An export is kept for good, so a bundle whose documentation read failed is never pinned.
    A retry once the read answers renders the complete file."""
    rig = _Rig("fetched", monkeypatch)

    with monkeypatch.context() as graph:
        graph.setattr(
            current_module,
            "bundle_citations",
            lambda citation, **_: (citation.sources_by_ref(), False),
        )
        refused = rig.prepare("csv", bundle=True)

    assert refused.status_code == 503, refused.text
    assert (refused.json()["code"], refused.json()["retryable"]) == ("service_unavailable", True)
    assert _spent(rig.ledger) == {}
    assert rig.references() == []
    assert rig.link(_export_id(rig.frame, "csv", bundle=True)).status_code == 404

    retried = rig.prepare("csv", bundle=True)

    assert retried.status_code == 200, retried.text
    assert rig.pinned_bytes(retried.json()["sha256"])


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


class _HeldRender:
    """Stands in for the render on one rig: `results` says how each call ends, each call holds
    until `release`, and `parked` is set once a concurrent prepare waits for the holder."""

    def __init__(self, kind: str, monkeypatch: pytest.MonkeyPatch, *, results: list[Any]) -> None:
        self.calls = 0
        self.entered = threading.Event()
        self.parked = threading.Event()
        self.release = threading.Event()
        self._results = results
        self._render = getattr(*_RENDERS[kind])
        monkeypatch.setattr(*_RENDERS[kind], self)
        read_claim = LiveAllowanceLedger.read_claim

        def parked_read(ledger: LiveAllowanceLedger, key: str) -> Any:
            claim = read_claim(ledger, key)
            if claim is not None:
                self.parked.set()
            return claim

        monkeypatch.setattr(LiveAllowanceLedger, "read_claim", parked_read)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.entered.set()
        assert self.release.wait(10)
        result = self._results.pop(0)
        if result is not None:
            raise result
        return self._render(*args, **kwargs)


@pytest.mark.parametrize("kind", ["fetched", "derived"])
def test_concurrent_prepares_of_one_id_render_once_and_answer_the_record(
    kind: str, monkeypatch: pytest.MonkeyPatch, caller_key: str
):
    rig = _Rig(kind, monkeypatch)
    render = _HeldRender(kind, monkeypatch, results=[None])

    with ThreadPoolExecutor(max_workers=3) as pool:
        leader = pool.submit(rig.prepare, "csv")
        assert render.entered.wait(10)
        followers = [pool.submit(rig.prepare, "csv") for _ in range(2)]
        assert render.parked.wait(10)
        render.release.set()
        responses = [leader.result(), *(follower.result() for follower in followers)]

    assert [r.status_code for r in responses] == [200, 200, 200], [r.text for r in responses]
    assert responses[0].json() == responses[1].json() == responses[2].json()
    assert render.calls == 1
    assert _spent(rig.ledger)[(caller_key, "pin")] == 1


@pytest.mark.parametrize("kind", ["fetched", "derived"])
def test_a_waiter_runs_its_own_prepare_after_the_one_it_waited_on_failed(
    kind: str, monkeypatch: pytest.MonkeyPatch, caller_key: str
):
    rig = _Rig(kind, monkeypatch)
    failure = ArtifactExpired(id=rig.frame["spec_id"], content_version=rig.frame["content_version"])
    render = _HeldRender(kind, monkeypatch, results=[failure, None])

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(rig.prepare, "csv")
        assert render.entered.wait(10)
        waiter = pool.submit(rig.prepare, "csv")
        assert render.parked.wait(10)
        render.release.set()
        failed, answered = leader.result(), waiter.result()

    assert failed.json()["code"] == "artifact_expired", failed.text
    assert answered.status_code == 200, answered.text
    assert render.calls == 2
    assert _spent(rig.ledger)[(caller_key, "pin")] == 1


@pytest.mark.parametrize("kind", ["fetched", "derived"])
@pytest.mark.parametrize(
    ("refusal", "code"),
    [
        (ExportOverCeiling(format="xlsx", count=2, cap=1), "over_ceiling"),
        (ExcelLimitError("too wide"), "excel_limit"),
        (ExcelIllegalCharacterError("bad cell", row=1, column=1), "excel_illegal_character"),
        (ParquetSizeBoundError("too large"), "parquet_size_bound"),
        (RollupNotSummableError("no", axis="a", precondition="b"), "rollup_not_summable"),
        (
            StoredTraceMissingResolvedStatVars(subject_id="s", content_version="c"),
            "internal_error",
        ),
    ],
    ids=["over-ceiling", "excel-limit", "excel-character", "parquet", "rollup", "stored-trace"],
)
def test_a_waiter_on_a_prepare_that_refused_prepares_in_turn_and_answers_the_same_refusal(
    kind: str,
    refusal: Exception,
    code: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """The request names the file, and the file decides the refusal, so each render gives it, and
    the units go back."""
    rig = _Rig(kind, monkeypatch)
    render = _HeldRender(kind, monkeypatch, results=[refusal, refusal])

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(rig.prepare, "xlsx")
        assert render.entered.wait(10)
        waiter = pool.submit(rig.prepare, "xlsx")
        assert render.parked.wait(10)
        render.release.set()
        responses = [leader.result(), waiter.result()]

    assert [r.json()["code"] for r in responses] == [code] * 2, [r.text for r in responses]
    assert len({(r.status_code, r.json()["detail"]) for r in responses}) == 1
    assert render.calls == 2
    assert _spent(rig.ledger) == {}


@pytest.mark.parametrize("kind", ["fetched", "derived"])
def test_a_burst_from_a_caller_with_one_unit_left_renders_one_file(
    kind: str, monkeypatch: pytest.MonkeyPatch, caller_key: str
):
    """The first prepare holds the unit through its render, so the rest refuse at once."""
    rig = _Rig(kind, monkeypatch, ledger=_ledger(pin=1))
    render = _HeldRender(kind, monkeypatch, results=[None])

    with ThreadPoolExecutor(max_workers=4) as pool:
        first = pool.submit(rig.prepare, "xlsx")
        assert render.entered.wait(10)
        try:
            burst = [
                pool.submit(rig.prepare, fmt, bundle=bundle).result(timeout=5)
                for fmt, bundle in (("xlsx", True), ("parquet", True), ("csv", True))
            ]
        finally:
            render.release.set()
        answered = first.result()

    assert [r.status_code for r in burst] == [503, 503, 503], [r.text for r in burst]
    assert {r.json()["retryable"] for r in burst} == {True}
    assert {r.headers["retry-after"] for r in burst} == {"10"}
    assert answered.status_code == 200, answered.text
    assert render.calls == 1
    assert _spent(rig.ledger)[(caller_key, "pin")] == 1
    assert rig.prepare("parquet").status_code == 429


def test_a_prepare_another_instance_holds_is_waited_for_and_answers_its_record(
    monkeypatch: pytest.MonkeyPatch,
):
    """The claim row alone says a prepare runs, so this process has no flight to look at."""
    rig = _Rig("fetched", monkeypatch)
    first = rig.prepare("csv")
    rig.forget_exports()
    export = first.json()["export_id"]
    now = datetime.now(UTC)
    claim = rig.ledger.claim(export, now=now, expires_at=now + timedelta(minutes=5))
    assert claim is not None

    def landed() -> None:
        time.sleep(0.4)
        with rig.store.transaction() as tx:
            tx.execute(
                "INSERT INTO datasets.export (export_id, subject_id, content_version, "
                "artifact_digest, bucket, object_name, body, recorded_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    export,
                    rig.frame["spec_id"],
                    rig.frame["content_version"],
                    f"sha256:{first.json()['sha256']}",
                    "bucket",
                    "object",
                    first.json(),
                    now,
                ),
            )
        rig.ledger.close_claim(claim)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(landed)
        answered = rig.prepare("csv")

    assert answered.status_code == 200, answered.text
    assert answered.json() == first.json()
    assert "rebuild_wait" in answered.headers["server-timing"]


def _dead_prepare(rig: _Rig, spends: list[Spend]) -> CopyClaim:
    """A held claim of the rig's csv export, whose lease has passed."""
    export = _export_id(rig.frame, "csv", bundle=False)
    past = datetime(2020, 1, 1, tzinfo=UTC)
    dead = rig.ledger.claim(export, now=past, expires_at=past)
    assert dead is not None
    held = rig.ledger.hold_claim(dead, spends, rig.spent_at)
    assert held is not None
    return held


def test_a_prepare_that_died_before_a_copy_refunds_as_its_retry_takes_over(
    monkeypatch: pytest.MonkeyPatch, caller_key: str
):
    """The last unit comes back, so the retry spends for itself and answers."""
    rig = _Rig("derived", monkeypatch, ledger=_ledger(pin=1))
    held = _dead_prepare(rig, [Spend(caller_key, AllowanceKind.pin)])

    retry = rig.prepare("csv")

    assert retry.status_code == 200, retry.text
    assert rig.ledger.read_claim(held.key) is None
    assert _spent(rig.ledger) == {
        (BUDGET_KEY, "pin_budget"): 1,
        (BUDGET_KEY, "open_pin_budget"): 1,
        (caller_key, "pin"): 1,
    }


def test_a_dead_prepares_unit_refuses_retryably_until_the_sweep_settles_it(
    monkeypatch: pytest.MonkeyPatch,
):
    rig = _Rig("derived", monkeypatch, ledger=_ledger(pin_budget=1))
    dead = _dead_prepare(rig, [Spend(BUDGET_KEY, AllowanceKind.pin_budget)])

    refused = rig.prepare("xlsx")
    rig.ledger.close_claim(dead)
    answered = rig.prepare("xlsx")

    assert refused.status_code == 503, refused.text
    assert refused.json()["detail"] == PinUnitHeld.detail
    assert refused.headers["retry-after"] == str(SWEEP_INTERVAL_SECONDS)
    assert answered.status_code == 200, answered.text


def test_waiters_past_the_read_pool_all_answer_the_record(monkeypatch: pytest.MonkeyPatch):
    """Waiters wake together, and each read after a wait queues for a permit in place of
    refusing."""
    monkeypatch.setenv("DATASETS_READ_MAX_CONCURRENCY", "2")
    rig = _Rig("derived", monkeypatch)
    render = _HeldRender("derived", monkeypatch, results=[None])
    read_claim = LiveAllowanceLedger.read_claim

    def slow(ledger: LiveAllowanceLedger, key: str) -> Any:
        time.sleep(0.02)
        return read_claim(ledger, key)

    with ThreadPoolExecutor(max_workers=8) as pool:
        leader = pool.submit(rig.prepare, "csv")
        assert render.entered.wait(10)
        waiters = []
        for _ in range(7):
            render.parked.clear()
            waiters.append(pool.submit(rig.prepare, "csv"))
            assert render.parked.wait(10)
        monkeypatch.setattr(LiveAllowanceLedger, "read_claim", slow)
        time.sleep(0.3)
        render.release.set()
        responses = [leader.result(), *(waiter.result() for waiter in waiters)]

    assert [r.status_code for r in responses] == [200] * 8, [r.text for r in responses]
    assert render.calls == 1


def test_waiters_on_one_export_hold_no_work_permit(monkeypatch: pytest.MonkeyPatch):
    """Three waiters and a pool of two: the other door's work still runs."""
    monkeypatch.setenv("DATASETS_WORK_MAX_CONCURRENCY", "2")
    rig = _Rig("fetched", monkeypatch)
    # Listed and pinned by now, so its second prepare takes a permit and renders nothing.
    assert rig.prepare("csv").status_code == 200
    rig.forget_exports()
    render = _HeldRender("fetched", monkeypatch, results=[None])

    with ThreadPoolExecutor(max_workers=5) as pool:
        leader = pool.submit(rig.prepare, "xlsx")
        assert render.entered.wait(10)
        waiters = [pool.submit(rig.prepare, "xlsx") for _ in range(3)]
        assert render.parked.wait(10)
        try:
            other = rig.prepare("csv")
        finally:
            render.release.set()
        responses = [leader.result(), *(waiter.result() for waiter in waiters)]

    assert other.status_code == 200, other.text
    assert [r.status_code for r in responses] == [200] * 4, [r.text for r in responses]
    assert render.calls == 1


def test_a_waiter_gives_up_at_the_materialise_wait_naming_the_running_prepare(
    monkeypatch: pytest.MonkeyPatch,
):
    """Waits a second, the shortest wait whose `Retry-After` differs from the backpressure one."""
    monkeypatch.setenv("DATASETS_MATERIALISE_WAIT_SECONDS", "1.1")
    rig = _Rig("fetched", monkeypatch)
    render = _HeldRender("fetched", monkeypatch, results=[None])

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(rig.prepare, "xlsx")
        assert render.entered.wait(10)
        try:
            refused = pool.submit(rig.prepare, "xlsx").result(timeout=5)
        finally:
            render.release.set()
        assert leader.result().status_code == 200

    assert refused.status_code == 503, refused.text
    assert (refused.json()["code"], refused.json()["retryable"]) == ("service_unavailable", True)
    assert refused.headers["retry-after"] == "2"


@pytest.mark.parametrize("kind", ["fetched", "derived"])
def test_a_prepare_answers_a_record_written_since_the_door_read_it(
    kind: str, monkeypatch: pytest.MonkeyPatch
):
    rig = _Rig(kind, monkeypatch)
    first = rig.prepare("csv")
    monkeypatch.setattr(exports_door, "recorded_export", lambda **_: None)
    monkeypatch.setattr(PreparingCopy, "hold", _forbidden)
    monkeypatch.setattr(PreparingCopy, "copy", _forbidden)

    again = rig.prepare("csv")

    assert again.status_code == 200, again.text
    assert again.json() == first.json()


def test_a_derived_file_is_rendered_and_stored_inside_one_slot_and_stays_there(
    monkeypatch: pytest.MonkeyPatch,
):
    """Each step sees a slot held, and the bytes are gone by the time the slot releases."""
    rig = _Rig("derived", monkeypatch)
    slots = 0
    inside = False
    steps: list[tuple[str, bool]] = []
    held: set[int] = set()
    held_at_release: list[int] = []

    class _Bytes(bytes):
        def __del__(self) -> None:
            held.discard(id(self))

    slot = exports_module.bounded_materialise_slot
    render = derived_door.render_derived_file
    put = rig.artifact_store.put

    @contextmanager
    def watched(*args: Any, **kwargs: Any) -> Iterator[None]:
        nonlocal slots, inside
        slots += 1
        with slot(*args, **kwargs):
            inside = True
            yield
            inside = False
            held_at_release.append(len(held))

    def rendered(**kwargs: Any) -> Any:
        steps.append(("render", inside))
        file = render(**kwargs)
        tracked = _Bytes(file.data)
        held.add(id(tracked))
        return replace(file, data=tracked)

    def stored(data: bytes, *, tier: Tier) -> Any:
        steps.append(("store", inside))
        return put(data, tier=tier)

    monkeypatch.setattr(derived_door, "bounded_materialise_slot", watched)
    monkeypatch.setattr(derived_door, "render_derived_file", rendered)
    monkeypatch.setattr(exports_module, "file_shape", _forbidden)
    monkeypatch.setattr(rig.artifact_store, "put", stored)

    assert rig.prepare("csv").status_code == 200

    assert steps == [("render", True), ("store", True)]
    assert slots == 1
    assert held_at_release == [0]


def test_both_doors_bound_the_derived_files_permit_wait_by_a_deadline(
    monkeypatch: pytest.MonkeyPatch,
):
    rig = _Rig("derived", monkeypatch)
    slot = derived_door.bounded_materialise_slot
    deadlines: list[float | None] = []

    def watched(*args: Any, deadline: float | None = None, **kwargs: Any) -> Any:
        deadlines.append(deadline)
        return slot(*args, deadline=deadline, **kwargs)

    monkeypatch.setattr(derived_door, "bounded_materialise_slot", watched)
    frame = rig.frame

    assert rig.prepare("csv").status_code == 200
    file_path = f"/api/datasets/derived/{frame['spec_id']}/{frame['content_version']}/file"
    assert rig.client.get(file_path).status_code == 200

    assert len(deadlines) == 2
    assert None not in deadlines


def _csv_bytes(rows: list[list[str]]) -> bytes:
    text = io.StringIO(newline="")
    csv.writer(text).writerows(rows)
    return text.getvalue().encode("utf-8")


def _bundled(data: bytes, *, fmt: str = "csv") -> bytes:
    """`data` as the first data resource of a bundle, beside a second CSV."""
    resources = [{"path": f"data/main.{fmt}"}, {"path": "data/main-totals.csv"}]
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"data/main.{fmt}", data)
        archive.writestr("data/main-totals.csv", b"a\r\n1\r\n2\r\n3\r\n")
        archive.writestr("datapackage.json", json.dumps({"resources": resources}))
    return out.getvalue()


@pytest.mark.parametrize("bundle", [False, True], ids=["bare", "bundle"])
def test_file_shape_counts_a_row_once_whatever_newlines_its_cells_quote(bundle: bool):
    data = _csv_bytes([["a", "b"], ["one\ntwo", "1"], ["three\r\nfour", "2"], ["fünf", "3"]])

    shape = exports_module.file_shape(
        _bundled(data) if bundle else data, format="csv", bundle=bundle
    )

    assert shape == (3, 2)


class _ChunkedReads(io.BytesIO):
    """A stream that refuses a read of everything left, which a whole copy would need."""

    def read(self, size: int | None = -1, /) -> bytes:
        assert size is not None and 0 <= size < len(self.getvalue())
        return super().read(size)

    def read1(self, size: int | None = -1, /) -> bytes:
        return self.read(size)


_LONG_CSV = _csv_bytes([["a", "b"], *([["text", "1"]] * 5_000)])


def test_a_csv_is_counted_from_chunks_of_the_stream():
    assert exports_module._table_shape(_ChunkedReads(_LONG_CSV), format="csv") == (5_000, 2)


def _parquet_bytes() -> bytes:
    out = io.BytesIO()
    pq.write_table(pa.table({"a": ["text"] * 5_000, "b": ["1"] * 5_000}), out)
    return out.getvalue()


@pytest.mark.parametrize(
    ("fmt", "data", "whole"),
    [
        ("csv", _LONG_CSV, ["datapackage.json"]),
        ("parquet", _parquet_bytes(), ["datapackage.json", "data/main.parquet"]),
    ],
    ids=["csv-streams", "parquet-seeks"],
)
def test_a_bundle_unzips_whole_only_what_seeks_backwards(
    fmt: str, data: bytes, whole: list[str], monkeypatch: pytest.MonkeyPatch
):
    """A zip member re-inflates from its start on each backward seek, so a member a reader
    seeks in is read whole, and a CSV, read forwards, is streamed."""
    read = zipfile.ZipFile.read
    unzipped: list[str] = []

    def recorded(archive: zipfile.ZipFile, name: str, *args: Any) -> bytes:
        unzipped.append(name)
        return read(archive, name, *args)

    monkeypatch.setattr(zipfile.ZipFile, "read", recorded)

    shape = exports_module.file_shape(_bundled(data, fmt=fmt), format=fmt, bundle=True)

    assert shape == (5_000, 2)
    assert unzipped == whole


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


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param("DELETE FROM datasets.frame_recipe", id="no-recipe"),
        pytest.param(
            "UPDATE datasets.frame_recipe SET recipe = json_set(recipe, '$.roots', json('[]'))",
            id="no-roots",
        ),
        pytest.param(
            f"DELETE FROM datasets.version_record WHERE subject_id = '{POPULATION}'",
            id="a-root-with-no-record",
        ),
    ],
)
def test_a_derived_frame_with_an_unnamed_source_refuses_provenance_unavailable(
    statement: str, monkeypatch: pytest.MonkeyPatch
):
    rig = _Rig("derived", monkeypatch)
    with rig.store.transaction() as tx:
        tx.execute(statement)

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


@pytest.mark.parametrize("revoked", ["served", "posted", "bound-later"])
def test_a_version_revoked_after_its_export_refuses_the_link_and_a_repeat_prepare(
    revoked: str, monkeypatch: pytest.MonkeyPatch
):
    rig = _Rig("fetched", monkeypatch)
    served = rig.frame["spec_id"]
    posted = rig.bind_posted_id()
    frame = {**rig.frame, "spec_id": posted}
    body = rig.prepare("csv", frame=frame).json()
    if revoked == "bound-later":
        rig.spec_store.state.spec_alias.append((posted, "spec_later"))
    rig.revoke({"served": served, "posted": posted, "bound-later": "spec_later"}[revoked])

    link = rig.link(body["export_id"])
    repeat = rig.prepare("csv", frame=frame)

    for resp in (link, repeat):
        assert resp.status_code == 410, resp.text
        assert resp.json()["code"] == "snapshot_revoked"
    assert rig.signings == []


def test_a_quarantined_identity_refuses_the_link_and_a_repeat_prepare(
    monkeypatch: pytest.MonkeyPatch,
):
    rig = _Rig("fetched", monkeypatch)
    body = rig.prepare("csv").json()
    _quarantine(rig, rig.frame["spec_id"])

    link = rig.link(body["export_id"])
    repeat = rig.prepare("csv")

    for resp in (link, repeat):
        assert resp.status_code == 500, resp.text
        assert resp.json()["code"] == "identity_quarantined"
    assert rig.signings == []


@pytest.mark.parametrize(
    ("kind", "table"),
    [
        pytest.param("fetched", "version_record", id="version-record"),
        pytest.param("derived", "frame_recipe", id="recipe"),
    ],
)
def test_an_export_whose_record_is_gone_refuses_rather_than_signing(
    kind: str, table: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard refuses what it cannot establish, in place of reading absence as cleared."""
    rig = _Rig(kind, monkeypatch)
    body = rig.prepare("csv").json()
    with rig.store.transaction() as tx:
        tx.execute(f"DELETE FROM datasets.{table}")

    link = rig.link(body["export_id"])
    repeat = rig.prepare("csv")

    for resp in (link, repeat):
        assert resp.status_code == 500, resp.text
        assert resp.json()["code"] == "internal_error"
    assert not [signed for signed in rig.signings if signed[0] == f"sha256:{body['sha256']}"][1:]


def test_a_repeat_prepare_needs_no_caller_and_a_new_one_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = _Rig("fetched", monkeypatch)
    body = rig.prepare("csv").json()

    def unresolved(*_: object, **__: object) -> None:
        raise CallerUnresolved("no caller")

    monkeypatch.setattr(exports_door, "resolve_caller", unresolved)

    repeat = rig.prepare("csv")
    fresh = rig.prepare("parquet")

    assert repeat.status_code == 200, repeat.text
    assert repeat.json() == body
    assert fresh.status_code == 503, fresh.text


def test_a_lasting_link_outlives_a_change_to_the_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stored body holds the reply of its day: a field since dropped, and none added since."""
    rig = _Rig("fetched", monkeypatch)
    body = rig.prepare("csv").json()
    stored = {key: value for key, value in body.items() if key != "rows"} | {"retired": 1}
    with rig.store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.export SET body = %s WHERE export_id = %s",
            (json.dumps(stored), body["export_id"]),
        )

    link = rig.link(body["export_id"])
    repeat = rig.prepare("csv")

    assert link.status_code == 302, link.text
    assert repeat.status_code == 200, repeat.text
    assert repeat.json() == stored


def test_a_root_revoked_after_a_derived_export_refuses_the_link_and_a_repeat_prepare(
    monkeypatch: pytest.MonkeyPatch,
):
    """A derived frame holds its roots' data, so a root's takedown reaches its export."""
    rig = _Rig("derived", monkeypatch)
    body = rig.prepare("csv").json()
    rig.revoke(POPULATION, POPULATION_V1)

    link = rig.link(body["export_id"])
    repeat = rig.prepare("csv")
    fresh = rig.prepare("parquet")

    for resp in (link, repeat, fresh):
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
    assert rig.link(_export_id(rig.frame, "parquet", bundle=False)).status_code == 404
    with pytest.raises(LicenceNotRedistributable) as raised:
        exports_module.check_licences([licence["source"]])
    assert raised.value.sources == ((licence["source"], "terms forbid it"),)


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


def _forbidden(*_args: object, **_: object) -> object:
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


def _quarantine(rig: _Rig, subject_id: str) -> None:
    rig.spec_store.state.identity_quarantine.append(
        {
            "id_kind": "spec",
            "subject_id": subject_id,
            "held_bytes": b"a",
            "incoming_bytes": b"b",
        }
    )


@pytest.mark.parametrize(
    ("takedown", "status", "code"),
    [
        pytest.param("revoked", 410, "snapshot_revoked", id="revoked"),
        pytest.param("quarantined", 500, "identity_quarantined", id="quarantined"),
    ],
)
def test_a_later_takedown_of_the_id_a_bare_request_sent_refuses_the_link(
    takedown: str, status: int, code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = _Rig("fetched", monkeypatch)
    served = rig.frame["spec_id"]
    posted = rig.bind_posted_id(_BARE)
    _resolving(monkeypatch, rig.frame)

    body = rig.prepare("csv", frame={"spec_id": posted}).json()

    assert body["frame"] == {**rig.frame, "spec_id": posted}
    assert body["export_id"] == _export_id(body["frame"], "csv", bundle=False)
    assert rig.recorded_subject(body["export_id"]) == served
    assert rig.link(body["export_id"]).status_code == 302
    rig.signings.clear()
    if takedown == "revoked":
        rig.revoke(posted)
    else:
        _quarantine(rig, posted)

    link = rig.link(body["export_id"])

    assert link.status_code == status, link.text
    assert link.json()["code"] == code
    assert rig.signings == []


def test_a_spec_root_refuses_a_version_revoked_under_the_id_it_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolve a bare frames read, a bare create and a bare export share
    (`analysis_roots.newest_spec_version`) honours the id sent, as a request naming it does."""
    rig = _Rig("fetched", monkeypatch)
    posted = rig.bind_posted_id(_BARE)
    canonical = analysis_roots.served_spec
    monkeypatch.setattr(
        analysis_roots,
        "served_spec",
        lambda *, spec_id, **kwargs: canonical(spec_id=rig.frame["spec_id"], **kwargs),
    )
    request = {
        "tree": {
            "verb": "aggregate",
            "arguments": {"aggregation": "sum", "group_by": ["stat_var"]},
            "inputs": {"data": {"spec_id": posted}},
        }
    }
    rig.revoke(posted)

    resp = rig.client.post("/api/datasets/analysis", json=request)

    assert resp.status_code == 410, resp.text
    assert resp.json()["code"] == "snapshot_revoked"


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


def test_a_refusal_that_passes_through_carries_no_cause_of_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refusal = SpecNotFound(id=_BARE)
    _resolving(monkeypatch, refusal)

    with pytest.raises(SpecNotFound) as raised:
        exports_module.resolve_export_frame(
            BareExportFrame(spec_id=_BARE),
            current=SimpleNamespace(recipe_store=None),  # ty: ignore[invalid-argument-type]
            deps=None,  # ty: ignore[invalid-argument-type]
        )

    assert raised.value is refusal
    assert raised.value.__cause__ is None


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
