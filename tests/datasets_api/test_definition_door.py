"""Tests exercise `GET /api/datasets/{id}` through the composed surface, including
re-materialisation, query options, current member validation, and unknown ids.
"""

from __future__ import annotations

import io
import json
import zipfile
from urllib.parse import urlsplit

import httpx
import pyarrow.parquet as pq
import pytest

import datasets_api.doors.definition as definition_module
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.config import DatasetsBuildSettings
from datasets_api.store_errors import DatasetsStoreUnavailable
from dc_sdmx import SdmxClient
from graph_generation import FakeGeneration
from materialise.graph import Facet
from series_coverage.index import CoverageIndex
from series_coverage.schema import ABSENT_ENTITY, CoverageManifest, FacetMetadata, RoleNames
from series_coverage.shared import CoverageSnapshot
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._identity_builders import described_identity, pinned_identity, register_body
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore, wire_definition_for
from .test_pin_door import FakeArtifactReferenceWriter

_BASE_URL = "https://datacommons.one.org/core/api"
_RECIPIENT = "country/KEN"
_STAT_VAR = "MalariaGrantDisbursements"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_DAC_RECIPE = "ConversionRecipeDAC"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_CSV_BODY = (
    _CSV_HEADER + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{_RECIPIENT},USD,GrantEquivalent,"
    f"NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2020",
    latest_date="2020",
)

_SCOPE = {
    "entities": {"recipient": [_RECIPIENT]},
    "facets": {
        "provenance": {"ref": "OECD_DAC2A"},
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}


def _graph(*, generation: FakeGeneration | None = None) -> FakeMintGraphReader:
    arcs: dict[str, tuple[str, ...]] = {
        "typeOf": ("StatisticalVariable",),
        "observationProperties": ("recipient",),
        "measuredProperty": ("amount",),
    }
    return FakeMintGraphReader(
        arcs={_STAT_VAR: arcs},
        # The graph facet probe keys on the bare dcid.
        facets={(_STAT_VAR, _RECIPIENT): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
        generation=generation,
    )


def _coverage_service(
    *,
    stat_vars: tuple[str, ...] = (_STAT_VAR,),
    donor_role: str | None = None,
    donor: str | None = None,
    recipient: str = _RECIPIENT,
    recipient_role: str = "recipient",
    facet_id: str = "facet1",
    provenance: str = "OECD_DAC2A",
    measurement_method: str = "GrantEquivalent",
    unit: str = "USD",
    years: tuple[int, ...] = (2020,),
) -> FakeCoverageService:
    """Builds a coverage facet per StatVar, matching the fixture's metadata and year range."""
    coverage_donor = "country/USA" if donor is None and donor_role is not None else donor
    coverage_donor = ABSENT_ENTITY if coverage_donor is None else coverage_donor
    index = build_coverage_index(
        [(stat_var, facet_id, coverage_donor, recipient, list(years)) for stat_var in stat_vars],
        role_names={
            stat_var: RoleNames(donor=donor_role, recipient=recipient_role)
            for stat_var in stat_vars
        },
        facet_metadata={
            facet_id: FacetMetadata(
                provenance_id=provenance, measurement_method=measurement_method, unit=unit
            )
        },
    )
    return FakeCoverageService(index)


class _MutableCoverageService:
    """Allows tests to replace coverage between facet probes."""

    def __init__(self, index: CoverageIndex, *, stamp: str = "test-coverage-stamp") -> None:
        self._index = index
        self.stamp = stamp

    def current(self) -> CoverageIndex:
        return self._index

    def replace(self, index: CoverageIndex) -> None:
        self._index = index

    def snapshot(self) -> CoverageSnapshot:
        return CoverageSnapshot(
            index=self._index,
            manifest=CoverageManifest(
                stamp=self.stamp,
                build_id=f"{self.stamp}-build",
                built_at="t",
                horizon=self._index.horizon,
            ),
            stamp=self.stamp,
        )


def _sdmx_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


class _MutableSdmxTransport:
    """Returns a mutable SDMX response body so tests can vary later fetch results."""

    def __init__(self, body: str = _CSV_BODY) -> None:
        self.body = body

    def __call__(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=self.body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )


def _multi_stat_var_sdmx_transport(
    csv_by_ref: dict[str, str], *, probe_only_rows_by_ref: dict[str, str] | None = None
) -> httpx.MockTransport:
    """Returns rows for each bare StatVar named in the combined request. Optional
    probe-only rows let tests model a StatVar with a facet but no observations."""
    probe_only_rows_by_ref = probe_only_rows_by_ref or {}

    def handler(request: httpx.Request) -> httpx.Response:
        variable_param = request.url.params.get("c[variableMeasured]", "")
        requested = variable_param.split(",")
        is_latest_probe = request.url.params.get("c[TIME_PERIOD]") == "LATEST"
        rows = "".join(
            probe_only_rows_by_ref[ref]
            if is_latest_probe and ref in probe_only_rows_by_ref
            else csv_by_ref[ref].removeprefix(_CSV_HEADER)
            for ref in requested
            if ref in csv_by_ref
        )
        body = _CSV_HEADER + rows
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    return httpx.MockTransport(handler)


def _client(
    *,
    graph: FakeMintGraphReader,
    store: FakeDatasetStore | None = None,
    artifact_store: FakeArtifactStore | None = None,
    sdmx_transport: httpx.MockTransport | None = None,
    version_writer=None,
    head_writer=None,
    version_store=None,
    snapshot_reader=None,
    code_index=None,
    coverage=None,
    settings: DatasetsBuildSettings | None = None,
    materialise_max_concurrency: int | None = None,
):
    """Builds a composed client with separate spec and version stores. A version store wires
    live readers and inventory-aware artifacts."""
    store = store if store is not None else FakeDatasetStore()
    if version_store is not None:
        from artifact_store import LiveArtifactInventory
        from datasets_api.version_store import LiveVersionRecordWriter

        from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

        if version_writer is None:
            version_writer = LiveVersionRecordWriter(store=version_store)
        if snapshot_reader is None:
            snapshot_reader = LiveSnapshotReader(store=version_store)
        if artifact_store is None:
            artifact_store = _InventoryRecordingArtifactStore(
                inventory=LiveArtifactInventory(store=version_store)
            )
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        snapshot_reader=snapshot_reader if snapshot_reader is not None else FakeSnapshotReader(),
        graph=graph,
        sdmx=SdmxClient(
            base_url=_BASE_URL,
            transport=sdmx_transport if sdmx_transport is not None else _sdmx_transport(),
        ),
        artifact_store=artifact_store if artifact_store is not None else FakeArtifactStore(),
        version_writer=version_writer,
        head_writer=head_writer,
        reference_writer=FakeArtifactReferenceWriter(),
        code_index=code_index,
        coverage=coverage if coverage is not None else _coverage_service(),
        settings=settings,
        materialise_max_concurrency=materialise_max_concurrency,
    )


def _mint_one_spec(client) -> str:
    body = {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
        "options": {"bundle": False},
    }
    resp = client.post("/api/datasets", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()["definition"]["spec_id"]


class _RaisingVersionRecordWriter:
    """Raises `DatasetsStoreUnavailable` for every version and artifact write."""

    def record_version(self, record, *, trace_id: str | None = None) -> None:
        raise DatasetsStoreUnavailable("exemplar diagnostics")

    def record_artifact(self, *, subject_id, content_version, artifact, trace_id=None) -> None:
        raise DatasetsStoreUnavailable("exemplar diagnostics")

    def write_version(self, write, *, trace_id: str | None = None) -> None:
        raise DatasetsStoreUnavailable("exemplar diagnostics")


def test_retrieval_wire_maps_a_version_writer_store_failure_to_retryable_503():
    """Retrieval maps a failing version write to a retryable 503."""
    store = FakeDatasetStore()
    graph = _graph()
    minting_client = _client(graph=graph, store=store)
    spec_id = _mint_one_spec(minting_client)
    retrieving_client = _client(
        graph=graph, store=store, version_writer=_RaisingVersionRecordWriter()
    )

    resp = retrieving_client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["code"] == "service_unavailable"
    assert body["retryable"] is True


def test_retrieval_re_materialises_a_minted_spec():
    graph = _graph()
    client = _client(graph=graph)
    spec_id = _mint_one_spec(client)

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["definition"]["spec_id"] == spec_id
    assert manifest["content_version"].startswith("cv_")
    # C79: the response lists the requested artifact and series statistics. Frame1 stays on
    # the version record.
    assert len(manifest["artifacts"]) == 2
    assert manifest["series_count"] is not None
    statistics_artifacts = [
        a for a in manifest["artifacts"] if a["rendering"] == {"statistics": "series"}
    ]
    assert len(statistics_artifacts) == 1


def test_retrieval_response_carries_cache_control_no_store():
    graph = _graph()
    client = _client(graph=graph)
    spec_id = _mint_one_spec(client)

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    assert resp.headers["cache-control"] == "no-store"


def test_bundled_retrieval_reproducing_call_targets_this_same_route():
    artifact_store = FakeArtifactStore()
    client = _client(graph=_graph(), artifact_store=artifact_store)
    spec_id = _mint_one_spec(client)

    resp = client.get(f"/api/datasets/{spec_id}", params={"bundle": "true"})

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    zip_bytes = artifact_store.fetch(manifest["artifacts"][0]["url"])
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        descriptor = json.loads(zf.read("datapackage.json"))
    call = descriptor["dfo"]["reproducing_call"]
    assert call["method"] == "GET"
    assert urlsplit(call["url"]).path == f"/api/datasets/{spec_id}"

    reproduced = client.get(urlsplit(call["url"]).path, params=call["options"])

    assert reproduced.status_code == 200, reproduced.text
    reproduced_manifest = reproduced.json()
    assert reproduced_manifest["definition"]["spec_id"] == manifest["definition"]["spec_id"]
    assert reproduced_manifest["content_version"] == manifest["content_version"]


def test_bundle_query_option_produces_a_zip_rendering():
    graph = _graph()
    client = _client(graph=graph)
    spec_id = _mint_one_spec(client)

    resp = client.get(f"/api/datasets/{spec_id}", params={"bundle": "true"})

    assert resp.status_code == 200, resp.text
    artifact = resp.json()["artifacts"][0]
    assert artifact["rendering"]["bundle"] is True
    assert artifact["media_type"] == "application/zip"


def test_unknown_query_key_refuses_request_invalid_with_the_equivalent_body_pointer():
    graph = _graph()
    client = _client(graph=graph)
    spec_id = _mint_one_spec(client)

    resp = client.get(f"/api/datasets/{spec_id}", params={"bogus": "1"})

    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/options/bogus"


def test_a_posted_definition_binding_and_opening_one_axis_refuses_request_invalid():
    """`creditor` names the origin axis, so an identity binding it and opening `origin`
    states both one creditor and every creditor."""
    client = _client(graph=_graph())
    spec_id = "spec_" + "0" * 32
    identity = described_identity(entities={"creditor": ["country/CHN"]}, open_roles=["origin"])
    definition = {"schema_version": 1, "spec_id": spec_id, "identity": identity}

    resp = client.post(f"/api/datasets/{spec_id}/options", json={"definition": definition})

    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/definition"


def test_unregistered_spec_id_refuses_spec_not_found():
    client = _client(graph=_graph())

    resp = client.get("/api/datasets/spec_never_registered")

    assert resp.status_code == 404
    assert resp.json()["code"] == "spec_not_found"
    # `dfo_platform.problem` sets `Cache-Control: no-store` on every
    # problem-details response.
    assert resp.headers["cache-control"] == "no-store"


def test_id_outside_the_spec_namespace_refuses_the_platform_routing_404():
    """Only `spec_` ids reach this door. Other ids get the platform routing 404,
    `not_found`."""
    client = _client(graph=_graph())

    resp = client.get("/api/datasets/not-a-real-namespace")

    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


def test_declared_role_drift_refuses_definition_not_reproducible():
    """The StatVar declares `recipient` at mint time, the role `_SCOPE`
    binds. A later graph write (simulated here by mutating the graph
    directly) drops that role, so the graph's current declared-role set
    does not reproduce against the entities the stored identity binds.
    """
    graph = _graph()
    client = _client(graph=graph)
    spec_id = _mint_one_spec(client)

    ok = client.get(f"/api/datasets/{spec_id}")
    assert ok.status_code == 200, ok.text

    graph._arcs = {
        **graph._arcs,
        _STAT_VAR: {
            "typeOf": ("StatisticalVariable",),
            "observationProperties": ("provider",),
            "measuredProperty": ("amount",),
        },
    }

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["code"] == "definition_not_reproducible"
    assert body["issues"] == [{"member": spec_id}, {"member": spec_id, "axis": "recipient"}]
    assert "pointer" not in body


def test_facet_drift_refuses_facet_not_available_without_body_pointers():
    """A facet-selection refusal from the shared materialise pipeline must
    not leak a `/facets/{axis}` pointer through a GET request that never
    posted a body for that pointer to name. This is reproduced by moving
    the series to another facet after mint, so the definition's bound
    provenance has no candidate.
    """
    graph = _graph()
    graph._arcs[_STAT_VAR] = {
        "typeOf": ("StatisticalVariable",),
        "observationProperties": ("provider", "recipient"),
        "measuredProperty": ("amount",),
    }
    two_role_header = _CSV_HEADER.replace(
        "variableMeasured,recipient,unit", "variableMeasured,provider,recipient,unit"
    )
    two_role_body = (
        two_role_header + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/USA,{_RECIPIENT},USD,"
        "GrantEquivalent,NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
    )
    transport = _MutableSdmxTransport()
    transport.body = two_role_body
    coverage = _MutableCoverageService(_coverage_service(donor_role="provider").current())
    client = _client(graph=graph, sdmx_transport=httpx.MockTransport(transport), coverage=coverage)
    spec_id = _mint_one_spec(client)

    transport.body = two_role_header
    coverage.replace(
        _coverage_service(donor_role="provider", facet_id="facet2", provenance="WB_IDS").current()
    )

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["code"] == "facet_not_available"
    assert "pointer" not in body
    assert body["issues"]
    for issue in body["issues"]:
        assert "pointer" not in issue
        assert issue.get("axis")


def _currency_facet(unit: str) -> Facet:
    return Facet(
        facet_id="facet1",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit=unit,
        obs_count=1,
        earliest_date="2020",
        latest_date="2020",
    )


def _currency_graph(*, unit: str, extra_arcs: dict | None = None) -> FakeMintGraphReader:
    """Mirrors `test_mint_error_classification.TestConversionRefusalsMapTo
    422ThroughTheRealMintDoor`'s own currency-graph fixture: the shape arcs
    a request-form unit binding needs, plus whatever `extra_arcs` names the
    native unit's own `CurrencyUnit` facts.
    """
    arcs: dict[str, tuple[str, ...]] = {
        "typeOf": ("StatisticalVariable",),
        "observationProperties": ("recipient",),
        "measuredProperty": ("amount",),
    }
    return FakeMintGraphReader(
        arcs={_STAT_VAR: arcs, **(extra_arcs or {})},
        facets={(_STAT_VAR, _RECIPIENT): [_currency_facet(unit)]},
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )


def _currency_sdmx_transport(
    unit: str, *, rates: dict[str, dict[int, float]] | None = None
) -> httpx.MockTransport:
    """Serves one observation row in `unit`. `rates` adds the DAC exchange-rate table a
    converting request loads."""
    row = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{_RECIPIENT},{unit},GrantEquivalent,"
        "NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
    )

    def observations(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_CSV_HEADER + row,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    bodies = recipe_table_bodies(_DAC_RECIPE, rates=rates) if rates is not None else {}
    return RateTableServer(bodies, fallback=observations).transport()


def _unit_request_identity(request: dict) -> dict:
    return pinned_identity(
        stat_var=_STAT_VAR,
        entities={"recipient": [_RECIPIENT]},
        facets={"unit": {"state": "value", "request": request}},
    )


def _register_currency_spec(client, *, identity: dict) -> str:
    """Registers an identity without materialising it. Retrieval runs the first conversion check."""
    spec_id = recompute_identity(identity).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )
    assert reg.status_code == 200, reg.text
    return spec_id


class TestConversionRefusalsMapTo422OnRetrieval:
    """Direct registration defers conversion checks until retrieval. These refusals carry no
    request pointer because the unit binding belongs to the stored identity."""

    def test_non_currency_native_unit_refuses_facet_binding_unsupported(self):
        graph = _currency_graph(unit="USD")
        client = _client(graph=graph, sdmx_transport=_currency_sdmx_transport("USD"))
        identity = _unit_request_identity({"currency": "USD", "prices": "current"})
        spec_id = _register_currency_spec(client, identity=identity)

        resp = client.get(f"/api/datasets/{spec_id}")

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "facet_binding_unsupported"
        assert problem["retryable"] is False
        assert "pointer" not in problem
        assert "cause" not in problem

    def test_declared_non_nominal_qualifier_refuses_recipe_not_declared_with_its_cause(self):
        graph = _currency_graph(
            unit="CurrencyUSD",
            extra_arcs={
                "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
                _STAT_VAR: {
                    "typeOf": ("StatisticalVariable",),
                    "observationProperties": ("recipient",),
                    "measuredProperty": ("amount",),
                    "measurementQualifier": ("RealValue",),
                },
            },
        )
        client = _client(
            graph=graph,
            sdmx_transport=_currency_sdmx_transport("CurrencyUSD"),
            coverage=_coverage_service(unit="CurrencyUSD"),
        )
        identity = _unit_request_identity({"currency": "USD", "prices": "current"})
        spec_id = _register_currency_spec(client, identity=identity)

        resp = client.get(f"/api/datasets/{spec_id}")

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "recipe_not_declared"
        assert problem["retryable"] is False
        assert "pointer" not in problem
        assert problem["cause"]["code"] == "nominal_undeclared"
        assert problem["cause"]["entities"] == []

    def test_no_qualifier_arc_at_all_natively_satisfied_serves_the_native_unit(self):
        # This fixture has no `measurementQualifier` arc, so native USD retrieval succeeds.

        graph = _currency_graph(
            unit="CurrencyUSD",
            extra_arcs={"CurrencyUSD": {"typeOf": ("CurrencyUnit",)}},
        )
        artifact_store = FakeArtifactStore()
        client = _client(
            graph=graph,
            sdmx_transport=_currency_sdmx_transport("CurrencyUSD"),
            coverage=_coverage_service(unit="CurrencyUSD"),
            artifact_store=artifact_store,
        )
        identity = _unit_request_identity({"currency": "USD", "prices": "current"})
        spec_id = _register_currency_spec(client, identity=identity)

        resp = client.get(f"/api/datasets/{spec_id}")

        assert resp.status_code == 200, resp.text
        manifest = resp.json()
        statistics_artifact = next(
            a for a in manifest["artifacts"] if a["rendering"] == {"statistics": "series"}
        )
        table = pq.read_table(io.BytesIO(artifact_store.fetch(statistics_artifact["url"])))
        assert json.loads(table.to_pylist()[0]["facet"])["unit"] == {
            "currency": "USD",
            "prices": "current",
        }

    def test_legacy_native_currency_refuses_rate_coverage_gap_with_its_cause(self):
        # `CurrencyFRF` is a withdrawn currency, so the exchange-rate table holds no row for it.
        graph = _currency_graph(
            unit="CurrencyFRF",
            extra_arcs={
                _STAT_VAR: {
                    "typeOf": ("StatisticalVariable",),
                    "observationProperties": ("recipient",),
                    "measuredProperty": ("amount",),
                    **CONVERTIBLE_STAT_VAR_ARCS,
                },
                **conversion_arcs(
                    recipe_id=_DAC_RECIPE,
                    provenance_id="OECD_DAC2A",
                    currencies=["CurrencyUSD", "CurrencyEUR", "CurrencyFRF"],
                ),
            },
        )
        client = _client(
            graph=graph,
            sdmx_transport=_currency_sdmx_transport(
                "CurrencyFRF", rates={"CurrencyUSD": {2020: 1.0}, "CurrencyEUR": {2020: 0.9}}
            ),
            coverage=_coverage_service(unit="CurrencyFRF"),
        )
        identity = _unit_request_identity({"currency": "EUR", "prices": "current"})
        spec_id = _register_currency_spec(client, identity=identity)

        resp = client.get(f"/api/datasets/{spec_id}")

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "rate_coverage_gap"
        assert problem["retryable"] is False
        assert "pointer" not in problem
        assert problem["cause"]["code"] == "rate_missing"
        assert problem["cause"]["entities"] == ["CurrencyFRF@2020"]
        assert problem["cause"]["truncated"] is False


def test_registered_pinned_identity_with_no_bound_entities_retrieves_every_covered_series():
    """A directly registered identity with no bound entities materialises every series
    coverage holds, under the identity it was registered with."""
    fetches: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetches.append(request)
        return httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(graph=_graph(), sdmx_transport=httpx.MockTransport(handler))
    identity = pinned_identity(stat_var=_STAT_VAR)
    spec_id = recompute_identity(identity).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )
    assert reg.status_code == 200, reg.text

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["definition"]["spec_id"] == spec_id
    assert manifest["definition"]["identity"]["entities"] == {}
    assert manifest["coverage"]["rows"] == 1
    # The fetch covers every series coverage holds and names its recipient axis.
    (fetch,) = fetches
    assert fetch.url.params["c[recipient]"] == _RECIPIENT
    assert "c[provider]" not in fetch.url.params


def _described_two_member_identity() -> dict:
    """A two-member described-arm identity that binds no entity."""
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "DevelopmentFinance",
            "measured_property": "amount",
        },
        "slots": [
            {
                "property": "DevelopmentFinanceScheme",
                "binding": {"state": "set", "refs": ["ODAGrants", "ODALoans"]},
            }
        ],
        "entities": {},
        "requested_window": {"kind": "all"},
    }


def test_registered_role_bearing_group_with_no_bound_entities_serves_options_and_mints():
    """An "every recipient" identity binds no entity. Its role-bearing members read coverage
    across every recipient, so `/options` and retrieval both materialise it."""
    csv_by_ref = {
        stat_var: _CSV_HEADER
        + f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{_RECIPIENT},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
        for stat_var in (_ODA_GRANTS, _ODA_LOANS)
    }
    client = _client(
        graph=_described_two_member_graph(),
        sdmx_transport=_multi_stat_var_sdmx_transport(csv_by_ref),
        coverage=_coverage_service(stat_vars=(_ODA_GRANTS, _ODA_LOANS)),
    )
    identity = _described_two_member_identity()
    spec_id = recompute_identity(identity).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )
    assert reg.status_code == 200, reg.text

    options = client.get(f"/api/datasets/{spec_id}/options")
    retrieved = client.get(f"/api/datasets/{spec_id}")

    assert options.status_code == 200, options.text
    assert options.json()["coverage"]["rows"] == 2
    assert retrieved.status_code == 200, retrieved.text
    assert retrieved.json()["coverage"]["rows"] == 2
    assert retrieved.json()["definition"]["identity"]["entities"] == {}


@pytest.mark.parametrize("registered", [True, False], ids=["registered-get", "unregistered-post"])
def test_options_refuses_an_estimate_over_the_inline_ceiling_before_any_fetch(
    monkeypatch: pytest.MonkeyPatch, registered: bool
):
    """Coverage prices the unbound role-bearing identity at two rows. With the inline ceiling
    at one row, `/options` refuses `over_ceiling` from the estimate and sends no SDMX request,
    whether it reads a registered spec or admits a posted definition."""
    monkeypatch.setattr(definition_module, "MATERIALISE_RAW_FETCH_GUARD_ROWS", 1)
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(500)

    client = _client(
        graph=_described_two_member_graph(),
        sdmx_transport=httpx.MockTransport(handler),
        coverage=_coverage_service(stat_vars=(_ODA_GRANTS, _ODA_LOANS)),
    )
    identity = _described_two_member_identity()
    spec_id = recompute_identity(identity).spec_id
    definition = wire_definition_for(spec_id=spec_id, identity=identity)
    if registered:
        reg = client.post("/api/datasets/register", json=register_body(definition=definition))
        assert reg.status_code == 200, reg.text
        options = client.get(f"/api/datasets/{spec_id}/options")
    else:
        options = client.post(f"/api/datasets/{spec_id}/options", json={"definition": definition})

    assert options.status_code == 422, options.text
    body = options.json()
    assert body["code"] == "over_ceiling"
    assert body["ceiling_kind"] == "rows"
    assert body["count"] == 2
    assert body["cap"] == 1
    assert sent == []


def test_retrieval_names_the_spec_id_when_the_statvars_own_graph_node_disappears():
    """`resolve_shape`'s own raise (the StatVar's graph node is gone, so its
    `typeOf` arc does not name `StatisticalVariable`) must surface
    `spec_id` on the wire, not the raw StatVar ref `resolve_shape` itself
    raises with.
    """
    graph = _graph()
    client = _client(graph=graph)
    spec_id = _mint_one_spec(client)

    graph._arcs = {}

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["code"] == "member_not_resolvable"
    assert body["issues"] == [{"member": spec_id}]


_STAT_VAR_B = "TbGrantDisbursements"


def test_group_retrieval_still_resolves_through_node_arcs_not_node_arcs_many():
    """A single definition resolves its shape through `node_arcs`. After a warmup GET, the two
    `node_arcs_many` calls belong to facet refresh and citation reads."""
    graph = _graph()
    client = _client(graph=graph)
    spec_id = _mint_one_spec(client)
    warmup = client.get(f"/api/datasets/{spec_id}")
    assert warmup.status_code == 200, warmup.text
    graph.calls = dict.fromkeys(graph.calls, 0)

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    assert graph.calls["node_arc_values"] > 0
    assert graph.calls["node_arcs_many"] == 2


def test_registered_breadth_only_identity_refuses_spec_not_exportable_on_retrieval():
    """A directly registered described identity with no enumerable members refuses retrieval
    as `spec_not_exportable`."""
    client = _client(graph=_graph())
    identity = described_identity()
    spec_id = recompute_identity(identity).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )
    assert reg.status_code == 200, reg.text

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "spec_not_exportable"


_ODA_GRANTS = "ODAGrantsAmount"
_ODA_GRANTS_REF = f"dcid:{_ODA_GRANTS}"
_ODA_LOANS = "ODALoansAmount"
_ODA_LOANS_REF = f"dcid:{_ODA_LOANS}"

_DESCRIBED_TWO_MEMBER_IDENTITY: dict = {
    "identity_version": "spec1",
    "shape": {
        "kind": "statvar_schema",
        "population_type": "DevelopmentFinance",
        "measured_property": "amount",
    },
    "slots": [
        {
            "property": "DevelopmentFinanceScheme",
            "binding": {"state": "set", "refs": ["ODAGrants", "ODALoans"]},
        }
    ],
    "entities": {"recipient": [_RECIPIENT]},
    "facets": {
        "provenance": {"state": "value", "ref": "OECD_DAC2A"},
        "measurementMethod": {"state": "value", "ref": "GrantEquivalent"},
        "unit": {"state": "value", "ref": "USD"},
    },
    "requested_window": {"kind": "all"},
}


def _described_two_member_graph() -> FakeMintGraphReader:
    member_arcs = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("DevelopmentFinance",),
        "measuredProperty": ("amount",),
        "observationProperties": ("recipient",),
        "constraintProperties": ("DevelopmentFinanceScheme",),
    }
    return FakeMintGraphReader(
        arcs={
            _ODA_GRANTS: {**member_arcs, "DevelopmentFinanceScheme": ("ODAGrants",)},
            _ODA_LOANS: {**member_arcs, "DevelopmentFinanceScheme": ("ODALoans",)},
        },
        facets={
            (_ODA_GRANTS, _RECIPIENT): [_FACET],
            (_ODA_LOANS, _RECIPIENT): [_FACET],
        },
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
    )


def test_registered_described_identity_with_real_members_retrieves_and_materialises_both():
    """A described identity that enumerates real members serves both and lists them in the trace."""
    graph = _described_two_member_graph()
    client = _client(
        graph=graph,
        sdmx_transport=_multi_stat_var_sdmx_transport(
            {
                _ODA_GRANTS: _CSV_HEADER
                + f"dataflow,DC:DF_OBS(1.0.0),I,{_ODA_GRANTS},{_RECIPIENT},USD,GrantEquivalent,"
                "NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n",
                _ODA_LOANS: _CSV_HEADER
                + f"dataflow,DC:DF_OBS(1.0.0),I,{_ODA_LOANS},{_RECIPIENT},USD,GrantEquivalent,"
                "NotApplicable,OECD_DAC2A,2020,500,,facet1\r\n",
            }
        ),
        coverage=_coverage_service(stat_vars=(_ODA_GRANTS, _ODA_LOANS)),
    )
    spec_id = recompute_identity(_DESCRIBED_TWO_MEMBER_IDENTITY).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(
            definition=wire_definition_for(spec_id=spec_id, identity=_DESCRIBED_TWO_MEMBER_IDENTITY)
        ),
    )
    assert reg.status_code == 200, reg.text

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["definition"]["spec_id"] == spec_id
    resolved_refs = {entry["ref"] for entry in manifest["trace"]["resolved_stat_vars"]}
    assert resolved_refs == {_ODA_GRANTS_REF, _ODA_LOANS_REF}


def test_registered_described_arm_group_with_no_bound_entities_retrieves_both_members():
    """A two-member described-arm identity with no bound entities materialises both members
    across every series coverage holds, under the identity it was registered with."""
    identity = _described_two_member_identity()
    serve_rows = _multi_stat_var_sdmx_transport(
        {
            _ODA_GRANTS: _CSV_HEADER
            + f"dataflow,DC:DF_OBS(1.0.0),I,{_ODA_GRANTS},{_RECIPIENT},USD,GrantEquivalent,"
            "NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n",
            _ODA_LOANS: _CSV_HEADER
            + f"dataflow,DC:DF_OBS(1.0.0),I,{_ODA_LOANS},{_RECIPIENT},USD,GrantEquivalent,"
            "NotApplicable,OECD_DAC2A,2020,500,,facet1\r\n",
        }
    )
    fetches: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetches.append(request)
        return serve_rows.handle_request(request)

    client = _client(
        graph=_described_two_member_graph(),
        sdmx_transport=httpx.MockTransport(handler),
        coverage=_coverage_service(stat_vars=(_ODA_GRANTS, _ODA_LOANS)),
    )
    spec_id = recompute_identity(identity).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )
    assert reg.status_code == 200, reg.text

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["definition"]["spec_id"] == spec_id
    assert manifest["definition"]["identity"]["entities"] == {}
    assert manifest["coverage"]["rows"] == 2
    resolved_refs = {entry["ref"] for entry in manifest["trace"]["resolved_stat_vars"]}
    assert resolved_refs == {_ODA_GRANTS_REF, _ODA_LOANS_REF}
    # Every fetch covers every series coverage holds, and between them they name both members.
    assert fetches
    for fetch in fetches:
        assert fetch.url.params["c[recipient]"] == _RECIPIENT
        assert "c[provider]" not in fetch.url.params
    assert {
        stat_var
        for fetch in fetches
        for stat_var in str(fetch.url.params["c[variableMeasured]"]).split(",")
    } == {_ODA_GRANTS, _ODA_LOANS}


def test_registered_described_identity_names_a_zero_row_member_with_its_own_count():
    """A graph-enumerated member with zero SDMX rows stays in the trace with count 0."""
    graph = _described_two_member_graph()
    client = _client(
        graph=graph,
        sdmx_transport=_multi_stat_var_sdmx_transport(
            {
                _ODA_GRANTS: _CSV_HEADER
                + f"dataflow,DC:DF_OBS(1.0.0),I,{_ODA_GRANTS},{_RECIPIENT},USD,GrantEquivalent,"
                "NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n",
                _ODA_LOANS: _CSV_HEADER,  # facet-viable, but zero data rows
            },
            probe_only_rows_by_ref={
                _ODA_LOANS: f"dataflow,DC:DF_OBS(1.0.0),I,{_ODA_LOANS},{_RECIPIENT},USD,"
                "GrantEquivalent,NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n",
            },
        ),
        coverage=_coverage_service(stat_vars=(_ODA_GRANTS, _ODA_LOANS)),
    )
    spec_id = recompute_identity(_DESCRIBED_TWO_MEMBER_IDENTITY).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(
            definition=wire_definition_for(spec_id=spec_id, identity=_DESCRIBED_TWO_MEMBER_IDENTITY)
        ),
    )
    assert reg.status_code == 200, reg.text

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    entries_by_ref = {entry["ref"]: entry for entry in manifest["trace"]["resolved_stat_vars"]}
    assert entries_by_ref.keys() == {_ODA_GRANTS_REF, _ODA_LOANS_REF}
    assert entries_by_ref[_ODA_GRANTS_REF]["observation_count"] == 1
    assert entries_by_ref[_ODA_LOANS_REF]["observation_count"] == 0


def test_registered_pin_id_the_graph_does_not_hold_refuses_member_not_resolvable_not_500():
    """A registered pin for a missing graph dcid returns 422 `member_not_resolvable`."""
    client = _client(graph=_graph())
    identity = pinned_identity(stat_var="NotAKnownStatVar")
    spec_id = recompute_identity(identity).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )
    assert reg.status_code == 200, reg.text

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["code"] == "member_not_resolvable"
    assert body["issues"] == [{"member": spec_id}]


def test_role_key_drift_refuses_definition_not_reproducible():
    """An entity role absent from the StatVar's current `observationProperties` makes its
    stored identity irreproducible."""
    client = _client(graph=_graph())
    identity = pinned_identity(stat_var=_STAT_VAR, entities={"donor": ["country/FRA"]})
    spec_id = recompute_identity(identity).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )
    assert reg.status_code == 200, reg.text

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["code"] == "definition_not_reproducible"
    assert body["issues"] == [{"member": spec_id}, {"member": spec_id, "axis": "donor"}]
    assert "pointer" not in body


def test_no_drift_register_then_get_round_trip_serves_200():
    """A directly registered pinned identity with no graph drift retrieves successfully."""
    client = _client(graph=_graph())
    identity = pinned_identity(
        stat_var=_STAT_VAR,
        entities={"recipient": [_RECIPIENT]},
        facets={
            "provenance": {"state": "value", "ref": "OECD_DAC2A"},
            "measurementMethod": {"state": "value", "ref": "GrantEquivalent"},
            "unit": {"state": "value", "ref": "USD"},
        },
    )
    spec_id = recompute_identity(identity).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )
    assert reg.status_code == 200, reg.text

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
