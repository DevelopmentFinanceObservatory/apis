"""`GET /api/datasets/{id}/options` returns output options, version, and coverage without
rendering artifacts or writing a version record. Registry and materialisation are real;
graph, SDMX, and artifact storage are faked.
"""

from __future__ import annotations

import json

import httpx
import pytest

from artifact_store import StoredArtifact, Tier
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.adapter import build_identity_document
from datasets_api.mint_models import Scope
from datasets_api.revision import SentinelRevisionReader
from datasets_api.schema import OptionsEnvelope as PublishedOptionsEnvelope
from datasets_api.shape import resolve_shape
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord, VersionRecord
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from statvar_members import POSTED_MEMBER_CAP
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.materialise._rate_table_support import CONVERTIBLE_STAT_VAR_ARCS, conversion_arcs

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._identity_builders import register_body
from ._snapshot_doubles import FakeSnapshotReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore, wire_definition_for
from .test_definition_door import (
    _DESCRIBED_TWO_MEMBER_IDENTITY,
    _currency_graph,
    _currency_sdmx_transport,
    _described_two_member_graph,
    _unit_request_identity,
)
from .test_definition_door import (
    _coverage_service as _currency_coverage_service,
)
from .test_pin_door import FakeArtifactReferenceWriter

_STAT_VAR = "MalariaGrantDisbursements"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_DAC_RECIPE = "ConversionRecipeDAC"
_BASE_URL = "https://datacommons.one.org/core/api"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_CSV_BODY = (
    _CSV_HEADER + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,USD,GrantEquivalent,"
    "NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
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
    "entities": {"recipient": ["country/KEN"]},
    "facets": {
        "provenance": {"ref": "OECD_DAC2A"},
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}

_FRAME_COLUMN_NAMES = (
    "stat_var",
    "recipient",
    "time_period",
    "value",
    "unit",
    "measurementMethod",
    "provenance",
    "status",
)

_ENVELOPE_TOP_LEVEL_FIELDS = {
    "schema_version",
    "definition",
    "collection",
    "content_version",
    "coverage",
    "output_options",
    "warnings",
    "trace",
}


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            },
            "country/KEN": {
                "countryAlpha3Code": ("KEN",),
                "countryAlpha2Code": ("KE",),
                "name": ("Kenya",),
            },
            "country/UGA": {"name": ("Uganda",)},
        },
        facets={
            (_STAT_VAR, "country/KEN"): [_FACET],
            (_STAT_VAR, "country/UGA"): [_FACET],
        },
        labels={"OECD_DAC2A": "OECD DAC2a", "country/KEN": "Kenya", "country/UGA": "Uganda"},
    )


def _sdmx_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


def _body() -> dict:
    return {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
        "options": {"bundle": False},
    }


class _CountingArtifactStore(FakeArtifactStore):
    """Counts stored artifacts."""

    def __init__(self) -> None:
        super().__init__()
        self.put_calls = 0

    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        self.put_calls += 1
        return super().put(data, tier=tier)


class _RecordingVersionRecordWriter:
    """Records version-write calls without persisting them."""

    def __init__(self) -> None:
        self.records: list[VersionRecord] = []

    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> None:
        self.records.append(record)

    def record_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: ArtifactRecord,
        trace_id: str | None = None,
    ) -> None:
        return None


def _coverage_service() -> FakeCoverageService:
    """Builds 2020 coverage rows for the two entities in `_graph()`."""
    index = build_coverage_index(
        [
            (_STAT_VAR, "facet1", "country/USA", "country/KEN", [2020]),
            (_STAT_VAR, "facet1", "country/USA", "country/UGA", [2020]),
        ],
        role_names={_STAT_VAR: RoleNames(donor="provider", recipient="recipient")},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _client(
    *,
    store: FakeDatasetStore | None = None,
    artifact_store=None,
    graph: FakeMintGraphReader | None = None,
    version_writer=None,
    head_writer=None,
    snapshot_reader=None,
):
    store = store if store is not None else FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph if graph is not None else _graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        revision_reader=SentinelRevisionReader(),
        artifact_store=artifact_store if artifact_store is not None else FakeArtifactStore(),
        version_writer=version_writer,
        head_writer=head_writer,
        snapshot_reader=snapshot_reader if snapshot_reader is not None else FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )


def _assert_output_options_shape(output_options: dict) -> None:
    assert output_options["capability_version"] == DATASETS_CAPABILITY_VERSION
    assert {"columns", "formats", "bundle", "enrichments", "pivots", "groupby"} <= set(
        output_options
    )
    assert [c["name"] for c in output_options["columns"]] == list(_FRAME_COLUMN_NAMES)


def test_definition_options_envelope_matches_a_minted_content_version():
    store = FakeDatasetStore()
    client = _client(store=store)
    mint_resp = client.post("/api/datasets", json=_body())
    assert mint_resp.status_code == 200, mint_resp.text
    manifest = mint_resp.json()
    spec_id = manifest["definition"]["spec_id"]

    resp = client.get(f"/api/datasets/{spec_id}/options")

    assert resp.status_code == 200, resp.text
    envelope = resp.json()
    assert set(envelope) <= _ENVELOPE_TOP_LEVEL_FIELDS
    assert "citation" not in envelope
    assert "artifacts" not in envelope
    assert "statistics" not in envelope
    assert envelope["definition"]["spec_id"] == spec_id
    assert envelope["content_version"] == manifest["content_version"]
    assert envelope["coverage"] == manifest["coverage"]
    _assert_output_options_shape(envelope["output_options"])
    PublishedOptionsEnvelope.model_validate(envelope)


def _member_sidecar(
    stat_var_ref: str, *, entities: dict[str, list[str]], graph: FakeMintGraphReader
) -> tuple[str, dict]:
    shape = resolve_shape(stat_var_ref, graph=graph)
    document = build_identity_document(
        shape, scope=Scope(entities=entities, facets=_SCOPE["facets"])
    )
    recomputed = recompute_identity(document)
    return recomputed.spec_id, {
        "schema_version": 1,
        "spec_id": recomputed.spec_id,
        "identity": document,
    }


def test_collection_options_envelope_carries_output_options_per_member_never_at_the_top_level():
    store = FakeDatasetStore()
    graph = _graph()
    client = _client(store=store, graph=graph)
    _spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    _spec_uga, def_uga = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/UGA"]}, graph=graph
    )
    reg = client.post(
        "/api/datasets/register", json={"schema_version": 1, "members": [def_ken, def_uga]}
    )
    assert reg.status_code == 200, reg.text
    collection_id = reg.json()["collection"]["collection_id"]

    resp = client.get(f"/api/datasets/{collection_id}/options")

    assert resp.status_code == 200, resp.text
    envelope = resp.json()
    assert set(envelope) <= _ENVELOPE_TOP_LEVEL_FIELDS
    assert "output_options" not in envelope
    members = envelope["collection"]["members"]
    assert len(members) == 2
    for member in members:
        _assert_output_options_shape(member["output_options"])
    PublishedOptionsEnvelope.model_validate(envelope)


def test_the_options_route_stores_no_artifact_and_writes_no_version_record():
    store = FakeDatasetStore()
    writer = _RecordingVersionRecordWriter()
    artifact_store = _CountingArtifactStore()
    client = _client(store=store, artifact_store=artifact_store, version_writer=writer)
    mint_resp = client.post("/api/datasets", json=_body())
    assert mint_resp.status_code == 200, mint_resp.text
    spec_id = mint_resp.json()["definition"]["spec_id"]
    records_after_mint = len(writer.records)
    puts_after_mint = artifact_store.put_calls
    assert records_after_mint > 0
    assert puts_after_mint > 0

    resp = client.get(f"/api/datasets/{spec_id}/options")

    assert resp.status_code == 200, resp.text
    assert len(writer.records) == records_after_mint
    assert artifact_store.put_calls == puts_after_mint


def _minted_manifest() -> dict:
    """Mints `_body()` into a store of its own, so the spec stays unregistered elsewhere."""
    resp = _client().post("/api/datasets", json=_body())
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_posted_definition_answers_for_an_unregistered_spec_as_its_mint_and_writes_nothing():
    """A search result's definition, posted before any mint in this deployment, gets the
    enrichments its mint reports and the envelope GET serves once a mint has registered it.
    The registry, version records and artifact store are untouched, and GET still 404s."""
    manifest = _minted_manifest()
    spec_id = manifest["definition"]["spec_id"]
    store = FakeDatasetStore()
    writer = _RecordingVersionRecordWriter()
    artifact_store = _CountingArtifactStore()
    head_store = LiveHeadStore(store=FakeVersionRecordDatasetStore())
    client = _client(
        store=store, artifact_store=artifact_store, version_writer=writer, head_writer=head_store
    )

    resp = client.post(
        f"/api/datasets/{spec_id}/options", json={"definition": manifest["definition"]}
    )

    assert resp.status_code == 200, resp.text
    envelope = resp.json()
    PublishedOptionsEnvelope.model_validate(envelope)
    assert envelope["output_options"]["enrichments"]
    assert envelope["output_options"]["enrichments"] == manifest["output_options"]["enrichments"]
    assert envelope["definition"] == manifest["definition"]
    assert envelope["content_version"] == manifest["content_version"]
    assert envelope["coverage"] == manifest["coverage"]
    assert store.state.spec_registry == {}
    assert store.state.spec_definition == {}
    assert writer.records == []
    assert head_store.read_head(spec_id) is None
    assert artifact_store.put_calls == 0
    assert client.get(f"/api/datasets/{spec_id}/options").json()["code"] == "spec_not_found"

    registered = _client()
    assert registered.post("/api/datasets", json=_body()).status_code == 200
    stored = registered.get(f"/api/datasets/{spec_id}/options").json()
    posted = registered.post(
        f"/api/datasets/{spec_id}/options", json={"definition": manifest["definition"]}
    ).json()
    for served in (stored, posted):
        assert {key: value for key, value in served.items() if key != "trace"} == {
            key: value for key, value in envelope.items() if key != "trace"
        }


def test_posted_dcids_foreign_to_a_registered_spec_are_ignored_and_the_envelope_matches_get():
    """A registered id resolves as GET resolves it, so the posted member dcids seed nothing."""
    client = _client()
    manifest = client.post("/api/datasets", json=_body()).json()
    spec_id = manifest["definition"]["spec_id"]

    got = client.get(f"/api/datasets/{spec_id}/options")
    posted = client.post(
        f"/api/datasets/{spec_id}/options",
        json={
            "definition": manifest["definition"],
            "resolved_stat_var_dcids": [_STAT_VAR, "SomeOtherStatVar"],
        },
    )

    assert got.status_code == 200, got.text
    assert posted.status_code == 200, posted.text
    assert {key: value for key, value in posted.json().items() if key != "trace"} == {
        key: value for key, value in got.json().items() if key != "trace"
    }


def _foreign_identity(definition: dict) -> dict:
    """The same identity bound to another recipient, so it hashes to another spec id."""
    identity = json.loads(json.dumps(definition["identity"]))
    identity["entities"] = {"recipient": ["country/UGA"]}
    return identity


def _identity_hashing_elsewhere(definition: dict) -> dict:
    return {"definition": {**definition, "identity": _foreign_identity(definition)}}


def _definition_naming_another_id(definition: dict) -> dict:
    foreign_spec_id = recompute_identity(_foreign_identity(definition)).spec_id
    return {"definition": {**definition, "spec_id": foreign_spec_id}}


@pytest.mark.parametrize(
    ("body_for", "expected_status", "expected_code", "expected_pointers"),
    [
        pytest.param(
            _identity_hashing_elsewhere,
            422,
            "spec_id_mismatch",
            [],
            id="identity-hashes-to-another-id",
        ),
        pytest.param(
            _definition_naming_another_id,
            422,
            "spec_id_mismatch",
            [],
            id="definition-names-another-id",
        ),
        pytest.param(
            lambda definition: {
                "definition": definition,
                "resolved_stat_var_dcids": [_STAT_VAR, "SomeOtherStatVar"],
            },
            422,
            "member_not_resolvable",
            ["/resolved_stat_var_dcids/1"],
            id="posted-dcid-outside-the-family",
        ),
        pytest.param(
            lambda definition: {
                "definition": definition,
                "resolved_stat_var_dcids": [_STAT_VAR]
                + [f"OtherStatVar{i}" for i in range(1, 1_626)],
            },
            422,
            "member_not_resolvable",
            [f"/resolved_stat_var_dcids/{i}" for i in range(1, 1_626)],
            id="1626-posted-dcids-clear-the-length-check",
        ),
        pytest.param(
            lambda definition: {
                "definition": definition,
                "resolved_stat_var_dcids": [_STAT_VAR] * (POSTED_MEMBER_CAP + 1),
            },
            400,
            "request_invalid",
            [],
            id="posted-dcids-over-the-ceiling",
        ),
    ],
)
def test_posted_definition_refusals_point_into_the_options_body_and_write_nothing(
    body_for, expected_status: int, expected_code: str, expected_pointers: list[str]
) -> None:
    definition = _minted_manifest()["definition"]
    store = FakeDatasetStore()
    client = _client(store=store)

    resp = client.post(f"/api/datasets/{definition['spec_id']}/options", json=body_for(definition))

    assert resp.status_code == expected_status, resp.text
    problem = resp.json()
    assert problem["code"] == expected_code
    assert [issue["pointer"] for issue in problem.get("issues", [])] == expected_pointers
    assert store.state.spec_registry == {}


@pytest.mark.parametrize(
    "body_for",
    [
        pytest.param(_identity_hashing_elsewhere, id="identity-hashes-to-another-id"),
        pytest.param(_definition_naming_another_id, id="definition-names-another-id"),
    ],
)
def test_a_posted_definition_for_another_spec_refuses_even_when_the_path_id_has_a_fresh_head(
    body_for,
) -> None:
    """A fresh head answers without resolving, so the body is checked before the head is read."""
    version_store = FakeVersionRecordDatasetStore()
    head_store = LiveHeadStore(store=version_store)
    client = _client(
        version_writer=LiveVersionRecordWriter(store=version_store),
        head_writer=head_store,
        snapshot_reader=LiveSnapshotReader(store=version_store),
    )
    definition = client.post("/api/datasets", json=_body()).json()["definition"]
    spec_id = definition["spec_id"]
    assert head_store.read_head(spec_id) is not None
    served = client.post(f"/api/datasets/{spec_id}/options", json={"definition": definition})
    assert served.json()["trace"]["served_from"] == "head"

    resp = client.post(f"/api/datasets/{spec_id}/options", json=body_for(definition))

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "spec_id_mismatch"


def test_a_query_key_on_the_options_route_refuses_request_invalid_with_an_options_pointer():
    store = FakeDatasetStore()
    client = _client(store=store)
    mint_resp = client.post("/api/datasets", json=_body())
    assert mint_resp.status_code == 200, mint_resp.text
    spec_id = mint_resp.json()["definition"]["spec_id"]

    resp = client.get(f"/api/datasets/{spec_id}/options", params={"format": "csv"})

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/options/format"


@pytest.mark.parametrize(
    ("route", "expected_status", "expected_code", "expected_detail"),
    [
        pytest.param(
            "/api/datasets/spec_unregistered/options",
            404,
            "spec_not_found",
            "no definition or collection is registered at id 'spec_unregistered'",
            id="unknown-spec-id",
        ),
        pytest.param(
            "/api/datasets/col_unregistered/options",
            404,
            "spec_not_found",
            "no definition or collection is registered at id 'col_unregistered'",
            id="unknown-collection-id",
        ),
    ],
)
def test_unknown_id_refuses_spec_not_found_on_the_options_route(
    route: str, expected_status: int, expected_code: str, expected_detail: str
) -> None:
    client = _client(graph=_graph())

    resp = client.get(route)

    assert resp.status_code == expected_status, resp.text
    problem = resp.json()
    assert problem["code"] == expected_code
    assert problem["detail"] == expected_detail


def _currency_client(
    *, graph: FakeMintGraphReader, unit: str, rates: dict[str, dict[int, float]] | None = None
):
    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_currency_sdmx_transport(unit, rates=rates)),
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_currency_coverage_service(unit=unit),
    )


def _register_currency_spec_for_options(client, *, identity: dict) -> str:
    spec_id = recompute_identity(identity).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )
    assert reg.status_code == 200, reg.text
    return spec_id


class TestConversionRefusalsMapTo422OnTheOptionsRoute:
    """The options route shares retrieval's conversion refusal mapping."""

    def test_non_currency_native_unit_refuses_facet_binding_unsupported(self):
        graph = _currency_graph(unit="USD")
        client = _currency_client(graph=graph, unit="USD")
        identity = _unit_request_identity({"currency": "USD", "prices": "current"})
        spec_id = _register_currency_spec_for_options(client, identity=identity)

        resp = client.get(f"/api/datasets/{spec_id}/options")

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "facet_binding_unsupported"
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
        client = _currency_client(graph=graph, unit="CurrencyUSD")
        identity = _unit_request_identity({"currency": "USD", "prices": "current"})
        spec_id = _register_currency_spec_for_options(client, identity=identity)

        resp = client.get(f"/api/datasets/{spec_id}/options")

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "recipe_not_declared"
        assert "pointer" not in problem
        assert problem["cause"]["code"] == "nominal_undeclared"

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
        client = _currency_client(
            graph=graph,
            unit="CurrencyFRF",
            rates={"CurrencyUSD": {2020: 1.0}, "CurrencyEUR": {2020: 0.9}},
        )
        identity = _unit_request_identity({"currency": "EUR", "prices": "current"})
        spec_id = _register_currency_spec_for_options(client, identity=identity)

        resp = client.get(f"/api/datasets/{spec_id}/options")

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "rate_coverage_gap"
        assert "pointer" not in problem
        assert problem["cause"]["code"] == "rate_missing"
        assert problem["cause"]["entities"] == ["CurrencyFRF@2020"]


def test_collection_member_enumerating_more_than_one_statvar_refuses_over_ceiling_on_the_options_route():
    """The options route refuses a member ask that enumerates multiple StatVars."""
    client = _client(graph=_described_two_member_graph())
    spec_id = recompute_identity(_DESCRIBED_TWO_MEMBER_IDENTITY).spec_id
    reg = client.post(
        "/api/datasets/register",
        json={
            "schema_version": 1,
            "members": [
                wire_definition_for(spec_id=spec_id, identity=_DESCRIBED_TWO_MEMBER_IDENTITY)
            ],
        },
    )
    assert reg.status_code == 200, reg.text
    collection_id = reg.json()["collection"]["collection_id"]

    resp = client.get(f"/api/datasets/{collection_id}/options")

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "over_ceiling"
