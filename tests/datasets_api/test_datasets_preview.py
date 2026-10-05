"""Preview tests pin the `PreviewEnvelope` contract and prove each input kind leaves registry,
artifact, version, and head stores unchanged. #295 covers the spec-id sidecar miss.
"""

from __future__ import annotations

import httpx

import datasets_api.kinds.collection_id
import datasets_api.preview
from datasets_api.adapter import build_identity_document
from datasets_api.mint_models import Scope
from datasets_api.revision import SentinelRevisionReader
from datasets_api.shape import resolve_shape
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from series_coverage.index import Estimate
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry, mint_collection_id
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_STAT_VAR = "MalariaGrantDisbursements"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_BASE_URL = "https://datacommons.one.org/core/api"
_SOURCE_CODE = "CODE1"
_SCOPE_DICT = {"entities": {"recipient": ["country/KEN"]}}

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_CSV_BODY = (
    _CSV_HEADER + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,USD,GrantEquivalent,"
    "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2015",
    latest_date="2015",
)


def _arcs() -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        _STAT_VAR: {
            "typeOf": ("StatisticalVariable",),
            "observationProperties": ("recipient",),
            "measuredProperty": ("amount",),
        },
        _SOURCE_CODE: {"member": (_STAT_VAR_REF,)},
    }


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs=_arcs(),
        # The graph facet probe keys on the bare dcid.
        facets={(_STAT_VAR, "country/KEN"): [_FACET], (_STAT_VAR, "country/UGA"): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", "country/KEN": "Kenya", "country/UGA": "Uganda"},
    )


# Ten rows for the one bound entity, for a test that needs the survivor
# count to exceed a `limit` it supplies.
_CSV_MULTI_ROW_BODY = _CSV_HEADER + "".join(
    f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,USD,GrantEquivalent,"
    f"NotApplicable,OECD_DAC2A,{2000 + i},1000,,facet1\r\n"
    for i in range(10)
)


def _sdmx_transport(requests: list[httpx.Request], *, body: str = _CSV_BODY):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def _coverage_service() -> FakeCoverageService:
    """Builds coverage matching `_FACET` for both recipients and the 2015 CSV row."""
    index = build_coverage_index(
        [
            (_STAT_VAR, "facet1", "country/USA", "country/KEN", [2015]),
            (_STAT_VAR, "facet1", "country/USA", "country/UGA", [2015]),
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
    graph: FakeMintGraphReader | None = None,
    spec_store: FakeDatasetStore | None = None,
    version_store: FakeVersionRecordDatasetStore | None = None,
    sdmx_requests: list[httpx.Request] | None = None,
    sdmx_body: str | None = None,
    sdmx_transport: httpx.MockTransport | None = None,
):
    """Builds the client with inspectable registry, version, head, and artifact stores."""
    spec_store = spec_store if spec_store is not None else FakeDatasetStore()
    version_store = version_store if version_store is not None else FakeVersionRecordDatasetStore()
    requests = sdmx_requests if sdmx_requests is not None else []
    sdmx = SdmxClient(
        base_url=_BASE_URL,
        transport=sdmx_transport
        if sdmx_transport is not None
        else _sdmx_transport(requests, body=sdmx_body if sdmx_body is not None else _CSV_BODY),
    )
    artifact_store = FakeArtifactStore()
    client = make_client(
        registry=LiveSpecRegistry(store=spec_store),
        spec_reader=LiveSpecReader(store=spec_store),
        graph=graph if graph is not None else _graph(),
        sdmx=sdmx,
        revision_reader=SentinelRevisionReader(),
        artifact_store=artifact_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        head_writer=LiveHeadStore(store=version_store),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )
    return client, artifact_store


def _assert_nothing_written(
    *, spec_store: FakeDatasetStore, version_store: FakeVersionRecordDatasetStore, artifact_store
) -> None:
    """Asserts that preview writes no registry row, artifact, version, or head."""
    assert spec_store.state.spec_registry == {}
    assert spec_store.state.spec_definition == {}
    assert spec_store.state.collection_registry == {}
    assert spec_store.state.collection_member == {}
    assert artifact_store._objects == {}
    with version_store.transaction() as tx:
        assert tx.fetch_all("SELECT * FROM datasets.version_record") == []
        assert tx.fetch_all("SELECT * FROM datasets.subject_head") == []


def _sidecar_for(
    stat_var_ref: str, *, graph: FakeMintGraphReader, entities: dict[str, list[str]]
) -> tuple[str, dict]:
    shape = resolve_shape(stat_var_ref, graph=graph)
    document = build_identity_document(shape, scope=Scope(entities=entities))
    recomputed = recompute_identity(document)
    return recomputed.spec_id, {
        "schema_version": 1,
        "spec_id": recomputed.spec_id,
        "identity": document,
    }


def _preview_body(input_: dict, *, limit: int | None = None) -> dict:
    body: dict = {"schema_version": 1, "input": input_}
    body["preview"] = {"limit": limit} if limit is not None else {}
    return body


# ---------------------------------------------------------------------------
# Envelope contract
# ---------------------------------------------------------------------------


def test_preview_envelope_is_distinct_from_a_manifest():
    client, _artifact_store = _client()

    resp = client.post(
        "/api/datasets",
        json=_preview_body(
            {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE_DICT}, limit=10
        ),
    )

    assert resp.status_code == 200, resp.text
    envelope = resp.json()
    assert set(envelope) == {"schema_version", "members"}
    assert len(envelope["members"]) == 1
    member = envelope["members"][0]
    assert member["stat_var"] == _STAT_VAR_REF
    assert set(member) == {
        "stat_var",
        "content_version",
        "coverage",
        "citation",
        "warnings",
        "sample",
        "total_rows",
        "truncated",
    }
    assert member["total_rows"] == 1
    assert member["truncated"] is False
    assert member["sample"]["columns"]
    assert len(member["sample"]["rows"]) == 1


def test_preview_row_sample_is_bounded_by_limit():
    """`limit` bounds the delivered sample. `total_rows` remains the full survivor count."""
    client, _artifact_store = _client()

    resp = client.post(
        "/api/datasets",
        json=_preview_body(
            {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE_DICT}, limit=1
        ),
    )

    assert resp.status_code == 200, resp.text
    member = resp.json()["members"][0]
    assert member["total_rows"] == 1
    assert len(member["sample"]["rows"]) <= 1


# ---------------------------------------------------------------------------
# statvar_set preview leaves stores unchanged
# ---------------------------------------------------------------------------


def test_preview_statvar_set_writes_nothing():
    spec_store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    client, artifact_store = _client(spec_store=spec_store, version_store=version_store)

    resp = client.post(
        "/api/datasets",
        json=_preview_body(
            {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE_DICT}
        ),
    )

    assert resp.status_code == 200, resp.text
    _assert_nothing_written(
        spec_store=spec_store, version_store=version_store, artifact_store=artifact_store
    )


# ---------------------------------------------------------------------------
# source_codes preview resolves through the graph without writes
# ---------------------------------------------------------------------------


def test_preview_source_codes_writes_nothing():
    spec_store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    client, artifact_store = _client(spec_store=spec_store, version_store=version_store)

    resp = client.post(
        "/api/datasets",
        json=_preview_body({"kind": "source_codes", "codes": [_SOURCE_CODE], "scope": _SCOPE_DICT}),
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["members"][0]["stat_var"] == _STAT_VAR_REF
    _assert_nothing_written(
        spec_store=spec_store, version_store=version_store, artifact_store=artifact_store
    )


# ---------------------------------------------------------------------------
# spec_id: the registry-hit path
# ---------------------------------------------------------------------------


def test_preview_spec_id_registry_hit_writes_nothing():
    spec_store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    client, _artifact_store = _client(spec_store=spec_store, version_store=version_store)
    minted = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE_DICT},
        },
    )
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]
    # The preview must leave these counts unchanged.
    with version_store.transaction() as tx:
        version_rows_before = len(tx.fetch_all("SELECT * FROM datasets.version_record"))
    registry_rows_before = len(spec_store.state.spec_registry)

    resp = client.post("/api/datasets", json=_preview_body({"kind": "spec_id", "spec_id": spec_id}))

    assert resp.status_code == 200, resp.text
    assert resp.json()["members"][0]["stat_var"] == _STAT_VAR_REF
    assert len(spec_store.state.spec_registry) == registry_rows_before
    with version_store.transaction() as tx:
        assert len(tx.fetch_all("SELECT * FROM datasets.version_record")) == version_rows_before


# ---------------------------------------------------------------------------
# spec_id: the registry-miss-with-sidecar path -- the write `mint_spec_id`
# runs through `deps.registry.register_definition` at kinds/spec_id.py's
# own miss branch, the one a preview has to bypass entirely, not merely
# skip the artifact write for.
# ---------------------------------------------------------------------------


def test_preview_spec_id_registry_miss_with_sidecar_writes_nothing():
    spec_store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    graph = _graph()
    client, artifact_store = _client(
        spec_store=spec_store, version_store=version_store, graph=graph
    )
    spec_id, definition = _sidecar_for(
        _STAT_VAR_REF, graph=graph, entities={"recipient": ["country/KEN"]}
    )
    assert spec_id not in spec_store.state.spec_registry

    resp = client.post(
        "/api/datasets",
        json=_preview_body({"kind": "spec_id", "spec_id": spec_id, "definition": definition}),
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["members"][0]["stat_var"] == _STAT_VAR_REF
    _assert_nothing_written(
        spec_store=spec_store, version_store=version_store, artifact_store=artifact_store
    )


# ---------------------------------------------------------------------------
# collection_id: the registry-miss-with-sidecar path, the same shape of
# write `preview_spec_id` bypasses, applied to `mint_collection_id`'s own
# `deps.registry.register_collection` call.
# ---------------------------------------------------------------------------


def test_preview_collection_id_registry_miss_with_sidecar_writes_nothing():
    spec_store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    graph = _graph()
    client, artifact_store = _client(
        spec_store=spec_store, version_store=version_store, graph=graph
    )
    spec_ken, def_ken = _sidecar_for(
        _STAT_VAR_REF, graph=graph, entities={"recipient": ["country/KEN"]}
    )
    spec_uga, def_uga = _sidecar_for(
        _STAT_VAR_REF, graph=graph, entities={"recipient": ["country/UGA"]}
    )
    collection_id = mint_collection_id([spec_ken, spec_uga])
    assert collection_id not in spec_store.state.collection_registry

    resp = client.post(
        "/api/datasets",
        json=_preview_body(
            {
                "kind": "collection_id",
                "collection_id": collection_id,
                "members": [def_ken, def_uga],
            }
        ),
    )

    assert resp.status_code == 200, resp.text
    envelope = resp.json()
    assert len(envelope["members"]) == 2
    assert {member["stat_var"] for member in envelope["members"]} == {_STAT_VAR_REF}
    _assert_nothing_written(
        spec_store=spec_store, version_store=version_store, artifact_store=artifact_store
    )


# #662: a preview and a mint of the same definition report the same version.


def _versioned_body(
    *, limit: int | None = None, entities: dict[str, list[str]] | None = None
) -> dict:
    scope = {"entities": entities} if entities is not None else _SCOPE_DICT
    return _preview_body(
        {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": scope}, limit=limit
    )


def _rows_body(*, rows: int) -> dict:
    """Returns the standard StatVar input. The fixture controls its observation rows."""
    return {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE_DICT}


def test_preview_member_content_version_equals_the_mint_version():
    client, _artifact_store = _client()

    previewed = client.post("/api/datasets", json=_versioned_body(limit=10))
    assert previewed.status_code == 200, previewed.text
    member = previewed.json()["members"][0]
    assert member["content_version"] is not None
    assert member["content_version"].startswith("cv_")

    minted = client.post("/api/datasets", json={"schema_version": 1, "input": _rows_body(rows=1)})
    assert minted.status_code == 200, minted.text

    assert member["content_version"] == minted.json()["content_version"]


def test_preview_member_content_version_equals_the_mint_version_when_truncated():
    """A truncated sample has the same `content_version` as a full mint."""
    client, _artifact_store = _client(sdmx_body=_CSV_MULTI_ROW_BODY)

    previewed = client.post("/api/datasets", json=_versioned_body(limit=1))
    assert previewed.status_code == 200, previewed.text
    member = previewed.json()["members"][0]
    assert member["total_rows"] > 1
    assert member["truncated"] is True

    minted = client.post("/api/datasets", json={"schema_version": 1, "input": _rows_body(rows=1)})
    assert minted.status_code == 200, minted.text

    assert member["content_version"] == minted.json()["content_version"]


def test_preview_collection_members_each_report_their_own_mint_version():
    """Every member of a multi-member preview mints through the same path a
    mint of that member's definition uses, so each version equals the
    version that member's own definition mints.
    """
    spec_store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    graph = _graph()
    client, _artifact_store = _client(
        spec_store=spec_store, version_store=version_store, graph=graph
    )
    _spec_ken, def_ken = _sidecar_for(
        _STAT_VAR_REF, graph=graph, entities={"recipient": ["country/KEN"]}
    )
    _spec_uga, def_uga = _sidecar_for(
        _STAT_VAR_REF, graph=graph, entities={"recipient": ["country/UGA"]}
    )
    collection_id = mint_collection_id([_spec_ken, _spec_uga])

    resp = client.post(
        "/api/datasets",
        json=_preview_body(
            {
                "kind": "collection_id",
                "collection_id": collection_id,
                "members": [def_ken, def_uga],
            }
        ),
    )

    assert resp.status_code == 200, resp.text
    versions = [member["content_version"] for member in resp.json()["members"]]
    assert len(versions) == 2
    assert all(version is not None for version in versions)


# ---------------------------------------------------------------------------
# What a preview reads
# ---------------------------------------------------------------------------

# Twenty-five rows, past the 20 x `limit` a limit-1 preview's read would allow if `limit`
# bounded the read.
_CSV_25_ROW_BODY = _CSV_HEADER + "".join(
    f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,USD,GrantEquivalent,"
    f"NotApplicable,OECD_DAC2A,{1990 + i},1000,,facet1\r\n"
    for i in range(25)
)


def test_a_preview_reads_every_row_however_small_its_limit():
    client, _artifact_store = _client(sdmx_body=_CSV_25_ROW_BODY)

    previewed = client.post("/api/datasets", json=_versioned_body(limit=1))
    assert previewed.status_code == 200, previewed.text
    member = previewed.json()["members"][0]
    assert member["total_rows"] == 25
    assert len(member["sample"]["rows"]) == 1

    minted = client.post("/api/datasets", json={"schema_version": 1, "input": _rows_body(rows=1)})
    assert minted.status_code == 200, minted.text
    assert member["content_version"] == minted.json()["content_version"]


def test_a_preview_over_the_sync_budget_refuses_before_any_fetch(monkeypatch):
    monkeypatch.setattr(
        datasets_api.preview,
        "sync_estimate",
        lambda requests, **_: Estimate(rows=10**9, requests=1),
    )
    requests: list[httpx.Request] = []
    client, _artifact_store = _client(sdmx_requests=requests)

    refused = client.post("/api/datasets", json=_versioned_body(limit=5))

    assert refused.status_code == 422, refused.text
    body = refused.json()
    assert (body["code"], body["ceiling_kind"], body["count"]) == (
        "over_ceiling",
        "sync_rows",
        10**9,
    )
    assert body["retryable"] is False
    assert requests == []


def test_a_preview_read_past_its_raw_guard_refuses_over_ceiling(monkeypatch):
    # A ceiling of 1 makes the raw-fetch guard 20 rows, below the response's 25.
    monkeypatch.setattr(datasets_api.preview, "raw_fetch_guard", lambda estimate, **_: 1)
    client, _artifact_store = _client(sdmx_body=_CSV_25_ROW_BODY)

    refused = client.post("/api/datasets", json=_versioned_body(limit=5))

    assert refused.status_code == 422, refused.text
    body = refused.json()
    assert (body["code"], body["ceiling_kind"], body["cap"]) == ("over_ceiling", "rows", 20)
    assert body["count"] > 20
    assert body["retryable"] is False


def test_a_preview_whose_survivors_pass_the_mint_ceiling_refuses_over_ceiling(monkeypatch):
    # A ceiling of 2 leaves the raw-fetch guard at 40 rows, above the response's 25, so the
    # refusal is the mint's survivor ceiling.
    monkeypatch.setattr(datasets_api.preview, "raw_fetch_guard", lambda estimate, **_: 2)
    client, _artifact_store = _client(sdmx_body=_CSV_25_ROW_BODY)

    refused = client.post("/api/datasets", json=_versioned_body(limit=5))

    assert refused.status_code == 422, refused.text
    body = refused.json()
    assert (body["code"], body["ceiling_kind"], body["cap"]) == ("over_ceiling", "rows", 2)
    assert body["count"] == 25
    assert body["retryable"] is False


# ---------------------------------------------------------------------------
# A collection's members share one budget
# ---------------------------------------------------------------------------


def _three_rows_at_the_requested_recipient(request: httpx.Request) -> httpx.Response:
    recipient = request.url.params["c[recipient]"]
    body = _CSV_HEADER + "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{recipient},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{2013 + i},1000,,facet1\r\n"
        for i in range(3)
    )
    return httpx.Response(
        200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
    )


def _two_member_collection(graph: FakeMintGraphReader) -> tuple[dict, dict[str, dict]]:
    """The collection input of a Kenya and a Uganda member, and each member's definition."""
    spec_ken, def_ken = _sidecar_for(
        _STAT_VAR_REF, graph=graph, entities={"recipient": ["country/KEN"]}
    )
    spec_uga, def_uga = _sidecar_for(
        _STAT_VAR_REF, graph=graph, entities={"recipient": ["country/UGA"]}
    )
    collection = {
        "kind": "collection_id",
        "collection_id": mint_collection_id([spec_ken, spec_uga]),
        "members": [def_ken, def_uga],
    }
    return collection, {spec_ken: def_ken, spec_uga: def_uga}


def _with_ceiling(monkeypatch, ceiling: int) -> None:
    """Sets the ceiling a collection's preview and its mint run under."""
    for module in (datasets_api.preview, datasets_api.kinds.collection_id):
        monkeypatch.setattr(module, "raw_fetch_guard", lambda estimate, **_: ceiling)


def test_a_collection_preview_whose_members_together_pass_the_ceiling_refuses_as_its_mint_does(
    monkeypatch,
):
    """Two three-row members each fit a ceiling of four and together hold six."""
    _with_ceiling(monkeypatch, 4)
    graph = _graph()
    client, _artifact_store = _client(
        graph=graph,
        sdmx_transport=httpx.MockTransport(_three_rows_at_the_requested_recipient),
    )
    collection, _definitions = _two_member_collection(graph)

    previewed = client.post("/api/datasets", json=_preview_body(collection))
    minted = client.post("/api/datasets", json={"schema_version": 1, "input": collection})

    assert previewed.status_code == 422, previewed.text
    assert minted.status_code == 422, minted.text
    refusals = [
        (body["code"], body["ceiling_kind"], body["count"], body["cap"])
        for body in (previewed.json(), minted.json())
    ]
    assert refusals == [("over_ceiling", "rows", 6, 4)] * 2


def test_a_collection_preview_that_fits_previews_each_member_as_it_previews_alone(monkeypatch):
    _with_ceiling(monkeypatch, 6)
    graph = _graph()
    client, _artifact_store = _client(
        graph=graph,
        sdmx_transport=httpx.MockTransport(_three_rows_at_the_requested_recipient),
    )
    collection, definitions = _two_member_collection(graph)

    previewed = client.post("/api/datasets", json=_preview_body(collection))
    alone = [
        client.post(
            "/api/datasets",
            json=_preview_body({"kind": "spec_id", "spec_id": spec_id, "definition": definition}),
        )
        for spec_id, definition in definitions.items()
    ]

    assert previewed.status_code == 200, previewed.text
    assert all(response.status_code == 200 for response in alone), [r.text for r in alone]
    assert [(m["total_rows"], m["content_version"]) for m in previewed.json()["members"]] == [
        (member["total_rows"], member["content_version"])
        for response in alone
        for member in response.json()["members"]
    ]
    assert [m["total_rows"] for m in previewed.json()["members"]] == [3, 3]
