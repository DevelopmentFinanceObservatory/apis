"""The spec door mints a composite definition as one frame of its components' rows.

The identity binds `provenance/ONEBilateralAndImputedMultilateral`, which composes CRS
bilateral education aid to Kenya with the ONE imputed multilateral sectors' share of it. The
SDMX double serves rows written for these tests, so they prove the mint's rules over the
fixture index's members, not recorded figures.
"""

from __future__ import annotations

import csv
import gc
import io
import threading
import weakref
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from artifact_store import Tier
from code_index.composites import (
    DECLARED_COMPOSITES,
    CompositeNotServable,
    component_identities,
)
from code_index.declarations import CRS_INCLUSIVE, DAC2A_INCLUSIVE
from code_index.shared import get_index
from datasets_api import serve
from datasets_api.adapter import resolve_definition_members
from datasets_api.config import DatasetsBuildSettings
from datasets_api.groupby import refuse_provenance_disagreement, refuse_stat_var_collision
from datasets_api.version_store import LiveHeadStore
from materialise.reshape import RollupNotSummableError, Table
from materialise.trace import MemberFamily, RecordedFamily, StatedFamily, recorded_family_json
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import recompute_identity
from spec_registry.reader import StoredDefinition
from statvar_members import MemberEnumerator
from tests.code_index.aid_identity import (
    CRS_MEMBER_COUNT,
    CRS_PROVENANCE,
    IMPUTED_MEMBER_COUNT,
    IMPUTED_PROVENANCE,
    education_aid_identity,
)
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.materialise import test_composite as composite_test

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._member_bounds import mint_bounds
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door import _client

_STACKED = DECLARED_COMPOSITES[CRS_INCLUSIVE]
_PROVIDER = "country/FRA"
_RECIPIENT = "country/KEN"
_CRS_FACET = "facet_crs"
_IMPUTED_FACET = "facet_imputed"
_CRS_YEARS = (2019, 2020, 2021)
_IMPUTED_YEARS = (2020, 2021)

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,financingProvider,financingRecipient,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,"
    "facetId\r\n"
)


def _identity() -> dict:
    return education_aid_identity(source=_STACKED.provenance)


def _members() -> dict[str, list[str]]:
    """The fixture index's members of each component, by role, as the mint resolves them."""
    resolved = resolve_definition_members(
        _identity(),
        spec_id="spec_test",
        graph=FakeMintGraphReader(),
        enumerator=MemberEnumerator(),
        code_index=get_index(),
        bounds=mint_bounds(),
    )
    members: dict[str, list[str]] = {}
    for shape in resolved.shapes:
        assert shape.component is not None
        members.setdefault(shape.component.role, []).append(shape.stat_var)
    return members


def _populated() -> dict[str, tuple[str, int, tuple[int, ...]]]:
    """Two CRS members and one imputed member holding rows: each one's facet, value and
    years, by bare dcid."""
    members = _members()
    crs = [member.removeprefix("dcid:") for member in members["bilateral"][:2]]
    imputed = members["imputed"][0].removeprefix("dcid:")
    return {
        crs[0]: (_CRS_FACET, 100, _CRS_YEARS),
        crs[1]: (_CRS_FACET, 50, _CRS_YEARS),
        imputed: (_IMPUTED_FACET, 7, _IMPUTED_YEARS),
    }


_DAC2A_FACET = "facet_dac2a"
_DAC2A = "provenance/OECDDAC2A"
# The provenance each facet's SDMX rows and coverage carry.
_SDMX_PROVENANCE = {
    _CRS_FACET: CRS_PROVENANCE,
    _IMPUTED_FACET: IMPUTED_PROVENANCE,
    _DAC2A_FACET: _DAC2A,
}


def _sdmx_response(
    populated: Mapping[str, tuple[str, int, tuple[int, ...]]], request: httpx.Request
) -> httpx.Response:
    """The rows `populated` holds for the members `request` names, as SDMX CSV."""
    provenance = _SDMX_PROVENANCE
    named = str(request.url.params.get("c[variableMeasured]", "")).split(",")
    body = _CSV_HEADER + "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{dcid},{_PROVIDER},{_RECIPIENT},USD,GrantEquivalent,"
        f"NotApplicable,{provenance[facet]},{year},{value},,{facet}\r\n"
        for dcid, (facet, value, years) in populated.items()
        if dcid in named
        for year in years
    )
    return httpx.Response(
        200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
    )


def _sdmx_transport(
    populated: Mapping[str, tuple[str, int, tuple[int, ...]]],
) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: _sdmx_response(populated, request))


def _coverage(populated: Mapping[str, tuple[str, int, tuple[int, ...]]]) -> FakeCoverageService:
    index = build_coverage_index(
        [
            (dcid, facet, _PROVIDER, _RECIPIENT, list(years))
            for dcid, (facet, _value, years) in populated.items()
        ],
        role_names={
            dcid: RoleNames(donor="financingProvider", recipient="financingRecipient")
            for dcid in populated
        },
        facet_metadata={
            facet: FacetMetadata(
                provenance_id=provenance, measurement_method="GrantEquivalent", unit="USD"
            )
            for facet, provenance in _SDMX_PROVENANCE.items()
        },
    )
    return FakeCoverageService(index)


def _composite_client(
    version_store: FakeVersionRecordDatasetStore | None = None,
    *,
    populated: Mapping[str, tuple[str, int, tuple[int, ...]]] | None = None,
    settings: DatasetsBuildSettings | None = None,
    transport: httpx.MockTransport | None = None,
    sdmx_hedge_after: float | None = None,
) -> tuple[FakeArtifactStore | None, TestClient]:
    """A client over the composite's rows (`_populated` unless given), and the artifact store
    its renderings are put in. With `version_store`, the client writes heads and records there
    and builds its own artifact store."""
    populated = _populated() if populated is None else populated
    graph = FakeMintGraphReader(labels={_PROVIDER: "France", _RECIPIENT: "Kenya"})
    transport = _sdmx_transport(populated) if transport is None else transport
    if version_store is not None:
        return None, _client(
            graph=graph,
            sdmx_transport=transport,
            coverage=_coverage(populated),
            version_store=version_store,
            head_writer=LiveHeadStore(store=version_store),
            settings=settings,
        )
    artifacts = FakeArtifactStore()
    return artifacts, _client(
        graph=graph,
        sdmx_transport=transport,
        coverage=_coverage(populated),
        artifact_store=artifacts,
        settings=settings,
        sdmx_hedge_after=sdmx_hedge_after,
    )


def _mint(
    client,
    *,
    identity: Mapping[str, object] | None = None,
    options: Mapping[str, object] | None = None,
) -> httpx.Response:
    identity = _identity() if identity is None else identity
    spec_id = recompute_identity(identity).spec_id
    return client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {
                "kind": "spec_id",
                "spec_id": spec_id,
                "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
            },
            "options": {"bundle": False, **(options or {})},
        },
    )


def _csv_rows(artifacts: FakeArtifactStore, manifest: dict) -> list[dict]:
    """The rows of `manifest`'s CSV rendering."""
    artifact = next(a for a in manifest["artifacts"] if a["rendering"].get("format") == "csv")
    data = artifacts.get(artifact_digest=artifact["artifact_digest"], tier=Tier.working)
    return list(csv.DictReader(io.StringIO(data.decode())))


def test_the_mint_resolves_each_component_from_its_own_table():
    members = _members()

    assert {role: len(refs) for role, refs in members.items()} == {
        "bilateral": CRS_MEMBER_COUNT,
        "imputed": IMPUTED_MEMBER_COUNT,
    }
    assert not set(members["bilateral"]) & set(members["imputed"])


def test_a_composite_mints_one_frame_over_the_years_both_components_hold():
    _artifacts, client = _composite_client()

    minted = _mint(client)

    assert minted.status_code == 200, minted.text
    manifest = minted.json()
    by_role = {entry["role"]: entry for entry in manifest["trace"]["components"]}
    assert set(by_role) == {"bilateral", "imputed"}
    assert by_role["bilateral"]["provenance"] == CRS_PROVENANCE
    assert by_role["imputed"]["provenance"] == IMPUTED_PROVENANCE
    assert by_role["bilateral"]["dropped_years"] == [2019]
    # Every imputed member but the one holding rows gets a zero row in each covered cell.
    assert by_role["imputed"]["zero_rows"] == (IMPUTED_MEMBER_COUNT - 1) * len(_IMPUTED_YEARS)
    crs_rows = 2 * len(_IMPUTED_YEARS)
    assert manifest["coverage"]["rows"] == crs_rows + IMPUTED_MEMBER_COUNT * len(_IMPUTED_YEARS)
    assert [source["ref"] for source in manifest["citation"]["sources"]] == [
        CRS_PROVENANCE,
        IMPUTED_PROVENANCE,
    ]
    provenance_facets = [f for f in manifest["trace"]["facets"] if f["axis"] == "provenance"]
    assert [f["ref"] for f in provenance_facets] == [_STACKED.provenance]


def test_a_pooled_hedged_mint_writes_the_serial_mint_s_bytes():
    """The first SDMX request stalls until the mint has returned and then answers other values,
    so its duplicate answers it. The pooled mint's frame, CSV and content version match a mint
    that reads one request at a time, so the duplicate's answer is the one the mint used."""
    populated = _populated()
    stale = {
        dcid: (facet, value + 1000, years) for dcid, (facet, value, years) in populated.items()
    }
    lock = threading.Lock()
    sent: Counter[str] = Counter()
    stalled: list[str] = []
    stalled_until_done = threading.Event()

    def stall_first(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        with lock:
            sent[url] += 1
            stall = not stalled
            if stall:
                stalled.append(url)
        if stall:
            stalled_until_done.wait(5)
            return _sdmx_response(stale, request)
        return _sdmx_response(populated, request)

    def mint(concurrency: int, transport: httpx.MockTransport, hedge_after: float | None):
        artifacts, client = _composite_client(
            settings=DatasetsBuildSettings.from_env().model_copy(
                update={"mint_member_max_concurrency": concurrency}
            ),
            transport=transport,
            sdmx_hedge_after=hedge_after,
        )
        assert artifacts is not None
        minted = _mint(client)
        assert minted.status_code == 200, minted.text
        manifest = minted.json()
        return manifest["content_version"], _csv_rows(artifacts, manifest)

    try:
        pooled = mint(4, httpx.MockTransport(stall_first), 0.05)
    finally:
        stalled_until_done.set()
    serial = mint(1, _sdmx_transport(populated), None)

    [stalled_url] = stalled
    assert sent[stalled_url] == 2
    assert max(sent.values()) == 2
    assert pooled[1]
    assert pooled == serial
    # The stalled request names members with rows, so its stale values would have shown.
    assert any(dcid in stalled_url for dcid in populated)


def test_a_groupby_collapse_of_a_composite_sums_both_components():
    artifacts, client = _composite_client()
    assert artifacts is not None

    minted = _mint(client, options={"groupby": {"by": ["time_period"], "op": "sum"}})

    assert minted.status_code == 200, minted.text
    manifest = minted.json()
    rows = _csv_rows(artifacts, manifest)
    assert {row["time_period"]: float(row["value"]) for row in rows} == {
        "2020": 100 + 50 + 7,
        "2021": 100 + 50 + 7,
    }
    assert all(_STACKED.label in row["stat_var_label"] for row in rows)


def test_a_sector_code_and_its_own_purpose_codes_each_count_once_in_the_sum():
    """The imputed table publishes spending imputed against a three-digit sector code (114)
    beside spending imputed against its five-digit purpose codes (11420). `crsPurpose` declares
    a code's rows disjoint from its children's, and the table's rows sum to its source total,
    so a sum over both counts each dollar once: the total is every row's value added once."""
    members = _members()
    imputed = {member.removeprefix("dcid:") for member in members["imputed"]}
    parent = next(dcid for dcid in imputed if dcid.endswith("_CRSPurposeCode114_Nominal"))
    child = next(dcid for dcid in imputed if dcid.endswith("_CRSPurposeCode11420_Nominal"))
    crs = members["bilateral"][0].removeprefix("dcid:")
    populated = {
        crs: (_CRS_FACET, 100, _IMPUTED_YEARS),
        parent: (_IMPUTED_FACET, 7, _IMPUTED_YEARS),
        child: (_IMPUTED_FACET, 3, _IMPUTED_YEARS),
    }
    artifacts, client = _composite_client(populated=populated)
    assert artifacts is not None

    minted = _mint(client, options={"groupby": {"by": ["time_period"], "op": "sum"}})

    assert minted.status_code == 200, minted.text
    rows = _csv_rows(artifacts, minted.json())
    assert {row["time_period"]: float(row["value"]) for row in rows} == {
        "2020": 100 + 7 + 3,
        "2021": 100 + 7 + 3,
    }


def test_the_estimate_counts_the_zero_rows_and_refuses_before_any_read():
    """The admission estimate counts every imputed member once per cell the composite's
    series hold (`CoverageIndex.cell_count`), so the row ceiling bounds the composed frame and
    a composite over it refuses before any SDMX request."""
    populated = _populated()
    read_rows = sum(len(years) for _facet, _value, years in populated.values())
    cells = len({year for _facet, _value, years in populated.values() for year in years})
    composed_bound = read_rows + IMPUTED_MEMBER_COUNT * cells
    requests: list[httpx.Request] = []

    def _counted(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _sdmx_response(populated, request)

    _artifacts, client = _composite_client(
        populated=populated,
        settings=DatasetsBuildSettings(sync_budget_rows=1, async_max_rows=read_rows + 1),
        transport=httpx.MockTransport(_counted),
    )

    refused = _mint(client)

    assert refused.status_code == 422, refused.text
    problem = refused.json()
    assert problem["code"] == "over_ceiling"
    assert problem["count"] == composed_bound
    assert requests == []


def test_the_stored_version_re_renders_the_composite_frame_and_its_citations():
    version_store = FakeVersionRecordDatasetStore()
    _artifacts, client = _composite_client(version_store)
    minted = _mint(client)
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]

    served = client.get(f"/api/datasets/{spec_id}?bundle=false&format=parquet")

    assert served.status_code == 200, served.text
    manifest = served.json()
    assert manifest["trace"]["served_from"] == "head"
    assert manifest["content_version"] == minted.json()["content_version"]
    assert manifest["trace"]["components"] == minted.json()["trace"]["components"]
    assert manifest["citation"] == minted.json()["citation"]


def test_a_recheck_of_a_composite_head_states_no_one_family():
    """A composite's components each read their own family, so the freshness check that
    confirms its head writes no members (`pipeline.definition_family`), whatever the head
    stated before."""
    version_store = FakeVersionRecordDatasetStore()
    _artifacts, client = _composite_client(version_store)
    minted = _mint(client)
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]
    # A head confirmed past the recheck window, inside the maximum age, stating members.
    confirmed_at = datetime.now(UTC) - timedelta(hours=1)
    stated = RecordedFamily(
        stated=StatedFamily(members=MemberFamily(digest="a" * 64, count=3)),
        confirmed_at=confirmed_at,
        basis="earlier-basis",
    )
    with version_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.subject_head SET confirmed_at = %s, family = %s WHERE subject_id = %s",
            (confirmed_at, recorded_family_json(stated), spec_id),
        )

    served = client.get(f"/api/datasets/{spec_id}?bundle=false&format=parquet")

    assert served.status_code == 200, served.text
    assert served.json()["trace"]["served_from"] == "head"
    head = LiveHeadStore(store=version_store).read_head(spec_id)
    assert head is not None and head.family is not None
    assert head.family.confirmed_at == head.confirmed_at > confirmed_at
    assert head.family.stated.members is None


def test_a_rebuild_of_a_composite_version_on_record_reads_no_origin_findings(monkeypatch):
    """A version on record serves the origin findings it recorded, so a rebuild that lands on
    it reads none for any component."""
    version_store = FakeVersionRecordDatasetStore()
    _artifacts, client = _composite_client(version_store)
    reads: list[object] = []
    origin_findings = serve._origin_findings

    def counted(*args, **kwargs):
        reads.append(args[0])
        return origin_findings(*args, **kwargs)

    monkeypatch.setattr(serve, "_origin_findings", counted)
    minted = _mint(client)
    assert minted.status_code == 200, minted.text
    first_mint_reads = len(reads)
    # A head past its maximum age rebuilds, and its version is on record.
    with version_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.subject_head SET confirmed_at = %s, materialised_at = %s "
            "WHERE subject_id = %s",
            (
                "2000-01-01T00:00:00+00:00",
                "2000-01-01T00:00:00+00:00",
                minted.json()["definition"]["spec_id"],
            ),
        )

    rebuilt = _mint(client)

    assert rebuilt.status_code == 200, rebuilt.text
    assert rebuilt.json()["content_version"] == minted.json()["content_version"]
    assert rebuilt.json()["trace"]["served_from"] != "head"
    assert first_mint_reads == 2
    assert len(reads) == first_mint_reads


def test_a_composite_s_component_frames_are_freed_before_its_origin_findings(monkeypatch):
    """A component run keeps only what its origin findings read, so no component's
    materialisation outlives the composition into the findings' graph reads."""
    _artifacts, client = _composite_client()
    component_results: list[weakref.ref] = []
    alive_at_findings: list[int] = []
    compose = serve.compose_components
    component_findings = serve._component_findings

    def composing(results, **kwargs):
        component_results.extend(weakref.ref(component.result) for component in results)
        return compose(results, **kwargs)

    def finding(runs, **kwargs):
        gc.collect()
        alive_at_findings.append(sum(ref() is not None for ref in component_results))
        return component_findings(runs, **kwargs)

    monkeypatch.setattr(serve, "compose_components", composing)
    monkeypatch.setattr(serve, "_component_findings", finding)

    minted = _mint(client)

    assert minted.status_code == 200, minted.text
    assert len(component_results) == 2
    assert alive_at_findings == [0]


def _same_purpose_pair() -> tuple[str, str]:
    """A CRS member and an imputed member of one purpose code."""
    index = get_index()
    rows = {row.dcid: row for row in index.statvars}
    members = _members()
    imputed = {
        rows[member.removeprefix("dcid:")].constraint_properties.get("crsPurpose"): member
        for member in members["imputed"]
    }
    for member in members["bilateral"]:
        purpose = rows[member.removeprefix("dcid:")].constraint_properties.get("crsPurpose")
        if purpose in imputed:
            return member, imputed[purpose]
    raise AssertionError("no CRS and imputed member share a purpose code")


def test_two_tables_in_one_frame_still_refuse_a_collapse_outside_a_composite():
    crs, imputed = _same_purpose_pair()
    table = Table(
        columns=("stat_var", "time_period", "value"),
        rows=((crs, "2020", 1.0), (imputed, "2020", 2.0)),
    )

    with pytest.raises(RollupNotSummableError) as ambiguous:
        refuse_stat_var_collision(table, index=get_index(), decoded_code_systems=["crsPurpose"])
    with pytest.raises(RollupNotSummableError) as disagreeing:
        refuse_provenance_disagreement(provenance_refs=[CRS_PROVENANCE, IMPUTED_PROVENANCE])

    assert ambiguous.value.precondition == "stat_var_ambiguous"
    assert disagreeing.value.precondition == "provenance_disagreement"
    refuse_stat_var_collision(
        table, index=get_index(), decoded_code_systems=["crsPurpose"], composite=_STACKED
    )
    refuse_provenance_disagreement(provenance_refs=[_STACKED.provenance])


def test_a_composite_binding_a_slot_the_imputed_rows_do_not_declare_is_refused():
    identity = _identity()
    for slot in identity["slots"]:
        if slot["property"] == "crsCooperationModality":
            slot["binding"] = {"state": "value", "ref": "CRSCooperationModalityC01"}

    with pytest.raises(CompositeNotServable) as refused:
        component_identities(get_index(), identity)

    assert refused.value.pointer == "/slots/0"


@pytest.mark.parametrize(
    ("edit", "detail"),
    [
        pytest.param(
            lambda identity: identity.update(requested_window={"kind": "latest"}),
            "latest window",
            id="latest-window",
        ),
        pytest.param(
            lambda identity: identity["slots"][0].update(
                binding={"state": "value", "ref": "CRSCooperationModalityC01"}
            ),
            "crsCooperationModality",
            id="bound-modality",
        ),
    ],
)
def test_the_door_refuses_a_composite_it_cannot_compose_before_any_read(edit, detail):
    _artifacts, client = _composite_client()
    identity = _identity()
    edit(identity)
    spec_id = recompute_identity(identity).spec_id

    refused = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {
                "kind": "spec_id",
                "spec_id": spec_id,
                "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
            },
            "options": {"bundle": False},
        },
    )

    assert refused.status_code == 400, refused.text
    problem = refused.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/input/spec_id"
    assert detail in problem["detail"]


def test_each_component_run_is_charged_only_the_rows_the_composite_keeps(monkeypatch):
    """The imputed table covers 2020 and 2021, so the bilateral run's 2019 rows are dropped
    from the composite and leave the ceiling to the imputed run."""
    populated = _populated()
    crs_rows = [
        (f"dcid:{dcid}", _RECIPIENT, year, float(value))
        for dcid, (facet, value, years) in populated.items()
        if facet == _CRS_FACET
        for year in years
    ]
    imputed_rows = [
        (f"dcid:{dcid}", _RECIPIENT, year, float(value))
        for dcid, (facet, value, years) in populated.items()
        if facet == _IMPUTED_FACET
        for year in years
    ]
    results = {
        "bilateral": composite_test._component(
            _STACKED,
            "bilateral",
            crs_rows,
            provenance=CRS_PROVENANCE,
            members=sorted({row[0] for row in crs_rows}),
        ),
        "imputed": composite_test._component(
            _STACKED,
            "imputed",
            imputed_rows,
            provenance=IMPUTED_PROVENANCE,
            members=sorted({row[0] for row in imputed_rows}),
        ),
    }
    ceilings: dict[str, int] = {}

    def _materialise(part, *, ceiling, precomputed_probe):
        role = part[0].component
        ceilings[role] = ceiling
        return results[role].result, None

    monkeypatch.setattr(serve, "_origin_findings", lambda *a, **k: serve._OriginFindings((), ()))
    identity = _identity()
    components = component_identities(get_index(), identity)
    assert components is not None
    ceiling = 1_000

    serve._materialise_components(
        [request for result in results.values() for request in result.requests],
        components=components,
        ceiling=ceiling,
        materialise=_materialise,
        stored=StoredDefinition(
            spec_id="spec_test",
            schema_version=1,
            identity=identity,
            identity_bytes=b"",
            identity_digest="",
        ),
        code_index=get_index(),
        graph=FakeMintGraphReader(),
        enrichment=composite_test._ENRICHMENT,
        deadline=None,
        coverage_index=_coverage(populated).snapshot().index,
    )

    kept_crs_rows = sum(1 for row in crs_rows if row[2] in _IMPUTED_YEARS)
    assert kept_crs_rows < len(crs_rows)
    assert ceilings == {"bilateral": ceiling, "imputed": ceiling - kept_crs_rows}


_DAC2A_IMPUTED = "AttributedAmount_FinancialFlow_Gross_Disbursement_DAC2aAidType106_Nominal"


def _dac2a_oda_identity(netting: str) -> dict:
    """DAC2a ODA from France to Kenya on the `netting` basis, including imputed multilateral
    ODA (`DAC2A_INCLUSIVE`)."""
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "FinancialFlow",
            "measured_property": "amount",
            "stat_type": "measuredValue",
            "measurement_qualifier": "Nominal",
        },
        "slots": [
            {"property": "dac2aAidType", "binding": {"state": "value", "ref": "DAC2aAidType206"}},
            {"property": "financialFlowType", "binding": {"state": "value", "ref": "Disbursement"}},
            {"property": "nettingBasis", "binding": {"state": "value", "ref": netting}},
        ],
        "entities": {"origin": [_PROVIDER], "destination": [_RECIPIENT]},
        "requested_window": {"kind": "all"},
        "facets": {"provenance": {"state": "value", "ref": DAC2A_INCLUSIVE}},
    }


@pytest.mark.parametrize("netting", ["Gross", "Net"])
def test_a_time_sum_of_the_dac2a_inclusive_composite_adds_total_oda_and_imputed_oda(
    netting: str,
):
    """DAC2a's aid type is no summable axis, so only the composite's declaration makes its
    two components additive: a sum over time adds total ODA on the asked basis to the gross
    imputed multilateral line, on gross and net alike."""
    bilateral = f"Amount_FinancialFlow_{netting}_Disbursement_DAC2aAidType206_Nominal"
    populated = {
        bilateral: (_DAC2A_FACET, 100, _IMPUTED_YEARS),
        _DAC2A_IMPUTED: (_DAC2A_FACET, 7, _IMPUTED_YEARS),
    }
    artifacts, client = _composite_client(populated=populated)
    assert artifacts is not None
    identity = _dac2a_oda_identity(netting)

    minted = _mint(
        client, identity=identity, options={"groupby": {"by": ["time_period"], "op": "sum"}}
    )

    assert minted.status_code == 200, minted.text
    rows = _csv_rows(artifacts, minted.json())
    assert {row["time_period"]: float(row["value"]) for row in rows} == {
        "2020": 100 + 7,
        "2021": 100 + 7,
    }
