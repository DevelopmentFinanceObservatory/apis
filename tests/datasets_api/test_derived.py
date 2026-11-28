"""Derived reads: the manifest of a frame the runner stored, with its retention, recipe, roots,
citation and the series every step of its recipe left out, and its rows as files, through the
app factory and live store SQL."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pyarrow.parquet as pq
import pytest
import rfc8785
from fastapi.testclient import TestClient

import datasets_api.doors.analysis as analysis_door
from artifact_store import Tier
from code_index.composites import DECLARED_COMPOSITES
from code_index.declarations import DAC2A_BILATERAL
from dataset_store.snapshot import ManifestFacts, StoredVersion
from datasets_api.analysis import AnalysisRequest, run_analysis
from datasets_api.config import DatasetsBuildSettings
from datasets_api.derived import root_reads, roots_citation
from datasets_api.doors.derived import DerivedManifestRoot, link_cache_control
from datasets_api.render import frame_file_location, stored_frame_location
from datasets_api.retention import WORKING_TIER_RETENTION
from datasets_api.version_store import LiveVersionLister
from materialise.formats import canonical_frame_document, read_canonical_frame_table
from materialise.frame_build import build_frame
from spec_registry import LiveSpecRegistry
from tests.materialise._ledger import LEDGER, MEMBERS, SOLO, ledger_index

from ._analysis_doubles import (
    AID,
    AID_V1,
    AID_V2,
    POPULATION,
    POPULATION_V1,
    FakeRootSource,
    Stores,
    dataset,
    step,
)
from ._app_factories import make_client
from ._deps_doubles import closable_permits
from ._frame_doubles import link_frame1_only
from ._release_doubles import IndexServingRelease, bound_index
from ._render_inputs import SITE
from .conftest import FakeDatasetStore


def _aid(provenance: str = "dcid:dc/base/OECD_CRS") -> Any:
    """ODA to three recipients over 2021 and 2022, each recipient a party, read from
    `provenance`."""
    return build_frame(
        role_keys=["financingRecipient", "financingRecipient_party_kind"],
        rows=[
            {
                "stat_var": "dcid:Amount_ODA",
                "financingRecipient": recipient,
                "financingRecipient_party_kind": "party",
                "time_period": year,
                "value": 10.0,
                "unit": "USDollar",
                "provenance": provenance,
            }
            for recipient in ("country/GHA", "country/KEN", "country/NGA")
            for year in ("2021", "2022")
        ],
    )


def _population() -> Any:
    """Population of Kenya in both years and of Nigeria in 2021 alone; none of Ghana."""
    return build_frame(
        role_keys=["observationAbout"],
        rows=[
            {
                "stat_var": "dcid:Count_Person",
                "observationAbout": place,
                "time_period": year,
                "value": 100.0,
                "unit": "Person",
            }
            for place, year in (
                ("country/KEN", "2021"),
                ("country/KEN", "2022"),
                ("country/NGA", "2021"),
            )
        ],
    )


# Aid per person, then its change from 2021 to 2022. The quotient drops Ghana, which has no
# population, and Nigeria's 2022, which its population lacks; the growth then drops Nigeria,
# which lacks its 2022 endpoint.
_CHAIN = step(
    "growth",
    step(
        "quotient_of",
        dataset(AID, AID_V1),
        dataset(POPULATION, POPULATION_V1),
        on={"destination": "place"},
    ),
    window_start="2021",
    window_end="2022",
    entity_by=["financingRecipient"],
    change="absolute",
)


def _roots(stores: Stores, *, aid: Any = None) -> FakeRootSource:
    return FakeRootSource(
        frames={
            (AID, AID_V1): _aid() if aid is None else aid,
            (POPULATION, POPULATION_V1): _population(),
        },
        newest={AID: AID_V1, POPULATION: POPULATION_V1},
        recipes=stores.recipe_store,
    )


def _stored(stores: Stores, *, aid: Any = None, tree: Any = _CHAIN) -> dict[str, Any]:
    result = run_analysis(
        AnalysisRequest.model_validate({"tree": tree}),
        source=_roots(stores, aid=aid),
        stores=stores.analysis(),
    )
    return result.ref()["members"][0]


def _client(stores: Stores, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(analysis_door, "LiveRootSource", lambda **_: _roots(stores))
    return make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        settings=DatasetsBuildSettings.from_env(),
        artifact_store=stores.artifact_store,
        snapshot_reader=stores.snapshot_reader,
        version_writer=stores.version_writer,
        recipe_store=stores.recipe_store,
        version_lister=LiveVersionLister(store=stores.store),
        now=lambda: stores.now,
    )


def _path(member: dict[str, Any], suffix: str = "") -> str:
    return f"/api/datasets/derived/{member['spec']['spec_id']}/{member['content_version']}{suffix}"


def test_the_manifest_lists_what_every_step_of_the_chain_left_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = Stores()
    member = _stored(stores)

    with _client(stores, monkeypatch) as client:
        response = client.get(_path(member))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["spec_id"] == member["spec"]["spec_id"]
    assert body["retention"]["pinned"] is False
    assert body["retention"]["held_until"].startswith("2026-11-27")
    assert body["recipe"]["replay_key"] == member["spec"]["recipe"]["replay_key"]
    assert body["roots"] == member["roots"]
    assert body["resolved_periods"] == [] and body["anchor"] is None
    left_out = [
        (
            row["step"]["node"],
            row["step"]["verb"],
            row["side"],
            row["reason"],
            row["coordinates"]["financingRecipient"],
            row["rows"],
        )
        for row in body["left_out"]
    ]
    assert left_out == [
        (0, "growth", "data", "endpoint_missing", "country/NGA", 1),
        (1, "quotient_of", "data", "unmatched_key", "country/GHA", 2),
        (1, "quotient_of", "data", "period_unmatched", "country/NGA", 1),
    ]
    assert body["left_out_complete"] is True


def test_a_derived_root_alone_recomputes_its_recipe_under_its_own_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = Stores()
    member = _stored(stores)
    tree = {"spec_id": member["spec"]["spec_id"], "content_version": member["content_version"]}

    with _client(stores, monkeypatch) as client:
        response = client.post("/api/datasets/analysis", json={"tree": tree, "latest": True})

    assert response.status_code == 200, response.text
    again = response.json()["ref"]["members"][0]
    assert again["spec"]["spec_id"] == member["spec"]["spec_id"]
    assert again["content_version"] == member["content_version"]
    assert again["output_digest"] == member["output_digest"]
    assert again["artifact"]["digest"] == member["artifact"]["digest"]


@pytest.mark.parametrize(
    ("query", "media_type"),
    [
        ("format=csv&bundle=false", "text/csv"),
        ("format=parquet&bundle=false&layout=frame", "application/vnd.apache.parquet"),
        ("format=xlsx&bundle=false", None),
        ("", "application/zip"),
    ],
)
def test_the_file_route_writes_the_frame_in_each_format(
    monkeypatch: pytest.MonkeyPatch, query: str, media_type: str | None
) -> None:
    stores = Stores()
    member = _stored(stores)

    with _client(stores, monkeypatch) as client:
        response = client.get(_path(member, f"/file?{query}"))

    assert response.status_code == 200, response.text
    if media_type is not None:
        assert response.headers["content-type"].startswith(media_type)
    assert "attachment" in response.headers["content-disposition"]
    if query.startswith("format=csv"):
        rows = list(csv.DictReader(io.StringIO(response.text)))
        assert [row["financingRecipient"] for row in rows] == ["country/KEN"]
    if "parquet" in query:
        table = pq.read_table(io.BytesIO(response.content))
        assert table.column("financingRecipient").to_pylist() == ["country/KEN"]
        # Bare Parquet is the stored frame file, the bytes the ref's digest names.
        digest = "sha256:" + hashlib.sha256(response.content).hexdigest()
        assert digest == member["artifact"]["digest"]
        assert table.column("unit").to_pylist() == _unit_texts(stores, member)
    if not query:
        names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
        assert "datapackage.json" in names
        assert any(name.startswith("data/") and name.endswith(".csv") for name in names)


def test_a_bundle_of_a_frame_over_a_composite_cites_its_tables_and_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A derived frame keeps its root's composite provenance. Its bundle cites the composite's
    component table, as every other bundle does, and names the composite with what it reads
    and the destinations it keeps."""
    stores = Stores()
    # Aid's change from 2021 to 2022, a step on one root, which keeps its provenance.
    growth = step(
        "growth",
        dataset(AID, AID_V1),
        window_start="2021",
        window_end="2022",
        entity_by=["financingRecipient"],
        change="absolute",
    )
    member = _stored(stores, aid=_aid(DAC2A_BILATERAL), tree=growth)
    composite = DECLARED_COMPOSITES[DAC2A_BILATERAL]
    (table,) = {recipe.dataset for recipe in composite.components}

    with _client(stores, monkeypatch) as client:
        response = client.get(_path(member, "/file"))

    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        files = {name: bundle.read(name).decode() for name in bundle.namelist()}
    descriptor = json.loads(files["datapackage.json"])
    # The roots here carry no stored citation, so the table is named by its ref.
    assert [source["title"] for source in descriptor["sources"]] == [table]
    cut = "counting bilateral flows to countries and regions, without core contributions"
    for text in (files["README.md"], files["SOURCES.md"], descriptor["description"]):
        assert "OECD DAC2a (bilateral flows)" in text
        assert DAC2A_BILATERAL in text
        assert cut in text


@pytest.mark.parametrize("declared", [LEDGER, SOLO], ids=["two-members", "one-member"])
def test_a_bundle_of_a_frame_over_a_declared_dataset_cites_it_once_and_no_member(
    monkeypatch: pytest.MonkeyPatch, declared: str
) -> None:
    """A dataset's rows carry its id, so the derived bundle cites the dataset once, and no
    file of it names a member provenance."""
    stores = Stores()
    growth = step(
        "growth",
        dataset(AID, AID_V1),
        window_start="2021",
        window_end="2022",
        entity_by=["financingRecipient"],
        change="absolute",
    )
    monkeypatch.setattr(analysis_door, "LiveRootSource", lambda **_: _roots(stores))

    with bound_index(ledger_index()):
        member = _stored(stores, aid=_aid(declared), tree=growth)
        client = make_client(
            registry=LiveSpecRegistry(store=FakeDatasetStore()),
            settings=DatasetsBuildSettings.from_env(),
            artifact_store=stores.artifact_store,
            snapshot_reader=stores.snapshot_reader,
            version_writer=stores.version_writer,
            recipe_store=stores.recipe_store,
            version_lister=LiveVersionLister(store=stores.store),
            now=lambda: stores.now,
            release=IndexServingRelease(ledger_index()),
        )
        with client:
            manifest = client.get(_path(member))
            response = client.get(_path(member, "/file"))

    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        files = {name: bundle.read(name).decode() for name in bundle.namelist()}
    descriptor = json.loads(files["datapackage.json"])
    assert [source["title"] for source in descriptor["sources"]] == [declared]
    for text in (manifest.text, *files.values()):
        assert not [m for m in MEMBERS if m in text]


def _locations(stores: Stores, member: dict[str, Any]) -> Any:
    return stores.snapshot_reader.read_artifact_locations(
        member["spec"]["spec_id"], member["content_version"]
    )


def _unit_texts(stores: Stores, member: dict[str, Any]) -> list[str]:
    """The `unit` cells of the member's canonical frame as a flat file writes them: a ref as
    its string, a converted or composed unit as its JCS text."""
    location = stored_frame_location(_locations(stores, member))
    assert location is not None
    raw = stores.artifact_store.get(artifact_digest=location.artifact_digest, tier=Tier.working)
    document = canonical_frame_document(read_canonical_frame_table(raw))
    unit = cast("list[str]", document["columns"]).index("unit")
    rows = cast("list[list[str | dict[str, Any]]]", document["rows"])
    cells = [row[unit] for row in rows]
    assert any(isinstance(cell, dict) for cell in cells)
    return [rfc8785.dumps(cell).decode() if isinstance(cell, dict) else cell for cell in cells]


def test_a_frame_file_whose_bytes_are_gone_renders_again_to_the_ref_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the stored frame file's object deleted, bare Parquet renders again from the
    canonical frame and gives the bytes the ref's digest names."""
    stores = Stores()
    member = _stored(stores)
    frame_file = frame_file_location(_locations(stores, member), site=SITE)
    assert frame_file is not None and frame_file.artifact_digest == member["artifact"]["digest"]
    hex_digest = frame_file.artifact_digest.removeprefix("sha256:")
    deleted = 0
    for bucket in stores.gcs.buckets.values():
        for key in [key for key in bucket.objects if hex_digest in str(key)]:
            del bucket.objects[key]
            deleted += 1
    assert deleted == 1

    with _client(stores, monkeypatch) as client:
        response = client.get(_path(member, "/file?format=parquet&bundle=false&layout=frame"))

    assert response.status_code == 200, response.text
    assert "sha256:" + hashlib.sha256(response.content).hexdigest() == member["artifact"]["digest"]


def test_a_frame_stored_without_its_file_links_the_file_its_first_read_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame published before frames stored their file links none. The first bare Parquet
    read renders and stores it, and links it under the version, so later reads serve it."""
    stores = Stores()
    member = _stored(stores)
    digest = member["artifact"]["digest"]
    with stores.store.transaction() as tx:
        tx.execute("DELETE FROM datasets.version_artifact WHERE artifact_digest = %s", (digest,))
    assert frame_file_location(_locations(stores, member), site=SITE) is None

    with _client(stores, monkeypatch) as client:
        response = client.get(_path(member, "/file?format=parquet&bundle=false&layout=frame"))

    assert response.status_code == 200, response.text
    assert "sha256:" + hashlib.sha256(response.content).hexdigest() == digest
    linked = frame_file_location(_locations(stores, member), site=SITE)
    assert linked is not None and linked.artifact_digest == digest


def test_a_frame1_file_past_the_decode_bound_refuses_with_every_permit_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A derived frame stored as a frame1 document recorded at 250 MB is past the 216,600,000
    bytes a read that decodes it alone takes. Its record decides that, so the file route
    refuses `over_ceiling` (`bytes`) with every materialise permit held, where a wait for one
    would refuse a retryable 503."""
    permits = closable_permits(monkeypatch)
    stores = Stores()
    member = _stored(stores)
    digest = link_frame1_only(
        stores.store,
        stores.artifact_store,
        subject_id=member["spec"]["spec_id"],
        content_version=member["content_version"],
    )
    with stores.store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_inventory SET size_bytes = %s WHERE artifact_digest = %s",
            (250_000_000, digest),
        )

    with _client(stores, monkeypatch) as client:
        for held in permits:
            held.closed = True
        response = client.get(_path(member, "/file?format=csv&bundle=false"))

    assert response.status_code == 422, response.text
    assert (response.json()["code"], response.json()["ceiling_kind"]) == ("over_ceiling", "bytes")


def test_an_unknown_frame_refuses_snapshot_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    stores = Stores()
    with _client(stores, monkeypatch) as client:
        manifest = client.get(f"/api/datasets/derived/ops_{'a' * 32}/ocv_{'b' * 32}")
        file = client.get(f"/api/datasets/derived/ops_{'a' * 32}/ocv_{'b' * 32}/file")

    assert manifest.status_code == 404
    assert manifest.json()["code"] == "snapshot_not_found"
    assert file.status_code == 404
    assert file.json()["code"] == "snapshot_not_found"


def _version(
    subject_id: str,
    content_version: str,
    *,
    citation: dict[str, Any],
    trace: dict[str, Any] | None = None,
) -> StoredVersion:
    return StoredVersion(
        subject_id=subject_id,
        subject_kind="spec",
        content_version=content_version,
        schema_version=1,
        facts=ManifestFacts(
            prepared_at=datetime(2026, 1, 1, tzinfo=UTC),
            definition={},
            coverage={},
            citation=citation,
            trace=trace or {},
            warnings=(),
        ),
    )


class _Versions:
    """Root versions holding one citation each, and the traces `traces` names."""

    def __init__(
        self,
        citations: dict[tuple[str, str], dict[str, Any]],
        traces: dict[tuple[str, str], dict[str, Any]] | None = None,
    ) -> None:
        self.citations = citations
        self.traces = traces or {}

    def read_version(self, subject_id: str, content_version: str) -> StoredVersion | None:
        citation = self.citations.get((subject_id, content_version))
        if citation is None:
            return None
        return _version(
            subject_id,
            content_version,
            citation=citation,
            trace=self.traces.get((subject_id, content_version)),
        )


_CUT_ENTRIES = [
    {
        "role": "financingRecipient",
        "provenance": "dcid:dc/base/OECD_CRS",
        "entities": ["dcid:DACCountriesTotal"],
    },
    {
        "role": "financingRecipient",
        "provenance": None,
        "entities": ["dcid:GroupOfSeven"],
        "kept_in": [
            {
                "series": {
                    "financingProvider": "dcid:DACCountriesTotal",
                    "measurementMethod": None,
                    "stat_var": "dcid:SV",
                    "unit": {"currency": "USD", "prices": "current"},
                },
                "time_periods": ["1968"],
            }
        ],
    },
]


def test_each_root_carries_the_party_totals_its_own_version_left_out() -> None:
    citation = {"text": "OECD.", "sources": []}
    reader = _Versions(
        {(AID, AID_V1): citation, (AID, AID_V2): citation, (POPULATION, POPULATION_V1): citation},
        traces={(AID, AID_V2): {"party_totals_left_out": _CUT_ENTRIES, "other": 1}},
    )
    roots = [
        {"spec_id": AID, "content_version": AID_V1, "rendering": {"format": "csv"}},
        {
            "spec_id": AID,
            "content_version": AID_V2,
            "rendering": {"format": "csv", "without_party_totals": True},
        },
        {"spec_id": POPULATION, "content_version": POPULATION_V1, "rendering": {}},
        {"spec_id": "spec_" + "f" * 32, "content_version": "cv_" + "f" * 32, "rendering": {}},
    ]

    served, _ = root_reads(reader, roots)  # ty: ignore[invalid-argument-type] -- a reader double

    wire = [DerivedManifestRoot.model_validate(root).model_dump(mode="json") for root in served]
    assert [root.get("party_totals_left_out") for root in wire] == [
        None,
        _CUT_ENTRIES,
        None,
        None,
    ]
    assert [set(root) for root in wire] == [
        {"spec_id", "content_version", "rendering"},
        {"spec_id", "content_version", "rendering", "party_totals_left_out"},
        {"spec_id", "content_version", "rendering"},
        {"spec_id", "content_version", "rendering"},
    ]


class _CutAidReader:
    """The stores' snapshot reader, with the aid root's version a cut whose trace lists
    `_CUT_ENTRIES`."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def read_version(self, subject_id: str, content_version: str) -> StoredVersion | None:
        if (subject_id, content_version) == (AID, AID_V1):
            return _version(
                AID,
                AID_V1,
                citation={"text": "OECD.", "sources": []},
                trace={"party_totals_left_out": _CUT_ENTRIES},
            )
        return self._inner.read_version(subject_id, content_version)


def test_the_derived_manifest_alone_names_the_totals_a_root_left_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = Stores()
    monkeypatch.setattr(analysis_door, "LiveRootSource", lambda **_: _roots(stores))
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        settings=DatasetsBuildSettings.from_env(),
        artifact_store=stores.artifact_store,
        snapshot_reader=_CutAidReader(stores.snapshot_reader),
        version_writer=stores.version_writer,
        recipe_store=stores.recipe_store,
        version_lister=LiveVersionLister(store=stores.store),
        now=lambda: stores.now,
    )

    with client:
        answer = client.post("/api/datasets/analysis", json={"tree": _CHAIN})
        member = answer.json()["ref"]["members"][0]
        manifest = client.get(_path(member))

    assert answer.status_code == 200, answer.text
    assert [set(root) for root in member["roots"]] == [
        {"spec_id", "content_version", "rendering"},
        {"spec_id", "content_version", "rendering"},
    ]
    assert manifest.status_code == 200, manifest.text
    roots = manifest.json()["roots"]
    assert [(root["spec_id"], root.get("party_totals_left_out")) for root in roots] == [
        (AID, _CUT_ENTRIES),
        (POPULATION, None),
    ]
    assert [{**root, "party_totals_left_out": None} for root in roots] == [
        {**root, "party_totals_left_out": None} for root in member["roots"]
    ]


def test_the_citation_names_each_roots_sources_once() -> None:
    crs = {"ref": "dcid:dc/base/OECD_CRS", "label": "OECD CRS"}
    wdi = {"ref": "dcid:dc/base/WorldBank_WDI", "label": "World Bank WDI"}
    reader = _Versions(
        {
            (AID, AID_V1): {"text": "OECD.", "sources": [crs]},
            (POPULATION, POPULATION_V1): {"text": "World Bank.", "sources": [wdi, crs]},
        }
    )
    roots = [
        {"spec_id": AID, "content_version": AID_V1},
        {"spec_id": POPULATION, "content_version": POPULATION_V1},
        {"spec_id": "spec_" + "f" * 32, "content_version": "cv_" + "f" * 32},
    ]

    citation = roots_citation(reader, roots)  # ty: ignore[invalid-argument-type] -- a reader double

    assert citation.text == "OECD.\n\nWorld Bank."
    assert [source.ref for source in citation.sources] == [crs["ref"], wdi["ref"]]
    assert json.loads(citation.model_dump_json())["sources"][1]["label"] == "World Bank WDI"


def _link(member: dict[str, Any]) -> str:
    return _path(member, f"/files/{member['artifact']['digest']}")


@pytest.fixture
def _signing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sign as a URL the object name, with no V4 date, so the redirect's life is the shortest a
    reused signature leaves."""
    monkeypatch.setattr(
        "artifact_store.store.sign_get_url",
        lambda **kwargs: f"https://storage.googleapis.com/b/{kwargs['object_name']}",
    )


def _pin(stores: Stores, member: dict[str, Any]) -> None:
    """Move every object of the member's version to the pinned tier."""
    digests = [location.artifact_digest for location in _locations(stores, member)]
    with stores.store.transaction() as tx:
        for digest in digests:
            tx.execute(
                "UPDATE datasets.artifact_location SET tier = %s, bucket = %s "
                "WHERE artifact_digest = %s",
                ("pinned", "pinned-bucket", digest),
            )


def _held_for(stores: Stores, member: dict[str, Any], seconds: int) -> None:
    """Store the member's objects so that the working-tier hold ends `seconds` from now."""
    stored_at = stores.now - WORKING_TIER_RETENTION + timedelta(seconds=seconds)
    digests = [location.artifact_digest for location in _locations(stores, member)]
    with stores.store.transaction() as tx:
        for digest in digests:
            tx.execute(
                "UPDATE datasets.artifact_location SET stored_at = %s WHERE artifact_digest = %s",
                (stored_at.isoformat(), digest),
            )


@pytest.mark.parametrize(
    ("hold", "tier", "cache_control"),
    [
        # The signed URL lives at least 240 s (its 5 minutes less the 1-minute reuse window),
        # and a cached redirect keeps 60 s of it.
        pytest.param(None, "working", "public, max-age=180", id="working-held-long"),
        pytest.param(100, "working", "public, max-age=100", id="working-held-100s"),
        pytest.param(30, "working", "no-store", id="working-held-30s"),
        pytest.param("pinned", "pinned", "public, max-age=180", id="pinned"),
    ],
)
@pytest.mark.usefixtures("_signing")
def test_the_digest_link_redirects_cacheably_within_the_url_life_and_the_hold(
    monkeypatch: pytest.MonkeyPatch, hold: int | str | None, tier: str, cache_control: str
) -> None:
    stores = Stores()
    member = _stored(stores)
    if hold == "pinned":
        _pin(stores, member)
    elif isinstance(hold, int):
        _held_for(stores, member, hold)

    with _client(stores, monkeypatch) as client:
        response = client.get(_link(member), follow_redirects=False)

    assert response.status_code == 302, response.text
    assert response.headers["cache-control"] == cache_control
    assert response.headers["x-dfo-artifact-tier"] == tier
    assert member["artifact"]["digest"].removeprefix("sha256:") in response.headers["location"]


def test_the_digest_link_reads_the_v4_expiry_of_the_signed_url() -> None:
    now = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
    url = (
        "https://storage.googleapis.com/b/o?X-Goog-Algorithm=GOOG4-RSA-SHA256"
        "&X-Goog-Date=20261122T115900Z&X-Goog-Expires=300&X-Goog-Signature=f"
    )

    assert link_cache_control(url, now=now, held_until=None) == "public, max-age=180"
    assert link_cache_control(url, now=now + timedelta(seconds=150), held_until=None) == (
        "no-store"
    )


def test_the_digest_link_refuses_as_the_dataset_link_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = Stores()
    member = _stored(stores)
    unknown = _path(member, "/files/sha256:" + "f" * 64)
    absent = f"/api/datasets/derived/ops_{'a' * 32}/ocv_{'b' * 32}/files/sha256:{'f' * 64}"
    with _client(stores, monkeypatch) as client:
        not_linked = client.get(unknown, follow_redirects=False)
        not_found = client.get(absent, follow_redirects=False)

    assert (not_linked.status_code, not_linked.json()["code"]) == (404, "artifact_not_found")
    assert (not_found.status_code, not_found.json()["code"]) == (404, "snapshot_not_found")
    assert not_linked.headers["cache-control"] == "no-store"


@pytest.mark.usefixtures("_signing")
def test_the_digest_link_renews_a_file_past_retention_whose_bytes_are_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frame file is past working retention and its object is deleted: the link renders
    it again from the stored frame, the same digest, and redirects to it."""
    stores = Stores()
    member = _stored(stores)
    digest = member["artifact"]["digest"]
    with stores.store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_location SET stored_at = %s WHERE artifact_digest = %s",
            ((stores.now - timedelta(days=60)).isoformat(), digest),
        )
    hex_digest = digest.removeprefix("sha256:")
    for bucket in stores.gcs.buckets.values():
        for key in [key for key in bucket.objects if hex_digest in str(key)]:
            del bucket.objects[key]

    with _client(stores, monkeypatch) as client:
        response = client.get(_link(member), follow_redirects=False)

    assert response.status_code == 302, response.text
    assert hex_digest in response.headers["location"]
