"""Derived reads: the manifest of a frame the runner stored, with its retention, recipe, roots,
citation and the series every step of its recipe left out, and its rows as files, through the
app factory and live store SQL."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from typing import Any, cast

import pyarrow.parquet as pq
import pytest
import rfc8785
from fastapi.testclient import TestClient

import datasets_api.doors.analysis as analysis_door
from artifact_store import Tier
from dataset_store.snapshot import ManifestFacts, StoredVersion
from datasets_api.analysis import AnalysisRequest, run_analysis
from datasets_api.config import DatasetsBuildSettings
from datasets_api.derived import roots_citation
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.render import frame_file_location, stored_frame_location
from datasets_api.version_store import LiveVersionLister
from materialise.formats import canonical_frame_document, read_canonical_frame_table
from materialise.frame_build import build_frame
from spec_registry import LiveSpecRegistry

from ._analysis_doubles import (
    AID,
    AID_V1,
    POPULATION,
    POPULATION_V1,
    FakeRootSource,
    Stores,
    dataset,
    step,
)
from ._app_factories import make_client
from ._render_inputs import SITE
from .conftest import FakeDatasetStore


def _aid() -> Any:
    """ODA to three recipients over 2021 and 2022, each recipient a party."""
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
                "provenance": "dcid:dc/base/OECD_CRS",
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


def _roots(stores: Stores) -> FakeRootSource:
    return FakeRootSource(
        frames={(AID, AID_V1): _aid(), (POPULATION, POPULATION_V1): _population()},
        newest={AID: AID_V1, POPULATION: POPULATION_V1},
        recipes=stores.recipe_store,
    )


def _stored(stores: Stores) -> dict[str, Any]:
    result = run_analysis(
        AnalysisRequest.model_validate({"tree": _CHAIN}),
        source=_roots(stores),
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
        recipe_store=LiveRecipeStore(store=stores.store),
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


def test_an_unknown_frame_refuses_snapshot_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    stores = Stores()
    with _client(stores, monkeypatch) as client:
        manifest = client.get(f"/api/datasets/derived/ops_{'a' * 32}/ocv_{'b' * 32}")
        file = client.get(f"/api/datasets/derived/ops_{'a' * 32}/ocv_{'b' * 32}/file")

    assert manifest.status_code == 404
    assert manifest.json()["code"] == "snapshot_not_found"
    assert file.status_code == 404
    assert file.json()["code"] == "snapshot_not_found"


class _Versions:
    """Root versions holding one citation each."""

    def __init__(self, citations: dict[tuple[str, str], dict[str, Any]]) -> None:
        self.citations = citations

    def read_version(self, subject_id: str, content_version: str) -> StoredVersion | None:
        citation = self.citations.get((subject_id, content_version))
        if citation is None:
            return None
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
                trace={},
                warnings=(),
            ),
        )


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
