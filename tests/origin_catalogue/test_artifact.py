"""Tests the artifact round trip, the manifest the writer computes, and the loader's refusals."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from origin_catalogue import UN_SYSTEM, CataloguePointer
from origin_catalogue.artifact import (
    CatalogueArtifactError,
    CatalogueTooLargeError,
    load_catalogue,
    publish_build,
    read_published_pointer,
    resident_bytes,
)
from origin_catalogue.schema import (
    CATALOGUE_FORMAT_VERSION,
    FORMAT_ROOT,
    UnCarryForward,
    identity_key,
)
from tests.origin_catalogue.support import (
    BUILT_AT,
    CENSUS_POVERTY,
    EUROSTAT_GDP,
    GDP,
    POPULATION,
    UN_WHO,
    WDI,
    WDI_DATASET,
    encode_sample,
    publish_sample,
    sample_tables,
)


def _uri(tmp_path: Path) -> str:
    return f"file://{tmp_path / 'catalogue'}"


def _build_dir(tmp_path: Path, build_id: str = "build-a") -> Path:
    return tmp_path / "catalogue" / FORMAT_ROOT / "versions" / build_id


def test_round_trip_reproduces_the_tables(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    snapshot = load_catalogue(uri)
    assert snapshot.tables == sample_tables()
    assert snapshot.un_system_enabled
    assert not load_catalogue(uri, un_system_enabled=False).un_system_enabled
    assert snapshot.admit_candidate("undata/who/MORT_U5_RATE") == UN_SYSTEM
    assert snapshot.admit_candidate(CENSUS_POVERTY) is None
    assert snapshot.base_places(GDP) == ("country/KEN", "country/USA", "nuts/DE11")


def test_manifest_records_counts_overlaps_identity_and_size(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    build = publish_sample(uri)
    manifest = load_catalogue(uri).manifest
    assert manifest == build.manifest
    assert manifest.built_at == BUILT_AT
    assert manifest.build_seconds == 12.5
    assert manifest.registry_hash == "r1"
    assert manifest.overlaps == (POPULATION,)
    assert manifest.row_counts == {
        "provenances": 4,
        "provenances_dfo": 1,
        "provenances_base": 2,
        "provenances_un_system": 1,
        "dfo_variables": 2,
        "base_variables": 2,
        "provenance_places": 2,
        "withheld_variables": 1,
        "topics": 1,
        "scan_places": 5,
        "un_blocked_places": 2,
        "dataset_facts": 3,
    }
    assert manifest.resident_bytes == resident_bytes(load_catalogue(uri).tables)
    assert manifest.un_carried_forward is None
    assert read_published_pointer(uri) == CataloguePointer(
        build_id="build-a",
        identity=identity_key({"registry": "r1", "dfo": "f1", "week": "2026-W40"}),
    )


def test_loaded_rows_share_one_object_per_provenance_list(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    tables = load_catalogue(uri).tables
    (pid,) = tables.base_variables[POPULATION].provenances
    assert pid is next(key for key in tables.provenances if key == WDI)


def test_identity_key_ignores_input_order() -> None:
    assert identity_key({"a": "1", "b": "2"}) == identity_key({"b": "2", "a": "1"})
    assert identity_key({"a": "1"}) != identity_key({"a": "2"})


def test_encode_build_refuses_a_naive_build_time() -> None:
    with pytest.raises(ValueError, match="UTC offset"):
        encode_sample(built_at=BUILT_AT.replace(tzinfo=None))


def test_publish_refuses_to_rewrite_a_build_directory(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    build = publish_sample(uri)
    with pytest.raises(CatalogueArtifactError, match="already exists"):
        publish_build(uri, build, expected_generation=None)


def test_publish_loses_the_pointer_race_on_a_stale_generation(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    assert not publish_build(uri, encode_sample(build_id="build-b"), expected_generation=None)
    assert load_catalogue(uri).build_id == "build-a"


def test_load_refuses_a_build_over_the_memory_budget_before_reading_tables(
    tmp_path: Path,
) -> None:
    uri = _uri(tmp_path)
    build = publish_sample(uri)
    (_build_dir(tmp_path) / "base_variables.parquet").unlink()
    with pytest.raises(CatalogueTooLargeError, match="over the"):
        load_catalogue(uri, max_resident_bytes=build.manifest.resident_bytes - 1)
    with pytest.raises(CatalogueArtifactError, match="missing"):
        load_catalogue(uri, max_resident_bytes=build.manifest.resident_bytes)


def test_load_refuses_a_missing_pointer(tmp_path: Path) -> None:
    with pytest.raises(CatalogueArtifactError, match="missing"):
        load_catalogue(_uri(tmp_path))
    assert read_published_pointer(_uri(tmp_path)) is None


def test_load_refuses_a_pointer_the_manifest_disagrees_with(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    pointer = tmp_path / "catalogue" / FORMAT_ROOT / "current.json"
    pointer.write_text(json.dumps({"build_id": "build-a", "identity": "other"}))
    with pytest.raises(CatalogueArtifactError, match="the manifest at"):
        load_catalogue(uri)


def test_load_refuses_another_format_version(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    path = _build_dir(tmp_path) / "manifest.json"
    payload = json.loads(path.read_text())
    payload["format_version"] = CATALOGUE_FORMAT_VERSION - 1
    path.write_text(json.dumps(payload))
    with pytest.raises(
        CatalogueArtifactError, match=f"format_version {CATALOGUE_FORMAT_VERSION - 1}"
    ):
        load_catalogue(uri)


def test_load_refuses_row_counts_the_manifest_does_not_record(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    path = _build_dir(tmp_path) / "manifest.json"
    payload = json.loads(path.read_text())
    payload["row_counts"]["base_variables"] = 3
    path.write_text(json.dumps(payload))
    with pytest.raises(CatalogueArtifactError, match="row counts"):
        load_catalogue(uri)


def _rewrite_base_variables(tmp_path: Path, table: pa.Table) -> None:
    sink = io.BytesIO()
    pq.write_table(table, sink)
    (_build_dir(tmp_path) / "base_variables.parquet").write_bytes(sink.getvalue())


def test_load_refuses_a_bitset_of_the_wrong_width(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    _rewrite_base_variables(
        tmp_path,
        pa.table(
            {
                "variable": [GDP, POPULATION],
                "provenances": [[WDI], [WDI]],
                "places": [b"\x01\x00", b"\x01"],
            }
        ),
    )
    with pytest.raises(CatalogueArtifactError, match="2-byte place bitset"):
        load_catalogue(uri)


def _rewrite_provenance_places(tmp_path: Path, table: pa.Table) -> None:
    sink = io.BytesIO()
    pq.write_table(table, sink)
    (_build_dir(tmp_path) / "provenance_places.parquet").write_bytes(sink.getvalue())


def test_load_refuses_a_provenance_bitset_of_the_wrong_width(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    _rewrite_provenance_places(
        tmp_path,
        pa.table({"provenance": [EUROSTAT_GDP, WDI], "places": [b"\x10\x00", b"\x07"]}),
    )
    with pytest.raises(CatalogueArtifactError, match="2-byte place bitset"):
        load_catalogue(uri)


def test_load_refuses_a_variable_place_none_of_its_provenances_holds(
    tmp_path: Path,
) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    # WDI holds GDP at Kenya and the USA; the NUTS region is Eurostat's alone.
    _rewrite_provenance_places(
        tmp_path,
        pa.table({"provenance": [EUROSTAT_GDP, WDI], "places": [b"\x10", b"\x07"]}),
    )
    assert load_catalogue(uri).tables.provenance_places == {EUROSTAT_GDP: 0b10000, WDI: 0b00111}
    _rewrite_provenance_places(tmp_path, pa.table({"provenance": [WDI], "places": [b"\x07"]}))
    with pytest.raises(CatalogueArtifactError, match="has places none of its provenances"):
        load_catalogue(uri)


def test_load_refuses_a_base_variable_under_a_provenance_outside_the_table(
    tmp_path: Path,
) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    _rewrite_base_variables(
        tmp_path,
        pa.table(
            {
                "variable": [GDP, POPULATION],
                "provenances": [[WDI], [UN_WHO]],
                "places": [b"\x01", b"\x01"],
            }
        ),
    )
    with pytest.raises(CatalogueArtifactError, match="must be a base provenance row"):
        load_catalogue(uri)


def test_load_refuses_a_null_provenance(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    _rewrite_base_variables(
        tmp_path,
        pa.table(
            {
                "variable": [GDP, POPULATION],
                "provenances": [[WDI, None], [WDI]],
                "places": [b"\x01", b"\x01"],
            }
        ),
    )
    with pytest.raises(CatalogueArtifactError, match="null"):
        load_catalogue(uri)


def test_load_refuses_an_unknown_origin(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    path = _build_dir(tmp_path) / "provenances.json"
    rows = json.loads(path.read_text())
    rows[0]["data_commons"] = "wikidata"
    path.write_text(json.dumps(rows))
    with pytest.raises(CatalogueArtifactError, match="wikidata"):
        load_catalogue(uri)


def test_load_refuses_a_build_whose_manifest_names_no_dataset_facts(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    path = _build_dir(tmp_path) / "manifest.json"
    payload = json.loads(path.read_text())
    del payload["files"]["dataset_facts"]
    path.write_text(json.dumps(payload))
    (_build_dir(tmp_path) / "dataset_facts.json").unlink()
    with pytest.raises(CatalogueArtifactError, match="names no 'dataset_facts' file"):
        load_catalogue(uri)


def test_load_refuses_dataset_facts_that_miss_an_entry(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    path = _build_dir(tmp_path) / "dataset_facts.json"
    path.write_text(
        json.dumps([f for f in json.loads(path.read_text()) if f["ref"] != WDI_DATASET])
    )
    with pytest.raises(CatalogueArtifactError, match="dataset facts miss 1"):
        load_catalogue(uri)


def test_dataset_facts_store_each_entry_s_card_facts(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    facts = {
        f["ref"]: f for f in json.loads((_build_dir(tmp_path) / "dataset_facts.json").read_text())
    }
    assert facts[WDI_DATASET] == {
        "ref": WDI_DATASET,
        "description": None,
        "license_type": "Creative Commons Attribution License",
        "update_cadence": "P3M",
        "years_covered": "1960-2024",
        "source_url": "https://datacatalog.worldbank.org/dataset/world-development-indicators",
    }
    assert WDI not in facts
    assert UN_WHO not in facts


def test_provenance_rows_carry_their_dataset_or_null(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    rows = {
        row["id"]: row
        for row in json.loads((_build_dir(tmp_path) / "provenances.json").read_text())
    }
    assert rows[WDI]["dataset"] == {
        "ref": "dc/d/WorldBank_WorldDevelopmentIndicators",
        "label": "World Development Indicators",
    }
    assert rows[UN_WHO]["dataset"] is None


def test_manifest_records_a_un_carry_forward(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    carried = UnCarryForward(from_build="build-0", reason="UN deployment read failed: HTTP 503")
    publish_sample(uri, un_carried_forward=carried)
    assert load_catalogue(uri).manifest.un_carried_forward == carried
