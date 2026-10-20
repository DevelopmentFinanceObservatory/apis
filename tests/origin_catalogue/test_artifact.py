"""Tests the artifact round trip, the manifest the writer computes, and the loader's refusals."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from origin_catalogue import UN_SYSTEM, CataloguePointer, artifact
from origin_catalogue.arc_tables import EMPTY_ARC_TABLES, ArcTables
from origin_catalogue.artifact import (
    CatalogueArtifactError,
    CatalogueTooLargeError,
    encode_build,
    load_catalogue,
    publish_build,
    read_arc_tables,
    read_published_pointer,
    read_un_mirror,
    resident_bytes,
)
from origin_catalogue.families import FamilyKey, family_tables, summarise_table
from origin_catalogue.schema import (
    ARC_TARGETS_FILE,
    BASE_ARCS_FILE,
    CATALOGUE_FORMAT_VERSION,
    DFO_ARCS_FILE,
    FAMILIES_FILE,
    FORMAT_ROOT,
    SCHEMA_ARCS_FILE,
    UN_TOPICS_FILE,
    CatalogueManifest,
    UnCarryForward,
    identity_key,
)
from origin_catalogue.un_mirror import EMPTY_UN_MIRROR, wire_arcs
from tests.origin_catalogue.support import (
    BUILT_AT,
    CENSUS_POVERTY,
    CRS,
    CRS_VARIABLE,
    EUROSTAT_GDP,
    GDP,
    HOUSEHOLDS,
    POPULATION,
    UN_SERIES,
    UN_VARIABLE,
    UN_VARIABLE_FEMALE,
    UN_WHO,
    WDI,
    WDI_DATASET,
    build_arc_tables,
    encode_sample,
    publish_sample,
    sample_arc_tables,
    sample_tables,
    sample_un_mirror,
)


def _uri(tmp_path: Path) -> str:
    return f"file://{tmp_path / 'catalogue'}"


def _build_dir(tmp_path: Path, build_id: str = "build-a") -> Path:
    return tmp_path / "catalogue" / FORMAT_ROOT / "versions" / build_id


def _rehash(tmp_path: Path) -> CatalogueManifest:
    """Records in the build's manifest the SHA-256 of each file as it now stands, so a
    test that rewrites a table reaches the decoder's own checks. Returns the manifest."""
    path = _build_dir(tmp_path) / "manifest.json"
    payload = json.loads(path.read_text())
    payload["file_sha256"] = {
        logical: hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
        for logical, name in payload["files"].items()
        if (path.parent / name).exists()
    }
    path.write_text(json.dumps(payload))
    return CatalogueManifest.from_json(path.read_text())


def test_round_trip_reproduces_the_tables(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri, un_mirror=sample_un_mirror())
    snapshot = load_catalogue(uri)
    assert snapshot.tables == sample_tables()
    assert snapshot.un.mirror == sample_un_mirror()
    assert snapshot.un_system_enabled
    assert not load_catalogue(uri, un_system_enabled=False).un_system_enabled
    assert snapshot.admit_candidate(UN_VARIABLE) == UN_SYSTEM
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
        "source_documentation": 3,
        "superseded_variables": 0,
        "successor_variables": 0,
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
    manifest = publish_sample(uri, un_mirror=sample_un_mirror(), arcs=sample_arc_tables()).manifest
    needed = (
        manifest.resident_bytes
        + manifest.un_mirror.resident_bytes
        + manifest.arc_tables.resident_bytes
        + manifest.families.resident_bytes
    )
    assert manifest.un_mirror.resident_bytes > 1
    assert manifest.arc_tables.resident_bytes > 1
    assert manifest.families.resident_bytes > 1
    (_build_dir(tmp_path) / "base_variables.parquet").unlink()
    with pytest.raises(CatalogueTooLargeError, match="over the"):
        load_catalogue(uri, max_resident_bytes=needed - 1)
    with pytest.raises(CatalogueArtifactError, match="missing"):
        load_catalogue(uri, max_resident_bytes=needed)


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
    _rehash(tmp_path)


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
    _rehash(tmp_path)


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
    _rehash(tmp_path)
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
    _rehash(tmp_path)
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


def test_source_documentation_stores_each_entry_s_documentation_facts(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    documentation = {
        d["ref"]: d
        for d in json.loads((_build_dir(tmp_path) / "source_documentation.json").read_text())
    }
    assert documentation[CRS] == {
        "ref": CRS,
        "provenances": [CRS],
        "publisher": "OECD",
        "source_data_url": "https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS.parquet",
        "citation": "OECD (2026), Creditor Reporting System",
        "license_url": "https://www.oecd.org/en/about/terms-conditions.html",
        "last_update": "2026-10-02",
        "next_expected_update": "2027-01-02",
        "documents": {"dataNotes": ["- Figures for the latest year are preliminary.\\n"]},
    }
    assert set(documentation) == {CRS, WDI_DATASET, EUROSTAT_GDP}


def test_provenance_rows_carry_their_dataset_or_null_and_their_licence(tmp_path: Path) -> None:
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
    assert rows[WDI]["licence"] == {
        "type": "CreativeCommonsAttributionLicense",
        "label": "Creative Commons Attribution License",
        "url": None,
    }
    assert rows[UN_WHO]["licence"] == {"type": None, "label": None, "url": None}


def test_manifest_records_a_un_carry_forward(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    carried = UnCarryForward(from_build="build-0", reason="UN deployment read failed: HTTP 503")
    publish_sample(uri, un_carried_forward=carried)
    assert load_catalogue(uri).manifest.un_carried_forward == carried


# --- UN tables ---------------------------------------------------------------


def _publish_with_un_tables(tmp_path: Path) -> tuple[str, CatalogueManifest]:
    uri = _uri(tmp_path)
    build = publish_sample(uri, un_mirror=sample_un_mirror())
    return uri, build.manifest


def test_the_superseded_variable_table_round_trips_and_is_counted(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    superseded = {CENSUS_POVERTY: UN_VARIABLE_FEMALE, "old/X": None}
    tables = replace(
        sample_tables(),
        withheld_variables=frozenset(superseded),
        superseded=superseded,
    )
    build = encode_build(
        tables,
        build_id="build-a",
        built_at=BUILT_AT,
        build_seconds=1.0,
        identity_inputs={"week": "w"},
        registry_hash="r1",
        un_mirror=sample_un_mirror(),
        arcs=sample_arc_tables(),
    )
    assert publish_build(uri, build, expected_generation=None)

    snapshot = load_catalogue(uri)

    assert snapshot.tables.superseded == superseded
    assert snapshot.successor(CENSUS_POVERTY) == UN_VARIABLE_FEMALE
    assert snapshot.is_superseded("old/X")
    assert snapshot.successor("old/X") is None
    counts = build.manifest.row_counts
    assert (counts["superseded_variables"], counts["successor_variables"]) == (2, 1)
    assert build.manifest.files["superseded_variables"] == "superseded_variables.parquet"


def test_load_refuses_a_build_without_its_superseded_variable_table(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    (_build_dir(tmp_path) / "superseded_variables.parquet").unlink()

    with pytest.raises(CatalogueArtifactError, match=r"superseded_variables\.parquet"):
        load_catalogue(uri)


def test_load_refuses_a_successor_the_un_tables_do_not_admit(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri, un_mirror=sample_un_mirror())
    path = _build_dir(tmp_path) / "superseded_variables.parquet"
    pq.write_table(pa.table({"variable": [CENSUS_POVERTY], "successor": ["undata/who/GONE"]}), path)
    manifest_path = _build_dir(tmp_path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["row_counts"]["superseded_variables"] = 1
    manifest["row_counts"]["successor_variables"] = 1
    manifest_path.write_text(json.dumps(manifest))
    _rehash(tmp_path)

    with pytest.raises(CatalogueArtifactError, match="not admitted UN variables"):
        load_catalogue(uri)


def test_the_un_tables_round_trip_beside_the_tables_the_loader_reads(tmp_path: Path) -> None:
    uri, manifest = _publish_with_un_tables(tmp_path)

    assert read_un_mirror(uri, manifest) == sample_un_mirror()
    assert load_catalogue(uri).tables == sample_tables()
    assert manifest.un_mirror.row_counts == {
        "un_variables": 2,
        "un_arcs": 6,
        "un_excluded_variables": 1,
        "un_excluded_series": 1,
        "un_series": 1,
        "un_topics": 1,
    }
    assert manifest.un_mirror.agencies == sample_un_mirror().agencies
    assert manifest.un_mirror.resident_bytes == resident_bytes(read_un_mirror(uri, manifest))
    assert CatalogueManifest.from_json(manifest.to_json()) == manifest


def test_a_build_without_un_agencies_writes_empty_un_tables(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    build = publish_sample(uri)

    assert read_un_mirror(uri, build.manifest) == EMPTY_UN_MIRROR
    assert set(build.manifest.un_mirror.row_counts.values()) == {0}


def test_load_refuses_un_tables_the_manifest_disagrees_with(tmp_path: Path) -> None:
    uri, _manifest = _publish_with_un_tables(tmp_path)
    # A well-formed topic table of another build: it decodes, and holds no topic.
    (_build_dir(tmp_path) / UN_TOPICS_FILE).write_bytes(encode_sample().files[UN_TOPICS_FILE])
    _rehash(tmp_path)

    with pytest.raises(CatalogueArtifactError, match="UN row counts"):
        load_catalogue(uri)


def test_decoded_un_variables_share_one_object_per_arc_row_and_series(tmp_path: Path) -> None:
    uri, manifest = _publish_with_un_tables(tmp_path)

    variables = read_un_mirror(uri, manifest).variables
    plain, female = variables[UN_VARIABLE], variables[UN_VARIABLE_FEMALE]
    assert plain.series is female.series
    shared = [arc for arc in plain.arcs if any(arc is other for other in female.arcs)]
    assert {arc.property for arc in shared} == {"measuredProperty", "populationType"}


def test_un_tables_decode_a_batch_at_a_time_and_read_no_file_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri, manifest = _publish_with_un_tables(tmp_path)
    whole: list[object] = []
    monkeypatch.setattr(artifact.pq, "read_table", lambda *args, **kwargs: whole.append(args))
    monkeypatch.setattr(artifact, "_UN_DECODE_ROWS", 1)

    mirror = read_un_mirror(uri, manifest)

    assert whole == []
    assert mirror == sample_un_mirror()
    # Rows of different batches share their strings and arc rows.
    plain, female = mirror.variables[UN_VARIABLE], mirror.variables[UN_VARIABLE_FEMALE]
    assert plain.series is female.series
    assert any(arc is other for arc in plain.arcs for other in female.arcs)


def test_reading_un_tables_refuses_counts_the_manifest_does_not_record(tmp_path: Path) -> None:
    uri, manifest = _publish_with_un_tables(tmp_path)
    recorded = replace(
        manifest.un_mirror, row_counts={**manifest.un_mirror.row_counts, "un_variables": 3}
    )

    with pytest.raises(CatalogueArtifactError, match="UN row counts"):
        read_un_mirror(uri, replace(manifest, un_mirror=recorded))


def test_reading_un_tables_refuses_tables_that_break_their_invariants(tmp_path: Path) -> None:
    uri, manifest = _publish_with_un_tables(tmp_path)
    path = _build_dir(tmp_path) / "un_variables.parquet"
    pq.write_table(pq.read_table(path).slice(0, 1), path)
    manifest = _rehash(tmp_path)

    with pytest.raises(CatalogueArtifactError, match="UnMirrorTableError"):
        read_un_mirror(uri, manifest)


def test_reading_un_tables_refuses_an_arc_row_out_of_range(tmp_path: Path) -> None:
    uri, manifest = _publish_with_un_tables(tmp_path)
    arcs = _build_dir(tmp_path) / "un_arcs.parquet"
    pq.write_table(pq.read_table(arcs).slice(0, 2), arcs)
    manifest = _rehash(tmp_path)

    with pytest.raises(CatalogueArtifactError, match="out of range"):
        read_un_mirror(uri, manifest)


def test_loading_refuses_a_un_variable_no_agency_prefix_starts(tmp_path: Path) -> None:
    mirror = sample_un_mirror()
    who = mirror.agencies["who"]
    foreign = replace(
        mirror,
        variables={**mirror.variables, "undata/unicef/X": mirror.variables[UN_VARIABLE]},
        agencies={"who": replace(who, variables=who.variables + 1)},
    )
    uri = _uri(tmp_path)
    publish_sample(uri, un_mirror=foreign)

    with pytest.raises(CatalogueArtifactError, match="no agency's id prefix starts"):
        load_catalogue(uri)


def test_reading_un_tables_refuses_a_missing_file(tmp_path: Path) -> None:
    uri, manifest = _publish_with_un_tables(tmp_path)
    (_build_dir(tmp_path) / "un_mirror.json").unlink()

    with pytest.raises(CatalogueArtifactError, match="missing"):
        read_un_mirror(uri, manifest)


def test_reading_un_tables_requires_the_excluded_series_names(tmp_path: Path) -> None:
    uri, manifest = _publish_with_un_tables(tmp_path)
    path = _build_dir(tmp_path) / "un_mirror.json"
    payload = json.loads(path.read_bytes())
    del payload["series_names"]
    path.write_text(json.dumps(payload))
    manifest = _rehash(tmp_path)

    with pytest.raises(CatalogueArtifactError, match="series_names"):
        read_un_mirror(uri, manifest)


# --- Arc tables --------------------------------------------------------------


def test_the_arc_tables_round_trip_and_the_manifest_counts_and_measures_them(
    tmp_path: Path,
) -> None:
    uri = _uri(tmp_path)
    manifest = publish_sample(uri, arcs=sample_arc_tables()).manifest
    snapshot = load_catalogue(uri)

    assert snapshot.arcs == sample_arc_tables()
    assert read_arc_tables(uri, manifest) == sample_arc_tables()
    assert manifest.arc_tables.row_counts == {
        "dfo_arcs": 1,
        "base_arcs": 2,
        "schema_arcs": 4,
        "dfo_arcs_empty": 0,
        "base_arcs_empty": 1,
        "schema_arcs_empty": 1,
        "arc_targets": 8,
    }
    assert manifest.arc_tables.resident_bytes == resident_bytes(snapshot.arcs)
    assert CatalogueManifest.from_json(manifest.to_json()) == manifest


def _arcs_moving_gdp_to_dfo(*, empty: bool) -> ArcTables:
    """The sample arc tables with GDP, a variable listed as base alone, left out of the
    `base` table, and held empty in the `dfo` table when `empty`."""
    arcs = sample_arc_tables()
    return build_arc_tables(
        dfo={
            **{node: wire_arcs(rows) for node, rows in arcs.dfo.items()},
            **({GDP: None} if empty else {}),
        },
        base={node: wire_arcs(rows) for node, rows in arcs.base.items() if node != GDP},
        schema={node: wire_arcs(rows) for node, rows in arcs.schema.items()},
        labels=arcs.labels,
    )


def test_a_build_without_arcs_for_its_listed_variables_is_refused() -> None:
    """A listed variable no arc table holds would read live, so encoding refuses it."""
    with pytest.raises(CatalogueArtifactError, match="hold none of 3 listed") as refused:
        encode_sample(arcs=EMPTY_ARC_TABLES)
    assert all(node in str(refused.value) for node in (CRS_VARIABLE, GDP, POPULATION))
    with pytest.raises(CatalogueArtifactError, match=rf"hold none of 1 listed.*{GDP}"):
        encode_sample(arcs=_arcs_moving_gdp_to_dfo(empty=False))


def test_equal_arc_tables_encode_to_equal_bytes_whatever_the_input_order() -> None:
    arcs = sample_arc_tables()
    reordered = build_arc_tables(
        **{
            name: {node: wire_arcs(table[node]) for node in reversed(list(table))}
            for name, table in (("dfo", arcs.dfo), ("base", arcs.base), ("schema", arcs.schema))
        },
        labels=arcs.labels,
    )

    first = encode_sample(arcs=arcs).files
    second = encode_sample(arcs=reordered).files

    assert reordered == arcs
    for name in (DFO_ARCS_FILE, BASE_ARCS_FILE, SCHEMA_ARCS_FILE, ARC_TARGETS_FILE):
        assert first[name] == second[name]


def test_decoded_arc_tables_share_one_row_per_target_across_tables(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri, arcs=sample_arc_tables())
    arcs = load_catalogue(uri).arcs

    crs_type = arcs.dfo[CRS_VARIABLE][-1]
    households_type = arcs.schema[HOUSEHOLDS][-1]
    assert crs_type.property == "typeOf"
    assert crs_type is households_type


@pytest.mark.parametrize(
    ("arcs", "table"),
    [
        pytest.param(
            build_arc_tables(
                dfo={CENSUS_POVERTY: {"name": {"nodes": [{"value": "Poverty"}]}}},
                base={},
                schema={},
            ),
            "dfo",
            id="dfo-node-with-arcs-not-a-dfo-variable",
        ),
        pytest.param(
            build_arc_tables(dfo={}, base={}, schema={GDP: None}),
            "schema",
            id="schema-node-a-listed-variable",
        ),
        pytest.param(
            build_arc_tables(
                dfo={},
                base={CENSUS_POVERTY: {"name": {"nodes": [{"value": "Poverty"}]}}},
                schema={},
            ),
            "base",
            id="base-node-a-withheld-variable",
        ),
        pytest.param(
            build_arc_tables(dfo={CENSUS_POVERTY: None}, base={}, schema={}),
            "dfo",
            id="dfo-empty-row-a-withheld-variable",
        ),
        pytest.param(
            _arcs_moving_gdp_to_dfo(empty=True), "dfo", id="dfo-empty-row-a-listed-base-variable"
        ),
    ],
)
def test_encoding_refuses_arc_tables_holding_a_node_outside_their_scope(
    arcs: object, table: str
) -> None:
    with pytest.raises(CatalogueArtifactError, match=f"the {table} arc table"):
        encode_sample(arcs=arcs)


def test_loading_refuses_arc_tables_holding_a_node_outside_their_scope(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri, arcs=sample_arc_tables())
    path = _build_dir(tmp_path) / DFO_ARCS_FILE
    table = pq.read_table(path)
    nodes = [
        CENSUS_POVERTY if node == CRS_VARIABLE else node
        for node in table.column("node").to_pylist()
    ]
    pq.write_table(table.set_column(0, "node", pa.array(nodes)), path)
    _rehash(tmp_path)

    with pytest.raises(CatalogueArtifactError, match="the dfo arc table"):
        load_catalogue(uri)


def test_a_dfo_variable_whose_node_is_base_data_commons_loads_from_the_base_table(
    tmp_path: Path,
) -> None:
    """A DFO variable Spanner declares no node for, as it declares none for
    `Count_Person`, holds its arcs in the `base` table though the base list does not
    name it."""
    uri = _uri(tmp_path)
    arcs = sample_arc_tables()
    moved = build_arc_tables(
        dfo={node: wire_arcs(rows) for node, rows in arcs.dfo.items() if node != CRS_VARIABLE},
        base={
            **{node: wire_arcs(rows) for node, rows in arcs.base.items()},
            CRS_VARIABLE: wire_arcs(arcs.dfo[CRS_VARIABLE]),
        },
        schema={node: wire_arcs(rows) for node, rows in arcs.schema.items()},
        labels=arcs.labels,
    )
    publish_sample(uri, arcs=moved)

    loaded = load_catalogue(uri)
    assert CRS_VARIABLE in loaded.tables.dfo_variables
    assert loaded.held_arcs(CRS_VARIABLE) == arcs.dfo[CRS_VARIABLE]


def test_loading_refuses_a_listed_base_variable_held_empty_in_place_of_its_base_row(
    tmp_path: Path,
) -> None:
    """A partial `base` table that leaves out a listed base variable, held empty in the `dfo`
    table with counts and hashes to match, would route the variable as absent. The loader
    refuses it."""
    uri = _uri(tmp_path)
    publish_sample(uri, arcs=sample_arc_tables())
    build = _build_dir(tmp_path)
    partial = _arcs_moving_gdp_to_dfo(empty=True)
    for name, data in artifact._encode_arc_tables(partial).items():
        (build / name).write_bytes(data)
    payload = json.loads((build / "manifest.json").read_text())
    payload["arc_tables"]["row_counts"] = artifact._arc_row_counts(partial)
    (build / "manifest.json").write_text(json.dumps(payload))
    _rehash(tmp_path)

    with pytest.raises(CatalogueArtifactError, match=rf"the dfo arc table.*{GDP}"):
        load_catalogue(uri)


def test_the_arc_tables_hold_resolver_index_nodes_beyond_the_listed_variables(
    tmp_path: Path,
) -> None:
    """Our index's subject the graph dropped is held empty in the `dfo` table, and a base
    node the base index names is held in the `base` table, though neither is a listed
    variable."""
    uri = _uri(tmp_path)
    dropped, indexed = "Amount_FinancialFlow_CRSPurposeCode43010", "Count_Person_Urban"
    arcs = sample_arc_tables()
    held = build_arc_tables(
        dfo={
            **{node: wire_arcs(rows) for node, rows in arcs.dfo.items()},
            dropped: None,
        },
        base={
            **{node: wire_arcs(rows) for node, rows in arcs.base.items()},
            indexed: {"name": {"nodes": [{"value": "Urban population"}]}},
        },
        schema={node: wire_arcs(rows) for node, rows in arcs.schema.items()},
        labels=arcs.labels,
    )
    publish_sample(uri, arcs=held)

    loaded = load_catalogue(uri)
    assert loaded.held_arcs(dropped) == ()
    assert [arc.value for arc in loaded.held_arcs(indexed) or ()] == ["Urban population"]


def test_reading_arc_tables_refuses_counts_the_manifest_does_not_record(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    manifest = publish_sample(uri, arcs=sample_arc_tables()).manifest
    recorded = replace(
        manifest.arc_tables, row_counts={**manifest.arc_tables.row_counts, "base_arcs_empty": 0}
    )

    with pytest.raises(CatalogueArtifactError, match="arc table row counts"):
        read_arc_tables(uri, replace(manifest, arc_tables=recorded))


def test_reading_arc_tables_refuses_a_row_out_of_range_naming_the_build(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    manifest = publish_sample(uri, arcs=sample_arc_tables()).manifest
    path = _build_dir(tmp_path) / ARC_TARGETS_FILE
    pq.write_table(pq.read_table(path).slice(0, 2), path)
    manifest = _rehash(tmp_path)

    with pytest.raises(
        CatalogueArtifactError, match=rf"^catalogue build {manifest.build_id!r} .*out of range"
    ):
        read_arc_tables(uri, manifest)


def test_reading_arc_tables_refuses_targets_without_labels(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    manifest = publish_sample(uri, arcs=sample_arc_tables()).manifest
    path = _build_dir(tmp_path) / ARC_TARGETS_FILE
    pq.write_table(pq.read_table(path).drop_columns(["label"]), path)
    manifest = _rehash(tmp_path)

    with pytest.raises(CatalogueArtifactError, match="label"):
        read_arc_tables(uri, manifest)


def test_reading_arc_tables_refuses_a_node_twice_in_one_table(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    manifest = publish_sample(uri, arcs=sample_arc_tables()).manifest
    path = _build_dir(tmp_path) / BASE_ARCS_FILE
    table = pq.read_table(path)
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), path)
    manifest = _rehash(tmp_path)

    with pytest.raises(CatalogueArtifactError, match="appears twice"):
        read_arc_tables(uri, manifest)


# --- Family tables -----------------------------------------------------------


def test_the_family_tables_round_trip_and_the_manifest_counts_and_measures_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The build holds one summary per family of its arc tables' and UN tables' variables,
    computed when it is encoded, and the loaded snapshot serves them. With
    `GRAPH_RELEASE_ORIGIN` on, loading also summarises the base variables alone."""
    monkeypatch.delenv("GRAPH_RELEASE_ORIGIN", raising=False)
    uri = _uri(tmp_path)
    arcs, mirror = sample_arc_tables(), sample_un_mirror()
    manifest = publish_sample(uri, un_mirror=mirror, arcs=arcs).manifest
    snapshot = load_catalogue(uri)
    expected = family_tables(dfo=arcs.dfo, base=arcs.base, un=mirror)

    assert snapshot.families == expected
    assert manifest.families.row_counts == {"families": 2, "un_families": 1}
    assert manifest.families.resident_bytes == resident_bytes(snapshot.families)
    family = snapshot.held_family(FamilyKey(UN_SERIES, "value", "", None, None, False, False))
    assert family is not None
    assert len(family.members) == 2
    assert CatalogueManifest.from_json(manifest.to_json()) == manifest
    assert snapshot.base_families is None
    monkeypatch.setenv("GRAPH_RELEASE_ORIGIN", "1")
    monkeypatch.setenv("GRAPH_RELEASE_COVERAGE", "1")
    base = summarise_table(arcs.base)
    assert base
    assert load_catalogue(uri).base_families == base


def _rewrite_families(path: Path, column: str, value: object) -> None:
    """Sets `column` of the family table at `path` to `value` in every row."""
    table = pq.read_table(path)
    at = table.schema.get_field_index(column)
    values = pa.array([value] * table.num_rows, table.schema.field(at).type)
    pq.write_table(table.set_column(at, column, values), path)


@pytest.mark.parametrize(
    ("corrupt", "match"),
    [
        pytest.param(
            lambda path: pq.write_table(
                pa.concat_tables([pq.read_table(path), pq.read_table(path).slice(0, 1)]), path
            ),
            "appears twice",
            id="family-twice",
        ),
        pytest.param(
            lambda path: _rewrite_families(path, "members", []),
            "lists its members malformed",
            id="family-of-no-member",
        ),
        pytest.param(
            lambda path: _rewrite_families(path, "members", ["Debt", "Debt"]),
            "lists its members malformed",
            id="member-twice",
        ),
        pytest.param(
            lambda path: _rewrite_families(path, "shared", ["missing"]),
            "shares a property",
            id="shared-property-not-listed",
        ),
        pytest.param(
            lambda path: _rewrite_families(path, "total_name", None),
            "total without a name",
            id="total-without-a-name",
        ),
    ],
)
def test_loading_refuses_malformed_family_rows(tmp_path: Path, corrupt, match: str) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri, un_mirror=sample_un_mirror(), arcs=sample_arc_tables())
    corrupt(_build_dir(tmp_path) / FAMILIES_FILE)
    _rehash(tmp_path)

    with pytest.raises(CatalogueArtifactError, match=match):
        load_catalogue(uri)


def test_loading_refuses_a_family_table_of_another_build(tmp_path: Path) -> None:
    """A family file another build wrote, with the same row counts, decodes and passes every
    row check, so the manifest's SHA-256 of the file is what refuses it."""
    uri = _uri(tmp_path)
    publish_sample(uri, un_mirror=sample_un_mirror(), arcs=sample_arc_tables())
    arcs = sample_arc_tables()
    renamed = build_arc_tables(
        dfo={node: wire_arcs(rows) for node, rows in arcs.dfo.items()},
        base={
            **{node: wire_arcs(rows) for node, rows in arcs.base.items()},
            GDP: {
                "name": {"nodes": [{"value": "GDP, renamed"}]},
                "typeOf": {"nodes": [{"dcid": "StatisticalVariable", "types": ["Class"]}]},
            },
        },
        schema={node: wire_arcs(rows) for node, rows in arcs.schema.items()},
        labels=arcs.labels,
    )
    other = encode_sample(build_id="build-b", un_mirror=sample_un_mirror(), arcs=renamed)
    assert (
        other.files[FAMILIES_FILE]
        != encode_sample(un_mirror=sample_un_mirror(), arcs=sample_arc_tables()).files[
            FAMILIES_FILE
        ]
    )
    (_build_dir(tmp_path) / FAMILIES_FILE).write_bytes(other.files[FAMILIES_FILE])

    with pytest.raises(CatalogueArtifactError, match="SHA-256"):
        load_catalogue(uri)


def test_loading_refuses_family_counts_the_manifest_does_not_record(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri, un_mirror=sample_un_mirror(), arcs=sample_arc_tables())
    path = _build_dir(tmp_path) / "manifest.json"
    payload = json.loads(path.read_text())
    payload["families"]["row_counts"]["un_families"] = 0
    path.write_text(json.dumps(payload))

    with pytest.raises(CatalogueArtifactError, match="family row counts"):
        load_catalogue(uri)
