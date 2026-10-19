"""Builds graph releases in the bucket layout from rows given in code."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from artifact_publication.uri import read_uri_bytes_and_generation
from code_index.loader import default_index_path
from graph_release.format import (
    PART_SCHEMAS,
    Manifest,
    StagedPart,
    pointer_uri,
    write_blob_part,
    write_part,
)
from graph_release.publish import PublishStatus, hold_release, publish_release

TEST_EXECUTION = "test-import-execution"
TEST_REVISION = "test-dc-revision"


def sample_tables() -> dict[str, pa.Table]:
    """Ten rows of each platform part, unsorted, over two provenances."""
    variables = [f"Var_{n}" for n in (3, 1, 4, 0, 2)]
    series = pa.table(
        {
            "variable_measured": [variables[n % 5] for n in range(10)],
            "entity1": ["country/GHA" if n % 2 else "country/DEU" for n in range(10)],
            "extra_entities_id": ["" if n < 7 else "x1" for n in range(10)],
            "facet_id": ["f1" if n % 3 else "f2" for n in range(10)],
            "entity2": [None if n % 4 else "country/KEN" for n in range(10)],
            "observation_count": list(range(1, 11)),
            "earliest_date": [date(1990 + n, 1, 1) for n in range(10)],
            "latest_date": [date(2010 + n, 1, 1) for n in range(10)],
            "period_mask_0": [(1 << n) | (1 << 63) for n in range(10)],
            "period_mask_1": [None if n == 4 else n for n in range(10)],
        },
        schema=pa.schema(
            [
                ("variable_measured", pa.string()),
                ("entity1", pa.string()),
                ("extra_entities_id", pa.string()),
                ("facet_id", pa.string()),
                ("entity2", pa.string()),
                ("observation_count", pa.int64()),
                ("earliest_date", pa.date32()),
                ("latest_date", pa.date32()),
                ("period_mask_0", pa.uint64()),
                ("period_mask_1", pa.int64()),
            ]
        ),
    )
    # Bit 63 set, stored signed as Spanner stores it.
    series = series.set_column(
        series.schema.get_field_index("period_mask_0"),
        "period_mask_0",
        series["period_mask_0"].cast(pa.int64(), safe=False),
    )
    facets = pa.table(
        {
            "facet_id": [f"f{n}" for n in (2, 1, 5, 3, 4, 9, 8, 6, 7, 0)],
            "provenance": ["dc/p2" if n % 2 else "dc/p1" for n in range(10)],
            "import_name": [f"Import{n}" for n in range(10)],
            "measurement_method": [None, "OECD", *(["Sum"] * 8)],
            "observation_period": ["P1Y"] * 10,
            "unit": ["USDollar"] * 9 + [None],
            "scaling_factor": [None] * 5 + ["1000000"] * 5,
            "is_dc_aggregate": [n == 3 for n in range(10)],
        }
    )
    edges = pa.table(
        {
            "subject_id": [variables[n % 5] for n in range(10)],
            "predicate": ["typeOf" if n % 2 else "measuredProperty" for n in range(10)],
            "object_id": [f"Node_{9 - n}" for n in range(10)],
            "provenance": ["dc/p2" if n % 3 else "dc/p1" for n in range(10)],
        }
    )
    nodes = pa.table(
        {
            "subject_id": [f"Node_{n}" for n in (7, 2, 9, 0, 4, 1, 8, 3, 6, 5)],
            "name": [None if n == 1 else f"Name {n}" for n in range(10)],
            "types": [["StatisticalVariable"], [], None, ["Thing", "Class"], *([["Thing"]] * 6)],
            "value": [None] * 8 + ["literal", "42"],
            "authored": [n >= 8 for n in range(10)],
        }
    )
    imports = pa.table(
        {
            "import_name": [f"Import{n}" for n in (4, 0, 9, 1, 6, 2, 8, 3, 7, 5)],
            "provenance": [f"dc/p{n}" if n % 2 else None for n in range(10)],
            "version": [f"v{n}" for n in range(10)],
            "state": ["SUCCESS"] * 10,
            "data_import_timestamp": [
                datetime(2026, 10, 1 + n, 12, 30, tzinfo=UTC) for n in range(10)
            ],
        }
    )
    return {
        "series": series,
        "facets": facets,
        "edges": edges,
        "nodes": nodes,
        "imports": imports,
    }


def build_test_release(
    root_uri: str,
    *,
    tables: Mapping[str, pa.Table],
    blobs: Mapping[str, bytes] | None = None,
    import_execution_id: str = TEST_EXECUTION,
    dc_serving_revision: str = TEST_REVISION,
    hold: bool = False,
) -> Manifest:
    """Stages `tables` and the serialised `blobs` as parts, publishes them under `root_uri`
    through `publish_release`, holds the release when `hold`, and returns
    the manifest. Raises `AssertionError` when a held pointer refuses it.

    `series` is written one file per facet provenance, so it needs `facets`.
    `edges` is written one file per its own `provenance` column.
    """
    with tempfile.TemporaryDirectory() as staging:
        staging_dir = Path(staging)
        staged: list[StagedPart] = []
        for name, table in tables.items():
            for provenance, rows in _provenance_files(name, table, tables):
                staged += write_part(
                    rows,
                    staging_dir=staging_dir,
                    name=name,
                    layer=PART_SCHEMAS[name].layer,
                    source="tests.graph_release.support",
                    produced_by_execution=import_execution_id,
                    provenance=provenance,
                )
        for name, data in (blobs or {}).items():
            staged.append(
                write_blob_part(
                    data,
                    staging_dir=staging_dir,
                    name=name,
                    source="tests.graph_release.support",
                    produced_by_execution=import_execution_id,
                )
            )
        manifest = Manifest(
            import_execution_id=import_execution_id,
            import_completed_at="2026-10-18T23:50:00+00:00",
            ingested_imports=("TestImport",),
            read_timestamp="2026-10-19T00:00:00+00:00",
            dc_serving_revision=dc_serving_revision,
            builder_code_hash="test-code-hash",
            declarations_hash="test-declarations-hash",
            parts=tuple(part.ref for part in staged),
        )
        found = read_uri_bytes_and_generation(pointer_uri(root_uri))
        outcome = publish_release(
            root_uri,
            staged=staged,
            manifest=manifest,
            expected_pointer_generation=found[1] if found is not None else None,
        )
    assert outcome.status is PublishStatus.PUBLISHED, outcome
    if hold:
        hold_release(root_uri, manifest.release_id)
    return manifest


def build_packaged_index_release(root_uri: str) -> Manifest:
    """Publishes a release under `root_uri` whose only part is the code index, the
    packaged `index.json.gz` document byte for byte."""
    return build_test_release(
        root_uri, tables={}, blobs={"code_index_blocks": default_index_path().read_bytes()}
    )


def _provenance_files(
    name: str, table: pa.Table, tables: Mapping[str, pa.Table]
) -> list[tuple[str | None, pa.Table]]:
    if name == "edges":
        provenance = table["provenance"]
    elif name == "series":
        facets = tables["facets"]
        provenance = pc.take(
            facets["provenance"], pc.index_in(table["facet_id"], value_set=facets["facet_id"])
        )
    else:
        return [(None, table)]
    return [
        (value, table.filter(pc.equal(provenance, value)))
        for value in sorted(pc.unique(provenance).to_pylist())
    ]
