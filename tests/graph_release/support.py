"""Builds graph releases in the bucket layout from rows given in code."""

from __future__ import annotations

import dataclasses
import json
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from artifact_publication.jobs import Ingestion
from artifact_publication.uri import read_uri_bytes_and_generation
from code_index.loader import default_index_path
from code_index.query import CodeIndex
from code_index.types import MeasureIndex
from graph_release.context import bound_release, current_release
from graph_release.coverage import HORIZON
from graph_release.declarations import DECLARATIONS
from graph_release.format import (
    PART_SCHEMAS,
    Manifest,
    StagedPart,
    pointer_uri,
    write_blob_part,
    write_part,
)
from graph_release.publish import PublishStatus, hold_release, publish_release
from graph_release.service import Release
from series_coverage.io import (
    write_coverage_table,
    write_current,
    write_facet_metadata,
    write_manifest,
    write_role_names,
)
from series_coverage.schema import (
    FACET_COLUMN,
    CoverageManifest,
    CoveragePointer,
    FacetMetadata,
    RoleNames,
)
from tests.graph_release.recorded import platform_parts, raw_inputs
from tests.series_coverage.support import build_dir, pointer_path

TEST_EXECUTION = "test-import-execution"
TEST_REVISION = "test-dc-revision"
_READ_AT = "2026-10-19T00:00:00Z"
"""`build_test_release`'s `read_timestamp`, as a coverage build's `built_at` writes it."""


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


def build_view_release(root_uri: str, *, dc_serving_revision: str = TEST_REVISION) -> Manifest:
    """Publishes a release under `root_uri` whose views build: the packaged `index.json.gz`
    document as its code index part, and the recorded slice's platform parts and the
    declarations as the parts its coverage view reads (`tests.graph_release.recorded`)."""
    return build_test_release(
        root_uri,
        tables=platform_parts().tables(),
        blobs={
            "code_index_blocks": default_index_path().read_bytes(),
            "declarations": DECLARATIONS.declarations_json(),
        },
        dc_serving_revision=dc_serving_revision,
    )


def publish_coverage_build(
    base: Path, published: dict, *, ingestion: Ingestion | None = None
) -> str:
    """Publishes `published`'s rows and role names under `base` as the coverage job writes a
    build, built at `build_test_release`'s read time from `ingestion`, each facet's metadata
    from the recorded facet JSON. Returns `base`'s URI."""
    prefix = build_dir(base, "build")
    prefix.mkdir(parents=True)
    table = pa.Table.from_pylist(
        [dict(zip(published["columns"], row, strict=True)) for row in published["rows"]],
        schema=HORIZON.schema,
    )
    write_coverage_table(table, prefix / "coverage.parquet", horizon=HORIZON)
    write_role_names(
        {variable: RoleNames(**names) for variable, names in published["role_names"].items()},
        prefix / "role_names.json",
    )
    covered = set(table[FACET_COLUMN].to_pylist())
    facets = raw_inputs().facets
    metadata: dict[str, FacetMetadata] = {}
    for facet_id, facet in zip(
        facets["facet_id"].to_pylist(), facets["facet_min"].to_pylist(), strict=True
    ):
        if facet_id in covered:
            fields = json.loads(facet)
            metadata[facet_id] = FacetMetadata(
                provenance_id=fields["provenance"],
                measurement_method=fields.get("measurementMethod"),
                unit=fields.get("unit"),
                observation_period=fields.get("observationPeriod"),
            )
    write_facet_metadata(metadata, prefix / "facets.json")
    write_manifest(
        CoverageManifest(
            stamp="stamp",
            build_id="build",
            built_at=_READ_AT,
            ingestion=ingestion,
            source="tests.graph_release.support",
            row_counts={"rows": table.num_rows},
            files={
                "coverage": "coverage.parquet",
                "role_names": "role_names.json",
                "facets": "facets.json",
            },
            horizon=HORIZON,
        ),
        prefix / "manifest.json",
    )
    write_current(CoveragePointer(stamp="stamp", build_id="build"), pointer_path(base))
    return base.as_uri()


@contextmanager
def serving_index(
    code_index: CodeIndex, measure_index: MeasureIndex | None = None
) -> Iterator[Release]:
    """Binds the bound release with `code_index`, and `measure_index` when given, as its views'
    indexes for the body of the `with` block, so `get_index()` and `get_measure_index()` serve
    them as they serve a release's. The release keeps its parts and its coverage view."""
    release = current_release()
    assert release.views is not None, "the bound release carries no views"
    views = dataclasses.replace(
        release.views,
        code_index=code_index,
        measure_index=measure_index if measure_index is not None else release.views.measure_index,
    )
    with bound_release(dataclasses.replace(release, views=views)) as bound:
        yield bound


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
