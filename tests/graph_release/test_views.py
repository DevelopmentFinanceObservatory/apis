"""Pins the serving views of a release against the loaders they stand in for.

The coverage view of a release built from the recorded `RawInputs` equals the `CoverageIndex`
`series_coverage.loader` loads from the published dev coverage build of the same variables
(`coverage_rows/published.json`), its facet metadata read from the recorded facets as the
coverage job reads it from `TimeSeries`. The code index of a release whose part is the packaged
`index.json.gz` equals the packaged index, its stamp naming the release. The recorded slice's
own DFO parts open as both views.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pyarrow as pa
import pytest

from code_index.export import content_hash
from code_index.loader import default_index_path, load_index, load_measure_axes
from graph_release.coverage import HORIZON
from graph_release.declarations import DECLARATIONS
from graph_release.dfo import build_dfo_parts, dfo_inputs
from graph_release.format import LoadedRelease, open_release
from graph_release.views import (
    ViewRefused,
    read_document,
    release_code_indexes,
    release_coverage_index,
)
from series_coverage.io import (
    write_coverage_table,
    write_current,
    write_facet_metadata,
    write_manifest,
    write_role_names,
)
from series_coverage.loader import load_coverage
from series_coverage.schema import (
    FACET_COLUMN,
    CoverageManifest,
    CoveragePointer,
    FacetMetadata,
    RoleNames,
)
from tests.graph_release.recorded import (
    FIXTURES,
    SLICE_BASE_PARTIES,
    SLICE_DECLARATIONS,
    platform_parts,
    raw_inputs,
    slice_platform_parts,
)
from tests.graph_release.support import build_packaged_index_release, build_test_release
from tests.series_coverage.support import build_dir, pointer_path

_READ_AT = "2026-10-19T00:00:00Z"
"""`build_test_release`'s `read_timestamp`, as a coverage build's `built_at` and a code index
stamp's `exported_at` write it."""


def _open(root: Path, manifest_id: str) -> LoadedRelease:
    return open_release(root.as_uri(), local_dir=root / "local", release_id=manifest_id)


def _publish_coverage(base: Path, published: dict) -> str:
    """Publishes `published`'s rows and role names as the coverage job writes a build, each
    facet's metadata from the recorded facet JSON."""
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
            source="tests.graph_release.test_views",
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


def test_the_coverage_view_equals_the_published_coverage_build(tmp_path: Path) -> None:
    published = json.loads((FIXTURES / "coverage_rows/published.json").read_text())
    manifest = build_test_release(
        (tmp_path / "release").as_uri(),
        tables=platform_parts().tables(),
        blobs={"declarations": DECLARATIONS.declarations_json()},
    )

    view = release_coverage_index(_open(tmp_path / "release", manifest.release_id))
    loaded = load_coverage(_publish_coverage(tmp_path / "coverage", published)).index

    assert view.num_rows == loaded.num_rows == len(published["rows"])
    assert view._table.to_pylist() == loaded._table.to_pylist()
    assert view.role_names == loaded.role_names
    assert view.facet_metadata == loaded.facet_metadata
    assert view.build_year == loaded.build_year == 2026
    assert view.variable_provenances() == loaded.variable_provenances()


def test_the_code_index_view_is_the_packaged_index_stamped_with_the_release(
    tmp_path: Path,
) -> None:
    manifest = build_packaged_index_release(tmp_path.as_uri())

    index, measures = release_code_indexes(_open(tmp_path, manifest.release_id))
    packaged = load_index(default_index_path())

    assert index.stamp == dataclasses.replace(
        packaged.stamp, coverage_build=manifest.release_id, exported_at=_READ_AT
    )
    for block in ("codes", "statvars", "listed_statvars", "source_codes"):
        assert getattr(index, block) == getattr(packaged, block), block
    assert measures == load_measure_axes(default_index_path())


def test_the_slice_release_opens_as_both_views(tmp_path: Path) -> None:
    parts = slice_platform_parts()
    dfo = build_dfo_parts(dfo_inputs(parts, SLICE_DECLARATIONS), SLICE_BASE_PARTIES)
    manifest = build_test_release(tmp_path.as_uri(), tables=parts.tables(), blobs=dfo.blobs())
    release = _open(tmp_path, manifest.release_id)

    index, _ = release_code_indexes(release)
    coverage = release_coverage_index(release)

    blocks = read_document(dfo.code_index_blocks)
    del blocks["stamp"]
    assert index.stamp.content_hash == content_hash(blocks)
    assert index.stamp.coverage_build == manifest.release_id
    assert {row.dcid for row in index.statvars} <= coverage.variable_provenances().keys()


def test_a_release_without_a_view_part_refuses_the_view(tmp_path: Path) -> None:
    manifest = build_packaged_index_release(tmp_path.as_uri())
    release = _open(tmp_path, manifest.release_id)

    with pytest.raises(ViewRefused, match="no series part"):
        release_coverage_index(release)
    stripped = dataclasses.replace(release, blobs={})
    with pytest.raises(ViewRefused, match="no code_index_blocks part"):
        release_code_indexes(stripped)
