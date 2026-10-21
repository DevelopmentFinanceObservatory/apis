"""Pins the serving views of a release against the loaders they stand in for.

The coverage view of a release built from the recorded `RawInputs` equals the `CoverageIndex`
`series_coverage.loader` loads from the published dev coverage build of the same variables
(`coverage_rows/published.json`), its facet metadata read from the recorded facets as the
coverage job reads it from `TimeSeries`. The code index of a release whose part is the packaged
`index.json.gz` equals the packaged index, its stamp naming the release. The recorded slice's
own DFO parts open as both views. The origin view of the recorded release is the DFO arc table
and family summaries the origin catalogue job reads from the same rows.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Collection
from pathlib import Path

import pytest

from code_index.export import content_hash
from code_index.loader import default_index_path, load_index, load_measure_axes
from graph_release.context import current_release
from graph_release.declarations import DECLARATIONS
from graph_release.dfo import build_dfo_parts, dfo_inputs
from graph_release.format import LoadedRelease, open_release
from graph_release.origin import release_origin
from graph_release.views import (
    ViewRefused,
    read_document,
    release_code_indexes,
    release_coverage_index,
)
from origin_catalogue.arc_tables import ArcTablesBuilder
from origin_catalogue.families import summarise_table
from presentation_api.app import presentation_surface
from series_coverage.index import CoverageIndex
from series_coverage.loader import load_coverage
from tests.graph_release.recorded import (
    FIXTURES,
    SLICE_BASE_PARTIES,
    SLICE_DECLARATIONS,
    platform_parts,
    recorded_dfo_arcs,
    slice_platform_parts,
)
from tests.graph_release.support import (
    COVERAGE_SLICE,
    build_packaged_index_release,
    build_test_release,
    origin_release,
    publish_coverage_build,
    release_at_revision,
    sample_tables,
    serving_release,
)

_READ_AT = "2026-10-19T00:00:00Z"
"""`build_test_release`'s `read_timestamp`, as a code index stamp's `exported_at` writes it."""


def _open(root: Path, manifest_id: str) -> LoadedRelease:
    return open_release(root.as_uri(), local_dir=root / "local", release_id=manifest_id)


def test_the_coverage_view_equals_the_published_coverage_build(tmp_path: Path) -> None:
    published = json.loads((FIXTURES / "coverage_rows/published.json").read_text())
    manifest = build_test_release(
        (tmp_path / "release").as_uri(),
        tables=platform_parts().tables(),
        blobs={"declarations": DECLARATIONS.declarations_json()},
    )

    view = release_coverage_index(_open(tmp_path / "release", manifest.release_id))
    loaded = load_coverage(publish_coverage_build(tmp_path / "coverage", published)).index

    assert view.num_rows == loaded.num_rows == len(published["rows"])
    assert view._table.to_pylist() == loaded._table.to_pylist()
    assert view.role_names == loaded.role_names
    assert view.facet_metadata == loaded.facet_metadata
    assert view.build_year == loaded.build_year == 2026
    assert view.variable_provenances() == loaded.variable_provenances()


def test_the_session_release_s_coverage_view_is_the_recorded_coverage_slice() -> None:
    """Each slice row, years and all, though the release counts years from its own mask
    epoch, and no other; its origin view holds no DFO variable."""
    release = current_release()
    view = release.views.coverage
    loaded = load_coverage(COVERAGE_SLICE.parent.as_uri()).index
    variables = loaded.role_names.keys()

    assert loaded.horizon != view.horizon
    assert view.num_rows == loaded.num_rows
    assert _years_by_row(view, variables) == _years_by_row(loaded, variables)
    assert {v: view.role_names[v] for v in variables} == loaded.role_names
    assert view.facet_metadata == loaded.facet_metadata
    assert view.variable_provenances() == loaded.variable_provenances()
    assert release.origin.arcs == {}


def _years_by_row(index: CoverageIndex, variables: Collection[str]) -> list[tuple]:
    """`index`'s rows of `variables`, each with the years its mask counts, sorted."""
    rows = []
    for row in index._table.to_pylist():
        if row["variable"] not in variables:
            continue
        mask = sum(
            row.pop(column) << (64 * word) for word, column in enumerate(index.horizon.mask_columns)
        )
        years = [
            index.horizon.epoch_year + bit for bit in range(mask.bit_length()) if mask >> bit & 1
        ]
        rows.append((*row.values(), years))
    return sorted(rows)


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


def test_the_origin_view_is_the_catalogue_job_s_dfo_table(tmp_path: Path) -> None:
    """Each DFO variable's rows, the import-added `sourceIndicatorCode` constraint and the
    authored base targets included, and each family's members in order."""
    manifest = build_test_release(tmp_path.as_uri(), tables=platform_parts().tables())
    builder = ArcTablesBuilder()
    read = recorded_dfo_arcs(builder)
    job = builder.table("dfo")

    origin = release_origin(_open(tmp_path, manifest.release_id))

    assert list(origin.arcs) == sorted(job)
    assert origin.arcs == job
    assert len(job) == 20
    assert read.base_targets
    assert list(origin.families.items()) == list(summarise_table(job).items())


def test_a_release_declaring_no_variable_has_an_empty_origin_view(tmp_path: Path) -> None:
    manifest = build_test_release(tmp_path.as_uri(), tables=sample_tables())

    origin = release_origin(_open(tmp_path, manifest.release_id))

    assert origin.arcs == {}
    assert origin.families == {}


def test_a_release_without_a_view_part_refuses_the_view(tmp_path: Path) -> None:
    manifest = build_packaged_index_release(tmp_path.as_uri())
    release = _open(tmp_path, manifest.release_id)

    with pytest.raises(ViewRefused, match="no series part"):
        release_coverage_index(release)
    with pytest.raises(ViewRefused, match="no series part"):
        release_origin(release)
    stripped = dataclasses.replace(release, blobs={})
    with pytest.raises(ViewRefused, match="no code_index_blocks part"):
        release_code_indexes(stripped)


def test_an_origin_release_answers_its_variables_arcs_as_the_arc_tables_hold_them() -> None:
    """Typed targets, an untyped one and a literal, each variable's family summarised, and the
    release under an id of its own."""
    variables = {
        "Debt_Total": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable", "types": ["Class"]}]},
            "name": {"nodes": [{"value": "Total debt"}]},
            "populationType": {"nodes": [{"dcid": "Debt", "name": "Debt", "types": ["Class"]}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        },
        "Debt_Gov": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable", "types": ["Class"]}]},
            "populationType": {"nodes": [{"dcid": "Debt", "name": "Debt", "types": ["Class"]}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "debtor": {"nodes": [{"dcid": "Government", "types": ["DebtorType"]}]},
            "constraintProperties": {"nodes": [{"dcid": "debtor", "types": ["Property"]}]},
        },
    }
    builder = ArcTablesBuilder()
    for variable, arcs in variables.items():
        builder.add("dfo", variable, arcs)
    held = builder.table("dfo")

    release = origin_release(variables)

    assert release.origin.arcs == held
    assert release.origin.families == summarise_table(held)
    assert release.release_id != current_release().release_id
    assert release.served


def test_a_surface_built_while_serving_a_release_binds_it_and_the_session_s_after() -> None:
    session = current_release()
    release = release_at_revision("serving-test")

    with serving_release(release):
        inside = presentation_surface(revision_store=object(), frame_store=object())
    outside = presentation_surface(revision_store=object(), frame_store=object())

    for surface, served in ((inside, release), (outside, session)):
        assert surface.request_scope is not None
        with surface.request_scope():
            assert current_release() is served
