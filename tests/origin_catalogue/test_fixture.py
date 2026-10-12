"""Checks the committed catalogue builds: the sample in `tests/fixtures/origin_catalogue`
and the dev build in `tests/fixtures/origin_catalogue_eval`."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from origin_catalogue import ORIGINS, UN_SYSTEM, CatalogueManifest, entry_ref
from origin_catalogue.artifact import load_catalogue
from origin_catalogue.service import DEFAULT_MAX_RESIDENT_BYTES
from scripts.stage_origin_catalogue_fixture import stage_fixture
from tests.origin_catalogue.support import (
    FIXTURE_BUILD_ID,
    FIXTURE_DIR,
    encode_sample,
    sample_tables,
    sample_un_mirror,
)

_EVAL_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "origin_catalogue_eval"

_REWRITE = "rewrite it with `uv run python scripts/record_origin_catalogue_fixture.py`"


def _without_sizes(manifest: CatalogueManifest) -> CatalogueManifest:
    return replace(
        manifest, resident_bytes=0, un_mirror=replace(manifest.un_mirror, resident_bytes=0)
    )


def test_committed_fixture_loads_as_the_sample_build(tmp_path: Path) -> None:
    snapshot = load_catalogue(stage_fixture(FIXTURE_DIR, tmp_path))
    expected = encode_sample(build_id=FIXTURE_BUILD_ID, un_mirror=sample_un_mirror()).manifest
    assert snapshot.tables == sample_tables(), _REWRITE
    assert snapshot.un.mirror == sample_un_mirror(), _REWRITE
    # `resident_bytes` sums `sys.getsizeof`, which varies across Python builds.
    assert _without_sizes(snapshot.manifest) == _without_sizes(expected), _REWRITE


def test_committed_eval_build_loads_under_the_service_budget(tmp_path: Path) -> None:
    snapshot = load_catalogue(
        stage_fixture(_EVAL_FIXTURE_DIR, tmp_path), max_resident_bytes=DEFAULT_MAX_RESIDENT_BYTES
    )
    counts = snapshot.manifest.row_counts
    assert all(counts[f"provenances_{origin.id}"] > 0 for origin in ORIGINS.values()), counts


def test_committed_eval_build_documents_every_listed_provenance(tmp_path: Path) -> None:
    """`materialise.provenance_doc._catalogued_documentation` gives a listed provenance no
    document when its entry's `source_documentation.provenances` omits it. Every non-UN row
    belongs to an entry that lists it."""
    tables = load_catalogue(stage_fixture(_EVAL_FIXTURE_DIR, tmp_path)).tables
    rows = [row for row in tables.provenances.values() if row.data_commons != UN_SYSTEM]
    assert rows
    unlisted = [
        row.id
        for row in rows
        if (documentation := tables.source_documentation.get(entry_ref(row))) is None
        or row.id not in documentation.provenances
    ]
    assert unlisted == []
