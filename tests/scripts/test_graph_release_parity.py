"""Pins `scripts/graph_release_parity.py` on the recorded release slice: a release and the
coverage build and code index of the same import report parity, and a difference in the rows,
the provenances or a code index block is unexplained unless a rule names its cause."""

from __future__ import annotations

import copy
import dataclasses
import gzip
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from artifact_publication.jobs import Ingestion
from code_index.loader import default_index_path
from graph_release.format import open_release
from graph_release.shadow import PINNED_INDEX_VARIABLES
from scripts.graph_release_parity import Refused, check_same_import, parity_report
from series_coverage.loader import load_coverage, read_table
from tests.graph_release.recorded import FIXTURES
from tests.graph_release.support import TEST_EXECUTION, build_view_release, publish_coverage_build
from tests.series_coverage.support import build_dir

_PINNED = next(iter(PINNED_INDEX_VARIABLES))


def _report(
    tmp_path: Path,
    *,
    published: Callable[[dict], None] = lambda _: None,
    facets: Callable[[dict], None] = lambda _: None,
    export: Callable[[dict], None] = lambda _: None,
) -> dict:
    """The report on `build_view_release` against the recorded coverage build and the packaged
    code index, after `published` edits the build's rows, `facets` its `facets.json` and
    `export` the code index document standing in for a fresh export."""
    manifest = build_view_release((tmp_path / "release").as_uri())
    release = open_release(
        (tmp_path / "release").as_uri(),
        local_dir=tmp_path / "local",
        release_id=manifest.release_id,
    )
    rows = json.loads((FIXTURES / "coverage_rows/published.json").read_text())
    published(rows)
    base = tmp_path / "coverage"
    publish_coverage_build(
        base, rows, ingestion=Ingestion(TEST_EXECUTION, "2026-10-18 23:50:00+00:00")
    )
    facets_file = build_dir(base, "build") / "facets.json"
    metadata = json.loads(facets_file.read_text())
    facets(metadata)
    facets_file.write_text(json.dumps(metadata))
    coverage = load_coverage(base.as_uri())
    document = json.loads(gzip.decompress(default_index_path().read_bytes()))
    export(document)
    return parity_report(
        release,
        coverage,
        read_table(build_dir(base, "build").as_uri(), coverage.manifest),
        document,
    )


def _extra_row(rows: dict) -> None:
    row = list(rows["rows"][0])
    row[rows["columns"].index("recipient")] = "country/XXX"
    rows["rows"].append(row)


def _moved_provenance(metadata: dict) -> None:
    facet = next(iter(metadata.values()))
    facet["provenance_id"] = "provenance/Elsewhere"


def _relabelled_statvar(document: dict) -> None:
    document["statvars"][0]["label"] = "another label"


def _pinned_row(document: dict) -> None:
    row = copy.deepcopy(document["listed_statvars"][0])
    row["dcid"] = _PINNED
    document["listed_statvars"].append(row)


def test_equal_inputs_report_parity(tmp_path: Path) -> None:
    report = _report(tmp_path)

    assert report["outcome"] == "parity"
    assert report["unexplained"] == 0
    categories = report["categories"]
    assert categories["coverage.rows"]["compared"] > 0
    assert categories["code_index.statvars"]["compared"] > 0
    assert all(category["differences"] == 0 for category in categories.values())


@pytest.mark.parametrize(
    ("edits", "category"),
    [
        ({"published": _extra_row}, "coverage.rows"),
        ({"facets": _moved_provenance}, "coverage.variable_provenances"),
        ({"export": _relabelled_statvar}, "code_index.statvars"),
    ],
)
def test_a_difference_is_unexplained(tmp_path: Path, edits: dict, category: str) -> None:
    report = _report(tmp_path, **edits)

    assert report["outcome"] == "differences"
    assert report["categories"][category]["unexplained"] >= 1


def test_a_difference_a_rule_names_is_explained(tmp_path: Path) -> None:
    report = _report(tmp_path, export=_pinned_row)

    listed = report["categories"]["code_index.listed_statvars"]
    assert report["outcome"] == "parity"
    assert listed["differences"] == 1
    assert listed["explained"] == {"pinned_index_variable": 1}
    assert listed["explained_examples"][0]["difference"]["side"] == "export_only"


@pytest.mark.parametrize(
    ("ingestion", "code"),
    [
        (Ingestion("another-execution", "2026-10-18 23:50:00+00:00"), "import_mismatch"),
        (None, "coverage_ingestion_unrecorded"),
    ],
)
def test_a_coverage_build_of_another_import_is_refused(
    tmp_path: Path, ingestion: Ingestion | None, code: str
) -> None:
    manifest = build_view_release(tmp_path.as_uri())
    rows = json.loads((FIXTURES / "coverage_rows/published.json").read_text())
    coverage = load_coverage(publish_coverage_build(tmp_path / "coverage", rows)).manifest

    with pytest.raises(Refused) as refused:
        check_same_import(manifest, dataclasses.replace(coverage, ingestion=ingestion))

    assert refused.value.code == code
