"""`series_coverage.verify_index`: the deploy-time check that the packaged code index carries
the provenances of the published coverage build. An index built from the coverage slice fixture
passes; a perturbed one fails, naming what differs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from code_index import parse_index
from code_index.loader import load_index
from series_coverage.compatibility import index_coverage_differences, read_compatible_pointer
from series_coverage.loader import load_coverage
from series_coverage.verify_index import main

_COVERAGE_SLICE_URI = (Path(__file__).resolve().parents[1] / "fixtures" / "coverage_slice").as_uri()
_CRS = "provenance/OECDCreditorReportingSystem"
_OTHER = "provenance/SomeOtherDataset"
_ROWS = 30


def _dataset(provenance: str, *, rank: int = 0) -> dict:
    return {
        "provenance": provenance,
        "label": provenance,
        "synonyms": [],
        "opt_in": False,
        "absent_observation": "unknown",
        "rank": rank,
        "invariants": {},
    }


def _slice_document() -> dict:
    """An index document whose rows are the slice's first variables with the slice's own
    provenances."""
    provenances = load_coverage(_COVERAGE_SLICE_URI).index.variable_provenances()
    return {
        "stamp": {
            "graph_endpoint": "https://fake.example",
            "exported_at": "2026-09-27T00:00:00Z",
            "code_counts": {},
            "statvar_shape_counts": {},
            "content_hash": "",
            "coverage_build": "20260101T000000Z-exported",
            "statvars_without_coverage": 0,
        },
        "declarations": [],
        "datasets": [_dataset(_CRS)],
        "codes": [],
        "statvars": [
            {
                "dcid": dcid,
                "population_type": "FinancialFlow",
                "measured_property": "amount",
                "stat_type": "measuredValue",
                "measurement_qualifier": None,
                "measurement_denominator": None,
                "observation_properties": [],
                "constraint_properties": {},
                "provenances": list(provenances[dcid]),
            }
            for dcid in sorted(provenances)[:_ROWS]
        ],
    }


def _run(tmp_path: Path, document: dict) -> int:
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(document))
    return main([_COVERAGE_SLICE_URI, "--index", str(index_path)])


def test_an_index_carrying_the_builds_provenances_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path, _slice_document()) == 0

    out = capsys.readouterr()
    assert f"{_ROWS} StatVar rows carry the build's provenances" in out.out
    assert "'20260101T000000Z-exported'" in out.out
    assert "'20260927T142252Z-slice'" in out.out
    assert out.err == ""


def test_a_row_the_build_does_not_cover_fails_and_is_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _slice_document()
    document["statvars"][0]["dcid"] = "StatVar_The_Build_Does_Not_Hold"

    assert _run(tmp_path, document) == 1

    err = capsys.readouterr().err
    assert f"StatVar_The_Build_Does_Not_Hold: index ['{_CRS}'], build []" in err
    assert "'20260101T000000Z-exported'" in err
    assert "'20260927T142252Z-slice'" in err


def test_a_row_missing_a_provenance_the_build_holds_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _slice_document()
    dcid = document["statvars"][3]["dcid"]
    document["statvars"][3]["provenances"] = []

    assert _run(tmp_path, document) == 1

    assert f"{dcid}: index [], build ['{_CRS}']" in capsys.readouterr().err


def test_a_row_carrying_a_provenance_the_build_lacks_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _slice_document()
    document["datasets"].append(_dataset(_OTHER, rank=1))
    dcid = document["statvars"][5]["dcid"]
    # The row moves to a measured property only the other dataset publishes.
    document["statvars"][5]["measured_property"] = "otherAmount"
    document["statvars"][5]["provenances"] = [_OTHER]

    assert _run(tmp_path, document) == 1

    err = capsys.readouterr().err
    assert f"{dcid}: index ['{_OTHER}'], build ['{_CRS}']" in err
    assert f"declared dataset '{_OTHER}' has no row in the build" in err


def test_at_most_twenty_differing_variables_are_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _slice_document()
    for row in document["statvars"][:25]:
        row["provenances"] = []

    assert _run(tmp_path, document) == 1

    err = capsys.readouterr().err
    assert err.count(": index [], build [") == 20
    assert "and 5 more of 25 differing variables" in err


def test_differences_report_a_declared_dataset_with_no_row_in_the_build() -> None:
    index = parse_index(_slice_document())
    coverage = {row.dcid: (_OTHER,) for row in index.statvars}

    differences = index_coverage_differences(index, coverage)

    assert differences.datasets_without_rows == (_CRS,)
    assert set(differences.variables) == {row.dcid for row in index.statvars}
    assert not index_coverage_differences(
        index, {row.dcid: row.provenances for row in index.statvars}
    )


def test_no_artifact_uri_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("COVERAGE_ARTIFACT_URI", raising=False)

    assert main([]) == 1
    assert "COVERAGE_ARTIFACT_URI" in capsys.readouterr().err


def test_an_unreadable_build_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([(tmp_path / "nothing-here").as_uri()]) == 1
    assert "could not load its inputs" in capsys.readouterr().err


def test_the_artifact_uri_env_var_is_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(_slice_document()))
    monkeypatch.setenv("COVERAGE_ARTIFACT_URI", _COVERAGE_SLICE_URI)

    assert main(["--index", str(index_path)]) == 0


def _copied_slice(tmp_path: Path) -> str:
    """A copy of the slice fixture, so a recorded pointer stays out of the committed one."""
    copy = tmp_path / "coverage"
    shutil.copytree(Path(_COVERAGE_SLICE_URI.removeprefix("file://")), copy)
    return copy.as_uri()


def _hashed_index(tmp_path: Path, document: dict) -> str:
    document["stamp"]["content_hash"] = "slice-index"
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(document))
    return str(index_path)


def test_record_points_the_index_pointer_at_a_build_that_passes(tmp_path: Path) -> None:
    uri = _copied_slice(tmp_path)
    index_path = _hashed_index(tmp_path, _slice_document())

    assert main([uri, "--index", index_path, "--record"]) == 0

    pointer = read_compatible_pointer(uri, load_index(index_path))
    assert pointer is not None
    assert pointer.build_id == "20260927T142252Z-slice"


def test_record_leaves_no_pointer_for_a_build_that_differs(tmp_path: Path) -> None:
    uri = _copied_slice(tmp_path)
    document = _slice_document()
    document["statvars"][0]["provenances"] = []
    index_path = _hashed_index(tmp_path, document)

    assert main([uri, "--index", index_path, "--record"]) == 1

    assert read_compatible_pointer(uri, load_index(index_path)) is None


def test_a_check_without_record_leaves_no_pointer(tmp_path: Path) -> None:
    uri = _copied_slice(tmp_path)
    index_path = _hashed_index(tmp_path, _slice_document())

    assert main([uri, "--index", index_path]) == 0

    assert read_compatible_pointer(uri, load_index(index_path)) is None
