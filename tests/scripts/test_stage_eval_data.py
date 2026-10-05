"""`scripts/stage_eval_data.py` against a local directory standing in for the deployed buckets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.stage_eval_data as m
from origin_catalogue.schema import FORMAT_ROOT as CATALOGUE_ROOT
from series_coverage.schema import FORMAT_ROOT as COVERAGE_ROOT

REPO = Path(__file__).resolve().parents[2]


def _publish(bucket: Path, root: str, pointer: dict, builds: dict[str, dict[str, bytes]]) -> None:
    """Lay out ``bucket/root`` as a publishing job does: builds under ``versions/``,
    then ``current.json``."""
    for build_id, files in builds.items():
        for name, data in files.items():
            path = bucket / root / "versions" / build_id / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    (bucket / root / "current.json").write_text(json.dumps(pointer))


@pytest.fixture
def buckets(tmp_path):
    coverage, catalogue = tmp_path / "coverage-bucket", tmp_path / "catalogue-bucket"
    _publish(
        coverage,
        "format-2",
        {"stamp": "s2", "build_id": "b2"},
        {
            "b1": {"manifest.json": b"old"},
            "b2": {"manifest.json": b"m2", "tables/coverage.parquet": b"p2"},
        },
    )
    _publish(
        coverage,
        "format-3",
        {"stamp": "s3", "build_id": "b3"},
        {"b3": {"manifest.json": b"m3"}},
    )
    _publish(
        catalogue,
        "format-1",
        {"build_id": "c1", "identity": "i1"},
        {"c1": {"manifest.json": b"c"}},
    )
    return coverage.as_uri(), catalogue.as_uri()


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_stages_the_current_build_of_every_root_the_checkouts_read(buckets, tmp_path):
    coverage_uri, catalogue_uri = buckets
    roots = {
        Path("head"): m.FormatRoots(coverage="format-3", catalogue="format-1"),
        Path("base"): m.FormatRoots(coverage="format-2", catalogue="format-1"),
    }
    dest = tmp_path / "staged"

    env = m.stage(
        list(roots),
        coverage_uri=coverage_uri,
        catalogue_uri=catalogue_uri,
        dest=dest,
        roots_of=roots.__getitem__,
    )

    assert env == {
        "COVERAGE_ARTIFACT_URI": (dest / "coverage").resolve().as_uri(),
        "ORIGIN_CATALOGUE_URI": (dest / "origin-catalogue").resolve().as_uri(),
    }
    # Only the pointed-at build is copied, and each pointer stays as published.
    assert _files(dest / "coverage") == {
        "format-2/current.json": json.dumps({"stamp": "s2", "build_id": "b2"}).encode(),
        "format-2/versions/b2/manifest.json": b"m2",
        "format-2/versions/b2/tables/coverage.parquet": b"p2",
        "format-3/current.json": json.dumps({"stamp": "s3", "build_id": "b3"}).encode(),
        "format-3/versions/b3/manifest.json": b"m3",
    }
    assert _files(dest / "origin-catalogue") == {
        "format-1/current.json": json.dumps({"build_id": "c1", "identity": "i1"}).encode(),
        "format-1/versions/c1/manifest.json": b"c",
    }


@pytest.mark.parametrize(
    ("coverage_root", "message"),
    [("format-9", "no current.json"), ("format-empty", "which is empty")],
)
def test_a_root_without_a_build_exits_3(
    buckets, tmp_path, monkeypatch, capsys, coverage_root, message
):
    coverage_uri, catalogue_uri = buckets
    bucket = Path(coverage_uri.removeprefix("file://"))
    (bucket / "format-empty").mkdir()
    (bucket / "format-empty" / "current.json").write_text('{"stamp": "s", "build_id": "gone"}')
    roots = m.FormatRoots(coverage=coverage_root, catalogue="format-1")
    monkeypatch.setattr(m, "checkout_format_roots", lambda _checkout: roots)

    code = m.main(
        [
            "--checkout",
            ".",
            "--dest",
            str(tmp_path / "staged"),
            "--coverage-uri",
            coverage_uri,
            "--catalogue-uri",
            catalogue_uri,
        ]
    )

    assert code == 3
    assert message in capsys.readouterr().err


def test_checkout_format_roots_reads_the_checkout_environment():
    assert m.checkout_format_roots(REPO) == m.FormatRoots(
        coverage=COVERAGE_ROOT, catalogue=CATALOGUE_ROOT
    )
