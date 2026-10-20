"""A coverage build is recorded as compatible with the code index only when it carries the
index's provenances, and the deploy records the packaged index as the one every service runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_index.loader import default_index_path, load_index
from series_coverage import record_serving_index as record_serving_index_cli
from series_coverage.compatibility import (
    read_compatible_pointer,
    read_serving_index,
    record_if_compatible,
)
from series_coverage.loader import load_coverage
from series_coverage.schema import CoveragePointer
from tests.series_coverage.support import build_test_artifact, index_for

_MOVED = "provenance/RepublishedUnderAnotherDataset"


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return tmp_path / "coverage"


@pytest.fixture
def index(tmp_path: Path):
    """An index carrying the provenances every build `build_test_artifact` writes by default."""
    return index_for(build_test_artifact(tmp_path / "reference"))


def test_load_coverage_reads_the_build_a_pointer_names(base: Path) -> None:
    uri = build_test_artifact(base, stamp="stamp-1")
    build_test_artifact(base, stamp="stamp-2")

    loaded = load_coverage(uri, pointer=CoveragePointer(stamp="stamp-1", build_id="stamp-1-build"))

    assert (loaded.manifest.stamp, loaded.manifest.build_id) == ("stamp-1", "stamp-1-build")


def test_only_a_compatible_build_is_recorded(base: Path, index) -> None:
    uri = build_test_artifact(base, stamp="stamp-1", provenance=_MOVED)

    assert record_if_compatible(uri, index, load_coverage(uri))
    assert read_compatible_pointer(uri, index) is None

    build_test_artifact(base, stamp="stamp-2")

    assert not record_if_compatible(uri, index, load_coverage(uri))
    assert read_compatible_pointer(uri, index) == CoveragePointer(
        stamp="stamp-2", build_id="stamp-2-build"
    )


def test_the_deploy_records_the_packaged_index_as_serving(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = build_test_artifact(base, stamp="stamp-1")
    assert read_serving_index(uri) is None
    monkeypatch.setenv("COVERAGE_ARTIFACT_URI", uri)

    assert record_serving_index_cli.main() == 0

    serving = read_serving_index(uri)
    assert serving is not None
    assert serving.content_hash == load_index(default_index_path()).stamp.content_hash
