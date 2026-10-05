"""The coverage service serves only a build that carries the code index's provenances.

A published build that differs is rejected. A running service keeps its build or moves to the
build the index's compatibility pointer names, and a cold start loads that build or fails.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from code_index.shared import get_index
from series_coverage import record_serving_index as record_serving_index_cli
from series_coverage import shared as coverage_shared
from series_coverage.compatibility import (
    compatible_pointer_uri,
    read_compatible_pointer,
    read_serving_index,
    record_if_compatible,
)
from series_coverage.io import write_current
from series_coverage.loader import LoadedCoverage, load_coverage
from series_coverage.schema import CoveragePointer
from series_coverage.shared import (
    CoverageIncompatibleError,
    CoverageIndexCheckValueError,
    CoverageService,
    get_coverage_service,
)
from tests.series_coverage.support import V1, build_test_artifact, index_for

_MOVED = "provenance/RepublishedUnderAnotherDataset"
# Build ids as `new_build_id` writes them, in publication order.
_BUILD_1, _BUILD_2, _BUILD_3 = (
    "20260924T000000Z-0a0a0a0a",
    "20260925T000000Z-0b0b0b0b",
    "20260926T000000Z-0c0c0c0c",
)
_SLICE_URI = (Path(__file__).resolve().parents[1] / "fixtures" / "coverage_slice").as_uri()


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return tmp_path / "coverage"


@pytest.fixture
def index(tmp_path: Path):
    """An index carrying the provenances every build `build_test_artifact` writes by default."""
    return index_for(build_test_artifact(tmp_path / "reference"))


def _record_published(uri: str, index) -> None:
    assert not record_if_compatible(uri, index, load_coverage(uri))


def _counting_loader(calls: list[str | None]):
    def loader(uri: str, *, pointer: CoveragePointer | None = None) -> LoadedCoverage:
        calls.append(None if pointer is None else pointer.build_id)
        return load_coverage(uri, pointer=pointer)

    return loader


def test_a_compatible_refresh_is_adopted(base: Path, index) -> None:
    uri = build_test_artifact(base, stamp="stamp-1")
    service = CoverageService(artifact_uri=uri, index=index, recheck_seconds=0)
    build_test_artifact(base, stamp="stamp-2")

    assert service.snapshot_current().stamp == "stamp-2"
    assert service.rejected_build is None


def test_an_incompatible_refresh_keeps_the_served_build(
    base: Path, index, caplog: pytest.LogCaptureFixture
) -> None:
    uri = build_test_artifact(base, stamp="stamp-1")
    calls: list[str | None] = []
    service = CoverageService(
        artifact_uri=uri, index=index, recheck_seconds=0, loader=_counting_loader(calls)
    )
    build_test_artifact(base, stamp="stamp-2", provenance=_MOVED)

    with caplog.at_level(logging.ERROR, logger=coverage_shared.__name__):
        served = service.snapshot_current()

    assert (served.stamp, service.stamp) == ("stamp-1", "stamp-1")
    assert service.rejected_build == "stamp-2-build"
    (record,) = caplog.records
    message = record.getMessage()
    assert "build 'stamp-2-build'" in message
    assert "serving build 'stamp-1-build'" in message
    assert f"{V1}: index ['provenance/OECDCreditorReportingSystem'], build ['{_MOVED}']" in message
    # The rejected build is not loaded again while `current.json` names it.
    assert service.snapshot_current().stamp == "stamp-1"
    assert calls == [None, None]


def test_an_incompatible_refresh_moves_to_a_newer_recorded_build(base: Path, index) -> None:
    uri = build_test_artifact(base, stamp="stamp-1", build_id=_BUILD_1)
    service = CoverageService(artifact_uri=uri, index=index, recheck_seconds=0)
    build_test_artifact(base, stamp="stamp-2", build_id=_BUILD_2)
    _record_published(uri, index)
    build_test_artifact(base, stamp="stamp-3", build_id=_BUILD_3, provenance=_MOVED)

    assert service.snapshot_current().stamp == "stamp-2"
    assert service.rejected_build == _BUILD_3


def test_an_incompatible_refresh_never_moves_back_to_an_older_recorded_build(
    base: Path, index
) -> None:
    """The pointer lags `current.json` while the job runs another index than the service."""
    uri = build_test_artifact(base, stamp="stamp-1", build_id=_BUILD_1)
    _record_published(uri, index)
    calls: list[str | None] = []
    service = CoverageService(
        artifact_uri=uri, index=index, recheck_seconds=0, loader=_counting_loader(calls)
    )
    build_test_artifact(base, stamp="stamp-2", build_id=_BUILD_2)
    assert service.snapshot_current().stamp == "stamp-2"
    build_test_artifact(base, stamp="stamp-3", build_id=_BUILD_3, provenance=_MOVED)

    assert service.snapshot_current().stamp == "stamp-2"
    assert service.rejected_build == _BUILD_3
    assert calls == [None, None, None]


def test_a_compatible_publish_after_a_rejection_is_adopted(base: Path, index) -> None:
    uri = build_test_artifact(base, stamp="stamp-1")
    service = CoverageService(artifact_uri=uri, index=index, recheck_seconds=0)
    build_test_artifact(base, stamp="stamp-2", provenance=_MOVED)
    service.snapshot_current()
    build_test_artifact(base, stamp="stamp-3")

    assert service.snapshot_current().stamp == "stamp-3"
    assert service.rejected_build is None


def test_a_cold_start_on_an_incompatible_build_serves_the_recorded_build(base: Path, index) -> None:
    uri = build_test_artifact(base, stamp="stamp-1")
    _record_published(uri, index)
    build_test_artifact(base, stamp="stamp-2", provenance=_MOVED)

    service = CoverageService(artifact_uri=uri, index=index, recheck_seconds=0)

    assert service.stamp == "stamp-1"
    assert service.rejected_build == "stamp-2-build"


def test_a_cold_start_with_no_recorded_build_fails(base: Path, index) -> None:
    uri = build_test_artifact(base, stamp="stamp-1", provenance=_MOVED)

    with pytest.raises(CoverageIncompatibleError, match="names no build"):
        CoverageService(artifact_uri=uri, index=index, recheck_seconds=0)


def test_a_cold_start_whose_recorded_build_also_differs_fails(base: Path, index) -> None:
    uri = build_test_artifact(base, stamp="stamp-1", provenance=_MOVED)
    build_test_artifact(base, stamp="stamp-2", provenance=_MOVED)
    pointer_file = Path(compatible_pointer_uri(uri, index).removeprefix("file://"))
    pointer_file.parent.mkdir(parents=True)
    write_current(CoveragePointer(stamp="stamp-1", build_id="stamp-1-build"), pointer_file)

    with pytest.raises(CoverageIncompatibleError, match="which differs too"):
        CoverageService(artifact_uri=uri, index=index, recheck_seconds=0)


def test_the_process_service_refuses_a_build_the_packaged_index_does_not_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface construction acquires this service, so the refusal fails readiness. The
    recorded slice covers a subset of the packaged index's StatVars."""
    monkeypatch.setenv("COVERAGE_ARTIFACT_URI", _SLICE_URI)
    monkeypatch.setenv("COVERAGE_RECHECK_SECONDS", "0")

    with coverage_shared._singleton.isolated(), pytest.raises(CoverageIncompatibleError):
        get_coverage_service()


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


def test_index_check_off_serves_the_build_unchecked_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("COVERAGE_ARTIFACT_URI", _SLICE_URI)
    monkeypatch.setenv("COVERAGE_RECHECK_SECONDS", "0")
    monkeypatch.setenv("COVERAGE_INDEX_CHECK", "off")

    with (
        coverage_shared._singleton.isolated(),
        caplog.at_level(logging.WARNING, logger=coverage_shared.__name__),
    ):
        service = get_coverage_service()

    assert service.manifest.build_id == "20260927T142252Z-slice"
    assert service.rejected_build is None
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "COVERAGE_INDEX_CHECK=off" in message
    assert "'20260927T142252Z-slice'" in message
    assert repr(get_index().stamp.content_hash) in message


def test_an_unknown_index_check_value_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COVERAGE_ARTIFACT_URI", _SLICE_URI)
    monkeypatch.setenv("COVERAGE_INDEX_CHECK", "false")

    with (
        coverage_shared._singleton.isolated(),
        pytest.raises(CoverageIndexCheckValueError, match="'false'"),
    ):
        get_coverage_service()


def test_the_deploy_records_the_packaged_index_as_serving(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = build_test_artifact(base, stamp="stamp-1")
    assert read_serving_index(uri) is None
    monkeypatch.setenv("COVERAGE_ARTIFACT_URI", uri)

    assert record_serving_index_cli.main() == 0

    serving = read_serving_index(uri)
    assert serving is not None
    assert serving.content_hash == get_index().stamp.content_hash
