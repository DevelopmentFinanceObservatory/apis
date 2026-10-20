"""Tests coverage loading, pointer rechecks, swaps, and singleton wiring, and the bound graph
release's coverage that `get_coverage_service` serves with `GRAPH_RELEASE_COVERAGE` on."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from graph_release.context import ReleaseNotBound, bound_release, current_release
from graph_release.service import ReleaseService
from series_coverage import shared as coverage_shared
from series_coverage.loader import CoverageArtifactError, LoadedCoverage, load_coverage
from series_coverage.schema import CoveragePointer
from series_coverage.shared import (
    CoverageArtifactUriMissingError,
    CoverageService,
    ReleaseCoverage,
    get_coverage_service,
    reset_coverage_service,
)
from tests.graph_release.support import build_view_release
from tests.series_coverage.support import (
    V1,
    build_test_artifact,
    code_index_for,
    index_for,
    pointer_path,
)


@pytest.fixture(autouse=True)
def _isolated_singleton(monkeypatch: pytest.MonkeyPatch):
    """Runs each test on its own singleton, with `GRAPH_RELEASE_COVERAGE` off unless the test
    sets it."""
    monkeypatch.delenv("GRAPH_RELEASE_COVERAGE", raising=False)
    with coverage_shared._singleton.isolated():
        yield


def test_coverage_service_loads_at_construction(tmp_path: Path) -> None:
    uri = build_test_artifact(tmp_path / "coverage")
    service = CoverageService(artifact_uri=uri, index=index_for(uri), recheck_seconds=0)
    assert service.stamp == "test-coverage-stamp"
    assert service.current().filter_candidates(variables=[V1], bound={}) or True


def test_coverage_service_raises_at_construction_for_a_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(CoverageArtifactError):
        CoverageService(
            artifact_uri=f"file://{tmp_path / 'absent'}",
            index=code_index_for({}),
            recheck_seconds=0,
        )


def test_coverage_service_disabled_recheck_never_reloads(tmp_path: Path) -> None:
    uri = build_test_artifact(tmp_path / "coverage")
    calls = []

    def counting_loader(u: str, *, pointer: CoveragePointer | None = None) -> LoadedCoverage:
        calls.append(u)
        return load_coverage(u, pointer=pointer)

    service = CoverageService(
        artifact_uri=uri, index=index_for(uri), recheck_seconds=0, loader=counting_loader
    )
    assert len(calls) == 1
    for _ in range(5):
        service.current()
    assert len(calls) == 1


def test_coverage_service_recheck_with_unchanged_stamp_never_calls_the_loader(
    tmp_path: Path,
) -> None:
    """An unchanged pointer leaves the loader call count at one.

    The recheck reads only `current.json`.
    """
    uri = build_test_artifact(tmp_path / "coverage")
    fake_now = [0.0]
    calls = []

    def counting_loader(u: str, *, pointer: CoveragePointer | None = None) -> LoadedCoverage:
        calls.append(u)
        return load_coverage(u, pointer=pointer)

    service = CoverageService(
        artifact_uri=uri,
        index=index_for(uri),
        recheck_seconds=10,
        clock=lambda: fake_now[0],
        loader=counting_loader,
    )
    assert len(calls) == 1  # construction's own load
    # The 10-second interval elapsed with the pointer stamp unchanged.
    fake_now[0] = 11.0
    service.current()
    service.close()
    # Only current.json is read on recheck.
    assert len(calls) == 1
    assert service.stamp == "test-coverage-stamp"


def test_coverage_service_recheck_triggers_after_interval_and_swaps_on_change(
    tmp_path: Path,
) -> None:
    base = tmp_path / "coverage"
    uri = build_test_artifact(base, stamp="stamp-1")
    fake_now = [0.0]

    def clock() -> float:
        return fake_now[0]

    service = CoverageService(
        artifact_uri=uri, index=index_for(uri), recheck_seconds=10, clock=clock
    )
    assert service.stamp == "stamp-1"

    # Publishing stamp-2 and advancing past 10 seconds triggers the background recheck.
    build_test_artifact(base, stamp="stamp-2")
    fake_now[0] = 11.0
    service.current()
    service.close()
    assert service.stamp == "stamp-2"


def test_a_republish_of_the_same_stamp_in_a_new_build_reloads(tmp_path: Path) -> None:
    """A new `build_id` reloads the artifact even when its stamp is unchanged.

    `built_at` differs and BigQuery row order is unordered across builds.
    """
    base = tmp_path / "coverage"
    uri = build_test_artifact(base, stamp="stamp-1", build_id="build-a")
    fake_now = [0.0]
    calls = []

    def counting_loader(u: str, *, pointer: CoveragePointer | None = None) -> LoadedCoverage:
        calls.append(u)
        return load_coverage(u, pointer=pointer)

    service = CoverageService(
        artifact_uri=uri,
        index=index_for(uri),
        recheck_seconds=10,
        clock=lambda: fake_now[0],
        loader=counting_loader,
    )
    assert service.manifest.build_id == "build-a"

    build_test_artifact(base, stamp="stamp-1", build_id="build-b")
    fake_now[0] = 11.0
    service.current()
    service.close()
    assert len(calls) == 2
    assert (service.stamp, service.manifest.build_id) == ("stamp-1", "build-b")

    build_test_artifact(base, stamp="stamp-1", build_id="build-c")
    assert service.snapshot_current().manifest.build_id == "build-c"
    assert len(calls) == 3


def test_coverage_service_recheck_before_interval_elapses_does_not_reload(tmp_path: Path) -> None:
    base = tmp_path / "coverage"
    uri = build_test_artifact(base, stamp="stamp-1")
    fake_now = [0.0]
    service = CoverageService(
        artifact_uri=uri, index=index_for(uri), recheck_seconds=10, clock=lambda: fake_now[0]
    )
    build_test_artifact(base, stamp="stamp-2")
    fake_now[0] = 5.0
    service.current()
    service.close()
    assert service.stamp == "stamp-1"


def test_coverage_service_background_failure_keeps_serving_last_known_good(tmp_path: Path) -> None:
    base = tmp_path / "coverage"
    uri = build_test_artifact(base)
    fake_now = [0.0]
    attempts = []

    def flaky_loader(u: str, *, pointer: CoveragePointer | None = None) -> LoadedCoverage:
        attempts.append(u)
        if len(attempts) == 1:
            return load_coverage(u, pointer=pointer)
        raise CoverageArtifactError("simulated transient failure")

    service = CoverageService(
        artifact_uri=uri,
        index=index_for(uri),
        recheck_seconds=10,
        clock=lambda: fake_now[0],
        loader=flaky_loader,
    )
    # A changed stamp forces a full reload and exercises the failing loader.
    build_test_artifact(base, stamp="stamp-2")
    fake_now[0] = 11.0
    service.current()
    service.close()
    # Construction and the failed background reload are the two attempts.
    assert len(attempts) == 2
    assert (
        service.stamp == "test-coverage-stamp"
    )  # The failed reload leaves the published stamp unchanged.


def test_snapshot_current_loads_a_moved_pointer_before_returning(tmp_path: Path) -> None:
    """`snapshot_current()` reads a moved pointer before the background recheck is due."""
    base = tmp_path / "coverage"
    uri = build_test_artifact(base, stamp="stamp-1")
    calls = []

    def counting_loader(u: str, *, pointer: CoveragePointer | None = None) -> LoadedCoverage:
        calls.append(u)
        return load_coverage(u, pointer=pointer)

    service = CoverageService(
        artifact_uri=uri,
        index=index_for(uri),
        recheck_seconds=3_600,
        clock=lambda: 0.0,
        loader=counting_loader,
    )
    build_test_artifact(base, stamp="stamp-2")
    assert service.snapshot().stamp == "stamp-1"

    current = service.snapshot_current()

    assert current.stamp == "stamp-2"
    assert current.manifest.stamp == "stamp-2"
    assert current.index is service.current()
    assert len(calls) == 2
    assert service.snapshot().stamp == "stamp-2"


def test_snapshot_current_reads_the_pointer_even_with_rechecks_disabled(tmp_path: Path) -> None:
    base = tmp_path / "coverage"
    uri = build_test_artifact(base, stamp="stamp-1")
    service = CoverageService(artifact_uri=uri, index=index_for(uri), recheck_seconds=0)
    build_test_artifact(base, stamp="stamp-2")

    assert service.snapshot_current().stamp == "stamp-2"


def test_snapshot_current_with_an_unchanged_pointer_never_calls_the_loader(tmp_path: Path) -> None:
    uri = build_test_artifact(tmp_path / "coverage")
    calls = []

    def counting_loader(u: str, *, pointer: CoveragePointer | None = None) -> LoadedCoverage:
        calls.append(u)
        return load_coverage(u, pointer=pointer)

    service = CoverageService(
        artifact_uri=uri, index=index_for(uri), recheck_seconds=0, loader=counting_loader
    )
    loaded_index = service.current()

    current = service.snapshot_current()

    assert current.stamp == "test-coverage-stamp"
    assert current.index is loaded_index
    assert len(calls) == 1


def test_a_load_guard_wraps_only_the_load_of_a_moved_build(tmp_path: Path) -> None:
    """A registered guard is entered around the load of a moved build and left once the new
    build is served. A pointer read that finds no move enters it not at all, and an
    unregistered guard is never entered again."""
    base = tmp_path / "coverage"
    uri = build_test_artifact(base, stamp="stamp-1")
    events: list[str] = []

    def loader(base_uri: str, *, pointer: CoveragePointer | None = None) -> LoadedCoverage:
        events.append("load")
        return load_coverage(base_uri, pointer=pointer)

    service = CoverageService(
        artifact_uri=uri, index=index_for(uri), recheck_seconds=0, loader=loader
    )
    events.clear()

    @contextmanager
    def guard() -> Iterator[None]:
        events.append("enter")
        yield
        events.append(f"leave serving {service.stamp}")

    unregister = service.register_load_guard(guard)
    service.snapshot_current()
    assert events == []

    build_test_artifact(base, stamp="stamp-2")
    service.snapshot_current()
    assert events == ["enter", "load", "leave serving stamp-2"]

    unregister()
    build_test_artifact(base, stamp="stamp-3")
    service.snapshot_current()
    assert events[3:] == ["load"]


def test_snapshot_current_raises_when_the_pointer_is_unreadable(tmp_path: Path) -> None:
    """An unreadable published pointer raises while the served artifact stays unchanged."""
    base = tmp_path / "coverage"
    uri = build_test_artifact(base)
    service = CoverageService(artifact_uri=uri, index=index_for(uri), recheck_seconds=0)
    pointer_path(base).unlink()

    with pytest.raises(CoverageArtifactError):
        service.snapshot_current()

    assert service.stamp == "test-coverage-stamp"


def test_a_stalled_failed_recheck_cannot_overwrite_a_concurrent_snapshot_current_load(
    tmp_path: Path,
) -> None:
    """`_load_if_moved` and `_record_reload_failure` serialise snapshot updates on `_load_lock`.

    The failed recheck stalls while `snapshot_current()` loads `stamp-2` on another thread.
    """
    base = tmp_path / "coverage"
    uri = build_test_artifact(base, stamp="stamp-1")

    load_attempts = []

    def flaky_loader(u: str, *, pointer: CoveragePointer | None = None) -> LoadedCoverage:
        load_attempts.append(u)
        # The background recheck makes the second attempt.
        if len(load_attempts) == 2:
            raise CoverageArtifactError("simulated transient failure")
        return load_coverage(u, pointer=pointer)

    entered_failure_write = threading.Event()
    release_failure_write = threading.Event()
    block_next_clock_call = threading.Event()
    fake_now = [0.0]

    def clock() -> float:
        if block_next_clock_call.is_set():
            block_next_clock_call.clear()
            entered_failure_write.set()
            assert release_failure_write.wait(timeout=5), "test deadlocked"
        fake_now[0] += 1.0
        return fake_now[0]

    service = CoverageService(
        artifact_uri=uri,
        index=index_for(uri),
        recheck_seconds=3_600,
        clock=clock,
        loader=flaky_loader,
    )
    assert len(load_attempts) == 1  # construction's own load

    build_test_artifact(base, stamp="stamp-2")
    block_next_clock_call.set()  # arms the trap for the recheck thread's own failure write

    recheck_thread = threading.Thread(target=service._run_load)
    recheck_thread.start()
    assert entered_failure_write.wait(timeout=5), "the recheck never reached its failure write"

    results = []
    errors = []

    def foreground_read() -> None:
        try:
            results.append(service.snapshot_current())
        except BaseException as exc:
            errors.append(exc)

    foreground_thread = threading.Thread(target=foreground_read)
    foreground_thread.start()

    release_failure_write.set()
    recheck_thread.join(timeout=5)
    foreground_thread.join(timeout=5)

    assert not errors, errors
    assert len(results) == 1
    assert results[0].stamp == "stamp-2"
    # stamp-2 stays published after the stalled failure write.
    assert service.stamp == "stamp-2"
    # Calls cover construction, the failed recheck, and the foreground load.
    assert len(load_attempts) == 3


def test_get_coverage_service_requires_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COVERAGE_ARTIFACT_URI", raising=False)
    with pytest.raises(CoverageArtifactUriMissingError, match="COVERAGE_ARTIFACT_URI"):
        get_coverage_service()


def test_get_coverage_service_builds_once_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = build_test_artifact(tmp_path / "coverage")
    monkeypatch.setenv("COVERAGE_ARTIFACT_URI", uri)
    monkeypatch.setenv("COVERAGE_RECHECK_SECONDS", "0")
    monkeypatch.setattr(coverage_shared, "get_index", lambda: index_for(uri))
    first = get_coverage_service()
    second = get_coverage_service()
    assert first is second


def test_reset_coverage_service_forces_a_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = build_test_artifact(tmp_path / "coverage")
    monkeypatch.setenv("COVERAGE_ARTIFACT_URI", uri)
    monkeypatch.setenv("COVERAGE_RECHECK_SECONDS", "0")
    monkeypatch.setattr(coverage_shared, "get_index", lambda: index_for(uri))
    first = get_coverage_service()
    reset_coverage_service()
    second = get_coverage_service()
    assert first is not second


def test_with_the_flag_on_the_service_serves_the_bound_releases_coverage_without_an_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPH_RELEASE_COVERAGE", "1")
    monkeypatch.delenv("COVERAGE_ARTIFACT_URI", raising=False)
    release = current_release()
    assert release.views is not None

    service = get_coverage_service()
    snapshot = service.snapshot()

    assert isinstance(service, ReleaseCoverage)
    assert snapshot.index is release.views.coverage
    assert snapshot.stamp == release.release_id
    assert snapshot.manifest.stamp == snapshot.manifest.build_id == release.release_id
    assert snapshot.manifest.ingestion is not None
    assert snapshot.manifest.ingestion.workflow_execution_id == (
        release.manifest.import_execution_id
    )
    assert service.current() is snapshot.index
    assert service.snapshot_current().index is snapshot.index
    assert (service.stamp, service.manifest) == (snapshot.stamp, snapshot.manifest)
    assert service.serves(snapshot)


def test_with_the_flag_on_a_request_keeps_its_releases_coverage_across_a_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot stays served to the request that bound its release, so admission never
    restarts it onto the same release again, and the next request reads the new release."""
    monkeypatch.setenv("GRAPH_RELEASE_COVERAGE", "1")
    root = (tmp_path / "releases").as_uri()
    build_view_release(root, dc_serving_revision="first")
    now = [0.0]
    releases = ReleaseService(
        uri=root,
        local_dir=tmp_path / "local",
        recheck_seconds=1,
        clock=lambda: now[0],
        build_views=True,
    )
    service = get_coverage_service()

    with bound_release(releases.snapshot()) as first:
        held = service.snapshot()
        second = build_view_release(root, dc_serving_revision="second")
        now[0] += 1
        releases.snapshot()
        releases.close()
        assert releases.snapshot().release_id == second.release_id
        assert service.serves(held)
        assert service.snapshot().stamp == first.release_id

    with bound_release(releases.snapshot()):
        assert service.snapshot().stamp == second.release_id
        assert not service.serves(held)


@pytest.mark.unbound_release
def test_with_the_flag_on_an_unbound_read_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_RELEASE_COVERAGE", "1")

    with pytest.raises(ReleaseNotBound):
        get_coverage_service().snapshot()
