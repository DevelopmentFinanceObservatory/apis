"""Tests the bound graph release's coverage that `get_coverage_service` serves."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_release.context import ReleaseNotBound, bound_release, current_release
from graph_release.service import ReleaseService
from series_coverage.shared import ReleaseCoverage, get_coverage_service
from tests.graph_release.support import build_view_release


def test_the_service_serves_the_bound_releases_coverage() -> None:
    release = current_release()

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
    assert (service.stamp, service.manifest) == (snapshot.stamp, snapshot.manifest)


def test_a_request_keeps_its_releases_coverage_across_a_swap(tmp_path: Path) -> None:
    """A request reads the coverage of the release it bound after the service swaps it out,
    and the next request reads the new release's."""
    root = (tmp_path / "releases").as_uri()
    build_view_release(root, dc_serving_revision="first")
    now = [0.0]
    releases = ReleaseService(
        uri=root, local_dir=tmp_path / "local", recheck_seconds=1, clock=lambda: now[0]
    )
    service = get_coverage_service()

    with bound_release(releases.snapshot()) as first:
        held = service.snapshot()
        second = build_view_release(root, dc_serving_revision="second")
        now[0] += 1
        releases.snapshot()
        releases.close()
        assert releases.snapshot().release_id == second.release_id
        assert service.snapshot().index is held.index
        assert service.snapshot().stamp == first.release_id

    with bound_release(releases.snapshot()):
        assert service.snapshot().stamp == second.release_id
        assert service.snapshot().index is not held.index


@pytest.mark.unbound_release
def test_an_unbound_read_raises() -> None:
    with pytest.raises(ReleaseNotBound):
        get_coverage_service().snapshot()
