"""The coverage a surface serves: the coverage view of the graph release bound to the current
request or operation.

`get_coverage_service()` returns `ReleaseCoverage`, which reads the bound release's coverage
view (`graph_release.service.ReleaseViews.coverage`). That release's code index comes from the same
release, so the two always agree. Its stamp and build id are the release id. A release swap
holds every materialise permit through the release service's swap guard
(`ReleaseService.register_swap_guard`). A request still bound to the replaced release is refused
at the Datasets admission checks, since it read its code index from that release.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from artifact_publication.jobs import Ingestion
from graph_release.context import current_release
from series_coverage.index import CoverageIndex
from series_coverage.schema import CoverageManifest

if TYPE_CHECKING:
    from graph_release.service import Release


@dataclass(frozen=True, slots=True)
class CoverageSnapshot:
    """Index, manifest, and stamp read together from one release's coverage view."""

    index: CoverageIndex
    manifest: CoverageManifest
    stamp: str


class ServedCoverage(Protocol):
    """The coverage a surface serves, as `ReleaseCoverage` serves it."""

    def current(self) -> CoverageIndex: ...

    @property
    def stamp(self) -> str: ...

    @property
    def manifest(self) -> CoverageManifest: ...

    def snapshot(self) -> CoverageSnapshot: ...


class ReleaseCoverage:
    """The coverage view of the graph release bound to the current context, read on every call,
    so a request serves the coverage of the release it bound and the next request after a swap
    serves the new one. Holds no index, so a swapped-out release dies with the last request that
    bound it.

    Every read raises `graph_release.context.ReleaseNotBound` when no release is bound.
    """

    def current(self) -> CoverageIndex:
        return self.snapshot().index

    @property
    def stamp(self) -> str:
        """The bound release's id."""
        return current_release().release_id

    @property
    def manifest(self) -> CoverageManifest:
        return self.snapshot().manifest

    def snapshot(self) -> CoverageSnapshot:
        return _release_snapshot(current_release())


def _release_snapshot(release: Release) -> CoverageSnapshot:
    """`release`'s coverage view, with a manifest naming the release as the build."""
    index = release.views.coverage
    manifest = release.manifest
    return CoverageSnapshot(
        index=index,
        manifest=CoverageManifest(
            stamp=release.release_id,
            build_id=release.release_id,
            built_at=manifest.read_timestamp,
            source="graph_release",
            row_counts={},
            files={},
            horizon=index.horizon,
            # The release records its import's completion time and no creation time, so the
            # `artifact_pair_stale` warning, which compares execution ids, names the former.
            ingestion=Ingestion(
                workflow_execution_id=manifest.import_execution_id,
                creation_timestamp=manifest.import_completed_at,
            ),
        ),
        stamp=release.release_id,
    )


_RELEASE_COVERAGE = ReleaseCoverage()


def get_coverage_service() -> ReleaseCoverage:
    """Returns the process's `ReleaseCoverage`, which reads the release bound to each call."""
    return _RELEASE_COVERAGE


__all__ = [
    "CoverageSnapshot",
    "ReleaseCoverage",
    "ServedCoverage",
    "get_coverage_service",
]
