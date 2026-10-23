"""Provide in-memory coverage fixtures for routes and direct engine tests.
`patch_coverage` patches the two request entry points; direct consumers need
an explicit `CoverageSnapshot` because their coverage is a required argument.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from search.engine import app as search_app
from search.engine import core as search_core
from series_coverage.index import CoverageIndex
from series_coverage.schema import CoverageManifest, FacetMetadata, RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.fixtures.provenance import index_provenance
from tests.series_coverage.support import coverage_table

# (variable, facet_id, donor, recipient, years)
CoverageRow = tuple[str, str, str, str, Sequence[int]]

# The provenance a fixture facet carries when a test names none.
FIXTURE_PROVENANCE = "dc/base/CoverageFixture"


def build_coverage_index(
    rows: Sequence[CoverageRow],
    *,
    role_names: Mapping[str, RoleNames],
    facet_metadata: Mapping[str, FacetMetadata] | None = None,
    build_year: int | None = None,
) -> CoverageIndex:
    """Build an in-memory index from `rows`, built in `build_year` (no year by default, so
    no year counts as a projection).

    `facet_metadata` mirrors an artifact's `facets.json`; supply it when tests resolve
    provenance, unit, or measurement method from coverage. A facet it leaves out gets a
    provenance: the one dataset the code index records its variables under
    (`fixtures.provenance.index_provenance`), else `FIXTURE_PROVENANCE`, and the `P1Y` period
    every DFO import states, since a coverage facet stating none is a cadence of its own
    (`materialise.facets.cadences_by_member`). The loader refuses
    an artifact without an entry for every facet, and the job reads each entry's provenance
    from the facet's time series.
    """
    table, horizon = coverage_table(rows)
    datasets: dict[str, set[str | None]] = {}
    for variable, facet_id, *_rest in rows:
        datasets.setdefault(facet_id, set()).add(index_provenance(variable))
    metadata = {
        facet_id: FacetMetadata(
            provenance_id=(next(iter(found)) if len(found) == 1 else None) or FIXTURE_PROVENANCE,
            observation_period="P1Y",
        )
        for facet_id, found in datasets.items()
    }
    metadata.update(facet_metadata or {})
    return CoverageIndex(
        table, role_names, facet_metadata=metadata, horizon=horizon, build_year=build_year
    )


class FakeCoverageService:
    """Return a fixed index, stamp, and snapshot with a synthetic manifest."""

    def __init__(self, index: CoverageIndex, *, stamp: str = "test-coverage-stamp") -> None:
        self._index = index
        self.stamp = stamp
        self._manifest = CoverageManifest(
            stamp=stamp,
            build_id=f"{stamp}-build",
            built_at="1970-01-01T00:00:00Z",
            horizon=index.horizon,
        )

    def current(self) -> CoverageIndex:
        return self._index

    def snapshot(self) -> CoverageSnapshot:
        return CoverageSnapshot(index=self._index, manifest=self._manifest, stamp=self.stamp)


def empty_coverage_snapshot(*, stamp: str = "test-coverage-stamp") -> CoverageSnapshot:
    """Build an empty snapshot for tests that do not exercise coverage resolution."""
    return FakeCoverageService(build_coverage_index([], role_names={}), stamp=stamp).snapshot()


def patch_coverage(
    monkeypatch: pytest.MonkeyPatch,
    rows: Sequence[CoverageRow],
    *,
    role_names: Mapping[str, RoleNames],
    facet_metadata: Mapping[str, FacetMetadata] | None = None,
) -> CoverageIndex:
    """Build an index and patch `app` and `core` service bindings for this test.

    Route tests receive snapshots through those entry points; direct consumers need an explicit
    snapshot because they take coverage as a required argument. `monkeypatch` reverts the bindings.
    Returns the index; `facet_metadata` passes to `build_coverage_index`.
    """
    index = build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)
    service = FakeCoverageService(index)
    monkeypatch.setattr(search_app, "get_coverage_service", lambda: service)
    monkeypatch.setattr(search_core, "get_coverage_service", lambda: service)
    return index
