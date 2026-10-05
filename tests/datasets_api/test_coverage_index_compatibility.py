"""A coverage publish whose provenances differ from the packaged code index leaves the mint
door serving the build the election agrees with.

Dataset election reads `StatVarRow.provenances` from the packaged index, and the mint reads
facet provenances from the coverage build the service serves. Each build here covers every
row of the packaged index under its own provenances; the moved build republishes the two
members populated at Kenya under another dataset.
"""

from __future__ import annotations

from pathlib import Path

from code_index.query import CodeIndex
from code_index.shared import get_index
from series_coverage.compatibility import record_if_compatible
from series_coverage.io import read_current, write_current
from series_coverage.loader import LoadedCoverage
from series_coverage.schema import CoverageManifest, CoveragePointer, FacetMetadata, RoleNames
from series_coverage.shared import CoverageService
from tests.code_index.aid_identity import CRS_PROVENANCE
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.series_coverage.support import pointer_path

from ._graph_doubles import FakeMintGraphReader
from .test_definition_door import _client
from .test_preview_family import (
    _PROVIDER,
    _RECIPIENT,
    _YEARS,
    _family,
    _input,
    _populated,
    _sdmx_transport,
)

_MOVED = "provenance/RepublishedUnderAnotherDataset"
_MINT_BODY = {"schema_version": 1, "options": {"bundle": False}}


def _build(
    index: CodeIndex, populated: tuple[str, ...], *, build_id: str, moved: bool
) -> LoadedCoverage:
    """A build holding one row per provenance of every index row. `populated` members hold
    their rows at Kenya, under `_MOVED` when `moved`."""
    rows = []
    facet_metadata: dict[str, FacetMetadata] = {}
    for row in index.statvars:
        for provenance in row.provenances:
            if moved and row.dcid in populated:
                provenance = _MOVED
            facet_id = "facet1" if provenance == CRS_PROVENANCE else f"facet-{provenance}"
            facet_metadata[facet_id] = FacetMetadata(
                provenance_id=provenance, measurement_method="GrantEquivalent", unit="USD"
            )
            if row.dcid in populated:
                rows.append((row.dcid, facet_id, _PROVIDER, _RECIPIENT, list(_YEARS)))
            else:
                rows.append((row.dcid, facet_id, "country/USA", "country/USA", [2000]))
    coverage = build_coverage_index(
        rows,
        role_names={
            row.dcid: RoleNames(donor="financingProvider", recipient="recipient")
            for row in index.statvars
        },
        facet_metadata=facet_metadata,
    )
    manifest = CoverageManifest(
        stamp=f"stamp-{build_id}",
        build_id=build_id,
        built_at="2026-10-01T00:00:00Z",
        source="tests",
        row_counts={"rows": len(rows)},
        files={},
        horizon=coverage.horizon,
    )
    return LoadedCoverage(manifest=manifest, index=coverage)


def _publish(base: Path, build: LoadedCoverage) -> None:
    path = pointer_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_current(
        CoveragePointer(stamp=build.manifest.stamp, build_id=build.manifest.build_id), path
    )


def _mint(coverage, populated: tuple[str, ...]):
    """Mints the education-aid-to-Kenya identity against an empty registry."""
    client = _client(
        graph=FakeMintGraphReader(labels={_PROVIDER: "France", _RECIPIENT: "Kenya"}),
        sdmx_transport=_sdmx_transport(populated),
        coverage=coverage,
    )
    return client.post("/api/datasets", json={**_MINT_BODY, "input": _input()})


def test_an_incompatible_publish_leaves_mint_and_election_agreeing(tmp_path: Path) -> None:
    index = get_index()
    populated = _populated(_family())
    builds = {
        build.manifest.build_id: build
        for build in (
            _build(index, populated, build_id="build-a", moved=False),
            _build(index, populated, build_id="build-b", moved=True),
        )
    }
    base = tmp_path / "coverage"
    uri = base.as_uri()

    def loader(base_uri: str, *, pointer: CoveragePointer | None = None) -> LoadedCoverage:
        return builds[(pointer or read_current(pointer_path(base))).build_id]

    _publish(base, builds["build-a"])
    assert not record_if_compatible(uri, index, builds["build-a"])
    service = CoverageService(artifact_uri=uri, index=index, recheck_seconds=0, loader=loader)
    before = _mint(service, populated)
    _publish(base, builds["build-b"])

    assert service.snapshot_current().manifest.build_id == "build-a"
    after = _mint(service, populated)

    assert service.rejected_build == "build-b"
    assert before.status_code == 200, before.text
    assert after.status_code == 200, after.text
    assert after.json()["content_version"] == before.json()["content_version"]
    assert [source["ref"] for source in after.json()["citation"]["sources"]] == [CRS_PROVENANCE]


def test_serving_the_rejected_build_would_split_the_mint_from_the_election() -> None:
    """The control: the moved build, served as it stands, refuses the mint."""
    populated = _populated(_family())
    moved = _build(get_index(), populated, build_id="build-b", moved=True)

    resp = _mint(FakeCoverageService(moved.index), populated)

    assert resp.status_code == 500, resp.text
    assert _MOVED in resp.json()["detail"]
