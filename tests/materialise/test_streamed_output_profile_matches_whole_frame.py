"""Checks output-profile invariance to chunk size, including pivot variation across chunks.

Two members share a `(recipient, time_period)` key, so `stat_var` must appear in `varying` in
both the single-chunk and streamed profiles. Setting `_FRAME_CHUNK_TARGET_ROWS` to 1 puts each
member in its own chunk; the default never splits these fixtures.
"""

from __future__ import annotations

import materialise.generation as generation_module
from materialise import EnrichmentLookup, MaterialisationResult, materialise_group
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._group_support import (
    _BARE_A,
    _BARE_B,
    _BARE_C,
    _BOUND_FACETS,
    _CSV_HEADER,
    _ENTITY,
    _OTHER_ENTITY,
    _ROLE,
    _SHARED_FACET,
    _SNAPSHOT,
    _STAT_VAR_A,
    _STAT_VAR_B,
    _STAT_VAR_C,
    _request,
    _sdmx_client_for,
)
from .conftest import FakeGraphReader, FakeRevisionReader

_STAT_VAR_D = "dcid:Stat_D"
_BARE_D = "Stat_D"

# `(bare_stat_var, dcid_stat_var, entity, year)`. A and B share one
# `(entity, year)` pair -- the duplicated pivot key this test exercises --
# while C and D each carry a distinct pair of their own.
_MEMBERS = (
    (_BARE_A, _STAT_VAR_A, _ENTITY, "2015"),
    (_BARE_B, _STAT_VAR_B, _ENTITY, "2015"),
    (_BARE_C, _STAT_VAR_C, _OTHER_ENTITY, "2016"),
    (_BARE_D, _STAT_VAR_D, _OTHER_ENTITY, "2017"),
)


def _row(*, bare_stat_var: str, entity: str, year: str, value: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{bare_stat_var},{entity},UNIT1,METH1,"
        f"NotApplicable,PROV1,{year},{value},,facet1\r\n"
    )


def _bodies() -> dict[str, str]:
    return {
        bare: _CSV_HEADER + _row(bare_stat_var=bare, entity=entity, year=year, value="100")
        for bare, _stat_var, entity, year in _MEMBERS
    }


def _coverage() -> CoverageIndex:
    """Builds coverage for members spread across `_ENTITY` and `_OTHER_ENTITY`."""
    role_names = {bare: RoleNames(donor="donor", recipient=_ROLE) for bare, *_rest in _MEMBERS}
    facet_metadata = {
        "facet1": FacetMetadata(provenance_id="PROV1", measurement_method="METH1", unit="UNIT1")
    }
    return build_coverage_index(
        [
            (bare, "facet1", "country/DONOR", entity, [2015, 2016, 2017])
            for bare, _stat_var, entity, _year in _MEMBERS
        ],
        role_names=role_names,
        facet_metadata=facet_metadata,
    )


def _run(monkeypatch, *, target_rows: int | None) -> MaterialisationResult:
    """Runs `materialise_group` with a fresh mock fetch and optional chunk-size override."""
    if target_rows is not None:
        monkeypatch.setattr(generation_module, "_FRAME_CHUNK_TARGET_ROWS", target_rows)
    graph = FakeGraphReader(
        facets={(bare, entity): [_SHARED_FACET] for bare, _, entity, _ in _MEMBERS}
    )
    sdmx, _requests = _sdmx_client_for(_bodies())
    requests = [
        _request(stat_var=stat_var, facet_bindings=_BOUND_FACETS, entity=entity)
        for _, stat_var, entity, _ in _MEMBERS
    ]
    return materialise_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=_coverage(),
    )


def test_streamed_output_profile_matches_the_single_chunk_case(monkeypatch):
    whole = _run(monkeypatch, target_rows=None)
    assert len(list(whole.frame_chunks())) == 1

    streamed = _run(monkeypatch, target_rows=1)
    assert len(list(streamed.frame_chunks())) == len(_MEMBERS)

    whole_profile = whole.output_profile
    streamed_profile = streamed.output_profile

    assert streamed_profile.rows == whole_profile.rows
    assert streamed_profile.columns == whole_profile.columns
    assert set(streamed_profile.pivots) == set(whole_profile.pivots)
    assert streamed_profile.enrichments == whole_profile.enrichments

    # The duplicated `(recipient, time_period)` key this fixture builds in:
    # `stat_var` must show up as `varying` on both sides, proving the
    # streamed run's own cross-chunk second pass agrees with the
    # whole-table run, not just the counts that never needed one.
    recipient_time = next(
        p for p in whole_profile.pivots if p.index == "recipient" and p.columns == "time_period"
    )
    assert recipient_time.duplicate_keys is True
    varying_names = {name for name, _distinct in recipient_time.varying}
    assert "stat_var" in varying_names

    streamed_recipient_time = next(
        p for p in streamed_profile.pivots if p.index == "recipient" and p.columns == "time_period"
    )
    assert streamed_recipient_time == recipient_time
