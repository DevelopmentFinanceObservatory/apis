"""Checks group `content_version` and rows across single- and multi-chunk runs.

Setting `_FRAME_CHUNK_TARGET_ROWS` to 1 puts each member in its own chunk. The default (100,000)
never splits these small fixtures.
"""

from __future__ import annotations

import materialise.generation as generation_module
from materialise import EnrichmentLookup, materialise_group

from ._group_support import (
    _BARE_A,
    _BARE_B,
    _BARE_C,
    _BOUND_FACETS,
    _CSV_HEADER,
    _ENTITY,
    _SHARED_FACET,
    _STAT_VAR_A,
    _STAT_VAR_B,
    _STAT_VAR_C,
    _coverage_for,
    _csv_row,
    _request,
    _sdmx_client_for,
)
from .conftest import FakeGraphReader

_STAT_VAR_D = "dcid:Stat_D"
_BARE_D = "Stat_D"
_STAT_VAR_E = "dcid:Stat_E"
_BARE_E = "Stat_E"

_MEMBERS = (
    (_BARE_A, _STAT_VAR_A, "2015"),
    (_BARE_B, _STAT_VAR_B, "2016"),
    (_BARE_C, _STAT_VAR_C, "2017"),
    (_BARE_D, _STAT_VAR_D, "2018"),
    (_BARE_E, _STAT_VAR_E, "2019"),
)


def _bodies() -> dict[str, str]:
    return {
        bare: _CSV_HEADER
        + _csv_row(
            stat_var=stat_var,
            year=year,
            value="100",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )
        for bare, stat_var, year in _MEMBERS
    }


def _run(
    monkeypatch, *, target_rows: int | None
) -> tuple[str, tuple[tuple[object, ...], ...], int]:
    """Runs a fresh group fetch and returns its version, concatenated rows, and chunk count."""
    if target_rows is not None:
        monkeypatch.setattr(generation_module, "_FRAME_CHUNK_TARGET_ROWS", target_rows)
    graph = FakeGraphReader(facets={(bare, _ENTITY): [_SHARED_FACET] for bare, _, _ in _MEMBERS})
    sdmx, _requests = _sdmx_client_for(_bodies())
    requests = [
        _request(stat_var=stat_var, facet_bindings=_BOUND_FACETS) for _, stat_var, _ in _MEMBERS
    ]
    coverage = _coverage_for([(bare, "facet1") for bare, _, _ in _MEMBERS])
    result = materialise_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties={}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )
    chunks = list(result.frame_chunks())
    rows = tuple(row for chunk in chunks for row in chunk.rows)
    return result.content_version, rows, len(chunks)


def test_streamed_content_version_matches_the_single_chunk_case(monkeypatch):
    whole_version, whole_rows, whole_chunk_count = _run(monkeypatch, target_rows=None)
    assert whole_chunk_count == 1

    streamed_version, streamed_rows, streamed_chunk_count = _run(monkeypatch, target_rows=1)
    assert streamed_chunk_count == len(_MEMBERS)

    assert streamed_version == whole_version
    assert streamed_rows == whole_rows
