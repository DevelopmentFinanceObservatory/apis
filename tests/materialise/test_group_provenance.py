"""Group frames use the single provenance selected for all members.

The frame-level assertion protects the groupby consumer from mixed provenances.
"""

from __future__ import annotations

from materialise import EnrichmentLookup, materialise_group
from materialise.ceiling import DEFAULT_SYNC_CEILING_ROWS
from materialise.facets import FacetBinding
from materialise.generation import _generate_group
from materialise.graph import Facet

from ._group_support import (
    _BARE_A,
    _BARE_B,
    _CSV_HEADER,
    _ENTITY,
    _STAT_VAR_A,
    _STAT_VAR_B,
    _coverage_for,
    _csv_row,
    _request,
    _sdmx_client_for,
)
from .conftest import FakeGraphReader, only_chunk

_SHARED_FACET = Facet(
    facet_id="facetShared",
    provenance_id="PROV1",
    measurement_method="METH1",
    unit="UNIT1",
    obs_count=1,
    earliest_date="2015",
    latest_date="2015",
)
_BOUND_FACETS = {
    "provenance": FacetBinding(state="bound", ref="PROV1"),
    "measurementMethod": FacetBinding(state="bound", ref="METH1"),
    "unit": FacetBinding(state="bound", ref="UNIT1"),
}


def test_group_frame_rows_carry_one_provenance_across_members():
    # Both members select the same provenance for the merged frame.
    request_a = _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS)
    request_b = _request(stat_var=_STAT_VAR_B, facet_bindings=_BOUND_FACETS)
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [_SHARED_FACET],
            (_BARE_B, _ENTITY): [_SHARED_FACET],
        }
    )
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="10",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facetShared",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2016",
            value="20",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facetShared",
        ),
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    coverage = _coverage_for([(_BARE_A, "facetShared"), (_BARE_B, "facetShared")])

    result = materialise_group(
        [request_a, request_b],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert result.coverage.rows == 2
    frame = only_chunk(result)
    assert set(frame.table["provenance"].to_pylist()) == {"PROV1"}


def test_group_generation_frame_carries_one_provenance_even_when_a_member_contributes_no_row():
    request_a = _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS)
    request_b = _request(stat_var=_STAT_VAR_B, facet_bindings=_BOUND_FACETS)
    graph = FakeGraphReader(facets={(_BARE_A, _ENTITY): [_SHARED_FACET]})
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="10",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facetShared",
        ),
        _BARE_B: _CSV_HEADER,
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    # B has no candidate at all (no `RoleNames` entry for `_BARE_B`): B's
    # own pool is empty, matching the fixtured graph facets above.
    coverage = _coverage_for([(_BARE_A, "facetShared")])

    outcome = _generate_group(
        [request_a, request_b],
        sdmx=sdmx,
        graph=graph,
        ceiling=DEFAULT_SYNC_CEILING_ROWS,
        member_concurrency=1,
        coverage_index=coverage,
    )

    assert outcome.member_observation_counts == {_STAT_VAR_A: 1, _STAT_VAR_B: 0}
    provenance_cells = only_chunk(outcome).table["provenance"].to_pylist()
    assert len(set(provenance_cells)) == 1
    assert set(provenance_cells) == {"PROV1"}
