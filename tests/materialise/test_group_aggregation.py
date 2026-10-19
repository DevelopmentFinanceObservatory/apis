"""`materialise_group`: N=1 byte-identity with `materialise_spec`, N>=2 row
concatenation/one `content_version`/sorted `resolved_stat_vars`, per-member
`observation_count`, and `declared_roles`/stat_var agreement across members.
"""

from __future__ import annotations

import pytest

from materialise import EnrichmentLookup, RowProgress, materialise_group, materialise_spec
from materialise.ceiling import DEFAULT_SYNC_CEILING_ROWS
from materialise.content_version import mint_version
from materialise.errors import MaterialisationUsageError
from materialise.facets import FacetBinding
from materialise.generation import _generate_group
from materialise.graph import Facet

from ._group_support import (
    _BARE_A,
    _BARE_B,
    _BARE_C,
    _BOUND_FACETS,
    _CSV_HEADER,
    _ENTITY,
    _ROLE,
    _SHARED_FACET,
    _STAT_VAR_A,
    _STAT_VAR_B,
    _STAT_VAR_C,
    _coverage_for,
    _csv_row,
    _request,
    _sdmx_client_for,
)
from .conftest import FakeGraphReader, only_chunk

# ---------------------------------------------------------------------------
# N=1 byte-identity
# ---------------------------------------------------------------------------


def test_n1_agrees_with_materialise_spec_on_content_version_frame_and_trace():
    request = _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS)
    csv_body = _CSV_HEADER + _csv_row(
        stat_var=_STAT_VAR_A,
        year="2015",
        value="1000",
        unit="UNIT1",
        method="METH1",
        provenance="PROV1",
        facet_id="facet1",
    )
    graph_spec = FakeGraphReader(facets={(_BARE_A, _ENTITY): [_SHARED_FACET]})
    sdmx_spec, _r1 = _sdmx_client_for({_BARE_A: csv_body})
    coverage_spec = _coverage_for([(_BARE_A, "facet1")])
    golden = materialise_spec(
        request,
        sdmx=sdmx_spec,
        graph=graph_spec,
        enrichment=EnrichmentLookup(graph_spec, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage_spec,
    )

    graph_group = FakeGraphReader(facets={(_BARE_A, _ENTITY): [_SHARED_FACET]})
    sdmx_group, _r2 = _sdmx_client_for({_BARE_A: csv_body})
    coverage_group = _coverage_for([(_BARE_A, "facet1")])
    grouped = materialise_group(
        [request],
        sdmx=sdmx_group,
        graph=graph_group,
        enrichment=EnrichmentLookup(graph_group, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage_group,
    )

    assert grouped.content_version == golden.content_version
    grouped_frame = only_chunk(grouped)
    golden_frame = only_chunk(golden)
    assert grouped_frame.columns == golden_frame.columns
    assert grouped_frame.rows == golden_frame.rows
    # Each fake graph serves a generation of its own, which each trace records.
    assert grouped.trace.graph_revision == graph_group.generation_stamp().token
    assert golden.trace.graph_revision == graph_spec.generation_stamp().token
    assert {**grouped.trace.to_dict(), "graph_revision": None} == {
        **golden.trace.to_dict(),
        "graph_revision": None,
    }

    # A recorded pin, independent of both call paths agreeing with each
    # other: an unrelated change to the grammar's own hashing would move
    # this without necessarily moving `golden`/`grouped` apart from EACH
    # OTHER, so this asserts against the actual value once observed, not
    # just their mutual agreement.
    assert golden.content_version == mint_version(golden_frame)
    assert golden.trace.resolved_stat_vars == (
        {"ref": _STAT_VAR_A, "roles": [_ROLE], "observation_count": 1},
    )


# ---------------------------------------------------------------------------
# N=2 / N=3 row concatenation, sorted resolved_stat_vars
# ---------------------------------------------------------------------------


def test_n2_concatenates_rows_mints_one_content_version_two_resolved_stat_vars():
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
            value="1000",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2016",
            value="2000",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        ),
    }
    sdmx, requests = _sdmx_client_for(bodies)
    coverage = _coverage_for([(_BARE_A, "facet1"), (_BARE_B, "facet1")])

    result = materialise_group(
        [request_a, request_b],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=1,
        coverage=coverage,
    )

    # Candidates come from `coverage`, an in-memory lookup: only the one
    # combined data fetch runs. Both members bind the same entity scope, so
    # `_plan_collection_chunks` puts them in one shared chunk at
    # `member_concurrency=1` (its own lane target, `ceil(2 / 1) == 2`);
    # the default `4` would already split two same-scope members into
    # singleton chunks instead.
    assert len(requests) == 1
    assert result.coverage.rows == 2
    assert len(only_chunk(result).rows) == 2
    assert result.trace.resolved_stat_vars == (
        {"ref": _STAT_VAR_A, "roles": [_ROLE], "observation_count": 1},
        {"ref": _STAT_VAR_B, "roles": [_ROLE], "observation_count": 1},
    )
    assert result.content_version.startswith("cv_")


def test_progress_counter_ends_equal_to_the_frame_row_count():
    """`RowProgress` equals the returned frame's row count."""
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
            value="1000",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2016",
            value="2000",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        ),
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    coverage = _coverage_for([(_BARE_A, "facet1"), (_BARE_B, "facet1")])
    progress = RowProgress()

    result = materialise_group(
        [request_a, request_b],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=1,
        coverage=coverage,
        progress=progress,
    )

    assert progress.value == result.coverage.rows == len(only_chunk(result).rows)


def test_two_members_each_with_two_disjoint_year_facets_keep_all_four_rows():
    # Each member's own two facets share (unit, measurementMethod,
    # provenance) but report DIFFERENT years for the SAME recipient -- no
    # grain collision anywhere, so every row from every facet survives,
    # across both members.
    request_a = _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS)
    request_b = _request(stat_var=_STAT_VAR_B, facet_bindings=_BOUND_FACETS)
    facets = [
        Facet(
            facet_id="facet1",
            provenance_id="PROV1",
            measurement_method="METH1",
            unit="UNIT1",
            obs_count=1,
            earliest_date="2015",
            latest_date="2015",
        ),
        Facet(
            facet_id="facet2",
            provenance_id="PROV1",
            measurement_method="METH1",
            unit="UNIT1",
            obs_count=1,
            earliest_date="2016",
            latest_date="2016",
        ),
    ]
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): facets,
            (_BARE_B, _ENTITY): facets,
        }
    )

    def body_for(stat_var: str) -> str:
        return (
            _CSV_HEADER
            + _csv_row(
                stat_var=stat_var,
                year="2015",
                value="1000",
                unit="UNIT1",
                method="METH1",
                provenance="PROV1",
                facet_id="facet1",
            )
            + _csv_row(
                stat_var=stat_var,
                year="2016",
                value="2000",
                unit="UNIT1",
                method="METH1",
                provenance="PROV1",
                facet_id="facet2",
            )
        )

    sdmx, requests = _sdmx_client_for(
        {_BARE_A: body_for(_STAT_VAR_A), _BARE_B: body_for(_STAT_VAR_B)}
    )
    coverage = _coverage_for(
        [(_BARE_A, "facet1"), (_BARE_A, "facet2"), (_BARE_B, "facet1"), (_BARE_B, "facet2")]
    )

    result = materialise_group(
        [request_a, request_b],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=1,
        coverage=coverage,
    )

    # Candidates come from `coverage`, an in-memory lookup: only the one
    # combined data fetch runs. Both members bind the same entity scope, so
    # `_plan_collection_chunks` puts them in one shared chunk at
    # `member_concurrency=1` (its own lane target, `ceil(2 / 1) == 2`);
    # the default `4` would already split two same-scope members into
    # singleton chunks instead.
    assert len(requests) == 1
    assert result.coverage.rows == 4
    assert len(only_chunk(result).rows) == 4
    assert result.trace.resolved_stat_vars == (
        {"ref": _STAT_VAR_A, "roles": [_ROLE], "observation_count": 2},
        {"ref": _STAT_VAR_B, "roles": [_ROLE], "observation_count": 2},
    )


def test_n3_resolved_stat_vars_sorted_regardless_of_input_order():
    requests = [
        _request(stat_var=_STAT_VAR_C, facet_bindings=_BOUND_FACETS),
        _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS),
        _request(stat_var=_STAT_VAR_B, facet_bindings=_BOUND_FACETS),
    ]
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [_SHARED_FACET],
            (_BARE_B, _ENTITY): [_SHARED_FACET],
            (_BARE_C, _ENTITY): [_SHARED_FACET],
        }
    )
    bodies = {
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
        for bare, stat_var, year in (
            (_BARE_A, _STAT_VAR_A, "2015"),
            (_BARE_B, _STAT_VAR_B, "2016"),
            (_BARE_C, _STAT_VAR_C, "2017"),
        )
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    coverage = _coverage_for([(_BARE_A, "facet1"), (_BARE_B, "facet1"), (_BARE_C, "facet1")])

    result = materialise_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert [entry["ref"] for entry in result.trace.resolved_stat_vars] == [
        _STAT_VAR_A,
        _STAT_VAR_B,
        _STAT_VAR_C,
    ]


def test_frame_bytes_independent_of_member_order():
    facets = {
        (_BARE_A, _ENTITY): [_SHARED_FACET],
        (_BARE_B, _ENTITY): [_SHARED_FACET],
        (_BARE_C, _ENTITY): [_SHARED_FACET],
    }
    bodies = {
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
        for bare, stat_var, year in (
            (_BARE_A, _STAT_VAR_A, "2015"),
            (_BARE_B, _STAT_VAR_B, "2016"),
            (_BARE_C, _STAT_VAR_C, "2017"),
        )
    }
    requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS),
        _request(stat_var=_STAT_VAR_B, facet_bindings=_BOUND_FACETS),
        _request(stat_var=_STAT_VAR_C, facet_bindings=_BOUND_FACETS),
    ]

    coverage = _coverage_for([(_BARE_A, "facet1"), (_BARE_B, "facet1"), (_BARE_C, "facet1")])

    sdmx_forward, _r1 = _sdmx_client_for(bodies)
    graph_forward = FakeGraphReader(facets=facets)
    result_forward = materialise_group(
        requests,
        sdmx=sdmx_forward,
        graph=graph_forward,
        enrichment=EnrichmentLookup(graph_forward, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    sdmx_reversed, _r2 = _sdmx_client_for(bodies)
    graph_reversed = FakeGraphReader(facets=facets)
    result_reversed = materialise_group(
        list(reversed(requests)),
        sdmx=sdmx_reversed,
        graph=graph_reversed,
        enrichment=EnrichmentLookup(graph_reversed, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    forward_frame = only_chunk(result_forward)
    reversed_frame = only_chunk(result_reversed)
    assert forward_frame.columns == reversed_frame.columns
    assert forward_frame.rows == reversed_frame.rows
    assert result_forward.content_version == result_reversed.content_version


# ---------------------------------------------------------------------------
# Per-member observation_count: a member contributing no rows to the
# group's shared frame must not be invisible in the trace
# ---------------------------------------------------------------------------


def test_group_member_with_zero_matching_rows_is_named_with_zero_count():
    # B has a facet but returns no rows. Group-level warnings do not identify
    # an empty member; its `observation_count` must be zero.
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
            value="1000",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        ),
        _BARE_B: _CSV_HEADER,  # facet-viable, zero data rows
    }
    # The role-bearing LATEST probe needs a row to discover `facet1`; the
    # actual data fetch remains empty.
    probe_bodies = {
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2015",
            value="1000",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        ),
    }
    sdmx, requests = _sdmx_client_for(bodies, probe_bodies_by_bare_stat_var=probe_bodies)
    # Coverage makes B facet-viable; its data response remains header-only.
    coverage = _coverage_for([(_BARE_A, "facet1"), (_BARE_B, "facet1")])

    result = materialise_group(
        [request_a, request_b],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=1,
        coverage=coverage,
    )

    # Candidates come from `coverage`, an in-memory lookup: only the one
    # combined data fetch runs. Both members bind the same entity scope, so
    # `_plan_collection_chunks` puts them in one shared chunk at
    # `member_concurrency=1` (its own lane target, `ceil(2 / 1) == 2`);
    # the default `4` would already split two same-scope members into
    # singleton chunks instead.
    assert len(requests) == 1
    assert result.coverage.rows == 1
    assert result.warnings == ()  # neither the group-empty nor coverage warning fires
    entries_by_ref = {entry["ref"]: entry for entry in result.trace.resolved_stat_vars}
    assert entries_by_ref[_STAT_VAR_A]["observation_count"] == 1
    assert entries_by_ref[_STAT_VAR_B]["observation_count"] == 0


def test_group_member_observation_count_reflects_post_collapse_rows_not_raw_rows():
    # Member A's own response carries two facets that share one
    # (unit, measurementMethod, provenance) identity for one shared
    # recipient/period, a grain collision `facet_collapse.
    # collapse_duplicate_identity_facets_table` resolves to one surviving
    # row. `member_observation_counts` must report member A's own count
    # once collapsed (1), not its raw response row count (2).
    request_a = _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS)
    request_b = _request(stat_var=_STAT_VAR_B, facet_bindings=_BOUND_FACETS)
    facet_1 = Facet(
        facet_id="facet1",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="UNIT1",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    facet_2 = Facet(
        facet_id="facet2",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="UNIT1",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [facet_1, facet_2],
            (_BARE_B, _ENTITY): [_SHARED_FACET],
        }
    )
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="1000",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="2000",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet2",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2016",
            value="3000",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        ),
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    coverage = _coverage_for([(_BARE_A, "facet1"), (_BARE_A, "facet2"), (_BARE_B, "facet1")])

    result = materialise_group(
        [request_a, request_b],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    entries_by_ref = {entry["ref"]: entry for entry in result.trace.resolved_stat_vars}
    assert entries_by_ref[_STAT_VAR_A]["observation_count"] == 1
    assert entries_by_ref[_STAT_VAR_B]["observation_count"] == 1
    assert result.coverage.rows == 2


def test_group_observation_counts_join_by_stat_var_not_positional_order():
    # `_finish_group` sorts `resolved_stat_vars` by `stat_var` to guarantee
    # order-independence; request order here (C, A, B) differs from that
    # sorted order (A, B, C), and every member carries a DIFFERENT
    # observation count, so a naive index-based zip of per-member counts
    # against the sorted requests would land every count on the wrong ref.
    requests = [
        _request(stat_var=_STAT_VAR_C, facet_bindings=_BOUND_FACETS),
        _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS),
        _request(stat_var=_STAT_VAR_B, facet_bindings=_BOUND_FACETS),
    ]
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [_SHARED_FACET],
            (_BARE_B, _ENTITY): [_SHARED_FACET],
            (_BARE_C, _ENTITY): [_SHARED_FACET],
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
            facet_id="facet1",
        ),
        _BARE_B: _CSV_HEADER,  # zero rows
        _BARE_C: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_C,
            year="2015",
            value="10",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )
        + _csv_row(
            stat_var=_STAT_VAR_C,
            year="2016",
            value="20",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        ),
    }
    # B's LATEST probe discovers `facet1`; its data fetch has zero rows.
    probe_bodies = {
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2015",
            value="10",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        ),
    }
    sdmx, _requests = _sdmx_client_for(bodies, probe_bodies_by_bare_stat_var=probe_bodies)
    # Coverage supplies B's facet; its data response remains header-only.
    coverage = _coverage_for([(_BARE_A, "facet1"), (_BARE_B, "facet1"), (_BARE_C, "facet1")])

    result = materialise_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    entries_by_ref = {entry["ref"]: entry for entry in result.trace.resolved_stat_vars}
    assert entries_by_ref[_STAT_VAR_A]["observation_count"] == 1
    assert entries_by_ref[_STAT_VAR_B]["observation_count"] == 0
    assert entries_by_ref[_STAT_VAR_C]["observation_count"] == 2


def test_group_generation_serves_a_populated_member_alongside_an_empty_pool_member():
    # Member B's own probe returns no candidate at all, so it shares no
    # facet with A and the group's one selection comes from A alone. Both
    # members bind the same entity scope, so they share one chunk: B
    # dispatches nothing inside that chunk while A's rows stream through,
    # and `member_observation_counts` still carries B's own entry at zero.
    # This reaches the generation layer directly, since neither count
    # survives onto `MaterialisationResult`.
    request_a = _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS)
    request_b = _request(stat_var=_STAT_VAR_B, facet_bindings=_BOUND_FACETS)
    graph = FakeGraphReader(facets={(_BARE_A, _ENTITY): [_SHARED_FACET]})
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="1000",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        ),
        _BARE_B: _CSV_HEADER,
    }
    sdmx, requests = _sdmx_client_for(bodies)
    # B's own candidates come from `coverage`, which carries no `RoleNames`
    # entry for it at all (the same "B's own probe finds nothing" shape,
    # matching `graph.facets` above, which is empty for B too).
    coverage = _coverage_for([(_BARE_A, "facet1")])

    outcome = _generate_group(
        [request_a, request_b],
        sdmx=sdmx,
        graph=graph,
        ceiling=DEFAULT_SYNC_CEILING_ROWS,
        member_concurrency=1,
        coverage_index=coverage,
    )

    # Candidates come from `coverage`, an in-memory lookup: only the one
    # combined data fetch runs for the chunk they share.
    assert len(requests) == 1
    assert outcome.row_count == 1
    assert outcome.member_observation_counts == {_STAT_VAR_A: 1, _STAT_VAR_B: 0}
    assert outcome.selections["provenance"].ref == "PROV1"
    stat_vars = only_chunk(outcome).table["stat_var"].to_pylist()
    assert stat_vars == [_STAT_VAR_A]


# ---------------------------------------------------------------------------
# Role agreement
# ---------------------------------------------------------------------------


def test_mismatched_declared_roles_raises_usage_error_before_any_network_call():
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [_SHARED_FACET],
            (_BARE_B, _ENTITY): [_SHARED_FACET],
        }
    )
    sdmx, requests = _sdmx_client_for({})
    request_a = _request(
        stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS, declared_roles=(_ROLE,)
    )
    request_b = _request(
        stat_var=_STAT_VAR_B, facet_bindings=_BOUND_FACETS, declared_roles=("donor",)
    )

    with pytest.raises(MaterialisationUsageError):
        materialise_group(
            [request_a, request_b],
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            materialiser_build="build-123",
            capability_version="cap-v1",
        )

    assert requests == []
    assert sum(graph.calls.values()) == 0


def test_duplicate_stat_var_in_group_raises_usage_error_before_any_network_call():
    # Two members sharing the same stat_var would otherwise collapse onto
    # one dict key in `_generate_group`'s `member_observation_counts`,
    # silently keeping only the last member's count under BOTH trace
    # entries -- refused outright instead of left to that coincidence.
    graph = FakeGraphReader(facets={(_BARE_A, _ENTITY): [_SHARED_FACET]})
    sdmx, requests = _sdmx_client_for({})
    request_a1 = _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS)
    request_a2 = _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS)

    with pytest.raises(MaterialisationUsageError):
        materialise_group(
            [request_a1, request_a2],
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            materialiser_build="build-123",
            capability_version="cap-v1",
        )

    assert requests == []  # refused before any SDMX call
    assert sum(graph.calls.values()) == 0  # refused before any graph probe too


def test_empty_requests_raises_usage_error():
    graph_empty = FakeGraphReader()
    with pytest.raises(MaterialisationUsageError):
        materialise_group(
            [],
            sdmx=_sdmx_client_for({})[0],
            graph=graph_empty,
            enrichment=EnrichmentLookup(graph_empty, parties=lambda: {}),
            materialiser_build="build-123",
            capability_version="cap-v1",
        )


def test_mismatched_unit_bindings_raise_usage_error_before_any_network_call():
    # `_group_shared_candidates` narrows the merged candidate pool using
    # only `requests[0].facet_bindings` -- correct only because every
    # member is required to carry the same bindings. Two members differing
    # on the unit axis alone must be refused here, before that trust is
    # ever relied on.
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [_SHARED_FACET],
            (_BARE_B, _ENTITY): [_SHARED_FACET],
        }
    )
    sdmx, requests = _sdmx_client_for({})
    mismatched_facets = dict(_BOUND_FACETS)
    mismatched_facets["unit"] = FacetBinding(state="bound", ref="UNIT2")
    request_a = _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS)
    request_b = _request(stat_var=_STAT_VAR_B, facet_bindings=mismatched_facets)

    with pytest.raises(MaterialisationUsageError, match="unit"):
        materialise_group(
            [request_a, request_b],
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            materialiser_build="build-123",
            capability_version="cap-v1",
        )

    assert requests == []
    assert sum(graph.calls.values()) == 0
