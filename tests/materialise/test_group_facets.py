"""`materialise_group`: group-level facet selection, the facet serving the most members
ranked by summed obs_count, members it leaves unserved named, honouring a graph-declared
default policy agreed across members, the per-member bound-axis refusal,
an axis absent from the chosen facet (null on `measurementMethod`, refused
via `DefaultFacetUndefinedError` on `provenance`), and members that bind
no entity minting every series coverage holds.
"""

from __future__ import annotations

import dataclasses
import time
from typing import NamedTuple

import pytest

from materialise import EnrichmentLookup, materialise_group
from materialise.errors import (
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
)
from materialise.facets import (
    DEPLOYMENT_FACET_POLICY,
    CoverageAdmission,
    FacetBinding,
    joint_dispatch_facet_ids,
)
from materialise.generation import (
    _generate_chunk,
    _MemberFetchPlan,
    _plan_collection_chunks,
    _RawRowBudget,
)
from materialise.graph import Facet, FacetDefaultPolicy
from materialise.group_facets import _group_pool, elect_group, select_group_facets
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._group_support import (
    _BARE_A,
    _BARE_B,
    _CSV_HEADER,
    _ENTITY,
    _STAT_VAR_A,
    _STAT_VAR_B,
    _STAT_VAR_C,
    _csv_row,
    _request,
    _sdmx_client_for,
)
from .conftest import FakeGraphReader


class _GroupFacet(NamedTuple):
    """One coverage row's own facet identity, for `_coverage_for` below --
    `stat_var` the bare form, matching a test's own `_BARE_A`/`_BARE_B`."""

    stat_var: str
    facet_id: str
    provenance: str | None
    method: str | None
    unit: str | None
    obs_count: int = 1


def _coverage_for(*entries: _GroupFacet, entity: str = _ENTITY, year: int = 1973) -> CoverageIndex:
    """Build coverage from `_GroupFacet` entries with `entity` as recipient.
    Unlisted StatVars return empty candidate lists. Donor is an unbound
    placeholder.
    """
    bare_stat_vars = {entry.stat_var for entry in entries}
    role_names = {bare: RoleNames(donor="donor", recipient="recipient") for bare in bare_stat_vars}
    facet_metadata = {
        # Every DFO import states `P1Y`; a coverage facet stating no period is a cadence of
        # its own (`materialise.facets.cadences_by_member`).
        entry.facet_id: FacetMetadata(
            provenance_id=entry.provenance,
            measurement_method=entry.method,
            unit=entry.unit,
            observation_period="P1Y",
        )
        for entry in entries
    }
    rows = [
        (
            entry.stat_var,
            entry.facet_id,
            "country/DONOR",
            entity,
            list(range(year, year + entry.obs_count)),
        )
        for entry in entries
    ]
    return build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)


def test_group_selection_ranks_by_summed_obs_count_not_a_single_members_local_max():
    # `facetShared` sums to 7 (3 + 4). `facetLocalMax` sums to 6 (5 + 1).
    # `facetPrivateA` belongs only to A, so the group cannot select it.
    facet_shared = Facet(
        facet_id="facetShared",
        provenance_id="PROV_SHARED",
        measurement_method="METH_SHARED",
        unit="UNIT_SHARED",
        obs_count=3,
        earliest_date="2015",
        latest_date="2015",
    )
    facet_local_max = Facet(
        facet_id="facetLocalMax",
        provenance_id="PROV_LOCAL",
        measurement_method="METH_LOCAL",
        unit="UNIT_LOCAL",
        obs_count=5,
        earliest_date="2015",
        latest_date="2015",
    )
    facet_private_a = Facet(
        facet_id="facetPrivateA",
        provenance_id="PROV_PRIVATE",
        measurement_method="METH_PRIVATE",
        unit="UNIT_PRIVATE",
        obs_count=100,
        earliest_date="2015",
        latest_date="2015",
    )
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [facet_shared, facet_local_max, facet_private_a],
            (_BARE_B, _ENTITY): [
                Facet(
                    facet_id="facetShared",
                    provenance_id="PROV_SHARED",
                    measurement_method="METH_SHARED",
                    unit="UNIT_SHARED",
                    obs_count=4,
                    earliest_date="2016",
                    latest_date="2016",
                ),
                Facet(
                    facet_id="facetLocalMax",
                    provenance_id="PROV_LOCAL",
                    measurement_method="METH_LOCAL",
                    unit="UNIT_LOCAL",
                    obs_count=1,
                    earliest_date="2016",
                    latest_date="2016",
                ),
            ],
        }
    )
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="10",
            unit="UNIT_SHARED",
            method="METH_SHARED",
            provenance="PROV_SHARED",
            facet_id="facetShared",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2016",
            value="20",
            unit="UNIT_SHARED",
            method="METH_SHARED",
            provenance="PROV_SHARED",
            facet_id="facetShared",
        ),
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    deadlines: list[float | None] = []
    original_stream_data_batches = sdmx.stream_data_batches

    def recording_stream_data_batches(**kwargs):
        deadlines.append(kwargs.get("deadline"))
        return original_stream_data_batches(**kwargs)

    sdmx.stream_data_batches = recording_stream_data_batches  # type: ignore[method-assign]
    deadline = time.monotonic() + 30
    coverage = _coverage_for(
        _GroupFacet(_BARE_A, "facetShared", "PROV_SHARED", "METH_SHARED", "UNIT_SHARED", 3),
        _GroupFacet(_BARE_A, "facetLocalMax", "PROV_LOCAL", "METH_LOCAL", "UNIT_LOCAL", 5),
        _GroupFacet(_BARE_A, "facetPrivateA", "PROV_PRIVATE", "METH_PRIVATE", "UNIT_PRIVATE", 60),
        _GroupFacet(_BARE_B, "facetShared", "PROV_SHARED", "METH_SHARED", "UNIT_SHARED", 4),
        _GroupFacet(_BARE_B, "facetLocalMax", "PROV_LOCAL", "METH_LOCAL", "UNIT_LOCAL", 1),
    )
    requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings={}),
        _request(stat_var=_STAT_VAR_B, facet_bindings={}),
    ]

    result = materialise_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        deadline=deadline,
        member_concurrency=1,
        coverage=coverage,
    )

    facets_by_axis = {entry["axis"]: entry for entry in result.trace.facets}
    assert facets_by_axis["provenance"]["ref"] == "PROV_SHARED"
    assert facets_by_axis["provenance"]["dispatch_facet_id"] == "facetShared"
    assert result.coverage.rows == 2
    # Both members bind the same entity scope and window kind, and
    # `member_concurrency=1` above (`_plan_collection_chunks`'s own lane
    # target is `ceil(2 / 1) == 2`. The default `4` would split them
    # into singletons) puts them in one shared chunk. One combined
    # One `stream_data_batches` call serves both members.
    assert deadlines == [deadline]
    # Neither member requested a unit conversion, so `_finish_group`'s
    # no-conversion branch fires: no group-level record, and both members'
    # own entries stay `None`.
    assert result.conversion_record is None
    assert result.member_conversion_records == {_STAT_VAR_A: None, _STAT_VAR_B: None}


def test_group_honours_graph_declared_policy_agreed_across_members():
    # Both members declare `facetPolicy` (count 1). `facetBig` has 100.
    # The policy selects provenance, while method and unit select `facetBig`.
    # Both facets share those axes, so the selection remains jointly valid.
    facet_policy = Facet(
        facet_id="facetPolicy",
        provenance_id="PROV_POLICY",
        measurement_method="METH_SHARED",
        unit="UNIT_SHARED",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    facet_big = Facet(
        facet_id="facetBig",
        provenance_id="PROV_BIG",
        measurement_method="METH_SHARED",
        unit="UNIT_SHARED",
        obs_count=100,
        earliest_date="2016",
        latest_date="2016",
    )
    policy = FacetDefaultPolicy(
        resolved_ref="PROV_POLICY",
        policy_node="dcid:GraphDeclaredProvenancePolicy",
        policy_version="graph-v1",
        selection_reason="graph_declared",
    )
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [facet_policy, facet_big],
            (_BARE_B, _ENTITY): [facet_policy, facet_big],
        },
        policies={
            (_STAT_VAR_A, "provenance"): policy,
            (_STAT_VAR_B, "provenance"): policy,
        },
    )
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="10",
            unit="UNIT_SHARED",
            method="METH_SHARED",
            provenance="PROV_POLICY",
            facet_id="facetPolicy",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2016",
            value="20",
            unit="UNIT_SHARED",
            method="METH_SHARED",
            provenance="PROV_POLICY",
            facet_id="facetPolicy",
        ),
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    coverage = _coverage_for(
        _GroupFacet(_BARE_A, "facetPolicy", "PROV_POLICY", "METH_SHARED", "UNIT_SHARED", 1),
        _GroupFacet(_BARE_A, "facetBig", "PROV_BIG", "METH_SHARED", "UNIT_SHARED", 60),
        _GroupFacet(_BARE_B, "facetPolicy", "PROV_POLICY", "METH_SHARED", "UNIT_SHARED", 1),
        _GroupFacet(_BARE_B, "facetBig", "PROV_BIG", "METH_SHARED", "UNIT_SHARED", 60),
    )
    requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings={}),
        _request(stat_var=_STAT_VAR_B, facet_bindings={}),
    ]

    result = materialise_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    facets_by_axis = {entry["axis"]: entry for entry in result.trace.facets}
    provenance = facets_by_axis["provenance"]
    assert provenance["ref"] == "PROV_POLICY"
    assert provenance["dispatch_facet_id"] == "facetPolicy"
    assert provenance["basis"] == "declared_default"
    assert provenance["default_policy"] == "dcid:GraphDeclaredProvenancePolicy"
    assert provenance["default_policy_version"] == "graph-v1"
    assert provenance["selection_reason"] == "graph_declared"
    assert provenance["default_policy"] != DEPLOYMENT_FACET_POLICY

    method = facets_by_axis["measurementMethod"]
    assert method["ref"] == "METH_SHARED"
    assert method["basis"] == "deployment_fallback"
    assert method["default_policy"] == DEPLOYMENT_FACET_POLICY
    assert result.coverage.rows == 2


def test_a_group_sharing_no_facet_serves_the_top_members_rows_and_names_the_other():
    graph = FakeGraphReader()
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="10",
            unit="UNIT_A",
            method="METH_A",
            provenance="PROV_A",
            facet_id="onlyA",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2015",
            value="10",
            unit="UNIT_B",
            method="METH_B",
            provenance="PROV_B",
            facet_id="onlyB",
        ),
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    coverage = _coverage_for(
        _GroupFacet(_BARE_A, "onlyA", "PROV_A", "METH_A", "UNIT_A"),
        _GroupFacet(_BARE_B, "onlyB", "PROV_B", "METH_B", "UNIT_B"),
    )
    group_requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings={}),
        _request(stat_var=_STAT_VAR_B, facet_bindings={}),
    ]

    result = materialise_group(
        group_requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    # The facets tie, and the ranking's provenance order elects `PROV_A`'s.
    facets_by_axis = {entry["axis"]: entry for entry in result.trace.facets}
    assert facets_by_axis["provenance"]["ref"] == "PROV_A"
    assert result.coverage.rows == 1
    counts = {entry["ref"]: entry["observation_count"] for entry in result.trace.resolved_stat_vars}
    assert counts == {_STAT_VAR_A: 1, _STAT_VAR_B: 0}
    (warning,) = [w for w in result.warnings if w["code"] == "coverage_below_scope"]
    assert warning["member_count"] == 2
    assert warning["unserved_members"] == {"count": 1, "refs": [_STAT_VAR_B]}


def test_member_missing_bound_axis_refuses_with_its_own_alternatives_and_pointer():
    graph = FakeGraphReader()
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="10",
            unit="UNIT1",
            method="METH1",
            provenance="REAL_PROV_A",
            facet_id="facetA",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2015",
            value="10",
            unit="UNIT1",
            method="METH1",
            provenance="REAL_PROV_B",
            facet_id="facetB",
        ),
    }
    sdmx, requests = _sdmx_client_for(bodies)
    coverage = _coverage_for(
        _GroupFacet(_BARE_A, "facetA", "REAL_PROV_A", "METH1", "UNIT1"),
        _GroupFacet(_BARE_B, "facetB", "REAL_PROV_B", "METH1", "UNIT1"),
    )
    request_a = _request(
        stat_var=_STAT_VAR_A,
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref="WRONG_PROV"),
            "measurementMethod": FacetBinding(state="bound", ref="METH1"),
            "unit": FacetBinding(state="bound", ref="UNIT1"),
        },
    )
    request_b = _request(
        stat_var=_STAT_VAR_B,
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref="REAL_PROV_B"),
            "measurementMethod": FacetBinding(state="bound", ref="METH1"),
            "unit": FacetBinding(state="bound", ref="UNIT1"),
        },
    )

    with pytest.raises(FacetNotAvailableError) as excinfo:
        materialise_group(
            [request_a, request_b],
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=coverage,
        )

    # Candidates come from `coverage`, an in-memory lookup: refused during
    # selection with no SDMX call at all.
    assert requests == []
    issues = excinfo.value.issues
    assert issues is not None
    assert len(issues) == 1
    issue = issues[0]
    assert issue.axis == "provenance"
    assert issue.member == _STAT_VAR_A
    assert issue.pointer == "/facets/provenance"
    alternative_refs = {alt.ref for alt in issue.alternatives or []}
    assert alternative_refs == {"REAL_PROV_A"}


def test_axis_absent_from_chosen_facet_is_none_in_trace_and_omitted_from_preimage():
    facet_no_method = Facet(
        facet_id="facetNoMethod",
        provenance_id="PROV1",
        measurement_method=None,
        unit="UNIT1",
        obs_count=5,
        earliest_date="2015",
        latest_date="2015",
    )
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [facet_no_method],
            (_BARE_B, _ENTITY): [
                Facet(
                    facet_id="facetNoMethod",
                    provenance_id="PROV1",
                    measurement_method=None,
                    unit="UNIT1",
                    obs_count=5,
                    earliest_date="2016",
                    latest_date="2016",
                )
            ],
        }
    )
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="10",
            unit="UNIT1",
            method="",
            provenance="PROV1",
            facet_id="facetNoMethod",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2016",
            value="20",
            unit="UNIT1",
            method="",
            provenance="PROV1",
            facet_id="facetNoMethod",
        ),
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    coverage = _coverage_for(
        _GroupFacet(_BARE_A, "facetNoMethod", "PROV1", None, "UNIT1", 5),
        _GroupFacet(_BARE_B, "facetNoMethod", "PROV1", None, "UNIT1", 5),
    )
    requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings={}),
        _request(stat_var=_STAT_VAR_B, facet_bindings={}),
    ]

    result = materialise_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    facets_by_axis = {entry["axis"]: entry for entry in result.trace.facets}
    assert facets_by_axis["measurementMethod"]["ref"] is None
    assert facets_by_axis["measurementMethod"]["basis"] == "deployment_fallback"
    assert facets_by_axis["measurementMethod"]["selection_reason"] == "absent_on_selected_facet"
    # The successful mint confirms the null measurement-method axis is
    # omitted from its preimage.
    assert result.content_version.startswith("cv_")
    assert result.coverage.rows == 2


def test_provenance_absent_from_chosen_facet_raises_default_facet_undefined():
    # Citation assembly requires a concrete provenance ref, so a null
    # provenance axis raises `DefaultFacetUndefinedError`.
    graph = FakeGraphReader()
    # The SDMX sentinel `NotApplicable` parses to `None` for provenance.
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="10",
            unit="UNIT1",
            method="METH1",
            provenance="NotApplicable",
            facet_id="facetNoProvenance",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2016",
            value="20",
            unit="UNIT1",
            method="METH1",
            provenance="NotApplicable",
            facet_id="facetNoProvenance",
        ),
    }
    sdmx, requests = _sdmx_client_for(bodies)
    coverage = _coverage_for(
        _GroupFacet(_BARE_A, "facetNoProvenance", None, "METH1", "UNIT1"),
        _GroupFacet(_BARE_B, "facetNoProvenance", None, "METH1", "UNIT1"),
    )
    group_requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings={}),
        _request(stat_var=_STAT_VAR_B, facet_bindings={}),
    ]

    with pytest.raises(DefaultFacetUndefinedError):
        materialise_group(
            group_requests,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=coverage,
        )

    # Candidates come from `coverage`, an in-memory lookup: refused during
    # selection with no SDMX call at all.
    assert requests == []


def _unbound_group_fixture():
    """Members A and B share one facet identity in coverage, with one row each."""
    bodies = {
        bare: _CSV_HEADER
        + _csv_row(
            stat_var=stat_var,
            year="2015",
            value="10",
            unit="UNIT_S",
            method="METH_S",
            provenance="PROV_S",
            facet_id="facetS",
        )
        for bare, stat_var in ((_BARE_A, _STAT_VAR_A), (_BARE_B, _STAT_VAR_B))
    }
    sdmx, requests = _sdmx_client_for(bodies)
    coverage = _coverage_for(
        _GroupFacet(_BARE_A, "facetS", "PROV_S", "METH_S", "UNIT_S"),
        _GroupFacet(_BARE_B, "facetS", "PROV_S", "METH_S", "UNIT_S"),
        year=2015,
    )
    return sdmx, requests, coverage


def _entity_filters(request) -> dict[str, str]:
    """A fetch's role filters, keyed by role."""
    return {
        key: value
        for key, value in request.url.params.items()
        if key in ("c[donor]", "c[recipient]")
    }


def test_one_member_with_no_bound_entity_mints_every_covered_series_beside_a_bound_member():
    graph = FakeGraphReader()
    sdmx, requests, coverage = _unbound_group_fixture()
    request_a = dataclasses.replace(
        _request(stat_var=_STAT_VAR_A, facet_bindings={}), role_bindings={}
    )
    request_b = _request(stat_var=_STAT_VAR_B, facet_bindings={})

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
    by_member = {str(r.url.params["c[variableMeasured]"]): _entity_filters(r) for r in requests}
    # A's fetch lists every entity coverage holds on one role axis. B's names its own binding.
    assert by_member == {
        _BARE_A: {"c[donor]": "country/DONOR"},
        _BARE_B: {"c[recipient]": _ENTITY},
    }
    # Coverage answers the probe, so no graph facet read or place scope read happens.
    assert graph.calls["variable_facets_batch"] == 0
    assert graph.calls["observation_facets_batch_many"] == 0
    assert graph.calls["child_places_batch"] == 0


def test_two_members_with_no_bound_entity_share_one_exhaustive_fetch():
    graph = FakeGraphReader()
    sdmx, requests, coverage = _unbound_group_fixture()
    request_a = dataclasses.replace(
        _request(stat_var=_STAT_VAR_A, facet_bindings={}), role_bindings={}
    )
    request_b = dataclasses.replace(
        _request(stat_var=_STAT_VAR_B, facet_bindings={}), role_bindings={}
    )

    result = materialise_group(
        [request_a, request_b],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
        # Keeps both members in one shared chunk.
        member_concurrency=1,
    )

    assert result.coverage.rows == 2
    (request,) = requests
    assert sorted(str(request.url.params["c[variableMeasured]"]).split(",")) == [_BARE_A, _BARE_B]
    assert _entity_filters(request) == {"c[donor]": "country/DONOR"}


def test_members_with_no_bound_entity_and_no_coverage_are_the_empty_result():
    # Coverage holds no series for either member, so no probe returns a candidate and nothing
    # is read.
    graph = FakeGraphReader()
    sdmx, requests = _sdmx_client_for({})
    coverage = _coverage_for()
    request_a = dataclasses.replace(
        _request(stat_var=_STAT_VAR_A, facet_bindings={}), role_bindings={}
    )
    request_b = dataclasses.replace(
        _request(stat_var=_STAT_VAR_B, facet_bindings={}), role_bindings={}
    )

    result = materialise_group(
        [request_a, request_b],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert result.probe_empty
    assert result.coverage.rows == 0
    assert [warning["code"] for warning in result.warnings] == ["empty_result"]
    assert result.trace.facets == ()
    assert graph.calls["variable_facets_batch"] == 0
    assert graph.calls["observation_facets_batch_many"] == 0
    assert graph.calls["child_places_batch"] == 0
    assert requests == []


def test_role_bearing_members_with_no_bound_entity_mint_across_every_entity():
    # "Every recipient": coverage supplies both members' candidates, and the
    # fetch filters on every covered dcid of the smaller role axis.
    graph = FakeGraphReader()
    bodies = {
        bare: _CSV_HEADER
        + _csv_row(
            stat_var=stat_var,
            year="2015",
            value="10",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )
        for bare, stat_var in ((_BARE_A, _STAT_VAR_A), (_BARE_B, _STAT_VAR_B))
    }
    sdmx, requests = _sdmx_client_for(bodies)
    coverage = _coverage_for(
        _GroupFacet(_BARE_A, "facet1", "PROV1", "METH1", "UNIT1"),
        _GroupFacet(_BARE_B, "facet1", "PROV1", "METH1", "UNIT1"),
    )
    group_requests = [
        dataclasses.replace(_request(stat_var=stat_var, facet_bindings={}), role_bindings={})
        for stat_var in (_STAT_VAR_A, _STAT_VAR_B)
    ]

    result = materialise_group(
        group_requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert result.coverage.rows == 2
    assert requests
    for request in requests:
        assert request.url.params["c[donor]"] == "country/DONOR"
        assert "c[recipient]" not in request.url.params


def test_group_selection_narrows_the_shared_pool_to_the_requested_window():
    """The recorded facet wins on count unwindowed. A 2027 window reaches only the projection."""
    recorded = Facet(
        facet_id="facetRecorded",
        provenance_id="PROV_SHARED",
        measurement_method=None,
        unit="UNIT_SHARED",
        obs_count=55,
        earliest_date="1970",
        latest_date="2024",
    )
    estimated = dataclasses.replace(
        recorded,
        facet_id="facetEstimated",
        measurement_method="Estimated",
        obs_count=8,
        earliest_date="2025",
        latest_date="2032",
    )
    candidates = [[recorded, estimated], [recorded, estimated]]
    window_2027 = RequestedWindow(kind="range", start_year=2027, end_year=2027)

    unwindowed = select_group_facets(
        [
            _request(stat_var=_STAT_VAR_A, facet_bindings={}),
            _request(stat_var=_STAT_VAR_B, facet_bindings={}),
        ],
        candidates,
        graph=FakeGraphReader(),
    ).election.selections
    windowed = select_group_facets(
        [
            _request(stat_var=_STAT_VAR_A, facet_bindings={}, window=window_2027),
            _request(stat_var=_STAT_VAR_B, facet_bindings={}, window=window_2027),
        ],
        candidates,
        graph=FakeGraphReader(),
    ).election.selections

    assert unwindowed["measurementMethod"].ref is None
    assert unwindowed["provenance"].dispatch_facet_id == "facetRecorded"
    assert windowed["measurementMethod"].ref == "Estimated"
    assert windowed["provenance"].dispatch_facet_id == "facetEstimated"
    assert windowed["provenance"].selection_reason == "sole_candidate"


@pytest.mark.parametrize(
    ("window", "continued_by", "dispatched"),
    [
        # The window runs past the recorded series, so the group takes its estimates for
        # the later years, as a member's election does.
        pytest.param(
            RequestedWindow(kind="range", start_year=2016, end_year=2030),
            "Estimated",
            ("facetEstimated", "facetRecorded"),
            id="window-past-the-series-continues",
        ),
        pytest.param(
            RequestedWindow(kind="range", start_year=2016, end_year=2020),
            None,
            ("facetRecorded",),
            id="control-window-within-the-series",
        ),
    ],
)
def test_a_group_continues_its_series_with_the_same_sources_estimates(
    window, continued_by, dispatched
):
    recorded = Facet(
        facet_id="facetRecorded",
        provenance_id="PROV_SHARED",
        measurement_method=None,
        unit="UNIT_SHARED",
        obs_count=55,
        earliest_date="1970",
        latest_date="2024",
    )
    estimated = dataclasses.replace(
        recorded,
        facet_id="facetEstimated",
        measurement_method="Estimated",
        obs_count=8,
        earliest_date="2025",
        latest_date="2032",
    )
    candidates = [[recorded, estimated], [recorded, estimated]]

    election = select_group_facets(
        [
            _request(stat_var=_STAT_VAR_A, facet_bindings={}, window=window),
            _request(stat_var=_STAT_VAR_B, facet_bindings={}, window=window),
        ],
        candidates,
        graph=FakeGraphReader(),
    ).election

    assert election.selections["measurementMethod"].ref is None
    assert election.selections["measurementMethod"].continued_by == continued_by
    for member, member_candidates in enumerate(candidates):
        assert joint_dispatch_facet_ids(member_candidates, election, member=member) == dispatched


# --- Unit-axis `state="request"` group selection --------------------------

from materialise.facets import UnitRequest  # noqa: E402
from materialise.recipes import NonCurrencyUnitError  # noqa: E402

_GROUP_UNIT_REQUEST_USD_CURRENT = UnitRequest(currency="USD", prices="current")
_GROUP_REQUEST_BINDINGS = {
    "unit": FacetBinding(state="request", request=_GROUP_UNIT_REQUEST_USD_CURRENT)
}


def _group_requests(*, facet_bindings: dict[str, FacetBinding]) -> list:
    return [
        _request(stat_var=_STAT_VAR_A, facet_bindings=facet_bindings),
        _request(stat_var=_STAT_VAR_B, facet_bindings=facet_bindings),
    ]


def test_group_request_prefers_native_currency_candidate_over_a_higher_ranked_alternative():
    # facetEUR sums to 2000 across the two members, far outranking
    # facetUSD's 2, but the group's own currency preference for
    # {USD, current} wins over the ranking, mirroring the member-level rule.
    facet_usd = Facet(
        facet_id="facetUSD",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="CurrencyUSD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    facet_eur = Facet(
        facet_id="facetEUR",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="CurrencyEUR",
        obs_count=1000,
        earliest_date="2020",
        latest_date="2020",
    )
    requests = _group_requests(facet_bindings=_GROUP_REQUEST_BINDINGS)

    selections = select_group_facets(
        requests, [[facet_usd, facet_eur], [facet_usd, facet_eur]], graph=FakeGraphReader()
    ).election.selections

    assert selections["unit"].ref == "CurrencyUSD"
    assert selections["unit"].selection_reason == "native_currency_preferred"
    assert selections["unit"].request is _GROUP_UNIT_REQUEST_USD_CURRENT


def test_group_request_falls_to_ordinary_default_when_no_native_currency_candidate():
    facet_eur = Facet(
        facet_id="facetEUR",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="CurrencyEUR",
        obs_count=5,
        earliest_date="2020",
        latest_date="2020",
    )
    requests = _group_requests(facet_bindings=_GROUP_REQUEST_BINDINGS)

    selections = select_group_facets(
        requests, [[facet_eur], [facet_eur]], graph=FakeGraphReader()
    ).election.selections

    assert selections["unit"].ref == "CurrencyEUR"
    assert selections["unit"].selection_reason == "sole_candidate"
    assert selections["unit"].request is _GROUP_UNIT_REQUEST_USD_CURRENT


def test_group_request_resolved_to_a_non_currency_ref_raises_non_currency_unit_error():
    facet_bare_usd = Facet(
        facet_id="facetBareUSD",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="USD",
        obs_count=5,
        earliest_date="2020",
        latest_date="2020",
    )
    requests = _group_requests(facet_bindings=_GROUP_REQUEST_BINDINGS)

    with pytest.raises(NonCurrencyUnitError) as excinfo:
        select_group_facets(requests, [[facet_bare_usd], [facet_bare_usd]], graph=FakeGraphReader())

    assert excinfo.value.unit == "USD"


def test_group_request_declared_policy_agreed_across_members_wins_over_ranking():
    policy = FacetDefaultPolicy(
        resolved_ref="CurrencyGBP",
        policy_node="policy/unit-default",
        policy_version="v1",
        selection_reason="most-widely-reported currency",
    )
    graph = FakeGraphReader(policies={(_STAT_VAR_A, "unit"): policy, (_STAT_VAR_B, "unit"): policy})
    facet_gbp = Facet(
        facet_id="facetGBP",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="CurrencyGBP",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    requests = _group_requests(facet_bindings=_GROUP_REQUEST_BINDINGS)

    selections = select_group_facets(
        requests, [[facet_gbp], [facet_gbp]], graph=graph
    ).election.selections

    assert selections["unit"].basis == "declared_default"
    assert selections["unit"].ref == "CurrencyGBP"
    assert selections["unit"].request is _GROUP_UNIT_REQUEST_USD_CURRENT


def test_group_request_narrows_provenance_and_measurement_method_to_the_currency_preferred_pool():
    # facet_p2 sums to 2,000 observations vs facet_p1's 2. USD/current
    # narrows all axes to facet_p1. An unbound unit ranks facet_p2.
    facet_p1 = Facet(
        facet_id="f1",
        provenance_id="P1",
        measurement_method="M1",
        unit="CurrencyUSD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    facet_p2 = Facet(
        facet_id="f2",
        provenance_id="P2",
        measurement_method="M2",
        unit="CurrencyEUR",
        obs_count=1000,
        earliest_date="2015",
        latest_date="2015",
    )
    candidates = [[facet_p1, facet_p2], [facet_p1, facet_p2]]
    requests_with_request_unit = _group_requests(facet_bindings=_GROUP_REQUEST_BINDINGS)
    requests_with_unbound_unit = _group_requests(facet_bindings={})

    election_request = select_group_facets(
        requests_with_request_unit, candidates, graph=FakeGraphReader()
    ).election
    selections_request = election_request.selections
    selections_unbound = select_group_facets(
        requests_with_unbound_unit, candidates, graph=FakeGraphReader()
    ).election.selections

    # An unbound unit ranks the full, disagreeing pool: facet_p2 (the
    # higher-summed-obs_count facet) wins.
    assert selections_unbound["provenance"].ref == "P2"
    assert selections_unbound["measurementMethod"].ref == "M2"
    assert (
        selections_unbound["provenance"].selection_reason == "most_observations_over_disagreement"
    )
    # The group's choice names both values in rank order, each with its merged count.
    assert [
        (value.ref, value.observations) for value in selections_unbound["provenance"].ranked_values
    ] == [("P2", 2000), ("P1", 2)]

    # A currency-native unit request narrows the shared pool to the sole
    # USD-carrying facet first, so these axes follow that narrower pool to
    # facet_p1 instead.
    assert selections_request["provenance"].ref == "P1"
    assert selections_request["measurementMethod"].ref == "M1"
    assert selections_request["provenance"].selection_reason == "sole_candidate"
    # Every member has the elected `facet_p1` in its candidate list.
    for member, member_candidates in enumerate(candidates):
        assert joint_dispatch_facet_ids(member_candidates, election_request, member=member) == (
            "f1",
        )


# --- Group axis resolution's own axis rule, and dispatch-facet agreement --

from materialise.errors import MaterialisationUsageError  # noqa: E402


def test_group_axis_resolution_rejects_a_request_binding_for_an_axis_other_than_unit():
    # Request construction rejects this binding first, so the request-free election
    # exercises group-level axis validation directly.
    candidate = Facet(
        facet_id="facet1",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="UNIT1",
        obs_count=1,
        earliest_date="2015",
        latest_date="2017",
    )

    with pytest.raises(MaterialisationUsageError):
        elect_group(
            [[candidate]],
            stat_vars=[_STAT_VAR_A],
            window=None,
            index_sourced=False,
            requires_provenance=True,
            bindings={
                "provenance": FacetBinding(state="request", request=_GROUP_UNIT_REQUEST_USD_CURRENT)
            },
        )


def test_group_unit_request_binding_dispatches_from_facetusd_not_top():
    # facetEUR has more observations, but the USD/current preference selects
    # facetUSD for dispatch.
    facet_usd = Facet(
        facet_id="facetUSD",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="CurrencyUSD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    facet_eur = Facet(
        facet_id="facetEUR",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="CurrencyEUR",
        obs_count=1000,
        earliest_date="2020",
        latest_date="2020",
    )
    requests = _group_requests(facet_bindings=_GROUP_REQUEST_BINDINGS)
    candidates = [[facet_usd, facet_eur], [facet_usd, facet_eur]]

    selections = select_group_facets(
        requests, candidates, graph=FakeGraphReader()
    ).election.selections

    assert selections["unit"].dispatch_facet_id == "facetUSD"


def test_group_shares_a_facet_even_when_each_members_own_currency_native_candidate_differs():
    # Each member has a private USD facet plus shared EUR. Per-member USD
    # narrowing before intersection would empty the candidate set. The group
    # intersects shared facets first, then selects EUR and carries the USD
    # request forward for conversion.
    member1_usd = Facet(
        facet_id="member1_USD",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="CurrencyUSD",
        obs_count=5,
        earliest_date="2015",
        latest_date="2015",
    )
    member2_usd = Facet(
        facet_id="member2_USD",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="CurrencyUSD",
        obs_count=5,
        earliest_date="2015",
        latest_date="2015",
    )
    shared_eur = Facet(
        facet_id="shared_EUR",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="CurrencyEUR",
        obs_count=3,
        earliest_date="2018",
        latest_date="2018",
    )
    requests = _group_requests(facet_bindings=_GROUP_REQUEST_BINDINGS)
    candidates = [[member1_usd, shared_eur], [member2_usd, shared_eur]]

    selections = select_group_facets(
        requests, candidates, graph=FakeGraphReader()
    ).election.selections
    shared_candidates, _holders = _group_pool(
        candidates, bindings=requests[0].facet_bindings, index_sourced=False
    )
    (shared_facet,) = shared_candidates

    assert shared_facet.facet_id == "shared_EUR"
    assert shared_facet.obs_count == 6
    assert selections["unit"].ref == "CurrencyEUR"
    assert selections["unit"].selection_reason == "sole_candidate"
    assert selections["unit"].dispatch_facet_id == "shared_EUR"


def test_group_shared_candidates_raise_when_a_member_with_candidates_matches_none_jointly():
    # Both candidates are shared, but policies select P1 from f1 and EUR
    # from f2. No candidate carries both values, so each member is refused.
    facet_f1 = Facet(
        facet_id="f1",
        provenance_id="P1",
        measurement_method="M1",
        unit="CurrencyUSD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    facet_f2 = Facet(
        facet_id="f2",
        provenance_id="P2",
        measurement_method="M2",
        unit="CurrencyEUR",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    provenance_policy = FacetDefaultPolicy(
        resolved_ref="P1",
        policy_node="dcid:GraphDeclaredProvenancePolicy",
        policy_version="graph-v1",
        selection_reason="graph_declared",
    )
    unit_policy = FacetDefaultPolicy(
        resolved_ref="CurrencyEUR",
        policy_node="dcid:GraphDeclaredUnitPolicy",
        policy_version="graph-v1",
        selection_reason="graph_declared",
    )
    graph = FakeGraphReader(
        policies={
            (_STAT_VAR_A, "provenance"): provenance_policy,
            (_STAT_VAR_B, "provenance"): provenance_policy,
            (_STAT_VAR_A, "unit"): unit_policy,
            (_STAT_VAR_B, "unit"): unit_policy,
        }
    )
    requests = _group_requests(facet_bindings={})
    candidates = [[facet_f1, facet_f2], [facet_f1, facet_f2]]

    with pytest.raises(FacetNotAvailableError) as excinfo:
        select_group_facets(requests, candidates, graph=graph)

    issues = excinfo.value.issues
    assert issues is not None
    assert {issue.member for issue in issues} == {_STAT_VAR_A, _STAT_VAR_B}
    assert {issue.axis for issue in issues} == {"provenance", "measurementMethod", "unit"}


def test_a_group_whose_members_hold_the_currency_under_their_own_facet_ids_serves_both():
    # The members share no facet id. The first member's facet wins the tie, and the second
    # member's own facet carries the same source, method and currency, so it serves too.
    member1_usd = Facet(
        facet_id="member1_USD",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="CurrencyUSD",
        obs_count=5,
        earliest_date="2015",
        latest_date="2015",
    )
    member2_usd = Facet(
        facet_id="member2_USD",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="CurrencyUSD",
        obs_count=5,
        earliest_date="2015",
        latest_date="2015",
    )
    requests = _group_requests(facet_bindings=_GROUP_REQUEST_BINDINGS)

    group = select_group_facets(requests, [[member1_usd], [member2_usd]], graph=FakeGraphReader())

    assert group.election.selections["unit"].dispatch_facet_id == "member1_USD"
    assert group.dispatch_ids == (("member1_USD",), ("member2_USD",))
    assert group.unserved == ()


def test_index_sourced_group_agrees_on_provenance_and_unit_despite_differing_facet_ids_and_methods():
    # Index-sourced CRS members can have different facet ids and methods.
    # They share provenance and unit, the axes required for group agreement.
    facet_a = Facet(
        facet_id="facetA",
        provenance_id="PROV_CRS",
        measurement_method="METH_A",
        unit="UNIT_USD",
        obs_count=5,
        earliest_date="2020",
        latest_date="2020",
    )
    facet_b = Facet(
        facet_id="facetB",
        provenance_id="PROV_CRS",
        measurement_method="METH_B",
        unit="UNIT_USD",
        obs_count=3,
        earliest_date="2020",
        latest_date="2020",
    )
    requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings={}, index_sourced=True),
        _request(stat_var=_STAT_VAR_B, facet_bindings={}, index_sourced=True),
    ]
    candidates = [[facet_a], [facet_b]]
    graph = FakeGraphReader()

    election = select_group_facets(requests, candidates, graph=graph).election
    selections = election.selections

    assert set(selections) == {"provenance", "unit"}
    assert selections["provenance"].ref == "PROV_CRS"
    assert selections["unit"].ref == "UNIT_USD"
    # `facetA` outranks `facetB` on summed `obs_count` (5 vs. 3), so it wins
    # the group's own representative pick.
    assert selections["provenance"].dispatch_facet_id == "facetA"
    # Each member dispatches from its own candidate, despite differing methods.
    assert joint_dispatch_facet_ids([facet_a], election, member=0) == ("facetA",)
    assert joint_dispatch_facet_ids([facet_b], election, member=1) == ("facetB",)


def test_index_sourced_group_with_differing_unit_serves_the_broader_unit_and_names_the_other():
    # The members disagree on unit, so no `(provenance, unit)` serves both: the broader
    # facet serves its member and the other member is named unserved.
    facet_a = Facet(
        facet_id="facetA",
        provenance_id="PROV_CRS",
        measurement_method="METH_A",
        unit="UNIT_USD",
        obs_count=5,
        earliest_date="2020",
        latest_date="2020",
    )
    facet_b = Facet(
        facet_id="facetB",
        provenance_id="PROV_CRS",
        measurement_method="METH_B",
        unit="UNIT_EUR",
        obs_count=3,
        earliest_date="2020",
        latest_date="2020",
    )
    requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings={}, index_sourced=True),
        _request(stat_var=_STAT_VAR_B, facet_bindings={}, index_sourced=True),
    ]

    group = select_group_facets(requests, [[facet_a], [facet_b]], graph=FakeGraphReader())

    assert group.election.selections["unit"].ref == "UNIT_USD"
    assert group.dispatch_ids == (("facetA",), ())
    assert group.unserved == (1,)


def _facet_of(facet_id: str, *, obs_count: int, provenance: str = "PROV") -> Facet:
    """A facet of one import, method and unit apart from its id and its size."""
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        measurement_method=f"METH_{facet_id}",
        unit="UNIT_USD",
        obs_count=obs_count,
        earliest_date="2020",
        latest_date="2020",
    )


@pytest.mark.parametrize(
    ("candidates", "bindings", "breadth", "dispatched", "unserved"),
    [
        # Non-index members share provenance and unit but no facet id: the broader facet
        # serves its member and the other is named.
        pytest.param(
            [[_facet_of("facetA", obs_count=5)], [_facet_of("facetB", obs_count=3)]],
            {},
            None,
            (("facetA",), ()),
            (1,),
            id="no-shared-facet-the-broader-serves",
        ),
        # A facet every member holds wins over a broader facet one member holds alone.
        pytest.param(
            [
                [_facet_of("facetA", obs_count=50), _facet_of("facetShared", obs_count=3)],
                [_facet_of("facetShared", obs_count=3)],
            ],
            {},
            None,
            (("facetShared",), ("facetShared",)),
            (),
            id="a-shared-facet-serves-every-member",
        ),
        # Among facets serving as many members, the larger summed breadth wins over the
        # larger count.
        pytest.param(
            [
                [_facet_of("facetA", obs_count=9)],
                [_facet_of("facetA", obs_count=9)],
                [_facet_of("facetB", obs_count=1)],
                [_facet_of("facetB", obs_count=1)],
            ],
            {},
            {"facetA": 2, "facetB": 6},
            ((), (), ("facetB",), ("facetB",)),
            (0, 1),
            id="a-tie-on-members-goes-to-summed-breadth",
        ),
        # A member holding no candidate serves nothing and is not named.
        pytest.param(
            [[_facet_of("facetA", obs_count=5)], []],
            {},
            None,
            (("facetA",), ()),
            (),
            id="an-unpopulated-member-is-not-named",
        ),
        # A member holding each bound ref, but on no one candidate, serves nothing and is
        # named.
        pytest.param(
            [
                [_facet_of("facetA", obs_count=5)],
                [
                    _facet_of("facetB", obs_count=5),
                    dataclasses.replace(
                        _facet_of("facetA", obs_count=5, provenance="OTHER"), facet_id="facetC"
                    ),
                ],
            ],
            {
                "provenance": FacetBinding(state="bound", ref="PROV"),
                "measurementMethod": FacetBinding(state="bound", ref="METH_facetA"),
            },
            None,
            (("facetA",), ()),
            (1,),
            id="a-member-no-joint-bound-combination-covers-is-named",
        ),
    ],
)
def test_a_group_elects_the_facet_serving_the_most_members(
    candidates, bindings, breadth, dispatched, unserved
):
    requests = [
        _request(stat_var=stat_var, facet_bindings=bindings)
        for stat_var in (_STAT_VAR_A, _STAT_VAR_B, _STAT_VAR_C, "dcid:Stat_D")[: len(candidates)]
    ]

    group = select_group_facets(requests, candidates, graph=FakeGraphReader(), breadth=breadth)

    assert group.dispatch_ids == dispatched
    assert group.unserved == unserved


# --- A member whose bound entities publish nothing reaches zero rows without refusing the group ---


def test_group_shared_candidates_ignores_a_member_with_an_empty_pool():
    # Member B's bound entities publish no observation for its StatVar, so
    # its probed pool is empty. Its `{provenance: PROV_SHARED, unit:
    # UNIT_SHARED}` binding has nothing to match against, and an empty pool
    # contributes no id set to the group's intersection. Member A carries
    # `facetShared`, which serves the whole group, so the group selects it
    # and B reaches zero rows.
    shared = Facet(
        facet_id="facetShared",
        provenance_id="PROV_SHARED",
        measurement_method="METH_SHARED",
        unit="UNIT_SHARED",
        obs_count=3,
        earliest_date="2015",
        latest_date="2015",
    )
    requests = [
        _request(
            stat_var=_STAT_VAR_A,
            facet_bindings={
                "provenance": FacetBinding(state="bound", ref="PROV_SHARED"),
                "unit": FacetBinding(state="bound", ref="UNIT_SHARED"),
            },
        ),
        _request(
            stat_var=_STAT_VAR_B,
            facet_bindings={
                "provenance": FacetBinding(state="bound", ref="PROV_SHARED"),
                "unit": FacetBinding(state="bound", ref="UNIT_SHARED"),
            },
        ),
    ]
    candidates = [[shared], []]

    selections = select_group_facets(
        requests, candidates, graph=FakeGraphReader()
    ).election.selections

    assert selections["provenance"].ref == "PROV_SHARED"
    assert selections["provenance"].dispatch_facet_id == "facetShared"


def test_index_sourced_group_materialises_both_members_rows_despite_differing_measurement_method():
    # The methods differ. Shared selection must preserve both rows.
    # Dropping either row would reduce coverage from 2 to 1.
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [
                Facet(
                    facet_id="facetA",
                    provenance_id="PROV_CRS",
                    measurement_method="METH_A",
                    unit="UNIT_USD",
                    obs_count=5,
                    earliest_date="2020",
                    latest_date="2020",
                )
            ],
            (_BARE_B, _ENTITY): [
                Facet(
                    facet_id="facetB",
                    provenance_id="PROV_CRS",
                    measurement_method="METH_B",
                    unit="UNIT_USD",
                    obs_count=3,
                    earliest_date="2020",
                    latest_date="2020",
                )
            ],
        }
    )
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2020",
            value="10",
            unit="UNIT_USD",
            method="METH_A",
            provenance="PROV_CRS",
            facet_id="facetA",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2020",
            value="20",
            unit="UNIT_USD",
            method="METH_B",
            provenance="PROV_CRS",
            facet_id="facetB",
        ),
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    coverage = _coverage_for(
        _GroupFacet(_BARE_A, "facetA", "PROV_CRS", "METH_A", "UNIT_USD", 5),
        _GroupFacet(_BARE_B, "facetB", "PROV_CRS", "METH_B", "UNIT_USD", 3),
    )
    requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings={}, index_sourced=True),
        _request(stat_var=_STAT_VAR_B, facet_bindings={}, index_sourced=True),
    ]

    result = materialise_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert result.coverage.rows == 2
    facets_by_axis = {entry["axis"]: entry for entry in result.trace.facets}
    assert set(facets_by_axis) == {"provenance", "unit"}
    assert facets_by_axis["provenance"]["ref"] == "PROV_CRS"
    assert facets_by_axis["unit"]["ref"] == "UNIT_USD"


def test_group_materialise_with_one_empty_and_one_populated_member_serves_the_populated_one():
    # The end-to-end shape. Both members bind the same entity and the same
    # facet refs, but B's LATEST probe returns no row naming `_STAT_VAR_B`,
    # so B discovers no facet while A discovers `facetShared`. The group
    # materialises one row from A and reports B at zero.
    request_a = _request(
        stat_var=_STAT_VAR_A,
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref="PROV_SHARED"),
            "unit": FacetBinding(state="bound", ref="UNIT_SHARED"),
        },
    )
    request_b = _request(
        stat_var=_STAT_VAR_B,
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref="PROV_SHARED"),
            "unit": FacetBinding(state="bound", ref="UNIT_SHARED"),
        },
    )
    graph = FakeGraphReader()
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="10",
            unit="UNIT_SHARED",
            method="METH_SHARED",
            provenance="PROV_SHARED",
            facet_id="facetShared",
        ),
        _BARE_B: _CSV_HEADER,
    }
    sdmx, requests = _sdmx_client_for(bodies)
    # B has no `RoleNames` entry or coverage rows, so its candidate list is empty.
    coverage = _coverage_for(
        _GroupFacet(_BARE_A, "facetShared", "PROV_SHARED", "METH_SHARED", "UNIT_SHARED")
    )

    result = materialise_group(
        [request_a, request_b],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    # Candidates come from `coverage`, an in-memory lookup, so only the
    # one combined data fetch runs, no separate probe call. The empty
    # member stays in the group's resolved population with a zero count.
    assert len(requests) == 1
    assert result.coverage.rows == 1
    entries_by_ref = {entry["ref"]: entry for entry in result.trace.resolved_stat_vars}
    assert entries_by_ref[_STAT_VAR_A]["observation_count"] == 1
    assert entries_by_ref[_STAT_VAR_B]["observation_count"] == 0
    facets_by_axis = {entry["axis"]: entry for entry in result.trace.facets}
    assert facets_by_axis["provenance"]["ref"] == "PROV_SHARED"


def test_group_shared_candidates_raises_when_every_member_has_an_empty_pool():
    # Neither member's bound entities publish an observation. There is no
    # shared facet to find, the same refusal a lone member's empty probe
    # reaches.
    requests = _group_requests(facet_bindings={})
    candidates: list[list[Facet]] = [[], []]

    with pytest.raises(FacetNotAvailableError):
        select_group_facets(requests, candidates, graph=FakeGraphReader())


# Empty candidate sets determine fetch routing and policy agreement.


def test_group_materialise_member_with_empty_probe_reports_zero_rows_despite_a_matching_fetch_row():
    # B has no candidate, so its dispatch ids are empty. Its SDMX body
    # carries the selected facet, but row routing drops it before survival
    # checks and records zero observations for B.
    request_a = _request(
        stat_var=_STAT_VAR_A,
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref="PROV_SHARED"),
            "unit": FacetBinding(state="bound", ref="UNIT_SHARED"),
        },
    )
    request_b = _request(
        stat_var=_STAT_VAR_B,
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref="PROV_SHARED"),
            "unit": FacetBinding(state="bound", ref="UNIT_SHARED"),
        },
    )
    graph = FakeGraphReader()
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="10",
            unit="UNIT_SHARED",
            method="METH_SHARED",
            provenance="PROV_SHARED",
            facet_id="facetShared",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2014",
            value="99",
            unit="UNIT_SHARED",
            method="METH_SHARED",
            provenance="PROV_SHARED",
            facet_id="facetShared",
        ),
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    # B has no candidate at all (no `RoleNames` entry for `_BARE_B`): B's
    # own facet probe finds nothing, disagreeing with the real (non-LATEST)
    # fetch above, which does carry a matching row for B.
    coverage = _coverage_for(
        _GroupFacet(_BARE_A, "facetShared", "PROV_SHARED", "METH_SHARED", "UNIT_SHARED")
    )

    result = materialise_group(
        [request_a, request_b],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    # One member holds a candidate, so the group is not the empty-probe result.
    assert not result.probe_empty
    assert result.coverage.rows == 1
    entries_by_ref = {entry["ref"]: entry for entry in result.trace.resolved_stat_vars}
    assert entries_by_ref[_STAT_VAR_A]["observation_count"] == 1
    assert entries_by_ref[_STAT_VAR_B]["observation_count"] == 0


def test_group_empty_member_does_not_break_the_populated_members_declared_policy_agreement():
    # A and C share the `facetPolicy` default and `facetBig`, which has a
    # higher count. B has no candidates or policy, so policy agreement uses
    # A and C. Including B would select the higher-count `facetBig` default.
    facet_policy = Facet(
        facet_id="facetPolicy",
        provenance_id="PROV_POLICY",
        measurement_method="METH1",
        unit="UNIT1",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    facet_big = Facet(
        facet_id="facetBig",
        provenance_id="PROV_BIG",
        measurement_method="METH1",
        unit="UNIT1",
        obs_count=100,
        earliest_date="2020",
        latest_date="2020",
    )
    policy = FacetDefaultPolicy(
        resolved_ref="PROV_POLICY",
        policy_node="dcid:GraphDeclaredProvenancePolicy",
        policy_version="graph-v1",
        selection_reason="graph_declared",
    )
    graph = FakeGraphReader(
        policies={
            (_STAT_VAR_A, "provenance"): policy,
            (_STAT_VAR_C, "provenance"): policy,
            # _STAT_VAR_B declares nothing for "provenance".
        }
    )
    requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings={}),
        _request(stat_var=_STAT_VAR_B, facet_bindings={}),
        _request(stat_var=_STAT_VAR_C, facet_bindings={}),
    ]
    candidates = [[facet_policy, facet_big], [], [facet_policy, facet_big]]

    selections = select_group_facets(requests, candidates, graph=graph).election.selections

    assert selections["provenance"].basis == "declared_default"
    assert selections["provenance"].ref == "PROV_POLICY"


def test_index_sourced_group_empty_member_does_not_break_the_populated_members_declared_policy_agreement():
    # Index-sourced members agree on provenance and unit despite differing
    # facet ids and methods. A and C share a provenance policy. B has no
    # candidates or policy. The selected provenance remains `PROV_POLICY`.
    facet_a = Facet(
        facet_id="facetA",
        provenance_id="PROV_POLICY",
        measurement_method="METH_A",
        unit="UNIT1",
        obs_count=1,
        earliest_date="2020",
        latest_date="2020",
    )
    facet_c = Facet(
        facet_id="facetC",
        provenance_id="PROV_POLICY",
        measurement_method="METH_C",
        unit="UNIT1",
        obs_count=100,
        earliest_date="2020",
        latest_date="2020",
    )
    policy = FacetDefaultPolicy(
        resolved_ref="PROV_POLICY",
        policy_node="dcid:GraphDeclaredProvenancePolicy",
        policy_version="graph-v1",
        selection_reason="graph_declared",
    )
    graph = FakeGraphReader(
        policies={
            (_STAT_VAR_A, "provenance"): policy,
            (_STAT_VAR_C, "provenance"): policy,
        }
    )
    requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings={}, index_sourced=True),
        _request(stat_var=_STAT_VAR_B, facet_bindings={}, index_sourced=True),
        _request(stat_var=_STAT_VAR_C, facet_bindings={}, index_sourced=True),
    ]
    candidates = [[facet_a], [], [facet_c]]

    selections = select_group_facets(requests, candidates, graph=graph).election.selections

    assert selections["provenance"].basis == "declared_default"
    assert selections["provenance"].ref == "PROV_POLICY"


@pytest.mark.parametrize("window_kind", ["latest", "all"])
def test_group_generate_chunk_role_bearing_member_with_empty_dispatch_ids_reports_zero_rows(
    window_kind: str,
):
    # The mock serves one body to the probe and the fetch, so call
    # `_generate_chunk` directly, under each window, with
    # B's dispatch ids empty while its body contains the selected facet and a row
    # naming no facet. A role-bearing member's rows pass the admission mask, so the
    # chunk skips a member that dispatches nothing.
    request_a = _request(
        stat_var=_STAT_VAR_A,
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref="PROV_SHARED"),
            "unit": FacetBinding(state="bound", ref="UNIT_SHARED"),
        },
        window=RequestedWindow(kind=window_kind),
    )
    request_b = _request(
        stat_var=_STAT_VAR_B,
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref="PROV_SHARED"),
            "unit": FacetBinding(state="bound", ref="UNIT_SHARED"),
        },
        window=RequestedWindow(kind=window_kind),
    )
    shared_facet = Facet(
        facet_id="facetShared",
        provenance_id="PROV_SHARED",
        measurement_method="METH_SHARED",
        unit="UNIT_SHARED",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    selections = select_group_facets(
        [request_a, request_b], [[shared_facet], [shared_facet]], graph=FakeGraphReader()
    ).election.selections
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="10",
            unit="UNIT_SHARED",
            method="METH_SHARED",
            provenance="PROV_SHARED",
            facet_id="facetShared",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2015",
            value="99",
            unit="UNIT_SHARED",
            method="METH_SHARED",
            provenance="PROV_SHARED",
            facet_id="facetShared",
        )
        # A malformed row naming no facet: coverage admission keeps every row, so only
        # skipping the member keeps it out.
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2016",
            value="99",
            unit="UNIT_SHARED",
            method="METH_SHARED",
            provenance="PROV_SHARED",
            facet_id="",
        ),
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    chunk_requests = [("0", request_a), ("1", request_b)]
    (chunk,) = _plan_collection_chunks(chunk_requests, places=[(), ()])
    plans = [
        _MemberFetchPlan(
            selections=selections,
            dispatch_ids=("facetShared",),
            admission=CoverageAdmission(),
            conversion_plan=None,
        ),
        # B's probe found no candidate, an empty dispatch set, even
        # though its real body above carries a matching row.
        _MemberFetchPlan(
            selections=selections,
            dispatch_ids=(),
            admission=CoverageAdmission(),
            conversion_plan=None,
        ),
    ]

    result = _generate_chunk(
        chunk,
        chunk_requests,
        plans,
        sdmx=sdmx,
        graph=FakeGraphReader(),
        ceiling=100,
        raw_budget=_RawRowBudget(100),
    )

    assert result[0].table.num_rows == 1
    assert result[1].table.num_rows == 0
