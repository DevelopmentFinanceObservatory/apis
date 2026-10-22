"""Collection facet probing batches members by entity scope and fetches by chunk."""

from __future__ import annotations

import random
import time
from collections.abc import Sequence

import httpx
import pytest

from dc_sdmx import SdmxClient
from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    materialise_group,
    place_scope,
)
from materialise.errors import CollectionProbeCeilingExceededError, MaterialisationUsageError
from materialise.facets import (
    _MAX_COLLECTION_PROBE_PAIRS,
    CollectionFacetProbe,
    probe_collection_facets,
    probe_facets,
)
from materialise.graph import Facet, ProvenanceRefresh
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._scope_availability import NO_SCOPE_READ
from .conftest import FakeGraphReader

_COUNTRIES = ("country/KEN", "country/TGO", "country/ETH")
_OTHER_COUNTRIES = ("country/USA", "country/CAN")


def _facet(facet_id: str, *, obs_count: int = 1) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=obs_count,
        earliest_date="2015",
        latest_date="2015",
    )


def _member_stat_vars(n: int) -> list[str]:
    return [f"dcid:CollectionProbeVar{i:03d}" for i in range(n)]


def _shared_scope_request(stat_var: str) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={"observationAbout": list(_COUNTRIES)},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )


_ROLE = "financingRecipient"
_ROLE_DONOR_NAME = "financingProvider"
_ROLE_RECIPIENT = "country/KEN"
_ROLE_DONOR = "country/FRA"


def _role_bearing_request(
    stat_var: str, *, recipient: str = _ROLE_RECIPIENT
) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={_ROLE: [recipient]},
        declared_roles=(_ROLE,),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )


def _role_coverage_for_variables(bare_stat_vars: Sequence[str]) -> CoverageIndex:
    """Build one metadata-bearing coverage facet for each bare StatVar."""
    return build_coverage_index(
        [(bare, f"facet-{bare}", _ROLE_DONOR, _ROLE_RECIPIENT, [2020]) for bare in bare_stat_vars],
        role_names={
            bare: RoleNames(donor=_ROLE_DONOR_NAME, recipient=_ROLE) for bare in bare_stat_vars
        },
        facet_metadata={
            f"facet-{bare}": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
            for bare in bare_stat_vars
        },
    )


def test_probe_collection_facets_one_call_covers_every_member():
    stat_vars = _member_stat_vars(5)
    graph = FakeGraphReader(
        facets={
            (sv.removeprefix("dcid:"), country): [_facet(f"facet-{sv}-{country}")]
            for sv in stat_vars
            for country in _COUNTRIES
        }
    )
    requests = [_shared_scope_request(sv) for sv in stat_vars]

    probe = probe_collection_facets(requests, availability=NO_SCOPE_READ, settle=True, graph=graph)

    assert isinstance(probe, CollectionFacetProbe)
    assert graph.calls["observation_facets_batch_many"] == 1
    assert graph.calls["variable_facets_batch"] == 0
    assert len(probe.candidates) == len(requests)


def test_probe_collection_facets_candidate_set_matches_probe_facets_per_member():
    """Each member gets the same candidates and admission as a standalone `probe_facets` call."""
    stat_vars = _member_stat_vars(4)
    facets = {
        (sv.removeprefix("dcid:"), country): [_facet(f"facet-{sv}-{country}", obs_count=idx + 1)]
        for idx, sv in enumerate(stat_vars)
        for country in _COUNTRIES
    }
    graph = FakeGraphReader(facets=facets)
    requests = [_shared_scope_request(sv) for sv in stat_vars]

    probe = probe_collection_facets(requests, availability=NO_SCOPE_READ, settle=True, graph=graph)

    for index, request in enumerate(requests):
        standalone = probe_facets(
            availability=NO_SCOPE_READ,
            stat_var=request.stat_var.removeprefix("dcid:"),
            bound_entities=list(_COUNTRIES),
            settle=True,
            graph=graph,
        )
        assert probe.members[index] == standalone


def test_probe_collection_facets_reverse_order_still_matches_per_member():
    """Candidate-set equality holds regardless of which order `requests` is
    probed in. Each member's own union is scoped to its own
    `bound_entities`, fully isolated from another member's entities, even
    though the underlying POST names every member's entities together.
    """
    stat_vars = _member_stat_vars(3)
    facets = {
        (sv.removeprefix("dcid:"), country): [_facet(f"facet-{sv}-{country}")]
        for sv in stat_vars
        for country in _COUNTRIES
    }
    graph = FakeGraphReader(facets=facets)
    requests = [_shared_scope_request(sv) for sv in stat_vars]

    forward = probe_collection_facets(
        requests, availability=NO_SCOPE_READ, settle=True, graph=graph
    )
    reversed_probe = probe_collection_facets(
        list(reversed(requests)), availability=NO_SCOPE_READ, settle=True, graph=graph
    )

    assert forward.candidates[0] == reversed_probe.candidates[-1]
    assert forward.candidates[-1] == reversed_probe.candidates[0]


def _other_scope_request(stat_var: str) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={"observationAbout": list(_OTHER_COUNTRIES)},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )


def test_probe_collection_facets_partitions_by_entity_scope_one_call_per_group():
    """Two entity scopes produce two batch calls; each member's candidates
    match its standalone `probe_facets` result.
    """
    shared_scope_vars = _member_stat_vars(3)
    other_scope_vars = [f"dcid:CollectionProbeOtherVar{i:03d}" for i in range(2)]
    facets = {
        (sv.removeprefix("dcid:"), country): [_facet(f"facet-{sv}-{country}")]
        for sv in shared_scope_vars
        for country in _COUNTRIES
    }
    facets.update(
        {
            (sv.removeprefix("dcid:"), country): [_facet(f"facet-{sv}-{country}")]
            for sv in other_scope_vars
            for country in _OTHER_COUNTRIES
        }
    )
    graph = FakeGraphReader(facets=facets)
    requests = [_shared_scope_request(sv) for sv in shared_scope_vars] + [
        _other_scope_request(sv) for sv in other_scope_vars
    ]

    probe = probe_collection_facets(requests, availability=NO_SCOPE_READ, settle=True, graph=graph)

    assert graph.calls["observation_facets_batch_many"] == 2
    assert graph.calls["variable_facets_batch"] == 0
    for index, request in enumerate(requests):
        bound_entities = (
            list(_COUNTRIES) if index < len(shared_scope_vars) else list(_OTHER_COUNTRIES)
        )
        standalone = probe_facets(
            availability=NO_SCOPE_READ,
            stat_var=request.stat_var.removeprefix("dcid:"),
            bound_entities=bound_entities,
            settle=True,
            graph=graph,
        )
        assert probe.members[index] == standalone


def test_probe_collection_facets_provenance_refresh_batched_once_across_groups():
    """(#439) Two entity-scope groups share one provenance id, so refresh
    metadata requires one `node_arcs_many` call.
    """
    shared_scope_vars = _member_stat_vars(3)
    other_scope_vars = [f"dcid:CollectionProbeOtherVar{i:03d}" for i in range(2)]
    facets = {
        (sv.removeprefix("dcid:"), country): [_facet(f"facet-{sv}-{country}")]
        for sv in shared_scope_vars
        for country in _COUNTRIES
    }
    facets.update(
        {
            (sv.removeprefix("dcid:"), country): [_facet(f"facet-{sv}-{country}")]
            for sv in other_scope_vars
            for country in _OTHER_COUNTRIES
        }
    )
    graph = FakeGraphReader(
        facets=facets,
        arcs={"OECD_DAC2A": {"lastDataRefreshDate": ("2026-03-01",)}},
    )
    requests = [_shared_scope_request(sv) for sv in shared_scope_vars] + [
        _other_scope_request(sv) for sv in other_scope_vars
    ]

    probe = probe_collection_facets(requests, availability=NO_SCOPE_READ, settle=True, graph=graph)

    assert graph.calls["observation_facets_batch_many"] == 2
    assert graph.calls["node_arcs_many"] == 1
    assert probe.provenance_refresh == {
        "OECD_DAC2A": ProvenanceRefresh(last_data_refresh_date="2026-03-01")
    }


def test_probe_collection_facets_provenance_refresh_absent_provenance_is_none_none():
    """A provenance without refresh dates maps to
    `ProvenanceRefresh(None)`.
    """
    stat_vars = _member_stat_vars(2)
    facets = {
        (sv.removeprefix("dcid:"), country): [_facet(f"facet-{sv}-{country}")]
        for sv in stat_vars
        for country in _COUNTRIES
    }
    graph = FakeGraphReader(facets=facets)  # No `arcs` at all.
    requests = [_shared_scope_request(sv) for sv in stat_vars]

    probe = probe_collection_facets(requests, availability=NO_SCOPE_READ, settle=True, graph=graph)

    assert graph.calls["node_arcs_many"] == 1
    assert probe.provenance_refresh == {
        "OECD_DAC2A": ProvenanceRefresh(last_data_refresh_date=None)
    }


def test_probe_collection_facets_no_provenance_ids_skips_the_node_arcs_many_call():
    """No candidate carrying a `provenance_id` at all (every member's
    facets are provenance-less) makes no `node_arcs_many` call, matching
    `node_arcs_many`'s own empty-input contract.
    """
    facets = {
        (sv.removeprefix("dcid:"), country): [
            Facet(
                facet_id=f"facet-{sv}-{country}",
                provenance_id=None,
                measurement_method="GrantEquivalent",
                unit="USD",
                obs_count=1,
                earliest_date="2015",
                latest_date="2015",
            )
        ]
        for sv in _member_stat_vars(2)
        for country in _COUNTRIES
    }
    graph = FakeGraphReader(facets=facets)
    requests = [_shared_scope_request(sv) for sv in _member_stat_vars(2)]

    probe = probe_collection_facets(requests, availability=NO_SCOPE_READ, settle=True, graph=graph)

    assert graph.calls["node_arcs_many"] == 0
    assert probe.provenance_refresh == {}


def test_probe_collection_facets_forwards_deadline_to_every_group_call():
    """`probe_collection_facets`'s own `deadline` reaches every entity-scope
    group's `observation_facets_batch_many` call unchanged, mirroring
    `test_group_facets.py`'s recording-double proof for the SDMX side.
    """
    shared_scope_vars = _member_stat_vars(2)
    other_scope_vars = [f"dcid:CollectionProbeOtherVar{i:03d}" for i in range(2)]
    facets = {
        (sv.removeprefix("dcid:"), country): [_facet(f"facet-{sv}-{country}")]
        for sv in shared_scope_vars
        for country in _COUNTRIES
    }
    facets.update(
        {
            (sv.removeprefix("dcid:"), country): [_facet(f"facet-{sv}-{country}")]
            for sv in other_scope_vars
            for country in _OTHER_COUNTRIES
        }
    )
    graph = FakeGraphReader(facets=facets)
    deadlines: list[float | None] = []
    original = graph.observation_facets_batch_many

    def recording_observation_facets_batch_many(**kwargs):
        deadlines.append(kwargs.get("deadline"))
        return original(**kwargs)

    graph.observation_facets_batch_many = recording_observation_facets_batch_many  # type: ignore[method-assign]
    deadline = time.monotonic() + 30
    requests = [_shared_scope_request(sv) for sv in shared_scope_vars] + [
        _other_scope_request(sv) for sv in other_scope_vars
    ]

    probe_collection_facets(
        requests, availability=NO_SCOPE_READ, settle=True, graph=graph, deadline=deadline
    )

    assert deadlines == [deadline, deadline]


def test_probe_collection_facets_over_pair_cap_raises_before_any_call():
    """A synthetic collection whose partitioned pair count crosses
    `_MAX_COLLECTION_PROBE_PAIRS` raises `CollectionProbeCeilingExceededError`
    before `observation_facets_batch_many` is ever called.
    """
    oversized_entities = [f"country/C{i}" for i in range(_MAX_COLLECTION_PROBE_PAIRS + 1)]
    request = MaterialisationRequest(
        stat_var="dcid:CollectionProbeOverCap",
        role_bindings={"observationAbout": oversized_entities},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )
    graph = FakeGraphReader()

    with pytest.raises(CollectionProbeCeilingExceededError) as excinfo:
        probe_collection_facets([request], availability=NO_SCOPE_READ, settle=True, graph=graph)

    assert str(_MAX_COLLECTION_PROBE_PAIRS + 1) in str(excinfo.value)
    assert str(_MAX_COLLECTION_PROBE_PAIRS) in str(excinfo.value)
    assert graph.calls["observation_facets_batch_many"] == 0


def _unbound_request(bare_stat_var: str) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=f"dcid:{bare_stat_var}",
        role_bindings={},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )


def test_members_binding_no_place_count_their_own_places_against_the_pair_cap(monkeypatch):
    # 50 members each hold data at 1,000 of 5,000 places, so their union is every one of the
    # 5,000. Counted at the union they are 250,000 pairs, past the cap; counted at their own
    # places they are 50,000.
    # Every member's availability read starts in this request.
    monkeypatch.setattr(place_scope, "AVAILABILITY_WORKERS", 50)
    pool = [f"geoId/{n:04d}" for n in range(5_000)]
    members = {
        f"CollectionProbeWide{i:02d}": [pool[(i * 100 + n) % 5_000] for n in range(1_000)]
        for i in range(50)
    }
    graph = FakeGraphReader(
        facets={
            (stat_var, place): [_facet("facet1")]
            for stat_var, places in members.items()
            for place in places
        }
    )

    probe = probe_collection_facets(
        [_unbound_request(stat_var) for stat_var in members],
        settle=True,
        graph=graph,
        availability=graph.scope_availability(),
    )

    assert [len(member.places) for member in probe.members] == [1_000] * 50
    assert all(member.every_place for member in probe.members)


def test_members_binding_no_place_past_the_pair_cap_refuse_with_a_remedy_and_keep_their_scopes(
    monkeypatch,
):
    # 41 members at a full 5,000-place scope each are 205,000 pairs. No member's scope shrinks
    # to fit, since a member's scope would then depend on its companions.
    monkeypatch.setattr(place_scope, "AVAILABILITY_WORKERS", 41)
    places = [f"geoId/{n:04d}" for n in range(5_000)]
    stat_vars = [f"CollectionProbeFull{i:02d}" for i in range(41)]
    graph = FakeGraphReader(
        facets={(stat_var, place): [_facet("facet1")] for stat_var in stat_vars for place in places}
    )

    with pytest.raises(CollectionProbeCeilingExceededError) as excinfo:
        probe_collection_facets(
            [_unbound_request(stat_var) for stat_var in stat_vars],
            settle=True,
            graph=graph,
            availability=graph.scope_availability(),
        )

    assert "205000" in str(excinfo.value)
    assert "name the places to cover or request fewer members" in str(excinfo.value)
    assert graph.presence_reads == []
    assert graph.calls["observation_facets_batch_many"] == 0


def test_probe_collection_facets_member_with_no_bound_entities_probes_the_default_places():
    """A subject-only member that binds no entity gets its facets at every place it holds data
    at, as availability lists them, and the probe returns those places as its `places`.
    """
    bound_request = _shared_scope_request(_member_stat_vars(1)[0])
    unbound_request = MaterialisationRequest(
        stat_var="dcid:CollectionProbeUnbound",
        role_bindings={},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )
    with_data, without_data = "country/AAA", "country/BBB"
    graph = FakeGraphReader(
        facets={
            **{
                (bound_request.stat_var.removeprefix("dcid:"), country): [_facet("facet1")]
                for country in _COUNTRIES
            },
            ("CollectionProbeUnbound", with_data): [_facet("facet2")],
        },
        child_places={("Earth", "Country"): [with_data, without_data]},
    )

    availability = graph.scope_availability()

    probe = probe_collection_facets(
        [bound_request, unbound_request], settle=True, graph=graph, availability=availability
    )

    assert probe.candidates[0] != ()
    assert probe.candidates[1] == (_facet("facet2"),)
    assert [member.places for member in probe.members] == [(), (with_data,)]
    assert availability.reads == [("CollectionProbeUnbound",)]
    assert graph.calls["child_places_batch"] == 0


def test_probe_collection_facets_empty_requests_returns_empty_probe_no_call():
    graph = FakeGraphReader()

    probe = probe_collection_facets([], availability=NO_SCOPE_READ, settle=True, graph=graph)

    assert probe.candidates == ()
    assert graph.calls["observation_facets_batch_many"] == 0


def test_probe_collection_facets_role_bearing_group_reads_candidates_from_coverage():
    stat_vars = _member_stat_vars(3)
    bare_stat_vars = [sv.removeprefix("dcid:") for sv in stat_vars]
    coverage = _role_coverage_for_variables(bare_stat_vars)
    requests = [_role_bearing_request(sv) for sv in stat_vars]

    probe = probe_collection_facets(
        requests,
        availability=NO_SCOPE_READ,
        settle=True,
        graph=FakeGraphReader(),
        coverage=coverage,
    )

    assert len(probe.candidates) == 3
    for index, bare in enumerate(bare_stat_vars):
        (candidate,) = probe.candidates[index]
        assert candidate.facet_id == f"facet-{bare}"
        assert candidate.provenance_id == "OECD_DAC2A"
        assert candidate.obs_count == 1


def test_probe_collection_facets_role_bearing_candidate_set_matches_probe_facets_per_member():
    """Each role-bearing member's own coverage-sourced candidates are
    byte-identical to what `probe_facets` returns calling it alone for that
    member's own `(stat_var, role_bindings)` pair, mirroring the graph
    path's own parity guarantee.
    """
    stat_vars = _member_stat_vars(2)
    bare_stat_vars = [sv.removeprefix("dcid:") for sv in stat_vars]
    coverage = _role_coverage_for_variables(bare_stat_vars)
    requests = [_role_bearing_request(sv) for sv in stat_vars]

    probe = probe_collection_facets(
        requests,
        availability=NO_SCOPE_READ,
        settle=True,
        graph=FakeGraphReader(),
        coverage=coverage,
    )

    for index, bare in enumerate(bare_stat_vars):
        standalone = probe_facets(
            availability=NO_SCOPE_READ,
            stat_var=bare,
            bound_entities=[_ROLE_RECIPIENT],
            settle=True,
            graph=FakeGraphReader(),
            declared_roles=(_ROLE,),
            role_bindings={_ROLE: [_ROLE_RECIPIENT]},
            coverage=coverage,
        )
        assert probe.members[index] == standalone


def test_probe_collection_facets_role_bearing_group_uncovered_stat_var_returns_no_candidates():
    """A covered and uncovered StatVar can share a role-bearing group. The
    uncovered member returns `()` while covered candidates remain.
    """
    covered, uncovered = _member_stat_vars(2)
    coverage = _role_coverage_for_variables([covered.removeprefix("dcid:")])
    requests = [_role_bearing_request(covered), _role_bearing_request(uncovered)]

    probe = probe_collection_facets(
        requests,
        availability=NO_SCOPE_READ,
        settle=True,
        graph=FakeGraphReader(),
        coverage=coverage,
    )

    assert probe.candidates[0] != ()
    assert probe.candidates[1] == ()


def test_probe_collection_facets_role_bearing_group_of_many_members_maps_each_to_its_own():
    """300 covered stat_vars, each named twice (with and without the
    `dcid:` prefix), one uncovered stat_var named three times, and a second
    recipient's group reusing some of the same stat_vars, all interleaved:
    every member gets its own stat_var's candidates from its own group, and
    members naming one stat_var in one group get the same tuple."""
    bare_stat_vars = [f"CollectionProbeManyVar{i:03d}" for i in range(300)]
    uncovered = "CollectionProbeManyUncovered"
    coverage = _role_coverage_for_variables(bare_stat_vars)
    members = (
        [(f"dcid:{bare}", _ROLE_RECIPIENT) for bare in bare_stat_vars]
        + [(bare, _ROLE_RECIPIENT) for bare in bare_stat_vars]
        + [(f"dcid:{uncovered}", _ROLE_RECIPIENT)] * 3
        + [(f"dcid:{bare}", "country/ETH") for bare in bare_stat_vars[::7]]
    )
    random.Random(4).shuffle(members)
    requests = [_role_bearing_request(sv, recipient=recipient) for sv, recipient in members]

    probe = probe_collection_facets(
        requests,
        availability=NO_SCOPE_READ,
        settle=True,
        graph=FakeGraphReader(),
        coverage=coverage,
    )

    assert len(probe.candidates) == len(members)
    by_member: dict[tuple[str, str], tuple[Facet, ...]] = {}
    for (stat_var, recipient), candidates in zip(members, probe.candidates, strict=True):
        bare = stat_var.removeprefix("dcid:")
        if bare == uncovered or recipient != _ROLE_RECIPIENT:
            assert candidates == ()
        else:
            (candidate,) = candidates
            assert (candidate.facet_id, candidate.obs_count) == (f"facet-{bare}", 1)
        assert by_member.setdefault((bare, recipient), candidates) == candidates


def test_probe_collection_facets_mixed_groups_route_each_group_independently():
    """One `observationAbout` group and one role-bearing group in the same
    call each take their own path: one `observation_facets_batch_many` call
    for the graph-path group, a `coverage`-sourced lookup per stat_var for
    the role-bearing group.
    """
    observation_about_vars = _member_stat_vars(2)
    role_bearing_vars = [f"dcid:CollectionProbeRoleVar{i:03d}" for i in range(2)]
    bare_role_vars = [sv.removeprefix("dcid:") for sv in role_bearing_vars]
    graph = FakeGraphReader(
        facets={
            (sv.removeprefix("dcid:"), country): [_facet(f"facet-{sv}-{country}")]
            for sv in observation_about_vars
            for country in _COUNTRIES
        }
    )
    coverage = _role_coverage_for_variables(bare_role_vars)
    requests = [_shared_scope_request(sv) for sv in observation_about_vars] + [
        _role_bearing_request(sv) for sv in role_bearing_vars
    ]

    probe = probe_collection_facets(
        requests, availability=NO_SCOPE_READ, settle=True, graph=graph, coverage=coverage
    )

    assert graph.calls["observation_facets_batch_many"] == 1
    for index in range(len(observation_about_vars)):
        assert probe.candidates[index] != ()
    for index, bare in enumerate(bare_role_vars, start=len(observation_about_vars)):
        (candidate,) = probe.candidates[index]
        assert candidate.facet_id == f"facet-{bare}"
        assert candidate.obs_count == 1


def test_probe_collection_facets_role_bearing_without_coverage_raises_before_any_call():
    graph = FakeGraphReader()
    requests = [_role_bearing_request(sv) for sv in _member_stat_vars(1)]

    with pytest.raises(MaterialisationUsageError, match="coverage"):
        probe_collection_facets(requests, availability=NO_SCOPE_READ, settle=True, graph=graph)

    assert graph.calls["observation_facets_batch_many"] == 0


def _single_variable_sdmx_client(bare_stat_var: str) -> SdmxClient:
    def handler(request: httpx.Request) -> httpx.Response:
        header = (
            "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,"
            "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
            "scalingFactor,facetId\r\n"
        )
        row = (
            f"dataflow,DC:DF_OBS(1.0.0),I,{bare_stat_var},{_COUNTRIES[0]},USD,GrantEquivalent,"
            f"NotApplicable,OECD_DAC2A,2015,1000,,facet-{bare_stat_var}\r\n"
        )
        return httpx.Response(
            200,
            text=header + row,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )


def test_materialise_group_single_member_precomputed_probe_skips_the_network_call():
    stat_var = _member_stat_vars(1)[0]
    bare_stat_var = stat_var.removeprefix("dcid:")
    graph = FakeGraphReader(
        facets={
            (bare_stat_var, country): [_facet(f"facet-{bare_stat_var}")] for country in _COUNTRIES
        }
    )
    request = _shared_scope_request(stat_var)
    probe = probe_collection_facets([request], availability=NO_SCOPE_READ, settle=True, graph=graph)
    assert graph.calls["observation_facets_batch_many"] == 1

    sdmx = _single_variable_sdmx_client(bare_stat_var)

    result = materialise_group(
        [request],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        precomputed_probe=probe,
    )

    assert graph.calls["observation_facets_batch_many"] == 1
    assert graph.calls["variable_facets_batch"] == 0
    assert result.coverage.rows == 1


def test_materialise_group_multi_member_precomputed_probe_skips_every_probe_call():
    stat_vars = _member_stat_vars(2)
    bare_stat_vars = [sv.removeprefix("dcid:") for sv in stat_vars]
    graph = FakeGraphReader(
        facets={
            (bare, country): [_facet("shared-facet")]
            for bare in bare_stat_vars
            for country in _COUNTRIES
        }
    )
    requests = [_shared_scope_request(sv) for sv in stat_vars]
    probe = probe_collection_facets(requests, availability=NO_SCOPE_READ, settle=True, graph=graph)
    assert graph.calls["observation_facets_batch_many"] == 1

    def handler(request: httpx.Request) -> httpx.Response:
        # Both members share `_shared_scope_request`'s own entity scope, so
        # `_plan_collection_chunks` puts them in one chunk. `_generate_chunk`
        # sends one combined request naming both stat_vars, comma-joined.
        requested_vars = str(request.url.params.get("c[variableMeasured]")).split(",")
        header = (
            "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,"
            "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
            "scalingFactor,facetId\r\n"
        )
        rows = "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{_COUNTRIES[0]},USD,GrantEquivalent,"
            f"NotApplicable,OECD_DAC2A,2015,1000,,shared-facet\r\n"
            for stat_var in requested_vars
        )
        return httpx.Response(
            200,
            text=header + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )

    result = materialise_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        precomputed_probe=probe,
    )

    # `_generate_group` probes each member individually when no precomputed
    # probe is given. With one, none of those per-member calls happen, so
    # the count stays at the one call the standalone probe above made.
    assert graph.calls["observation_facets_batch_many"] == 1
    assert graph.calls["variable_facets_batch"] == 0
    assert result.coverage.rows == 2
