"""`generation._generate_group`'s probe phase: with no `precomputed_probe`,
one `facets.probe_collection_candidates` call covers the whole group (one
batched request per entity-scope group) and its `members[i]` pairs with
`requests[i]`. Members that bind no entity share the default place scope,
the fetch names the places that probe returned, and each member's admission
holds its own facets at those places. Also pins
`probe_collection_candidates` itself against `probe_collection_facets`:
same member probes, no provenance refresh call.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence

import httpx
import pytest

from dc_sdmx import SdmxClient
from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    materialise_group,
)
from materialise import facets as facets_module
from materialise.errors import CollectionProbeCeilingExceededError
from materialise.facets import (
    CollectionFacetProbe,
    CoverageAdmission,
    MemberProbe,
    PlaceAdmission,
    probe_collection_candidates,
    probe_collection_facets,
)
from materialise.graph import Facet
from materialise.window import RequestedWindow

from ._group_support import (
    _BARE_A,
    _BARE_B,
    _BARE_C,
    _BOUND_FACETS,
    _CSV_HEADER,
    _ENTITY,
    _STAT_VAR_A,
    _STAT_VAR_B,
    _STAT_VAR_C,
    _coverage_for,
    _csv_row,
    _request,
    _sdmx_client_for,
)
from ._scope_availability import NO_SCOPE_READ, ScopeAvailability
from .conftest import FakeGraphReader

_ABOUT = "observationAbout"
_COUNTRY_X = "country/XXX"
_COUNTRY_Y = "country/YYY"
_ABOUT_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _facet(facet_id: str) -> Facet:
    """Every facet here carries the same provenance/method/unit triple, so
    any of them serves a group whose members share `f-shared`; only the
    `facet_id` tells a member's own candidates apart.
    """
    return Facet(
        facet_id=facet_id,
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="UNIT1",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )


def _admitted_at(entity: str, *facets: Facet) -> MemberProbe:
    """A subject-only member probe whose every candidate was admitted at `entity`."""
    return MemberProbe(
        candidates=facets,
        admission=PlaceAdmission(
            places_by_facet={
                facet.facet_id: frozenset({entity}) for facet in facets if facet.facet_id
            }
        ),
    )


def _about_request(bare_stat_var: str, entity: str) -> MaterialisationRequest:
    """A subject-only member: its probe rides the graph's batched
    `/v2/observation` path, not SDMX.
    """
    return MaterialisationRequest(
        stat_var=f"dcid:{bare_stat_var}",
        role_bindings={_ABOUT: [entity]},
        declared_roles=(_ABOUT,),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )


class _RecordingGraph(FakeGraphReader):
    """`FakeGraphReader` recording each batched probe's own arguments, so a
    test can name which stat_vars and entities one call carried.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.batch_many_args: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def observation_facets_batch_many(self, *, stat_vars, entities, deadline=None):
        self.batch_many_args.append((tuple(stat_vars), tuple(entities)))
        return super().observation_facets_batch_many(
            stat_vars=stat_vars, entities=entities, deadline=deadline
        )


class _ProbeTransportError(RuntimeError):
    """A hard failure raised by a batched graph probe."""


class _FailingGraph(_RecordingGraph):
    """Raises `_ProbeTransportError` from every batched probe whose
    entities include one of `failing_entities`, naming the first such
    entity in the message.
    """

    def __init__(self, *, failing_entities: Sequence[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._failing_entities = tuple(failing_entities)

    def observation_facets_batch_many(self, *, stat_vars, entities, deadline=None):
        for entity in entities:
            if entity in self._failing_entities:
                self.batch_many_args.append((tuple(stat_vars), tuple(entities)))
                raise _ProbeTransportError(entity)
        return super().observation_facets_batch_many(
            stat_vars=stat_vars, entities=entities, deadline=deadline
        )


def _about_sdmx(
    entity_by_bare_stat_var: dict[str, str],
    *,
    availability: ScopeAvailability | None = None,
) -> tuple[SdmxClient, list[httpx.Request]]:
    """Answers each data fetch with one `f-shared` row per named stat_var
    in `entity_by_bare_stat_var`, and the default place scope's
    availability reads from `availability`. Records every data request. A
    subject-only group makes no SDMX facet probe, so every recorded request
    is a fetch.
    """
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if availability is not None and (response := availability.respond(request)) is not None:
            return response
        requests.append(request)
        named = str(request.url.params.get("c[variableMeasured]")).split(",")
        rows = "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{bare},{entity_by_bare_stat_var[bare]},UNIT1,METH1,"
            f"NotApplicable,PROV1,2015,10,,f-shared\r\n"
            for bare in named
            if bare in entity_by_bare_stat_var
        )
        return httpx.Response(
            200,
            text=_ABOUT_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    ), requests


def _group(requests, *, sdmx, graph, **kwargs):
    return materialise_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        **kwargs,
    )


def _fetch_wire(requests: list[httpx.Request]) -> dict[frozenset[str], tuple[str, ...]]:
    """`c[variableMeasured]` set to `c[facetId]` tuple, for every fetch
    request (the ones carrying a facet filter).
    """
    return {
        frozenset(str(r.url.params["c[variableMeasured]"]).split(",")): tuple(
            str(r.url.params["c[facetId]"]).split(",")
        )
        for r in requests
        if "c[facetId]" in r.url.params
    }


# ---------------------------------------------------------------------------
# (a) One batched probe for the whole group
# ---------------------------------------------------------------------------


def test_subject_only_group_issues_one_batched_graph_probe_and_no_per_member_probe():
    graph = _RecordingGraph(
        facets={(bare, _COUNTRY_X): [_facet("f-shared")] for bare in (_BARE_A, _BARE_B, _BARE_C)}
    )
    sdmx, _sdmx_requests = _about_sdmx(dict.fromkeys((_BARE_A, _BARE_B, _BARE_C), _COUNTRY_X))

    result = _group(
        [_about_request(bare, _COUNTRY_X) for bare in (_BARE_A, _BARE_B, _BARE_C)],
        sdmx=sdmx,
        graph=graph,
    )

    assert graph.batch_many_args == [((_BARE_A, _BARE_B, _BARE_C), (_COUNTRY_X,))]
    assert graph.calls["observation_facets_batch_many"] == 1
    assert graph.calls["variable_facets_batch"] == 0
    assert result.coverage.rows == 3


def test_role_bearing_group_reads_facets_from_coverage_and_issues_no_probe_request():
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
        for bare, stat_var in (
            (_BARE_A, _STAT_VAR_A),
            (_BARE_B, _STAT_VAR_B),
            (_BARE_C, _STAT_VAR_C),
        )
    }
    sdmx, requests = _sdmx_client_for(bodies)
    coverage = _coverage_for([(_BARE_A, "facet1"), (_BARE_B, "facet1"), (_BARE_C, "facet1")])

    result = _group(
        [
            _request(stat_var=stat_var, facet_bindings=_BOUND_FACETS)
            for stat_var in (_STAT_VAR_A, _STAT_VAR_B, _STAT_VAR_C)
        ],
        sdmx=sdmx,
        graph=graph,
        coverage=coverage,
        # `_plan_collection_chunks`'s own lane target is `ceil(3 / lanes)`;
        # `member_concurrency=1` (`ceil(3 / 1) == 3`) keeps all three
        # same-scope members in the one shared chunk this test means to
        # exercise, instead of the default `4`'s own split.
        member_concurrency=1,
    )

    # No LATEST-scoped facet probe: the facet candidates come from
    # `coverage`, an in-memory read. Only the one shared chunk fetch
    # reaches the transport.
    probes = [r for r in requests if r.url.params.get("c[TIME_PERIOD]") == "LATEST"]
    assert probes == []
    assert len(requests) == 1
    assert result.coverage.rows == 3


# ---------------------------------------------------------------------------
# (b) candidates[i] pairs with requests[i]
# ---------------------------------------------------------------------------


def test_each_members_own_candidates_reach_its_own_fetch_wire_across_interleaved_scopes():
    """A and C share entity scope X, B binds Y, so the batched probe splits
    into two groups (`[A, C]`, `[B]`) and stitches candidates back into
    request order. Each member carries `f-shared` plus one facet only it
    has. C's `latest` window keeps it out of A's fetch chunk, so every
    member is fetched alone and its `c[facetId]` is exactly its own
    dispatch ids: a member handed another member's candidates puts that
    member's facet on the wire instead of its own.
    """
    graph = _RecordingGraph(
        facets={
            (_BARE_A, _COUNTRY_X): [_facet("f-shared"), _facet("f-a")],
            (_BARE_B, _COUNTRY_Y): [_facet("f-shared"), _facet("f-b")],
            (_BARE_C, _COUNTRY_X): [_facet("f-shared"), _facet("f-c")],
        }
    )
    sdmx, sdmx_requests = _about_sdmx(
        {_BARE_A: _COUNTRY_X, _BARE_B: _COUNTRY_Y, _BARE_C: _COUNTRY_X}
    )

    result = _group(
        [
            _about_request(_BARE_A, _COUNTRY_X),
            _about_request(_BARE_B, _COUNTRY_Y),
            dataclasses.replace(
                _about_request(_BARE_C, _COUNTRY_X), window=RequestedWindow(kind="latest")
            ),
        ],
        sdmx=sdmx,
        graph=graph,
    )

    assert graph.batch_many_args == [
        ((_BARE_A, _BARE_C), (_COUNTRY_X,)),
        ((_BARE_B,), (_COUNTRY_Y,)),
    ]
    assert _fetch_wire(sdmx_requests) == {
        frozenset({_BARE_A}): ("f-a", "f-shared"),
        frozenset({_BARE_C}): ("f-c", "f-shared"),
        frozenset({_BARE_B}): ("f-b", "f-shared"),
    }
    assert result.coverage.rows == 3


# ---------------------------------------------------------------------------
# (c) Members that bind no entity, and hard-failure precedence
# ---------------------------------------------------------------------------

_COUNTRY_Z = "country/ZZZ"
_EARTH_COUNTRIES = {("Earth", "Country"): [_COUNTRY_X, _COUNTRY_Y, _COUNTRY_Z]}


def _unbound(request: MaterialisationRequest) -> MaterialisationRequest:
    return dataclasses.replace(request, role_bindings={})


def _fetched_places(requests: list[httpx.Request]) -> dict[frozenset[str], tuple[str, ...]]:
    """`c[variableMeasured]` set to the `c[observationAbout]` tuple of every fetch."""
    return {
        frozenset(str(r.url.params["c[variableMeasured]"]).split(",")): tuple(
            str(r.url.params["c[observationAbout]"]).split(",")
        )
        for r in requests
    }


def test_unbound_members_are_each_probed_and_fetched_at_their_own_countries_with_data():
    # A holds data at X, C at Y, and neither at Z. B binds X itself.
    graph = _RecordingGraph(
        facets={
            (_BARE_A, _COUNTRY_X): [_facet("f-shared")],
            (_BARE_C, _COUNTRY_Y): [_facet("f-shared")],
            (_BARE_B, _COUNTRY_X): [_facet("f-shared")],
        },
        child_places=_EARTH_COUNTRIES,
    )
    sdmx, sdmx_requests = _about_sdmx(
        {_BARE_A: _COUNTRY_X, _BARE_B: _COUNTRY_X, _BARE_C: _COUNTRY_Y},
        availability=graph.scope_availability(),
    )

    result = _group(
        [
            _unbound(_about_request(_BARE_A, _COUNTRY_X)),
            _about_request(_BARE_B, _COUNTRY_X),
            _unbound(_about_request(_BARE_C, _COUNTRY_X)),
        ],
        sdmx=sdmx,
        graph=graph,
        # Keeps the two unbound members in one shared chunk.
        member_concurrency=1,
    )

    # Availability names the places with data, so the graph reads no containment: one batch for
    # each unbound member at its own places, and one for the bound member.
    assert graph.calls["child_places_batch"] == 0
    assert graph.batch_many_args == [
        ((_BARE_A,), (_COUNTRY_X,)),
        ((_BARE_C,), (_COUNTRY_Y,)),
        ((_BARE_B,), (_COUNTRY_X,)),
    ]
    # The fetch names the countries the probe found data at, each member's own.
    assert _fetched_places(sdmx_requests) == {
        frozenset({_BARE_A}): (_COUNTRY_X,),
        frozenset({_BARE_B}): (_COUNTRY_X,),
        frozenset({_BARE_C}): (_COUNTRY_Y,),
    }
    assert result.coverage.rows == 3


def test_unbound_members_with_data_only_at_earth_fetch_earth():
    graph = _RecordingGraph(
        facets={(bare, "Earth"): [_facet("f-shared")] for bare in (_BARE_A, _BARE_B)},
        child_places=_EARTH_COUNTRIES,
    )
    sdmx, sdmx_requests = _about_sdmx(
        {_BARE_A: "Earth", _BARE_B: "Earth"}, availability=graph.scope_availability()
    )

    result = _group(
        [
            _unbound(_about_request(_BARE_A, _COUNTRY_X)),
            _unbound(_about_request(_BARE_B, _COUNTRY_X)),
        ],
        sdmx=sdmx,
        graph=graph,
        member_concurrency=1,
    )

    assert _fetched_places(sdmx_requests) == {frozenset({_BARE_A, _BARE_B}): ("Earth",)}
    assert result.coverage.rows == 2


@pytest.mark.parametrize("bound", [False, True], ids=["default_scope", "bound_place"])
def test_a_group_whose_every_probe_returns_no_candidate_is_the_empty_result(bound: bool):
    # No default place and no bound place holds data for either member.
    graph = _RecordingGraph(child_places=_EARTH_COUNTRIES)
    sdmx, sdmx_requests = _about_sdmx({}, availability=graph.scope_availability())
    members = [_about_request(_BARE_A, _COUNTRY_X), _about_request(_BARE_B, _COUNTRY_X)]

    result = _group(
        members if bound else [_unbound(member) for member in members], sdmx=sdmx, graph=graph
    )

    assert result.probe_empty
    assert result.coverage.rows == 0
    assert [warning["code"] for warning in result.warnings] == ["empty_result"]
    assert result.trace.facets == ()
    assert [entry["observation_count"] for entry in result.trace.resolved_stat_vars] == [0, 0]
    assert sdmx_requests == []


def test_a_member_with_a_worldwide_total_alone_adds_its_earth_row_beside_country_data():
    # Each member is fetched at its own places, so B's worldwide total sits in the same table
    # as A's country row.
    graph = _RecordingGraph(
        facets={
            (_BARE_A, _COUNTRY_X): [_facet("f-shared")],
            (_BARE_B, "Earth"): [_facet("f-shared")],
        },
        child_places=_EARTH_COUNTRIES,
    )
    sdmx, sdmx_requests = _about_sdmx(
        {_BARE_A: _COUNTRY_X, _BARE_B: "Earth"}, availability=graph.scope_availability()
    )
    requests = [
        _unbound(_about_request(_BARE_A, _COUNTRY_X)),
        _unbound(_about_request(_BARE_B, _COUNTRY_X)),
    ]

    result = _group(requests, sdmx=sdmx, graph=graph, member_concurrency=1)

    assert _fetched_places(sdmx_requests) == {
        frozenset({_BARE_A}): (_COUNTRY_X,),
        frozenset({_BARE_B}): ("Earth",),
    }
    assert result.coverage.rows == 2


def test_a_supplied_probe_carries_its_default_places_to_the_fetch():
    graph = _RecordingGraph()
    sdmx, sdmx_requests = _about_sdmx({_BARE_A: _COUNTRY_Y, _BARE_B: _COUNTRY_Y})
    supplied = CollectionFacetProbe(
        members=(
            dataclasses.replace(_admitted_at(_COUNTRY_Y, _facet("f-shared")), places=(_COUNTRY_Y,)),
        )
        * 2,
    )

    result = _group(
        [
            _unbound(_about_request(_BARE_A, _COUNTRY_X)),
            _unbound(_about_request(_BARE_B, _COUNTRY_X)),
        ],
        sdmx=sdmx,
        graph=graph,
        precomputed_probe=supplied,
        member_concurrency=1,
    )

    assert graph.calls["child_places_batch"] == 0
    assert graph.batch_many_args == []
    assert _fetched_places(sdmx_requests) == {frozenset({_BARE_A, _BARE_B}): (_COUNTRY_Y,)}
    assert result.coverage.rows == 2


def test_a_batched_probe_failure_surfaces_from_the_bound_member_beside_an_unbound_one():
    graph = _FailingGraph(
        failing_entities=[_COUNTRY_X],
        child_places={("Earth", "Country"): [_COUNTRY_Y]},
        arcs={_COUNTRY_Y: {"typeOf": ("Country",)}},
    )
    sdmx, sdmx_requests = _about_sdmx(
        {}, availability=ScopeAvailability(data=[(_BARE_A, _COUNTRY_Y)])
    )

    with pytest.raises(_ProbeTransportError) as excinfo:
        _group(
            [
                _unbound(_about_request(_BARE_A, _COUNTRY_X)),
                _about_request(_BARE_B, _COUNTRY_X),
            ],
            sdmx=sdmx,
            graph=graph,
        )

    assert str(excinfo.value) == _COUNTRY_X
    # The unbound member's probe at the country availability named ran first, then the bound
    # member's failed.
    assert graph.batch_many_args == [
        ((_BARE_A,), (_COUNTRY_Y,)),
        ((_BARE_B,), (_COUNTRY_X,)),
    ]
    assert sdmx_requests == []


@pytest.mark.parametrize(
    ("first_entity", "second_entity"),
    [(_COUNTRY_X, _COUNTRY_Y), (_COUNTRY_Y, _COUNTRY_X)],
)
def test_the_earliest_failing_scope_group_in_request_order_surfaces_first(
    first_entity, second_entity
):
    graph = _FailingGraph(failing_entities=[_COUNTRY_X, _COUNTRY_Y])
    sdmx, _ = _about_sdmx({})

    with pytest.raises(_ProbeTransportError) as excinfo:
        _group(
            [_about_request(_BARE_A, first_entity), _about_request(_BARE_B, second_entity)],
            sdmx=sdmx,
            graph=graph,
        )

    assert str(excinfo.value) == first_entity
    assert graph.batch_many_args == [((_BARE_A,), (first_entity,))]


# ---------------------------------------------------------------------------
# (d) A supplied probe is used unchanged
# ---------------------------------------------------------------------------


def test_supplied_probe_reaches_the_fetch_wire_and_issues_no_probe_call():
    graph = _RecordingGraph()
    sdmx, sdmx_requests = _about_sdmx({_BARE_A: _COUNTRY_X, _BARE_B: _COUNTRY_X})
    supplied = CollectionFacetProbe(
        members=(
            _admitted_at(_COUNTRY_X, _facet("f-shared"), _facet("f-a")),
            _admitted_at(_COUNTRY_X, _facet("f-shared"), _facet("f-b")),
        ),
    )

    result = _group(
        [_about_request(_BARE_A, _COUNTRY_X), _about_request(_BARE_B, _COUNTRY_X)],
        sdmx=sdmx,
        graph=graph,
        precomputed_probe=supplied,
        # `_plan_collection_chunks`' lane target is `ceil(2 / lanes)`, so one lane keeps both
        # same-scope members in one shared chunk.
        member_concurrency=1,
    )

    assert graph.batch_many_args == []
    assert graph.calls["observation_facets_batch_many"] == 0
    assert graph.calls["variable_facets_batch"] == 0
    assert _fetch_wire(sdmx_requests) == {frozenset({_BARE_A, _BARE_B}): ("f-a", "f-b", "f-shared")}
    assert result.coverage.rows == 2


# ---------------------------------------------------------------------------
# (f) probe_collection_candidates against probe_collection_facets
# ---------------------------------------------------------------------------


def _refresh_graph(facets) -> FakeGraphReader:
    return FakeGraphReader(facets=facets, arcs={"PROV1": {"lastDataRefreshDate": ("2024-01-01",)}})


def test_candidates_match_probe_collection_facets_on_the_graph_path_without_the_refresh_call():
    facets = {
        (_BARE_A, _COUNTRY_X): [_facet("f-shared"), _facet("f-a")],
        (_BARE_B, _COUNTRY_Y): [_facet("f-b")],
    }
    requests = [_about_request(_BARE_A, _COUNTRY_X), _about_request(_BARE_B, _COUNTRY_Y)]
    graph_candidates = _refresh_graph(facets)
    graph_facets = _refresh_graph(facets)

    members = probe_collection_candidates(
        requests, availability=NO_SCOPE_READ, settle=True, graph=graph_candidates
    )
    probe = probe_collection_facets(
        requests, availability=NO_SCOPE_READ, settle=True, graph=graph_facets
    )

    assert members == probe.members
    assert members == (
        _admitted_at(_COUNTRY_X, _facet("f-shared"), _facet("f-a")),
        _admitted_at(_COUNTRY_Y, _facet("f-b")),
    )
    assert graph_candidates.calls["observation_facets_batch_many"] == 2
    assert graph_candidates.calls["node_arcs_many"] == 0
    assert graph_facets.calls["node_arcs_many"] == 1
    assert set(probe.provenance_refresh) == {"PROV1"}


def test_candidates_match_probe_collection_facets_on_the_coverage_path_without_the_refresh_call():
    coverage = _coverage_for([(_BARE_A, f"facet-{_BARE_A}"), (_BARE_B, f"facet-{_BARE_B}")])
    requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS),
        _request(stat_var=_STAT_VAR_B, facet_bindings=_BOUND_FACETS),
    ]
    graph_candidates = _refresh_graph({})
    graph_facets = _refresh_graph({})

    members = probe_collection_candidates(
        requests, availability=NO_SCOPE_READ, settle=True, graph=graph_candidates, coverage=coverage
    )
    probe = probe_collection_facets(
        requests, availability=NO_SCOPE_READ, settle=True, graph=graph_facets, coverage=coverage
    )

    assert members == probe.members
    assert [c.facet_id for member in members for c in member.candidates] == [
        f"facet-{_BARE_A}",
        f"facet-{_BARE_B}",
    ]
    assert graph_candidates.calls["node_arcs_many"] == 0
    assert graph_facets.calls["node_arcs_many"] == 1


def test_candidates_of_no_requests_are_empty_and_make_no_call():
    graph = _refresh_graph({})

    assert (
        probe_collection_candidates([], availability=NO_SCOPE_READ, settle=True, graph=graph) == ()
    )
    assert sum(graph.calls.values()) == 0


# ---------------------------------------------------------------------------
# Members that bind no entity, at the probe itself
# ---------------------------------------------------------------------------


def test_candidates_give_an_unbound_member_the_default_places_and_its_facets_there():
    graph = _RecordingGraph(
        facets={
            (_BARE_A, _COUNTRY_Y): [_facet("f-a")],
            (_BARE_A, "Earth"): [_facet("f-world")],
            (_BARE_B, _COUNTRY_X): [_facet("f-shared")],
        },
        child_places=_EARTH_COUNTRIES,
    )
    requests = [
        _unbound(_about_request(_BARE_A, _COUNTRY_X)),
        _about_request(_BARE_B, _COUNTRY_X),
    ]

    probed = probe_collection_candidates(
        requests, settle=True, graph=graph, availability=graph.scope_availability()
    )

    # The unbound member is probed at every place it holds data at, Earth's total among them,
    # and admitted at each place where its facet was read.
    assert probed == (
        MemberProbe(
            candidates=(_facet("f-a"), _facet("f-world")),
            admission=PlaceAdmission(
                places_by_facet={"f-a": frozenset({_COUNTRY_Y}), "f-world": frozenset({"Earth"})}
            ),
            places=(_COUNTRY_Y, "Earth"),
        ),
        _admitted_at(_COUNTRY_X, _facet("f-shared")),
    )
    probe = probe_collection_facets(
        requests, settle=True, graph=graph, availability=graph.scope_availability()
    )
    assert probe.default_scope == (_COUNTRY_Y, "Earth")


def test_candidates_of_an_unbound_role_bearing_group_cover_every_coverage_row():
    # Each member's coverage sits at a recipient the request never names, under its own facet.
    coverage = _coverage_for([(_BARE_A, "facet1"), (_BARE_B, "facet2")], entity="country/ANY")
    requests = [
        dataclasses.replace(
            _request(stat_var=stat_var, facet_bindings=_BOUND_FACETS, entity=_ENTITY),
            role_bindings={},
        )
        for stat_var in (_STAT_VAR_A, _STAT_VAR_B)
    ]
    graph = FakeGraphReader()

    probed = probe_collection_candidates(
        requests, availability=NO_SCOPE_READ, settle=True, graph=graph, coverage=coverage
    )

    assert [[c.facet_id for c in member.candidates] for member in probed] == [
        ["facet1"],
        ["facet2"],
    ]
    # Coverage answers a role-bearing group, so no place scope is read and its rows serve unmasked.
    assert [member.admission for member in probed] == [CoverageAdmission()] * 2
    assert [member.places for member in probed] == [(), ()]
    assert sum(graph.calls.values()) == 0


def test_candidates_of_a_role_bearing_group_bound_to_an_empty_entity_list_match_no_coverage_row():
    # A role bound to an empty entity list binds no entity, so the coverage lookup matches none
    # of the members' rows and each member has no candidates.
    coverage = _coverage_for([(_BARE_A, "facet1"), (_BARE_B, "facet1")])
    requests = [
        dataclasses.replace(
            _request(stat_var=stat_var, facet_bindings=_BOUND_FACETS, entity=_ENTITY),
            role_bindings={"recipient": []},
        )
        for stat_var in (_STAT_VAR_A, _STAT_VAR_B)
    ]

    probed = probe_collection_candidates(
        requests,
        availability=NO_SCOPE_READ,
        settle=True,
        graph=FakeGraphReader(),
        coverage=coverage,
    )

    assert probed == (MemberProbe(candidates=(), admission=CoverageAdmission()),) * 2


# ---------------------------------------------------------------------------
# (g) Refusals that fire before the probe reaches any transport
# ---------------------------------------------------------------------------


def test_group_over_the_probe_pair_cap_is_refused_before_any_request_when_no_probe_is_supplied(
    monkeypatch,
):
    monkeypatch.setattr(facets_module, "_MAX_COLLECTION_PROBE_PAIRS", 1)
    graph = _RecordingGraph()
    sdmx, sdmx_requests = _about_sdmx({_BARE_A: _COUNTRY_X, _BARE_B: _COUNTRY_X})

    with pytest.raises(CollectionProbeCeilingExceededError):
        _group(
            [_about_request(_BARE_A, _COUNTRY_X), _about_request(_BARE_B, _COUNTRY_X)],
            sdmx=sdmx,
            graph=graph,
        )

    assert graph.batch_many_args == []
    assert sdmx_requests == []


def test_expired_deadline_refuses_before_the_fetch_reaches_the_transport():
    # Coverage supplies facet candidates in memory, so the expired deadline
    # applies to the fetch, not a probe call.
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
    coverage = _coverage_for([(_BARE_A, "facet1"), (_BARE_B, "facet1")])

    with pytest.raises(TimeoutError):
        _group(
            [
                _request(stat_var=stat_var, facet_bindings=_BOUND_FACETS)
                for stat_var in (_STAT_VAR_A, _STAT_VAR_B)
            ],
            sdmx=sdmx,
            graph=FakeGraphReader(),
            coverage=coverage,
            deadline=time.monotonic() - 1.0,
        )

    assert requests == []
