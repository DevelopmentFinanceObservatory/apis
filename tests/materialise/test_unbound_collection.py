"""Group members that bind no entity: each one gets the candidates, rows and
`content_version` the same request gets alone, whatever its siblings hold and whatever the
request order. Members are fetched at every place where any of them holds data, as availability
reports them (`materialise.place_scope.places_with_data`).
"""

from __future__ import annotations

import dataclasses
import itertools

import httpx
import pytest

from dc_sdmx import SdmxClient
from graph_release.context import current_release
from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    materialise_group,
    materialise_spec,
    place_scope,
)
from materialise import facets as facets_module
from materialise.errors import MaterialisationUsageError
from materialise.facets import (
    CoverageAdmission,
    MemberProbe,
    PlaceAdmission,
    probe_collection_candidates,
    probe_collection_facets,
    probe_facets,
)
from materialise.generation import _plan_collection_chunks
from materialise.pipeline import ConcatenatedFrameChunks
from materialise.window import RequestedWindow
from origin_catalogue.origins import DFO
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, provenance_row

from ._group_support import (
    _BARE_A,
    _BARE_B,
    _BARE_C,
    _CSV_HEADER,
    _SHARED_FACET,
    _STAT_VAR_A,
    _coverage_for,
    _csv_row,
    _request,
)
from ._scope_availability import ScopeAvailability
from .conftest import FakeGraphReader

_ABOUT = "observationAbout"
_KEN = "country/KEN"
_TGO = "country/TGO"
_ABOUT_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)

# `_BARE_A` is role-bearing and lives in coverage. `_BARE_B` holds data at Kenya and a
# worldwide total. `_BARE_C` holds a worldwide total alone.
_SUBJECT_DATA = ((_BARE_B, _KEN), (_BARE_B, "Earth"), (_BARE_C, "Earth"))


def _graph() -> FakeGraphReader:
    return FakeGraphReader(facets={key: [_SHARED_FACET] for key in _SUBJECT_DATA})


def _role_bearing() -> MaterialisationRequest:
    return dataclasses.replace(_request(stat_var=_STAT_VAR_A, facet_bindings={}), role_bindings={})


def _subject_only(bare_stat_var: str) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=f"dcid:{bare_stat_var}",
        role_bindings={},
        declared_roles=(_ABOUT,),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )


_ROLE_SPEC = "spec_" + "a" * 32
_COUNTRY_SPEC = "spec_" + "b" * 32
_WORLD_SPEC = "spec_" + "c" * 32
_MEMBERS = {
    _ROLE_SPEC: _role_bearing,
    _COUNTRY_SPEC: lambda: _subject_only(_BARE_B),
    _WORLD_SPEC: lambda: _subject_only(_BARE_C),
}
_ORDERS = list(itertools.permutations(_MEMBERS))


def _alone(request: MaterialisationRequest, graph: FakeGraphReader, coverage) -> MemberProbe:
    return probe_facets(
        stat_var=request.stat_var.removeprefix("dcid:"),
        bound_entities=(),
        settle=True,
        graph=graph,
        availability=graph.scope_availability(),
        declared_roles=request.declared_roles,
        role_bindings=request.role_bindings,
        coverage=coverage,
    )


@pytest.mark.parametrize("order", _ORDERS)
def test_each_unbound_member_gets_the_candidates_it_gets_alone(order):
    coverage = _coverage_for([(_BARE_A, "facet1")])
    requests = [_MEMBERS[spec_id]() for spec_id in order]

    graph = _graph()
    probed = probe_collection_candidates(
        requests,
        settle=True,
        graph=graph,
        availability=graph.scope_availability(),
        coverage=coverage,
    )

    for index, request in enumerate(requests):
        # The same probe: its own places, candidates and admission for a subject-only member,
        # coverage for the role-bearing one.
        alone = _alone(request, _graph(), coverage)
        assert alone.candidates != ()
        assert probed[index] == alone
    assert [type(probed[order.index(spec_id)].admission) for spec_id in _MEMBERS] == [
        CoverageAdmission,
        PlaceAdmission,
        PlaceAdmission,
    ]
    # The role-bearing member has no place scope; coverage answers it. Each subject-only
    # member is fetched at every place it holds data at.
    assert {spec_id: probed[order.index(spec_id)].places for spec_id in _MEMBERS} == {
        _ROLE_SPEC: (),
        _COUNTRY_SPEC: (_KEN, "Earth"),
        _WORLD_SPEC: ("Earth",),
    }


@pytest.mark.parametrize("order", [(_ROLE_SPEC, _COUNTRY_SPEC), (_COUNTRY_SPEC, _ROLE_SPEC)])
def test_a_role_bearing_member_beside_an_unbound_subject_only_one_needs_coverage(order):
    requests = [_MEMBERS[spec_id]() for spec_id in order]

    with pytest.raises(MaterialisationUsageError, match="require a coverage index"):
        probe_collection_candidates(
            requests, settle=True, graph=_graph(), availability=_graph().scope_availability()
        )


def test_members_are_probed_and_admitted_at_their_own_places():
    requests = [_subject_only(_BARE_B), _subject_only(_BARE_C)]

    graph = _graph()
    probed = probe_collection_candidates(
        requests, settle=True, graph=graph, availability=graph.scope_availability()
    )

    # Kenya and Earth's total both stay in the first member's scope. The worldwide-total
    # member's scope is Earth alone.
    facet_id = _SHARED_FACET.facet_id
    assert probed == (
        MemberProbe(
            # Kenya's and Earth's counts merge under the one facet id.
            candidates=(dataclasses.replace(_SHARED_FACET, obs_count=2, annual_evidence=True),),
            admission=PlaceAdmission(places_by_facet={facet_id: frozenset({_KEN, "Earth"})}),
            places=(_KEN, "Earth"),
        ),
        MemberProbe(
            candidates=(_SHARED_FACET,),
            admission=PlaceAdmission(places_by_facet={facet_id: frozenset({"Earth"})}),
            places=("Earth",),
        ),
    )


def test_unbound_members_fetch_together_only_when_the_probe_gave_them_the_same_places():
    requests = [
        (spec_id, _subject_only(bare))
        for spec_id, bare in (("s0", _BARE_A), ("s1", _BARE_B), ("s2", _BARE_C))
    ]

    chunks = _plan_collection_chunks(requests, places=[(_KEN, _TGO), ("Earth",), (_KEN, _TGO)])

    assert [(chunk.bare_stat_vars, chunk.role_bindings) for chunk in chunks] == [
        ((_BARE_A, _BARE_C), {_ABOUT: (_KEN, _TGO)}),
        ((_BARE_B,), {_ABOUT: ("Earth",)}),
    ]


def _sdmx() -> tuple[SdmxClient, list[httpx.Request]]:
    """Serves one row per subject-only variable at each named place it holds data at, and
    one role-bearing row for `_BARE_A`."""
    recorded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        availability = ScopeAvailability(data=_SUBJECT_DATA)
        if (response := availability.respond(request)) is not None:
            return response
        recorded.append(request)
        named = str(request.url.params["c[variableMeasured]"]).split(",")
        places = request.url.params.get(f"c[{_ABOUT}]")
        if places is None:
            text = _CSV_HEADER + "".join(
                _csv_row(
                    stat_var=bare,
                    year="2015",
                    value="10",
                    unit="UNIT1",
                    method="METH1",
                    provenance="PROV1",
                    facet_id="facet1",
                )
                for bare in named
            )
        else:
            text = _ABOUT_HEADER + "".join(
                f"dataflow,DC:DF_OBS(1.0.0),I,{bare},{place},UNIT1,METH1,NotApplicable,PROV1,"
                f"2015,10,,facet1\r\n"
                for bare in named
                for place in places.split(",")
                if (bare, place) in _SUBJECT_DATA
            )
        return httpx.Response(
            200, text=text, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    return SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    ), recorded


_SERVICE = {"materialiser_build": "build-123", "capability_version": "cap-v1"}


def _group_of_country_and_world_members(*, probe=None):
    """A group of `_BARE_B`, with Kenya data and a worldwide total, fetched at Kenya and Earth,
    and `_BARE_C`, a worldwide total alone, fetched at Earth."""
    graph = _graph()
    sdmx, _recorded = _sdmx()
    return materialise_group(
        [_subject_only(_BARE_B), _subject_only(_BARE_C)],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        precomputed_probe=probe,
        **_SERVICE,
    )


def test_a_group_reusing_a_probe_mints_what_it_mints_fresh():
    requests = [_subject_only(_BARE_B), _subject_only(_BARE_C)]
    graph = _graph()
    probe = probe_collection_facets(
        requests, settle=True, graph=graph, availability=graph.scope_availability()
    )

    reused = _group_of_country_and_world_members(probe=probe)
    fresh = _group_of_country_and_world_members()

    assert reused.coverage.rows == fresh.coverage.rows == 3
    assert reused.content_version == fresh.content_version


def _rows_by_member(result) -> dict[str, list[tuple[object, ...]]]:
    """Each StatVar's rows in `result`'s table, each row without its StatVar column."""
    table = ConcatenatedFrameChunks.from_chunks(result.frame_chunks())
    column = table.columns.index("stat_var")
    rows: dict[str, list[tuple[object, ...]]] = {}
    for row in table.rows:
        rest = tuple(value for index, value in enumerate(row) if index != column)
        rows.setdefault(str(row[column]).removeprefix("dcid:"), []).append(rest)
    return {stat_var: sorted(member_rows, key=repr) for stat_var, member_rows in rows.items()}


def _mint(mint, requests):
    graph = _graph()
    sdmx, _recorded = _sdmx()
    return mint(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        **_SERVICE,
    )


@pytest.mark.parametrize("order", list(itertools.permutations([_BARE_B, _BARE_C])))
def test_a_member_returns_the_rows_it_returns_alone_in_a_group(order):
    # `_BARE_B` holds Kenya data and a worldwide total; `_BARE_C` a worldwide total alone.
    # Each returns its own places' rows whatever its companions hold.
    alone = {
        bare: _rows_by_member(_mint(materialise_spec, _subject_only(bare)))[bare] for bare in order
    }
    group = _rows_by_member(_mint(materialise_group, [_subject_only(bare) for bare in order]))

    assert group == alone
    assert {_KEN in repr(row) for row in alone[_BARE_B]} == {True, False}
    assert alone[_BARE_C] and all("Earth" in repr(row) for row in alone[_BARE_C])


_DFO_PROV1 = FakeCatalogue(
    provenances={"PROV1": provenance_row("PROV1", DFO)},
    dfo_variables=frozenset({_BARE_B, _BARE_C}),
)


@pytest.mark.parametrize("unsettled", [False, True], ids=["settled", "unsettled"])
@pytest.mark.parametrize("mint", [materialise_spec, materialise_group], ids=["member", "group"])
def test_a_run_whose_default_scope_this_request_alone_took_keys_on_no_import(
    monkeypatch, bind_catalogue, mint, unsettled
):
    """A countries fallback this request alone took (`DefaultPresence.unsettled`) is a scope a
    later request can answer differently, so its run is keyed on no import and its head keeps
    the day limit."""
    read = facets_module.read_default_presence

    def _read(**kwargs):
        presence = read(**kwargs)
        held = frozenset(presence.scopes) if unsettled else frozenset()
        return dataclasses.replace(presence, unsettled=held)

    monkeypatch.setattr(facets_module, "read_default_presence", _read)
    bind_catalogue(_DFO_PROV1)
    requests = (
        _subject_only(_BARE_B)
        if mint is materialise_spec
        else [_subject_only(_BARE_B), _subject_only(_BARE_C)]
    )

    result = _mint(mint, requests)

    assert result.coverage.rows
    expected = None if unsettled else current_release().imported_data
    assert result.dependencies.imported_data == expected


# Twenty StatVars, each with data at Kenya, Togo and Earth, so every place and the countries
# fallback (Kenya and Togo) differ for each of them.
_WIDE = [f"Count_Wide_{index:02d}" for index in range(20)]
_WIDE_DATA = tuple((bare, place) for bare in _WIDE for place in (_KEN, _TGO, "Earth"))


def _wide_graph() -> FakeGraphReader:
    return FakeGraphReader(
        facets={key: [_SHARED_FACET] for key in _WIDE_DATA},
        child_places={("Earth", "Country"): [_KEN, _TGO]},
    )


def _wide_sdmx() -> SdmxClient:
    availability = ScopeAvailability(data=_WIDE_DATA)

    def handler(request: httpx.Request) -> httpx.Response:
        if (response := availability.respond(request)) is not None:
            return response
        named = str(request.url.params["c[variableMeasured]"]).split(",")
        places = str(request.url.params[f"c[{_ABOUT}]"]).split(",")
        text = _ABOUT_HEADER + "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{bare},{place},UNIT1,METH1,NotApplicable,PROV1,"
            f"2015,10,,facet1\r\n"
            for bare in named
            for place in places
            if (bare, place) in _WIDE_DATA
        )
        return httpx.Response(
            200, text=text, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    return SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )


def _wide_mint(graph: FakeGraphReader) -> tuple[str, dict[str, int]]:
    """The group's `content_version`, and each member's row count."""
    result = materialise_group(
        [_subject_only(bare) for bare in _WIDE],
        sdmx=_wide_sdmx(),
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        **_SERVICE,
    )
    return result.content_version, {
        stat_var: len(rows) for stat_var, rows in _rows_by_member(result).items()
    }


def test_a_cold_mint_past_a_busy_settle_pool_mints_what_a_warm_re_mint_mints(monkeypatch):
    # Four settle threads for twenty cold StatVars. Search, which does not settle, takes the
    # countries fallback for the sixteen the pool cannot start; the mint waits for every read.
    monkeypatch.setattr(place_scope, "AVAILABILITY_WORKERS", 4)
    requests = [_subject_only(bare) for bare in _WIDE]
    searched = probe_collection_candidates(
        requests, settle=False, graph=_wide_graph(), availability=ScopeAvailability(data=_WIDE_DATA)
    )
    assert sum(not member.every_place for member in searched) == 16

    graph = _wide_graph()
    cold = _wide_mint(graph)
    warm = _wide_mint(graph)

    assert cold == warm
    # Kenya, Togo and Earth: every member at every place it holds data at.
    assert set(cold[1]) == set(_WIDE)
    assert set(cold[1].values()) == {3}
    probed = probe_collection_candidates(
        requests, settle=True, graph=graph, availability=ScopeAvailability(data=_WIDE_DATA)
    )
    assert {(member.places, member.every_place) for member in probed} == {
        ((_KEN, _TGO, "Earth"), True)
    }
