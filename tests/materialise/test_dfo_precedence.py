"""Facet selection ranks DFO's facet first for a variable in the origin catalogue's overlap set,
and a fetched row serves only where its facet was admitted at its place, so a base facet never
serves a place where DFO holds data, even when a binding selects it; elsewhere ranking is
unchanged."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from urllib.parse import unquote

import httpx
import pytest

from dc_sdmx import SdmxClient
from dc_sdmx.errors import SdmxResponseError
from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    materialise_collection,
    materialise_group,
    materialise_preview,
)
from materialise.facets import (
    FacetBinding,
    UnitRequest,
    declared_default_policies,
    probe_facets,
    rank_candidates,
    select_facet,
)
from materialise.generation import _select_member_facets
from materialise.graph import Facet
from materialise.group_facets import select_group_facets
from materialise.revision import RevisionSnapshot
from materialise.window import RequestedWindow
from origin_catalogue import BASE, DFO, Catalogue
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

from ._scope_availability import NO_SCOPE_READ
from .conftest import FakeGraphReader, FakeRevisionReader, only_chunk

_VARIABLE = "Count_Person"
_OTHER = "Count_Household"
_P1, _P2 = "country/P1", "country/P2"
_DFO_PROVENANCE = "provenance/DfoImport"
_BASE_PROVENANCE = "dc/base/WorldDevelopmentIndicators"

# Admission already applied per place: DFO's facet alone at P1, the base facet alone at P2.
_DFO_FACET = Facet(
    facet_id="fD",
    provenance_id=_DFO_PROVENANCE,
    measurement_method="Census",
    unit="CurrencyUSD",
    obs_count=5,
    earliest_date="2019",
    latest_date="2023",
)
_BASE_FACET = Facet(
    facet_id="fB",
    provenance_id=_BASE_PROVENANCE,
    measurement_method="Census",
    unit="CurrencyUSD",
    obs_count=60,
    earliest_date="1960",
    latest_date="2023",
)

_SDMX_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


@pytest.fixture
def overlap(bind_catalogue: Callable[[Catalogue], None]) -> None:
    """`_VARIABLE` is held by DFO and by an admitted base provenance; `_OTHER` by base alone."""
    bind_catalogue(
        restrictive_catalogue(
            provenances=[
                provenance_row(_DFO_PROVENANCE, DFO),
                provenance_row(_BASE_PROVENANCE, BASE),
            ],
            dfo_variables=[_VARIABLE],
            base_variables={
                _VARIABLE: ([_BASE_PROVENANCE], []),
                _OTHER: ([_BASE_PROVENANCE], []),
            },
        )
    )


def _graph(stat_var: str) -> FakeGraphReader:
    return FakeGraphReader(facets={(stat_var, _P1): [_DFO_FACET], (stat_var, _P2): [_BASE_FACET]})


def _candidates(stat_var: str) -> tuple[Facet, ...]:
    return probe_facets(
        availability=NO_SCOPE_READ,
        stat_var=stat_var,
        bound_entities=[_P1, _P2],
        settle=True,
        graph=_graph(stat_var),
    ).candidates


_ALL = RequestedWindow(kind="all")


def _request(
    stat_var: str, window: RequestedWindow = _ALL, **facet_bindings: FacetBinding
) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=f"dcid:{stat_var}",
        role_bindings={"observationAbout": [_P1, _P2]},
        declared_roles=("observationAbout",),
        facet_bindings=facet_bindings,
        window=window,
    )


@pytest.mark.usefixtures("overlap")
def test_dfo_data_at_any_place_selects_dfo_facet_over_more_base_observations():
    selections = _select_member_facets(
        _request(_VARIABLE), _candidates(_VARIABLE), graph=_graph(_VARIABLE)
    )

    provenance = selections["provenance"]
    assert provenance.ref == _DFO_PROVENANCE
    assert provenance.dispatch_facet_id == "fD"
    assert provenance.selection_reason == "dfo_precedence"
    # Both candidates share the unit, so its axis reason stays the count ranking's.
    assert selections["unit"].selection_reason == "most_observations"


@pytest.mark.usefixtures("overlap")
def test_a_variable_outside_the_overlap_set_keeps_the_count_ranking():
    selection = select_facet(
        "provenance",
        FacetBinding(state="unbound"),
        candidates=_candidates(_OTHER),
        graph=_graph(_OTHER),
        stat_var=f"dcid:{_OTHER}",
        window=None,
        policies=declared_default_policies(
            {"provenance": FacetBinding(state="unbound")},
            graph=_graph(_OTHER),
            stat_var=f"dcid:{_OTHER}",
        ),
    )

    assert selection.ref == _BASE_PROVENANCE
    assert selection.selection_reason == "most_observations_over_disagreement"


def test_the_permissive_test_catalogue_leaves_ranking_unchanged():
    # The suite-wide `FakeCatalogue` lists every variable under one origin, so no fixture's
    # selection, and so no content version, moves.
    assert _select_member_facets(
        _request(_VARIABLE), _candidates(_VARIABLE), graph=_graph(_VARIABLE)
    )["provenance"].ref == (_BASE_PROVENANCE)


def test_an_empty_precedence_is_the_count_ranking():
    candidates = [_DFO_FACET, _BASE_FACET]

    assert rank_candidates(candidates, window=None) == (_BASE_FACET, _DFO_FACET)
    assert rank_candidates(candidates, window=None, precedence=frozenset({_DFO_PROVENANCE})) == (
        _DFO_FACET,
        _BASE_FACET,
    )


@pytest.mark.usefixtures("overlap")
def test_a_window_only_the_base_facet_reaches_serves_the_base_facet():
    # Narrowing runs before precedence: DFO's facet starts in 2019, so a 1970s window leaves
    # only the base facet, where precedence first would serve an empty DFO series.
    selection = select_facet(
        "provenance",
        FacetBinding(state="unbound"),
        candidates=_candidates(_VARIABLE),
        graph=_graph(_VARIABLE),
        stat_var=f"dcid:{_VARIABLE}",
        window=RequestedWindow(kind="range", start_year=1970, end_year=1979),
        policies=declared_default_policies(
            {"provenance": FacetBinding(state="unbound")},
            graph=_graph(_VARIABLE),
            stat_var=f"dcid:{_VARIABLE}",
        ),
    )

    assert selection.ref == _BASE_PROVENANCE
    assert selection.selection_reason == "sole_candidate"


@pytest.mark.usefixtures("overlap")
def test_a_currency_request_prefers_dfo_among_native_currency_candidates():
    selection = select_facet(
        "unit",
        FacetBinding(state="request", request=UnitRequest(currency="USD", prices="current")),
        candidates=_candidates(_VARIABLE),
        graph=_graph(_VARIABLE),
        stat_var=f"dcid:{_VARIABLE}",
        window=None,
        policies=declared_default_policies(
            {
                "unit": FacetBinding(
                    state="request", request=UnitRequest(currency="USD", prices="current")
                )
            },
            graph=_graph(_VARIABLE),
            stat_var=f"dcid:{_VARIABLE}",
        ),
    )

    assert selection.dispatch_facet_id == "fD"


@pytest.mark.usefixtures("overlap")
def test_a_group_with_an_overlap_member_shares_dfo_facet():
    candidates = [[_DFO_FACET, _BASE_FACET], [_DFO_FACET, _BASE_FACET]]

    selections = select_group_facets(
        [_request(_VARIABLE), _request(_OTHER)], candidates, graph=FakeGraphReader()
    )

    assert selections["provenance"].ref == _DFO_PROVENANCE
    assert selections["provenance"].selection_reason == "dfo_precedence"


@pytest.mark.usefixtures("overlap")
@pytest.mark.parametrize(
    ("window", "provenance", "reason"),
    [
        # Both facets reach 2020-2023, so precedence elects DFO's over the base's larger count.
        (
            RequestedWindow(kind="range", start_year=2020, end_year=2023),
            _DFO_PROVENANCE,
            "dfo_precedence",
        ),
        # DFO's facet starts in 2019, so a 1970s window leaves the base facet alone.
        (
            RequestedWindow(kind="range", start_year=1970, end_year=1979),
            _BASE_PROVENANCE,
            "sole_candidate",
        ),
    ],
    ids=["both_overlap", "base_only"],
)
def test_a_group_narrows_to_the_window_before_precedence(window, provenance, reason):
    candidates = [[_DFO_FACET, _BASE_FACET], [_DFO_FACET, _BASE_FACET]]

    selections = select_group_facets(
        [_request(_VARIABLE, window), _request(_OTHER, window)],
        candidates,
        graph=FakeGraphReader(),
    )

    assert selections["provenance"].ref == provenance
    assert selections["provenance"].selection_reason == reason


@pytest.mark.usefixtures("overlap")
def test_a_preview_fetches_only_dfo_facet_and_serves_nothing_at_base_only_places():
    fetched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetched.append(unquote(str(request.url)))
        body = _SDMX_HEADER + (
            f"dataflow,DC:DF_OBS(1.0.0),I,{_VARIABLE},{_P1},CurrencyUSD,Census,P1Y,"
            f"{_DFO_PROVENANCE},2023,10,,fD\r\n"
        )
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    sdmx = SdmxClient(base_url="https://ours.example.org", transport=httpx.MockTransport(handler))

    outcome = materialise_preview(_request(_VARIABLE), sdmx=sdmx, graph=_graph(_VARIABLE), limit=10)

    assert outcome.selections["provenance"].ref == _DFO_PROVENANCE
    assert fetched and all("fD" in url and "fB" not in url for url in fetched)
    assert outcome.coverage.entity_refs["observationAbout"] == frozenset({_P1})


# The SDMX service knows nothing of admission: it holds the base facet at both places for every
# variable it is asked for, and DFO's facet at P1.
def _admission_blind_sdmx(fetched: list[str] | None = None) -> SdmxClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if fetched is not None:
            fetched.append(unquote(str(request.url)))
        variables = str(request.url.params.get("c[variableMeasured]")).split(",")
        facet_ids = str(request.url.params.get("c[facetId]")).split(",")
        rows = []
        for variable in variables:
            if "fB" in facet_ids:
                rows += [
                    f"dataflow,DC:DF_OBS(1.0.0),I,{variable},{place},CurrencyUSD,Census,P1Y,"
                    f"{_BASE_PROVENANCE},2023,99,,fB\r\n"
                    for place in (_P1, _P2)
                ]
            if "fD" in facet_ids:
                rows.append(
                    f"dataflow,DC:DF_OBS(1.0.0),I,{variable},{_P1},CurrencyUSD,Census,P1Y,"
                    f"{_DFO_PROVENANCE},2023,10,,fD\r\n"
                )
        return httpx.Response(
            200,
            text=_SDMX_HEADER + "".join(rows),
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(base_url="https://ours.example.org", transport=httpx.MockTransport(handler))


_BOUND_BASE = FacetBinding(state="bound", ref=_BASE_PROVENANCE)


@pytest.mark.usefixtures("overlap")
def test_a_bound_base_provenance_serves_no_row_where_admission_withheld_its_facet():
    outcome = materialise_preview(
        _request(_VARIABLE, provenance=_BOUND_BASE),
        sdmx=_admission_blind_sdmx(),
        graph=_graph(_VARIABLE),
        limit=10,
    )

    assert outcome.selections["provenance"].ref == _BASE_PROVENANCE
    # Admission withheld fB at P1 in favour of fD, so only P2 serves.
    assert outcome.coverage.entity_refs["observationAbout"] == frozenset({_P2})


@pytest.mark.usefixtures("overlap")
def test_a_bound_unit_only_the_base_facet_carries_serves_no_row_where_admission_withheld_it():
    graph = FakeGraphReader(
        facets={
            (_VARIABLE, _P1): [replace(_DFO_FACET, unit="CurrencyEUR")],
            (_VARIABLE, _P2): [_BASE_FACET],
        }
    )

    outcome = materialise_preview(
        _request(_VARIABLE, unit=FacetBinding(state="bound", ref="CurrencyUSD")),
        sdmx=_admission_blind_sdmx(),
        graph=graph,
        limit=10,
    )

    assert outcome.selections["provenance"].ref == _BASE_PROVENANCE
    assert outcome.coverage.entity_refs["observationAbout"] == frozenset({_P2})


@pytest.mark.usefixtures("overlap")
def test_an_unbound_request_against_the_same_service_serves_dfo_at_p1_only():
    outcome = materialise_preview(
        _request(_VARIABLE), sdmx=_admission_blind_sdmx(), graph=_graph(_VARIABLE), limit=10
    )

    assert outcome.selections["provenance"].ref == _DFO_PROVENANCE
    assert outcome.coverage.entity_refs["observationAbout"] == frozenset({_P1})


@pytest.mark.usefixtures("overlap")
def test_a_response_without_a_facet_id_column_fails_instead_of_serving_nothing():
    def handler(request: httpx.Request) -> httpx.Response:
        body = _SDMX_HEADER.replace(",facetId", "") + (
            f"dataflow,DC:DF_OBS(1.0.0),I,{_VARIABLE},{_P2},CurrencyUSD,Census,P1Y,"
            f"{_BASE_PROVENANCE},2023,99,\r\n"
        )
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    sdmx = SdmxClient(base_url="https://ours.example.org", transport=httpx.MockTransport(handler))

    with pytest.raises(SdmxResponseError, match="carry no facetId"):
        materialise_preview(
            _request(_VARIABLE, provenance=_BOUND_BASE),
            sdmx=sdmx,
            graph=_graph(_VARIABLE),
            limit=10,
        )


def _sdmx_serving(*rows: str) -> SdmxClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_SDMX_HEADER + "".join(rows),
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(base_url="https://ours.example.org", transport=httpx.MockTransport(handler))


def _base_row(place: str, year: str, *, variable: str = _VARIABLE, facet_id: str = "fB") -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{variable},{place},CurrencyUSD,Census,P1Y,"
        f"{_BASE_PROVENANCE},{year},1,,{facet_id}\r\n"
    )


@pytest.mark.usefixtures("overlap")
def test_a_withheld_row_does_not_set_the_latest_period():
    # fB is withheld at P1, where it holds the newest period, and admitted at P2.
    outcome = materialise_preview(
        replace(_request(_VARIABLE, provenance=_BOUND_BASE), window=RequestedWindow(kind="latest")),
        sdmx=_sdmx_serving(_base_row(_P1, "2024"), _base_row(_P2, "2023")),
        graph=_graph(_VARIABLE),
        limit=10,
    )

    assert outcome.coverage.entity_refs["observationAbout"] == frozenset({_P2})
    assert outcome.coverage.entity_refs["years"] == frozenset({2023})


@pytest.mark.usefixtures("overlap")
@pytest.mark.parametrize("place", ["NotApplicable", ""])
def test_a_subject_row_without_a_place_fails_instead_of_being_dropped(place: str):
    no_place = _base_row(place, "2023")

    with pytest.raises(SdmxResponseError, match="carry no observationAbout"):
        materialise_preview(
            _request(_VARIABLE, provenance=_BOUND_BASE),
            sdmx=_sdmx_serving(no_place, _base_row(_P2, "2023")),
            graph=_graph(_VARIABLE),
            limit=10,
        )


_SNAPSHOT = RevisionSnapshot(graph_revision="graph-rev", source_revisions=())


def _two_variable_graph() -> FakeGraphReader:
    """`_VARIABLE` as in `_graph`; `_OTHER` holds the base facet, admitted at both places."""
    return FakeGraphReader(
        facets={
            (_VARIABLE, _P1): [_DFO_FACET],
            (_VARIABLE, _P2): [_BASE_FACET],
            (_OTHER, _P1): [_BASE_FACET],
            (_OTHER, _P2): [_BASE_FACET],
        }
    )


@pytest.mark.usefixtures("overlap")
def test_a_collection_chunk_serves_each_member_only_where_its_own_probe_admitted_the_facet():
    fetched: list[str] = []
    graph = _two_variable_graph()

    result = materialise_collection(
        [
            ("spec_" + "0" * 32, _request(_VARIABLE, provenance=_BOUND_BASE)),
            ("spec_" + "1" * 32, _request(_OTHER, provenance=_BOUND_BASE)),
        ],
        sdmx=_admission_blind_sdmx(fetched),
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        # One lane keeps both same-scope members in one shared chunk fetch.
        member_concurrency=1,
    )

    assert len(fetched) == 1
    (_, overlap), (_, other) = result.members
    assert overlap.coverage.entity_refs["observationAbout"] == frozenset({_P2})
    assert other.coverage.entity_refs["observationAbout"] == frozenset({_P1, _P2})


@pytest.mark.usefixtures("overlap")
def test_a_group_fetch_serves_each_member_only_where_its_own_probe_admitted_the_facet():
    graph = _two_variable_graph()

    result = materialise_group(
        [_request(_VARIABLE, provenance=_BOUND_BASE), _request(_OTHER, provenance=_BOUND_BASE)],
        sdmx=_admission_blind_sdmx(),
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=1,
    )

    table = only_chunk(result).table
    served = set(
        zip(table["stat_var"].to_pylist(), table["observationAbout"].to_pylist(), strict=True)
    )
    assert served == {(f"dcid:{_VARIABLE}", _P2), (f"dcid:{_OTHER}", _P1), (f"dcid:{_OTHER}", _P2)}


@pytest.mark.usefixtures("overlap")
def test_a_single_member_mint_serves_no_row_where_admission_withheld_its_facet():
    graph = _graph(_VARIABLE)

    result = materialise_group(
        [_request(_VARIABLE, provenance=_BOUND_BASE)],
        sdmx=_admission_blind_sdmx(),
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
    )

    assert set(only_chunk(result).table["observationAbout"].to_pylist()) == {_P2}


def _latest(request: MaterialisationRequest) -> MaterialisationRequest:
    return replace(request, window=RequestedWindow(kind="latest"))


# Both variables hold fB at P1 for 2024 and at P2 for 2023. Admission withheld fB at P1 for
# `_VARIABLE` alone.
_NEWEST_AT_P1 = tuple(
    _base_row(place, year, variable=variable)
    for variable in (_VARIABLE, _OTHER)
    for place, year in ((_P1, "2024"), (_P2, "2023"))
)


@pytest.mark.usefixtures("overlap")
def test_a_withheld_row_does_not_set_a_collection_member_latest_period():
    graph = _two_variable_graph()

    result = materialise_collection(
        [
            ("spec_" + "0" * 32, _latest(_request(_VARIABLE, provenance=_BOUND_BASE))),
            ("spec_" + "1" * 32, _latest(_request(_OTHER, provenance=_BOUND_BASE))),
        ],
        sdmx=_sdmx_serving(*_NEWEST_AT_P1),
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=1,
    )

    (_, overlap), (_, other) = result.members
    assert overlap.coverage.entity_refs["observationAbout"] == frozenset({_P2})
    assert overlap.coverage.entity_refs["years"] == frozenset({2023})
    assert other.coverage.entity_refs["observationAbout"] == frozenset({_P1})
    assert other.coverage.entity_refs["years"] == frozenset({2024})


@pytest.mark.usefixtures("overlap")
def test_a_withheld_row_does_not_set_a_group_member_latest_period():
    graph = _two_variable_graph()

    result = materialise_group(
        [
            _latest(_request(_VARIABLE, provenance=_BOUND_BASE)),
            _latest(_request(_OTHER, provenance=_BOUND_BASE)),
        ],
        sdmx=_sdmx_serving(*_NEWEST_AT_P1),
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=1,
    )

    table = only_chunk(result).table
    served = set(
        zip(
            table["stat_var"].to_pylist(),
            table["observationAbout"].to_pylist(),
            table["time_period"].to_pylist(),
            strict=True,
        )
    )
    assert served == {(f"dcid:{_VARIABLE}", _P2, "2023"), (f"dcid:{_OTHER}", _P1, "2024")}


# A row that names no place, or no facet, after rows that serve.
_MALFORMED_AFTER_GOOD = {
    "observationAbout": _base_row("NotApplicable", "2023", variable=_OTHER),
    "facetId": _base_row(_P2, "2023", variable=_OTHER, facet_id=""),
}


@pytest.mark.usefixtures("overlap")
@pytest.mark.parametrize("missing", sorted(_MALFORMED_AFTER_GOOD))
def test_a_malformed_row_mid_stream_fails_a_collection_fetch(missing: str):
    graph = _two_variable_graph()
    good = [_base_row(_P2, "2023", variable=v) for v in (_VARIABLE, _OTHER)]

    with pytest.raises(SdmxResponseError, match=f"carry no {missing}"):
        materialise_collection(
            [
                ("spec_" + "0" * 32, _request(_VARIABLE, provenance=_BOUND_BASE)),
                ("spec_" + "1" * 32, _request(_OTHER, provenance=_BOUND_BASE)),
            ],
            sdmx=_sdmx_serving(*good, _MALFORMED_AFTER_GOOD[missing]),
            graph=graph,
            enrichment=EnrichmentLookup(graph),
            revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
            materialiser_build="build-123",
            capability_version="cap-v1",
            member_concurrency=1,
        )


@pytest.mark.usefixtures("overlap")
@pytest.mark.parametrize("missing", sorted(_MALFORMED_AFTER_GOOD))
def test_a_malformed_row_mid_stream_fails_a_group_fetch(missing: str):
    graph = _two_variable_graph()
    good = [_base_row(_P2, "2023", variable=v) for v in (_VARIABLE, _OTHER)]

    with pytest.raises(SdmxResponseError, match=f"carry no {missing}"):
        materialise_group(
            [_request(_VARIABLE, provenance=_BOUND_BASE), _request(_OTHER, provenance=_BOUND_BASE)],
            sdmx=_sdmx_serving(*good, _MALFORMED_AFTER_GOOD[missing]),
            graph=graph,
            enrichment=EnrichmentLookup(graph),
            revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
            materialiser_build="build-123",
            capability_version="cap-v1",
            member_concurrency=1,
        )


@pytest.mark.usefixtures("overlap")
def test_rows_for_a_member_whose_probe_admitted_nothing_serve_nothing():
    """Admission withheld every facet of `_VARIABLE` at both places, so its probe is empty and it
    dispatches nothing. The shared fetch still filters on fB for `_OTHER`, and the service
    returns `_VARIABLE`'s fB rows under that same filter; they do not serve."""
    graph = FakeGraphReader(facets={(_OTHER, _P1): [_BASE_FACET], (_OTHER, _P2): [_BASE_FACET]})

    result = materialise_group(
        [_request(_VARIABLE, provenance=_BOUND_BASE), _request(_OTHER, provenance=_BOUND_BASE)],
        sdmx=_admission_blind_sdmx(),
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=1,
    )

    table = only_chunk(result).table
    served = set(
        zip(table["stat_var"].to_pylist(), table["observationAbout"].to_pylist(), strict=True)
    )
    assert served == {(f"dcid:{_OTHER}", _P1), (f"dcid:{_OTHER}", _P2)}


@pytest.mark.usefixtures("overlap")
@pytest.mark.parametrize("missing", sorted(_MALFORMED_AFTER_GOOD))
def test_a_malformed_row_for_a_member_whose_probe_admitted_nothing_refuses(missing: str):
    graph = FakeGraphReader(facets={(_OTHER, _P1): [_BASE_FACET], (_OTHER, _P2): [_BASE_FACET]})
    malformed = _MALFORMED_AFTER_GOOD[missing].replace(f",{_OTHER},", f",{_VARIABLE},")

    with pytest.raises(SdmxResponseError, match=f"carry no {missing}"):
        materialise_group(
            [_request(_VARIABLE, provenance=_BOUND_BASE), _request(_OTHER, provenance=_BOUND_BASE)],
            sdmx=_sdmx_serving(_base_row(_P2, "2023", variable=_OTHER), malformed),
            graph=graph,
            enrichment=EnrichmentLookup(graph),
            revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
            materialiser_build="build-123",
            capability_version="cap-v1",
            member_concurrency=1,
        )
