"""The bulk role preview and the per-pair preview elect the same facet for one scope.

Both rank with `materialise.facets.rank_pool`. The bulk read leads with the number of role
pairs a facet covers, after `dfo_precedence`, so where every candidate covers the same pairs
the two paths agree pair by pair.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from types import MappingProxyType

import httpx
import pytest

from code_index.types import StatVarRow
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dc_sdmx import SdmxObservation
from materialise import MaterialisationRequest
from materialise.facets import probe_facets
from materialise.generation import _select_member_facets
from materialise.graph import Facet as MaterialiseFacet
from materialise.window import RequestedWindow
from origin_catalogue import BASE, DFO, Catalogue
from search.engine.clients.graph import Facet
from search.engine.extract import DateRequest
from search.engine.region import resolve_variable
from search.engine.region.codes import OBSERVATIONS_SCOPE_PARTIAL
from search.engine.region.state import EngineOptions
from search.engine.retrieval import role_resolve
from search.engine.retrieval.member_facts import IndexMemberFacts
from search.engine.retrieval.read_plan import PairPlan, PlanSeries, in_elected_datasets, plan_pairs
from search.models import Spec, TimeWindow
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests.engine import test_latest_projection_facet as latest_projection
from tests.engine import test_place_scope_default as place_scope
from tests.engine import test_role_bearing_pipeline_e2e as role_bearing
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    _SDMX_CSV_CONTENT_TYPE,
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    bulk_plan,
)
from tests.fixtures import make_resolver
from tests.materialise._scope_availability import NO_SCOPE_READ
from tests.materialise.conftest import FakeGraphReader
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

_SV = "TESTFIXTURE/ElectionAgreement/Flow"
_DONOR = "TESTFIXTURE/ElectionAgreement/UnboundDonor"
_KEN, _NOR = "country/KEN", "country/NOR"
_ROLES = ("creditor", "debtor")
_ROLE_NAMES = {_SV: RoleNames(donor="creditor", recipient="debtor")}
_DFO_PROVENANCE = "provenance/DfoImport"
_BASE_PROVENANCE = "dc/base/WorldBankIDS"


def _rows(facets: dict[str, list[int]]) -> list:
    return [
        (_SV, facet_id, _DONOR, recipient, years)
        for facet_id, years in facets.items()
        for recipient in (_KEN, _NOR)
    ]


def _elections(
    facets: dict[str, list[int]], metadata: dict[str, FacetMetadata]
) -> tuple[str, dict[str, str]]:
    """The bulk preview's facet for the Kenya-and-Norway scope, and each pair's facet from
    materialise's per-pair selection under a `latest` window."""
    index = build_coverage_index(_rows(facets), role_names=_ROLE_NAMES, facet_metadata=metadata)
    bulk = bulk_plan(
        [_SV], {"debtor": [_KEN, _NOR]}, coverage=FakeCoverageService(index).snapshot()
    ).elected
    assert bulk is not None
    per_pair: dict[str, str] = {}
    for recipient in (_KEN, _NOR):
        request = MaterialisationRequest(
            stat_var=f"dcid:{_SV}",
            role_bindings={"debtor": [recipient]},
            declared_roles=_ROLES,
            facet_bindings={},
            window=RequestedWindow(kind="latest"),
        )
        probe = probe_facets(
            availability=NO_SCOPE_READ,
            stat_var=_SV,
            bound_entities=[recipient],
            settle=False,
            graph=FakeGraphReader(),
            declared_roles=_ROLES,
            role_bindings=request.role_bindings,
            coverage=index,
        )
        selection = _select_member_facets(request, probe.candidates, graph=FakeGraphReader())
        dispatch_facet_id = selection["provenance"].dispatch_facet_id
        assert dispatch_facet_id is not None
        per_pair[recipient] = dispatch_facet_id
    return bulk.facet_id, per_pair


def test_a_recorded_facet_wins_on_both_paths_over_a_projection_covering_the_same_pairs():
    bulk, per_pair = _elections(
        {"f-rec": list(range(2010, 2025)), "f-est": list(range(2025, 2033))},
        {
            "f-rec": FacetMetadata(provenance_id=_BASE_PROVENANCE),
            "f-est": FacetMetadata(provenance_id=_BASE_PROVENANCE, measurement_method="Estimated"),
        },
    )

    assert bulk == "f-rec"
    assert per_pair == {_KEN: "f-rec", _NOR: "f-rec"}


def test_dfo_precedence_elects_the_dfo_facet_on_both_paths(
    bind_catalogue: Callable[[Catalogue], None],
):
    """The base facet covers the same pairs with more observations. Precedence ranks DFO's
    first on the per-pair path, and the bulk path ranks the same way."""
    bind_catalogue(
        restrictive_catalogue(
            provenances=[
                provenance_row(_DFO_PROVENANCE, DFO),
                provenance_row(_BASE_PROVENANCE, BASE),
            ],
            dfo_variables=[_SV],
            base_variables={_SV: ([_BASE_PROVENANCE], [])},
        )
    )

    bulk, per_pair = _elections(
        {"f-dfo": list(range(2019, 2024)), "f-base": list(range(1970, 2024))},
        {
            "f-dfo": FacetMetadata(provenance_id=_DFO_PROVENANCE),
            "f-base": FacetMetadata(provenance_id=_BASE_PROVENANCE),
        },
    )

    assert bulk == "f-dfo"
    assert per_pair == {_KEN: "f-dfo", _NOR: "f-dfo"}


# ---------------------------------------------------------------------------
# Coverage names the years the preview serves, on every preview path
# ---------------------------------------------------------------------------


def _assert_dated_by_served_points(spec: Spec) -> None:
    """Coverage, the `when` slot and any headline name the years the delivered points fall
    in: the window spans them and the headline takes the latest."""
    observations = spec.resolved.observations
    assert observations is not None
    years = [int(p.date[:4]) for s in observations.series for p in s.points]
    assert years, "the preview must deliver a point, or the comparison is vacuous"
    served = TimeWindow(start_year=min(years), end_year=max(years))
    assert spec.resolved.coverage.window == served
    when_slot = next(slot for slot in spec.resolved.slots if slot.key.axis == "when")
    assert when_slot.binding.value.time_window == served
    if spec.resolved.headline is not None:
        assert spec.resolved.headline.year == served.end_year


def test_pair_path_names_the_variables_latest_period():
    """Standard per-pair preview: Kenya's recorded 2024, the variable's latest period, which
    Ethiopia's 2019 does not reach, never Kenya's 2032 projection."""
    spec = latest_projection._resolve_buffered("Kenya", "Ethiopia")

    _assert_dated_by_served_points(spec)
    assert spec.resolved.coverage.window == TimeWindow(start_year=2024, end_year=2024)


@pytest.mark.parametrize(
    ("a2_indexed_to", "a2_read"),
    [
        pytest.param(2021, "2021", id="index-agrees"),
        pytest.param(2024, "2021", id="index-ahead"),
        pytest.param(2021, "2025", id="unserved-place-read-newer"),
    ],
)
def test_scope_bulk_path_names_the_served_facets_latest_period(
    a2_indexed_to: int, a2_read: str, caplog: pytest.LogCaptureFixture
):
    """Scope-defaulted preview: `f-rec` covers both places and wins on places covered over the
    projection `f-est`, which only A1 carries. The variable serves `f-rec`'s latest period,
    2024, which A1 holds and A2, whose read ends in 2021, does not, even where A2's facet
    states it runs to 2024 and the plan reads A2 too. Where A2's facet ends in 2021 the plan
    serves only A1, and a newer A2 point the read returns cannot set the period. A2 is
    unserved on every row, so the scope-partial warning counts one of two places. Only where
    the index dates A2 past its read is the drop logged as a disagreement."""
    graph = place_scope._make_graph(
        obs={
            f"{place_scope._SV}|country/A1": [
                {
                    "facetId": "f-rec",
                    "earliestDate": "2010",
                    "latestDate": "2024",
                    "obsCount": 15,
                    "dates": [str(y) for y in range(2010, 2025)],
                    "values": [1.0] * 15,
                },
                {
                    "facetId": "f-est",
                    "earliestDate": "2025",
                    "latestDate": "2032",
                    "obsCount": 8,
                    "measurementMethod": "Estimated",
                    "dates": [str(y) for y in range(2025, 2033)],
                    "values": [1.0] * 8,
                },
            ],
            f"{place_scope._SV}|country/A2": [
                {
                    "facetId": "f-rec",
                    "earliestDate": "2010",
                    "latestDate": str(a2_indexed_to),
                    "obsCount": a2_indexed_to - 2009,
                    "dates": [str(y) for y in range(2010, a2_indexed_to + 1)],
                    "values": [1.0] * (a2_indexed_to - 2009),
                },
            ],
        }
    )
    sdmx = FakeSdmxClient(
        observations={},
        availability={},
        subject_observations={
            place_scope._SV: {
                "country/A1": [_scope_row("country/A1", "f-rec", "2024")],
                "country/A2": [_scope_row("country/A2", "f-rec", a2_read)],
            }
        },
        scope_availability=graph.scope_availability,
    )
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(
            resolve_variable(
                place_scope._DETECT_QUERY,
                entities=[],
                date_request=DateRequest(window=None, latest=True),
                detect_query=place_scope._DETECT_QUERY,
                role_query=place_scope._DETECT_QUERY,
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=place_scope._FixedBindLLM(),
                base_timing={},
                sdmx=sdmx,
                coverage=empty_coverage_snapshot(),
                options=EngineOptions(include_observations=True),
            )
        )

    assert result.status == "definite", result
    spec = result.specs[0]
    assert [call["facet_ids"] for call in sdmx.subject_observations_calls] == [["f-rec"]]
    _assert_dated_by_served_points(spec)
    assert spec.resolved.coverage.window == TimeWindow(start_year=2024, end_year=2024)
    assert [w.message for w in result.warnings if w.code == OBSERVATIONS_SCOPE_PARTIAL] == [
        "series for 1 of 2 places in the resolved scope; 1 place has no data on the "
        "selected observation facet in its latest period."
    ]
    dropped = [r for r in caplog.records if "the scope read disagree" in r.getMessage()]
    assert [r.args for r in dropped] == (
        [(1, [(place_scope._SV, "country/A2")])] if a2_indexed_to > int(a2_read) else []
    )


def _scope_row(entity: str, facet_id: str, year: str) -> SdmxObservation:
    return SdmxObservation(
        variable=place_scope._SV,
        role_values={"observationAbout": entity},
        unit=None,
        measurement_method=None,
        observation_period=None,
        provenance=None,
        time_period=year,
        value="1.0",
        scaling_factor=None,
        facet_id=facet_id,
    )


def _role_row(recipient: str, facet_id: str, years: range) -> tuple:
    return (role_bearing._SV_DCID, facet_id, ABSENT_ENTITY, recipient, list(years))


# A one-role debtor variable, as IDS totals are. Kenya holds recorded `f-rec` values to 2024
# and an `Estimated` projection `f-est` to 2032; Norway holds only `f-old`, which ends in 2019
# and has the most observations of any facet.
_ROLE_FACETS: dict[str, list[tuple[str, range, str | None]]] = {
    _KEN: [("f-rec", range(2010, 2025), None), ("f-est", range(2025, 2033), "Estimated")],
    _NOR: [("f-old", range(1990, 2020), None)],
}
type _RoleFacets = dict[str, list[tuple[str, range, str | None]]]


def _role_metadata(
    role_facets: _RoleFacets, provenances: Mapping[str, str] = MappingProxyType({})
) -> dict[str, FacetMetadata]:
    return {
        facet_id: FacetMetadata(
            provenance_id=provenances.get(facet_id, _BASE_PROVENANCE), measurement_method=method
        )
        for facets in role_facets.values()
        for facet_id, _years, method in facets
    }


def _role_materialise_sdmx(
    role_facets: _RoleFacets, provenances: Mapping[str, str] = MappingProxyType({})
) -> MaterialiseSdmxClient:
    """Serve each dispatched facet's rows for the bound debtor, or its latest row under
    `LATEST`, as the service does."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        body = _csv_header(["debtor"])
        recipient = params.get("c[debtor]", "")
        facet_ids = params.get("c[facetId]")
        latest = params.get("c[TIME_PERIOD]") == "LATEST"
        for facet_id, years, method in role_facets.get(recipient, []):
            if facet_ids is not None and facet_id not in facet_ids.split(","):
                continue
            fixture = ObsFacetFixture(
                facet_id=facet_id,
                provenance_id=provenances.get(facet_id, _BASE_PROVENANCE),
                measurement_method=method,
            )
            for year in [years[-1]] if latest else years:
                body += _csv_row(
                    stat_var=role_bearing._SV_DCID,
                    role_dcids=[recipient],
                    fixture=fixture,
                    date_str=str(year),
                    value=float(year),
                )
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    return MaterialiseSdmxClient(
        base_url=MATERIALISE_BASE_URL, transport=httpx.MockTransport(handler)
    )


def _role_answer(
    monkeypatch,
    search_settings,
    *,
    max_pairs: int,
    role_facets: _RoleFacets = _ROLE_FACETS,
    discovered: bool = False,
    provenances: Mapping[str, str] | None = None,
) -> Spec:
    """Kenya's and Norway's latest debt flows, named as destinations, or with `discovered`
    found on the open destination axis. `provenances` names a facet's source where it is not
    the base one."""
    provenances = provenances or {}
    search_settings(obs_preview_max_pairs=max_pairs)
    rows = [
        _role_row(recipient, facet_id, years)
        for recipient, facets in role_facets.items()
        for facet_id, years, _method in facets
    ]
    index = patch_coverage(
        monkeypatch,
        rows,
        role_names={role_bearing._SV_DCID: RoleNames(donor=None, recipient="debtor")},
        facet_metadata=_role_metadata(role_facets, provenances),
    )
    sdmx = FakeSdmxClient(
        observations={
            role_bearing._SV_DCID: {
                (("debtor", recipient),): [
                    Facet(
                        facet_id=facet_id,
                        earliest_date=str(years[0]),
                        latest_date=str(years[-1]),
                        obs_count=len(years),
                        dates=[str(y) for y in years],
                        values=[float(y) for y in years],
                    )
                    for facet_id, years, _method in facets
                ]
                for recipient, facets in role_facets.items()
            }
        },
        availability={},
    )
    result = role_bearing._run(
        entities=[] if discovered else ["Kenya", "Norway"],
        sv_dcid=role_bearing._SV_DCID,
        sv_arcs=role_bearing._declaring_arcs(observation_properties=["debtor"]),
        resolve={"Kenya": _KEN, "Norway": _NOR},
        sv_score=0.9,
        role_query="debt flows" if discovered else "debt flows to Kenya and Norway",
        sdmx=sdmx,
        include_observations=True,
        materialise_graph=FakeGraphReader(),
        materialise_sdmx=_role_materialise_sdmx(role_facets, provenances),
        date_request=DateRequest(window=None, latest=True),
        coverage=FakeCoverageService(index).snapshot(),
    )
    assert result.status == "definite", result
    return result.specs[0]


def test_role_pair_path_elects_each_recipients_facet_as_the_preview_does(
    monkeypatch, search_settings
):
    """Per-pair role preview: Kenya's pair elects `f-rec` (2024) and Norway's `f-old`
    (2019), so the variable's latest period is Kenya's 2024, which Norway does not reach. A
    variable-wide election would pick `f-old` for both and date the answer 2019."""
    spec = _role_answer(monkeypatch, search_settings, max_pairs=120)

    observations = spec.resolved.observations
    assert observations is not None
    served = {
        next(e.ref for e in s.entities if e.role == "destination"): [p.date for p in s.points]
        for s in observations.series
    }
    assert served == {_KEN: ["2024"]}
    _assert_dated_by_served_points(spec)


def test_role_pair_path_dates_each_pair_by_its_joint_dispatch_group(monkeypatch, search_settings):
    """Kenya holds `f-a` (1990-2019) and `f-b` (2015-2024) from one import. The per-pair
    preview dispatches both and serves Kenya's 2024 point, the variable's latest period, which
    Norway's `f-a`, ending in 2022, does not reach, so coverage names 2024. Electing `f-a`
    alone would date Kenya 2019 and serve Norway's 2022."""
    spec = _role_answer(
        monkeypatch,
        search_settings,
        max_pairs=120,
        role_facets={
            _KEN: [("f-a", range(1990, 2020), None), ("f-b", range(2015, 2025), None)],
            _NOR: [("f-a", range(1990, 2023), None)],
        },
    )

    observations = spec.resolved.observations
    assert observations is not None
    served = {
        next(e.ref for e in s.entities if e.role == "destination"): [p.date for p in s.points]
        for s in observations.series
    }
    assert served == {_KEN: ["2024"]}
    _assert_dated_by_served_points(spec)
    assert spec.resolved.coverage.window == TimeWindow(start_year=2024, end_year=2024)
    headline = spec.resolved.headline
    assert headline is not None
    assert (headline.year, headline.series) == (2024, 1)


def test_discovered_pair_path_serves_the_variables_latest_period(monkeypatch, search_settings):
    """Two discovered debtors fit the pair cap, so the preview reads each pair on its own:
    Kenya's series ends in 2020 and Norway's in 2021, so the variable serves 2021 from Norway
    alone, and coverage, the `when` slot and the headline name it."""
    spec = _role_answer(
        monkeypatch,
        search_settings,
        max_pairs=120,
        role_facets={
            _KEN: [("f-a", range(2010, 2021), None)],
            _NOR: [("f-a", range(2012, 2022), None)],
        },
        discovered=True,
    )

    observations = spec.resolved.observations
    assert observations is not None
    served = {
        next(e.ref for e in s.entities if e.role == "destination"): [p.date for p in s.points]
        for s in observations.series
    }
    assert served == {_NOR: ["2021"]}
    _assert_dated_by_served_points(spec)
    assert spec.resolved.coverage.window == TimeWindow(start_year=2021, end_year=2021)


def test_role_bulk_path_names_the_served_facets_pairs(monkeypatch, search_settings):
    """Bulk role preview: one facet serves every pair. `f-old` leads on observations among
    facets covering one pair each, so Norway's 2019 point is the answer."""
    spec = _role_answer(monkeypatch, search_settings, max_pairs=1)

    _assert_dated_by_served_points(spec)


# ---------------------------------------------------------------------------
# A member of an index-resolved family is read from its elected dataset alone.
# ---------------------------------------------------------------------------

_ELECTED, _OTHER = "provenance/Elected", "provenance/Other"


def _dataset_member(dataset: str | None) -> IndexMemberFacts:
    """The role variable as an index row recorded under two datasets, read into the family
    `dataset` was elected for."""
    return IndexMemberFacts(
        row=StatVarRow(
            dcid=role_bearing._SV_DCID,
            population_type="EconomicActivity",
            measured_property="amount",
            stat_type="measuredValue",
            measurement_qualifier=None,
            measurement_denominator=None,
            observation_properties=("debtor",),
            constraint_properties={},
            provenances=(_ELECTED, _OTHER),
        ),
        dataset=dataset,
    )


def test_a_plan_and_a_mint_keep_a_member_by_one_dataset_test_before_they_rank():
    """The other dataset's facet covers more series and holds more points. A mint of the
    member keeps the elected dataset's facet (`dataset_candidates`), and a per-series plan
    and a bulk plan passed through `in_elected_datasets` elect that facet too."""

    def facet(provenance: str, obs: int) -> MaterialiseFacet:
        return MaterialiseFacet(
            facet_id=f"facet-{provenance}",
            provenance_id=provenance,
            measurement_method="METH",
            unit="UNIT",
            obs_count=obs,
            earliest_date="2015",
            latest_date="2020",
        )

    sv = role_bearing._SV_DCID
    candidates = [facet(_OTHER, 9), facet(_ELECTED, 1)]
    request = MaterialisationRequest(
        stat_var=f"dcid:{sv}",
        role_bindings={"debtor": [_KEN]},
        declared_roles=("debtor",),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
        dataset=_dataset_member(_ELECTED).read_dataset(),
    )
    mint = _select_member_facets(request, candidates, graph=FakeGraphReader())
    assert mint["provenance"].ref == _ELECTED

    series = [
        PlanSeries(stat_var=sv, role_tuple=(("debtor", place),), candidates=tuple(candidates))
        for place in (_KEN, _NOR)
    ]
    members = {sv: _dataset_member(_ELECTED)}
    pairs = in_elected_datasets(plan_pairs(series, members=[sv], years=None), members)
    assert isinstance(pairs, PairPlan)
    assert {f.provenance_id for s in pairs.served for f in s.group} == {_ELECTED}
    assert {f.provenance_id for f in pairs.probes[sv].candidates} == {_ELECTED}

    unelected = plan_pairs(series, members=[sv], years=None)
    assert in_elected_datasets(unelected, {sv: _dataset_member(None)}) is unelected
    assert {f.provenance_id for s in unelected.served for f in s.group} == {_OTHER}


def test_a_role_answer_serves_an_index_member_from_its_elected_dataset(
    monkeypatch, search_settings
):
    """Kenya's and Norway's flows hold `f-other` from 1990 and `f-elected` from 2015, so
    ranking alone serves `f-other`. Read into a family that elected the second dataset, the
    member is served from `f-elected` per pair and in bulk, at named and at discovered
    destinations."""
    role_facets: _RoleFacets = {
        place: [("f-other", range(1990, 2025), None), ("f-elected", range(2015, 2025), None)]
        for place in (_KEN, _NOR)
    }
    provenances = {"f-other": _OTHER, "f-elected": _ELECTED}

    def answer(dataset: str | None, *, max_pairs: int, discovered: bool) -> Spec:
        monkeypatch.setattr(
            role_resolve,
            "shape_member_facts",
            lambda shape, query: ({role_bearing._SV_DCID: _dataset_member(dataset)}, True),
        )
        return _role_answer(
            monkeypatch,
            search_settings,
            max_pairs=max_pairs,
            role_facets=role_facets,
            provenances=provenances,
            discovered=discovered,
        )

    for max_pairs, discovered in ((120, False), (1, False), (120, True), (1, True)):
        elected = answer(_ELECTED, max_pairs=max_pairs, discovered=discovered)
        assert elected.resolved.facets["provenance"].ref == _ELECTED
        observations = elected.resolved.observations
        assert observations is not None and observations.series
        assert {s.facet["provenance"].ref for s in observations.series} == {_ELECTED}

        ranked = answer(None, max_pairs=max_pairs, discovered=discovered)
        assert ranked.resolved.facets["provenance"].ref == _OTHER
