"""A dispatch serves one observation period (`materialise.facets.dispatch_cadence`), and a
group serves one for every member.

#347: the UN graph publishes Kenya's ILO unemployment rate as an annual (`P1Y`) and a
quarterly (`P3M`) facet with the same provenance and unit. Both matched the elected facet's
axes, so both were fetched, and the frame held `2019` beside `2019-01`; the presentation
assembler could then bind no chart over it (`'2022-10' is not a year period`). The facets,
dates and values below are the UN graph's for `country/KEN`, recorded 2026-10-07.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import httpx
import pytest

from assembler.bind import count_periods
from assembler.frame_view import compute_structural_view
from dc_sdmx import SdmxClient
from materialise import EnrichmentLookup, MaterialisationRequest, generation, materialise_group
from materialise.codes import MaterialisationWarning
from materialise.errors import MixedGrainError
from materialise.facets import (
    Cadence,
    FacetBinding,
    PlaceAdmission,
    joint_dispatch_facet_ids,
    subject_probe,
)
from materialise.generation import (
    _generate_chunk,
    _MemberFetchPlan,
    _plan_collection_chunks,
    _RawRowBudget,
    _select_member_facets,
)
from materialise.graph import Facet, FacetDefaultPolicy
from materialise.group_facets import select_group_facets
from materialise.window import RequestedWindow

from ._group_support import served_alone
from .conftest import FakeGraphReader, only_chunk

_RATE = "UnemploymentRate_ILO"
_YOUTH = "UnemploymentRate_ILO_Youth"
_KENYA = "country/KEN"
_ANNUAL_ID, _QUARTERLY_ID = "12900944295343888504", "10046015425536511672"
_ANNUAL = {"2019": 7.487, "2021": 8.874, "2022": 8.16}
_QUARTERLY = {
    "2019-01": 8.992, "2019-10": 6.999, "2020-01": 8.269, "2020-04": 15.202,
    "2021-01": 8.543, "2021-04": 9.496, "2021-07": 7.92, "2021-10": 9.545,
    "2022-01": 10.051, "2022-04": 8.452, "2022-07": 6.803, "2022-10": 7.357,
}  # fmt: skip
# A second annual facet, as an import publishing the rate twice would carry.
_REISSUE_ID = "77777777777777777777"
# A facet stating no period, written with an empty `observationPeriod` cell.
_UNSTATED_ID = "55555555555555555555"
_PERIOD_OF = {_ANNUAL_ID: "P1Y", _QUARTERLY_ID: "P3M", _REISSUE_ID: "P1Y", _UNSTATED_ID: ""}
# More annual points than quarterly ones in 2019-2022, so the youth rate alone elects its
# annual facet; the group's summed counts elect the quarterly one.
_YOUTH_ANNUAL = {"2018": 15.1, "2019": 15.3, "2020": 16.0, "2021": 15.8, "2022": 15.5}
_YOUTH_QUARTERLY = {"2019-01": 16.2, "2020-04": 25.1, "2022-04": 15.9}
_ALL_BOUND = {
    "provenance": FacetBinding(state="bound", ref="undata/p/ILO"),
    "measurementMethod": FacetBinding(state="bound", ref=None),
    "unit": FacetBinding(state="bound", ref="Percent"),
}
_SDMX_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_UNBOUND = {
    axis: FacetBinding(state="unbound") for axis in ("provenance", "measurementMethod", "unit")
}


def _facet(facet_id: str, values: dict[str, float]) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id="undata/p/ILO",
        unit="Percent",
        observation_period=_PERIOD_OF[facet_id],
        obs_count=len(values),
        earliest_date=min(values),
        latest_date=max(values),
    )


def _sdmx(rows: dict[tuple[str, str], dict[str, float]], fetched: list[set[str]]) -> SdmxClient:
    """Serves `rows`, keyed `(variable, facet id)`, for the facets each request names, as the
    service filters on `c[facetId]`."""

    def handler(request: httpx.Request) -> httpx.Response:
        variables = str(request.url.params.get("c[variableMeasured]")).split(",")
        facet_ids = set(str(request.url.params.get("c[facetId]")).split(","))
        fetched.append(facet_ids)
        body = "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{variable},{_KENYA},Percent,NotApplicable,"
            f"{_PERIOD_OF[facet_id]},undata/p/ILO,{period},{value},,{facet_id}\r\n"
            for (variable, facet_id), values in rows.items()
            if variable in variables and facet_id in facet_ids
            for period, value in values.items()
        )
        return httpx.Response(
            200,
            text=_SDMX_HEADER + body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(base_url="https://ours.example.org", transport=httpx.MockTransport(handler))


def _request(stat_var: str, **bound: FacetBinding) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=f"dcid:{stat_var}",
        role_bindings={"observationAbout": [_KENYA]},
        declared_roles=("observationAbout",),
        facet_bindings={**_UNBOUND, **bound},
        window=RequestedWindow(kind="range", start_year=2019, end_year=2026, end_unstated=True),
    )


def _materialise(
    requests: Sequence[MaterialisationRequest],
    graph: FakeGraphReader,
    sdmx: SdmxClient,
    warnings: dict[str, MaterialisationWarning] | None = None,
    witnesses: set[str] | None = None,
    absent_reads_zero: Callable[[str], bool] | None = None,
) -> dict[str, list[str]]:
    """Each StatVar's served periods, after a default-window bind check over the frame. The
    result's warnings go to `warnings`, by code, and the trace's per-axis
    `dispatch_facet_id`s to `witnesses`, when given."""
    result = materialise_group(
        list(requests),
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=1,
        absent_reads_zero=absent_reads_zero,
    )
    if warnings is not None:
        warnings.update((warning["code"], warning) for warning in result.warnings)
    if witnesses is not None:
        witnesses.update(str(entry["dispatch_facet_id"]) for entry in result.trace.facets)
    frame = only_chunk(result)
    rows = [list(row) for row in frame.rows]
    view = compute_structural_view(
        frame={"columns": list(frame.columns), "rows": rows}, continuation=None
    )
    # The presentation assembler's default window over the frame binds without refusal.
    assert view.coverage is not None
    count_periods(
        start=view.coverage[0], end=view.coverage[1], grain=view.period_grain, span=view.span
    )
    periods: dict[str, list[str]] = {}
    for stat_var, period in zip(
        frame.table["stat_var"].to_pylist(), frame.table["time_period"].to_pylist(), strict=True
    ):
        periods.setdefault(stat_var, []).append(period)
    return {stat_var: sorted(found) for stat_var, found in periods.items()}


def test_kenya_unemployment_serves_the_elected_facets_quarters_alone():
    graph = FakeGraphReader(
        facets={(_RATE, _KENYA): [_facet(_QUARTERLY_ID, _QUARTERLY), _facet(_ANNUAL_ID, _ANNUAL)]}
    )
    fetched: list[set[str]] = []
    sdmx = _sdmx({(_RATE, _QUARTERLY_ID): _QUARTERLY, (_RATE, _ANNUAL_ID): _ANNUAL}, fetched)

    periods = _materialise([_request(_RATE)], graph, sdmx)

    assert fetched == [{_QUARTERLY_ID}]
    assert periods == {f"dcid:{_RATE}": sorted(_QUARTERLY)}


def test_a_group_member_serves_no_row_from_a_facet_only_another_member_dispatched():
    # Both members elect the annual facet; the rate's probe also holds a reissue of it, so
    # the chunk fetches the reissue's id for both variables. The youth rate's probe reports
    # the reissue under another source, so its place admission admits the id at Kenya but
    # the id is no joint match of its own. Its rows under that id come back under the
    # shared filter and do not serve: only the filter to its own dispatch ids stops them.
    reissue = {"2023": 1.0}
    graph = FakeGraphReader(
        facets={
            (_RATE, _KENYA): [_facet(_ANNUAL_ID, _ANNUAL), _facet(_REISSUE_ID, reissue)],
            (_YOUTH, _KENYA): [
                _facet(_ANNUAL_ID, _ANNUAL),
                replace(_facet(_REISSUE_ID, reissue), provenance_id="other/p/Source"),
            ],
        }
    )
    fetched: list[set[str]] = []
    sdmx = _sdmx(
        {
            (_RATE, _ANNUAL_ID): _ANNUAL,
            (_RATE, _REISSUE_ID): reissue,
            (_YOUTH, _ANNUAL_ID): _ANNUAL,
            (_YOUTH, _REISSUE_ID): reissue,
        },
        fetched,
    )

    result = materialise_group(
        [_request(_RATE), _request(_YOUTH)],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=1,
    )

    table = only_chunk(result).table
    served = set(zip(table["stat_var"].to_pylist(), table["time_period"].to_pylist(), strict=True))
    assert fetched == [{_ANNUAL_ID, _REISSUE_ID}]
    assert {period for stat_var, period in served if stat_var == f"dcid:{_YOUTH}"} == set(_ANNUAL)


@pytest.mark.parametrize("annual_first", [True, False])
def test_a_bound_source_serves_the_window_whatever_the_candidate_order(annual_first):
    # An old annual series ending in 2009 beside the quarterly one: binding the source must
    # not let its first-matching witness pick the annual facet and serve nothing in 2019-2026.
    old_annual = {str(year): 5.0 for year in range(1960, 2010)}
    facets = [_facet(_ANNUAL_ID, old_annual), _facet(_QUARTERLY_ID, _QUARTERLY)]
    graph = FakeGraphReader(facets={(_RATE, _KENYA): facets if annual_first else facets[::-1]})
    fetched: list[set[str]] = []
    sdmx = _sdmx({(_RATE, _QUARTERLY_ID): _QUARTERLY, (_RATE, _ANNUAL_ID): old_annual}, fetched)
    request = _request(_RATE, provenance=FacetBinding(state="bound", ref="undata/p/ILO"))

    witnesses: set[str] = set()

    periods = _materialise([request], graph, sdmx, witnesses=witnesses)

    assert fetched == [{_QUARTERLY_ID}]
    assert periods == {f"dcid:{_RATE}": sorted(_QUARTERLY)}
    # Every axis's trace witness is a facet the dispatch fetched.
    assert witnesses == {_QUARTERLY_ID}


@pytest.mark.parametrize("annual_first", [True, False])
def test_a_declared_default_source_serves_the_window_whatever_the_candidate_order(
    annual_first,
):
    # A graph-declared default's witness is the first candidate
    # carrying its value too, so it must not pick the period either.
    old_annual = {str(year): 5.0 for year in range(1960, 2010)}
    facets = [_facet(_ANNUAL_ID, old_annual), _facet(_QUARTERLY_ID, _QUARTERLY)]
    request = _request(
        _RATE,
        measurementMethod=FacetBinding(state="bound", ref=None),
        unit=FacetBinding(state="bound", ref="Percent"),
    )
    policy = FacetDefaultPolicy(
        resolved_ref="undata/p/ILO",
        policy_node="policy/ilo",
        policy_version="v1",
        selection_reason="default source",
    )
    graph = FakeGraphReader(
        facets={(_RATE, _KENYA): facets if annual_first else facets[::-1]},
        policies={(request.stat_var, "provenance"): policy},
    )
    fetched: list[set[str]] = []
    sdmx = _sdmx({(_RATE, _QUARTERLY_ID): _QUARTERLY, (_RATE, _ANNUAL_ID): old_annual}, fetched)

    witnesses: set[str] = set()

    periods = _materialise([request], graph, sdmx, witnesses=witnesses)

    assert fetched == [{_QUARTERLY_ID}]
    assert periods == {f"dcid:{_RATE}": sorted(_QUARTERLY)}
    # Every axis's trace witness is a facet the dispatch fetched.
    assert witnesses == {_QUARTERLY_ID}


def test_a_fully_bound_group_serves_the_groups_one_cadence_for_every_member():
    # Every axis bound: the group's ranking over its summed counts elects the quarterly
    # facet, and the youth rate, which alone would elect its annual one, serves quarters too.
    graph = FakeGraphReader(
        facets={
            (_RATE, _KENYA): [_facet(_ANNUAL_ID, _ANNUAL), _facet(_QUARTERLY_ID, _QUARTERLY)],
            (_YOUTH, _KENYA): [
                _facet(_ANNUAL_ID, _YOUTH_ANNUAL),
                _facet(_QUARTERLY_ID, _YOUTH_QUARTERLY),
            ],
        }
    )
    fetched: list[set[str]] = []
    sdmx = _sdmx(
        {
            (_RATE, _ANNUAL_ID): _ANNUAL,
            (_RATE, _QUARTERLY_ID): _QUARTERLY,
            (_YOUTH, _ANNUAL_ID): _YOUTH_ANNUAL,
            (_YOUTH, _QUARTERLY_ID): _YOUTH_QUARTERLY,
        },
        fetched,
    )

    warnings: dict[str, MaterialisationWarning] = {}

    periods = _materialise(
        [_request(_RATE, **_ALL_BOUND), _request(_YOUTH, **_ALL_BOUND)], graph, sdmx, warnings
    )

    assert fetched == [{_QUARTERLY_ID}]
    assert periods == {
        f"dcid:{_RATE}": sorted(_QUARTERLY),
        f"dcid:{_YOUTH}": sorted(_YOUTH_QUARTERLY),
    }
    # Every member served, so coverage does not fall short.
    assert "coverage_below_scope" not in warnings


def test_a_group_elects_the_cadence_serving_the_most_members():
    # An index-sourced group agrees on provenance and unit only. The rate's quarterly facet
    # holds the most observations, but only the rate holds quarters, while both members
    # hold an annual facet, so the group serves years for both.
    rate = [_facet(_ANNUAL_ID, _ANNUAL), _facet(_QUARTERLY_ID, _QUARTERLY)]
    youth = [_facet(_REISSUE_ID, _YOUTH_ANNUAL)]
    graph = FakeGraphReader(facets={(_RATE, _KENYA): rate, (_YOUTH, _KENYA): youth})
    requests = [
        replace(_request(_RATE), index_sourced=True),
        replace(_request(_YOUTH), index_sourced=True),
    ]
    fetched: list[set[str]] = []
    sdmx = _sdmx(
        {
            (_RATE, _ANNUAL_ID): _ANNUAL,
            (_RATE, _QUARTERLY_ID): _QUARTERLY,
            (_YOUTH, _REISSUE_ID): _YOUTH_ANNUAL,
        },
        fetched,
    )
    warnings: dict[str, MaterialisationWarning] = {}

    periods = _materialise(requests, graph, sdmx, warnings)

    assert fetched == [{_ANNUAL_ID, _REISSUE_ID}]
    assert periods == {
        f"dcid:{_RATE}": sorted(_ANNUAL),
        f"dcid:{_YOUTH}": sorted(period for period in _YOUTH_ANNUAL if period >= "2019"),
    }
    assert "coverage_below_scope" not in warnings


@pytest.mark.parametrize(
    ("rate", "youth"),
    [
        # The merged facet's bounds are bare years, though the youth rate's copy is monthly.
        pytest.param(
            {str(year): 1.0 for year in range(2019, 2025)},
            {f"2020-{month:02d}": 2.0 for month in range(1, 13)},
            id="annual-bounds",
        ),
        # The merged facet's bounds are months, though the rate's copy is annual.
        pytest.param(
            {str(year): 1.0 for year in range(2020, 2023)},
            {f"{year}-{month:02d}": 2.0 for year in (2019, 2023) for month in range(1, 13)},
            id="monthly-bounds",
        ),
    ],
)
def test_a_group_serves_a_shared_unstated_facet_only_to_members_dating_it_alike(rate, youth):
    # One facet id states no period and holds years for the rate and months for the youth
    # rate. Each member's copy is judged apart, so the rate's is annual and the youth's is
    # not, no cadence holds both, and the earlier member's wins: the frame holds years alone
    # and binds, and the youth rate is reported as serving nothing.
    def unstated(values: dict[str, float]) -> Facet:
        return Facet(
            facet_id=_UNSTATED_ID,
            provenance_id="undata/p/ILO",
            unit="Percent",
            obs_count=len(values),
            earliest_date=min(values),
            latest_date=max(values),
        )

    graph = FakeGraphReader(
        facets={(_RATE, _KENYA): [unstated(rate)], (_YOUTH, _KENYA): [unstated(youth)]}
    )
    fetched: list[set[str]] = []
    sdmx = _sdmx({(_RATE, _UNSTATED_ID): rate, (_YOUTH, _UNSTATED_ID): youth}, fetched)
    warnings: dict[str, MaterialisationWarning] = {}

    periods = _materialise([_request(_RATE), _request(_YOUTH)], graph, sdmx, warnings)

    assert fetched == [{_UNSTATED_ID}]
    assert periods == {f"dcid:{_RATE}": sorted(rate)}
    assert warnings["coverage_below_scope"].get("off_cadence_members") == {
        "count": 1,
        "refs": [f"dcid:{_YOUTH}"],
    }


def test_a_chunk_serves_nothing_for_a_member_whose_populated_probe_dispatches_no_facet():
    # The youth rate's probe reported the quarterly facet, so its place admission admits it
    # at Kenya, but its plan dispatches no facet. The chunk fetches the quarterly id for the
    # rate, and the youth rate's rows under it do not serve.
    rate, youth = _request(_RATE), _request(_YOUTH)
    quarterly = _facet(_QUARTERLY_ID, _QUARTERLY)
    selections = select_group_facets(
        [rate, youth], [[quarterly], [quarterly]], graph=FakeGraphReader()
    ).election.selections
    admission = PlaceAdmission(places_by_facet={_QUARTERLY_ID: frozenset({_KENYA})})
    fetched: list[set[str]] = []
    sdmx = _sdmx(
        {(_RATE, _QUARTERLY_ID): _QUARTERLY, (_YOUTH, _QUARTERLY_ID): _YOUTH_QUARTERLY}, fetched
    )
    chunk_requests = [("0", rate), ("1", youth)]
    (chunk,) = _plan_collection_chunks(chunk_requests, places=[(), ()])
    plans = [
        _MemberFetchPlan(
            selections=selections,
            dispatch_ids=(_QUARTERLY_ID,),
            admission=admission,
            conversion_plan=None,
        ),
        _MemberFetchPlan(
            selections=selections, dispatch_ids=(), admission=admission, conversion_plan=None
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

    assert fetched == [{_QUARTERLY_ID}]
    assert result[0].table.num_rows == len(_QUARTERLY)
    assert result[1].table.num_rows == 0


@pytest.mark.parametrize(
    ("rows", "chunk_rows"),
    [
        # One member's facet dates its rows by year and by month.
        pytest.param({_RATE: {"2019": 1.0, "2020-06": 2.0}}, None, id="member"),
        # Two members share the facet, the rate's rows dated by year and the youth rate's by
        # month; neither copy reads as annual, so both dispatch it.
        pytest.param(
            {_RATE: {"2019": 1.0, "2020": 2.0}, _YOUTH: {"2020-01": 3.0, "2020-06": 4.0}},
            None,
            id="group",
        ),
        # The same group with each member in its own chunk, each of one grain.
        pytest.param(
            {_RATE: {"2019": 1.0, "2020": 2.0}, _YOUTH: {"2020-01": 3.0, "2020-06": 4.0}},
            1,
            id="group-across-chunks",
        ),
    ],
)
def test_a_mint_refuses_rows_of_two_grains(rows, chunk_rows, monkeypatch):
    if chunk_rows is not None:
        monkeypatch.setattr(generation, "_FRAME_CHUNK_TARGET_ROWS", chunk_rows)
    # The facet states no period, and its bounds, `2019` and `2020-06`, are no evidence of
    # one grain, so nothing tells the election its rows mix until they arrive.
    facet = Facet(
        facet_id=_UNSTATED_ID,
        provenance_id="undata/p/ILO",
        unit="Percent",
        obs_count=2,
        earliest_date="2019",
        latest_date="2020-06",
    )
    graph = FakeGraphReader(facets={(stat_var, _KENYA): [facet] for stat_var in rows})
    sdmx = _sdmx({(stat_var, _UNSTATED_ID): values for stat_var, values in rows.items()}, [])

    with pytest.raises(MixedGrainError) as excinfo:
        _materialise([_request(stat_var) for stat_var in rows], graph, sdmx)

    assert excinfo.value.code == "internal_error"
    assert set(excinfo.value.grains) == {"year", "month"}


def _unstated(facet_id: str, values: dict[str, float], *, obs_count: int | None = None) -> Facet:
    """An ILO facet stating no period, dated by `values`' keys."""
    return Facet(
        facet_id=facet_id,
        provenance_id="undata/p/ILO",
        unit="Percent",
        obs_count=len(values) if obs_count is None else obs_count,
        earliest_date=min(values),
        latest_date=max(values),
    )


def test_an_unstated_facet_rides_with_an_annual_facet_only_for_a_member_dating_it_by_year():
    # Both members hold a stated annual facet and one facet stating no period, which the
    # rate dates by year and the youth rate by month. The merged facet's bounds are bare
    # years, but each member's copy is judged apart: the rate dispatches the unstated facet
    # beside the annual one, and the youth rate the annual one alone, so no member serves
    # months beside years.
    years = {str(year): 1.0 for year in range(2019, 2025)}
    months = {f"2020-{month:02d}": 3.0 for month in range(1, 13)}
    annual = replace(_facet(_ANNUAL_ID, years), obs_count=40)
    graph = FakeGraphReader(
        facets={
            (_RATE, _KENYA): [annual, _unstated(_UNSTATED_ID, years)],
            (_YOUTH, _KENYA): [annual, _unstated(_UNSTATED_ID, months)],
        }
    )
    fetched: list[set[str]] = []
    sdmx = _sdmx(
        {
            (_RATE, _ANNUAL_ID): years,
            (_RATE, _UNSTATED_ID): years,
            (_YOUTH, _ANNUAL_ID): years,
            (_YOUTH, _UNSTATED_ID): months,
        },
        fetched,
    )

    periods = _materialise([_request(_RATE), _request(_YOUTH)], graph, sdmx)

    assert fetched == [{_ANNUAL_ID, _UNSTATED_ID}]
    assert periods == {f"dcid:{_RATE}": sorted(years), f"dcid:{_YOUTH}": sorted(years)}


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        # Two of three members hold an unstated id, one dated by year and one by month: the
        # id is outside the group's shared facets, and each copy is judged apart, so the
        # member dating it by year dispatches it beside the annual facet.
        pytest.param(
            [
                [_ANNUAL_ID, ("X", {"2021": 1.0, "2023": 1.0})],
                [_ANNUAL_ID, ("X", {"2021-01": 1.0, "2023-12": 1.0})],
                [_ANNUAL_ID],
            ],
            [(_ANNUAL_ID, "X"), (_ANNUAL_ID,), (_ANNUAL_ID,)],
            id="held-by-some",
        ),
        # Every member holds both ids; the unstated one is annual for every copy, so it
        # serves beside the stated annual facet.
        pytest.param(
            [
                [_ANNUAL_ID, ("X", {"2021": 1.0, "2023": 1.0})],
                [_ANNUAL_ID, ("X", {"2020": 1.0, "2022": 1.0})],
            ],
            [(_ANNUAL_ID, "X"), (_ANNUAL_ID, "X")],
            id="annual-everywhere",
        ),
    ],
)
def test_a_group_judges_an_unstated_facet_id_by_each_members_own_copies(members, expected):
    stat_vars = [_RATE, _YOUTH, f"{_RATE}_Adult"][: len(members)]
    annual = replace(_facet(_ANNUAL_ID, _ANNUAL), obs_count=40)
    candidates = [
        [annual if held == _ANNUAL_ID else _unstated(held[0], held[1]) for held in member]
        for member in members
    ]
    election = select_group_facets(
        [_request(stat_var) for stat_var in stat_vars], candidates, graph=FakeGraphReader()
    ).election

    assert [
        joint_dispatch_facet_ids(member, election, member=index)
        for index, member in enumerate(candidates)
    ] == expected


@pytest.mark.parametrize(
    ("uganda", "expected"),
    [
        # One place dates the unstated facet by month: the merged bounds are bare years, but
        # the place's own dates are not, so the facet is a cadence of its own.
        pytest.param({f"2020-{month:02d}": 3.0 for month in range(1, 13)}, ("A",), id="monthly"),
        # Every place dates it by year, so it serves beside the stated annual facet.
        pytest.param({str(year): 3.0 for year in range(2019, 2022)}, ("A", "X"), id="annual"),
    ],
)
def test_a_member_judges_an_unstated_facet_by_every_places_dates_before_they_merge(
    uganda, expected
):
    years = {str(year): 1.0 for year in range(2018, 2024)}
    annual = replace(_facet(_ANNUAL_ID, years), facet_id="A", obs_count=40)
    by_place = {
        _KENYA: [annual, _unstated("X", years)],
        "country/UGA": [annual, _unstated("X", uganda)],
    }
    probe = subject_probe(by_place, list(by_place))
    request = replace(_request(_RATE), role_bindings={"observationAbout": list(by_place)})

    election = _select_member_facets(
        request, probe.candidates, graph=FakeGraphReader(), breadth=probe.breadth_in(request.window)
    )

    assert joint_dispatch_facet_ids(probe.candidates, election, member=0) == expected


def _group_facet(
    facet_id: str, provenance: str, unit: str, obs_count: int, period: str = "P1Y"
) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        unit=unit,
        observation_period=period,
        obs_count=obs_count,
        earliest_date="2019",
        latest_date="2022",
    )


_OLD_ANNUAL = replace(
    _group_facet("A", "P", "U", 10), earliest_date="2000", latest_date="2010", obs_count=11
)


@pytest.mark.parametrize(
    ("declares", "candidates", "expected"),
    [
        # Both members declare provenance `P` and unit `U`; no shared facet carries both, but
        # each member holds its own facet that does. The cadence is ranked over those, and
        # each member dispatches its own.
        pytest.param(
            True,
            [
                [
                    _group_facet("A", "P", "U1", 50),
                    _group_facet("B", "P2", "U", 40),
                    _group_facet("C1", "P", "U", 5),
                ],
                [
                    _group_facet("A", "P", "U1", 50),
                    _group_facet("B", "P2", "U", 40),
                    _group_facet("C2", "P", "U", 6),
                ],
            ],
            [("C1",), ("C2",)],
            id="no-shared-match",
        ),
        # The rate also holds a quarterly facet of its own with more points than the shared
        # annual one, but only the rate holds quarters, so both members serve years.
        pytest.param(
            False,
            [
                [_group_facet("A", "P", "U", 10), _group_facet("Q", "P", "U", 90, "P3M")],
                [_group_facet("A", "P", "U", 10)],
            ],
            [("A",), ("A",)],
            id="member-only-outranks-shared-top",
        ),
        # Every member holds an annual facet, but its points all fall before the window,
        # while the third member's quarterly facet holds points in it. Only facets the window
        # keeps count, so the group serves quarters, and only the third member has them.
        pytest.param(
            False,
            [
                [_OLD_ANNUAL],
                [_OLD_ANNUAL],
                [_OLD_ANNUAL, _group_facet("Q", "P", "U", 12, "P3M")],
            ],
            [(), (), ("Q",)],
            id="out-of-window-majority",
        ),
        # The rate's copy of the annual facet holds points in the window, the other members'
        # copies do not, and those members hold quarters in it. Each member counts through its
        # own copies, so quarters serve two members against the annual facet's one.
        pytest.param(
            False,
            [
                [replace(_OLD_ANNUAL, earliest_date="2019", latest_date="2021")],
                [_OLD_ANNUAL, _group_facet("Q", "P", "U", 12, "P3M")],
                [_OLD_ANNUAL, _group_facet("Q", "P", "U", 12, "P3M")],
            ],
            [(), ("Q",), ("Q",)],
            id="one-copy-in-window",
        ),
        # The two members' copies of the annual facet fall either side of the window, so their
        # merged span reaches it while neither copy does.
        pytest.param(
            False,
            [
                [_OLD_ANNUAL],
                [
                    replace(_OLD_ANNUAL, earliest_date="2027", latest_date="2030"),
                    _group_facet("Q", "P", "U", 12, "P3M"),
                ],
            ],
            [(), ("Q",)],
            id="copies-straddle-the-window",
        ),
    ],
)
def test_a_groups_cadence_ranks_over_its_members_matching_facets(declares, candidates, expected):
    requests = [_request(stat_var) for stat_var in (_RATE, _YOUTH, f"{_RATE}_Adult")]
    requests = requests[: len(candidates)]
    declared = {
        (request.stat_var, axis): FacetDefaultPolicy(
            resolved_ref=ref, policy_node="policy/n", policy_version="v1", selection_reason="r"
        )
        for request in requests
        for axis, ref in (("provenance", "P"), ("unit", "U"))
    }
    graph = FakeGraphReader(policies=declared if declares else {})

    election = select_group_facets(requests, candidates, graph=graph).election

    assert [
        joint_dispatch_facet_ids(member, election, member=index)
        for index, member in enumerate(candidates)
    ] == expected


_EARLY_WINDOW = RequestedWindow(kind="range", start_year=2012, end_year=2015)


_MEMBER_FIELDS = ("member_count", "off_cadence_members", "out_of_window_members")


@pytest.mark.parametrize(
    ("youth_in_window", "reads_zero", "fields"),
    [
        # Its quarters fall in the window, so the figures leave out what it holds there.
        pytest.param(True, False, {"off_cadence_members"}, id="other-frequency"),
        pytest.param(True, True, {"off_cadence_members"}, id="other-frequency-reads-zero"),
        # It holds nothing in the window, a gap the dataset may read as zero.
        pytest.param(False, False, {"out_of_window_members"}, id="out-of-window"),
        pytest.param(False, True, None, id="out-of-window-reads-zero"),
    ],
)
def test_a_member_holding_the_groups_period_only_before_the_window_is_reported(
    youth_in_window, reads_zero, fields
):
    # Two members hold the annual facet in the window. The youth rate holds it only before
    # the window, so the group serves years and the youth rate, though it dispatches the
    # annual facet, serves nothing. Holding quarters in the window, it reports there at
    # another frequency; holding none, it reports nothing there, and a dataset reading an
    # absent member as zero raises no warning for it.
    in_window = {str(year): 1.0 for year in range(2012, 2016)}
    early = {str(year): 1.0 for year in range(2000, 2006)}
    quarters = {f"{year}-{month:02d}": 1.0 for year in range(2012, 2016) for month in (1, 4, 7, 10)}
    youth = [_facet(_ANNUAL_ID, early)]
    if youth_in_window:
        youth.insert(0, _facet(_QUARTERLY_ID, quarters))
    adult = f"{_RATE}_Adult"
    graph = FakeGraphReader(
        facets={
            (_RATE, _KENYA): [_facet(_ANNUAL_ID, in_window)],
            (adult, _KENYA): [_facet(_ANNUAL_ID, in_window)],
            (_YOUTH, _KENYA): youth,
        }
    )
    sdmx = _sdmx(
        {
            (_RATE, _ANNUAL_ID): in_window,
            (adult, _ANNUAL_ID): in_window,
            (_YOUTH, _ANNUAL_ID): early,
            (_YOUTH, _QUARTERLY_ID): quarters,
        },
        [],
    )
    requests = [
        replace(_request(stat_var), window=_EARLY_WINDOW) for stat_var in (_RATE, _YOUTH, adult)
    ]
    warnings: dict[str, MaterialisationWarning] = {}
    read: list[str] = []

    def absent_reads_zero(provenance: str) -> bool:
        read.append(provenance)
        return reads_zero

    periods = _materialise(requests, graph, sdmx, warnings, absent_reads_zero=absent_reads_zero)

    assert periods == {f"dcid:{_RATE}": sorted(in_window), f"dcid:{adult}": sorted(in_window)}
    # The rule is the served provenance's.
    assert read == ["undata/p/ILO"]
    if fields is None:
        assert "coverage_below_scope" not in warnings
    else:
        warning = warnings["coverage_below_scope"]
        unserved = {"count": 1, "refs": [f"dcid:{_YOUTH}"]}
        assert {field: warning[field] for field in _MEMBER_FIELDS if field in warning} == {
            "member_count": 3,
            **dict.fromkeys(fields, unserved),
        }


@pytest.mark.parametrize("month_first", [True, False])
def test_a_tie_goes_to_the_period_of_the_best_ranked_facet_a_counted_member_holds(month_first):
    # The rate dates unstated facet X by month in the window. The youth rate dates its copy
    # of X by year before the window, so that copy does not count, and holds an annual facet
    # Y in it. Each period serves one member; X ranks first, but only the rate's copy of it
    # counts, so the rate's period wins whichever member comes first.
    def dated(facet_id: str, dates: list[str]) -> Facet:
        return Facet(
            facet_id=facet_id,
            provenance_id="undata/p/ILO",
            unit="Percent",
            obs_count=len(dates),
            earliest_date=dates[0],
            latest_date=dates[-1],
            dates=tuple(dates),
            dates_complete=True,
        )

    monthly = [dated("X", [f"{year}-01" for year in range(2012, 2016)])]
    annual = [dated("X", [str(year) for year in range(2000, 2004)]), dated("Y", ["2014"])]
    candidates = [monthly, annual] if month_first else [annual, monthly]
    expected = [("X",), ()] if month_first else [(), ("X",)]
    requests = [replace(_request(stat_var), window=_EARLY_WINDOW) for stat_var in (_RATE, _YOUTH)]

    election = select_group_facets(requests, candidates, graph=FakeGraphReader()).election

    assert election.cadence == Cadence(facet_id="X")
    assert [
        joint_dispatch_facet_ids(member, election, member=index)
        for index, member in enumerate(candidates)
    ] == expected


@pytest.mark.parametrize("annual_first", [True, False])
def test_a_null_provenance_read_cites_only_facets_its_served_group_holds(annual_first):
    # A search read does not require provenance, so an unsourced top facet is selected
    # (`group_served`), and search refuses a pair whose selection witnesses fall outside the
    # served group. The annual facet carries no provenance either and comes first in graph
    # order, but the quarterly one is elected and served, so every witness must name it.
    def unsourced(facet_id: str, period: str, obs_count: int, latest: str) -> Facet:
        return Facet(
            facet_id=facet_id,
            unit="Percent",
            observation_period=period,
            obs_count=obs_count,
            earliest_date="2019",
            latest_date=latest,
        )

    facets = [unsourced("fA", "P1Y", 3, "2022"), unsourced("fQ", "P3M", 12, "2022-10")]
    candidates = facets if annual_first else facets[::-1]
    request = replace(_request(_RATE), requires_provenance=False)

    election = _select_member_facets(request, candidates, graph=FakeGraphReader())
    group = served_alone(candidates, stat_var=request.stat_var, window=request.window)

    assert joint_dispatch_facet_ids(candidates, election, member=0) == ("fQ",)
    assert [facet.facet_id for facet in group] == ["fQ"]
    assert {selection.dispatch_facet_id for selection in election.selections.values()} == {"fQ"}
