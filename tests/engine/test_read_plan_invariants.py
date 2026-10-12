"""An answer states what its preview serves: coverage's count and window, its sources and
served-member dimensions, and the headline's year are computed from the read plan, and the
preview reads that same plan.

Each role path (named or discovered destinations, read per pair or in bulk) runs end to end
through the fake coverage index and SDMX doubles under each kind of request, and the answer
is compared with the points its preview delivered.
"""

from __future__ import annotations

import functools
from dataclasses import replace

import httpx
import pytest

import search.engine.region.observations.fetch as region_fetch
import search.engine.region.phase_answer as phase_answer
import search.engine.retrieval.read_plan as read_plan
import search.engine.retrieval.role_resolve as role_resolve
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise import MaterialisationRequest
from materialise.generation import _select_member_facets
from materialise.window import RequestedWindow
from search.engine.clients.graph import Facet
from search.engine.extract import DateRequest
from search.engine.retrieval.coverage import (
    CoverageForm,
    answer_coverage,
    answer_headline,
    citable_source_ids,
)
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.read_plan import BulkPlan, PairPlan, ReadPlan
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.role_resolve import _role_plan
from search.engine.retrieval.role_scope import discover_role_scope
from search.models import BreadthDim, CoverageBreadth, CoverageExact, Spec, TimeWindow
from search.render.plan import render_sentence
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests.engine import test_role_bearing_pipeline_e2e as role_bearing
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    _SDMX_CSV_CONTENT_TYPE,
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
)
from tests.engine.test_facet_election_agreement import _role_materialise_sdmx
from tests.engine.test_population_headline import _yearly
from tests.materialise.conftest import FakeGraphReader

_KEN, _NOR, _UGA = "country/KEN", "country/NOR", "country/UGA"
_BASE = "dc/base/WorldBankIDS"

# Kenya holds recorded `f-rec` values to 2024 and an `Estimated` projection `f-est` to 2032;
# Norway holds only `f-old`, which ends in 2019; Uganda holds `f-rec` from 2012 to 2022.
_FACETS: dict[str, list[tuple[str, range, str | None]]] = {
    _KEN: [("f-rec", range(2010, 2025), None), ("f-est", range(2025, 2033), "Estimated")],
    _NOR: [("f-old", range(1990, 2020), None)],
    _UGA: [("f-rec", range(2012, 2023), None)],
}
_NAMES = {"Kenya": _KEN, "Norway": _NOR, "Uganda": _UGA}

_REQUESTS = {
    "latest": DateRequest(window=None, latest=True),
    "closed": DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False),
    "open": DateRequest(window=TimeWindow(start_year=2020, end_year=None), latest=False),
    "until": DateRequest(window=TimeWindow(start_year=None, end_year=2020), latest=False),
    "future": DateRequest(window=TimeWindow(start_year=2028, end_year=None), latest=False),
    "dateless": None,
}
# The pinned clock reads 2026, so "since 2020" reads 2020 to 2026 and states its open end at
# the latest year served.
_READ_WINDOW = {
    "closed": TimeWindow(start_year=2015, end_year=2020),
    "open": TimeWindow(start_year=2020, end_year=2026),
    # The coverage horizon starts in 1990 here, Norway's first year.
    "until": TimeWindow(start_year=1990, end_year=2020),
    # "Since 2028" asks about the future, so it reads to the horizon's last year, 2032.
    "future": TimeWindow(start_year=2028, end_year=2032),
}


def _answer(monkeypatch, search_settings, *, bulk: bool, discovered: bool, request: str) -> Spec:
    """Kenya's, Norway's and Uganda's debt flows, named as destinations or discovered on
    the open axis, read per pair or, with `bulk`, in one read."""
    result = _resolve(
        monkeypatch, search_settings, bulk=bulk, discovered=discovered, request=request
    )
    assert result.status == "definite", result
    return result.specs[0]


def _resolve(
    monkeypatch,
    search_settings,
    *,
    bulk: bool,
    discovered: bool,
    request: str | DateRequest,
):
    search_settings(obs_preview_max_pairs=1 if bulk else 120)
    rows = [
        (role_bearing._SV_DCID, facet_id, ABSENT_ENTITY, recipient, list(years))
        for recipient, facets in _FACETS.items()
        for facet_id, years, _method in facets
    ]
    index = patch_coverage(
        monkeypatch,
        rows,
        role_names={role_bearing._SV_DCID: RoleNames(donor=None, recipient="debtor")},
        # Every DFO import states `P1Y`; a coverage facet stating no period is a cadence of
        # its own (`materialise.facets.cadences_by_member`).
        facet_metadata={
            facet_id: FacetMetadata(
                provenance_id=_BASE, measurement_method=method, observation_period="P1Y"
            )
            for facets in _FACETS.values()
            for facet_id, _years, method in facets
        },
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
                for recipient, facets in _FACETS.items()
            }
        },
        availability={},
    )
    return role_bearing._run(
        entities=[] if discovered else list(_NAMES),
        sv_dcid=role_bearing._SV_DCID,
        sv_arcs=role_bearing._declaring_arcs(observation_properties=["debtor"]),
        resolve=_NAMES,
        sv_score=0.9,
        role_query="debt flows" if discovered else "debt flows to Kenya, Norway and Uganda",
        sdmx=sdmx,
        include_observations=True,
        materialise_graph=FakeGraphReader(),
        materialise_sdmx=_role_materialise_sdmx(_FACETS),
        date_request=_REQUESTS[request] if isinstance(request, str) else request,
        coverage=FakeCoverageService(index).snapshot(),
    )


@pytest.mark.parametrize("request_kind", list(_REQUESTS))
@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
@pytest.mark.parametrize("discovered", [False, True], ids=["named", "discovered"])
def test_an_answer_states_the_points_its_preview_serves(
    monkeypatch, search_settings, request_kind: str, bulk: bool, discovered: bool
) -> None:
    spec = _answer(
        monkeypatch, search_settings, bulk=bulk, discovered=discovered, request=request_kind
    )
    observations = spec.resolved.observations
    assert observations is not None and observations.omitted is None
    points = [(s, p) for s in observations.series for p in s.points]
    years = [int(p.date[:4]) for _s, p in points]
    assert years, "the preview must deliver a point, or the comparison is vacuous"
    coverage = spec.resolved.coverage
    dims = {d.label: d.count for d in coverage.dimensions or []}

    if request_kind == "latest":
        assert coverage.window == TimeWindow(start_year=min(years), end_year=max(years))
    elif request_kind == "dateless":
        assert coverage.window is None
    else:
        read = _READ_WINDOW[request_kind]
        assert all(read.start_year <= y <= read.end_year for y in years)
        # Every bound, named or open, is stated at the years served.
        assert coverage.window == TimeWindow(start_year=min(years), end_year=max(years))

    if isinstance(coverage, CoverageExact):
        assert coverage.observation_count == len(points)
    else:
        assert isinstance(coverage, CoverageBreadth)
        served_recipients = {
            e.ref for s in observations.series for e in s.entities if e.role == "destination"
        }
        assert dims["recipients"] == len(served_recipients)
    assert dims["sources"] == len({s.facet["provenance"].ref for s in observations.series})
    assert dims["variables"] == len({s.stat_var.ref for s in observations.series})
    headline = spec.resolved.headline
    assert headline is not None and headline.year == max(years)


# ---------------------------------------------------------------------------
# The cases where coverage and the preview once disagreed.
# ---------------------------------------------------------------------------

_DEBTOR = {"V": RoleNames(donor=None, recipient="debtor")}


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
@pytest.mark.parametrize("discovered", [False, True], ids=["named", "discovered"])
def test_an_open_start_states_the_first_year_served_not_the_horizon_floor(
    monkeypatch, search_settings, bulk: bool, discovered: bool
) -> None:
    """ "Until 2020" reads from the coverage horizon's first year, but the answer serves
    points from 1990 (Norway) onward: coverage's window, the `when` slot and the sentence
    state 1990 to 2020, and the horizon floor appears nowhere a reader sees."""
    spec = _answer(monkeypatch, search_settings, bulk=bulk, discovered=discovered, request="until")
    observations = spec.resolved.observations
    assert observations is not None
    first = min(int(p.date[:4]) for s in observations.series for p in s.points)
    served = TimeWindow(start_year=first, end_year=2020)

    assert spec.resolved.coverage.window == served
    when_slot = next(slot for slot in spec.resolved.slots if slot.key.axis == "when")
    assert when_slot.binding.value.time_window == served
    sentence = render_sentence(spec)
    assert str(first) in sentence and "1960" not in sentence, sentence


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
@pytest.mark.parametrize("discovered", [False, True], ids=["named", "discovered"])
def test_a_future_open_window_states_the_projection_years_it_serves(
    monkeypatch, search_settings, bulk: bool, discovered: bool
) -> None:
    """ "Since 2028" under the 2026 clock reads Kenya's projections to the horizon's last
    year on every route, and the window, `when` slot and sentence state 2028 to 2032."""
    spec = _answer(monkeypatch, search_settings, bulk=bulk, discovered=discovered, request="future")
    observations = spec.resolved.observations
    assert observations is not None and observations.omitted is None
    served = TimeWindow(start_year=2028, end_year=2032)
    assert sorted(p.date for s in observations.series for p in s.points) == [
        str(y) for y in range(2028, 2033)
    ]
    assert spec.resolved.coverage.window == served
    when_slot = next(slot for slot in spec.resolved.slots if slot.key.axis == "when")
    assert when_slot.binding.value.time_window == served
    assert "2028 to 2032" in render_sentence(spec)


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
@pytest.mark.parametrize("discovered", [False, True], ids=["named", "discovered"])
def test_a_window_past_the_horizon_answers_no_observations(
    monkeypatch, search_settings, bulk: bool, discovered: bool
) -> None:
    """A window starting after the horizon's last year (2032) has no year to read: every
    route answers `no_observations`, never an inverted window or a too-large preview."""
    for window in (TimeWindow(start_year=2040), TimeWindow(start_year=2040, end_year=2045)):
        result = _resolve(
            monkeypatch,
            search_settings,
            bulk=bulk,
            discovered=discovered,
            request=DateRequest(window=window, latest=False),
        )
        assert result.status == "no_data", result
        assert result.no_data_reason == "no_observations"


@pytest.mark.parametrize("discovered", [False, True], ids=["named", "discovered"])
@pytest.mark.parametrize("request_kind", ["latest", "closed", "open", "until", "dateless"])
def test_a_two_facet_group_counts_the_points_materialise_collapses_it_to(
    monkeypatch, search_settings, request_kind: str, discovered: bool
) -> None:
    """Kenya's `f-rec` (2010 to 2024) and `f-dup` (2015 to 2024) share a provenance, method
    and unit, so the preview dispatches both and materialise keeps one row per date:
    coverage and the headline count one point per series and date."""
    monkeypatch.setattr(
        "tests.engine.test_read_plan_invariants._FACETS",
        {
            _KEN: [("f-rec", range(2010, 2025), None), ("f-dup", range(2015, 2025), None)],
            _NOR: [("f-old", range(1990, 2020), None)],
            _UGA: [("f-rec", range(2012, 2023), None)],
        },
    )
    spec = _answer(
        monkeypatch, search_settings, bulk=False, discovered=discovered, request=request_kind
    )
    observations = spec.resolved.observations
    assert observations is not None
    points = [
        (next(e.ref for e in s.entities if e.role == "destination"), p.date)
        for s in observations.series
        for p in s.points
    ]
    assert len(points) == len(set(points))
    coverage = spec.resolved.coverage
    if isinstance(coverage, CoverageExact):
        assert coverage.observation_count == len(points)
        assert {d.label: d.count for d in coverage.dimensions}["observations"] == max(
            sum(1 for place, _d in points if place == p) for p, _d in points
        )
    headline = spec.resolved.headline
    assert headline is not None
    in_year = {place for place, date in points if int(date[:4]) == headline.year}
    assert headline.series == headline.observations == len(in_year)


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
@pytest.mark.parametrize("discovered", [False, True], ids=["named", "discovered"])
def test_a_window_between_recorded_and_projected_years_answers_no_observations(
    monkeypatch, search_settings, bulk: bool, discovered: bool
) -> None:
    """Every place's recorded data ends in 2019 and Kenya's projections start in 2028, so
    "since 2020" (2020 to the 2026 clock) reads nothing. Confirmation uses the plan's
    years, so every route answers `no_observations`, never a definite answer stating
    2020 to 2026 with no point."""
    monkeypatch.setattr(
        "tests.engine.test_read_plan_invariants._FACETS",
        {
            _KEN: [("f-rec", range(2010, 2020), None), ("f-est", range(2028, 2033), "Estimated")],
            _NOR: [("f-old", range(1990, 2020), None)],
            _UGA: [("f-rec", range(2012, 2020), None)],
        },
    )
    since = DateRequest(window=TimeWindow(start_year=2020, end_year=None), latest=False)
    result = _resolve(monkeypatch, search_settings, bulk=bulk, discovered=discovered, request=since)
    assert result.status == "no_data", result
    assert result.no_data_reason == "no_observations"


def test_the_unit_election_reads_the_plans_years_not_the_stated_window(
    monkeypatch, search_settings
) -> None:
    """ "Since 2018 in euros": the plan reads 2018 to 2026, while coverage states the years
    served, 2018 to 2024. The unit election picks among the facets of the years the
    preview reads, so it takes the plan's years."""
    captured: list[object] = []
    real = phase_answer.decide_part_unit

    def _spy(*args, **kwargs):
        captured.append(kwargs["years"])
        return real(*args, **kwargs)

    monkeypatch.setattr(phase_answer, "decide_part_unit", _spy)
    since = DateRequest(window=TimeWindow(start_year=2018, end_year=None), latest=False)
    monkeypatch.setitem(_REQUESTS, "since_2018", since)
    monkeypatch.setattr(role_bearing, "_run", functools.partial(role_bearing._run, unit_text="EUR"))

    _resolve(monkeypatch, search_settings, bulk=False, discovered=False, request="since_2018")

    assert captured == [TimeWindow(start_year=2018, end_year=2026)]


def _discover(index, date_request, search_settings, max_pairs: int = 120) -> Materialised:
    search_settings(obs_preview_max_pairs=max_pairs)
    result = discover_role_scope(
        surviving_svs=["V"],
        admitted_dcids=["V"],
        member_facts={},
        member_population_exact=True,
        open_flow_axis="destination",
        role_name="debtor",
        opposite_role_name=None,
        opposite_dcids=(),
        facet_filters={},
        date_request=date_request,
        coverage_snapshot=FakeCoverageService(index).snapshot(),
    )
    assert isinstance(result, Materialised)
    return result


def _named(index, recipients: list[str], date_request) -> ReadPlan:
    return _role_plan(
        ["V"],
        member_facts={"V": ArcsMemberFacts(arcs={})},
        names={"destination": "debtor"},
        subject_dcids=recipients,
        origin_dcids=[],
        bound_roles={"recipient": frozenset(recipients)},
        years=read_plan.role_years(date_request, index.horizon),
        coverage_index=index,
    )


def test_a_windowed_discovered_pair_dates_and_cites_the_facet_materialise_serves(
    search_settings,
) -> None:
    """Kenya holds `f-long` 1970-2021 (Long) and `f-short` 2020-2022 (Short). Inside 2020 to
    2022 the plan elects `f-short`, which holds three points there to `f-long`'s two;
    materialise, handed the plan's candidates and group, dispatches it; coverage cites Short
    and the headline names 2022."""
    index = build_coverage_index(
        [
            ("V", "f-long", ABSENT_ENTITY, _KEN, list(range(1970, 2022))),
            ("V", "f-short", ABSENT_ENTITY, _KEN, [2020, 2021, 2022]),
        ],
        role_names=_DEBTOR,
        facet_metadata={
            "f-long": FacetMetadata(provenance_id="dc/base/Long"),
            "f-short": FacetMetadata(provenance_id="dc/base/Short"),
        },
    )
    window = DateRequest(window=TimeWindow(start_year=2020, end_year=2022), latest=False)
    result = _discover(index, window, search_settings)
    plan = result.read_plan
    assert isinstance(plan, PairPlan)
    (series,) = plan.served
    request = MaterialisationRequest(
        stat_var="dcid:V",
        role_bindings={"debtor": [_KEN]},
        declared_roles=("debtor",),
        facet_bindings={},
        window=RequestedWindow(kind="range", start_year=2020, end_year=2022),
        candidate_filter=lambda f: f.facet_id in series.group_ids,
    )
    selection = _select_member_facets(
        request,
        [f for f in series.candidates if request.candidate_filter(f)],
        graph=FakeGraphReader(),
    ).selections

    assert selection["provenance"].dispatch_facet_id == "f-short"
    assert result.coverage.dimensions[1] == BreadthDim(label="sources", count=1)
    assert result.headline is not None and result.headline.year == 2022


def test_a_discovered_bulk_scope_counts_the_recipients_its_one_facet_serves(
    search_settings,
) -> None:
    """Kenya holds `f-a` to 2024 and Norway `f-b` to 2022; above the cap one facet serves,
    so coverage counts Kenya alone, as the headline does."""
    index = build_coverage_index(
        [
            ("V", "f-a", ABSENT_ENTITY, _KEN, list(range(2010, 2025))),
            ("V", "f-b", ABSENT_ENTITY, _NOR, list(range(2010, 2023))),
        ],
        role_names=_DEBTOR,
        facet_metadata={
            "f-a": FacetMetadata(provenance_id="dc/base/ImportA"),
            "f-b": FacetMetadata(provenance_id="dc/base/ImportB"),
        },
    )
    result = _discover(index, DateRequest(window=None, latest=True), search_settings, max_pairs=1)
    dims = {d.label: d.count for d in result.coverage.dimensions or []}

    assert isinstance(result.read_plan, BulkPlan)
    assert result.headline is not None
    assert dims["recipients"] == result.headline.role_counts["debtor"] == 1


_USA, _FRA, _CAN, _GBR = "country/USA", "country/FRA", "country/CAN", "country/GBR"


@pytest.mark.parametrize("max_pairs", [1, 120], ids=["bulk", "per_pair"])
@pytest.mark.parametrize(
    ("request_", "window"),
    [
        (DateRequest(window=None, latest=True), TimeWindow(start_year=2024, end_year=2024)),
        (
            DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False),
            TimeWindow(start_year=2015, end_year=2020),
        ),
    ],
    ids=["latest", "closed"],
)
def test_a_discovered_provider_scope_counts_the_creditors_of_the_named_debtors(
    search_settings, max_pairs: int, request_: DateRequest, window: TimeWindow
) -> None:
    """Debt flows to Kenya and Nigeria from any creditor: the United States and France lend
    to Kenya and Canada to Nigeria, while Britain lends only to Tanzania, which the question
    does not name. Coverage counts three providers on the creditor axis, never the two
    debtors and never Britain, states the years served, and the headline counts the same
    three creditors."""
    every_year = list(range(2010, 2025))
    index = build_coverage_index(
        [
            ("V", "f-a", _USA, _KEN, every_year),
            ("V", "f-a", _FRA, _KEN, every_year),
            ("V", "f-a", _CAN, "country/NGA", every_year),
            ("V", "f-a", _GBR, "country/TZA", every_year),
        ],
        role_names={"V": RoleNames(donor="creditor", recipient="debtor")},
        facet_metadata={"f-a": FacetMetadata(provenance_id="dc/base/ImportA")},
    )
    search_settings(obs_preview_max_pairs=max_pairs)
    result = discover_role_scope(
        surviving_svs=["V"],
        admitted_dcids=["V"],
        member_facts={},
        member_population_exact=True,
        open_flow_axis="origin",
        role_name="creditor",
        opposite_role_name="debtor",
        opposite_dcids=(_KEN, "country/NGA"),
        facet_filters={},
        date_request=request_,
        coverage_snapshot=FakeCoverageService(index).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert isinstance(result.read_plan, BulkPlan if max_pairs == 1 else PairPlan)
    assert isinstance(result.coverage, CoverageBreadth)
    assert result.coverage.dimensions == [
        BreadthDim(label="providers", count=3),
        BreadthDim(label="sources", count=1),
        BreadthDim(label="variables", count=1),
    ]
    assert result.scope_member_dcids == [_CAN, _FRA, _USA]
    assert result.coverage.window == window
    assert result.headline is not None
    assert result.headline.role_counts["creditor"] == 3


def test_a_named_pair_with_no_point_in_the_window_cites_no_source() -> None:
    """Kenya's `f-a` (ImportA) reaches 2020 to 2022; Norway's `f-b` (ImportB) ends in 2005,
    so the plan serves Kenya alone and coverage cites one source."""
    recipients = [_KEN, _NOR]
    index = build_coverage_index(
        [
            ("V", "f-a", ABSENT_ENTITY, _KEN, list(range(2010, 2025))),
            ("V", "f-b", ABSENT_ENTITY, _NOR, list(range(2000, 2006))),
        ],
        role_names=_DEBTOR,
        facet_metadata={
            "f-a": FacetMetadata(provenance_id="dc/base/ImportA"),
            "f-b": FacetMetadata(provenance_id="dc/base/ImportB"),
        },
    )
    window = DateRequest(window=TimeWindow(start_year=2020, end_year=2022), latest=False)

    coverage = answer_coverage(_named(index, recipients, window), CoverageForm())

    assert coverage.dimensions[0] == BreadthDim(label="sources", count=1)


def test_an_open_ended_bulk_read_counts_and_dates_only_the_years_it_reads(search_settings) -> None:
    """Three debtors hold `f1` from 2015 to 2030. "Since 2020" reads 2020 to 2026 in bulk,
    so coverage counts 21 points and the headline names 2026, not the 2030 projection."""
    search_settings(obs_preview_max_pairs=2)
    recipients = [_KEN, _NOR, _UGA]
    index = build_coverage_index(
        [("V", "f1", ABSENT_ENTITY, r, list(range(2015, 2031))) for r in recipients],
        role_names=_DEBTOR,
        facet_metadata={"f1": FacetMetadata(provenance_id="dc/base/Ids")},
    )
    since = DateRequest(window=TimeWindow(start_year=2020, end_year=None), latest=False)
    plan = _named(index, recipients, since)

    coverage = answer_coverage(plan, CoverageForm())
    headline = answer_headline(plan, index, date_request=since)

    assert isinstance(plan, BulkPlan)
    assert plan.read_years == tuple(str(y) for y in range(2020, 2027))
    assert isinstance(coverage, CoverageExact)
    assert (coverage.observation_count, coverage.window) == (
        21,
        TimeWindow(start_year=2020, end_year=2026),
    )
    assert headline is not None and headline.year == 2026


# ---------------------------------------------------------------------------
# A named destination with the origin left open.
# ---------------------------------------------------------------------------

_AAA, _CCC = "country/AAA", "country/CCC"
_CRA, _CRB, _CRC = "country/CRA", "country/CRB", "country/CRC"
_DEBTOR_NAMES = {"Aland": _AAA, "Cland": _CCC}
_IDS = "dc/base/Ids"
# Each facet's provenance: `f2` comes from another source, so it never joins `f1`'s group.
_SOURCE = {"f1": _IDS, "f2": "dc/base/Other"}
type _OpenOriginHoldings = dict[tuple[str, str], list[tuple[str, range]]]

# Two debtors and two creditors left open, one facet per pair.
_OPEN_ORIGIN: _OpenOriginHoldings = {
    (creditor, debtor): [("f1", years)]
    for debtor, years in ((_AAA, range(2010, 2024)), (_CCC, range(2012, 2025)))
    for creditor in (_CRA, _CRB)
}


def _open_origin_handler(holdings: _OpenOriginHoldings):
    """Serve each dispatched facet for every creditor of the debtor bound, or for the
    creditor bound too, or each one's latest row under `LATEST`."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        body = _csv_header(["creditor", "debtor"])
        debtors = params.get("c[debtor]", "").split(",")
        creditor = params.get("c[creditor]")
        facet_ids = params.get("c[facetId]")
        latest = params.get("c[TIME_PERIOD]") == "LATEST"
        for (c, d), held in holdings.items():
            if d not in debtors or (creditor is not None and c not in creditor.split(",")):
                continue
            for facet_id, years in held:
                if facet_ids is not None and facet_id not in facet_ids.split(","):
                    continue
                fixture = ObsFacetFixture(facet_id=facet_id, provenance_id=_SOURCE[facet_id])
                for year in [years[-1]] if latest else years:
                    body += _csv_row(
                        stat_var=role_bearing._SV_DCID,
                        role_dcids=[c, d],
                        fixture=fixture,
                        date_str=str(year),
                        value=1.0,
                    )
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    return handler


def _open_origin_answer(
    monkeypatch,
    search_settings,
    *,
    request: str | DateRequest,
    bulk: bool = False,
    max_pairs: int | None = None,
    discovered: bool = False,
    holdings: _OpenOriginHoldings = _OPEN_ORIGIN,
    live_holdings: _OpenOriginHoldings | None = None,
    requests: list[httpx.Request] | None = None,
) -> Spec:
    """Debt flows to the debtors of `holdings`, named or discovered, with the creditors
    left open. `requests` records each SDMX request the preview makes. `live_holdings`, when
    given, is what the preview's SDMX read returns in place of `holdings`, as when coverage
    lags the live data."""
    serve = _open_origin_handler(holdings if live_holdings is None else live_holdings)

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return serve(request)

    search_settings(obs_preview_max_pairs=max_pairs or (1 if bulk else 120))
    index = patch_coverage(
        monkeypatch,
        [
            (role_bearing._SV_DCID, facet_id, c, d, list(years))
            for (c, d), held in holdings.items()
            for facet_id, years in held
        ],
        role_names={role_bearing._SV_DCID: RoleNames(donor="creditor", recipient="debtor")},
        facet_metadata={
            facet_id: FacetMetadata(provenance_id=_SOURCE[facet_id])
            for held in holdings.values()
            for facet_id, _years in held
        },
    )
    sdmx = FakeSdmxClient(
        observations={
            role_bearing._SV_DCID: {
                (("creditor", c), ("debtor", d)): [
                    replace(_yearly(years), facet_id=facet_id, provenance_id=_SOURCE[facet_id])
                    for facet_id, years in held
                ]
                for (c, d), held in holdings.items()
            }
        },
        availability={},
    )
    debtors = sorted({d for _c, d in holdings})
    names = [name for name, dcid in _DEBTOR_NAMES.items() if dcid in debtors]
    result = role_bearing._run(
        entities=[] if discovered else names,
        sv_dcid=role_bearing._SV_DCID,
        sv_arcs=role_bearing._declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve=_DEBTOR_NAMES,
        sv_score=0.9,
        role_query="debt flows" if discovered else "debt flows to " + " and ".join(names),
        sdmx=sdmx,
        include_observations=True,
        materialise_graph=FakeGraphReader(),
        materialise_sdmx=MaterialiseSdmxClient(
            base_url=MATERIALISE_BASE_URL,
            transport=httpx.MockTransport(handler),
        ),
        provenance_labels={source: source for source in _SOURCE.values()},
        date_request=_REQUESTS[request] if isinstance(request, str) else request,
        coverage=FakeCoverageService(index).snapshot(),
    )
    assert result.status == "definite", result
    return result.specs[0]


def _served_by_pair(spec: Spec) -> dict[tuple[str, str], list[str]]:
    """Each served series' dates, keyed by its creditor and debtor."""
    observations = spec.resolved.observations
    assert observations is not None and observations.omitted is None
    return {
        (
            next(e.ref for e in s.entities if e.role == "origin"),
            next(e.ref for e in s.entities if e.role == "destination"),
        ): [p.date for p in s.points]
        for s in observations.series
    }


def _assert_counts_what_it_serves(spec: Spec, served: dict[tuple[str, str], list[str]]) -> None:
    """Coverage counts the served points (or, for a discovered scope, the debtors served);
    the headline names the latest served year and counts the series, creditors and debtors
    with a point in it."""
    points = [(key, date) for key, dates in served.items() for date in dates]
    coverage = spec.resolved.coverage
    if isinstance(coverage, CoverageBreadth):
        dims = {d.label: d.count for d in coverage.dimensions or []}
        assert dims["recipients"] == len({d for _c, d in served})
    else:
        assert isinstance(coverage, CoverageExact)
        assert coverage.observation_count == len(points)
    headline = spec.resolved.headline
    assert headline is not None
    assert headline.year == max(int(date[:4]) for _key, date in points)
    in_year = [key for key, date in points if int(date[:4]) == headline.year]
    assert headline.series == headline.observations == len(in_year)
    assert headline.role_counts == {
        "creditor": len({c for c, _d in in_year}),
        "debtor": len({d for _c, d in in_year}),
    }


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
@pytest.mark.parametrize("request_kind", ["latest", "closed", "dateless"])
def test_an_open_origin_serves_one_series_per_creditor(
    monkeypatch, search_settings, request_kind: str, bulk: bool
) -> None:
    """Two named debtors and two creditors left open: both routes serve one series per
    creditor and debtor, under `latest` those reaching the variable's latest year, and
    coverage and the headline count those series and points."""
    spec = _open_origin_answer(monkeypatch, search_settings, bulk=bulk, request=request_kind)
    served = _served_by_pair(spec)
    expected: dict[tuple[str, str], list[str]] = {}
    for (creditor, debtor), [(_facet_id, years)] in _OPEN_ORIGIN.items():
        if request_kind == "closed":
            expected[creditor, debtor] = [str(y) for y in range(2015, 2021)]
        elif request_kind == "dateless":
            expected[creditor, debtor] = [str(y) for y in years]
        elif years[-1] == 2024:
            # A `latest` read serves the variable's latest year, 2024, alone, on both routes.
            expected[creditor, debtor] = [str(years[-1])]
    assert served == expected
    _assert_counts_what_it_serves(spec, served)


@pytest.mark.parametrize("discovered", [False, True], ids=["named", "discovered"])
@pytest.mark.parametrize("request_kind", ["latest", "closed", "dateless"])
@pytest.mark.parametrize("creditors", [1, 5, 40])
def test_an_open_origin_preview_makes_one_read_per_debtor_however_many_creditors(
    monkeypatch, search_settings, creditors: int, request_kind: str, discovered: bool
) -> None:
    """Two debtors and 1, 5 or 40 creditors left open, each creditor's series ending in its
    own year. The preview makes one read per debtor, binding the debtor, leaving the creditor
    open and dispatching the variable's group, and still serves one series per creditor with
    that creditor's own dates, and under `latest` one per creditor reaching the variable's
    latest year."""
    holdings: _OpenOriginHoldings = {
        (f"country/CR{i:02d}", debtor): [("f1", range(2008 + i % 5, 2019 + i % 6))]
        for i in range(creditors)
        for debtor in (_AAA, _CCC)
    }
    requests: list[httpx.Request] = []
    spec = _open_origin_answer(
        monkeypatch,
        search_settings,
        request=request_kind,
        discovered=discovered,
        holdings=holdings,
        requests=requests,
    )

    assert sorted(
        (r.url.params.get("c[debtor]"), r.url.params.get("c[creditor]"), r.url.params["c[facetId]"])
        for r in requests
    ) == [(_AAA, None, "f1"), (_CCC, None, "f1")]
    served = _served_by_pair(spec)
    newest = max(years[-1] for [(_facet_id, years)] in holdings.values())
    expected: dict[tuple[str, str], list[str]] = {}
    for pair, [(_facet_id, years)] in holdings.items():
        if request_kind == "closed":
            dates = [str(y) for y in years if 2015 <= y <= 2020]
        elif request_kind == "dateless":
            dates = [str(y) for y in years]
        else:
            dates = [str(years[-1])] if years[-1] == newest else []
        if dates:
            expected[pair] = dates
    assert served == expected
    _assert_counts_what_it_serves(spec, served)


@pytest.mark.parametrize(
    ("request_kind", "headline_year"),
    [
        (None, 2020),
        (DateRequest(window=TimeWindow(start_year=None, end_year=2019), latest=False), 2019),
    ],
    ids=["dateless", "until_2019"],
)
def test_an_open_origin_headline_reads_the_group_the_variable_serves(
    monkeypatch, search_settings, request_kind: DateRequest | None, headline_year: int
) -> None:
    """At Aland, CRA holds `f1` to 2020; CRC holds `f1` to 2018 and `f2` from 2015 to 2024.
    `f1` covers both creditors, so it is the variable's one group and no series serves `f2`.
    The headline counts only the rows of the group served, so it names 2020 (or 2019 until
    2019) and one series, never CRC's unserved `f2` rows."""
    holdings: _OpenOriginHoldings = {
        (_CRA, _AAA): [("f1", range(2010, 2021))],
        (_CRC, _AAA): [("f1", range(1980, 2019)), ("f2", range(2015, 2025))],
    }
    request = request_kind or DateRequest(window=None, latest=False)
    spec = _open_origin_answer(
        monkeypatch,
        search_settings,
        request=request if request_kind is not None else "dateless",
        holdings=holdings,
    )
    served = _served_by_pair(spec)
    assert max(served[_CRC, _AAA]) == "2018"
    _assert_counts_what_it_serves(spec, served)
    headline = spec.resolved.headline
    assert headline is not None
    assert (headline.year, headline.series) == (headline_year, 1)


@pytest.mark.parametrize("discovered", [False, True], ids=["named", "discovered"])
@pytest.mark.parametrize(("max_pairs", "route"), [(3, "bulk"), (4, "per_pair")])
def test_an_open_origin_switches_to_bulk_when_its_creditor_series_pass_the_cap(
    monkeypatch, search_settings, max_pairs: int, route: str, discovered: bool
) -> None:
    """Two debtors bound and two creditors open are two bound pairs but four series. A cap
    of 3 sends the read to bulk, a cap of 4 keeps it per pair, and both serve the same four
    series."""
    calls: dict[str, int] = {"bulk": 0, "per_pair": 0}
    real_bulk, real_pair = region_fetch._bulk_role_series, region_fetch._preview_one

    def _bulk(*args, **kwargs):
        calls["bulk"] += 1
        return real_bulk(*args, **kwargs)

    def _pair(*args, **kwargs):
        calls["per_pair"] += 1
        return real_pair(*args, **kwargs)

    monkeypatch.setattr(region_fetch, "_bulk_role_series", _bulk)
    monkeypatch.setattr(region_fetch, "_preview_one", _pair)
    spec = _open_origin_answer(
        monkeypatch,
        search_settings,
        request="closed",
        max_pairs=max_pairs,
        discovered=discovered,
    )

    assert calls[route] > 0 and calls["bulk" if route == "per_pair" else "per_pair"] == 0
    served = _served_by_pair(spec)
    assert served == {key: [str(y) for y in range(2015, 2021)] for key in _OPEN_ORIGIN}
    _assert_counts_what_it_serves(spec, served)


# ---------------------------------------------------------------------------
# A role answer cites the sources its plan serves.
# ---------------------------------------------------------------------------

_ALAND, _BLAND, _CREDITOR = "country/AAA", "country/BBB", "country/CRA"
_SOURCE_LABELS = {"dc/base/Ids": "IDS", "dc/base/Proj": "Projections", "dc/base/Alt": "Alternative"}
# Aland and Bland hold the recorded `f1` (2010 to 2020), and Aland a second source the 2015
# to 2020 plan serves no point from: a projection outside those years, or a shorter series inside
# them that the election ranks below `f1`.
_UNSERVED_SOURCE: dict[str, tuple[range, FacetMetadata]] = {
    "outside_the_years": (
        range(2028, 2031),
        FacetMetadata(provenance_id="dc/base/Proj", measurement_method="Projected"),
    ),
    "not_elected": (range(2016, 2019), FacetMetadata(provenance_id="dc/base/Alt")),
}


@pytest.mark.parametrize("unserved", list(_UNSERVED_SOURCE))
@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
@pytest.mark.parametrize("discovered", [False, True], ids=["named", "discovered"])
def test_a_role_answer_cites_the_sources_its_plan_serves(
    monkeypatch, search_settings, discovered: bool, bulk: bool, unserved: str
) -> None:
    """`resolved_sources`, the sentence's source and coverage's source count all state the
    facets the read plan serves, on every role route."""
    search_settings(obs_preview_max_pairs=1 if bulk else 120)
    years, metadata = _UNSERVED_SOURCE[unserved]
    flow = role_bearing._SV_DCID
    index = patch_coverage(
        monkeypatch,
        [
            (flow, "f1", _CREDITOR, _ALAND, list(range(2010, 2021))),
            (flow, "f1", _CREDITOR, _BLAND, list(range(2010, 2021))),
            (flow, "f9", _CREDITOR, _ALAND, list(years)),
        ],
        role_names={flow: RoleNames(donor="creditor", recipient="debtor")},
        facet_metadata={"f1": FacetMetadata(provenance_id="dc/base/Ids"), "f9": metadata},
    )
    plans: list[ReadPlan | None] = []
    real_filter_answer = phase_answer.filter_answer

    def _filter_answer(*args, **kwargs):
        filtered = real_filter_answer(*args, **kwargs)
        plans.append(filtered.plan)
        return filtered

    monkeypatch.setattr(phase_answer, "filter_answer", _filter_answer)

    result = role_bearing._run(
        entities=[] if discovered else ["Aland", "Bland"],
        sv_dcid=flow,
        sv_arcs=role_bearing._declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Aland": _ALAND, "Bland": _BLAND},
        sv_score=0.9,
        role_query="debt flows" if discovered else "debt flows to Aland and Bland",
        sdmx=FakeSdmxClient(observations={}, availability={}),
        provenance_labels=_SOURCE_LABELS,
        date_request=_REQUESTS["closed"],
        coverage=FakeCoverageService(index).snapshot(),
    )

    assert result.status == "definite", (result.status, result.no_data_reason)
    (plan,) = plans
    assert isinstance(plan, BulkPlan if bulk else PairPlan)
    served = citable_source_ids(read_plan.facets_served(plan))
    assert served == ["dc/base/Ids"]
    resolved = result.spec.resolved
    assert [ref.dcid for ref in resolved.trace.resolved_sources] == served
    dims = {d.label: d.count for d in resolved.coverage.dimensions or []}
    assert dims["sources"] == len(served)
    sentence = render_sentence(result.spec)
    assert sentence.endswith(" IDS.")
    assert "Projections" not in sentence and "Alternative" not in sentence


def test_a_named_route_over_the_pair_cap_builds_no_pair_series(
    monkeypatch, search_settings
) -> None:
    """Three named recipients against a cap of one pair read in bulk, and the per-pair
    series are never built."""

    def _pair_series(*args, **kwargs):
        raise AssertionError("a bulk read builds no per-pair series")

    monkeypatch.setattr(role_resolve, "pair_series", _pair_series)

    spec = _answer(monkeypatch, search_settings, bulk=True, discovered=False, request="closed")

    assert spec.resolved.coverage.has_data
