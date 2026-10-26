"""Resolve plans an answer's preview read once, from the members and entities the preview
reads, and coverage follows that plan."""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field, replace

import pytest

import search.engine.retrieval.read_plan as read_plan
from materialise.facets import bound_variable_breadth, coverage_probe
from materialise.group_facets import group_served
from materialise.series_conversion import series_key
from search.engine.app import _note_breadth_fallback
from search.engine.clients.graph import Facet
from search.engine.config import settings
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.errors import GraphInfraError
from search.engine.extract import DateRequest
from search.engine.region.convertibility import _served_keys
from search.engine.region.facets import FacetAdmission, filter_answer
from search.engine.region.observations.bulk_role_series import (
    _bulk_role_series,
    _plan_role_bulk_fetch,
)
from search.engine.region.observations.fetch import _dispatched_series, _dispatches
from search.engine.region.option_coverage import RoleServedFilter
from search.engine.retrieval.coverage import CoverageForm, answer_coverage, answer_headline
from search.engine.retrieval.download import download_facts
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.read_plan import (
    BulkPlan,
    NoYearsToRead,
    PairPlan,
    PlanRoles,
    PlanSeries,
    ReadPlan,
    WindowGap,
    asked_years,
    plan_role_read,
    plan_standard,
    plan_window,
    plan_years,
    read_range,
    served_dates,
    unserved_entities,
    window_gap,
)
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.role_resolve import _elect_role_class, _role_plan
from search.engine.retrieval.role_scope import discover_role_scope, pair_series
from search.models import CoverageBreadth, CoverageExact, TimeWindow
from search.observations import observation_settings
from series_coverage.coverage_types import BoundRoles
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames, YearHorizon
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._fake_sdmx import CountingSdmxClient, FakeSdmxClient, plan_observations
from tests.fixtures import FakeGraph

_TOTAL = "TEST/PreviewRoute_Total"
_HEALTH = "TEST/PreviewRoute_BySector_Health"
_CONSTRAINTS = {_TOTAL: {}, _HEALTH: {"sector": "Health"}}


@dataclass(frozen=True)
class _Constraints:
    """Member facts that read only what the plan and the role-class election need: resolved
    constraints, one declared role and no elected dataset."""

    constraints: dict[str, str] = field(default_factory=dict)

    def constraints_known(self) -> bool:
        return True

    def read_constraints(self) -> dict[str, str]:
        return dict(self.constraints)

    def read_unresolved_constraints(self) -> frozenset[str]:
        return frozenset()

    def read_declared_roles(self) -> tuple[str, ...]:
        return ("debtor",)

    def read_dataset(self) -> None:
        return None


_FAMILY_FACTS = {sv: _Constraints(c) for sv, c in _CONSTRAINTS.items()}


def _recipients(count: int) -> list[str]:
    return [f"country/R{i:03d}" for i in range(count)]


def _plan(sv_dcids, entity_lists) -> ReadPlan:
    return plan_role_read(
        sv_dcids,
        entity_lists,
        years=None,
        read_years=None,
        candidates=lambda members: {},
        series_count=lambda members: 0,
        pair_series=lambda members: [],
        member_probes=lambda members: {},
        roles=PlanRoles(bound={}, role_names={"recipient": "debtor"}),
        elect=lambda plan: {},
    )


def _elected(sv_dcids: list[str]) -> list[str]:
    """The members the role path's one reducer, the role-class election, hands the plan."""
    elected = _elect_role_class(sv_dcids, _FAMILY_FACTS, ["destination"], ())
    assert elected is not None
    return elected


def test_the_plan_reads_the_members_the_role_class_election_keeps() -> None:
    """A total and its health part at 70 recipients are 140 pairs. The election keeps the
    total, so the preview reads 70 pairs, within the 120-pair cap. The plan reduces nothing
    itself: handed both members it counts 140 pairs and reads in bulk."""
    recipients = _recipients(70)
    assert 2 * len(recipients) > observation_settings().obs_preview_max_pairs

    assert _elected([_TOTAL, _HEALTH]) == [_TOTAL]
    plan = _plan(_elected([_TOTAL, _HEALTH]), [recipients])
    assert isinstance(plan, PairPlan) and plan.members == (_TOTAL,)
    handed_both = _plan([_TOTAL, _HEALTH], [recipients])
    assert isinstance(handed_both, BulkPlan) and handed_both.members == (_TOTAL, _HEALTH)
    over_cap = _recipients(observation_settings().obs_preview_max_pairs + 1)
    assert isinstance(_plan(_elected([_TOTAL, _HEALTH]), [over_cap]), BulkPlan)


def test_the_pair_count_multiplies_each_bound_roles_entities() -> None:
    """One member at 11 donors and 11 recipients is 121 pairs, over the cap."""
    assert isinstance(_plan(["sv"], [_recipients(11), _recipients(11)]), BulkPlan)
    assert isinstance(_plan(["sv"], [_recipients(11)]), PairPlan)


def test_an_open_role_reads_in_bulk_when_its_series_pass_the_cap() -> None:
    """One debtor bound and the creditors open is one bound pair, but the preview reads one
    series per creditor: 121 creditors with data send it to the bulk read, decided from the
    series count before any series is built."""
    debtor = ("debtor", "country/D")
    built: list[int] = []

    def series(creditors: int) -> list[PlanSeries]:
        built.append(creditors)
        return [
            PlanSeries(
                stat_var="sv",
                role_tuple=(debtor, ("creditor", f"country/C{i}")),
                candidates=(_facet("f1", range(2020, 2022)),),
            )
            for i in range(creditors)
        ]

    def plan(creditors: int) -> ReadPlan:
        return plan_role_read(
            ["sv"],
            [["country/D"]],
            years=None,
            read_years=None,
            candidates=lambda members: {},
            series_count=lambda members: creditors,
            pair_series=lambda members: series(creditors),
            member_probes=lambda members: {},
            roles=PlanRoles(bound={}, role_names={"recipient": "debtor", "donor": "creditor"}),
            elect=lambda plan: {},
        )

    cap = observation_settings().obs_preview_max_pairs
    assert isinstance(plan(cap), PairPlan)
    assert built == [cap]
    assert isinstance(plan(cap + 1), BulkPlan)
    assert built == [cap]


def test_named_destinations_plan_on_the_members_the_preview_reads() -> None:
    """Resolve records the per-pair plan for a total and part at 70 recipients: the
    election keeps the total alone, so the plan reads 70 pairs where both members would send
    it to bulk."""
    recipients = _recipients(70)
    index = build_coverage_index(
        [
            (sv, "f1", ABSENT_ENTITY, recipient, [2024])
            for sv in (_TOTAL, _HEALTH)
            for recipient in recipients
        ],
        role_names={sv: RoleNames(donor=None, recipient="debtor") for sv in (_TOTAL, _HEALTH)},
    )

    plan = _role_plan(
        _elected([_TOTAL, _HEALTH]),
        member_facts=_FAMILY_FACTS,
        names={"destination": "debtor"},
        subject_dcids=recipients,
        origin_dcids=[],
        bound_roles={"recipient": frozenset(recipients)},
        years=read_plan.role_years(None, index.horizon),
        coverage_index=index,
    )

    assert isinstance(plan, PairPlan) and plan.members == (_TOTAL,)
    assert {s.stat_var for s in plan.series} == {_TOTAL}
    assert plan.series[0].rows == (recipients[0], None)


def test_a_named_destination_plan_reads_a_member_whose_constraints_are_unknown_beside_the_total():
    """The election keeps a member whose constraints are unknown beside the total, as the
    mint's walk does, and the plan reads both. Reduced again by resolved constraints alone,
    the unread member would read as asserting nothing and displace the total."""
    unread = "TEST/PreviewRoute_Unread"
    total_arcs = {
        "constraintProperties": {"nodes": [{"dcid": "flowType"}]},
        "observationProperties": {"nodes": [{"dcid": "debtor"}]},
        "flowType": {"nodes": [{"dcid": "X"}]},
    }
    facts = {_TOTAL: ArcsMemberFacts(arcs=total_arcs), unread: ArcsMemberFacts(arcs={})}
    assert facts[unread].constraints_known() is False
    index = build_coverage_index(
        [
            (sv, "f1", ABSENT_ENTITY, debtor, [2024])
            for sv in (_TOTAL, unread)
            for debtor in _DEBTORS
        ],
        role_names={sv: RoleNames(donor=None, recipient="debtor") for sv in (_TOTAL, unread)},
    )

    elected = _elect_role_class([_TOTAL, unread], facts, ["destination"], ())
    plan = _role_plan(
        elected or [],
        member_facts=facts,
        names={"destination": "debtor"},
        subject_dcids=_DEBTORS,
        origin_dcids=[],
        bound_roles={"recipient": frozenset(_DEBTORS)},
        years=read_plan.role_years(None, index.horizon),
        coverage_index=index,
    )

    assert elected == [_TOTAL, unread]
    assert plan.members == (_TOTAL, unread)
    assert read_plan.minimal_constraint_svs(
        elected, {sv: member.read_constraints() for sv, member in facts.items()}
    ) == [unread]


# Kenya's pair holds `f-a` from one import to 2024, Norway's `f-b` from another to 2022.
_DISCOVERY_ROWS = [
    ("V", "f-a", ABSENT_ENTITY, "country/KEN", list(range(2010, 2025))),
    ("V", "f-b", ABSENT_ENTITY, "country/NOR", list(range(2010, 2023))),
]
_DISCOVERY_METADATA = {
    "f-a": FacetMetadata(provenance_id="dc/base/ImportA"),
    "f-b": FacetMetadata(provenance_id="dc/base/ImportB"),
}


def _discover(max_pairs: int, search_settings) -> Materialised:
    search_settings(obs_preview_max_pairs=max_pairs)
    index = build_coverage_index(
        _DISCOVERY_ROWS,
        role_names={"V": RoleNames(donor=None, recipient="debtor")},
        facet_metadata=_DISCOVERY_METADATA,
    )
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
        date_request=DateRequest(window=None, latest=True),
        coverage_snapshot=FakeCoverageService(index).snapshot(),
    )
    assert isinstance(result, Materialised)
    return result


def _sources(result: Materialised) -> int:
    assert isinstance(result.coverage, CoverageBreadth)
    return next(d.count for d in result.coverage.dimensions or [] if d.label == "sources")


def test_discovery_within_the_cap_serves_one_group_across_its_pairs(search_settings) -> None:
    """Two discovered recipients fit the cap, and the variable serves one group at every
    pair: each import covers one recipient, so the one holding more observations leads.
    Kenya's `f-a` serves, Norway's `f-b` does not, and the answer cites one source and
    dates its latest values 2024."""
    result = _discover(120, search_settings)

    plan = result.read_plan
    assert isinstance(plan, PairPlan) and plan.members == ("V",)
    assert {facet_id for s in plan.series for facet_id in s.group_ids} == {"f-a"}
    assert [s.role_tuple for s in plan.uncovered] == [(("debtor", "country/NOR"),)]
    assert _sources(result) == 1
    assert result.coverage.window == TimeWindow(start_year=2024, end_year=2024)
    assert result.headline is not None
    assert (result.headline.year, result.headline.series) == (2024, 1)


def test_discovery_above_the_cap_covers_the_one_elected_facet(search_settings) -> None:
    """Above the cap one facet serves every pair: `f-a` leads on observations, so one
    source counts and the window is its latest year."""
    result = _discover(1, search_settings)

    plan = result.read_plan
    assert isinstance(plan, BulkPlan)
    assert (plan.members, plan.years, plan.facet_id) == (("V",), "latest", "f-a")
    assert _sources(result) == 1
    assert result.coverage.window == TimeWindow(start_year=2024, end_year=2024)


# A one-role variable at three named debtors, read 2015 to 2020. `fA` holds AAA through the
# window but BBB and CCC only before it; `fB` holds AAA and BBB inside it. Both are in USD.
_FLOW = "TEST/PreviewRoute_Flow"
_AAA, _BBB, _CCC = "country/AAA", "country/BBB", "country/CCC"
_BULK_ROWS = [
    (_FLOW, "fA", ABSENT_ENTITY, _AAA, list(range(2010, 2021))),
    (_FLOW, "fA", ABSENT_ENTITY, _BBB, list(range(2000, 2006))),
    (_FLOW, "fA", ABSENT_ENTITY, _CCC, list(range(2000, 2006))),
    (_FLOW, "fB", ABSENT_ENTITY, _AAA, list(range(2015, 2021))),
    (_FLOW, "fB", ABSENT_ENTITY, _BBB, list(range(2015, 2021))),
]
_WINDOW = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)
_YEARS = [str(y) for y in range(2015, 2021)]


def _bulk_named_answer(search_settings, metadata: dict[str, FacetMetadata]):
    search_settings(obs_preview_max_pairs=1)
    index = build_coverage_index(
        _BULK_ROWS,
        role_names={_FLOW: RoleNames(donor=None, recipient="debtor")},
        facet_metadata=metadata,
    )
    plan = _role_plan(
        [_FLOW],
        member_facts={_FLOW: _Constraints()},
        names={"destination": "debtor"},
        subject_dcids=[_AAA, _BBB, _CCC],
        origin_dcids=[],
        bound_roles={"recipient": frozenset({_AAA, _BBB, _CCC})},
        years=read_plan.role_years(_WINDOW, index.horizon),
        coverage_index=index,
    )
    return index, plan


def _read_facets(plan: BulkPlan, sv_dcids: list[str]) -> set[str]:
    """The facet ids the bulk read requests for `plan`, the same in each of its reads."""
    sdmx = FakeSdmxClient(observations={})
    _bulk_role_series(plan, sv_dcids, sdmx=sdmx, graph=FakeGraph(), facet_filter=None)
    (facet_ids,) = {frozenset(call["facet_ids"]) for call in sdmx.series_calls}
    return set(facet_ids)


def _preview_facet(index, plan, unit: str) -> str | None:
    (facet_id,) = _read_facets(
        plan.narrowed(lambda _sv, candidate: candidate.unit == unit), [_FLOW]
    )
    return facet_id


def _filtered_coverage(plan, unit: str):
    """Coverage under a unit filter, as `filter_answer` builds it: the plan narrows to the
    admitted facets and coverage states the narrowed plan."""
    return answer_coverage(
        plan.narrowed(lambda _sv, candidate: candidate.unit == unit), CoverageForm()
    )


def test_named_bulk_election_ranks_the_window_the_preview_reads(search_settings) -> None:
    """Inside 2015 to 2020 `fB` covers two debtors and `fA` one, so resolve elects `fB` as
    the preview does, where lifetime series would elect `fA` over three. A unit filter that
    keeps both leaves coverage and the preview on `fB` and its 12 points."""
    metadata = {
        "fA": FacetMetadata(provenance_id="dc/base/A", unit="USD"),
        "fB": FacetMetadata(provenance_id="dc/base/B", unit="USD"),
    }
    index, plan = _bulk_named_answer(search_settings, metadata)

    assert isinstance(plan, BulkPlan)
    assert plan.facet_id == "fB"
    assert _preview_facet(index, plan, "USD") == "fB"
    coverage = _filtered_coverage(plan, "USD")
    assert isinstance(coverage, CoverageExact)
    assert coverage.observation_count == 12


def test_a_filter_that_drops_the_elected_facet_reelects_alike(search_settings) -> None:
    """A EUR filter removes `fB`, so coverage and the preview both fall back to `fA`, the
    only EUR facet, and coverage counts its one pair's six points in the window."""
    metadata = {
        "fA": FacetMetadata(provenance_id="dc/base/A", unit="EUR"),
        "fB": FacetMetadata(provenance_id="dc/base/B", unit="USD"),
    }
    index, plan = _bulk_named_answer(search_settings, metadata)

    assert plan.facet_id == "fB"
    assert _preview_facet(index, plan, "EUR") == "fA"
    assert plan.narrowed(lambda _sv, c: c.unit == "EUR").facet_id == "fA"
    coverage = _filtered_coverage(plan, "EUR")
    assert isinstance(coverage, CoverageExact)
    assert coverage.observation_count == 6


# A total and its health part at three debtors. The preview reads the total alone.
_DEBTORS = ["country/AAA", "country/BBB", "country/CCC"]
_LATEST = DateRequest(window=None, latest=True)
_FAMILY_ROLES = {sv: RoleNames(donor=None, recipient="debtor") for sv in _CONSTRAINTS}


def test_named_family_coverage_dates_the_total_the_preview_reads() -> None:
    """The total reaches 2022 at every debtor and its health part 2024. The preview reads the
    total alone, so coverage names 2022 with three points and the headline 2022 with three
    series, where counting the part would say 2022 to 2024 and six."""
    index = build_coverage_index(
        [(_TOTAL, "f1", ABSENT_ENTITY, r, list(range(2015, 2023))) for r in _DEBTORS]
        + [(_HEALTH, "f1", ABSENT_ENTITY, r, list(range(2015, 2025))) for r in _DEBTORS],
        role_names=_FAMILY_ROLES,
    )
    plan = _role_plan(
        _elected([_TOTAL, _HEALTH]),
        member_facts=_FAMILY_FACTS,
        names={"destination": "debtor"},
        subject_dcids=_DEBTORS,
        origin_dcids=[],
        bound_roles={"recipient": frozenset(_DEBTORS)},
        years=read_plan.role_years(_LATEST, index.horizon),
        coverage_index=index,
    )

    assert plan.members == (_TOTAL,)
    coverage = answer_coverage(plan, CoverageForm())
    headline = answer_headline(plan, index, date_request=_LATEST)
    assert isinstance(coverage, CoverageExact)
    assert (coverage.window, coverage.observation_count) == (
        TimeWindow(start_year=2022, end_year=2022),
        3,
    )
    assert headline is not None
    assert (headline.year, headline.series) == (2022, 3)


_ELECTION_DEBTORS = [f"country/E{i:02d}" for i in range(11)]


def _election_rows(sv: str, series_by_year: dict[int, int]) -> list:
    """One series per debtor, the first `series_by_year[year]` debtors holding `year`."""
    return [
        (sv, "f1", ABSENT_ENTITY, debtor, [y for y, held in series_by_year.items() if i < held])
        for i, debtor in enumerate(_ELECTION_DEBTORS[: max(series_by_year.values())])
    ]


def _election_plan(index, sv_dcids: list[str], date_request: DateRequest) -> ReadPlan:
    return _role_plan(
        sv_dcids,
        member_facts={sv: _Constraints({"sector": sv}) for sv in sv_dcids},
        names={"destination": "debtor"},
        subject_dcids=_ELECTION_DEBTORS,
        origin_dcids=[],
        bound_roles={"recipient": frozenset(_ELECTION_DEBTORS)},
        years=read_plan.role_years(date_request, index.horizon),
        coverage_index=index,
    )


def _bulk_reads(plan: BulkPlan) -> list[tuple[list[str], str | list[str]]]:
    """The `(variables, time_period)` of each `role_series` call a bulk preview of `plan`
    makes, in order, against an SDMX that agrees with coverage (`plan_observations`)."""
    sdmx = FakeSdmxClient(observations=plan_observations(plan))
    _bulk_role_series(plan, list(plan.members), sdmx=sdmx, graph=FakeGraph(), facet_filter=None)
    return [(call["variables"], call["time_period"]) for call in sdmx.series_calls]


def _bulk_pairs(plan: BulkPlan) -> dict[str, int]:
    """The pairs a bulk preview of `plan` dispatches for each member."""
    fetch = _plan_role_bulk_fetch(plan)
    return {
        sv: sum(1 for member, _role_tuple in fetch.to_fetch if member == sv)
        for sv in fetch.facet_svs
    }


def _latest_read(plan: PairPlan) -> dict[str, tuple[int | None, int]]:
    """What a per-pair `latest` preview of `plan` reads for each member: the year it names
    beside `LATEST` (`_Dispatch.elected_year`), None for `LATEST` alone, and the series it
    dispatches."""
    read: dict[str, tuple[int | None, int]] = {}
    for dispatch in _dispatches(
        _dispatched_series(plan, plan.members),
        {"debtor": _ELECTION_DEBTORS},
        years=plan.years,
        elected=plan.elected_years or {},
    ):
        year = dispatch.elected_year
        read[dispatch.stat_var] = (year, read.get(dispatch.stat_var, (year, 0))[1] + 1)
    return read


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
@pytest.mark.parametrize(
    ("date_request", "year", "series"),
    [
        pytest.param(_LATEST, 2024, 9, id="latest"),
        pytest.param(
            DateRequest(window=TimeWindow(start_year=2025, end_year=2025), latest=False),
            2025,
            5,
            id="named_year",
        ),
    ],
)
def test_a_latest_headline_and_download_count_the_year_the_held_series_elect(
    monkeypatch, search_settings, bulk: bool, date_request: DateRequest, year: int, series: int
) -> None:
    """Five debtors hold 2025, nine 2024 and eleven 2023. A `latest` answer heads 2024 with
    the nine holding it, its preview reads 2024, in bulk for those nine and per pair beside
    the `LATEST` row of all eleven, and its file and an option's served test count them, on
    both routes, from one election. The series at the newest
    period, or the nine alone, would elect 2025, and the headline keeps the raw counts of
    all three years. A request naming 2025 heads it with five and elects nothing."""
    if bulk:
        search_settings(obs_preview_max_pairs=1)
    index = build_coverage_index(
        _election_rows(_TOTAL, {2025: 5, 2024: 9, 2023: 11}),
        role_names={_TOTAL: RoleNames(donor=None, recipient="debtor")},
    )
    elections: list[object] = []
    elect = index.elected_years
    monkeypatch.setattr(
        index, "elected_years", lambda **rows: elections.append(rows) or elect(**rows)
    )
    plan = _election_plan(index, [_TOTAL], date_request)
    assert isinstance(plan, BulkPlan if bulk else PairPlan)

    headline = answer_headline(plan, index, date_request=date_request)
    download = download_facts(
        plan,
        index,
        requested_window="latest" if date_request.latest else None,
        conversion=None,
        provenance=None,
    )

    assert headline is not None
    assert (headline.year, headline.series, headline.role_counts) == (
        year,
        series,
        {"debtor": series},
    )
    assert served_dates(plan) == {_TOTAL: (str(year), str(year))}
    assert isinstance(answer_coverage(plan, CoverageForm()), CoverageExact)
    assert len(elections) == (1 if date_request.latest else 0)
    if date_request.latest:
        if isinstance(plan, BulkPlan):
            assert _bulk_pairs(plan) == {_TOTAL: 9}
            # Five pairs end after 2024, so the year is read once more.
            assert _bulk_reads(plan) == [([_TOTAL], "LATEST"), ([_TOTAL], ["2024"])]
        else:
            assert _latest_read(plan) == {_TOTAL: (2024, 11)}
        assert [(c.year, c.series) for c in headline.recent_years] == [
            (2025, 5),
            (2024, 9),
            (2023, 11),
        ]
        option = RoleServedFilter(
            probe_entities={"debtor": _ELECTION_DEBTORS},
            open_axes=frozenset(),
            years="latest",
            coverage=index,
        ).served({_TOTAL: _Constraints()})
        epoch = index.horizon.epoch_year
        assert {rows.years for rows in option[_TOTAL].values()} == {1 << (2024 - epoch)}
        assert len(option[_TOTAL]) == series
    assert (download.rows, download.series, download.window) == (
        series,
        series,
        TimeWindow(start_year=year, end_year=year),
    )


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
def test_a_two_member_latest_headline_counts_the_member_serving_its_year(
    search_settings, bulk: bool
) -> None:
    """The total holds 2025 at four of five debtors and elects it. Its health part holds
    2025 at one of six and elects 2024. The headline names 2025 with the total's four
    series, the preview reads the total in 2025, per pair at all five debtors, and the
    part's six series in 2024, and the file holds those ten rows, on both routes."""
    if bulk:
        search_settings(obs_preview_max_pairs=1)
    index = build_coverage_index(
        _election_rows(_TOTAL, {2025: 4, 2024: 5}) + _election_rows(_HEALTH, {2025: 1, 2024: 6}),
        role_names=_FAMILY_ROLES,
    )
    plan = _election_plan(index, [_TOTAL, _HEALTH], _LATEST)
    assert isinstance(plan, BulkPlan if bulk else PairPlan)
    assert plan.elected_years == {_TOTAL: 2025, _HEALTH: 2024}
    # A plan narrowed by a facet filter elects for itself, here over the same rows.
    assert plan.narrowed(lambda _sv, _candidate: True).elected_years == plan.elected_years
    if isinstance(plan, BulkPlan):
        assert _bulk_pairs(plan) == {_TOTAL: 4, _HEALTH: 6}
        # Only the part ends after its year, and its read names no year of the total's.
        assert _bulk_reads(plan) == [([_TOTAL, _HEALTH], "LATEST"), ([_HEALTH], ["2024"])]
    else:
        assert _latest_read(plan) == {_TOTAL: (2025, 5), _HEALTH: (2024, 6)}
    assert served_dates(plan) == {_TOTAL: ("2025", "2025"), _HEALTH: ("2024", "2024")}
    # A conversion checks each member's rows in the year it serves them.
    assert {
        sv: {block.year for rows in keys.values() for block in rows()}
        for sv, keys in _served_keys(plan).items()
    } == {_TOTAL: {2025}, _HEALTH: {2024}}

    headline = answer_headline(plan, index, date_request=_LATEST)
    download = download_facts(
        plan, index, requested_window="latest", conversion=None, provenance=None
    )

    assert headline is not None
    assert (headline.year, headline.series, headline.role_counts) == (2025, 4, {"debtor": 4})
    assert [(c.year, c.series) for c in headline.recent_years[:2]] == [(2025, 5), (2024, 11)]
    assert (download.rows, download.series, download.window) == (
        10,
        10,
        TimeWindow(start_year=2024, end_year=2025),
    )


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
@pytest.mark.parametrize(
    ("period", "build_year"),
    [pytest.param("P1M", None, id="sub_annual"), pytest.param("P1Y", 2022, id="projected")],
)
def test_a_member_the_index_cannot_elect_for_serves_its_newest_year(
    search_settings, bulk: bool, period: str, build_year: int | None
) -> None:
    """Five debtors hold 2025, nine 2024 and eleven 2023, counts that elect 2024. A facet
    stating months, or an index built in 2022, before every year held, elects no year. The
    preview reads `LATEST` and the answer serves, heads and counts the five series at
    2025, on both routes. A per-pair read fetches the latest row of all eleven to find
    that period, and a bulk read the five pairs coverage dates in it."""
    if bulk:
        search_settings(obs_preview_max_pairs=1)
    index = build_coverage_index(
        _election_rows(_TOTAL, {2025: 5, 2024: 9, 2023: 11}),
        role_names={_TOTAL: RoleNames(donor=None, recipient="debtor")},
        facet_metadata={
            "f1": FacetMetadata(provenance_id="dc/base/Ids", observation_period=period)
        },
        build_year=build_year,
    )
    plan = _election_plan(index, [_TOTAL], _LATEST)
    assert isinstance(plan, BulkPlan if bulk else PairPlan)

    headline = answer_headline(plan, index, date_request=_LATEST)
    download = download_facts(
        plan, index, requested_window="latest", conversion=None, provenance=None
    )

    assert plan.elected_years == {}
    if isinstance(plan, BulkPlan):
        assert _bulk_pairs(plan) == {_TOTAL: 5}
        assert _bulk_reads(plan) == [([_TOTAL], "LATEST")]
    else:
        assert _latest_read(plan) == {_TOTAL: (None, 11)}
    assert served_dates(plan) == {_TOTAL: ("2025", "2025")}
    assert headline is not None and (headline.year, headline.series) == (2025, 5)
    assert (download.rows, download.series, download.window) == (
        5,
        5,
        TimeWindow(start_year=2025, end_year=2025),
    )
    # A plan holding no election elects nothing too, and its file holds the same rows.
    assert (
        download_facts(
            replace(plan, elect=None),
            index,
            requested_window="latest",
            conversion=None,
            provenance=None,
        )
        == download
    )


def test_a_second_bulk_read_names_each_member_only_the_year_it_elected(search_settings) -> None:
    """Fifty members each hold every year from 1975 at 180 debtors, member `i` up to
    `1975 + i`, and one debtor holds the year after, so each elects a different year and
    ends after it. The answer keeps 9,000 rows. Each read after `LATEST` names its members
    only the year they elected, so the reads return 9,000 rows each way, where one read
    naming every member every elected year would return 229,549, past the SDMX row cap."""
    search_settings(obs_preview_max_pairs=1)
    debtors = [f"country/R{i:03d}" for i in range(180)]
    members = [f"TEST/PreviewRoute_Member{i:02d}" for i in range(50)]
    index = build_coverage_index(
        [
            (
                sv,
                "f1",
                ABSENT_ENTITY,
                debtor,
                [*range(1975, 1976 + i), *([1976 + i] if d == 0 else [])],
            )
            for i, sv in enumerate(members)
            for d, debtor in enumerate(debtors)
        ],
        role_names={sv: RoleNames(donor=None, recipient="debtor") for sv in members},
    )
    plan = _role_plan(
        members,
        member_facts={sv: _Constraints({"sector": sv}) for sv in members},
        names={"destination": "debtor"},
        subject_dcids=debtors,
        origin_dcids=[],
        bound_roles={"recipient": frozenset(debtors)},
        years=read_plan.role_years(_LATEST, index.horizon),
        coverage_index=index,
    )
    assert isinstance(plan, BulkPlan) and plan.elected_years is not None
    sdmx = CountingSdmxClient(observations=plan_observations(plan))

    observations, _warning = _bulk_role_series(
        plan, members, sdmx=sdmx, graph=FakeGraph(), facet_filter=None
    )

    newest, *named = sdmx.series_calls
    assert (newest["time_period"], newest["rows"]) == ("LATEST", 9_000)
    assert sorted(sv for call in named for sv in call["variables"]) == sorted(members)
    for call in named:
        assert all(call["time_period"] == [str(plan.elected_years[sv])] for sv in call["variables"])
        assert call["rows"] <= settings().sdmx_max_rows
    assert sum(call["rows"] for call in named) == plan.observation_count == 9_000
    assert observations.omitted is None


def _two_year_plan(search_settings) -> BulkPlan:
    """A bulk plan whose total elects 2024 and whose health part elects 2023, each with
    debtors ending after its year, so a `latest` read asks `LATEST` and then each year."""
    search_settings(obs_preview_max_pairs=1)
    index = build_coverage_index(
        _election_rows(_TOTAL, {2025: 5, 2024: 9, 2023: 11})
        + _election_rows(_HEALTH, {2024: 2, 2023: 9, 2022: 11}),
        role_names=_FAMILY_ROLES,
    )
    plan = _election_plan(index, [_TOTAL, _HEALTH], _LATEST)
    assert isinstance(plan, BulkPlan)
    assert plan.elected_years == {_TOTAL: 2024, _HEALTH: 2023}
    return plan


class _MeetingSdmx(FakeSdmxClient):
    """Holds each year-named `role_series` call until `together` of them are in flight,
    and breaks when they do not meet."""

    def __init__(self, *, together: int, **fixtures) -> None:
        super().__init__(**fixtures)
        self._meet = threading.Barrier(together, timeout=10)

    def role_series(self, **read):
        if read["time_period"] != "LATEST":
            self._meet.wait()
        return super().role_series(**read)


def test_the_bulk_reads_of_each_elected_year_are_in_flight_together(search_settings) -> None:
    """The reads of 2023 and 2024 that follow `LATEST` run at once, each under the
    preview's deadline and naming its member its own year, and both members are served."""
    plan = _two_year_plan(search_settings)
    sdmx = _MeetingSdmx(together=2, observations=plan_observations(plan))

    observations, _warning = _bulk_role_series(
        plan, [_TOTAL, _HEALTH], sdmx=sdmx, graph=FakeGraph(), facet_filter=None, deadline=123.0
    )

    assert sorted(
        (call["variables"], call["time_period"], call["deadline"]) for call in sdmx.series_calls[1:]
    ) == [([_HEALTH], ["2023"], 123.0), ([_TOTAL], ["2024"], 123.0)]
    assert sdmx.series_calls[0]["deadline"] == 123.0
    assert observations.omitted is None
    assert {(s.stat_var.ref, p.date) for s in observations.series for p in s.points} == {
        (_TOTAL, "2024"),
        (_HEALTH, "2023"),
    }


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        pytest.param(
            GraphInfraError("year read failed"), "observations_bulk_fetch_failed", id="failed"
        ),
        pytest.param(TimeoutError(), "observations_timed_out", id="timed_out"),
    ],
)
def test_a_failed_year_read_of_a_bulk_preview_serves_no_series(
    search_settings, failure: BaseException, reason: str
) -> None:
    """The bulk read asks `LATEST` and then 2023 and 2024. When one of those two reads
    fails or crosses the deadline the preview is omitted for that reason with no series,
    as when the first read fails: the member whose year was read is not served alone, and
    no `LATEST` row stands in for the year that could not be read."""
    plan = _two_year_plan(search_settings)
    sdmx = FakeSdmxClient(
        observations=plan_observations(plan), raise_series=lambda i: failure if i == 2 else None
    )

    observations, warning = _bulk_role_series(
        plan, [_TOTAL, _HEALTH], sdmx=sdmx, graph=FakeGraph(), facet_filter=None
    )

    assert len(sdmx.series_calls) == 3
    assert observations.omitted is not None and observations.omitted.reason == reason
    assert observations.series == []
    assert warning is not None and warning.code == reason


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
def test_a_dateless_role_plan_runs_no_election(monkeypatch, search_settings, bulk: bool) -> None:
    """A read of every date serves no elected year, so its plan never asks the index for
    one."""
    if bulk:
        search_settings(obs_preview_max_pairs=1)
    index = build_coverage_index(
        _election_rows(_TOTAL, {2025: 5, 2024: 9, 2023: 11}),
        role_names={_TOTAL: RoleNames(donor=None, recipient="debtor")},
    )
    elections: list[object] = []
    monkeypatch.setattr(index, "elected_years", lambda **rows: elections.append(rows) or {})
    plan = _role_plan(
        [_TOTAL],
        member_facts={_TOTAL: _Constraints({"sector": _TOTAL})},
        names={"destination": "debtor"},
        subject_dcids=_ELECTION_DEBTORS,
        origin_dcids=[],
        bound_roles={"recipient": frozenset(_ELECTION_DEBTORS)},
        years=read_plan.role_years(None, index.horizon),
        coverage_index=index,
    )
    assert isinstance(plan, BulkPlan if bulk else PairPlan)

    assert plan.elected_years is None
    assert elections == []


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
def test_a_plan_less_omitted_rows_elects_again_over_the_rows_it_keeps(
    monkeypatch, search_settings, bulk: bool
) -> None:
    """Five debtors hold 2025, nine 2024 and eleven 2023, which elects 2024. Without one
    debtor that holds 2025 and the four that stop at 2024 the counts are four, four and
    six, and the plan less their rows elects 2025 and serves its four series there. A
    per-pair preview reads every series the plan holds, eleven and six. Each plan elects
    once, however many counts read it."""
    if bulk:
        search_settings(obs_preview_max_pairs=1)
    index = build_coverage_index(
        _election_rows(_TOTAL, {2025: 5, 2024: 9, 2023: 11}),
        role_names={_TOTAL: RoleNames(donor=None, recipient="debtor")},
    )
    elections: list[object] = []
    elect = index.elected_years
    monkeypatch.setattr(
        index, "elected_years", lambda **rows: elections.append(rows) or elect(**rows)
    )
    plan = _election_plan(index, [_TOTAL], _LATEST)
    kept = plan.omitting(frozenset((_TOTAL, "debtor", d) for d in _ELECTION_DEBTORS[4:9]))

    for read, year, series, held in ((plan, 2024, 9, 11), (kept, 2025, 4, 6)):
        headline = answer_headline(read, index, date_request=_LATEST)
        download = download_facts(
            read, index, requested_window="latest", conversion=None, provenance=None
        )
        if isinstance(read, BulkPlan):
            assert _bulk_pairs(read) == {_TOTAL: series}
            # No pair the plan keeps ends after 2025, so that plan reads `LATEST` alone.
            assert (
                _bulk_reads(read)
                == [([_TOTAL], "LATEST"), ([_TOTAL], ["2024"])][: 2 if year == 2024 else 1]
            )
        else:
            assert _latest_read(read) == {_TOTAL: (year, held)}
        assert served_dates(read) == {_TOTAL: (str(year), str(year))}
        assert headline is not None and (headline.year, headline.series) == (year, series)
        assert (download.rows, download.series) == (series, series)
    assert len(elections) == 2


def test_discovered_family_elects_over_the_total_the_preview_reads(search_settings) -> None:
    """`f-a` holds the total at two debtors to 2020; `f-b` holds only the health part, at
    three debtors to 2024. The election keeps the total, and discovery elects among its
    facets, as the bulk preview reads it, so it plans `f-a` and names 2020; electing over
    the part would name 2024."""
    search_settings(obs_preview_max_pairs=1)
    index = build_coverage_index(
        [(_TOTAL, "f-a", ABSENT_ENTITY, r, list(range(2010, 2021))) for r in _DEBTORS[:2]]
        + [(_HEALTH, "f-b", ABSENT_ENTITY, r, list(range(2010, 2025))) for r in _DEBTORS],
        role_names=_FAMILY_ROLES,
    )
    snapshot = FakeCoverageService(index).snapshot()
    result = discover_role_scope(
        surviving_svs=_elected([_TOTAL, _HEALTH]),
        admitted_dcids=_elected([_TOTAL, _HEALTH]),
        member_facts=_FAMILY_FACTS,
        member_population_exact=True,
        open_flow_axis="destination",
        role_name="debtor",
        opposite_role_name=None,
        opposite_dcids=(),
        facet_filters={},
        date_request=_LATEST,
        coverage_snapshot=snapshot,
    )

    assert isinstance(result, Materialised)
    plan = result.read_plan
    assert isinstance(plan, BulkPlan)
    assert (plan.members, plan.facet_id) == ((_TOTAL,), "f-a")
    assert result.coverage.window == TimeWindow(start_year=2020, end_year=2020)
    assert result.headline is not None and result.headline.year == 2020
    assert _read_facets(plan, list(plan.members)) == {"f-a"}


_SPAN_ROWS = [
    (_FLOW, facet_id, ABSENT_ENTITY, recipient, list(years))
    for facet_id, years in (("fShort", range(2018, 2021)), ("fLong", range(2015, 2020)))
    for recipient in (_AAA, _BBB)
]


def test_a_bulk_election_ranks_the_span_and_count_a_mint_ranks(search_settings) -> None:
    """Two facets hold two pairs each inside 2015 to 2020. `fShort` runs 2018 to 2020 and
    `fLong` 2015 to 2019, so `fLong`'s span covers more of the window. The bulk plan ranks
    each facet's span and count over every year, the values the mint's coverage probe
    holds, and both elect `fLong`."""
    search_settings(obs_preview_max_pairs=1)
    index = build_coverage_index(
        _SPAN_ROWS,
        role_names={_FLOW: RoleNames(donor=None, recipient="debtor")},
        facet_metadata={
            "fShort": FacetMetadata(provenance_id="dc/base/Short"),
            "fLong": FacetMetadata(provenance_id="dc/base/Long"),
        },
    )
    bound: BoundRoles = {"recipient": frozenset({_AAA, _BBB})}
    years = read_plan.role_years(_WINDOW, index.horizon)
    plan = _role_plan(
        [_FLOW],
        member_facts={_FLOW: _Constraints()},
        names={"destination": "debtor"},
        subject_dcids=[_AAA, _BBB],
        origin_dcids=[],
        bound_roles=bound,
        years=years,
        coverage_index=index,
    )
    probe = coverage_probe(stat_var=_FLOW, bound=bound, coverage=index)
    window = plan_window(years.years)
    (minted,) = group_served({_FLOW: probe}, window=window)[_FLOW]

    assert isinstance(plan, BulkPlan)
    assert {
        facet_id: (c.lifetime[_FLOW].first_year, c.lifetime[_FLOW].last_year)
        for facet_id, c in plan.candidates.items()
    } == {"fShort": (2018, 2020), "fLong": (2015, 2019)}
    assert plan.facet_id == minted.facet_id == "fLong"


_FLOW_B = "TEST/PreviewRoute_FlowB"
_FLOW_C = "TEST/PreviewRoute_FlowC"
_GROUP_ROWS = [
    *(
        (_FLOW, "fWide", ABSENT_ENTITY, recipient, list(range(2015, 2021)))
        for recipient in (_AAA, _BBB, _CCC)
    ),
    (_FLOW, "fShared", ABSENT_ENTITY, _AAA, list(range(2015, 2021))),
    (_FLOW_B, "fShared", ABSENT_ENTITY, _AAA, list(range(2015, 2021))),
    (_FLOW_C, "fOwn", ABSENT_ENTITY, _BBB, list(range(2015, 2021))),
]


@pytest.mark.parametrize("max_pairs", [1, 1_000], ids=["bulk", "pairs"])
def test_a_plan_of_several_members_serves_the_group_a_mint_elects(
    search_settings, max_pairs: int
) -> None:
    """`fWide` holds three of `_FLOW`'s pairs, `fShared` one pair each of `_FLOW` and
    `_FLOW_B`, and `fOwn` one of `_FLOW_C`'s. The members serve one group, the facet held
    by the most of them, `fShared`, as the mint's election over the same probes does,
    though `fWide` holds the most pairs. `_FLOW_C` holds none of it and serves nothing, so
    its series and the debtors only it and `fWide` hold go unserved, and the plan names it as
    a member the group leaves out (`unserved_members`)."""
    search_settings(obs_preview_max_pairs=max_pairs)
    members = [_FLOW, _FLOW_B, _FLOW_C]
    index = build_coverage_index(
        _GROUP_ROWS,
        role_names={sv: RoleNames(donor=None, recipient="debtor") for sv in members},
        facet_metadata={
            facet_id: FacetMetadata(provenance_id=f"dc/base/{facet_id}")
            for facet_id in ("fWide", "fShared", "fOwn")
        },
    )
    bound: BoundRoles = {"recipient": frozenset({_AAA, _BBB, _CCC})}
    years = read_plan.role_years(_WINDOW, index.horizon)
    plan = _role_plan(
        members,
        member_facts=dict.fromkeys(members, _Constraints()),
        names={"destination": "debtor"},
        subject_dcids=[_AAA, _BBB, _CCC],
        origin_dcids=[],
        bound_roles=bound,
        years=years,
        coverage_index=index,
    )
    minted = group_served(
        {sv: coverage_probe(stat_var=sv, bound=bound, coverage=index) for sv in members},
        window=plan_window(years.years),
    )

    assert {sv: [f.facet_id for f in facets] for sv, facets in minted.items()} == {
        _FLOW: ["fShared"],
        _FLOW_B: ["fShared"],
        _FLOW_C: [],
    }
    if isinstance(plan, BulkPlan):
        assert max_pairs == 1
        assert (plan.facet_id, plan.served_members) == ("fShared", (_FLOW, _FLOW_B))
    else:
        assert max_pairs == 1_000
        assert plan.selections == dict.fromkeys((_FLOW, _FLOW_B), frozenset({"fShared"}))
        assert {s.key for s in plan.uncovered} == {
            (_FLOW, _BBB),
            (_FLOW, _CCC),
            (_FLOW_C, _BBB),
        }
    assert read_plan.unserved_entities(plan) == {("destination", _BBB), ("destination", _CCC)}
    # The mint names `_FLOW_C` in `coverage_below_scope`'s `unserved_members`, on both routes.
    assert read_plan.unserved_members(plan) == (_FLOW_C,)


def test_a_plan_names_the_entities_it_was_asked_for_and_serves_no_point_at(
    search_settings,
) -> None:
    """A standard plan asked for three places serves the one group: Ghana holds another
    source alone and Togo holds nothing, so neither is served. A scope read names no place.
    A bulk plan serves the bound recipients its elected facet holds a pair for in its
    years."""
    wide = _facet("fWide", range(2010, 2021), provenance="dc/base/Wide")
    other = _facet("fOther", range(2015, 2021), provenance="dc/base/Other")
    places = ["country/KEN", "country/UGA", "country/GHA", "country/TGO"]
    series = {
        ("sv", "country/KEN"): [wide],
        ("sv", "country/UGA"): [wide],
        ("sv", "country/GHA"): [other],
    }

    plan = plan_standard(series, members=["sv"], years=None, places=places)

    assert unserved_entities(plan) == {
        ("observationAbout", "country/GHA"),
        ("observationAbout", "country/TGO"),
    }
    assert unserved_entities(plan.narrowed(lambda _sv, _candidate: True)) == unserved_entities(plan)
    assert unserved_entities(plan.every_date()) == unserved_entities(plan)
    assert (
        unserved_entities(
            plan_standard(series, members=["sv"], years=None, scope=True, places=places)
        )
        == set()
    )
    assert unserved_entities(plan_standard({}, members=["sv"], years=None, places=places)) == set()
    assert unserved_entities(None) == set()

    _index, bulk = _bulk_named_answer(
        search_settings,
        {"fA": FacetMetadata(provenance_id="P"), "fB": FacetMetadata(provenance_id="Q")},
    )
    assert isinstance(bulk, BulkPlan)
    assert bulk.facet_id == "fB"
    assert unserved_entities(bulk) == {("destination", _CCC)}


_NOR = "country/NOR"


def test_a_bulk_plan_states_each_entity_unserved_on_its_own_role(search_settings) -> None:
    """mls-026 "How much does Norway give to the UN?": Norway is the donor and one of the
    recipients the group binds. It gives to AAA and BBB and never to itself, so it is
    unserved as a recipient and served as the donor."""
    search_settings(obs_preview_max_pairs=1)
    index = build_coverage_index(
        [(_FLOW, "fA", _NOR, recipient, list(range(2015, 2021))) for recipient in (_AAA, _BBB)],
        role_names={_FLOW: RoleNames(donor="lender", recipient="debtor")},
    )
    plan = _role_plan(
        [_FLOW],
        member_facts={_FLOW: _Constraints()},
        names={"destination": "debtor", "origin": "lender"},
        subject_dcids=[_AAA, _BBB, _NOR],
        origin_dcids=[_NOR],
        bound_roles={"recipient": frozenset({_AAA, _BBB, _NOR}), "donor": frozenset({_NOR})},
        years=read_plan.role_years(_WINDOW, index.horizon),
        coverage_index=index,
    )

    assert isinstance(plan, BulkPlan)
    assert unserved_entities(plan) == {("destination", _NOR)}


# ---------------------------------------------------------------------------
# The plan's years, groups and narrowing.
# ---------------------------------------------------------------------------


def _facet(
    facet_id: str,
    years: range,
    *,
    provenance: str = "dc/base/P",
    method: str | None = None,
    unit: str | None = None,
) -> Facet:
    dates = tuple(str(y) for y in years)
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        measurement_method=method,
        unit=unit,
        obs_count=len(dates),
        earliest_date=dates[0],
        latest_date=dates[-1],
        dates=dates,
        dates_complete=True,
    )


def _dateless(facets: list[Facet]) -> PairPlan:
    """The plan a one-place answer holding `facets` builds with no window."""
    return plan_standard({("sv", "country/KEN"): facets}, members=["sv"], years=None)


def test_a_window_gap_spans_the_group_an_every_date_read_serves() -> None:
    """Facets sharing one provenance, method and unit serve together, so the gap's span runs
    across all of them, reading a facet's dates when it states no bound and skipping a facet
    with no data."""
    years = TimeWindow(start_year=2021, end_year=2026)
    early, late = (
        replace(_facet(fid, span), earliest_date=None, latest_date=None)
        for fid, span in (("fB", range(2000, 2003)), ("fD", range(2028, 2031)))
    )
    empty = replace(_facet("fE", range(1990, 1991)), obs_count=0)
    gap = window_gap(_dateless([_facet("fA", range(2010, 2021)), early, late, empty]), years)
    assert gap == WindowGap(years=years, first="2000", last="2030")
    assert window_gap(_dateless([empty]), years) == WindowGap(years=years, first=None, last=None)
    assert window_gap(None, years) == WindowGap(years=years, first=None, last=None)


def test_a_window_gap_leaves_out_a_facet_the_election_passes_over() -> None:
    """`fA` from one source outranks `fB` from another, so the span is `fA`'s 2010 to 2020:
    the years a dateless answer serving `fA` names, not `fB`'s 2000 start."""
    years = TimeWindow(start_year=2023, end_year=2025)
    facets = [
        _facet("fA", range(2010, 2021)),
        _facet("fB", range(2000, 2005), provenance="dc/base/Other"),
    ]
    assert window_gap(_dateless(facets), years) == WindowGap(years=years, first="2010", last="2020")


@pytest.mark.parametrize("scope", [False, True], ids=["named", "scope"])
def test_a_window_gap_is_the_span_the_dateless_plan_of_the_same_series_serves(scope: bool) -> None:
    """Whatever the places hold, the gap names the first and last dates the dateless plan
    of the same series serves, so the note states the years a dateless answer states."""
    years = TimeWindow(start_year=2028, end_year=2030)
    alpha = {"provenance": "dc/base/Alpha"}
    beta = {"provenance": "dc/base/Beta"}
    for kenya, ghana in itertools.product(
        [
            [_facet("fA", range(2010, 2021), **alpha), _facet("fB", range(2015, 2025), **beta)],
            [_facet("fB", range(2019, 2022), **beta)],
            [_facet("fA", range(2005, 2023), **alpha)],
        ],
        [
            [_facet("fA", range(2012, 2019), **alpha), _facet("fB", range(2016, 2025), **beta)],
            [_facet("fB", range(2000, 2024), **beta), _facet("fA", range(2018, 2021), **alpha)],
            [],
        ],
    ):
        series = {("sv", "country/KEN"): kenya, ("sv", "country/GHA"): ghana}
        windowed = plan_standard(series, members=["sv"], years=years, scope=scope)
        dateless = plan_standard(series, members=["sv"], years=None, scope=scope)
        gap = read_plan.plan_window_gap(
            windowed, DateRequest(window=TimeWindow(start_year=2028, end_year=2030), latest=False)
        )
        first, last = read_plan.served_dates(dateless)["sv"]
        assert gap == WindowGap(years=years, first=first, last=last)


@pytest.mark.parametrize(
    ("asked", "years"),
    [
        (TimeWindow(start_year=None, end_year=2005), TimeWindow(start_year=None, end_year=2005)),
        (TimeWindow(start_year=2040, end_year=None), TimeWindow(start_year=2040, end_year=None)),
        (TimeWindow(start_year=2020, end_year=None), TimeWindow(start_year=2020, end_year=2026)),
        (TimeWindow(start_year=1950, end_year=2090), TimeWindow(start_year=1950, end_year=2090)),
    ],
    ids=["open_start", "future_start", "open_end", "past_the_horizon"],
)
def test_the_years_asked_keep_no_coverage_floor_or_cap(
    asked: TimeWindow, years: TimeWindow
) -> None:
    """The years a window answer names are the standard route's read of the request: an
    open start stays open, an open end runs to the pinned 2026 unless the request starts
    after it, and no coverage horizon bounds them."""
    assert asked_years(DateRequest(window=asked, latest=False)) == years
    assert asked_years(DateRequest(window=None, latest=True)) is None
    assert asked_years(None) is None


def test_an_open_end_stops_at_the_current_year_and_an_explicit_end_is_kept() -> None:
    """The pinned clock reads 2026: "since 2020" reads 2020 to 2026 on every route, a
    coverage horizon ending earlier stops it there, and "2028 to 2030" keeps its end."""
    since = DateRequest(window=TimeWindow(start_year=2020, end_year=None), latest=False)
    ahead = DateRequest(window=TimeWindow(start_year=2028, end_year=2030), latest=False)

    assert plan_years(since) == TimeWindow(start_year=2020, end_year=2026)
    assert plan_years(since, last_year=2024) == TimeWindow(start_year=2020, end_year=2024)
    assert plan_years(ahead, last_year=2087) == TimeWindow(start_year=2028, end_year=2030)
    assert plan_years(DateRequest(window=None, latest=True)) == "latest"
    assert plan_years(None) is None


def test_an_open_end_starting_in_the_future_reads_to_the_horizon() -> None:
    """ "Since 2028" under the 2026 clock asks about the future: it reads to the coverage
    horizon's last year, or every later year without one. A range with no year left once
    floored and capped raises `NoYearsToRead`, never an inverted window."""
    future = DateRequest(window=TimeWindow(start_year=2028, end_year=None), latest=False)
    this_year = DateRequest(window=TimeWindow(start_year=2026, end_year=None), latest=False)

    assert plan_years(future, last_year=2032) == TimeWindow(start_year=2028, end_year=2032)
    assert plan_years(future) == TimeWindow(start_year=2028, end_year=None)
    assert plan_years(this_year, last_year=2032) == TimeWindow(start_year=2026, end_year=2026)
    for window, first, last in (
        (TimeWindow(start_year=2040, end_year=None), 1960, 2032),
        (TimeWindow(start_year=2040, end_year=2045), 1960, 2032),
        (TimeWindow(start_year=1950, end_year=1955), 1960, 2032),
    ):
        assert read_range(window, first_year=first, last_year=last) is None
        with pytest.raises(NoYearsToRead):
            plan_years(DateRequest(window=window, latest=False), first_year=first, last_year=last)


# A horizon whose last year is 2032 and whose first, 1969, sits above `scope_min_year`.
_HORIZON = YearHorizon(epoch_year=1969, mask_words=1)


def _role_years(window: TimeWindow | None, *, latest: bool = False) -> read_plan.RoleYears:
    request = None if window is None and not latest else DateRequest(window=window, latest=latest)
    return read_plan.role_years(request, _HORIZON)


def test_role_years_confirm_plan_and_list_one_range() -> None:
    """A role read confirms on, plans and lists the same years: floored at the horizon's
    first year, capped at its last, and an open end stopping at the 2026 clock unless the
    window starts after it."""
    cases = [
        (TimeWindow(start_year=2020, end_year=2020), (2020, 2020)),
        (TimeWindow(end_year=1972), (1969, 1972)),
        (TimeWindow(start_year=2024), (2024, 2026)),
        (TimeWindow(start_year=2025, end_year=2040), (2025, 2032)),
        (TimeWindow(start_year=2028), (2028, 2032)),
    ]
    for window, (first, last) in cases:
        years = _role_years(window)
        assert years.years == TimeWindow(start_year=first, end_year=last)
        assert years.bounds == (first, last)
        assert years.listed == tuple(str(y) for y in range(first, last + 1))


def test_role_years_refuse_a_range_longer_than_the_cap(search_settings) -> None:
    """A range whose year list passes `scope_max_window_years` raises `WindowTooLong`, which
    every role route refuses as `scope_too_broad`."""
    search_settings(scope_max_window_years=5)
    with pytest.raises(read_plan.WindowTooLong):
        _role_years(TimeWindow(start_year=2000, end_year=2020))
    assert _role_years(TimeWindow(start_year=2016, end_year=2020)).listed == tuple(
        str(y) for y in range(2016, 2021)
    )


def test_latest_and_dateless_role_years_confirm_on_every_year() -> None:
    for years, planned in ((_role_years(None, latest=True), "latest"), (_role_years(None), None)):
        assert years.years == planned
        assert years.bounds == (None, None)
        assert years.listed is None


def test_role_years_with_no_year_left_raise() -> None:
    for window in (
        TimeWindow(start_year=1950, end_year=1955),
        TimeWindow(start_year=2040),
        TimeWindow(start_year=2040, end_year=2045),
    ):
        with pytest.raises(NoYearsToRead):
            _role_years(window)


def test_each_series_serves_its_joint_dispatch_group() -> None:
    """Kenya's `f-a` and `f-b` share a provenance, method and unit, so its series holds both.
    Norway's `f-c` comes from another source and is passed over. Under `latest` the preview
    dispatches Kenya's series alone, since Norway's ends in 2022, before the variable's 2024.
    A place-keyed variable lies outside the coverage index, elects no year and serves its
    newest period."""
    plan = plan_standard(
        {
            ("sv", "country/KEN"): [
                _facet("f-a", range(1990, 2020)),
                _facet("f-b", range(2015, 2025)),
            ],
            ("sv", "country/NOR"): [
                _facet("f-a", range(1990, 2023)),
                _facet("f-c", range(2000, 2005), provenance="dc/base/Q"),
            ],
        },
        members=["sv"],
        years="latest",
    )

    assert {s.key: s.group_ids for s in plan.held} == {
        ("sv", "country/KEN"): frozenset({"f-a", "f-b"}),
        ("sv", "country/NOR"): frozenset({"f-a"}),
    }
    assert plan.elected_years is None
    assert [s.key for s in plan.served] == [("sv", "country/KEN")]
    assert all(s.role_tuple == (("observationAbout", s.key[1]),) for s in plan.held)


def test_a_series_whose_group_misses_the_window_is_not_served() -> None:
    """Inside 2020 to 2022 Norway's group ends in 2005, so the preview reads Kenya alone."""
    plan = plan_standard(
        {
            ("sv", "country/KEN"): [_facet("f-a", range(2010, 2025))],
            ("sv", "country/NOR"): [_facet("f-b", range(2000, 2006))],
        },
        members=["sv"],
        years=TimeWindow(start_year=2020, end_year=2022),
    )

    assert [s.key for s in plan.series] == [("sv", "country/KEN"), ("sv", "country/NOR")]
    assert [s.key for s in plan.served] == [("sv", "country/KEN")]


def test_a_scope_read_serves_one_facet_at_every_place() -> None:
    """`f-wide` covers three places and `f-deep` one with more observations; a scope read
    leads with places covered, so every place serves `f-wide`."""
    places = ["country/A", "country/B", "country/C"]
    series = {("sv", p): [_facet("f-wide", range(2015, 2021))] for p in places}
    series[("sv", "country/A")].append(_facet("f-deep", range(1950, 2021), provenance="dc/base/Q"))

    plan = plan_standard(series, members=["sv"], years="latest", scope=True)

    assert plan.scope_facet_id == "f-wide"
    assert {s.group_ids for s in plan.served} == {frozenset({"f-wide"})}


_UNIT_SERIES = {
    ("sv", "country/KEN"): [
        _facet("f-usd", range(2010, 2021), unit="USD"),
        _facet("f-eur", range(2010, 2025), provenance="dc/base/Q", unit="EUR"),
    ],
    ("sv", "country/NOR"): [_facet("f-eur", range(2010, 2023), provenance="dc/base/Q", unit="EUR")],
}


@pytest.mark.parametrize("scope", [False, True])
def test_narrowing_a_pair_plan_equals_planning_the_admitted_candidates(scope: bool) -> None:
    """A USD filter drops `f-eur`: the narrowed plan elects Kenya's `f-usd` and serves no
    Norway series, as a plan built from the USD facets alone does."""

    def usd(_sv: str, candidate) -> bool:
        return candidate.unit == "USD"

    plan = plan_standard(_UNIT_SERIES, members=["sv"], years="latest", scope=scope)
    admitted = {key: [f for f in facets if usd(key[0], f)] for key, facets in _UNIT_SERIES.items()}

    narrowed = plan.narrowed(usd)

    assert narrowed == plan_standard(admitted, members=["sv"], years="latest", scope=scope)
    assert [(s.key, s.group_ids) for s in narrowed.served] == [
        (("sv", "country/KEN"), frozenset({"f-usd"}))
    ]


def test_narrowing_a_bulk_plan_equals_planning_the_admitted_candidates(search_settings) -> None:
    """Above the cap `fB` covers two debtors inside 2015 to 2020 and is elected; a EUR filter
    leaves `fA`, which a plan built from the EUR rows alone also elects."""
    metadata = {
        "fA": FacetMetadata(provenance_id="dc/base/A", unit="EUR"),
        "fB": FacetMetadata(provenance_id="dc/base/B", unit="USD"),
    }
    _index, plan = _bulk_named_answer(search_settings, metadata)
    eur_index = build_coverage_index(
        [row for row in _BULK_ROWS if row[1] == "fA"],
        role_names={_FLOW: RoleNames(donor=None, recipient="debtor")},
        facet_metadata=metadata,
    )
    eur_plan = _role_plan(
        [_FLOW],
        member_facts={_FLOW: _Constraints()},
        names={"destination": "debtor"},
        subject_dcids=[_AAA, _BBB, _CCC],
        origin_dcids=[],
        bound_roles={"recipient": frozenset({_AAA, _BBB, _CCC})},
        years=read_plan.role_years(_WINDOW, eur_index.horizon),
        coverage_index=eur_index,
    )

    narrowed = plan.narrowed(lambda _sv, c: c.unit == "EUR")

    assert isinstance(narrowed, BulkPlan) and isinstance(eur_plan, BulkPlan)
    assert narrowed == eur_plan
    assert narrowed.facet_id == "fA"
    assert narrowed.read_years == tuple(_YEARS)


def test_named_and_discovered_pairs_elect_over_the_same_in_window_counts(search_settings) -> None:
    """Kenya holds `f-long` 1970-2021 and `f-short` 2020-2022. Inside 2020 to 2022 a named
    Kenya and a discovered Kenya plan the same group, elected over the points each facet
    holds in those years (`materialise.facets.rank_pool`): `f-short` holds three, `f-long`
    two, though `f-long` holds 52 over its lifetime."""
    index = build_coverage_index(
        [
            ("V", "f-long", ABSENT_ENTITY, "country/KEN", list(range(1970, 2022))),
            ("V", "f-short", ABSENT_ENTITY, "country/KEN", [2020, 2021, 2022]),
        ],
        role_names={"V": RoleNames(donor=None, recipient="debtor")},
        facet_metadata={
            "f-long": FacetMetadata(provenance_id="dc/base/Long"),
            "f-short": FacetMetadata(provenance_id="dc/base/Short"),
        },
    )
    window = DateRequest(window=TimeWindow(start_year=2020, end_year=2022), latest=False)
    named = _role_plan(
        ["V"],
        member_facts={"V": _Constraints()},
        names={"destination": "debtor"},
        subject_dcids=["country/KEN"],
        origin_dcids=[],
        bound_roles={"recipient": frozenset({"country/KEN"})},
        years=read_plan.role_years(window, index.horizon),
        coverage_index=index,
    )
    discovered = discover_role_scope(
        surviving_svs=["V"],
        admitted_dcids=["V"],
        member_facts={},
        member_population_exact=True,
        open_flow_axis="destination",
        role_name="debtor",
        opposite_role_name=None,
        opposite_dcids=(),
        facet_filters={},
        date_request=window,
        coverage_snapshot=FakeCoverageService(index).snapshot(),
    )

    assert isinstance(discovered, Materialised)
    assert isinstance(named, PairPlan) and isinstance(discovered.read_plan, PairPlan)
    assert named.served == discovered.read_plan.served
    assert [(s.role_tuple, s.group_ids) for s in named.served] == [
        ((("debtor", "country/KEN"),), frozenset({"f-short"}))
    ]


# A scope read whose places hold one facet's dates in part: a dated graph read can return
# fewer dates than its observation count (`dates_complete=False`).
_SCOPE_WINDOW = TimeWindow(start_year=2021, end_year=2026)


def test_a_scope_facet_with_a_short_read_spanning_the_window_stays_elected() -> None:
    """`fX` (p1) covers both places: complete dates 2010 to 2012 at A, and at B a short
    read, one date of twenty, spanning 2004 to 2023. `fY` (p2) holds 2022 and 2023 at B.
    B's `fX` cannot be ruled out of 2021 to 2026, so `fX` keeps its place in the election
    and leads it on places covered, and the answer cites p1, stating the most B's points
    can reach, 2021 to 2023."""
    fx_a = _facet("fX", range(2010, 2013), provenance="p1")
    fx_b = replace(
        _facet("fX", range(2004, 2024), provenance="p1"),
        obs_count=20,
        dates=("2004",),
        dates_complete=False,
    )
    fy_b = _facet("fY", range(2022, 2024), provenance="p2")
    plan = plan_standard(
        {("sv", "A"): [fx_a], ("sv", "B"): [fx_b, fy_b]},
        members=["sv"],
        years=_SCOPE_WINDOW,
        scope=True,
    )
    coverage = answer_coverage(plan, CoverageForm())

    assert plan.scope_facet_id == "fX"
    assert [s.role_tuple[0][1] for s in plan.served] == ["B"]
    assert {d.label: d.count for d in coverage.dimensions or []}["sources"] == 1
    assert coverage.window == TimeWindow(start_year=2021, end_year=2023)


def test_a_scope_place_whose_short_read_ends_before_the_window_is_not_served() -> None:
    """`fX` at A is a short read spanning 2004 to 2012; at B it holds 2022 and 2023. Under
    2021 to 2026, A's span proves it holds no point, so the plan serves B alone, and
    coverage counts one place, B's two points and states 2022 to 2023."""
    fx_a = replace(
        _facet("fX", range(2004, 2013), provenance="p1"),
        obs_count=20,
        dates=("2004",),
        dates_complete=False,
    )
    fx_b = _facet("fX", range(2022, 2024), provenance="p1")
    plan = plan_standard(
        {("sv", "A"): [fx_a], ("sv", "B"): [fx_b]},
        members=["sv"],
        years=_SCOPE_WINDOW,
        scope=True,
    )
    coverage = answer_coverage(plan, CoverageForm(facet_label="places", counts_places=True))

    assert [s.role_tuple[0][1] for s in plan.served] == ["B"]
    assert isinstance(coverage, CoverageExact)
    assert coverage.observation_count == 2
    assert {d.label: d.count for d in coverage.dimensions or []}["places"] == 1
    assert coverage.window == TimeWindow(start_year=2022, end_year=2023)


def test_a_filtered_latest_bulk_plan_serves_the_latest_year_its_pairs_keep(search_settings):
    """`latest` in bulk over flow A (to 2018) and flow B (to 2024) serves each flow its own
    latest year, A's 2018 and B's 2024, one series each. A filter keeping flow A alone
    keeps A's 2018, with one series."""
    search_settings(obs_preview_max_pairs=1)
    flow_a, flow_b = "TEST/FlowA", "TEST/FlowB"
    index = build_coverage_index(
        [
            (flow_a, "f", ABSENT_ENTITY, "country/AAA", list(range(2015, 2019))),
            (flow_b, "f", ABSENT_ENTITY, "country/BBB", list(range(2015, 2025))),
        ],
        role_names={sv: RoleNames(donor=None, recipient="debtor") for sv in (flow_a, flow_b)},
        facet_metadata={"f": FacetMetadata(provenance_id="dc/base/P", unit="USD")},
    )
    places = frozenset({"country/AAA", "country/BBB"})
    plan = _role_plan(
        [flow_a, flow_b],
        member_facts={flow_a: _Constraints({"x": "a"}), flow_b: _Constraints({"x": "b"})},
        names={"destination": "debtor"},
        subject_dcids=sorted(places),
        origin_dcids=[],
        bound_roles={"recipient": places},
        years=read_plan.role_years(DateRequest(window=None, latest=True), index.horizon),
        coverage_index=index,
    )
    assert isinstance(plan, BulkPlan)
    assert plan.member_years(flow_a) == (2018, 2018)
    assert plan.member_years(flow_b) == (2024, 2024)
    assert plan.population == 2

    narrowed = plan.narrowed(lambda sv, _candidate: sv == flow_a)
    assert narrowed.member_years(flow_a) == (2018, 2018)
    assert narrowed.population == 1


def test_a_bulk_plan_narrowed_per_statvar_serves_one_set_of_members(
    monkeypatch, search_settings
) -> None:
    """Health and Education both carry `fU` (USD) at three debtors; Education also carries
    `fE` (EUR) at one. Electing USD for Health and EUR for Education keeps `fU` elected for
    Health alone, so coverage, the population, the bulk read and the headline all serve
    Health's three series and count Education's none."""
    monkeypatch.setattr(read_plan, "_absent_readings", lambda: {})
    search_settings(obs_preview_max_pairs=1)
    health, education = "TEST/Flow_Health", "TEST/Flow_Education"
    debtors = ["country/AAA", "country/BBB", "country/CCC"]
    rows = [
        (sv, "fU", ABSENT_ENTITY, debtor, list(range(2015, 2021)))
        for sv in (health, education)
        for debtor in debtors
    ]
    rows.append((education, "fE", ABSENT_ENTITY, debtors[0], list(range(2015, 2021))))
    index = build_coverage_index(
        rows,
        role_names={sv: RoleNames(donor=None, recipient="debtor") for sv in (health, education)},
        facet_metadata={
            "fU": FacetMetadata(provenance_id="dc/base/U", unit="USD"),
            "fE": FacetMetadata(provenance_id="dc/base/E", unit="EUR"),
        },
    )
    latest = DateRequest(window=None, latest=True)
    plan = _role_plan(
        [health, education],
        member_facts={
            health: _Constraints({"sector": "Health"}),
            education: _Constraints({"sector": "Education"}),
        },
        names={"destination": "debtor"},
        subject_dcids=debtors,
        origin_dcids=[],
        bound_roles={"recipient": frozenset(debtors)},
        years=read_plan.role_years(latest, index.horizon),
        coverage_index=index,
    )
    assert isinstance(plan, BulkPlan)
    form = CoverageForm(variables=True)
    answer = Materialised(
        sv_dcids=[health, education],
        has_data=True,
        coverage=answer_coverage(plan, form),
        coverage_form=form,
        read_plan=plan,
        headline=answer_headline(plan, index, date_request=latest),
    )

    filtered = filter_answer(
        answer,
        FacetAdmission(elected_units={health: frozenset({"USD"}), education: frozenset({"EUR"})}),
        date_request=latest,
        coverage_index=index,
    )

    narrowed = filtered.plan
    assert isinstance(narrowed, BulkPlan)
    assert narrowed.facet_id == "fU"
    assert narrowed.served_members == (health,)
    assert narrowed.population == 3
    assert narrowed.elected is not None
    fetch = _plan_role_bulk_fetch(narrowed)
    assert {sv for sv, _ in fetch.to_fetch} == {health}
    assert isinstance(filtered.coverage, CoverageExact)
    assert filtered.coverage.observation_count == 3
    assert filtered.headline is not None
    assert (filtered.headline.series, filtered.headline.observations) == (3, 3)


def test_a_plan_lists_its_served_facets_by_points_served_whatever_its_series_order() -> None:
    """A plan whose four series each serve one source: `fp` (Projections, ten points), `fw`
    (WDI, eleven) and two of ten. The plan lists WDI first whatever its series' order, and
    sources tied on points by dcid. The series' groups are set, since an election serves
    one source per Spec."""
    by_series = {
        "V1": _facet("fp", range(2021, 2031), provenance="dc/base/Proj"),
        "V2": _facet("fw", range(2010, 2021), provenance="dc/base/Wdi"),
        "V3": _facet("fa", range(2010, 2020), provenance="dc/base/B"),
        "V4": _facet("fb", range(2010, 2020), provenance="dc/base/A"),
    }
    series = [
        PlanSeries(
            stat_var=sv,
            role_tuple=(("observationAbout", "country/ZZZ"),),
            candidates=(facet,),
            group=(facet,),
        )
        for sv, facet in by_series.items()
    ]
    orders = [series, list(reversed(series))]

    listed = [
        [
            f.provenance_id
            for f in read_plan.facets_served(
                PairPlan(members=tuple(by_series), years=None, series=tuple(order))
            )
        ]
        for order in orders
    ]

    assert listed[0] == listed[1] == ["dc/base/Wdi", "dc/base/A", "dc/base/B", "dc/base/Proj"]


def test_a_series_elects_the_facet_holding_the_most_points_in_the_years_read() -> None:
    """`f-long` holds 52 points from 1970 to 2021 and `f-short` three from 2020 to 2022.
    Inside 2020 to 2022 `f-short` holds more, so it serves; over every year `f-long` does."""
    candidates = [
        _facet("f-long", range(1970, 2022), provenance="dc/base/Long"),
        _facet("f-short", range(2020, 2023), provenance="dc/base/Short"),
    ]
    window = TimeWindow(start_year=2020, end_year=2022)

    def group(years) -> frozenset[str]:
        (series,) = plan_standard(
            {("V", "country/KEN"): candidates}, members=["V"], years=years
        ).series
        return series.group_ids

    assert group(window) == {"f-short"}
    assert group(None) == {"f-long"}
    assert [read_plan.points_in_years(f, window) for f in candidates] == [2, 3]


def test_a_plan_holding_only_unservable_series_still_answers() -> None:
    """A series holding points in the years but no facet id the preview can dispatch is
    unservable: the plan serves nothing yet keeps its answer, whose preview says why."""
    undispatchable = replace(_facet("f", range(2010, 2021)), facet_id=None)
    plan = plan_standard({("V", "country/KEN"): [undispatchable]}, members=["V"], years=None)

    assert plan.served == ()
    assert len(plan.unservable) == 1
    assert not read_plan.serves_nothing(plan)


def test_a_pair_series_role_tuple_is_the_minted_series_key() -> None:
    """A two-role pair keys its series as a mint keys it (`series_key`, roles sorted by
    name), whatever order the plan's role names come in. Destination comes first here, which
    unsorted would put `financingRecipient` before `financingProvider`."""
    index = build_coverage_index(
        [("V", "f-a", "country/DEU", "country/KEN", [2020, 2021])],
        role_names={"V": RoleNames(donor="financingProvider", recipient="financingRecipient")},
        facet_metadata={"f-a": FacetMetadata(provenance_id="dc/base/A")},
    )

    (series,) = pair_series(
        index,
        variables=["V"],
        bound={"recipient": frozenset({"country/KEN"}), "donor": frozenset({"country/DEU"})},
        role_names={"recipient": "financingRecipient", "donor": "financingProvider"},
        years=read_plan.role_years(None, index.horizon),
    )

    assert (series.stat_var, series.role_tuple) == series_key(
        "V", ["financingRecipient", "financingProvider"], ["country/KEN", "country/DEU"]
    )
    assert series.rows == ("country/KEN", "country/DEU")


@pytest.mark.parametrize(
    ("places", "usd_count", "group", "reads"),
    [
        pytest.param(("country/FRA",), 46, {"fUSD"}, [("sv",)], id="one-place-broader-serves"),
        # The dollars hold more observations, so nothing ties and no breadth is read.
        pytest.param(("country/FRA",), 47, {"fUSD"}, [], id="one-place-no-tie-reads-none"),
        pytest.param(("country/FRA", "country/DEU"), 46, {"fEUR"}, [], id="two-places-read-none"),
    ],
)
def test_a_plan_at_one_place_breaks_a_tie_on_the_variable_s_whole_breadth(
    places, usd_count, group, reads
) -> None:
    """At one place the dollar and euro facets tie on years, count and latest date, so the
    facet broader across the whole variable serves, as a mint of the answer's Spec elects.
    Without a tie the breadth is never read. A plan naming several places reads no breadth
    and elects as before; here euros, which hold both places and lead on the unit's name."""
    eur = Facet(
        facet_id="fEUR",
        provenance_id="P",
        unit="CurrencyEUR",
        obs_count=46,
        earliest_date="1980",
        latest_date="2025",
    )
    usd = replace(eur, facet_id="fUSD", unit="CurrencyUSD", obs_count=usd_count)
    recorded: list[tuple[str, ...]] = []

    def reader(stat_vars, *, deadline):
        recorded.append(tuple(stat_vars))
        return {sv: {"fUSD": 197, "fEUR": 25} for sv in stat_vars}

    with bound_variable_breadth(reader, refuse_unread=False):
        plan = plan_standard(
            {("sv", place): (eur, usd) for place in places},
            members=["sv"],
            years=None,
            places=places,
        )

    assert {f.facet_id for s in plan.series for f in s.group} == group
    assert recorded == reads


@pytest.mark.parametrize(
    ("profiled", "usd_count", "group", "degraded"),
    [
        pytest.param({"sv": {"fUSD": 197, "fEUR": 25}}, 1, {"fUSD"}, [], id="profiled"),
        pytest.param({}, 1, {"fEUR"}, ["whole-variable breadth"], id="unread-tie-falls-back"),
        pytest.param({}, 2, {"fUSD"}, [], id="unread-without-a-tie-changes-nothing"),
    ],
)
def test_search_falls_back_on_an_unread_breadth_and_notes_a_degraded_read(
    profiled, usd_count, group, degraded
) -> None:
    """At one place the euro and dollar facets tie on every key before breadth. Search stores
    no election, so a failed breadth read leaves the tie to the unit's name and notes a
    degraded read, which marks the answer degraded and keeps it out of the cache. When the
    dollars hold more observations there is no tie, so the failed read changes nothing and
    the answer stays cacheable."""
    eur = Facet(
        facet_id="fEUR",
        provenance_id="P",
        unit="CurrencyEUR",
        obs_count=1,
        earliest_date="2020",
        latest_date="2020",
    )
    usd = replace(eur, facet_id="fUSD", unit="CurrencyUSD", obs_count=usd_count)

    with (
        recording_degraded_reads() as record,
        bound_variable_breadth(
            lambda stat_vars, *, deadline: profiled,
            refuse_unread=False,
            on_fallback=_note_breadth_fallback,
        ),
    ):
        plan = plan_standard(
            {("sv", "country/FRA"): (eur, usd)},
            members=["sv"],
            years=None,
            places=["country/FRA"],
        )

    assert {f.facet_id for s in plan.series for f in s.group} == group
    assert record.reads == degraded


def test_a_one_role_continuation_counts_the_observations_of_every_row_of_a_member():
    """A one-role read of a two-role variable sums the rows each debtor holds under every
    creditor. Kenya's reported rows end in 2022 and both creditors' estimates run to 2025,
    so each later year adds two observations: the count a year-by-year read of the same
    pairs gives."""
    from search.engine.retrieval.facet_election import merge_role_candidates
    from search.engine.retrieval.read_plan import BulkPlan
    from search.engine.retrieval.role_scope import role_index_for_named_entities

    sv, kenya = "TESTFIXTURE/OneRoleContinuation/Flow", "country/KEN"
    index = build_coverage_index(
        [
            (sv, "rec", "country/CHN", kenya, [2020, 2021, 2022]),
            (sv, "rec", "country/FRA", kenya, [2020, 2021]),
            (sv, "est", "country/CHN", kenya, [2023, 2024, 2025]),
            (sv, "est", "country/FRA", kenya, [2022, 2023, 2024, 2025]),
        ],
        role_names={sv: RoleNames(donor="creditor", recipient="debtor")},
    )
    first, last = 2020, 2025
    candidates = merge_role_candidates(
        role_index_for_named_entities(
            [sv],
            {"debtor": [kenya]},
            time_period=[str(y) for y in range(first, last + 1)],
            coverage_index=index,
        ),
        [sv],
    )
    plan = BulkPlan(
        members=(sv,),
        years=TimeWindow(start_year=first, end_year=last),
        facet_id="rec",
        continuing=("est",),
        candidates=candidates,
    )

    def counts(facet_id: str, start: int, end: int) -> dict:
        (candidate,) = candidates[facet_id].by_sv[sv]
        return dict(
            candidate.pairs.head(
                candidate.pair_count, first_year=start, last_year=end, with_data=True
            )
        )

    expected = counts("rec", first, last)
    for year in range(first, last + 1):
        later = counts("rec", year, last)
        for role_tuple, n_obs in counts("est", year, year).items():
            if role_tuple not in later:
                expected[role_tuple] = expected.get(role_tuple, 0) + n_obs

    assert expected == {(("debtor", kenya),): 11}
    assert plan.continued_pairs == {sv: expected}
    assert plan.observation_count == 11
    assert plan.max_pair_observations == 11
