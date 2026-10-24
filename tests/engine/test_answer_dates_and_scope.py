"""An answer dates its StatVars and lists its scope from the read plan it serves.

`stat_vars.items[].data_date_range`, the years a dateless sentence names and the window
note read `read_plan.served_dates` of the answer's plan. `scope.members`, coverage's breadth
dimension and the `role_scope_defaulted` warning read `coverage.scope_members` of it. Both
hold on the standard route and the four role routes, on a first answer and an unchanged
resubmit of it, with the graph requests each route made before. A role plan decides between
a per-pair and a bulk read before it builds a single series. A facet declaring no provenance
serves per pair as it does in bulk, and a future window inside the coverage horizon reads
coverage on every route.
"""

from __future__ import annotations

import contextvars
import dataclasses
import threading

import httpx
import pytest

import search.engine.region.phase_answer as phase_answer
import search.engine.region.promote_ground as promote_ground
import search.engine.region.related_read as related_read
import search.engine.retrieval.role_resolve as role_resolve
import search.engine.retrieval.role_scope as role_scope
import tests.engine.test_cited_sources as cited_sources
import tests.engine.test_parsed_pipeline as parsed
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from search.engine.clients.graph import EngineGraphClient, Facet
from search.engine.departures import spec_departure_facts
from search.engine.extract import DateRequest
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.engine.retrieval.coverage import CoverageForm, scope_members
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.read_plan import (
    BulkPlan,
    PairPlan,
    ReadPlan,
    role_years,
    served_dates,
)
from search.engine.retrieval.retrieve import Materialised
from search.models import (
    DateRange,
    IdentityBindingValue,
    IdentityWindowRange,
    ResubmitIdentityPayload,
    Spec,
    TimeWindow,
)
from search.render.plan import render_sentence
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    _SDMX_CSV_CONTENT_TYPE,
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    walk_options,
)
from tests.engine._request_log import _UNRECORDED_METHODS
from tests.engine.test_cited_sources import (
    _ALAND,
    _ALT,
    _BLAND,
    _CREDITOR,
    _FLOW,
    _IDS,
    _ROUTES,
    _TWO_SOURCES,
    _fertility,
    _resubmit,
    _resubmit_fertility,
    _role_answer,
    _spy_plans,
)
from tests.fixtures import FakeGraph
from tests.materialise.conftest import FakeGraphReader

_CLAND = "country/CCC"
_GRAPH_METHODS = frozenset(
    name
    for name in dir(EngineGraphClient)
    if not name.startswith("_") and name not in _UNRECORDED_METHODS
)


def _year_unavailable_shown(spec: Spec) -> list[str | None]:
    """What each `year_unavailable` departure the Spec raises says it shows instead."""
    return [
        fact.shown
        for fact in spec_departure_facts(spec, operations=())
        if fact.code == "year_unavailable"
    ]


_RELATED_READ: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "answer_dates_related_read", default=False
)


def _record_graph(monkeypatch, *modules, related: list[str] | None = None) -> list[str]:
    """Swap each module's `FakeGraph` for one that logs every graph request by method name,
    in call order. A request the fake serves through another counts once. With `related`,
    the related-measures reads (`related_read.read_related`), which run beside the answer's
    own, are logged there instead, so the answer's own order stays fixed."""
    log: list[str] = []
    depth = threading.local()

    class _RecordingGraph(FakeGraph):
        def __getattribute__(self, name: str):
            attr = super().__getattribute__(name)
            if name not in _GRAPH_METHODS or not callable(attr):
                return attr

            def _request(*args, **kwargs):
                if getattr(depth, "n", 0) == 0:
                    (related if related is not None and _RELATED_READ.get() else log).append(name)
                depth.n = getattr(depth, "n", 0) + 1
                try:
                    return attr(*args, **kwargs)
                finally:
                    depth.n -= 1

            return _request

    if related is not None:
        read_related = related_read.read_related

        def _marked(**kwargs):
            _RELATED_READ.set(True)
            return read_related(**kwargs)

        monkeypatch.setattr(related_read, "read_related", _marked)

    for module in modules:
        monkeypatch.setattr(module, "FakeGraph", _RecordingGraph)
    return log


def _record_sdmx(monkeypatch) -> list[FakeSdmxClient]:
    """Every SDMX fake the role helpers build, to read the requests each recorded."""
    clients: list[FakeSdmxClient] = []

    class _RecordingSdmx(FakeSdmxClient):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            clients.append(self)

    monkeypatch.setattr(cited_sources, "FakeSdmxClient", _RecordingSdmx)
    return clients


def _sdmx_requests(clients: list[FakeSdmxClient]) -> int:
    return sum(
        len(c.series_calls) + len(c.availability_calls) + len(c.subject_observations_calls)
        for c in clients
    )


# The graph requests each route makes for a first answer and an unchanged resubmit: the
# shape, labels, types and facet policies, and for an answer resolved through the event
# stream whether the graph declares facet defaults at all. Dating, scoping and the read decision read the plan and
# the coverage index alone, so they add none. The trailing label read of each names the values
# its facet options offer (`_maybe_build_facet_options`), after a facets read on the standard
# route; a role route reads its facets, and the named places' availability, from the coverage
# index.
_NAMED_FIRST = [
    "detect_svs",
    "resolve_by_arc",
    "node_names_many",
    "node_arcs_many",
    "node_arcs_batch",
    "node_labels_batch",
    "node_arcs_many",
    *["default_facet_policy"] * 3,
    "node_labels_batch",
    "node_types_batch",
    "node_labels_batch",
    "node_labels_batch",
]
_NAMED_RESUBMIT = [
    "node_arcs_batch",
    "node_labels_batch",
    *["default_facet_policy"] * 3,
    "node_labels_batch",
    "node_types_batch",
    "node_labels_batch",
    "node_labels_batch",
]
_DISCOVERED_FIRST = [
    "detect_svs",
    "node_arcs_batch",
    "node_labels_batch",
    *["default_facet_policy"] * 3,
    *["node_labels_batch"] * 3,
    "node_types_batch",
]
_DISCOVERED_RESUBMIT = [
    "node_arcs_batch",
    *["default_facet_policy"] * 3,
    "node_labels_batch",
    "node_labels_batch",
    "node_types_batch",
    "node_labels_batch",
]
# A resubmit pinning a source also reads that source's label.
_DISCOVERED_PINNED_RESUBMIT = [
    "node_arcs_batch",
    *["default_facet_policy"] * 3,
    *["node_labels_batch"] * 3,
    "node_types_batch",
    "node_labels_batch",
]
_STANDARD_FIRST = [
    "detect_svs",
    "node_arcs_many",
    "node_arcs_batch",
    "observation_facets_batch",
    "node_labels_batch",
    *["node_arcs_many"] * 3,
    # The menu's StatVar denominators, read for bind's list and the quantity gate
    # (`phase_rank_bind._read_denominators`).
    "node_arcs_batch",
    "node_labels_batch",
    *["default_facet_policy"] * 3,
    "observation_facets_batch",
    "node_labels_batch",
    "node_types_batch",
    "node_labels_batch",
    "observation_facets_batch",
    "node_labels_batch",
]
# The fertility rate's StatVarGroups, then its related group's members' arcs beside their
# facets at Kenya (`related.read_related`).
_STANDARD_RELATED = ["node_subjects_many", "node_arcs_batch", "observation_facets_batch"]
_STANDARD_RESUBMIT = [
    "node_arcs_batch",
    "node_labels_batch",
    "node_labels_batch",
    *["default_facet_policy"] * 3,
    "observation_facets_batch",
    "node_labels_batch",
    "node_types_batch",
    "node_labels_batch",
    "observation_facets_batch",
    "node_labels_batch",
]


def _route_requests(discovered: bool) -> tuple[list[str], list[str]]:
    """A role route's graph requests for a first answer and for its resubmit."""
    if discovered:
        return _DISCOVERED_FIRST, _DISCOVERED_RESUBMIT
    return _NAMED_FIRST, _NAMED_RESUBMIT


def _dates(spec: Spec) -> list[DateRange | None]:
    return [sv.data_date_range for sv in spec.resolved.members]


def _assert_dated_by(spec: Spec, plan: ReadPlan | None, expected: DateRange) -> None:
    """`spec` dates its StatVar `expected`, the dates `plan` serves for it."""
    assert plan is not None
    assert served_dates(plan) == {_FLOW: (expected.start, expected.end)}
    assert _dates(spec) == [expected]


# ---------------------------------------------------------------------------
# Dates: a dateless answer names the years its plan serves.
# ---------------------------------------------------------------------------


def _early_unserved_rows() -> list:
    """`f1` (IDS) runs 2010 to 2020 at Aland and Bland; Aland also holds `f9`
    (Alternative) for 2000 to 2004, which no election serves."""
    return [
        (_FLOW, "f1", _CREDITOR, _ALAND, list(range(2010, 2021))),
        (_FLOW, "f1", _CREDITOR, _BLAND, list(range(2010, 2021))),
        (_FLOW, "f9", _CREDITOR, _ALAND, list(range(2000, 2005))),
    ]


@_ROUTES
def test_a_dateless_role_answer_names_the_years_its_plan_serves(
    monkeypatch, search_settings, discovered: bool, bulk: bool
) -> None:
    """The sentence says 2010 to 2020, the span of `f1`, on a first answer and on an
    unchanged resubmit of it: Alternative's 2000 to 2004 is never read."""
    requests = _record_graph(monkeypatch, cited_sources)
    sdmx = _record_sdmx(monkeypatch)
    first_requests, resubmit_requests = _route_requests(discovered)
    first_plans = _spy_plans(monkeypatch, phase_answer)
    first, index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=_early_unserved_rows(),
        metadata=_TWO_SOURCES,
        date_request=None,
    )
    assert first.status == "definite", (first.status, first.no_data_reason)
    spec = first.specs[0]
    (plan,) = first_plans
    assert isinstance(plan, BulkPlan if bulk else PairPlan)
    _assert_dated_by(spec, plan, DateRange(start="2010", end="2020"))
    sentence = render_sentence(spec)
    assert ", 2010 to 2020. IDS." in sentence
    assert "2000" not in sentence
    assert requests == first_requests
    requests.clear()

    resubmit_plans = _spy_plans(monkeypatch, promote_ground)
    again = _resubmit(spec, index)

    assert again.status == "definite", (again.status, again.no_data_reason)
    (resubmit_plan,) = resubmit_plans
    _assert_dated_by(again.specs[0], resubmit_plan, DateRange(start="2010", end="2020"))
    assert render_sentence(again.specs[0]) == sentence
    assert requests == resubmit_requests
    assert _sdmx_requests(sdmx) == 0


@_ROUTES
def test_a_role_range_answer_states_the_span_its_served_data_holds(
    monkeypatch, search_settings, discovered: bool, bulk: bool
) -> None:
    """A 2002 to 2020 request reads `f1` from 2010, so the answer states 2010 to 2020 and
    departs from the years asked (`year_unavailable`). Alternative's 2002 to 2004 points
    sit inside the request but are never read, so they neither date the StatVar nor
    clear the departure."""
    window = DateRequest(window=TimeWindow(start_year=2002, end_year=2020), latest=False)
    requests = _record_graph(monkeypatch, cited_sources)
    first_requests, resubmit_requests = _route_requests(discovered)
    first, index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=_early_unserved_rows(),
        metadata=_TWO_SOURCES,
        date_request=window,
    )
    assert first.status == "definite", (first.status, first.no_data_reason)
    spec = first.specs[0]
    assert _dates(spec) == [DateRange(start="2010", end="2020")]
    sentence = render_sentence(spec)
    assert sentence.endswith(", 2010 to 2020. IDS.")
    assert _year_unavailable_shown(spec) == ["2010 to 2020"]
    assert requests == first_requests
    requests.clear()

    again = _resubmit(spec, index)

    assert again.status == "definite", (again.status, again.no_data_reason)
    assert render_sentence(again.specs[0]) == sentence
    assert requests == resubmit_requests


def _recorded_then_projected_rows() -> list:
    """IDS (`f1`) runs 2010 to 2020 and Alternative (`f9`) 2021 to 2028 at Aland and
    Bland."""
    return [
        (_FLOW, facet_id, _CREDITOR, place, list(years))
        for facet_id, years in (("f1", range(2010, 2021)), ("f9", range(2021, 2029)))
        for place in (_ALAND, _BLAND)
    ]


@_ROUTES
def test_a_clipped_window_states_the_served_series_span(
    monkeypatch, search_settings, discovered: bool, bulk: bool
) -> None:
    """Inside 2020 to 2022 Alternative holds four points and IDS two, so the answer serves
    Alternative from 2021. IDS publishes 2020, so the answer and its `year_unavailable`
    departure state where the served series runs, not where published data does."""
    window = DateRequest(window=TimeWindow(start_year=2020, end_year=2022), latest=False)
    first, index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=_recorded_then_projected_rows(),
        metadata=_TWO_SOURCES,
        date_request=window,
    )
    assert first.status == "definite", (first.status, first.no_data_reason)
    sentence = render_sentence(first.specs[0])
    assert sentence.endswith(", 2021 to 2022. Alternative.")
    assert _year_unavailable_shown(first.specs[0]) == ["2021 to 2022"]

    again = _resubmit(first.specs[0], index)

    assert again.status == "definite", (again.status, again.no_data_reason)
    assert render_sentence(again.specs[0]) == sentence


@pytest.mark.parametrize("time_text", [None, "2015 to 2020"], ids=["dateless", "range"])
def test_a_standard_answer_names_the_years_its_plan_serves(monkeypatch, time_text) -> None:
    """Kenya's fertility rate holds USD Source for 2018 to 2020 and EUR Source for 2015 to
    2016. The plan serves USD Source's three points, so the answer is dated 2018 to 2020,
    and a 2015 to 2020 request departs from the years asked with the served span."""
    related: list[str] = []
    requests = _record_graph(monkeypatch, parsed, related=related)
    first, plan = _fertility(monkeypatch, time_text=time_text)

    assert plan is not None
    assert list(served_dates(plan).values()) == [("2018", "2020")]
    assert [(d.start, d.end) for d in _dates(first) if d is not None] == [("2018", "2020")]
    sentence = render_sentence(first)
    assert "2015" not in sentence.split(".")[0]
    assert "2018 to 2020" in sentence
    assert _year_unavailable_shown(first) == (["2018 to 2020"] if time_text else [])
    assert requests == _STANDARD_FIRST
    assert related == _STANDARD_RELATED
    requests.clear()

    again, _resubmit_plan = _resubmit_fertility(monkeypatch, first)

    assert again.spec_id == first.spec_id
    assert _dates(again) == _dates(first)
    assert render_sentence(again) == sentence
    assert requests == _STANDARD_RESUBMIT
    assert related == _STANDARD_RELATED * 2


def test_a_source_pin_dates_the_answer_from_the_pinned_source(monkeypatch, search_settings) -> None:
    """A resubmit pinning Alternative serves `f9` alone, so the answer is dated by its
    2016 to 2018, as its sources and coverage are."""
    rows = [
        (_FLOW, "f1", _CREDITOR, _ALAND, list(range(2010, 2021))),
        (_FLOW, "f1", _CREDITOR, _BLAND, list(range(2010, 2021))),
        (_FLOW, "f9", _CREDITOR, _ALAND, list(range(2016, 2019))),
    ]
    first, index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=False,
        bulk=False,
        rows=rows,
        metadata=_TWO_SOURCES,
        date_request=None,
    )
    assert first.status == "definite", (first.status, first.no_data_reason)

    pinned = _resubmit(first.specs[0], index, facets={"provenance": IdentityBindingValue(ref=_ALT)})

    assert pinned.status == "definite", (pinned.status, pinned.no_data_reason)
    spec = pinned.specs[0]
    assert [ref.ref for ref in spec.resolved.trace.resolved_sources] == [_ALT]
    assert _dates(spec) == [DateRange(start="2016", end="2018")]


# ---------------------------------------------------------------------------
# Scope: a discovered scope lists, counts and warns about the members its plan serves.
# ---------------------------------------------------------------------------


def _three_recipient_rows() -> list:
    """`f1` (IDS) at Aland and Bland, `f9` (Alternative) at Cland alone. Per pair and in
    bulk the plan elects `f1`, which covers two recipients, and serves Aland and Bland."""
    return [
        (_FLOW, "f1", _CREDITOR, _ALAND, list(range(2010, 2021))),
        (_FLOW, "f1", _CREDITOR, _BLAND, list(range(2010, 2021))),
        (_FLOW, "f9", _CREDITOR, _CLAND, list(range(2016, 2019))),
    ]


_DISCOVERED = CoverageForm(variables=True, open_axis=("recipients", "recipient"))


def _stated_scope(result) -> tuple[list[str], int, str]:
    """The scope a discovered answer lists, its coverage recipient count and the warning
    that counts the scope."""
    resolved = result.specs[0].resolved
    assert resolved.scope is not None
    dims = {d.label: d.count for d in resolved.coverage.dimensions}
    (warning,) = [
        w.message
        for w in result.warnings
        if w.code == "role_scope_defaulted" and "no recipient was named" in w.message
    ]
    return [m.ref for m in resolved.scope.members], dims["recipients"], warning


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
@pytest.mark.parametrize(
    "date_request", [None, DateRequest(window=None, latest=True)], ids=["dateless", "latest"]
)
def test_a_discovered_scope_lists_the_members_its_plan_serves(
    monkeypatch, search_settings, bulk: bool, date_request
) -> None:
    first_plans = _spy_plans(monkeypatch, phase_answer)
    first, index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=True,
        bulk=bulk,
        rows=_three_recipient_rows(),
        metadata=_TWO_SOURCES,
        date_request=date_request,
    )
    assert first.status == "definite", (first.status, first.no_data_reason)
    (plan,) = first_plans
    assert plan is not None
    expected = [_ALAND, _BLAND]
    assert scope_members(plan, _DISCOVERED) == expected

    members, count, warning = _stated_scope(first)
    assert members == expected
    assert count == len(expected)
    assert f"resolved across the {len(expected)} recipients with data" in warning

    resubmit_plans = _spy_plans(monkeypatch, promote_ground)
    again = _resubmit(first.specs[0], index)

    assert again.status == "definite", (again.status, again.no_data_reason)
    (resubmit_plan,) = resubmit_plans
    assert resubmit_plan is not None
    assert scope_members(resubmit_plan, _DISCOVERED) == expected
    resolved = again.specs[0].resolved
    assert resolved.scope is not None
    assert [m.ref for m in resolved.scope.members] == expected
    assert {d.label: d.count for d in resolved.coverage.dimensions}["recipients"] == len(expected)


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
def test_a_source_pin_lists_the_scope_members_the_pinned_source_serves(
    monkeypatch, search_settings, bulk: bool
) -> None:
    """A resubmit pinning Alternative serves `f9` at Cland alone: the scope lists Cland, as
    coverage counts it."""
    first, index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=True,
        bulk=bulk,
        rows=_three_recipient_rows(),
        metadata=_TWO_SOURCES,
        date_request=None,
    )
    assert first.status == "definite", (first.status, first.no_data_reason)

    requests = _record_graph(monkeypatch, cited_sources)
    pinned = _resubmit(first.specs[0], index, facets={"provenance": IdentityBindingValue(ref=_ALT)})

    assert pinned.status == "definite", (pinned.status, pinned.no_data_reason)
    resolved = pinned.specs[0].resolved
    assert resolved.scope is not None
    assert [m.ref for m in resolved.scope.members] == [_CLAND]
    assert {d.label: d.count for d in resolved.coverage.dimensions}["recipients"] == 1
    assert [ref.ref for ref in resolved.trace.resolved_sources] == [_ALT]
    # The warning counts the one member, in the place the materialised warning held.
    assert [(w.code, w.message) for w in pinned.warnings if w.code == "role_scope_defaulted"] == [
        (
            "role_scope_defaulted",
            "no recipient was named; the answer was resolved across the 1 recipient with data.",
        ),
        (
            "role_scope_defaulted",
            "no provider was named; the answer was resolved across every provider with data.",
        ),
    ]
    assert requests == _DISCOVERED_PINNED_RESUBMIT


def _fan_out_rows() -> list:
    """Flow holds IDS and Alternative at Aland alone; FlowAlt holds both at Aland and
    Bland."""
    alt_flow = cited_sources._ALT_FLOW
    return [
        (_FLOW, "f1", _CREDITOR, _ALAND, list(range(2010, 2021))),
        (_FLOW, "f9", _CREDITOR, _ALAND, list(range(2016, 2019))),
        (alt_flow, "f1", _CREDITOR, _ALAND, list(range(2010, 2021))),
        (alt_flow, "f1", _CREDITOR, _BLAND, list(range(2010, 2021))),
        (alt_flow, "f9", _CREDITOR, _ALAND, list(range(2016, 2019))),
        (alt_flow, "f9", _CREDITOR, _BLAND, list(range(2016, 2019))),
    ]


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
@pytest.mark.parametrize("unit", [None, "runner_up_alone"], ids=["no_unit", "usd"])
def test_candidates_serving_different_scopes_share_a_warning_without_a_count(
    monkeypatch, search_settings, bulk: bool, unit: str | None
) -> None:
    """Flow serves Aland and FlowAlt serves Aland and Bland. Each candidate's scope and
    coverage count its own recipients, and the response's one warning, which speaks for
    both, names every recipient with data instead of the elected candidate's count."""
    flows = (_FLOW, cited_sources._ALT_FLOW)
    kwargs: dict = {}
    metadata = _TWO_SOURCES
    if unit is not None:
        case = cited_sources._DOLLAR_CASES[unit]
        metadata = case.metadata
        kwargs = {
            "unit_text": "USD",
            "sdmx": cited_sources._dollar_series(case, flows),
            "options": cited_sources._dollar_options(case, flows),
        }
    result, _index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=True,
        bulk=bulk,
        rows=_fan_out_rows(),
        metadata=metadata,
        flows=flows,
        llm=cited_sources._BindLLM(also_plausible=(1,)),
        date_request=cited_sources._CLOSED,
        **kwargs,
    )

    assert result.status == "candidates", (result.status, result.no_data_reason)
    scopes = {}
    for spec in result.specs:
        resolved = spec.resolved
        assert resolved.scope is not None
        members = [m.ref for m in resolved.scope.members]
        assert {d.label: d.count for d in resolved.coverage.dimensions}["recipients"] == len(
            members
        )
        scopes[resolved.members[0].ref.ref] = members
    assert scopes == {_FLOW: [_ALAND], cited_sources._ALT_FLOW: [_ALAND, _BLAND]}
    warnings = [w.message for w in result.warnings if w.code == "role_scope_defaulted"]
    assert warnings[0] == (
        "no recipient was named; the answer was resolved across every recipient with data."
    )


@_ROUTES
@pytest.mark.parametrize(
    "date_request", [None, cited_sources._CLOSED], ids=["dateless", "2015_to_2020"]
)
@pytest.mark.parametrize("also_plausible", [False, True], ids=["one_answer", "candidates"])
def test_a_unit_request_dates_every_answer_from_its_narrowed_plan(
    monkeypatch, search_settings, discovered: bool, bulk: bool, date_request, also_plausible: bool
) -> None:
    """IDS reports no unit and Alternative US dollars at Aland for 2016 to 2018, so a "USD"
    request serves Alternative alone: every answer is dated 2016 to 2018, where its
    unfiltered plan serves IDS's 2010 to 2020 (2015 to 2020 inside the range), and a
    discovered scope, its coverage and its warning count Aland alone, a candidate's as the
    single answer's."""
    case = cited_sources._DOLLAR_CASES["runner_up_alone"]
    flows = (_FLOW, cited_sources._ALT_FLOW) if also_plausible else (_FLOW,)
    result, _index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=cited_sources._candidate_rows()
        if also_plausible
        else cited_sources._two_source_rows(),
        metadata=case.metadata,
        flows=flows,
        date_request=date_request,
        llm=cited_sources._BindLLM(also_plausible=(1,) if also_plausible else ()),
        unit_text="USD",
        sdmx=cited_sources._dollar_series(case, flows),
        options=cited_sources._dollar_options(case, flows),
    )

    assert result.status == ("candidates" if also_plausible else "definite"), (
        result.status,
        result.no_data_reason,
    )
    assert len(result.specs) == len(flows)
    for spec in result.specs:
        assert _dates(spec) == [DateRange(start="2016", end="2018")]
        sentence = render_sentence(spec)
        assert ", 2016 to 2018. Current US dollars · Alternative." in sentence
        if date_request is not None:
            # IDS publishes 2015 to 2020 in another unit, so the served span departs.
            assert _year_unavailable_shown(spec) == ["2016 to 2018"]
        # Bland holds IDS alone, so a query naming it states the series serves nothing there.
        assert (spec.resolved.unserved_entities == (("destination", _BLAND),)) is not discovered
    if discovered:
        for spec in result.specs:
            assert spec.resolved.scope is not None
            assert [m.ref for m in spec.resolved.scope.members] == [_ALAND]
            assert {d.label: d.count for d in spec.resolved.coverage.dimensions}["recipients"] == 1
        warnings = [w.message for w in result.warnings if w.code == "role_scope_defaulted"]
        assert warnings == [
            "no recipient was named; the answer was resolved across the 1 recipient with data.",
            "no provider was named; the answer was resolved across every provider with data.",
        ]


@pytest.mark.parametrize("discovered", [False, True], ids=["named", "discovered"])
def test_a_unit_request_reports_its_facets_from_the_plan_with_or_without_a_preview(
    monkeypatch, search_settings, discovered: bool
) -> None:
    """IDS and Alternative both report US dollars, and a "USD" request serves IDS. The first
    answer and an unchanged resubmit of it report IDS and US dollars in `resolved.facets`
    and name the unit in the sentence, with the preview on and off alike."""
    case = cited_sources._DOLLAR_CASES["both_sources"]
    stated = set()
    for preview in (True, False):
        options = dataclasses.replace(
            cited_sources._dollar_options(case), include_observations=preview
        )
        first, index = _role_answer(
            monkeypatch,
            search_settings,
            discovered=discovered,
            bulk=False,
            rows=cited_sources._two_source_rows(),
            metadata=case.metadata,
            unit_text="USD",
            sdmx=cited_sources._dollar_series(case),
            options=options,
        )
        assert first.status == "definite", (first.status, first.no_data_reason)
        spec = first.specs[0]
        assert (spec.resolved.observations is not None) == preview

        again = _resubmit(spec, index, sdmx=cited_sources._dollar_series(case), options=options)

        assert again.status == "definite", (again.status, again.no_data_reason)
        for answer in (spec, again.specs[0]):
            facets = {axis: f.ref for axis, f in answer.resolved.facets.items()}
            sentence = render_sentence(answer)
            assert facets == {"provenance": _IDS, "unit": cited_sources._USD}
            # US dollars keep the magnitude, so the heading leaves the unit to the byline.
            assert "USD" not in sentence
            stated.add((tuple(sorted(facets.items())), sentence))
    assert len(stated) == 1, stated


@pytest.mark.parametrize("preview", [True, False], ids=["preview", "no_preview"])
def test_a_unit_request_warns_alike_on_a_first_answer_and_its_resubmit(
    monkeypatch, search_settings, preview: bool
) -> None:
    """IDS reports no unit and Alternative US dollars at Aland alone, so a "USD" request for
    Aland and Bland serves Aland and leaves Bland with no matching source. A first answer
    fetches observations to check the unit whether or not a preview is asked for; it states
    that fetch's warning only with the preview, as its unchanged resubmit does."""
    case = cited_sources._DOLLAR_CASES["runner_up_alone"]
    options = dataclasses.replace(cited_sources._dollar_options(case), include_observations=preview)
    first, index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=False,
        bulk=False,
        rows=cited_sources._two_source_rows(),
        metadata=case.metadata,
        unit_text="USD",
        sdmx=cited_sources._dollar_series(case),
        options=options,
    )
    assert first.status == "definite", (first.status, first.no_data_reason)

    again = _resubmit(
        first.specs[0], index, sdmx=cited_sources._dollar_series(case), options=options
    )

    assert again.status == "definite", (again.status, again.no_data_reason)
    stated = [
        [(w.code, w.message) for w in answer.warnings if w.code.startswith("observations_")]
        for answer in (first, again)
    ]
    assert stated[0] == stated[1]
    assert [code for code, _message in stated[0]] == (
        ["observations_no_matching_source"] if preview else []
    )


# ---------------------------------------------------------------------------
# The read decision: no series is built for a plan that reads in bulk.
# ---------------------------------------------------------------------------

_RECIPIENTS = [f"country/R{i:03d}" for i in range(100)]
_ROLE_NAMES = {_FLOW: RoleNames(donor="creditor", recipient="debtor")}


def _open_origin_index(creditors: int):
    """Every one of 100 recipients holds `f1` from each of `creditors` creditors in 2020."""
    return build_coverage_index(
        [
            (_FLOW, "f1", f"country/C{c:02d}", recipient, [2020])
            for recipient in _RECIPIENTS
            for c in range(creditors)
        ],
        role_names=_ROLE_NAMES,
        facet_metadata={"f1": FacetMetadata(provenance_id=_IDS)},
    )


def _spy_builder(monkeypatch, module) -> list[int]:
    """Record how many series each `pair_series` call `module` makes builds."""
    built: list[int] = []
    real = module.pair_series

    def _pair_series(*args, **kwargs):
        series = real(*args, **kwargs)
        built.append(len(series))
        return series

    monkeypatch.setattr(module, "pair_series", _pair_series)
    return built


def _named_plan(index) -> ReadPlan:
    return role_resolve._role_plan(
        [_FLOW],
        member_facts={_FLOW: ArcsMemberFacts(arcs={})},
        names={"destination": "debtor", "origin": "creditor"},
        subject_dcids=_RECIPIENTS,
        origin_dcids=[],
        bound_roles={"recipient": frozenset(_RECIPIENTS)},
        years=role_years(None, index.horizon),
        coverage_index=index,
    )


def _discovered_plan(index) -> ReadPlan | None:
    result = role_scope.discover_role_scope(
        surviving_svs=[_FLOW],
        admitted_dcids=[_FLOW],
        member_facts={_FLOW: ArcsMemberFacts(arcs={})},
        member_population_exact=True,
        open_flow_axis="destination",
        role_name="debtor",
        opposite_role_name="creditor",
        opposite_dcids=(),
        facet_filters={},
        date_request=None,
        coverage_snapshot=FakeCoverageService(index).snapshot(),
    )
    assert isinstance(result, Materialised)
    return result.read_plan


@pytest.mark.parametrize(
    ("module", "plan_of"),
    [(role_resolve, _named_plan), (role_scope, _discovered_plan)],
    ids=["named", "discovered"],
)
def test_an_open_origin_over_the_series_cap_reads_in_bulk_without_building_a_series(
    monkeypatch, search_settings, module, plan_of
) -> None:
    """100 recipients fit the 120-pair cap, but an open origin of 60 creditors makes 6,000
    series: the plan reads in bulk and builds none. With one creditor the 100 series fit,
    and the plan builds and reads them per pair."""
    search_settings(obs_preview_max_pairs=120)
    built = _spy_builder(monkeypatch, module)

    assert isinstance(plan_of(_open_origin_index(60)), BulkPlan)
    assert built == []

    plan = plan_of(_open_origin_index(1))
    assert isinstance(plan, PairPlan)
    assert built == [100]
    assert len(plan.served) == 100


def _two_creditor_rows() -> list:
    """Aland and Bland each hold `f1` (IDS) from two creditors: four series."""
    return [
        (_FLOW, "f1", creditor, recipient, list(range(2010, 2021)))
        for recipient in (_ALAND, _BLAND)
        for creditor in (_CREDITOR, "country/CRB")
    ]


@pytest.mark.parametrize("discovered", [False, True], ids=["named", "discovered"])
def test_an_answer_over_the_series_cap_reads_in_bulk_without_building_a_series(
    monkeypatch, search_settings, discovered: bool
) -> None:
    """Two recipients against a cap of three fit the bound product, but their two creditors
    make four series: the first answer and its resubmit read in bulk, build no series and
    make the route's usual graph requests. A cap of four reads the same rows per pair."""
    requests = _record_graph(monkeypatch, cited_sources)
    first_requests, resubmit_requests = _route_requests(discovered)
    built_named = _spy_builder(monkeypatch, role_resolve)
    built_discovered = _spy_builder(monkeypatch, role_scope)

    def _answer(cap: int):
        requests.clear()
        plans = _spy_plans(monkeypatch, phase_answer)
        first, index = _role_answer(
            monkeypatch,
            lambda **_: search_settings(obs_preview_max_pairs=cap),
            discovered=discovered,
            bulk=False,
            rows=_two_creditor_rows(),
            metadata=_TWO_SOURCES,
        )
        assert first.status == "definite", (first.status, first.no_data_reason)
        assert requests == first_requests
        requests.clear()
        resubmit_plans = _spy_plans(monkeypatch, promote_ground)
        again = _resubmit(first.specs[0], index)
        assert again.status == "definite", (again.status, again.no_data_reason)
        assert requests == resubmit_requests
        assert render_sentence(again.specs[0]) == render_sentence(first.specs[0])
        return [*plans, *resubmit_plans]

    plans = _answer(3)
    assert [type(plan) for plan in plans] == [BulkPlan, BulkPlan]
    assert built_named == [] and built_discovered == []

    plans = _answer(4)
    assert [type(plan) for plan in plans] == [PairPlan, PairPlan]
    assert (built_discovered if discovered else built_named) == [4, 4]


_OLD_PARTIES = [f"country/O{i:03d}" for i in range(300)]
_RECENT = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)


@pytest.mark.parametrize("discovered", [False, True], ids=["named", "discovered"])
def test_a_per_pair_plan_builds_only_the_series_holding_a_point_in_its_years(
    monkeypatch, search_settings, discovered: bool
) -> None:
    """Aland and Bland hold `f1` in 2005 and for 2015 to 2020, and 300 other parties hold it
    only in 2000 and 2001: 300 recipients on the discovered route, 300 creditors at Aland on
    the named route with an open origin. Inside 2015 to 2020 the plan counts two series and
    builds those two, each with its facets from every year, 2005 included, and serves
    both."""
    search_settings(obs_preview_max_pairs=120)
    built = _spy_builder(monkeypatch, role_scope if discovered else role_resolve)
    recent = [
        (_FLOW, "f1", _CREDITOR, recipient, [2005, *range(2015, 2021)])
        for recipient in (_ALAND, _BLAND)
    ]
    old = (
        [(_FLOW, "f1", _CREDITOR, recipient, [2000, 2001]) for recipient in _OLD_PARTIES]
        if discovered
        else [(_FLOW, "f1", creditor, _ALAND, [2000, 2001]) for creditor in _OLD_PARTIES]
    )
    index = build_coverage_index(
        [*recent, *old],
        role_names=_ROLE_NAMES,
        facet_metadata={"f1": FacetMetadata(provenance_id=_IDS)},
    )
    years = role_years(_RECENT, index.horizon)
    if discovered:
        result = role_scope.discover_role_scope(
            surviving_svs=[_FLOW],
            admitted_dcids=[_FLOW],
            member_facts={_FLOW: ArcsMemberFacts(arcs={})},
            member_population_exact=True,
            open_flow_axis="destination",
            role_name="debtor",
            opposite_role_name="creditor",
            opposite_dcids=(),
            facet_filters={},
            date_request=_RECENT,
            coverage_snapshot=FakeCoverageService(index).snapshot(),
        )
        assert isinstance(result, Materialised)
        plan = result.read_plan
    else:
        plan = role_resolve._role_plan(
            [_FLOW],
            member_facts={_FLOW: ArcsMemberFacts(arcs={})},
            names={"destination": "debtor", "origin": "creditor"},
            subject_dcids=[_ALAND, _BLAND],
            origin_dcids=[],
            bound_roles={"recipient": frozenset({_ALAND, _BLAND})},
            years=years,
            coverage_index=index,
        )

    assert isinstance(plan, PairPlan)
    assert built == [2]
    assert [s.rows for s in plan.served] == [(_ALAND, _CREDITOR), (_BLAND, _CREDITOR)]
    assert [(f.earliest_date, f.latest_date) for f in plan.served[0].candidates] == [
        ("2005", "2020")
    ]


# ---------------------------------------------------------------------------
# A facet declaring no provenance serves per pair as it does in bulk.
# ---------------------------------------------------------------------------

_NO_PROVENANCE = {"f1": FacetMetadata(), "f9": FacetMetadata(provenance_id=_ALT)}


def _unsourced_rows() -> list:
    """`f1`, declaring no provenance, runs 2010 to 2020 at Aland and Bland and outranks
    Alternative's `f9`, 2016 to 2018 at Aland."""
    return [
        (_FLOW, "f1", _CREDITOR, _ALAND, list(range(2010, 2021))),
        (_FLOW, "f1", _CREDITOR, _BLAND, list(range(2010, 2021))),
        (_FLOW, "f9", _CREDITOR, _ALAND, list(range(2016, 2019))),
    ]


_UNSOURCED_REQUESTS = {
    "dateless": (None, ", 2010 to 2020.", 22),
    "range": (
        DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False),
        ", 2015 to 2020.",
        12,
    ),
    "latest": (DateRequest(window=None, latest=True), " in 2020. Latest year with data.", 2),
}


@pytest.mark.parametrize("request_kind", list(_UNSOURCED_REQUESTS))
@_ROUTES
def test_a_facet_without_provenance_answers_alike_per_pair_and_in_bulk(
    monkeypatch, search_settings, discovered: bool, bulk: bool, request_kind: str
) -> None:
    """Per pair or in bulk, named or discovered, the plan serves `f1` at both places: the
    answer states its points and years and names no source, on a first answer and on an
    unchanged resubmit of it."""
    date_request, years_text, observations = _UNSOURCED_REQUESTS[request_kind]
    requests = _record_graph(monkeypatch, cited_sources)
    first_requests, resubmit_requests = _route_requests(discovered)
    first_plans = _spy_plans(monkeypatch, phase_answer)
    first, index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=_unsourced_rows(),
        metadata=_NO_PROVENANCE,
        date_request=date_request,
    )
    assert first.status == "definite", (first.status, first.no_data_reason)
    (plan,) = first_plans
    if isinstance(plan, PairPlan):
        assert len(plan.served) == 2
        assert plan.unservable == ()
    else:
        assert isinstance(plan, BulkPlan) and plan.facet_id == "f1"
    sentence = render_sentence(first.specs[0])
    place = "to 2 recipients" if discovered else "to Aland and Bland"
    assert sentence == f"Test flow {place}{years_text}"
    resolved = first.specs[0].resolved
    assert resolved.coverage.has_data
    assert resolved.trace.resolved_sources == []
    dims = {d.label: d.count for d in resolved.coverage.dimensions or []}
    assert dims["sources"] == 0
    if not discovered:
        assert resolved.coverage.observation_count == observations
    assert requests == first_requests
    requests.clear()

    again = _resubmit(first.specs[0], index)

    assert again.status == "definite", (again.status, again.no_data_reason)
    assert render_sentence(again.specs[0]) == sentence
    assert again.specs[0].resolved.coverage == resolved.coverage
    assert requests == resubmit_requests


def _unsourced_preview(urls: list[str]) -> MaterialiseSdmxClient:
    """The per-pair read's rows: `f1` at the debtor a request binds, declaring no
    provenance, for 2010 to 2020. Records each request's URL."""
    fixture = ObsFacetFixture(facet_id="f1")

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        debtor = request.url.params.get("c[debtor]", "")
        body = _csv_header(["creditor", "debtor"]) + "".join(
            _csv_row(
                stat_var=_FLOW,
                role_dcids=[_CREDITOR, debtor],
                fixture=fixture,
                date_str=str(year),
                value=float(year),
            )
            for year in range(2010, 2021)
        )
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    return MaterialiseSdmxClient(
        base_url=MATERIALISE_BASE_URL, transport=httpx.MockTransport(handler)
    )


def _unsourced_bulk() -> FakeSdmxClient:
    """The bulk read's rows: `f1` at Aland and Bland for 2010 to 2020."""
    years = [str(year) for year in range(2010, 2021)]
    facet = Facet(
        facet_id="f1",
        earliest_date=years[0],
        latest_date=years[-1],
        obs_count=len(years),
        dates=years,
        values=[float(year) for year in years],
    )
    return FakeSdmxClient(
        observations={
            _FLOW: {
                (("creditor", _CREDITOR), ("debtor", place)): [facet] for place in (_ALAND, _BLAND)
            }
        },
        availability={},
    )


@_ROUTES
def test_a_facet_without_provenance_is_read_by_its_facet_id(
    monkeypatch, search_settings, discovered: bool, bulk: bool
) -> None:
    """The per-pair read binds the provenance axis to the null ref and fetches `f1` by its id
    at each pair, one request per pair; the bulk read fetches it by id in one request. Both
    deliver the same 22 points."""
    urls: list[str] = []
    sdmx = _unsourced_bulk()
    result, _index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=_unsourced_rows(),
        metadata=_NO_PROVENANCE,
        date_request=None,
        sdmx=sdmx,
        options=EngineOptions(
            include_observations=True,
            materialise_graph=FakeGraphReader(),
            materialise_sdmx=_unsourced_preview(urls),
        ),
    )

    assert result.status == "definite", (result.status, result.no_data_reason)
    observations = result.specs[0].resolved.observations
    assert observations is not None and observations.omitted is None
    assert sorted(len(series.points) for series in observations.series) == [11, 11]
    assert "provenance" not in result.specs[0].resolved.facets
    if bulk:
        assert urls == []
        assert [call["facet_ids"] for call in sdmx.series_calls] == [{"f1"}]
    else:
        assert sdmx.series_calls == []
        assert len(urls) == 2
        assert all(httpx.URL(url).params.get("c[facetId]") == "f1" for url in urls)


def _unsourced_kenya_obs() -> dict:
    """Kenya's fertility rate in one facet declaring no provenance, 2018 to 2020."""
    return {
        parsed._FERTILITY_KEN_KEY: [
            {
                "earliestDate": "2018",
                "latestDate": "2020",
                "obsCount": 3,
                "dates": ["2018", "2019", "2020"],
                "values": [1.0, 2.0, 3.0],
                "facetId": "fk",
            }
        ]
    }


@pytest.mark.parametrize("time_text", [None, "2018 to 2020"], ids=["dateless", "range"])
def test_a_standard_facet_without_provenance_states_its_years(monkeypatch, time_text) -> None:
    """The standard route serves Kenya's one unsourced facet: the answer names its years and
    no source, on a first answer and on an unchanged resubmit."""
    monkeypatch.setattr(parsed, "_two_unit_facets_obs", _unsourced_kenya_obs)
    first, plan = _fertility(monkeypatch, time_text=time_text)

    assert isinstance(plan, PairPlan) and len(plan.served) == 1
    sentence = render_sentence(first)
    assert sentence == "Kenya's female fertility rate, 2018 to 2020."
    assert first.resolved.trace.resolved_sources == []

    again, _plan = _resubmit_fertility(monkeypatch, first)

    assert render_sentence(again) == sentence


# ---------------------------------------------------------------------------
# A future window inside the coverage horizon reads coverage on every route.
# ---------------------------------------------------------------------------

_FUTURE = {
    "2027": (TimeWindow(start_year=2027, end_year=2027), " in 2027"),
    "2027_to_2028": (TimeWindow(start_year=2027, end_year=2028), ", 2027 to 2028"),
    "since_2027": (TimeWindow(start_year=2027, end_year=None), ", 2027 to 2028"),
}


@pytest.mark.parametrize("window_name", list(_FUTURE))
@_ROUTES
def test_a_future_window_inside_the_horizon_answers_from_coverage(
    monkeypatch, search_settings, discovered: bool, bulk: bool, window_name: str
) -> None:
    """Coverage holds `f1` to 2028, past the pinned 2026 clock. A window in 2027 reads it on
    every role route, on a first answer and on an unchanged resubmit."""
    window, years_text = _FUTURE[window_name]
    rows = [(_FLOW, "f1", _CREDITOR, place, list(range(2015, 2029))) for place in (_ALAND, _BLAND)]
    first, index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=rows,
        metadata=_TWO_SOURCES,
        date_request=DateRequest(window=window, latest=False),
    )

    assert first.status == "definite", (first.status, first.no_data_reason)
    place = "to 2 recipients" if discovered else "to Aland and Bland"
    sentence = render_sentence(first.specs[0])
    assert sentence == f"Test flow {place}{years_text}. IDS."
    assert first.specs[0].resolved.coverage.has_data

    again = _resubmit(first.specs[0], index)

    assert again.status == "definite", (again.status, again.no_data_reason)
    assert render_sentence(again.specs[0]) == sentence


def _future_kenya_obs() -> dict:
    """Kenya's fertility rate from USD Source, 2024 to 2028."""
    years = [str(year) for year in range(2024, 2029)]
    return {
        parsed._FERTILITY_KEN_KEY: [
            {
                "earliestDate": years[0],
                "latestDate": years[-1],
                "obsCount": len(years),
                "dates": years,
                "values": [1.0] * len(years),
                "unit": "CurrencyUSD",
                "provenanceId": "dc/base/UsdSource",
                "importName": "UsdSource",
                "facetId": "fx",
            }
        ]
    }


@pytest.mark.parametrize(
    ("time_text", "years_text", "points"),
    [("2027", " in 2027", 1), ("2027 to 2028", ", 2027 to 2028", 2)],
    ids=["2027", "2027_to_2028"],
)
def test_a_standard_future_window_answers_from_its_facets(
    monkeypatch, time_text, years_text, points
) -> None:
    monkeypatch.setattr(parsed, "_two_unit_facets_obs", _future_kenya_obs)
    first, _plan = _fertility(monkeypatch, time_text=time_text)

    sentence = render_sentence(first)
    assert sentence == (f"Kenya's female fertility rate{years_text}. US dollars · USD Source.")
    assert first.resolved.coverage.observation_count == points

    again, _plan = _resubmit_fertility(monkeypatch, first)

    assert render_sentence(again) == sentence


# ---------------------------------------------------------------------------
# A window with no point names the span of the series a dateless answer serves.
# ---------------------------------------------------------------------------

_LATER = DateRequest(window=TimeWindow(start_year=2023, end_year=2025), latest=False)
# The `year_unavailable` fact of a 2023 to 2025 window over a series running 2010 to 2020.
_LATER_GAP = ("2023 to 2025", (2010, 2020))


def _resubmit_window(spec: Spec, index, window: TimeWindow):
    """Post `spec`'s identity back with its requested window changed to `window`."""
    edited = spec.model_copy(
        update={
            "definition": spec.definition.model_copy(
                update={
                    "identity": spec.definition.identity.model_copy(
                        update={
                            "requested_window": IdentityWindowRange(
                                start_year=window.start_year, end_year=window.end_year
                            )
                        }
                    )
                }
            )
        }
    )
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=edited.definition.identity,
        ),
        graph=FakeGraph(nodes=cited_sources._nodes(), detect={}, resolve={}, obs={}),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=FakeCoverageService(index).snapshot(),
        options=walk_options(
            FakeGraph(nodes=cited_sources._nodes(), detect={}, resolve={}, obs={}), EngineOptions()
        ),
    )


@_ROUTES
def test_a_window_without_points_names_the_span_a_dateless_answer_serves(
    monkeypatch, search_settings, discovered: bool, bulk: bool
) -> None:
    """`f1` (IDS) runs 2010 to 2020 and the unelected `f9` (Alternative) 2000 to 2004. A
    dateless answer names 2010 to 2020; a 2023 to 2025 request, first asked or edited into
    a resubmit of the dateless answer, says the series runs 2010 to 2020, not 2000."""
    dateless, index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=_early_unserved_rows(),
        metadata=_TWO_SOURCES,
        date_request=None,
    )
    assert ", 2010 to 2020. IDS." in render_sentence(dateless.specs[0])

    later, _index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=_early_unserved_rows(),
        metadata=_TWO_SOURCES,
        date_request=_LATER,
    )
    edited = _resubmit_window(dateless.specs[0], index, _LATER.window)

    for answer in (later, edited):
        assert (answer.status, answer.no_data_reason) == ("no_data", "no_observations")
        assert answer.window_gap_fact is not None
        assert (answer.window_gap_fact.asked, answer.window_gap_fact.data_years) == _LATER_GAP


def test_a_standard_window_without_points_names_the_span_a_dateless_answer_serves(
    monkeypatch,
) -> None:
    """Kenya's fertility rate serves USD Source, 2018 to 2020; EUR Source's 2015 to 2016 is
    passed over. A 2023 to 2025 request, first asked or edited into a resubmit of the
    dateless answer, says the series runs 2018 to 2020, as the dateless answer does."""
    dateless, _plan = _fertility(monkeypatch)
    assert ", 2018 to 2020. US dollars · USD Source." in render_sentence(dateless)

    first = parsed._resolve_parsed(
        [parsed._fertility_kenya_part(time_text="2023 to 2025")],
        graph=parsed._graph_with_two_unit_facets_on_fertility_kenya(),
    )
    assert first.status == "no_data"
    assert [(d.code, d.asked) for d in first.parts[0].departures] == [
        ("year_unavailable", "2023 to 2025")
    ]

    identity = dateless.definition.identity.model_copy(
        update={"requested_window": IdentityWindowRange(start_year=2023, end_year=2025)}
    )
    edited = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=identity,
        ),
        graph=parsed._graph_with_two_unit_facets_on_fertility_kenya(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            parsed._graph_with_two_unit_facets_on_fertility_kenya(), EngineOptions()
        ),
    )
    assert (edited.status, edited.no_data_reason) == ("no_data", "no_observations")
    assert edited.window_gap_fact is not None
    assert (edited.window_gap_fact.asked, edited.window_gap_fact.data_years) == (
        "2023 to 2025",
        (2018, 2020),
    )
