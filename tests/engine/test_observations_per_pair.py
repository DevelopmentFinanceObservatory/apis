"""Tests per-pair observation request construction, preview limits, and rate-cache forwarding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from unittest.mock import patch

import pytest

import search.engine.region.observations.fetch as region_fetch
from materialise import (
    MaterialisationRequest,
    SpanRead,
)
from materialise.graph import Facet as MaterialiseFacet
from materialise.rate_tables import RateTableCache
from materialise.window import RequestedWindow
from search.engine import config
from search.engine.app import create_app, search_surface
from search.engine.retrieval.facet_policy import NO_DECLARED_POLICIES
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.read_plan import PairPlan, PlanSeries, plan_pairs, plan_window
from search.models import (
    TimeWindow,
)
from search.observations import OBSERVATIONS_PREVIEW_TOO_LARGE, observation_settings
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    ObsFacetFixture,
    build_materialise_double,
    build_observations_planned,
    harness_reader,
    sdmx_double,
    stub_preview_result,
)
from tests.fixtures import FakeGraph, FakeLLM
from tests.materialise.conftest import FakeGraphReader

# ---------------------------------------------------------------------------
# Per-pair materialise_preview request construction: real materialise_preview
# (a FakeGraphReader + dc_sdmx.SdmxClient(transport=httpx.MockTransport(...)),
# the same fixture pattern tests/materialise/test_preview.py uses) or a
# monkeypatched materialise_preview spy when only _build_observations' own
# per-pair dispatch/aggregation is under test.
# ---------------------------------------------------------------------------


def test_search_surface_accepts_materialise_graph_and_sdmx_injection():
    """Explicit materialise clients construct `search_surface` successfully."""
    surface = search_surface(
        graph=FakeGraph(),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
        materialise_graph=FakeGraphReader(),
        materialise_sdmx=sdmx_double(),
    )
    assert surface is not None


def test_create_app_accepts_materialise_graph_and_sdmx_injection():
    """Same construction/injection check, one layer up at create_app()."""
    app = create_app(
        graph=FakeGraph(),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
        materialise_graph=FakeGraphReader(),
        materialise_sdmx=sdmx_double(),
    )
    assert app is not None


def test_per_pair_request_construction_standard_shape():
    """A standard map builds one request per (StatVar, entity) pair."""
    captured: list[MaterialisationRequest] = []

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
        span=None,
    ):
        captured.append(request)
        entity = next(iter(request.role_bindings["observationAbout"]))
        return stub_preview_result(sv="Count_Person", entity=entity)

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND", "country/USA"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/IND", "country/USA"]),
            materialise_sdmx=sdmx_double(),
            window=TimeWindow(start_year=2015, end_year=2020),
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert observations is not None
    assert len(captured) == 2
    by_entity = {next(iter(r.role_bindings["observationAbout"])): r for r in captured}
    assert set(by_entity) == {"country/IND", "country/USA"}
    for entity, req in by_entity.items():
        assert req.stat_var == "dcid:Count_Person"
        assert req.role_bindings == {"observationAbout": [entity]}
        assert req.declared_roles == ("observationAbout",)
        assert req.facet_bindings == {}
        assert req.window == RequestedWindow(kind="range", start_year=2015, end_year=2020)


def test_per_pair_preview_reads_under_the_obs_preview_max_rows_ceiling():
    """Search's preview keeps `obs_preview_max_rows` per series and bounds its raw read by
    that many per planned series, as `limit` and as the mint's `ceiling`."""
    bounds: list[tuple[int, int, int | None]] = []

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
        span=None,
    ):
        bounds.append((limit, ceiling, series_limit))
        return stub_preview_result(sv="Count_Person", entity="country/IND")

    with patch.object(region_fetch, "materialise_preview", _spy):
        build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/IND"]),
            materialise_sdmx=sdmx_double(),
            window=TimeWindow(start_year=2015, end_year=2020),
            coverage=empty_coverage_snapshot(),
        )

    max_rows = config.settings().obs_preview_max_rows
    assert bounds == [(max_rows, max_rows, max_rows)]


def test_per_pair_request_construction_unwindowed_query_is_kind_all():
    captured: list[MaterialisationRequest] = []

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
        span=None,
    ):
        captured.append(request)
        return stub_preview_result(entity="country/IND")

    with patch.object(region_fetch, "materialise_preview", _spy):
        build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/IND"]),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert captured[0].window == RequestedWindow(kind="all")


def test_per_pair_request_construction_role_bearing_shape():
    """A role-bearing map builds one request for each role-tuple combination."""
    captured: list[MaterialisationRequest] = []

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
        span=None,
    ):
        captured.append(request)
        entity = next(iter(request.role_bindings["provider"]))
        return stub_preview_result(sv="Test_Flow_SV", role_key="provider", entity=entity)

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            ["Test_Flow_SV"],
            {"provider": ["country/NOR", "country/SWE"], "recipient": ["country/KEN"]},
            graph=FakeGraph(),
            materialise_graph=FakeGraphReader(),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert observations is not None
    assert len(captured) == 2  # 2 providers x 1 recipient
    pairs = {
        (dict(r.role_bindings)["provider"][0], dict(r.role_bindings)["recipient"][0])
        for r in captured
    }
    assert pairs == {("country/NOR", "country/KEN"), ("country/SWE", "country/KEN")}
    for r in captured:
        assert set(r.declared_roles) == {"provider", "recipient"}


def test_declared_roles_widen_to_the_family_signature_when_only_one_role_is_bound():
    """Issues #881/#627: keep `role_bindings` on the bound recipient while declaring
    the full provider/recipient signature required by the SDMX response. A one-role
    signature made the role-divergence check fail every pair."""
    two_role_arcs = {
        "observationProperties": {
            "nodes": [{"dcid": "provider"}, {"dcid": "recipient"}],
        }
    }
    captured: list[MaterialisationRequest] = []

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        rate_cache=None,
        deadline=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
        span=None,
    ):
        captured.append(request)
        entity = next(iter(request.role_bindings["recipient"]))
        return stub_preview_result(sv="Test_Flow_SV", role_key="recipient", entity=entity)

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            ["Test_Flow_SV"],
            {"recipient": ["country/KEN"]},
            graph=FakeGraph(),
            materialise_graph=FakeGraphReader(),
            materialise_sdmx=sdmx_double(),
            member_facts={"Test_Flow_SV": ArcsMemberFacts(arcs=two_role_arcs)},
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert observations is not None
    assert len(captured) == 1
    request = captured[0]
    assert request.role_bindings == {"recipient": ["country/KEN"]}
    assert request.declared_roles == ("provider", "recipient")


def test_declared_roles_falls_back_to_bound_roles_when_member_facts_has_no_entry():
    """Without member facts for an SV, `declared_roles` contains only the bound roles."""
    captured: list[MaterialisationRequest] = []

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        rate_cache=None,
        deadline=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
        span=None,
    ):
        captured.append(request)
        entity = next(iter(request.role_bindings["recipient"]))
        return stub_preview_result(sv="Test_Flow_SV", role_key="recipient", entity=entity)

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            ["Test_Flow_SV"],
            {"recipient": ["country/KEN"]},
            graph=FakeGraph(),
            materialise_graph=FakeGraphReader(),
            materialise_sdmx=sdmx_double(),
            member_facts=None,
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert observations is not None
    assert len(captured) == 1
    assert captured[0].declared_roles == ("recipient",)


def test_pair_count_at_ceiling_still_previews():
    """The pair-count ceiling is inclusive; every pair contributes at the limit."""
    entities = [f"country/E{i}" for i in range(observation_settings().obs_preview_max_pairs)]

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
        span=None,
    ):
        entity = next(iter(request.role_bindings["observationAbout"]))
        return stub_preview_result(sv="Count_Person", entity=entity)

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": entities},
            graph=FakeGraph(),
            materialise_graph=harness_reader(entities),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert observations is not None
    assert len(observations.series) == observation_settings().obs_preview_max_pairs


def test_pair_count_over_ceiling_degrades_whole_preview_without_fetching_any_pair():
    """More planned series than the ceiling omits the whole preview before fetching.

    The plan holds only series with data, so this fixture gives every entity data to reach the
    ceiling."""
    entities = [f"country/E{i}" for i in range(observation_settings().obs_preview_max_pairs + 1)]

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
        span=None,
    ):
        raise AssertionError(
            "materialise_preview must not be called once the pair-count ceiling is exceeded"
        )

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": entities},
            graph=FakeGraph(),
            materialise_graph=harness_reader(entities),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_PREVIEW_TOO_LARGE
    assert warn is not None
    assert warn.code == OBSERVATIONS_PREVIEW_TOO_LARGE


def test_pair_count_planned_over_ceiling_but_pairs_with_data_under_ceiling_previews():
    """The ceiling applies to the planned series with data, even when the full product exceeds
    it."""
    entities = [f"country/E{i}" for i in range(observation_settings().obs_preview_max_pairs + 10)]
    with_data = entities[:100]
    captured: list[MaterialisationRequest] = []

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
        span=None,
    ):
        entity = next(iter(request.role_bindings["observationAbout"]))
        captured.append(request)
        return stub_preview_result(sv="Count_Person", entity=entity)

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": entities},
            graph=FakeGraph(),
            materialise_graph=harness_reader(with_data),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert observations is not None
    fetched_entities = {next(iter(r.role_bindings["observationAbout"])) for r in captured}
    assert fetched_entities == set(with_data)


def test_rate_cache_forwarded_to_materialise_preview():
    """Each pair receives the same `rate_cache` instance."""
    sentinel_cache = RateTableCache()
    captured: list[object] = []

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
        span=None,
    ):
        captured.append(rate_cache)
        return stub_preview_result(sv="Count_Person", entity="country/IND")

    with patch.object(region_fetch, "materialise_preview", _spy):
        build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/IND"]),
            materialise_sdmx=sdmx_double(),
            rate_cache=sentinel_cache,
            coverage=empty_coverage_snapshot(),
        )

    assert captured == [sentinel_cache]


def test_rate_cache_omitted_serves_existing_fixture_unchanged():
    """Omitting `rate_cache` forwards `None` to each preview call."""
    captured: list[object] = []

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
        span=None,
    ):
        captured.append(rate_cache)
        return stub_preview_result(sv="Count_Person", entity="country/IND")

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/IND"]),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert observations is not None
    assert captured == [None]


# ---------------------------------------------------------------------------
# A composite component's read in the years it serves: the plan elected under 1973 to
# 2024, the composite serving a span inside it (`fetch._span_read`).
# ---------------------------------------------------------------------------

_KEN = (("observationAbout", "country/KEN"),)


def _points(years: range | Sequence[int], value: float) -> tuple[tuple[str, float], ...]:
    return tuple((str(year), value) for year in years)


_SPAN_FACETS = {
    "one-facet": [ObsFacetFixture("1", _points(range(1973, 2025), 1.0), provenance_id="p")],
    # Recorded to 2015 and in 2019, estimated from 2010 on: the estimates serve the years
    # after the last recorded one.
    "continued": [
        ObsFacetFixture("1", _points([*range(1973, 2016), 2019], 1.0), provenance_id="p"),
        ObsFacetFixture(
            "2", _points(range(2010, 2025), 2.0), provenance_id="p", measurement_method="Estimated"
        ),
    ],
    # Two facets under one method contest 2011 to 2018. The first holds 46 rows in the
    # window and 8 in the span, the second 14 in both.
    "contested": [
        ObsFacetFixture("1", _points(range(1973, 2019), 1.0), provenance_id="p"),
        ObsFacetFixture("2", _points(range(2011, 2025), 2.0), provenance_id="p"),
    ],
    # Two facets under one method contest 2011 to 2018, the first holding more rows in the
    # window and in the span alike.
    "outranked": [
        ObsFacetFixture("1", _points(range(2011, 2025), 1.0), provenance_id="p"),
        ObsFacetFixture("2", _points(range(2011, 2019), 2.0), provenance_id="p"),
    ],
}


def _span_preview(
    case: str,
    served: frozenset[int],
    *,
    candidates: dict[str, MaterialiseFacet] | None = None,
) -> tuple[list[list[tuple[str, float | None]]], list[list[str] | None]]:
    """Kenya's series points read in `served` and each SDMX request's periods (None for
    every period), for `case`'s facets under a plan elected under 1973 to 2024.
    `candidates` replaces the plan's candidate of a facet id, the coverage a read ranks
    contested facets by, leaving the rows SDMX holds."""
    graph, sdmx, requests = build_materialise_double(
        {("Count_Person", _KEN): _SPAN_FACETS[case]}, honour_periods=True
    )
    held = graph.variable_facets_batch(
        stat_var="Count_Person", entities=["country/KEN"], needs_dates=True
    )["country/KEN"]
    plan = _plan([(candidates or {}).get(f.facet_id or "", f) for f in held])
    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/KEN"]},
        plan=plan,
        graph=FakeGraph(),
        materialise_graph=graph,
        materialise_sdmx=sdmx,
        coverage=empty_coverage_snapshot(),
        years=served,
    )
    assert warn is None
    data = [r for r in requests if "c[variableMeasured]" in r.url.params]
    periods = [
        r.url.params["c[TIME_PERIOD]"].split(",") if "c[TIME_PERIOD]" in r.url.params else None
        for r in data
    ]
    return [[(p.date, p.value) for p in s.points] for s in observations.series], periods


def _plan(candidates: Sequence[MaterialiseFacet]) -> PairPlan:
    return plan_pairs(
        [PlanSeries(stat_var="Count_Person", role_tuple=_KEN, candidates=tuple(candidates))],
        members=["Count_Person"],
        years=TimeWindow(start_year=1973, end_year=2024),
        declared=NO_DECLARED_POLICIES,
    )


def _whole_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every read asks SDMX for the whole window, ranks contested facets by its rows and
    keeps the span afterwards."""
    monkeypatch.setattr(
        region_fetch,
        "_span_read",
        lambda _dispatch, *, served, selected: SpanRead(served=plan_window(served)),
    )


_SERVED = frozenset(range(2011, 2025))


@pytest.mark.parametrize("case", ["one-facet", "continued", "contested"])
def test_a_component_read_in_its_served_years_asks_sdmx_for_none_before_them(case):
    """Elected under 1973 to 2024 and serving 2011 to 2024, the read asks SDMX for those
    years alone: no row before the span changes one inside it."""
    _points_read, periods = _span_preview(case, _SERVED)

    assert periods == [[str(year) for year in range(2011, 2025)]]


@pytest.mark.parametrize(
    ("case", "served"),
    [
        pytest.param("one-facet", _SERVED, id="one-facet"),
        # The recorded 2019 cuts the estimates up to it, so 2016 to 2018 serve none.
        pytest.param("continued", frozenset(range(2011, 2019)), id="continued"),
        # The first facet, the more rows in the window, keeps 2011 to 2018.
        pytest.param("contested", _SERVED, id="contested"),
    ],
)
def test_a_component_read_in_its_served_years_serves_the_whole_window_read_s_points(
    monkeypatch, case, served
):
    """The span read serves the points a read of the whole window keeps in the span: a
    continued series keeps its estimates past the last recorded year in the window, and
    contested facets rank over the window."""
    bounded, _periods = _span_preview(case, served)
    _whole_window(monkeypatch)
    whole, whole_periods = _span_preview(case, served)

    assert whole_periods == [None]
    assert bounded == whole


def test_a_component_read_in_its_served_years_ranks_contested_facets_by_coverage(monkeypatch):
    """Coverage holds the second facet over the whole window, 52 years to the first's 14,
    while SDMX holds its rows in 2011 to 2018 alone. The span read ranks by coverage and
    serves the second facet's 2011 to 2018; a read of the whole window ranks by its rows,
    as a read of the span would, and serves the first's."""
    second = replace(
        _facet("outranked", "2"),
        dates=tuple(str(year) for year in range(1973, 2025)),
        values=(2.0,) * 52,
        obs_count=52,
        earliest_date="1973",
    )

    (bounded,), _periods = _span_preview("outranked", _SERVED, candidates={"2": second})
    _whole_window(monkeypatch)
    (whole,), _periods = _span_preview("outranked", _SERVED, candidates={"2": second})

    assert bounded == [*_points(range(2011, 2019), 2.0), *_points(range(2019, 2025), 1.0)]
    assert whole == list(_points(range(2011, 2025), 1.0))


def _facet(case: str, facet_id: str) -> MaterialiseFacet:
    """The candidate a dated graph read gives `case`'s facet `facet_id`."""
    graph, _sdmx, _requests = build_materialise_double({("Count_Person", _KEN): _SPAN_FACETS[case]})
    (facet,) = [
        f
        for f in graph.variable_facets_batch(
            stat_var="Count_Person", entities=["country/KEN"], needs_dates=True
        )["country/KEN"]
        if f.facet_id == facet_id
    ]
    return facet
