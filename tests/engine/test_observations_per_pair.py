"""Tests per-pair observation request construction, preview limits, and rate-cache forwarding."""

from __future__ import annotations

from unittest.mock import patch

import search.engine.region.observations.fetch as region_fetch
from materialise import (
    MaterialisationRequest,
)
from materialise.rate_tables import RateTableCache
from materialise.window import RequestedWindow
from search.engine import config
from search.engine.app import create_app, search_surface
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.models import (
    TimeWindow,
)
from search.observations import OBSERVATIONS_PREVIEW_TOO_LARGE, observation_settings
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
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
