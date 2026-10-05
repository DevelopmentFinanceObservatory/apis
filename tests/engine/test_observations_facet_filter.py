"""Tests that the preview's `facet_filter` narrows each pair's candidates before
materialise elects, using single-facet and multi-axis fixtures.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import search.engine.region.observations.fetch as region_fetch
from materialise.graph import Facet as MaterialiseFacet
from search.engine.region.codes import (
    OBSERVATIONS_NO_MATCHING_SOURCE,
    OBSERVATIONS_PROBE_FAILED,
)
from search.engine.region.facets import (
    _facet_source_dcid,
)
from search.engine.region.resubmit_request import _facet_filter_from_bindings
from search.engine.region.result import FacetAxisFilter
from search.models import (
    IdentityFacetValueRequest,
    IdentityUnitRequest,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import (
    build_observations_planned,
    harness_reader,
    materialise_double_from_obs,
    sdmx_double,
    stub_preview_result,
)
from tests.engine._observations_doubles import (
    _MULTI_AXIS_SV,
    _TWO_ENTITY_SV,
    _facet_axis_materialise_double,
)
from tests.fixtures import FakeGraph
from tests.materialise.conftest import FakeGraphReader

# ---------------------------------------------------------------------------
# region/observations/fetch.py: _build_observations -- facet_filter (generalises the old
# source_filter kwarg to every facet axis; the source-switch carve-out
# region/resubmit_request.py's _rebind_source_slot threads through here as the
# "provenance" entry).
# ---------------------------------------------------------------------------


def _candidate(provenance_id: str) -> MaterialiseFacet:
    return MaterialiseFacet(
        earliest_date="2020", latest_date="2020", obs_count=1, provenance_id=provenance_id
    )


def _provenance_filter(*allowed: str) -> dict:
    """Build a provenance filter admitting each of `allowed`."""
    allowed_set = frozenset(allowed)
    return {
        "provenance": FacetAxisFilter(
            predicate=lambda f: _facet_source_dcid(f) in allowed_set,
        )
    }


def test_build_observations_facet_filter_single_ref_steers_dispatch_to_the_bound_source():
    """The 3-row excluded facet would win without a filter. Binding provenance
    to the 2-row allowed facet steers dispatch to it.
    """
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/IND": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2021",
                    "obsCount": 2,
                    "provenanceId": "dc/base/allowed",
                    "observations": [
                        {"date": "2020", "value": 1.0},
                        {"date": "2021", "value": 2.0},
                    ],
                },
                {
                    # The unconstrained dispatch winner (3 > 2 obsCount), but
                    # a different source than the one bound below.
                    "earliestDate": "2022",
                    "latestDate": "2024",
                    "obsCount": 3,
                    "provenanceId": "dc/base/excluded",
                    "observations": [
                        {"date": "2022", "value": 3.0},
                        {"date": "2023", "value": 4.0},
                        {"date": "2024", "value": 5.0},
                    ],
                },
            ],
        }
    )
    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        facet_filter=_provenance_filter("dc/base/allowed"),
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    assert observations.series[0].facet["provenance"].ref == "dc/base/allowed"
    assert observations.total_obs == 2


def test_build_observations_facet_filter_no_matching_facet_returns_warning():
    """When every facet is filtered out, the preview reports no matching source."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/IND": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": "dc/base/other",
                    "observations": [{"date": "2020", "value": 1.0}],
                }
            ],
        }
    )
    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        facet_filter=_provenance_filter("dc/base/requested"),
        coverage=empty_coverage_snapshot(),
    )
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == "observations_no_matching_source"
    assert warn is not None
    assert warn.code == "observations_no_matching_source"


def test_build_observations_facet_filter_none_is_byte_identical_to_omitting_it():
    """Explicit `None` matches the default call."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/IND": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": "dc/base/Src",
                    "observations": [{"date": "2020", "value": 1.0}],
                }
            ],
        }
    )
    without_kwarg = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    with_none = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        facet_filter=None,
        coverage=empty_coverage_snapshot(),
    )
    assert without_kwarg == with_none


# ---------------------------------------------------------------------------
# _build_observations' facet_filter against the multi-axis fixture group
# (tests.engine._observations_doubles): per-axis single-ref dispatch
# steering (measurementMethod, unit set, unit request-form).
# ---------------------------------------------------------------------------


def test_facet_filter_measurement_method_single_ref_steers_dispatch_to_the_bound_method():
    """The 18-candidate fixture's default is GrossDisbursement/USD. Binding
    `GrantEquivalent` selects its richest matching candidate.
    """
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double()
    sv = [_MULTI_AXIS_SV]
    entity = ["country/USA"]

    unconstrained, warn = build_observations_planned(
        sv,
        {"observationAbout": entity},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert unconstrained.series[0].facet["measurementMethod"].ref == "GrossDisbursement"

    constrained, warn = build_observations_planned(
        sv,
        {"observationAbout": entity},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        facet_filter={
            "measurementMethod": FacetAxisFilter(
                predicate=lambda f: f.measurement_method == "GrantEquivalent",
            )
        },
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert constrained is not None
    assert constrained.series[0].facet["measurementMethod"].ref == "GrantEquivalent"


def test_facet_filter_single_ref_binding_absent_everywhere_still_refuses_no_matching_source():
    """A bound method absent from the candidate pool returns
    `OBSERVATIONS_NO_MATCHING_SOURCE`, not a probe-failure warning.
    """
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double()
    observations, warn = build_observations_planned(
        [_MULTI_AXIS_SV],
        {"observationAbout": ["country/USA"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        facet_filter={
            "measurementMethod": FacetAxisFilter(
                predicate=lambda f: f.measurement_method == "NoSuchMethod",
            )
        },
        coverage=empty_coverage_snapshot(),
    )
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_NO_MATCHING_SOURCE
    assert warn is not None
    assert warn.code == OBSERVATIONS_NO_MATCHING_SOURCE


def test_facet_filter_set_binding_drops_entities_that_fail_it():
    """USA has the sole EUR facet; GBR has the sole USD facet. An EUR filter
    keeps the USA series, drops GBR, and reports one rejection out of two.
    """
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double()
    observations, warn = build_observations_planned(
        [_TWO_ENTITY_SV],
        {"observationAbout": ["country/USA", "country/GBR"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        facet_filter={"unit": FacetAxisFilter(predicate=lambda f: f.unit in frozenset({"EUR"}))},
        coverage=empty_coverage_snapshot(),
    )
    assert warn is not None
    assert warn.code == OBSERVATIONS_NO_MATCHING_SOURCE
    assert "1" in warn.message and "2" in warn.message
    assert observations is not None
    assert [s.entities[0].ref for s in observations.series] == ["country/USA"]
    assert observations.series[0].facet["unit"].ref == "EUR"


def test_facet_filter_request_form_unit_matching_the_unconstrained_pick_carries_ref():
    """A USD/current request has no single facet ref to steer dispatch with.
    The unconstrained USD winner satisfies its post-fetch predicate.
    """
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double()
    graph = FakeGraph(currency_units={"USD": "USD", "EUR": "EUR", "GBP": "GBP"})
    request = IdentityUnitRequest(currency="USD", prices="current")
    filter_ = _facet_filter_from_bindings(
        graph, [], {"unit": IdentityFacetValueRequest(request=request)}, deadline=None
    )
    observations, warn = build_observations_planned(
        [_MULTI_AXIS_SV],
        {"observationAbout": ["country/USA"]},
        graph=graph,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        facet_filter=filter_,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    assert observations.series[0].facet["unit"].ref == "USD"


def test_facet_filter_request_form_unit_no_table_entry_yields_predicate_rejecting_everything():
    """An unmapped JPY request rejects every candidate facet."""
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double()
    graph = FakeGraph(currency_units={"USD": "USD", "EUR": "EUR", "GBP": "GBP"})
    request = IdentityUnitRequest(currency="JPY", prices="current")
    filter_ = _facet_filter_from_bindings(
        graph, [], {"unit": IdentityFacetValueRequest(request=request)}, deadline=None
    )
    observations, warn = build_observations_planned(
        [_MULTI_AXIS_SV],
        {"observationAbout": ["country/USA"]},
        graph=graph,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        facet_filter=filter_,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is not None
    assert observations.series == []
    assert observations.omitted is not None


# ---------------------------------------------------------------------------
# _build_observations' facet_filter, isolated from _build_observations' own
# pre-fetch steering: a monkeypatched materialise_preview spy proves
# _preview_satisfies_filter's belt-and-braces post-hoc re-check on its own.
# ---------------------------------------------------------------------------


def test_facet_filter_simple_single_ref_predicate_admits_a_satisfying_pair():
    """A mocked preview with the requested provenance passes the post-fetch
    predicate check.
    """

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
        return stub_preview_result(provenance="prov1")

    filter_ = {"provenance": FacetAxisFilter(predicate=lambda f: f.provenance_id == "prov1")}
    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/IND"]),
            materialise_sdmx=sdmx_double(),
            facet_filter=filter_,
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert observations is not None
    assert observations.series[0].facet["provenance"].ref == "prov1"


def _sourced(facet_id: str, provenance_id: str, obs_count: int) -> MaterialiseFacet:
    return MaterialiseFacet(
        facet_id=facet_id,
        earliest_date="2020",
        latest_date="2020",
        obs_count=obs_count,
        provenance_id=provenance_id,
    )


def _record_requests(requests: list) -> object:
    """A `materialise_preview` spy recording each request with its precomputed candidates;
    every result dispatches `f1`."""

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
        requests.append((request, precomputed))
        return stub_preview_result(provenance="prov2")

    return _spy


def test_the_filtered_plan_reaches_materialise_as_candidates_and_its_group():
    """A filter narrows the plan once: materialise receives the admitted candidates and a
    `candidate_filter` admitting exactly the planned group, and binds no axis, so it elects
    nothing the plan did not."""
    reader = FakeGraphReader(
        facets={
            ("Count_Person", "country/IND"): [
                _sourced("f0", "prov1", 2),
                _sourced("f1", "prov2", 5),
                _sourced("f9", "prov3", 9),
            ]
        }
    )
    requests: list = []
    with patch.object(region_fetch, "materialise_preview", _record_requests(requests)):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(),
            materialise_graph=reader,
            materialise_sdmx=sdmx_double(),
            facet_filter=_provenance_filter("prov1", "prov2"),
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert observations.series[0].facet["provenance"].ref == "prov2"
    ((request, precomputed),) = requests
    assert request.facet_bindings == {}
    assert [f.facet_id for f in precomputed] == ["f0", "f1"]
    assert request.candidate_filter is not None
    assert [request.candidate_filter(f) for f in precomputed] == [False, True]


def test_a_dispatched_facet_the_filter_excludes_is_never_served(caplog):
    """The filter admits `prov1` alone, so the plan's group is `f0`. Materialise
    dispatching `f1` (`prov2`, which the filter excludes) is a plan mismatch: the pair is
    logged, counted as failed, and its `prov2` points are not served."""
    reader = FakeGraphReader(
        facets={
            ("Count_Person", "country/IND"): [
                _sourced("f0", "prov1", 2),
                _sourced("f1", "prov2", 5),
            ]
        }
    )
    requests: list = []
    with (
        patch.object(region_fetch, "materialise_preview", _record_requests(requests)),
        caplog.at_level(logging.WARNING, logger=region_fetch.__name__),
    ):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(),
            materialise_graph=reader,
            materialise_sdmx=sdmx_double(),
            facet_filter=_provenance_filter("prov1"),
            coverage=empty_coverage_snapshot(),
        )

    ((_request, precomputed),) = requests
    assert [f.facet_id for f in precomputed] == ["f0"]
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_PROBE_FAILED
    assert observations.omitted.hard_failed == 1
    assert warn is not None
    assert warn.code == OBSERVATIONS_PROBE_FAILED
    assert "outside the preview's read plan" in warn.message
    assert any("outside its planned group" in r.getMessage() for r in caplog.records)
