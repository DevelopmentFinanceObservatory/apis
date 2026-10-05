"""Tests per-pair observation failure isolation, warnings, and request-error propagation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import search.engine.region.observations.fetch as region_fetch
from materialise import (
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    MaterialisationUsageError,
    UpstreamResponseError,
)
from materialise.graph import Facet as MaterialiseFacet
from search.engine.region.codes import (
    OBSERVATIONS_NO_MATCHING_SOURCE,
    OBSERVATIONS_PROBE_FAILED,
)
from search.engine.region.result import FacetAxisFilter
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import (
    ObsFacetFixture,
    build_materialise_double,
    build_observations_planned,
    harness_reader,
    pair_plan,
    sdmx_double,
    stub_preview_result,
)
from tests.fixtures import FakeGraph
from tests.materialise.conftest import FakeGraphReader

# ---------------------------------------------------------------------------
# Upstream failures degrade one pair; request-building errors propagate.
# ---------------------------------------------------------------------------


def _make_raising_spy(exc_for_entity: dict[str, Exception]):
    """Raises the mapped exception for selected entities and returns a stub for the rest.

    Reusing each exception instance keeps all-fail classification deterministic."""

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
        if entity in exc_for_entity:
            raise exc_for_entity[entity]
        return stub_preview_result(sv="Count_Person", entity=entity)

    return _spy


def test_facet_not_available_without_filter_is_silent_and_isolates_other_pairs():
    """An unfiltered no-data pair is silent while another pair contributes."""
    spy = _make_raising_spy({"country/BAD": FacetNotAvailableError()})
    with patch.object(region_fetch, "materialise_preview", spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/BAD", "country/GOOD"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/BAD", "country/GOOD"]),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert observations is not None
    assert {s.entities[0].ref for s in observations.series} == {"country/GOOD"}


def test_facet_not_available_with_active_filter_and_every_pair_empty_warns_no_matching_source():
    """A facet_filter active, and EVERY pair ends up contributing nothing:
    surfaces OBSERVATIONS_NO_MATCHING_SOURCE."""
    spy = _make_raising_spy({"country/IND": FacetNotAvailableError()})
    filter_ = {"provenance": FacetAxisFilter(predicate=lambda f: True)}
    with patch.object(region_fetch, "materialise_preview", spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/IND"]),
            materialise_sdmx=sdmx_double(),
            facet_filter=filter_,
            coverage=empty_coverage_snapshot(),
        )

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_NO_MATCHING_SOURCE
    assert warn is not None
    assert warn.code == OBSERVATIONS_NO_MATCHING_SOURCE


def test_default_facet_undefined_degrades_pair_and_warns_but_other_pairs_still_contribute():
    """A missing default facet warns for this pair without blocking other pairs."""
    spy = _make_raising_spy({"country/BAD": DefaultFacetUndefinedError()})
    with patch.object(region_fetch, "materialise_preview", spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/BAD", "country/GOOD"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/BAD", "country/GOOD"]),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert observations is not None
    assert {s.entities[0].ref for s in observations.series} == {"country/GOOD"}
    assert warn is not None
    assert warn.code == OBSERVATIONS_PROBE_FAILED
    assert warn.message == "1 of 2 requested series are included; 1 could not be served."


def test_upstream_response_error_degrades_pair_and_warns_but_other_pairs_still_contribute():
    """Role divergence degrades its pair with a warning and leaves other pairs intact."""
    spy = _make_raising_spy({"country/BAD": UpstreamResponseError("role divergence")})
    with patch.object(region_fetch, "materialise_preview", spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/BAD", "country/GOOD"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/BAD", "country/GOOD"]),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert observations is not None
    assert {s.entities[0].ref for s in observations.series} == {"country/GOOD"}
    assert warn is not None
    assert warn.code == OBSERVATIONS_PROBE_FAILED
    assert warn.message == "1 of 2 requested series are included; 1 could not be served."


def _one_failing_policy_read(error: Exception):
    """Plan `Count_Person` and `Count_Bad` at `country/GOOD` from one reader, and preview them
    through a reader whose facet-policy read fails for `Count_Bad` alone with `error`."""
    pairs = {
        (sv, (("observationAbout", "country/GOOD"),)): [
            ObsFacetFixture(facet_id="facet1", points=(("2020", 100.0),), provenance_id="prov1")
        ]
        for sv in ("Count_Person", "Count_Bad")
    }
    materialise_graph, materialise_sdmx, _requests = build_materialise_double(pairs)

    class _RaisingGraphReader(FakeGraphReader):
        def default_facet_policy(self, *, stat_var, axis):
            if stat_var.endswith("Count_Bad"):
                raise error
            return super().default_facet_policy(stat_var=stat_var, axis=axis)

    entities = {"observationAbout": ["country/GOOD"]}
    return build_observations_planned(
        ["Count_Person", "Count_Bad"],
        entities,
        graph=FakeGraph(),
        materialise_graph=_RaisingGraphReader(facets=materialise_graph._facets),
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
        plan=pair_plan(["Count_Person", "Count_Bad"], entities, reader=materialise_graph),
    )


def test_graph_read_error_degrades_pair_and_warns_but_other_pairs_still_contribute():
    """A graph transport failure warns for its pair while another pair contributes. The
    preview's graph read is materialise's facet-policy read; the candidates come from the
    plan."""
    from materialise.graph import GraphReadError

    observations, warn = _one_failing_policy_read(GraphReadError("simulated transport error"))

    assert observations is not None
    assert {s.stat_var.ref for s in observations.series} == {"Count_Person"}
    assert warn is not None
    assert warn.code == OBSERVATIONS_PROBE_FAILED
    assert warn.message == "1 of 2 requested series are included; 1 could not be served."


def test_graph_infra_error_from_the_shared_search_client_degrades_pair_and_warns():
    """`GraphReadFailure` subclasses `GraphReadError`, so the per-pair handler must isolate it.

    This pins the shared Search graph client's transport-error translation."""
    from search.engine.errors import GraphReadFailure

    observations, warn = _one_failing_policy_read(
        GraphReadFailure("simulated transport error", upstream_status=503)
    )

    assert observations is not None
    assert {s.stat_var.ref for s in observations.series} == {"Count_Person"}
    assert warn is not None
    assert warn.code == OBSERVATIONS_PROBE_FAILED
    assert warn.message == "1 of 2 requested series are included; 1 could not be served."


def test_sdmx_transport_error_degrades_pair_and_warns_but_other_pairs_still_contribute():
    """A mocked SDMX connection failure warns for its pair while another pair contributes.

    The test uses a real `SdmxClient` with `httpx.MockTransport`."""
    import httpx

    from dc_sdmx import SdmxClient as MaterialiseSdmxClient

    graph = FakeGraphReader(
        facets={
            ("Count_Person", "country/BAD"): [
                MaterialiseFacet(
                    facet_id="facet1",
                    provenance_id="prov1",
                    measurement_method=None,
                    unit=None,
                    obs_count=1,
                    earliest_date="2020",
                    latest_date="2020",
                )
            ],
            ("Count_Person", "country/GOOD"): [
                MaterialiseFacet(
                    facet_id="facet1",
                    provenance_id="prov1",
                    measurement_method=None,
                    unit=None,
                    obs_count=1,
                    earliest_date="2020",
                    latest_date="2020",
                )
            ],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("c[observationAbout]") == "country/BAD":
            raise httpx.ConnectError("connection refused", request=request)
        csv_body = (
            "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,"
            "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
            "scalingFactor,facetId\r\n"
            "dataflow,DC:DF_OBS(1.0.0),I,dcid:Count_Person,country/GOOD,,,"
            "NotApplicable,prov1,2020,100,,facet1\r\n"
        )
        return httpx.Response(
            200,
            text=csv_body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    sdmx = MaterialiseSdmxClient(
        base_url="https://example.org/core/api", transport=httpx.MockTransport(handler)
    )

    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/BAD", "country/GOOD"]},
        graph=FakeGraph(),
        materialise_graph=graph,
        materialise_sdmx=sdmx,
        coverage=empty_coverage_snapshot(),
    )

    assert observations is not None
    assert {s.entities[0].ref for s in observations.series} == {"country/GOOD"}
    assert warn is not None
    assert warn.code == OBSERVATIONS_PROBE_FAILED


# ---------------------------------------------------------------------------
# All-fail messages name each failed read in a stable order.
# ---------------------------------------------------------------------------

# The one, stable rendering of each read name in the all-fail message --
# asserted by content, so a message that names the wrong read (or drops the
# count) fails even though the code and counts stay right.
_ALL_SDMX_FAILURE = (
    "all 1 requested series could not be served; the sdmx read failed, so "
    "Spec.resolved.observations was not populated."
)
_ALL_GRAPH_FAILURE = (
    "all 1 requested series could not be served; the graph read failed, so "
    "Spec.resolved.observations was not populated."
)
_ALL_UPSTREAM_FAILURE = (
    "all 1 requested series could not be served; the upstream response read failed, so "
    "Spec.resolved.observations was not populated."
)


def _assert_all_fail_names(exc: Exception, expected: str) -> None:
    """Checks that the omission and Warning name the read represented by `exc`."""
    spy = _make_raising_spy({"country/BAD": exc})
    with patch.object(region_fetch, "materialise_preview", spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/BAD"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/BAD"]),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_PROBE_FAILED
    assert observations.omitted.message == expected
    assert observations.omitted.dispatched == 1
    assert observations.omitted.hard_failed == 1
    assert warn is not None
    assert warn.code == OBSERVATIONS_PROBE_FAILED
    assert warn.message == expected


def test_all_pairs_sdmx_client_error_names_the_sdmx_read():
    """When every pair fails with `SdmxClientError`, the message names the SDMX read."""
    from dc_sdmx import SdmxClientError

    _assert_all_fail_names(SdmxClientError("simulated SDMX failure"), _ALL_SDMX_FAILURE)


def test_all_pairs_graph_read_error_names_the_graph_read():
    """When every pair fails with `GraphReadError`, the message names the graph read."""
    from materialise.graph import GraphReadError

    _assert_all_fail_names(GraphReadError("simulated transport error"), _ALL_GRAPH_FAILURE)


def test_all_pairs_upstream_response_error_names_the_upstream_response_read():
    """Role divergence occurs after graph and SDMX reads, so the message names the upstream read."""
    _assert_all_fail_names(UpstreamResponseError("role divergence"), _ALL_UPSTREAM_FAILURE)


def test_all_pairs_mixed_failures_name_every_failing_read():
    """Mixed graph and SDMX failures appear in the fixed graph-then-SDMX order."""
    from dc_sdmx import SdmxClientError
    from materialise.graph import GraphReadError

    spy = _make_raising_spy(
        {
            "country/BAD": GraphReadError("simulated transport error"),
            "country/WORSE": SdmxClientError("simulated SDMX failure"),
        }
    )
    with patch.object(region_fetch, "materialise_preview", spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/BAD", "country/WORSE"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/BAD", "country/WORSE"]),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_PROBE_FAILED
    assert observations.omitted.message == (
        "all 2 requested series could not be served; the graph and sdmx reads failed, so "
        "Spec.resolved.observations was not populated."
    )
    assert observations.omitted.hard_failed == 2
    assert warn is not None
    assert warn.message == observations.omitted.message


def test_all_pairs_no_dispatchable_facet_names_no_read():
    """Pairs without dispatchable facets are named without inventing a failed read."""
    spy = _make_raising_spy({"country/BAD": DefaultFacetUndefinedError()})
    with patch.object(region_fetch, "materialise_preview", spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/BAD"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/BAD"]),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_PROBE_FAILED
    assert observations.omitted.message == (
        "all 1 requested series could not be served; no (stat_var, entity) pair had "
        "a dispatchable facet, so Spec.resolved.observations was not populated."
    )
    assert observations.omitted.hard_failed == 1
    assert "graph" not in observations.omitted.message
    assert "sdmx" not in observations.omitted.message
    assert warn is not None
    assert warn.message == observations.omitted.message


def test_materialisation_usage_error_propagates_uncaught():
    """Request-building errors propagate instead of being classified as upstream failures."""

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
        raise MaterialisationUsageError("malformed request")

    with (
        patch.object(region_fetch, "materialise_preview", _spy),
        pytest.raises(MaterialisationUsageError),
    ):
        build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/IND"]),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )


def test_real_request_building_never_triggers_materialisation_usage_error():
    """A real per-pair fetch completes with `limit >= 1` using mocked SDMX and a `FakeGraphReader`."""
    materialise_graph, materialise_sdmx, _requests = build_materialise_double(
        {
            ("Count_Person", (("observationAbout", "country/IND"),)): [
                ObsFacetFixture(
                    facet_id="facet1",
                    points=(("2020", 100.0),),
                    provenance_id="prov1",
                    measurement_method="Test",
                    unit="USD",
                )
            ]
        }
    )

    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )

    assert warn is None
    assert observations is not None
    assert observations.total_obs == 1
    assert observations.series[0].points[0].value == 100.0
