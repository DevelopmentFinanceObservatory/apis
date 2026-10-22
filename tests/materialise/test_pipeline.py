"""Tests `materialise_spec` success, empty results, ceiling refusals, and coverage warnings."""

from __future__ import annotations

import io
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pyarrow.parquet as pq
import pytest

from dc_sdmx import SdmxClient, SdmxResponseError
from dfo_platform.problem import Alternative
from materialise import (
    EnrichmentLookup,
    MaterialisationCancelled,
    MaterialisationRequest,
    RowProgress,
    materialise_spec,
)
from materialise import finish as finish_module
from materialise.content_version import mint_version
from materialise.errors import (
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    MaterialisationError,
    MaterialisationUsageError,
    OverCeilingError,
)
from materialise.facets import (
    _MAX_ALTERNATIVES_PER_ISSUE,
    FacetBinding,
    UnitRequest,
    declared_default_policies,
    select_facet,
)
from materialise.graph import Facet
from materialise.group_facets import _as_issue
from materialise.place_scope import MAX_EVERY_PLACE, DefaultPlaceScope
from materialise.window import AppliedWindow, Coverage, RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._scope_availability import ScopeAvailability
from .conftest import FakeGraphReader, only_chunk

_BASE_URL = "https://datacommons.one.org/core/api"
_STAT_VAR = "dcid:MalariaGrantDisbursements"
# The graph wire keys on the bare dcid; `FakeGraphReader` fixtures below
# use that bare form to match what `generation.py` sends.
_BARE_STAT_VAR = "MalariaGrantDisbursements"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=2,
    earliest_date="2015",
    latest_date="2016",
)

_BOUND_FACETS = {
    "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
    "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
    "unit": FacetBinding(state="bound", ref="USD"),
}


def _csv_row(*, recipient: str, year: str, value: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{recipient},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{year},{value},,facet1\r\n"
    )


def _sdmx_client(
    csv_body: str, *, availability: ScopeAvailability | None = None
) -> tuple[SdmxClient, list[httpx.Request]]:
    """Serves `csv_body` to every data read, and the default place scope's availability reads
    from `availability`. Records the data reads."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if availability is not None and (response := availability.respond(request)) is not None:
            return response
        requests.append(request)
        return httpx.Response(
            200,
            text=csv_body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)), requests


def _coverage_for(facets: list[Facet], *, entities: list[str], year: int = 2015) -> CoverageIndex:
    """Builds coverage for each facet and recipient; the request binds only recipient."""
    role_names = {_BARE_STAT_VAR: RoleNames(donor="donor", recipient="recipient")}
    facet_metadata = {
        facet.facet_id: FacetMetadata(
            provenance_id=facet.provenance_id,
            measurement_method=facet.measurement_method,
            unit=facet.unit,
        )
        for facet in facets
        if facet.facet_id is not None
    }
    rows = [
        (_BARE_STAT_VAR, facet.facet_id, "country/DONOR", entity, [year])
        for facet in facets
        for entity in entities
        if facet.facet_id is not None
    ]
    return build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)


def _request(
    *,
    role_bindings: dict[str, list[str]],
    window: RequestedWindow,
) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings=role_bindings,
        declared_roles=("recipient",),
        facet_bindings=_BOUND_FACETS,
        window=window,
    )


def test_happy_path_bound_facets_all_three_axes_non_empty_result():
    csv_body = (
        _CSV_HEADER
        + _csv_row(recipient="country/TGO", year="2015", value="1000")
        + _csv_row(recipient="country/TGO", year="2016", value="2000")
    )
    sdmx, requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )

    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    # The one dispatched fetch: candidates come from `coverage`, an
    # in-memory lookup, so no SDMX probe call precedes it.
    assert len(requests) == 1
    assert len(only_chunk(result).rows) == 2
    assert result.coverage.rows == 2
    assert result.warnings == ()
    assert result.trace.applied_window == {
        "kind": "all",
        "start_year": 2015,
        "end_year": 2016,
        "origin": "default",
    }
    assert {entry["axis"] for entry in result.trace.facets} == {
        "provenance",
        "measurementMethod",
        "unit",
    }
    assert result.content_version.startswith("cv_")
    assert isinstance(result.output_profile_seconds, float)
    assert result.output_profile_seconds >= 0.0


def test_progress_counter_ends_equal_to_the_frame_row_count():
    """Progress counts surviving rows on the single-member path."""
    csv_body = (
        _CSV_HEADER
        + _csv_row(recipient="country/TGO", year="2015", value="1000")
        + _csv_row(recipient="country/TGO", year="2016", value="2000")
    )
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )
    progress = RowProgress()

    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
        progress=progress,
    )

    assert progress.value == result.coverage.rows == len(only_chunk(result).rows)


def test_a_cancelled_progress_stops_the_spec_at_its_first_chunk_boundary():
    """A cancelled progress counter stops materialisation at the next chunk boundary."""
    csv_body = _CSV_HEADER + _csv_row(recipient="country/TGO", year="2015", value="1000")
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )
    progress = RowProgress()
    progress.cancel()

    with pytest.raises(MaterialisationCancelled):
        materialise_spec(
            request,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=coverage,
            progress=progress,
        )
    assert progress.value == 0


def test_output_profile_seconds_measures_only_the_profile_call(monkeypatch: pytest.MonkeyPatch):
    """The reported duration includes the profile call's 20 ms sleep."""
    real_compute_output_profile = finish_module.compute_output_profile

    def _slow_compute_output_profile(*args: object, **kwargs: object):
        time.sleep(0.02)
        return real_compute_output_profile(*args, **kwargs)

    monkeypatch.setattr(finish_module, "compute_output_profile", _slow_compute_output_profile)

    csv_body = _CSV_HEADER + _csv_row(recipient="country/TGO", year="2015", value="1000")
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )

    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert result.output_profile_seconds >= 0.02


def test_role_set_divergence_between_declared_roles_and_the_sdmx_response_refuses():
    """If the SDMX header reports `donor` after shape resolution declared `recipient`,
    streaming raises `SdmxResponseError` before yielding a batch or building a frame."""
    diverged_header = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,donor,unit,measurementMethod,"
        "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
    )
    csv_body = diverged_header + (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/TGO,USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
    )
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )

    with pytest.raises(SdmxResponseError) as excinfo:
        materialise_spec(
            request,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=coverage,
        )

    assert "recipient" in str(excinfo.value)
    assert "donor" in str(excinfo.value)


def _spec(request: MaterialisationRequest, *, sdmx, graph, coverage=None):
    return materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )


_SUBJECT_STAT_VAR = "dcid:Count_Person"
_SUBJECT_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_EARTH_COUNTRIES = {("Earth", "Country"): ["country/KEN", "country/TGO", "country/UGA"]}


def _subject_row(place: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,Count_Person,{place},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,2015,10,,facet1\r\n"
    )


def _subject_request() -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=_SUBJECT_STAT_VAR,
        role_bindings={},
        declared_roles=("observationAbout",),
        facet_bindings=_BOUND_FACETS,
        window=RequestedWindow(kind="all"),
    )


def test_subject_only_request_with_no_bound_entity_mints_every_place_that_holds_data():
    graph = FakeGraphReader(
        facets={
            ("Count_Person", "country/KEN"): [_FACET],
            ("Count_Person", "country/TGO"): [_FACET],
            ("Count_Person", "Earth"): [_FACET],
        },
        child_places=_EARTH_COUNTRIES,
    )
    availability = graph.scope_availability()
    sdmx, requests = _sdmx_client(
        _SUBJECT_CSV_HEADER
        + _subject_row("country/KEN")
        + _subject_row("country/TGO")
        + _subject_row("Earth"),
        availability=availability,
    )

    result = _spec(_subject_request(), sdmx=sdmx, graph=graph)

    # The fetch names every place availability lists, Earth's total among them. Uganda holds
    # no data. The graph reads no containment, and the answer covers every place with data, so
    # it carries no countries warning.
    assert result.coverage.rows == 3
    (fetch,) = requests
    assert fetch.url.params["c[observationAbout]"] == "country/KEN,country/TGO,Earth"
    assert availability.reads == [("Count_Person",)]
    assert graph.calls["child_places_batch"] == 0
    assert result.default_scope == ("country/KEN", "country/TGO", "Earth")
    assert [warning["code"] for warning in result.warnings] == []


def test_subject_only_request_over_more_places_than_one_request_carries_fetches_in_slices():
    places = [f"wikidataId/Q{index:07d}" for index in range(1_200)]
    graph = FakeGraphReader(facets={("Count_Person", place): [_FACET] for place in places})
    availability = graph.scope_availability()
    named: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if (response := availability.respond(request)) is not None:
            return response
        assert len(str(request.url.params)) <= 7_800
        slice_places = str(request.url.params["c[observationAbout]"]).split(",")
        named.append(slice_places)
        return httpx.Response(
            200,
            text=_SUBJECT_CSV_HEADER + "".join(_subject_row(place) for place in slice_places),
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    sdmx = SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler))

    result = _spec(_subject_request(), sdmx=sdmx, graph=graph)

    # Every place is fetched once, across several requests, and every row serves.
    assert len(named) > 1
    assert [place for slice_places in named for place in slice_places] == places
    assert result.coverage.rows == len(places)


def test_subject_only_request_with_no_bound_entity_over_the_row_ceiling_refuses_over_ceiling():
    # Admission has no price for a subject-only request, so this ceiling is what bounds a
    # mint of every country.
    graph = FakeGraphReader(
        facets={
            ("Count_Person", "country/KEN"): [_FACET],
            ("Count_Person", "country/TGO"): [_FACET],
        },
        child_places=_EARTH_COUNTRIES,
    )
    sdmx, _requests = _sdmx_client(
        _SUBJECT_CSV_HEADER + _subject_row("country/KEN") + _subject_row("country/TGO"),
        availability=graph.scope_availability(),
    )

    with pytest.raises(OverCeilingError) as excinfo:
        materialise_spec(
            _subject_request(),
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            materialiser_build="build-123",
            capability_version="cap-v1",
            ceiling=1,
        )

    assert excinfo.value.code == "over_ceiling"


def test_subject_only_request_past_the_availability_cap_mints_its_countries_and_says_so():
    # Availability lists more places than `place_scope.MAX_EVERY_PLACE`, so the scope falls
    # back to the countries with data, and the mint warns that it covers those alone.
    graph = FakeGraphReader(
        facets={
            ("Count_Person", "country/KEN"): [_FACET],
            ("Count_Person", "country/TGO"): [_FACET],
            ("Count_Person", "Earth"): [_FACET],
        },
        child_places=_EARTH_COUNTRIES,
    )
    listed = [("Count_Person", f"geoId/{n:04d}") for n in range(MAX_EVERY_PLACE + 1)]
    sdmx, requests = _sdmx_client(
        _SUBJECT_CSV_HEADER + _subject_row("country/KEN") + _subject_row("country/TGO"),
        availability=ScopeAvailability(data=listed),
    )

    result = _spec(_subject_request(), sdmx=sdmx, graph=graph)

    (fetch,) = requests
    assert fetch.url.params["c[observationAbout]"] == "country/KEN,country/TGO"
    assert result.default_scope == ("country/KEN", "country/TGO")
    (warning,) = result.warnings
    assert warning["code"] == "coverage_below_scope"
    assert warning["message"].endswith(
        "No place was named, so the answer covers the countries with data, because the full "
        "list of places with data was too large or unavailable."
    )


_COUNTRIES_SCOPE_SENTENCE = (
    "No place was named, so the answer covers the countries with data, because the full list "
    "of places with data was too large or unavailable."
)


@pytest.mark.parametrize(
    ("rows", "origin", "every_place", "codes", "countries_stated"),
    [
        (2, "default", False, ["coverage_below_scope"], True),
        (2, "coverage_clamp", False, ["coverage_below_scope"], True),
        (2, "coverage_clamp", True, ["coverage_below_scope"], False),
        (0, "default", False, ["empty_result"], False),
        (0, "coverage_clamp", False, ["empty_result"], False),
    ],
    ids=["countries", "countries-and-clamp", "clamp", "empty-countries", "empty-clamp"],
)
def test_a_countries_scope_is_stated_once_and_never_beside_an_empty_result(
    rows, origin, every_place, codes, countries_stated
):
    outcome = SimpleNamespace(
        total_rows=rows,
        conversion_record=None,
        served_entities_by_role={},
        coverage=Coverage(rows=rows, dimensions=(), window=None),
        applied_window=AppliedWindow(kind="all", start_year=None, end_year=None, origin=origin),
        default_scope=DefaultPlaceScope(places=("country/KEN",), every_place=every_place),
    )

    # A preview-shaped stand-in carries only the fields `_build_warnings` reads.
    warnings = finish_module._build_warnings(_subject_request(), outcome)  # ty: ignore[invalid-argument-type]

    assert [warning["code"] for warning in warnings] == codes
    assert warnings[0]["message"].endswith(_COUNTRIES_SCOPE_SENTENCE) is countries_stated


def test_subject_only_request_with_a_worldwide_total_alone_mints_earth():
    graph = FakeGraphReader(
        facets={("Count_Person", "Earth"): [_FACET]}, child_places=_EARTH_COUNTRIES
    )
    availability = graph.scope_availability()
    sdmx, requests = _sdmx_client(
        _SUBJECT_CSV_HEADER + _subject_row("Earth"), availability=availability
    )

    result = _spec(_subject_request(), sdmx=sdmx, graph=graph)

    assert result.coverage.rows == 1
    (fetch,) = requests
    assert fetch.url.params["c[observationAbout]"] == "Earth"
    assert availability.reads == [("Count_Person",)]
    assert result.default_scope == ("Earth",)


@pytest.mark.parametrize("places", [(), ("country/KEN",)], ids=["default_scope", "bound_place"])
@pytest.mark.parametrize(
    "facet_bindings",
    [
        {},
        _BOUND_FACETS,
        # No rate cache is supplied, so planning this conversion would raise.
        {
            "unit": FacetBinding(
                state="request", request=UnitRequest(currency="USD", prices="current")
            )
        },
    ],
    ids=["unbound_facets", "bound_facets", "unit_request"],
)
def test_a_member_whose_probe_returns_no_candidate_is_the_empty_result(places, facet_bindings):
    # Neither a country nor Earth holds an observation, so the probe is empty whichever places
    # the request covers. The result selects no facet and plans no conversion.
    graph = FakeGraphReader(child_places=_EARTH_COUNTRIES)
    sdmx, requests = _sdmx_client(_SUBJECT_CSV_HEADER, availability=graph.scope_availability())
    request = replace(
        _subject_request(),
        role_bindings={"observationAbout": list(places)} if places else {},
        facet_bindings=facet_bindings,
    )

    result = _spec(request, sdmx=sdmx, graph=graph)

    assert result.probe_empty
    assert result.coverage.rows == 0
    assert [warning["code"] for warning in result.warnings] == ["empty_result"]
    assert result.trace.facets == ()
    assert result.conversion_record is None
    assert requests == []


def test_role_bearing_request_with_no_bound_entity_fetches_every_covered_entity():
    """An unbound role means every entity: coverage supplies the candidates and the fetch
    filters on every covered dcid of the smaller role axis."""
    csv_body = (
        _CSV_HEADER
        + _csv_row(recipient="country/TGO", year="2015", value="1000")
        + _csv_row(recipient="country/KEN", year="2015", value="2000")
    )
    sdmx, requests = _sdmx_client(csv_body)
    graph = FakeGraphReader()
    coverage = _coverage_for([_FACET], entities=["country/TGO", "country/KEN"])
    request = _request(role_bindings={}, window=RequestedWindow(kind="all"))

    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert result.coverage.rows == 2
    assert [request.url.params["c[donor]"] for request in requests] == ["country/DONOR"]
    assert all("c[recipient]" not in request.url.params for request in requests)
    assert graph.calls["variable_facets_batch"] == 0
    assert graph.calls["observation_facets_batch_many"] == 0
    assert graph.calls["child_places_batch"] == 0


def test_unbound_role_bearing_request_with_a_window_outside_coverage_materialises_no_rows():
    """Coverage holds 2015 only. A 1990 to 1995 window still fetches on every covered donor,
    and the window drops each returned row."""
    csv_body = _CSV_HEADER + _csv_row(recipient="country/TGO", year="2015", value="1000")
    sdmx, requests = _sdmx_client(csv_body)
    graph = FakeGraphReader()
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={}, window=RequestedWindow(kind="range", start_year=1990, end_year=1995)
    )

    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert result.coverage.rows == 0
    assert [request.url.params["c[donor]"] for request in requests] == ["country/DONOR"]


def test_series_statistics_populated_and_content_version_is_independent_of_them():
    csv_body = (
        _CSV_HEADER
        + _csv_row(recipient="country/TGO", year="2015", value="1000")
        + _csv_row(recipient="country/TGO", year="2016", value="2000")
    )
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )

    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert result.series_statistics.series_count == 1
    statistics_table = pq.read_table(io.BytesIO(result.series_statistics.data))
    assert statistics_table.num_rows == 1
    (series,) = statistics_table.to_pylist()
    assert series["stat_var"] == _STAT_VAR
    assert series["observations"] == 2
    assert series["entities"] == [{"role": "recipient", "ref": "country/TGO"}]

    # Series statistics are outside the content-version preimage.
    assert mint_version(only_chunk(result)) == result.content_version


def test_empty_result_zero_rows_survive_window_filter():
    csv_body = _CSV_HEADER + _csv_row(recipient="country/TGO", year="2015", value="1000")
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]},
        window=RequestedWindow(kind="range", start_year=2030, end_year=2031),
    )

    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert only_chunk(result).rows == ()
    assert result.trace.applied_window is None
    assert result.coverage.rows == 0
    # A zero-row result reports only `empty_result`.
    assert len(result.warnings) == 1
    assert result.warnings[0]["code"] == "empty_result"
    assert result.warnings[0]["severity"] == "warn"  # the shared severity vocabulary, not "warning"
    # The probe held a candidate, so the facets it selected stay in the trace.
    assert not result.probe_empty
    assert result.trace.facets


def test_blank_obs_value_row_surviving_filters_is_empty_result_not_a_stale_window():
    # Blank values are dropped before window, coverage, and ceiling calculations.
    csv_body = _CSV_HEADER + _csv_row(recipient="country/TGO", year="2015", value="")
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )

    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert only_chunk(result).rows == ()
    assert result.trace.applied_window is None
    assert result.coverage.rows == 0
    codes = {w["code"] for w in result.warnings}
    assert codes == {"empty_result"}


def test_over_ceiling_raises_with_real_coverage_named_in_detail():
    # `latest` is a post-fetch max reduction. Both rows use 2016, so both survive and exceed
    # the ceiling of 1 after coverage is measured.
    csv_body = (
        _CSV_HEADER
        + _csv_row(recipient="country/TGO", year="2016", value="1000")
        + _csv_row(recipient="country/TGO", year="2016", value="2000")
    )
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="latest")
    )

    with pytest.raises(MaterialisationError) as excinfo:
        materialise_spec(
            request,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            ceiling=1,
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=coverage,
        )

    assert isinstance(excinfo.value, OverCeilingError)
    assert excinfo.value.code == "over_ceiling"
    assert excinfo.value.detail is not None
    assert "2" in excinfo.value.detail
    assert "1" in excinfo.value.detail


def test_narrow_window_over_a_raw_response_past_the_ceiling_still_succeeds():
    # SDMX returns full history; the 2025 window keeps one of 50 rows, below `ceiling=10`.
    # The raw fetch therefore needs a guard wider than `ceiling + 1`.
    ceiling = 10
    csv_body = _CSV_HEADER + "".join(
        _csv_row(recipient="country/TGO", year=str(year), value="1000")
        for year in range(2000, 2050)
    )
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]},
        window=RequestedWindow(kind="range", start_year=2025, end_year=2025),
    )

    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        ceiling=ceiling,
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert result.coverage.rows == 1
    assert only_chunk(result).rows != ()


def test_raw_fetch_guard_trip_refuses_over_ceiling_naming_the_raw_abort():
    # With `ceiling=5`, the raw guard is 100; only one of 150 rows matches the window.
    # The refusal names the raw guard, and the batch count may exceed 100 by one batch.
    ceiling = 5
    raw_guard = 20 * ceiling
    csv_body = _CSV_HEADER + "".join(
        _csv_row(recipient="country/TGO", year=str(2000 + i), value="1000")
        for i in range(raw_guard + 50)
    )
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]},
        window=RequestedWindow(kind="range", start_year=2025, end_year=2025),
    )

    with pytest.raises(MaterialisationError) as excinfo:
        materialise_spec(
            request,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            ceiling=ceiling,
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=coverage,
        )

    assert isinstance(excinfo.value, OverCeilingError)
    assert excinfo.value.code == "over_ceiling"
    assert excinfo.value.detail is not None
    assert f"raw fetch guard of {raw_guard} rows" in excinfo.value.detail
    assert "was not measured" in excinfo.value.detail


def test_counted_early_refusal_aborts_the_stream_before_reading_every_chunk():
    # Every row matches; the survivor count crosses 5 before the raw guard of 100.
    # The 80-block body exceeds Arrow read-ahead, so `chunks_yielded` proves reading stops early.
    from dc_sdmx.client import _STREAM_BATCH_BLOCK_SIZE
    from tests.dc_sdmx.conftest import StreamingTransport

    ceiling = 5
    one_row = _csv_row(recipient="country/TGO", year="2000", value="1000")
    rows = -(-80 * _STREAM_BATCH_BLOCK_SIZE // len(one_row))  # well over the read-ahead window
    csv_body = (_CSV_HEADER + one_row * rows).encode()
    stream = StreamingTransport(body=csv_body, chunk_size=4096)
    sdmx = SdmxClient(base_url=_BASE_URL, transport=stream)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )

    with pytest.raises(OverCeilingError) as excinfo:
        materialise_spec(
            request,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            ceiling=ceiling,
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=coverage,
        )

    assert excinfo.value.code == "over_ceiling"
    assert excinfo.value.detail is not None
    assert "was not measured" in excinfo.value.detail
    assert stream.chunks_yielded < stream.total_chunks


def test_latest_window_over_a_huge_raw_response_does_not_early_refuse_and_succeeds():
    # `latest` reduces all 90 raw rows to one survivor before applying the ceiling of 5.
    ceiling = 5
    csv_body = _CSV_HEADER + "".join(
        _csv_row(recipient="country/TGO", year=str(2000 + i), value="1000") for i in range(90)
    )
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="latest")
    )

    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        ceiling=ceiling,
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert result.coverage.rows == 1
    assert result.trace.applied_window is not None
    assert result.trace.applied_window["kind"] == "latest"


@pytest.mark.parametrize(
    ("csv_body", "role_bindings", "window", "expected_origin", "expected_message"),
    [
        pytest.param(
            _CSV_HEADER + _csv_row(recipient="country/TGO", year="2015", value="1000"),
            {"recipient": ["country/TGO", "country/KEN"]},
            RequestedWindow(kind="all"),
            "default",
            "The served coverage falls short of the requested scope. "
            "No observations were served for country/KEN.",
            id="entity_shortfall",
        ),
        pytest.param(
            _CSV_HEADER
            + _csv_row(recipient="country/TGO", year="2015", value="1000")
            + _csv_row(recipient="country/TGO", year="2016", value="2000"),
            {"recipient": ["country/TGO"]},
            RequestedWindow(kind="range", start_year=2010, end_year=2020),
            "coverage_clamp",
            "The served coverage falls short of the requested scope.",
            id="window_clamp",
        ),
    ],
)
def test_coverage_below_scope_warning(
    csv_body: str,
    role_bindings: dict[str, list[str]],
    window: RequestedWindow,
    expected_origin: str,
    expected_message: str,
):
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings=role_bindings,
        window=window,
    )

    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert result.trace.applied_window["origin"] == expected_origin
    warnings_by_code = {w["code"]: w for w in result.warnings}
    assert "coverage_below_scope" in warnings_by_code
    warning = warnings_by_code["coverage_below_scope"]
    assert warning["severity"] == "warn"
    assert warning["message"] == expected_message


def test_coverage_below_scope_names_only_the_unserved_entity():
    csv_body = _CSV_HEADER + _csv_row(recipient="country/TGO", year="2015", value="1000")
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO", "country/SEN"]},
        window=RequestedWindow(kind="all"),
    )

    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    warning = {w["code"]: w for w in result.warnings}["coverage_below_scope"]
    assert "country/SEN" in warning["message"]
    assert "country/TGO" not in warning["message"]


def test_coverage_below_scope_caps_named_entities_at_five():
    bound = ["country/TGO"] + [f"country/Unserved{i}" for i in range(6)]
    csv_body = _CSV_HEADER + _csv_row(recipient="country/TGO", year="2015", value="1000")
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(role_bindings={"recipient": bound}, window=RequestedWindow(kind="all"))

    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    warning = {w["code"]: w for w in result.warnings}["coverage_below_scope"]
    assert warning["message"] == (
        "The served coverage falls short of the requested scope. "
        "No observations were served for country/Unserved0, country/Unserved1, "
        "country/Unserved2, country/Unserved3, country/Unserved4 and 1 more."
    )


def test_as_issue_carries_the_wrapped_errors_code_and_message():
    exc = DefaultFacetUndefinedError("axis 'unit' has no default", facet_pointer="/facets/unit")
    issue = _as_issue("unit", exc)
    assert issue.code == exc.code == "default_facet_undefined"
    assert issue.detail == str(exc)


def test_as_issue_leaves_detail_unset_for_an_error_with_no_message():
    issue = _as_issue("unit", FacetNotAvailableError(facet_pointer="/facets/unit"))
    assert issue.code == "facet_not_available"
    assert issue.detail is None


def test_as_issue_truncated_is_none_when_no_alternatives():
    exc = DefaultFacetUndefinedError(facet_pointer="/facets/provenance")
    issue = _as_issue("provenance", exc)
    assert issue.alternatives is None
    assert issue.truncated is None


def test_as_issue_truncated_is_false_below_the_cap():
    alts = [Alternative(ref=f"r{i}", label=f"l{i}") for i in range(5)]
    exc = FacetNotAvailableError(facet_pointer="/facets/unit", alternatives=alts)
    issue = _as_issue("unit", exc)
    assert issue.alternatives == alts
    assert issue.truncated is False


def test_as_issue_truncated_is_true_above_the_cap():
    alts = [Alternative(ref=f"r{i}", label=f"l{i}") for i in range(_MAX_ALTERNATIVES_PER_ISSUE + 5)]
    exc = FacetNotAvailableError(facet_pointer="/facets/unit", alternatives=alts)
    issue = _as_issue("unit", exc)
    assert issue.alternatives is not None
    assert len(issue.alternatives) == _MAX_ALTERNATIVES_PER_ISSUE
    assert issue.truncated is True


def test_as_issue_reports_truncated_end_to_end_when_select_facet_dedups_over_the_cap():
    # `select_facet` sees more dispatchable unit values than the cap and sets `truncated`.
    candidates = [
        Facet(
            facet_id=f"facet{i}",
            provenance_id="OECD_DAC2A",
            measurement_method="GrantEquivalent",
            unit=f"UNIT{i}",
            obs_count=1,
            earliest_date="2015",
            latest_date="2015",
        )
        for i in range(_MAX_ALTERNATIVES_PER_ISSUE + 3)
    ]
    graph = FakeGraphReader()

    with pytest.raises(FacetNotAvailableError) as excinfo:
        select_facet(
            "unit",
            FacetBinding(state="bound", ref="NOMATCH"),
            candidates=candidates,
            graph=graph,
            stat_var=_STAT_VAR,
            window=None,
            policies=declared_default_policies(
                {"unit": FacetBinding(state="bound", ref="NOMATCH")},
                graph=graph,
                stat_var=_STAT_VAR,
            ),
        )

    issue = _as_issue("unit", excinfo.value)
    assert issue.alternatives is not None
    assert len(issue.alternatives) == _MAX_ALTERNATIVES_PER_ISSUE
    assert issue.truncated is True


# --- `MaterialisationRequest.__post_init__`'s own axis rule ---------------


def test_materialisation_request_rejects_a_request_binding_under_a_non_unit_key():
    # The axis lives in exactly one place, the `facet_bindings` key --
    # `FacetBinding` itself carries no axis of its own -- so this is the one
    # place a request binding stored under the wrong key can be caught.
    with pytest.raises(MaterialisationUsageError):
        MaterialisationRequest(
            stat_var=_STAT_VAR,
            role_bindings={"recipient": ["country/KEN"]},
            declared_roles=("recipient",),
            facet_bindings={
                "provenance": FacetBinding(
                    state="request", request=UnitRequest(currency="USD", prices="current")
                )
            },
            window=RequestedWindow(kind="all"),
        )


def test_materialisation_request_accepts_a_request_binding_under_the_unit_key():
    request = MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings={"recipient": ["country/KEN"]},
        declared_roles=("recipient",),
        facet_bindings={
            "unit": FacetBinding(
                state="request", request=UnitRequest(currency="USD", prices="current")
            )
        },
        window=RequestedWindow(kind="all"),
    )

    assert request.facet_bindings["unit"].state == "request"
