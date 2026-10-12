"""Tests bounded previews, full survivor counts, facet selection, and write-free materialisation."""

from __future__ import annotations

import time
import unicodedata

import httpx
import pytest

from dc_sdmx import SdmxCapExceededError, SdmxClient
from materialise import (
    MaterialisationRequest,
    finish_materialise_preview,
    materialise_preview,
    materialise_spec,
    prepare_materialise_preview,
)
from materialise.enrichment import EnrichmentLookup
from materialise.errors import (
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    MaterialisationUsageError,
    UpstreamResponseError,
)
from materialise.facets import FacetBinding, UnitRequest
from materialise.graph import Facet
from materialise.rate_tables import RateTableCache
from materialise.recipe_bodies import RECIPE_BODIES
from materialise.recipes import RateCoverageGapError
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._rate_table_support import RateTableServer, rate_csv, rate_row
from ._scope_availability import ScopeAvailability
from .conftest import FakeGraphReader

_BASE_URL = "https://datacommons.one.org/core/api"
_STAT_VAR = "dcid:MalariaGrantDisbursements"
# The graph wire keys on the bare dcid; `FakeGraphReader` fixtures below use
# that bare form to match what `generation.py` sends.
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


def _time_periods(frame) -> list[object]:
    """Returns each row's `time_period` cell by its index in `frame.columns`."""
    index = frame.columns.index("time_period")
    return [row[index] for row in frame.rows]


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


def _coverage_for(facets: list[Facet], *, entities: list[str], year: int = 2015) -> CoverageIndex:
    """Builds coverage rows for each facet and recipient.

    The request binds only recipient. `year` keeps rows in range while tests focus on facet selection.
    """
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


_CONVERSION_BARE_STAT_VAR = "Amount_FinancialFlow_DevelopmentAssistance_Nominal"
_CONVERSION_STAT_VAR = f"dcid:{_CONVERSION_BARE_STAT_VAR}"
_CONVERSION_PROVENANCE = "provenance/OECDCreditorReportingSystem"
_CONVERSION_RECIPE = RECIPE_BODIES["ConversionRecipeDAC"]
_CONVERSION_ROLES = ("financingProvider", "financingRecipient")
_CONVERSION_RECIPIENT = "country/KEN"
_CONVERSION_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,financingProvider,financingRecipient,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _conversion_row(*, provider: str, year: int, value: float = 100.0) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_CONVERSION_BARE_STAT_VAR},{provider},"
        f"{_CONVERSION_RECIPIENT},CurrencyUSD,GrantEquivalent,NotApplicable,"
        f"{_CONVERSION_PROVENANCE},{year},{value},,f1\r\n"
    )


def _conversion_request(*, currency: str = "EUR") -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=_CONVERSION_STAT_VAR,
        role_bindings={"financingRecipient": [_CONVERSION_RECIPIENT]},
        declared_roles=_CONVERSION_ROLES,
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref=_CONVERSION_PROVENANCE),
            "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
            "unit": FacetBinding(
                state="request", request=UnitRequest(currency=currency, prices="current")
            ),
        },
        window=RequestedWindow(kind="all"),
    )


def _conversion_graph(
    *, national_currencies: dict[str, tuple[str, ...]] | None = None
) -> FakeGraphReader:
    currency_arcs = {
        currency: {"typeOf": ("CurrencyUnit",), "nationalCurrencyOf": places}
        for currency, places in (national_currencies or {}).items()
    }
    return FakeGraphReader(
        arcs={
            _CONVERSION_BARE_STAT_VAR: {
                "observationProperties": _CONVERSION_ROLES,
                "measurementQualifier": ("Nominal",),
                "populationType": ("FinancialFlow",),
            },
            _CONVERSION_PROVENANCE: {"conversionRecipe": (_CONVERSION_RECIPE.recipe_id,)},
            _CONVERSION_RECIPE.table_provenance: {"lastDataRefreshDate": ("2026-01-01T00:00:00Z",)},
            "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
            "CurrencyEUR": {"typeOf": ("CurrencyUnit",)},
            **currency_arcs,
        }
    )


def _conversion_coverage(providers: list[str]) -> CoverageIndex:
    return build_coverage_index(
        [
            (_CONVERSION_BARE_STAT_VAR, "f1", provider, _CONVERSION_RECIPIENT, [2024])
            for provider in providers
        ],
        role_names={
            _CONVERSION_BARE_STAT_VAR: RoleNames(
                donor="financingProvider", recipient="financingRecipient"
            )
        },
        facet_metadata={
            "f1": FacetMetadata(
                provenance_id=_CONVERSION_PROVENANCE,
                measurement_method="GrantEquivalent",
                unit="CurrencyUSD",
            )
        },
    )


def _conversion_sdmx(
    *,
    rows: list[str],
    rate_years: list[int],
    currencies: tuple[tuple[str, str], ...] = (("CurrencyUSD", "1.0"), ("CurrencyEUR", "0.9")),
) -> tuple[SdmxClient, RateTableServer]:
    observation_body = _CONVERSION_HEADER + "".join(rows)

    def observations(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=observation_body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    rate_body = rate_csv(
        *(
            rate_row(
                stat_var=_CONVERSION_RECIPE.exchange_rate_variable,
                entity=currency,
                year=str(year),
                value=value,
                provenance=_CONVERSION_RECIPE.table_provenance,
            )
            for year in rate_years
            for currency, value in currencies
        )
    )
    server = RateTableServer(
        {_CONVERSION_RECIPE.exchange_rate_variable: rate_body}, fallback=observations
    )
    return server.client(base_url=_BASE_URL), server


def _observation_data_reads(server: RateTableServer) -> list[httpx.Request]:
    return [
        request
        for request in server.data_requests()
        if request.url.params.get("c[variableMeasured]") == _CONVERSION_BARE_STAT_VAR
    ]


def test_prepared_preview_defers_an_earlier_series_rate_gap_until_after_exact_period_selection():
    """An open provider series at 2024 lacks a euro rate, while the global 2025 row is
    convertible. Preparing reads observations without applying either conversion; finishing
    at 2025 converts the retained row without a second observation read."""
    earlier, latest = "country/DEU", "country/FRA"
    sdmx, server = _conversion_sdmx(
        rows=[
            _conversion_row(provider=earlier, year=2024),
            _conversion_row(provider=latest, year=2025),
        ],
        rate_years=[2025],
    )
    prepared = prepare_materialise_preview(
        _conversion_request(),
        sdmx=sdmx,
        graph=_conversion_graph(),
        coverage=_conversion_coverage([earlier, latest]),
        limit=10,
        rate_cache=RateTableCache(),
        series_roles=("financingProvider",),
        series_limit=10,
    )

    assert prepared.latest_period == "2025"
    assert len(_observation_data_reads(server)) == 1

    result = finish_materialise_preview(prepared, time_period="2025")

    provider = result.frame.columns.index("financingProvider")
    assert [
        (row[provider], period)
        for row, period in zip(result.frame.rows, _time_periods(result.frame), strict=True)
    ] == [(latest, "2025")]
    assert result.frame.table["value"].to_pylist() == [pytest.approx(90.0)]
    assert result.conversion_record is not None
    assert len(_observation_data_reads(server)) == 1


def test_prepared_preview_refuses_a_rate_gap_in_the_retained_exact_period():
    provider = "country/DEU"
    sdmx, _server = _conversion_sdmx(
        rows=[_conversion_row(provider=provider, year=2024)], rate_years=[2025]
    )
    prepared = prepare_materialise_preview(
        _conversion_request(),
        sdmx=sdmx,
        graph=_conversion_graph(),
        coverage=_conversion_coverage([provider]),
        limit=10,
        rate_cache=RateTableCache(),
        series_roles=("financingProvider",),
        series_limit=10,
    )

    with pytest.raises(RateCoverageGapError) as missing:
        finish_materialise_preview(prepared, time_period="2024")

    assert missing.value.cause == "rate_missing"


def test_prepared_preview_discards_unplanned_open_role_rows_before_finalisation():
    """An allow-list drops an unexpected open-provider tuple before a latest period is
    selected or arithmetic touches a row, so a response containing only that tuple finalises
    empty."""
    planned, unplanned = "country/FRA", "country/DEU"
    sdmx, server = _conversion_sdmx(
        rows=[_conversion_row(provider=unplanned, year=2024)], rate_years=[2025]
    )
    prepared = prepare_materialise_preview(
        _conversion_request(),
        sdmx=sdmx,
        graph=_conversion_graph(),
        coverage=_conversion_coverage([planned, unplanned]),
        limit=10,
        rate_cache=RateTableCache(),
        series_roles=("financingProvider",),
        series_limit=10,
        allowed_series_keys=frozenset({(planned,)}),
    )

    result = finish_materialise_preview(prepared)

    assert prepared.latest_period is None
    assert result.total_rows == 0
    assert result.frame.rows == ()
    assert result.conversion_record is not None
    assert result.series_conversion_records == {}
    assert len(_observation_data_reads(server)) == 1


@pytest.mark.parametrize(
    ("series_roles", "allowed_series_keys", "series_limit", "message"),
    [
        ((), frozenset({()}), None, "allowed_series_keys"),
        (("notADeclaredRole",), frozenset({("country/FRA",)}), 10, "series_roles"),
        (
            ("financingProvider",),
            frozenset({("country/FRA", "country/DEU")}),
            10,
            "allowed_series_keys",
        ),
    ],
)
def test_prepared_preview_validates_allowed_series_keys(
    series_roles: tuple[str, ...],
    allowed_series_keys: frozenset[tuple[str, ...]],
    series_limit: int | None,
    message: str,
) -> None:
    sdmx, _server = _conversion_sdmx(rows=[], rate_years=[2025])

    with pytest.raises(MaterialisationUsageError, match=message):
        prepare_materialise_preview(
            _conversion_request(),
            sdmx=sdmx,
            graph=_conversion_graph(),
            coverage=_conversion_coverage([]),
            limit=10,
            rate_cache=RateTableCache(),
            series_roles=series_roles,
            series_limit=series_limit,
            allowed_series_keys=allowed_series_keys,
        )


def test_rows_under_limit_are_not_truncated_and_every_survivor_is_in_the_frame():
    csv_body = (
        _CSV_HEADER
        + _csv_row(recipient="country/TGO", year="2015", value="1000")
        + _csv_row(recipient="country/TGO", year="2016", value="2000")
    )
    sdmx, requests = _sdmx_client(csv_body)
    deadlines: list[float | None] = []
    original_stream_data_batches = sdmx.stream_data_batches

    def recording_stream_data_batches(**kwargs):
        deadlines.append(kwargs.get("deadline"))
        return original_stream_data_batches(**kwargs)

    sdmx.stream_data_batches = recording_stream_data_batches  # type: ignore[method-assign]
    deadline = time.monotonic() + 30
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )

    result = materialise_preview(
        request, sdmx=sdmx, graph=graph, coverage=coverage, limit=10, deadline=deadline
    )

    # The one dispatched fetch: candidates come from `coverage`, an
    # in-memory lookup, so no SDMX probe call precedes it.
    assert len(requests) == 1
    assert deadlines == [deadline]
    assert result.truncated is False
    assert result.total_rows == 2
    assert len(result.frame.rows) == result.total_rows
    assert result.warnings == ()


def test_a_role_bearing_member_serves_its_rows_unmasked():
    """Role-bearing candidates come from coverage, which holds DFO's data alone, so no place
    admission masks their rows: a recipient the coverage rows omit still serves."""
    csv_body = (
        _CSV_HEADER
        + _csv_row(recipient="country/TGO", year="2015", value="1000")
        + _csv_row(recipient="country/KEN", year="2015", value="2000")
    )
    sdmx, _requests = _sdmx_client(csv_body)
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO", "country/KEN"]},
        window=RequestedWindow(kind="all"),
    )

    result = materialise_preview(
        request, sdmx=sdmx, graph=FakeGraphReader(), coverage=coverage, limit=10
    )

    assert result.coverage.entity_refs["recipient"] == frozenset({"country/TGO", "country/KEN"})


def test_rows_over_limit_truncate_the_frame_but_total_rows_is_the_true_full_count():
    csv_body = _CSV_HEADER + "".join(
        _csv_row(recipient="country/TGO", year=str(2000 + i), value="1000") for i in range(10)
    )
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )

    result = materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=3)

    assert result.truncated is True
    assert (
        result.total_rows == 10
    )  # NOT capped at limit, contrast test_ceiling.py's OverCeilingError
    assert len(result.frame.rows) == 3

    # The preview returns the most recent periods.
    assert set(_time_periods(result.frame)) == {"2007", "2008", "2009"}


def test_truncation_drops_null_periods_first():
    # `NotApplicable` sorts first and truncation drops it.
    dated_row_a = _csv_row(recipient="country/TGO", year="2019", value="1000")
    dated_row_b = _csv_row(recipient="country/KEN", year="2020", value="2000")
    null_period_row = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/MOZ,USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,NotApplicable,3000,,facet1\r\n"
    )
    csv_body = _CSV_HEADER + dated_row_a + dated_row_b + null_period_row
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(
        facets={
            (_BARE_STAT_VAR, "country/TGO"): [_FACET],
            (_BARE_STAT_VAR, "country/KEN"): [_FACET],
            (_BARE_STAT_VAR, "country/MOZ"): [_FACET],
        }
    )
    coverage = _coverage_for([_FACET], entities=["country/TGO", "country/KEN", "country/MOZ"])
    request = _request(
        role_bindings={"recipient": ["country/TGO", "country/KEN", "country/MOZ"]},
        window=RequestedWindow(kind="all"),
    )

    result = materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=2)

    assert result.truncated is True
    assert result.total_rows == 3
    assert len(result.frame.rows) == 2
    recipient_index = result.frame.columns.index("recipient")
    delivered_recipients = {row[recipient_index] for row in result.frame.rows}
    assert delivered_recipients == {"country/TGO", "country/KEN"}


def test_truncation_ties_at_the_boundary_period_keep_the_later_rows_in_response_order():
    # A period tie keeps the last rows in response order.
    csv_body = _CSV_HEADER + "".join(
        _csv_row(recipient=f"country/R{i}", year="2020", value="1000") for i in range(3)
    )
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, f"country/R{i}"): [_FACET] for i in range(3)})
    coverage = _coverage_for([_FACET], entities=[f"country/R{i}" for i in range(3)])
    request = _request(
        role_bindings={"recipient": [f"country/R{i}" for i in range(3)]},
        window=RequestedWindow(kind="all"),
    )

    result = materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=2)

    assert result.truncated is True
    assert result.total_rows == 3
    assert len(result.frame.rows) == 2
    recipient_index = result.frame.columns.index("recipient")
    delivered_recipients = {row[recipient_index] for row in result.frame.rows}
    assert delivered_recipients == {"country/R1", "country/R2"}


def test_coverage_reflects_the_full_survivor_set_even_when_the_frame_is_truncated():
    csv_body = _CSV_HEADER + "".join(
        _csv_row(recipient="country/TGO", year=str(2000 + i), value="1000") for i in range(10)
    )
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )

    result = materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=3)

    assert result.truncated is True
    # Coverage counts 10 survivors while the preview delivers 3.
    assert result.coverage.rows == 10
    years_dimension = next(d for d in result.coverage.dimensions if d["label"] == "years")
    assert years_dimension["count"] == 10


def test_latest_window_carve_out_over_a_huge_raw_response_true_counts_then_truncates():
    # Eight distinct entities survive period 2016; the preview delivers 3.
    entities = [f"country/E{i}" for i in range(8)]
    csv_body = _CSV_HEADER + "".join(
        _csv_row(recipient=entity, year="2016", value="1000") for entity in entities
    )
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, entity): [_FACET] for entity in entities})
    coverage = _coverage_for([_FACET], entities=entities)
    request = _request(role_bindings={"recipient": entities}, window=RequestedWindow(kind="latest"))

    result = materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=3)

    assert result.truncated is True
    assert result.total_rows == 8
    assert len(result.frame.rows) == 3
    assert all(period == "2016" for period in _time_periods(result.frame))


_SUBJECT_ONLY_HEADER = _CSV_HEADER.replace(",recipient,", ",observationAbout,")
_SUBJECT_ONLY_ROW = (
    "dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/KEN,USD,GrantEquivalent,"
    "NotApplicable,OECD_DAC2A,2015,10,,facet1\r\n"
)


def _subject_only_graph() -> FakeGraphReader:
    """Kenya holds data among Earth's two countries."""
    return FakeGraphReader(
        facets={("Count_Person", "country/KEN"): [_FACET]},
        child_places={("Earth", "Country"): ["country/KEN", "country/TGO"]},
    )


def _subject_only_request() -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var="dcid:Count_Person",
        role_bindings={},
        declared_roles=("observationAbout",),
        facet_bindings=_BOUND_FACETS,
        window=RequestedWindow(kind="all"),
    )


def test_subject_only_request_with_no_bound_entity_previews_the_countries_that_hold_data():
    graph = _subject_only_graph()
    sdmx, requests = _sdmx_client(
        _SUBJECT_ONLY_HEADER + _SUBJECT_ONLY_ROW, availability=graph.scope_availability()
    )

    result = materialise_preview(_subject_only_request(), sdmx=sdmx, graph=graph, limit=10)

    assert result.total_rows == 1
    assert result.truncated is False
    (fetch,) = requests
    assert fetch.url.params["c[observationAbout]"] == "country/KEN"


def test_precomputed_candidates_for_a_request_binding_no_place_are_a_caller_fault():
    """Precomputed candidates cannot say at which of a default scope's places each facet was
    admitted, so a subject-only request binding no place refuses them before any read."""
    graph = _subject_only_graph()
    sdmx, requests = _sdmx_client(_SUBJECT_ONLY_HEADER + _SUBJECT_ONLY_ROW)

    with pytest.raises(MaterialisationUsageError, match="bind 0 places"):
        materialise_preview(
            _subject_only_request(), sdmx=sdmx, graph=graph, limit=10, precomputed=[_FACET]
        )

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

    result = materialise_preview(
        request,
        sdmx=sdmx,
        graph=graph,
        limit=10,
        coverage=coverage,
    )

    assert result.total_rows == 2
    assert [request.url.params["c[donor]"] for request in requests] == ["country/DONOR"]
    assert all("c[recipient]" not in request.url.params for request in requests)
    assert graph.calls["variable_facets_batch"] == 0
    assert graph.calls["observation_facets_batch_many"] == 0


def test_unbound_role_bearing_request_with_a_window_outside_coverage_previews_no_rows():
    """Coverage holds 2015 only. A 1990 to 1995 window still fetches on every covered donor,
    and the window drops each returned row."""
    csv_body = _CSV_HEADER + _csv_row(recipient="country/TGO", year="2015", value="1000")
    sdmx, requests = _sdmx_client(csv_body)
    graph = FakeGraphReader()
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={}, window=RequestedWindow(kind="range", start_year=1990, end_year=1995)
    )

    result = materialise_preview(request, sdmx=sdmx, graph=graph, limit=10, coverage=coverage)

    assert result.total_rows == 0
    assert len(result.frame.rows) == 0
    assert [request.url.params["c[donor]"] for request in requests] == ["country/DONOR"]


def test_default_facet_undefined_raises_same_as_materialise_spec():
    # The unbound provenance candidate has no ref, so selection raises.
    no_provenance_facet = Facet(
        facet_id="facet1",
        provenance_id=None,
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    # SDMX `NotApplicable` provenance parses to `None`.
    csv_body = (
        _CSV_HEADER + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/TGO,USD,GrantEquivalent,"
        "NotApplicable,NotApplicable,2015,1000,,facet1\r\n"
    )
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [no_provenance_facet]})
    coverage = _coverage_for([no_provenance_facet], entities=["country/TGO"])
    unbound_provenance = {
        "provenance": FacetBinding(state="unbound"),
        "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
        "unit": FacetBinding(state="bound", ref="USD"),
    }
    request = MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings={"recipient": ["country/TGO"]},
        declared_roles=("recipient",),
        facet_bindings=unbound_provenance,
        window=RequestedWindow(kind="all"),
    )

    with pytest.raises(DefaultFacetUndefinedError):
        materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=10)


def test_facet_not_available_raises_same_as_materialise_spec():
    csv_body = _CSV_HEADER + _csv_row(recipient="country/TGO", year="2015", value="1000")
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    nomatch_unit = {
        "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
        "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
        "unit": FacetBinding(state="bound", ref="NOMATCH"),
    }
    request = MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings={"recipient": ["country/TGO"]},
        declared_roles=("recipient",),
        facet_bindings=nomatch_unit,
        window=RequestedWindow(kind="all"),
    )

    with pytest.raises(FacetNotAvailableError):
        materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=10)


def test_role_divergence_raises_upstream_response_error_same_as_materialise_spec():
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

    with pytest.raises(UpstreamResponseError) as excinfo:
        materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=10)

    assert "recipient" in str(excinfo.value)
    assert "donor" in str(excinfo.value)


def test_non_positive_limit_raises_materialisation_usage_error_before_any_network_call():
    sdmx, requests = _sdmx_client(_CSV_HEADER)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )

    with pytest.raises(MaterialisationUsageError):
        materialise_preview(request, sdmx=sdmx, graph=graph, limit=0)

    assert requests == []


def test_two_facets_sharing_identity_collapse_to_one_and_the_preview_builds():
    # The facets differ only by id and period, so their rows share one output grain.
    period_row = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/TGO,USD,GrantEquivalent,"
        f"P1Y,OECD_DAC2A,2015,1000,,facet1\r\n"
    )
    periodless_row = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/TGO,USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,2015,1000,,facet2\r\n"
    )
    csv_body = _CSV_HEADER + period_row + periodless_row
    sdmx, _requests = _sdmx_client(csv_body)
    facet_with_period = Facet(
        facet_id="facet1",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    facet_without_period = Facet(
        facet_id="facet2",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    graph = FakeGraphReader(
        facets={(_BARE_STAT_VAR, "country/TGO"): [facet_with_period, facet_without_period]}
    )
    coverage = _coverage_for([facet_with_period, facet_without_period], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )

    result = materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=10)

    assert result.total_rows == 1
    assert len(result.frame.rows) == 1
    assert result.coverage.rows == 1


def test_two_facets_on_the_same_identity_but_disjoint_recipients_both_survive():
    # The facets serve different recipients, so both rows survive.
    tgo_row = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/TGO,USD,GrantEquivalent,"
        f"P1Y,OECD_DAC2A,2015,1000,,facet1\r\n"
    )
    ken_row = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,USD,GrantEquivalent,"
        f"P1Y,OECD_DAC2A,2015,2000,,facet2\r\n"
    )
    csv_body = _CSV_HEADER + tgo_row + ken_row
    sdmx, _requests = _sdmx_client(csv_body)
    facets = [
        Facet(
            facet_id="facet1",
            provenance_id="OECD_DAC2A",
            measurement_method="GrantEquivalent",
            unit="USD",
            obs_count=1,
            earliest_date="2015",
            latest_date="2015",
        ),
        Facet(
            facet_id="facet2",
            provenance_id="OECD_DAC2A",
            measurement_method="GrantEquivalent",
            unit="USD",
            obs_count=1,
            earliest_date="2015",
            latest_date="2015",
        ),
    ]
    graph = FakeGraphReader(
        facets={
            (_BARE_STAT_VAR, "country/TGO"): facets,
            (_BARE_STAT_VAR, "country/KEN"): facets,
        }
    )
    coverage = _coverage_for(facets, entities=["country/TGO", "country/KEN"])
    request = _request(
        role_bindings={"recipient": ["country/TGO", "country/KEN"]},
        window=RequestedWindow(kind="all"),
    )

    result = materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=10)

    assert result.total_rows == 2
    assert len(result.frame.rows) == 2
    assert result.coverage.rows == 2
    assert result.warnings == ()


def test_two_facets_on_the_same_identity_but_disjoint_years_both_survive():
    # Different years keep both same-recipient rows and widen the applied window.
    row_2015 = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/TGO,USD,GrantEquivalent,"
        f"P1Y,OECD_DAC2A,2015,1000,,facet1\r\n"
    )
    row_2016 = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/TGO,USD,GrantEquivalent,"
        f"P1Y,OECD_DAC2A,2016,2000,,facet2\r\n"
    )
    csv_body = _CSV_HEADER + row_2015 + row_2016
    sdmx, _requests = _sdmx_client(csv_body)
    facets = [
        Facet(
            facet_id="facet1",
            provenance_id="OECD_DAC2A",
            measurement_method="GrantEquivalent",
            unit="USD",
            obs_count=1,
            earliest_date="2015",
            latest_date="2015",
        ),
        Facet(
            facet_id="facet2",
            provenance_id="OECD_DAC2A",
            measurement_method="GrantEquivalent",
            unit="USD",
            obs_count=1,
            earliest_date="2016",
            latest_date="2016",
        ),
    ]
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): facets})
    coverage = _coverage_for(facets, entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )

    result = materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=10)

    assert result.total_rows == 2
    assert len(result.frame.rows) == 2
    assert set(_time_periods(result.frame)) == {"2015", "2016"}
    assert result.coverage.window is not None
    assert result.coverage.window["start_year"] == 2015
    assert result.coverage.window["end_year"] == 2016
    assert result.warnings == ()


def test_two_facets_with_nfc_variant_recipients_collapse_via_the_normalised_grain():
    # Role names match despite different NFC and NFD recipient spellings.
    nfc_recipient = f"country/{unicodedata.normalize('NFC', 'Réunion')}"
    nfd_recipient = f"country/{unicodedata.normalize('NFD', 'Réunion')}"
    assert nfc_recipient != nfd_recipient  # sanity: genuinely different raw strings

    # Distinct values identify the surviving row because `Frame.build` normalises both recipients.
    nfc_row = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{nfc_recipient},USD,GrantEquivalent,"
        f"P1Y,OECD_DAC2A,2015,1000,,facet1\r\n"
    )
    nfd_row = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{nfd_recipient},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,2015,2000,,facet2\r\n"
    )
    csv_body = _CSV_HEADER + nfc_row + nfd_row
    sdmx, _requests = _sdmx_client(csv_body)
    facet_with_period = Facet(
        facet_id="facet1",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    facet_without_period = Facet(
        facet_id="facet2",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    graph = FakeGraphReader(
        facets={
            (_BARE_STAT_VAR, nfc_recipient): [facet_with_period, facet_without_period],
            (_BARE_STAT_VAR, nfd_recipient): [facet_with_period, facet_without_period],
        }
    )
    coverage = _coverage_for(
        [facet_with_period, facet_without_period], entities=[nfc_recipient, nfd_recipient]
    )
    request = _request(
        role_bindings={"recipient": [nfc_recipient, nfd_recipient]},
        window=RequestedWindow(kind="all"),
    )

    result = materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=10)

    assert result.total_rows == 1
    assert len(result.frame.rows) == 1
    recipient_index = result.frame.columns.index("recipient")
    assert result.frame.rows[0][recipient_index] == unicodedata.normalize("NFC", nfc_recipient)
    # `OBS_VALUE` 1000 identifies facet1 after recipient normalisation.
    value_index = result.frame.columns.index("value")
    assert result.frame.rows[0][value_index] == 1000.0


def test_two_facets_with_bare_and_prefixed_variable_collapse_via_the_canonicalised_grain():
    # The response mixes bare and prefixed `variableMeasured` values for one observation.
    prefixed_row = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/TGO,USD,GrantEquivalent,"
        f"P1Y,OECD_DAC2A,2015,1000,,facet1\r\n"
    )
    bare_row = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_BARE_STAT_VAR},country/TGO,USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,2015,1000,,facet2\r\n"
    )
    csv_body = _CSV_HEADER + prefixed_row + bare_row
    sdmx, _requests = _sdmx_client(csv_body)
    facet_with_period = Facet(
        facet_id="facet1",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    facet_without_period = Facet(
        facet_id="facet2",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    graph = FakeGraphReader(
        facets={(_BARE_STAT_VAR, "country/TGO"): [facet_with_period, facet_without_period]}
    )
    coverage = _coverage_for([facet_with_period, facet_without_period], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )

    result = materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=10)

    assert result.total_rows == 1
    assert len(result.frame.rows) == 1


# ---------------------------------------------------------------------------
# `content_version` (#662): a preview minted under a `RevisionReader` reports
# the same version a mint of the same request reports, at any `limit`.
# ---------------------------------------------------------------------------


def _single_entity_setup(
    *, rows: int
) -> tuple[SdmxClient, FakeGraphReader, CoverageIndex, MaterialisationRequest]:
    csv_body = _CSV_HEADER + "".join(
        _csv_row(recipient="country/TGO", year=str(2000 + i), value="1000") for i in range(rows)
    )
    sdmx, _requests = _sdmx_client(csv_body)
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET], entities=["country/TGO"])
    request = _request(
        role_bindings={"recipient": ["country/TGO"]}, window=RequestedWindow(kind="all")
    )
    return sdmx, graph, coverage, request


def _spec_version(
    sdmx: SdmxClient,
    graph: FakeGraphReader,
    coverage: CoverageIndex,
    request: MaterialisationRequest,
) -> str:
    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        materialiser_build="test-build",
        capability_version="cap-v1",
        enrichment=EnrichmentLookup(graph, type_kinds={}, parties={}),
        coverage=coverage,
    )
    return result.content_version


def test_preview_content_version_equals_the_mint_version_for_the_same_request():
    sdmx, graph, coverage, request = _single_entity_setup(rows=2)

    mint_version = _spec_version(sdmx, graph, coverage, request)
    preview = materialise_preview(
        request, sdmx=sdmx, graph=graph, coverage=coverage, with_content_version=True, limit=10
    )

    assert preview.content_version is not None
    assert preview.content_version.startswith("cv_")
    assert preview.content_version == mint_version


def test_preview_content_version_equals_the_mint_version_when_truncated():
    """The token uses all survivors and is independent of the delivered row limit."""
    sdmx, graph, coverage, request = _single_entity_setup(rows=10)

    mint_version = _spec_version(sdmx, graph, coverage, request)
    truncated_preview = materialise_preview(
        request, sdmx=sdmx, graph=graph, coverage=coverage, with_content_version=True, limit=3
    )

    assert truncated_preview.truncated is True
    assert truncated_preview.total_rows == 10
    assert len(truncated_preview.frame.rows) == 3
    assert truncated_preview.content_version == mint_version


def test_preview_content_version_holds_across_two_different_limits():
    """The token stays fixed across preview limits."""
    sdmx, graph, coverage, request = _single_entity_setup(rows=10)

    narrow = materialise_preview(
        request, sdmx=sdmx, graph=graph, coverage=coverage, with_content_version=True, limit=1
    )
    wide = materialise_preview(
        request, sdmx=sdmx, graph=graph, coverage=coverage, with_content_version=True, limit=10
    )

    assert narrow.content_version == wide.content_version


def test_a_read_past_twenty_times_limit_counts_every_row_and_mints_the_mint_version():
    """`limit` bounds the sample only, so a member 25 times `limit` previews whole."""
    sdmx, graph, coverage, request = _single_entity_setup(rows=25)

    mint_version = _spec_version(sdmx, graph, coverage, request)
    preview = materialise_preview(
        request, sdmx=sdmx, graph=graph, coverage=coverage, with_content_version=True, limit=1
    )

    assert preview.total_rows == 25
    assert len(preview.frame.rows) == 1
    assert preview.content_version == mint_version


def test_a_read_past_twenty_times_ceiling_raises_the_sdmx_cap():
    sdmx, graph, coverage, request = _single_entity_setup(rows=25)

    with pytest.raises(SdmxCapExceededError) as capped:
        materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=10, ceiling=1)

    assert (capped.value.cap_kind, capped.value.cap) == ("rows", 20)


def test_a_preview_counts_every_survivor_past_its_ceiling():
    """Search passes its `limit` as `ceiling` and truncates, so survivors past it are counted."""
    sdmx, graph, coverage, request = _single_entity_setup(rows=25)

    preview = materialise_preview(
        request, sdmx=sdmx, graph=graph, coverage=coverage, limit=2, ceiling=2
    )

    assert (preview.total_rows, preview.truncated) == (25, True)


def test_preview_without_with_content_version_reports_no_content_version():
    sdmx, graph, coverage, request = _single_entity_setup(rows=2)

    preview = materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=10)

    assert preview.content_version is None
