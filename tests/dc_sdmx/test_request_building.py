"""Pin SDMX request paths, query strings, and headers with a recording transport."""

from __future__ import annotations

import httpx
import pytest

from dc_sdmx.client import ContainedIn, SdmxClient
from dc_sdmx.errors import SdmxQueryError, SdmxUnsupportedFilterError

from .conftest import read_fixture_text

_BASE_URL = "https://datacommons.one.org/core/api"
_DEFAULT_USER_AGENT = "dfo-apis (+https://github.com/ONEcampaign/dfo-apis)"
_CSV_RESPONSE = httpx.Response(
    200,
    text=read_fixture_text("data_header_only.csv"),
    headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
)
_AVAILABILITY_RESPONSE = httpx.Response(
    200, text=read_fixture_text("availability_observation_about.json")
)


def _client(transport: httpx.BaseTransport, *, token: str | None = None) -> SdmxClient:
    return SdmxClient(base_url=_BASE_URL, token=token, transport=transport)


def test_data_single_entity_url_and_headers(recording_transport):
    transport = recording_transport(_CSV_RESPONSE)
    _client(transport).data(
        variables=["Count_Person"], entities={"observationAbout": ["country/USA"]}
    )

    request = transport.last_request
    assert request.method == "GET"
    assert (
        str(request.url) == f"{_BASE_URL}/sdmx/v3/data/dataflow/DC/DF_OBS/1.0.0/*"
        "?c%5BvariableMeasured%5D=Count_Person"
        "&c%5BobservationAbout%5D=country%2FUSA"
    )
    assert request.headers["accept"] == "application/vnd.sdmx.data+csv;version=2.0.0"
    assert "x-graph-token" not in request.headers


def test_data_sends_token_header_when_given(recording_transport):
    transport = recording_transport(_CSV_RESPONSE)
    _client(transport, token="secret-token").data(
        variables=["Count_Person"], entities={"observationAbout": ["country/USA"]}
    )
    assert transport.last_request.headers["x-graph-token"] == "secret-token"


def test_data_sends_default_user_agent(recording_transport):
    transport = recording_transport(_CSV_RESPONSE)
    _client(transport).data(
        variables=["Count_Person"], entities={"observationAbout": ["country/USA"]}
    )
    assert transport.last_request.headers["user-agent"] == _DEFAULT_USER_AGENT


def test_data_honors_user_agent_override(recording_transport):
    transport = recording_transport(_CSV_RESPONSE)
    SdmxClient(base_url=_BASE_URL, transport=transport, user_agent="custom-ua/1.0").data(
        variables=["Count_Person"], entities={"observationAbout": ["country/USA"]}
    )
    assert transport.last_request.headers["user-agent"] == "custom-ua/1.0"


def test_data_multi_role_entities(recording_transport):
    transport = recording_transport(_CSV_RESPONSE)
    _client(transport).data(
        variables=["Count_Person"],
        entities={"donor": ["country/USA"], "recipient": ["country/KEN"]},
    )
    url = str(transport.last_request.url)
    assert "c%5Bdonor%5D=country%2FUSA" in url
    assert "c%5Brecipient%5D=country%2FKEN" in url


def test_data_contained_in_expands_to_property_pair(recording_transport):
    transport = recording_transport(_CSV_RESPONSE)
    _client(transport).data(
        variables=["Count_Person"],
        entities={"observationAbout": ContainedIn(parent="africa", child_type="Country")},
    )
    url = str(transport.last_request.url)
    assert "c%5BobservationAbout.containedInPlace%2B%5D=africa" in url
    assert "c%5BobservationAbout.typeOf%5D=Country" in url


@pytest.mark.parametrize(
    ("data_kwargs", "expected_fragment"),
    [
        pytest.param(
            {
                "variables": ["Count_Person"],
                "entities": {"observationAbout": ["country/USA"]},
                "facet_ids": ["facet-1", "facet-2"],
            },
            "c%5BfacetId%5D=facet-1%2Cfacet-2",
            id="facet_ids_or_list",
        ),
        pytest.param(
            {
                "variables": ["Count_Person"],
                "entities": {"observationAbout": ["country/KEN", "country/NGA"]},
            },
            "c%5BobservationAbout%5D=country%2FKEN%2Ccountry%2FNGA",
            id="entity_or_list",
        ),
        pytest.param(
            {
                "variables": ["Count_Person"],
                "entities": {"observationAbout": ["country/USA"]},
                "time_period": "LATEST",
            },
            "c%5BTIME_PERIOD%5D=LATEST",
            id="time_period_latest",
        ),
        pytest.param(
            {
                "variables": ["Count_Person"],
                "entities": {"observationAbout": ["country/USA"]},
                "time_period": ["2020", "2021"],
            },
            "c%5BTIME_PERIOD%5D=2020%2C2021",
            id="time_period_explicit_or_list",
        ),
    ],
)
def test_data_request_filters(recording_transport, data_kwargs, expected_fragment):
    transport = recording_transport(_CSV_RESPONSE)
    _client(transport).data(**data_kwargs)
    assert expected_fragment in str(transport.last_request.url)


def test_availability_url_and_headers(recording_transport):
    transport = recording_transport(_AVAILABILITY_RESPONSE)
    _client(transport).availability(variables=["Count_Person"], component="observationAbout")

    request = transport.last_request
    assert (
        str(request.url)
        == f"{_BASE_URL}/sdmx/v3/availability/dataflow/DC/DF_OBS/1.0.0/*/observationAbout"
        "?c%5BvariableMeasured%5D=Count_Person"
    )
    assert request.headers["accept"] == "application/vnd.sdmx.structure+json;version=2.0.0"
    assert request.headers["user-agent"] == _DEFAULT_USER_AGENT


def test_availability_component_is_required(recording_transport):
    # The mixer rejects SDMX's `all` wildcard, so callers must name a component.
    transport = recording_transport(_AVAILABILITY_RESPONSE)
    with pytest.raises(TypeError):
        _client(transport).availability(variables=["Count_Person"])  # ty: ignore[missing-argument]
    assert transport.requests == []


def test_availability_allows_empty_filters(recording_transport):
    transport = recording_transport(_AVAILABILITY_RESPONSE)
    # The mixer requires only `variableMeasured` for availability.
    _client(transport).availability(
        variables=["Count_Person"], component="observationAbout", filters={}
    )
    assert transport.requests


def test_availability_with_filters(recording_transport):
    transport = recording_transport(_AVAILABILITY_RESPONSE)
    _client(transport).availability(
        variables=["Count_Person"],
        component="observationAbout",
        filters={"observationAbout": ["country/USA"]},
    )
    assert "c%5BobservationAbout%5D=country%2FUSA" in str(transport.last_request.url)


def test_availability_filters_rejects_variable_measured_key(recording_transport):
    transport = recording_transport(_AVAILABILITY_RESPONSE)
    with pytest.raises(SdmxQueryError, match="variableMeasured"):
        _client(transport).availability(
            variables=["Count_Person"],
            component="observationAbout",
            filters={"variableMeasured": ["Count_Person"]},
        )
    assert transport.requests == []


def test_availability_time_period_exact_value(recording_transport):
    transport = recording_transport(_AVAILABILITY_RESPONSE)
    _client(transport).availability(
        variables=["Count_Person"], component="observationAbout", time_period="2020"
    )
    assert "c%5BTIME_PERIOD%5D=2020" in str(transport.last_request.url)


def test_availability_time_period_or_list_is_comma_joined_and_deduplicated(recording_transport):
    transport = recording_transport(_AVAILABILITY_RESPONSE)
    _client(transport).availability(
        variables=["Count_Person"],
        component="observationAbout",
        time_period=["2020", "2021", "2020"],
    )
    assert "c%5BTIME_PERIOD%5D=2020%2C2021" in str(transport.last_request.url)


@pytest.mark.parametrize(
    "bad_time_period",
    [
        "LATEST",
        ["LATEST"],
        [],
        "ge:2020",
        ["ge:2020"],
        "gt:2020",
        "le:2020",
        "lt:2020",
        "ne:2020",
        "",
        "   ",
        [""],
        ["   "],
        ["2020", ""],
    ],
)
def test_availability_time_period_unsupported_forms_rejected_before_http(
    recording_transport, bad_time_period
):
    transport = recording_transport(_AVAILABILITY_RESPONSE)
    with pytest.raises(SdmxUnsupportedFilterError):
        _client(transport).availability(
            variables=["Count_Person"],
            component="observationAbout",
            time_period=bad_time_period,
        )
    assert transport.requests == []


def test_invalid_component_id_rejected_before_http(recording_transport):
    transport = recording_transport(_AVAILABILITY_RESPONSE)
    with pytest.raises(SdmxQueryError):
        _client(transport).availability(variables=["Count_Person"], component="bad id")
    assert transport.requests == []
