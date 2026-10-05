"""Pin SDMX HTTP-error mapping, transport failures, and upstream status on malformed 2xx bodies."""

from __future__ import annotations

import json

import httpx
import pytest

from dc_sdmx.client import SdmxClient
from dc_sdmx.errors import (
    SdmxContainmentFilterError,
    SdmxHttpError,
    SdmxIncompatibleVariablesError,
    SdmxMissingFilterError,
    SdmxResponseError,
    SdmxRoleCeilingError,
    SdmxTransportError,
    SdmxUnexpectedResponseShapeError,
    SdmxUnsupportedFilterError,
)

from .conftest import read_fixture_bytes, read_fixture_text

_BASE_URL = "https://datacommons.one.org/core/api"


def _client_raising(
    body: str, *, status: int, content_type: str = "application/json"
) -> SdmxClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body, headers={"content-type": content_type})

    return SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    ("fixture", "status", "error_cls"),
    [
        ("missing_filter.json", 400, SdmxMissingFilterError),
        ("unsupported_filter_data.json", 501, SdmxUnsupportedFilterError),
        ("unsupported_availability_component.json", 501, SdmxUnsupportedFilterError),
        ("containment_filter.json", 400, SdmxContainmentFilterError),
        ("incompatible_variables.json", 400, SdmxIncompatibleVariablesError),
        ("role_ceiling.json", 400, SdmxRoleCeilingError),
    ],
)
def test_typed_error_mapping(fixture, status, error_cls):
    client = _client_raising(read_fixture_text(f"errors/{fixture}"), status=status)
    with pytest.raises(error_cls) as excinfo:
        client.data(variables=["Count_Person"], entities={"observationAbout": ["country/USA"]})
    assert excinfo.value.upstream_status == status
    assert excinfo.value.message  # the transcoded message was extracted, not swallowed


def test_unmatched_message_falls_back_to_generic_http_error():
    client = _client_raising(read_fixture_text("errors/generic_500.json"), status=500)
    with pytest.raises(SdmxHttpError) as excinfo:
        client.data(variables=["Count_Person"], entities={"observationAbout": ["country/USA"]})
    assert type(excinfo.value) is SdmxHttpError
    assert excinfo.value.upstream_status == 500
    assert excinfo.value.message == "internal error"


def test_non_json_error_body_still_raises_with_raw_text():
    body = read_fixture_text("errors/non_json_error.txt")
    client = _client_raising(body, status=502, content_type="text/plain")
    with pytest.raises(SdmxHttpError) as excinfo:
        client.data(variables=["Count_Person"], entities={"observationAbout": ["country/USA"]})
    assert excinfo.value.upstream_status == 502
    assert excinfo.value.message == body


def test_html_challenge_body_raises_unexpected_response_shape_error():
    # A Cloudflare WAF challenge is HTML, not the mixer's JSON error envelope.
    body = "<html><head><title>Just a moment...</title></head><body>Checking your browser</body></html>"
    client = _client_raising(body, status=403, content_type="text/html; charset=UTF-8")
    with pytest.raises(SdmxUnexpectedResponseShapeError) as excinfo:
        client.data(variables=["Count_Person"], entities={"observationAbout": ["country/USA"]})
    assert excinfo.value.upstream_status == 403
    assert "403" in excinfo.value.message
    assert "text/html; charset=UTF-8" in excinfo.value.message
    assert "Just a moment" in excinfo.value.message


def test_transport_error_raises_sdmx_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler))
    with pytest.raises(SdmxTransportError):
        client.data(variables=["Count_Person"], entities={"observationAbout": ["country/USA"]})


def test_availability_error_mapping_uses_same_hierarchy():
    client = _client_raising(read_fixture_text("errors/missing_filter.json"), status=400)
    with pytest.raises(SdmxMissingFilterError):
        client.availability(variables=["Count_Person"], component="observationAbout")


def test_data_200_with_non_csv_content_type_raises_response_error():
    # Live-verified deployment quirk; see `SdmxResponseError`'s docstring.
    body = read_fixture_text("data_empty_json_body.json")
    client = _client_raising(body, status=200, content_type="application/json")
    with pytest.raises(SdmxResponseError) as excinfo:
        client.data(variables=["Count_Person"], entities={"observationAbout": ["country/USA"]})
    assert excinfo.value.upstream_status == 200


def test_data_200_with_csv_content_type_parses_normally():
    client = _client_raising(
        read_fixture_text("data_header_only.csv"),
        status=200,
        content_type="application/vnd.sdmx.data+csv;version=2.0.0",
    )
    result = client.data(variables=["Count_Person"], entities={"observationAbout": ["country/USA"]})
    assert result.roles == ("observationAbout",)


def test_availability_200_with_non_json_body_raises_response_error():
    client = _client_raising("not json at all", status=200, content_type="application/json")
    with pytest.raises(SdmxResponseError) as excinfo:
        client.availability(variables=["Count_Person"], component="observationAbout")
    assert excinfo.value.upstream_status == 200


def test_availability_200_with_non_object_json_body_raises_response_error():
    # `_json_body` rejects the array before `parse_availability_json` runs.
    client = _client_raising("[1, 2, 3]", status=200, content_type="application/json")
    with pytest.raises(SdmxResponseError, match="JSON body is not an object") as excinfo:
        client.availability(variables=["Count_Person"], component="observationAbout")
    assert excinfo.value.upstream_status == 200


def test_data_200_with_malformed_csv_header_sets_upstream_status():
    # The parser has no response status; the client attaches the real 200.
    client = _client_raising(
        "not,a,valid,header\r\n",
        status=200,
        content_type="application/vnd.sdmx.data+csv;version=2.0.0",
    )
    with pytest.raises(SdmxResponseError, match="unexpected SDMX-CSV header") as excinfo:
        client.data(variables=["Count_Person"], entities={"observationAbout": ["country/USA"]})
    assert excinfo.value.upstream_status == 200


def test_data_streamed_malformed_utf8_row_raises_response_error():
    # Malformed UTF-8 must become `SdmxResponseError`, not `UnicodeDecodeError`.
    body = read_fixture_bytes("data_header_only.csv") + b"\xff\xfe invalid\r\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler))
    with pytest.raises(SdmxResponseError) as excinfo:
        client.data(variables=["Count_Person"], entities={"observationAbout": ["country/USA"]})
    assert excinfo.value.upstream_status == 200


def test_data_streamed_malformed_utf8_row_past_header_chunk_raises_response_error():
    # TextIOWrapper reads about 8 KB at a time; this row tests status handling
    # when malformed bytes reach `_capped_rows`, after header parsing.
    valid_row = read_fixture_bytes("data_single_entity.csv").splitlines()[1] + b"\r\n"
    padding = valid_row * (16 * 1024 // len(valid_row) + 1)
    body = read_fixture_bytes("data_header_only.csv") + padding + b"\xff\xfe invalid\r\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler))
    with pytest.raises(SdmxResponseError) as excinfo:
        client.data(variables=["Count_Person"], entities={"observationAbout": ["country/USA"]})
    assert excinfo.value.upstream_status == 200


@pytest.mark.parametrize(
    "body",
    [
        {"data": {"dataConstraints": None}},
        {"data": {"dataConstraints": [{"cubeRegions": None}]}},
        {"data": {"dataConstraints": [{"cubeRegions": [{"keyValues": None}]}]}},
        {
            "data": {
                "dataConstraints": [
                    {"cubeRegions": [{"keyValues": [{"id": "observationAbout", "values": None}]}]}
                ]
            }
        },
    ],
    ids=["dataConstraints", "cubeRegions", "keyValues", "values"],
)
def test_availability_200_with_malformed_nested_shape_sets_upstream_status(body):
    # Exercise each malformed level through the client to pin status attachment.
    client = _client_raising(json.dumps(body), status=200, content_type="application/json")
    with pytest.raises(SdmxResponseError) as excinfo:
        client.availability(variables=["Count_Person"], component="observationAbout")
    assert excinfo.value.upstream_status == 200


@pytest.mark.parametrize(
    "body",
    [{}, {"message": "upstream failure"}, {"data": {}}],
    ids=["empty-body", "error-envelope", "data-with-no-dataConstraints"],
)
def test_availability_200_missing_required_field_sets_upstream_status(body):
    # The mixer includes both fields even for empty results; missing either
    # means a malformed response, including a 2xx error envelope.
    client = _client_raising(json.dumps(body), status=200, content_type="application/json")
    with pytest.raises(SdmxResponseError, match="missing required field") as excinfo:
        client.availability(variables=["Count_Person"], component="observationAbout")
    assert excinfo.value.upstream_status == 200


def test_availability_200_empty_golden_still_parses_successfully():
    # Empty results retain `dataConstraints` and omit `cubeRegions`.
    client = _client_raising(
        read_fixture_text("availability_time_period_empty.json"),
        status=200,
        content_type="application/json",
    )
    result = client.availability(variables=["Count_Person"], component="observationAbout")
    assert result.components == {}


@pytest.mark.parametrize(
    "body",
    [
        # keyValues: null inside an include=false region -- validated (and
        # must raise) BEFORE the include=false diagnostic ever inspects it.
        {"data": {"dataConstraints": [{"cubeRegions": [{"include": False, "keyValues": None}]}]}},
        # id present and truthy, but not a string.
        {
            "data": {
                "dataConstraints": [
                    {"cubeRegions": [{"keyValues": [{"id": ["x"], "values": ["country/USA"]}]}]}
                ]
            }
        },
        # a non-string, unhashable member of values.
        {
            "data": {
                "dataConstraints": [
                    {"cubeRegions": [{"keyValues": [{"id": "observationAbout", "values": [{}]}]}]}
                ]
            }
        },
        # a non-string, hashable member of values, after a valid one.
        {
            "data": {
                "dataConstraints": [
                    {
                        "cubeRegions": [
                            {"keyValues": [{"id": "observationAbout", "values": ["ok", 42]}]}
                        ]
                    }
                ]
            }
        },
    ],
    ids=[
        "null-keyValues-in-excluded-region",
        "non-string-id",
        "unhashable-value",
        "hashable-non-string-value",
    ],
)
def test_availability_200_with_malformed_scalar_sets_upstream_status(body):
    client = _client_raising(json.dumps(body), status=200, content_type="application/json")
    with pytest.raises(SdmxResponseError) as excinfo:
        client.availability(variables=["Count_Person"], component="observationAbout")
    assert excinfo.value.upstream_status == 200
