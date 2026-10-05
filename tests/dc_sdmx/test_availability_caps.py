"""Pin `/availability` limits: bytes during streaming, values in the requested component."""

from __future__ import annotations

import json

import httpx
import pytest

from dc_sdmx.client import SdmxClient
from dc_sdmx.errors import SdmxCapExceededError

_BASE_URL = "https://datacommons.one.org/core/api"


def _availability_body(values: list[str], *, component: str = "observationAbout") -> bytes:
    body = {
        "data": {
            "dataConstraints": [
                {"cubeRegions": [{"keyValues": [{"id": component, "values": list(values)}]}]}
            ]
        }
    }
    return json.dumps(body).encode("utf-8")


def _multi_component_body(component_values: dict[str, list[str]]) -> bytes:
    body = {
        "data": {
            "dataConstraints": [
                {
                    "cubeRegions": [
                        {
                            "keyValues": [
                                {"id": component, "values": list(values)}
                                for component, values in component_values.items()
                            ]
                        }
                    ]
                }
            ]
        }
    }
    return json.dumps(body).encode("utf-8")


def _client(transport: httpx.BaseTransport) -> SdmxClient:
    return SdmxClient(base_url=_BASE_URL, transport=transport)


def test_max_values_none_is_uncapped(streaming_transport):
    values = [f"country/E{i}" for i in range(50)]
    body = _availability_body(values)
    transport = streaming_transport(body, headers={"content-type": "application/json"})
    result = _client(transport).availability(
        variables=["Count_Person"], component="observationAbout"
    )
    assert len(result.components["observationAbout"]) == 50


def test_within_both_caps_succeeds(streaming_transport):
    values = [f"country/E{i}" for i in range(5)]
    body = _availability_body(values)
    transport = streaming_transport(body, headers={"content-type": "application/json"})
    result = _client(transport).availability(
        variables=["Count_Person"],
        component="observationAbout",
        max_values=5,
        max_bytes=len(body),
    )
    assert len(result.components["observationAbout"]) == 5


def test_max_values_counts_only_the_requested_component(streaming_transport):
    # The requested 10 fit cap 50; counting the other 45 would trip it.
    requested = [f"country/E{i}" for i in range(10)]
    echoed = [f"20{i:02d}" for i in range(45)]
    body = _multi_component_body({"observationAbout": requested, "TIME_PERIOD": echoed})
    transport = streaming_transport(body, headers={"content-type": "application/json"})
    result = _client(transport).availability(
        variables=["Count_Person"], component="observationAbout", max_values=50
    )
    assert list(result.components["observationAbout"]) == requested


def test_max_values_raises_when_requested_component_alone_exceeds_cap(streaming_transport):
    # The requested 20 values exceed the cap of 3; the echoed value is excluded.
    requested = [f"country/E{i}" for i in range(20)]
    body = _multi_component_body({"observationAbout": requested, "TIME_PERIOD": ["2020"]})
    transport = streaming_transport(body, headers={"content-type": "application/json"})
    with pytest.raises(SdmxCapExceededError) as excinfo:
        _client(transport).availability(
            variables=["Count_Person"], component="observationAbout", max_values=3
        )
    err = excinfo.value
    assert err.cap_kind == "values"
    assert err.cap == 3
    assert err.observed == 20


def test_byte_cap_exceeded_raises_and_aborts_early(streaming_transport):
    values = [f"country/E{i}" for i in range(200)]
    body = _availability_body(values)
    small_cap = 50
    transport = streaming_transport(
        body, headers={"content-type": "application/json"}, chunk_size=8
    )
    with pytest.raises(SdmxCapExceededError) as excinfo:
        _client(transport).availability(
            variables=["Count_Person"], component="observationAbout", max_bytes=small_cap
        )
    err = excinfo.value
    assert err.cap_kind == "bytes"
    assert err.cap == small_cap
    assert err.observed > small_cap
    assert transport.chunks_yielded < transport.total_chunks


def test_byte_cap_trips_on_pre_buffered_response(recording_transport):
    # Pre-buffered responses leave `num_bytes_downloaded` at zero, so the cap
    # must count the bytes served by `iter_bytes()` too.
    values = [f"country/E{i}" for i in range(200)]
    body = _availability_body(values)
    response = httpx.Response(200, headers={"content-type": "application/json"}, content=body)
    transport = recording_transport(response)
    small_cap = 50
    with pytest.raises(SdmxCapExceededError) as excinfo:
        _client(transport).availability(
            variables=["Count_Person"], component="observationAbout", max_bytes=small_cap
        )
    assert excinfo.value.cap_kind == "bytes"


def test_byte_cap_wins_over_value_cap_when_bytes_trip_first(streaming_transport):
    # Byte limits run during streaming; value limits run after parsing.
    values = [f"country/E{i}" for i in range(200)]
    body = _availability_body(values)
    transport = streaming_transport(
        body, headers={"content-type": "application/json"}, chunk_size=8
    )
    with pytest.raises(SdmxCapExceededError) as excinfo:
        _client(transport).availability(
            variables=["Count_Person"],
            component="observationAbout",
            max_bytes=50,
            max_values=1_000_000,
        )
    assert excinfo.value.cap_kind == "bytes"


@pytest.mark.parametrize("status_code", [400, 500])
def test_byte_cap_aborts_oversized_non_success_body(streaming_transport, status_code):
    """Availability's cap applies while reading typed 4xx/5xx bodies too.

    The error payload is deliberately JSON-shaped, so this covers the path
    that would otherwise read its complete body merely to extract `message`.
    """
    body = json.dumps({"message": "upstream failure " * 200}).encode("utf-8")
    transport = streaming_transport(
        body,
        status_code=status_code,
        headers={"content-type": "application/json"},
        chunk_size=8,
    )

    with pytest.raises(SdmxCapExceededError) as excinfo:
        _client(transport).availability(
            variables=["Count_Person"], component="observationAbout", max_bytes=50
        )

    assert excinfo.value.cap_kind == "bytes"
    assert transport.chunks_yielded < transport.total_chunks
