"""Pin `data()` row and byte caps on streamed and pre-buffered responses."""

from __future__ import annotations

import httpx
import pytest

from dc_sdmx.client import SdmxClient
from dc_sdmx.errors import SdmxCapExceededError
from dc_sdmx.results import parse_data_csv

_HEADER = (
    b"STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,"
    b"measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
    b"scalingFactor,facetId\r\n"
)
_BASE_URL = "https://datacommons.one.org/core/api"


def _row(year: int) -> bytes:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/USA,Person,Census,P1Y,"
        f"dc/base,{year},1.50,0,stored-facet-id\r\n"
    ).encode()


def _row_with_embedded_newline() -> bytes:
    return (
        b'dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/USA,"Persons\ncounted",'
        b"Census,P1Y,dc/base,2020,1.50,0,stored-facet-id\r\n"
    )


def _body(row_count: int) -> bytes:
    return _HEADER + b"".join(_row(2000 + i) for i in range(row_count))


def _client(transport) -> SdmxClient:
    return SdmxClient(base_url=_BASE_URL, transport=transport)


def test_max_rows_none_is_uncapped(streaming_transport):
    transport = streaming_transport(_body(50))
    result = _client(transport).data(
        variables=["Count_Person"], entities={"observationAbout": ["country/USA"]}
    )
    assert len(result.rows) == 50


def test_within_both_caps_succeeds(streaming_transport):
    transport = streaming_transport(_body(5))
    result = _client(transport).data(
        variables=["Count_Person"],
        entities={"observationAbout": ["country/USA"]},
        max_rows=5,
        max_bytes=len(_body(5)),
    )
    assert len(result.rows) == 5


def test_row_cap_exceeded_raises_and_aborts_early(streaming_transport):
    transport = streaming_transport(_body(20))
    with pytest.raises(SdmxCapExceededError) as excinfo:
        _client(transport).data(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            max_rows=3,
        )
    err = excinfo.value
    assert err.cap_kind == "rows"
    assert err.cap == 3
    assert err.observed == 4
    # The stream aborted well before every chunk of a 20-row body was pulled.
    assert transport.chunks_yielded < transport.total_chunks


def test_byte_cap_exceeded_raises_and_aborts_early(streaming_transport):
    body = _body(20)
    small_cap = len(_HEADER) + len(_row(2000))  # room for the header + ~1 row
    transport = streaming_transport(body, chunk_size=8)
    with pytest.raises(SdmxCapExceededError) as excinfo:
        _client(transport).data(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            max_bytes=small_cap,
        )
    err = excinfo.value
    assert err.cap_kind == "bytes"
    assert err.cap == small_cap
    assert err.observed > small_cap
    assert transport.chunks_yielded < transport.total_chunks


def test_byte_cap_trips_on_header_only_response(streaming_transport):
    # The byte cap includes the header, even when the response has no rows.
    transport = streaming_transport(_body(0), chunk_size=8)
    small_cap = len(_HEADER) // 2
    with pytest.raises(SdmxCapExceededError) as excinfo:
        _client(transport).data(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            max_bytes=small_cap,
        )
    err = excinfo.value
    assert err.cap_kind == "bytes"
    assert err.cap == small_cap
    assert err.observed > small_cap
    assert transport.chunks_yielded < transport.total_chunks


def test_byte_cap_trips_on_pre_buffered_response(recording_transport):
    # Buffered `httpx.Response` objects leave `num_bytes_downloaded` at zero.
    body = _body(20)
    response = httpx.Response(
        200,
        headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        content=body,
    )
    transport = recording_transport(response)
    small_cap = len(_HEADER) + len(_row(2000))
    with pytest.raises(SdmxCapExceededError) as excinfo:
        _client(transport).data(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            max_bytes=small_cap,
        )
    assert excinfo.value.cap_kind == "bytes"


def test_byte_cap_trips_mid_oversized_record(streaming_transport):
    # The cap must trip mid-record, before the oversized record is buffered.
    body = _HEADER + _row(2000)
    small_cap = len(_HEADER) + 5
    transport = streaming_transport(body, chunk_size=8)
    with pytest.raises(SdmxCapExceededError) as excinfo:
        _client(transport).data(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            max_bytes=small_cap,
        )
    err = excinfo.value
    assert err.cap_kind == "bytes"
    assert err.observed > small_cap
    assert transport.chunks_yielded < transport.total_chunks


@pytest.mark.parametrize("status_code", [400, 500])
def test_byte_cap_aborts_oversized_non_success_body(streaming_transport, status_code):
    """The eager row API must cap a typed error body before mapping it."""
    body = b'{"message":"upstream failure ' + b"x" * 2000 + b'"}'
    transport = streaming_transport(
        body,
        status_code=status_code,
        headers={"content-type": "application/json"},
        chunk_size=8,
    )

    with pytest.raises(SdmxCapExceededError) as excinfo:
        _client(transport).data(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            max_bytes=50,
        )

    assert excinfo.value.cap_kind == "bytes"
    assert excinfo.value.cap == 50
    assert excinfo.value.observed > 50
    assert transport.chunks_yielded < transport.total_chunks


def test_streamed_quoted_embedded_newline_matches_eager_parse(streaming_transport):
    # Five-byte chunks split the quoted field containing the newline.
    body = _HEADER + _row_with_embedded_newline()
    transport = streaming_transport(body, chunk_size=5)
    streamed = _client(transport).data(
        variables=["Count_Person"], entities={"observationAbout": ["country/USA"]}
    )
    eager = parse_data_csv(body.decode())
    assert streamed.rows == eager.rows
    assert streamed.rows[0].unit == "Persons\ncounted"
