"""Test CSV batch parsing, streaming caps, deadlines, and parity with the row reader."""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

import httpx
import pyarrow as pa
import pytest

from dc_sdmx import client as _client_module
from dc_sdmx.client import SdmxClient
from dc_sdmx.errors import SdmxCapExceededError, SdmxResponseError
from dc_sdmx.results import (
    iter_data_csv_batches,
    iter_data_csv_rows,
    observation_schema,
)
from tests.materialise._table_support import table_from_observations

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer

from .conftest import read_fixture_bytes

_BASE_URL = "https://datacommons.one.org/core/api"

_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
    "scalingFactor,facetId\r\n"
)


def _row(year: int) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/USA,Person,Census,P1Y,"
        f"dc/base,{year},1.50,0,stored-facet-id\r\n"
    )


class _RawBytes(io.RawIOBase):
    """Expose an in-memory byte body through the `RawIOBase` interface."""

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: WriteableBuffer, /) -> int:
        return self._buf.readinto(buffer)


def _batches_of(body: str, *, block_size: int = 1 << 20) -> tuple[tuple[str, ...], pa.Table]:
    roles, batches = iter_data_csv_batches(_RawBytes(body.encode()), block_size=block_size)
    table = pa.Table.from_batches(list(batches), schema=observation_schema(roles))
    return roles, table


def _row_path_error(body: str) -> str:
    with pytest.raises(SdmxResponseError) as excinfo:
        _roles, rows = iter_data_csv_rows(body.splitlines(keepends=True))
        list(rows)
    return str(excinfo.value)


def _batch_path_error(body: str) -> str:
    with pytest.raises(SdmxResponseError) as excinfo:
        _roles, batches = iter_data_csv_batches(_RawBytes(body.encode()))
        list(batches)
    return str(excinfo.value)


# --- Header errors: identical messages to the row path ---------------------


def test_malformed_header_raises_same_message_as_row_path():
    body = "not,a,valid,header\r\n"
    assert _batch_path_error(body) == _row_path_error(body)


def test_unrecognized_trailing_column_raises_same_message_as_row_path():
    body = "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,bogus\r\n"
    assert _batch_path_error(body) == _row_path_error(body)


def test_missing_required_trailing_column_raises_same_message_as_row_path():
    body = "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,OBS_VALUE\r\n"
    batch_message = _batch_path_error(body)
    assert "missing required column" in batch_message
    assert batch_message == _row_path_error(body)


def test_empty_body_raises_no_header_row():
    with pytest.raises(SdmxResponseError, match="no header row"):
        list(iter_data_csv_batches(_RawBytes(b""))[1])


# --- Header-only body -------------------------------------------------------


def test_header_only_body_yields_zero_batches():
    roles, batches = iter_data_csv_batches(_RawBytes(read_fixture_bytes("data_header_only.csv")))
    assert roles == ("observationAbout",)
    assert list(batches) == []


def test_header_only_body_table_has_the_right_columns():
    roles, batches = iter_data_csv_batches(_RawBytes(read_fixture_bytes("data_header_only.csv")))
    table = pa.Table.from_batches(list(batches), schema=observation_schema(roles))
    assert table.schema == observation_schema(roles)
    assert table.num_rows == 0


@pytest.mark.parametrize(
    "body",
    (
        _HEADER + "\r\n" + _row(2020),
        _HEADER + _row(2020) + "\r\n" + _row(2021),
        _HEADER + _row(2020) + "\r\n",
    ),
)
def test_blank_data_record_is_rejected(body: str):
    with pytest.raises(SdmxResponseError, match=r"data row .* is empty"):
        _, batches = iter_data_csv_batches(_RawBytes(body.encode()))
        list(batches)


# --- Sentinel and empty-cell mapping per column kind ------------------------


@pytest.mark.parametrize("cell", ["NotApplicable", ""])
def test_a_role_cell_naming_no_entity_maps_to_null_on_both_paths(cell: str):
    body = _HEADER + _row(2020).replace("country/USA", cell)

    _roles, table = _batches_of(body)
    _roles, rows = iter_data_csv_rows(body.splitlines(keepends=True))

    (row,) = table.to_pylist()
    assert row["observationAbout"] is None
    (observation,) = list(rows)
    assert observation.role_values == {}


def test_na_unit_cell_maps_to_null():
    _roles, table = _batches_of(
        _HEADER + _row(2020).replace(",Person,Census,", ",NotApplicable,Census,")
    )
    (row,) = table.to_pylist()
    assert row["unit"] is None


def test_empty_unit_cell_is_kept_as_empty_string():
    body = (
        _HEADER
        + "dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/USA,,Census,P1Y,dc/base,2020,1.50,0,stored-facet-id\r\n"
    )
    _roles, table = _batches_of(body)
    (row,) = table.to_pylist()
    assert row["unit"] == ""


def test_empty_facet_id_cell_maps_to_null():
    body = (
        _HEADER
        + "dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/USA,Person,Census,P1Y,dc/base,2020,1.50,0,\r\n"
    )
    _roles, table = _batches_of(body)
    (row,) = table.to_pylist()
    assert row["facetId"] is None


# --- Absent optional/required columns ---------------------------------------


def test_absent_facet_id_column_yields_all_null_column():
    roles, batches = iter_data_csv_batches(
        _RawBytes(read_fixture_bytes("data_no_facet_id_column.csv"))
    )
    table = pa.Table.from_batches(list(batches), schema=observation_schema(roles))
    assert table.column("facetId").to_pylist() == [None]


# --- Quoted embedded newline and comma --------------------------------------


def test_quoted_embedded_newline_and_comma_parse_to_one_cell():
    body = (
        _HEADER + 'dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/USA,"line1\nline1b, more",'
        "Census,P1Y,dc/base,2020,1.50,0,stored-facet-id\r\n"
    )
    _roles, table = _batches_of(body)
    (row,) = table.to_pylist()
    assert row["unit"] == "line1\nline1b, more"


def test_quoted_embedded_newline_straddling_a_block_boundary_parses_to_one_cell():
    # The quoted field crosses 64-byte blocks, pinning multiline CSV parsing.
    body = (
        _HEADER + 'dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/USA,"line1\nline1b, more",'
        "Census,P1Y,dc/base,2020,1.50,0,stored-facet-id\r\n"
    )
    _roles, table = _batches_of(body, block_size=64)
    (row,) = table.to_pylist()
    assert row["unit"] == "line1\nline1b, more"


# --- Malformed rows ----------------------------------------------------------


def test_wrong_column_count_raises_naming_the_data_row_range():
    body = _HEADER + "dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/USA\r\n"
    with pytest.raises(SdmxResponseError, match=r"in the batch after data row 0"):
        list(iter_data_csv_batches(_RawBytes(body.encode()))[1])


def test_malformed_utf8_in_a_later_chunk_raises_response_error(streaming_transport):
    # 200 valid rows place the malformed bytes past pyarrow's first read.
    padding = (_row(2020)).encode() * 200
    body = (
        _HEADER.encode()
        + padding
        + b"dataflow,DC:DF_OBS(1.0.0),I,Count_Person,\xff\xfe,Person,Census,P1Y,dc/base,2020,1.50,0,stored-facet-id\r\n"
    )
    transport = streaming_transport(body)
    client = SdmxClient(base_url=_BASE_URL, transport=transport)
    with (
        pytest.raises(SdmxResponseError, match="malformed UTF-8") as excinfo,
        client.stream_data_batches(
            variables=["Count_Person"], entities={"observationAbout": ["country/USA"]}
        ) as (_roles, batches),
    ):
        list(batches)
    assert excinfo.value.upstream_status == 200


def _truncated_in_last_field() -> bytes:
    # The cut keeps all 13 fields, so the CSV parser alone reads a whole row.
    return (_HEADER + _row(2020) + _row(2021)[:-8]).encode()


def test_batches_raise_on_a_body_cut_inside_its_last_field(streaming_transport):
    client = SdmxClient(
        base_url=_BASE_URL, transport=streaming_transport(_truncated_in_last_field())
    )
    with (
        pytest.raises(SdmxResponseError, match="ended mid-record") as excinfo,
        client.stream_data_batches(
            variables=["Count_Person"], entities={"observationAbout": ["country/USA"]}
        ) as (_roles, batches),
    ):
        list(batches)
    assert excinfo.value.upstream_status == 200


def test_rows_raise_on_a_body_cut_inside_its_last_field(streaming_transport):
    client = SdmxClient(
        base_url=_BASE_URL, transport=streaming_transport(_truncated_in_last_field())
    )
    with (
        pytest.raises(SdmxResponseError, match="ended mid-record"),
        client.stream_data(
            variables=["Count_Person"], entities={"observationAbout": ["country/USA"]}
        ) as (_roles, rows),
    ):
        list(rows)


def test_body_ending_on_a_record_boundary_reads_every_row(streaming_transport):
    body = (_HEADER + _row(2020) + _row(2021)).encode()
    client = SdmxClient(base_url=_BASE_URL, transport=streaming_transport(body))
    with client.stream_data_batches(
        variables=["Count_Person"], entities={"observationAbout": ["country/USA"]}
    ) as (roles, batches):
        table = pa.Table.from_batches(list(batches), schema=observation_schema(roles))
    assert table.num_rows == 2


# --- max_rows ----------------------------------------------------------------


def test_max_rows_within_cap_succeeds(streaming_transport):
    body = (_HEADER + "".join(_row(2000 + i) for i in range(5))).encode()
    transport = streaming_transport(body)
    client = SdmxClient(base_url=_BASE_URL, transport=transport)
    with client.stream_data_batches(
        variables=["Count_Person"], entities={"observationAbout": ["country/USA"]}, max_rows=5
    ) as (roles, batches):
        table = pa.Table.from_batches(list(batches), schema=observation_schema(roles))
    assert table.num_rows == 5


def test_max_rows_exceeded_raises_with_cumulative_observed(streaming_transport):
    body = (_HEADER + "".join(_row(2000 + i) for i in range(2000))).encode()
    transport = streaming_transport(body)
    client = SdmxClient(base_url=_BASE_URL, transport=transport)
    with (
        pytest.raises(SdmxCapExceededError) as excinfo,
        client.stream_data_batches(
            variables=["Count_Person"], entities={"observationAbout": ["country/USA"]}, max_rows=3
        ) as (_roles, batches),
    ):
        list(batches)
    err = excinfo.value
    assert err.cap_kind == "rows"
    assert err.cap == 3
    # Batch checks can exceed the cap by one batch; row parsing trips at cap + 1.
    assert err.observed >= 3


def test_max_rows_early_abort_bounds_bytes_pulled_to_one_read_ahead_window(streaming_transport):
    # PyArrow reads ahead about 34 blocks before the first batch; 64 blocks
    # safely bounds that buffer while remaining below this body's size.
    block_size = _client_module._STREAM_BATCH_BLOCK_SIZE
    one_row = _row(2000)
    rows = -(-80 * block_size // len(one_row))  # well over the 64-block bound
    chunk_size = 4096
    body = (_HEADER + one_row * rows).encode()
    transport = streaming_transport(body, chunk_size=chunk_size)
    client = SdmxClient(base_url=_BASE_URL, transport=transport)
    with (
        pytest.raises(SdmxCapExceededError),
        client.stream_data_batches(
            variables=["Count_Person"], entities={"observationAbout": ["country/USA"]}, max_rows=3
        ) as (_roles, batches),
    ):
        list(batches)
    bytes_pulled = transport.chunks_yielded * chunk_size
    bound = 64 * block_size
    assert bytes_pulled <= bound
    assert transport.chunks_yielded < transport.total_chunks


# --- max_bytes -----------------------------------------------------------------


def test_max_bytes_trips_on_header_only_response(streaming_transport):
    transport = streaming_transport(_HEADER.encode(), chunk_size=8)
    client = SdmxClient(base_url=_BASE_URL, transport=transport)
    small_cap = len(_HEADER) // 2
    with (
        pytest.raises(SdmxCapExceededError) as excinfo,
        client.stream_data_batches(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            max_bytes=small_cap,
        ) as (_roles, batches),
    ):
        list(batches)
    assert excinfo.value.cap_kind == "bytes"
    assert transport.chunks_yielded < transport.total_chunks


def test_max_bytes_trips_mid_record(streaming_transport):
    body = (_HEADER + _row(2000)).encode()
    small_cap = len(_HEADER) + 5
    transport = streaming_transport(body, chunk_size=8)
    client = SdmxClient(base_url=_BASE_URL, transport=transport)
    with (
        pytest.raises(SdmxCapExceededError) as excinfo,
        client.stream_data_batches(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            max_bytes=small_cap,
        ) as (_roles, batches),
    ):
        list(batches)
    err = excinfo.value
    assert err.cap_kind == "bytes"
    assert err.observed > small_cap
    assert transport.chunks_yielded < transport.total_chunks


def test_max_bytes_trips_on_pre_buffered_response(recording_transport):
    body = (_HEADER + "".join(_row(2000 + i) for i in range(20))).encode()
    response = httpx.Response(
        200,
        headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        content=body,
    )
    transport = recording_transport(response)
    client = SdmxClient(base_url=_BASE_URL, transport=transport)
    small_cap = len(_HEADER) + len(_row(2000))
    with (
        pytest.raises(SdmxCapExceededError) as excinfo,
        client.stream_data_batches(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            max_bytes=small_cap,
        ) as (_roles, batches),
    ):
        list(batches)
    assert excinfo.value.cap_kind == "bytes"


@pytest.mark.parametrize("status_code", [400, 500])
def test_max_bytes_aborts_oversized_non_success_body(streaming_transport, status_code):
    """The public batch API shares the row path's cap on 4xx/5xx bodies."""
    body = b'{"message":"upstream failure ' + b"x" * 2000 + b'"}'
    transport = streaming_transport(
        body,
        status_code=status_code,
        headers={"content-type": "application/json"},
        chunk_size=8,
    )
    client = SdmxClient(base_url=_BASE_URL, transport=transport)

    with (
        pytest.raises(SdmxCapExceededError) as excinfo,
        client.stream_data_batches(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            max_bytes=50,
        ) as (_, batches),
    ):
        list(batches)

    assert excinfo.value.cap_kind == "bytes"
    assert excinfo.value.cap == 50
    assert excinfo.value.observed > 50
    assert transport.chunks_yielded < transport.total_chunks


def test_byte_cap_wins_when_it_would_trip_inside_the_same_read_ahead_window(streaming_transport):
    # Byte caps run during reads; batch row caps run after a batch returns.
    # The byte cap therefore trips first when both thresholds are near.
    body = (_HEADER + "".join(_row(2000 + i) for i in range(2000))).encode()
    transport = streaming_transport(body)
    client = SdmxClient(base_url=_BASE_URL, transport=transport)
    small_byte_cap = len(_HEADER) + 20
    with (
        pytest.raises(SdmxCapExceededError) as excinfo,
        client.stream_data_batches(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            max_rows=3,
            max_bytes=small_byte_cap,
        ) as (_roles, batches),
    ):
        list(batches)
    assert excinfo.value.cap_kind == "bytes"


# --- Deadline ------------------------------------------------------------------


def test_deadline_expired_before_dispatch_raises_before_transport_call(recording_transport):
    import time

    response = httpx.Response(
        200,
        headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        content=_HEADER.encode(),
    )
    transport = recording_transport(response)
    client = SdmxClient(base_url=_BASE_URL, transport=transport)
    with (
        pytest.raises(TimeoutError, match="absolute caller deadline"),
        client.stream_data_batches(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            deadline=time.monotonic() - 1,
        ) as (_roles, batches),
    ):
        list(batches)
    assert transport.requests == []


# --- Equivalence with the row path ----------------------------------------------

_MULTI_ROLE_COLUMNS = (
    "STRUCTURE",
    "STRUCTURE_ID",
    "ACTION",
    "variableMeasured",
    "destinationCountry",
    "sourceCountry",
    "unit",
    "measurementMethod",
    "observationPeriod",
    "provenance",
    "TIME_PERIOD",
    "OBS_VALUE",
    "scalingFactor",
    "facetId",
)


def _multi_role_row(**overrides: str) -> str:
    """Build a CSV row and quote overridden cells with `csv.writer`."""
    defaults = {
        "STRUCTURE": "dataflow",
        "STRUCTURE_ID": "DC:DF_OBS(1.0.0)",
        "ACTION": "I",
        "variableMeasured": "Count_Person",
        "destinationCountry": "country/CAN",
        "sourceCountry": "country/USA",
        "unit": "Person",
        "measurementMethod": "Census",
        "observationPeriod": "P1Y",
        "provenance": "dc/base",
        "TIME_PERIOD": "2020",
        "OBS_VALUE": "1.50",
        "scalingFactor": "0",
        "facetId": "stored-facet-id",
    }
    defaults.update(overrides)
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\r\n").writerow(
        [defaults[name] for name in _MULTI_ROLE_COLUMNS]
    )
    return buf.getvalue()


def test_stream_data_and_stream_data_batches_agree_on_the_same_body(streaming_transport):
    body_text = (
        ",".join(_MULTI_ROLE_COLUMNS)
        + "\r\n"
        + _multi_role_row(TIME_PERIOD="2020")
        + _multi_role_row(
            sourceCountry="NotApplicable", TIME_PERIOD="2021", OBS_VALUE="", facetId=""
        )
        + _multi_role_row(TIME_PERIOD="2022", OBS_VALUE="line1\nline1b, more")
    )
    body = body_text.encode()

    row_transport = streaming_transport(body)
    row_client = SdmxClient(base_url=_BASE_URL, transport=row_transport)
    with row_client.stream_data(
        variables=["Count_Person"],
        entities={"destinationCountry": ["country/CAN"], "sourceCountry": ["country/USA"]},
    ) as (row_roles, rows):
        row_table = table_from_observations(list(rows), roles=row_roles)

    batch_transport = streaming_transport(body)
    batch_client = SdmxClient(base_url=_BASE_URL, transport=batch_transport)
    with batch_client.stream_data_batches(
        variables=["Count_Person"],
        entities={"destinationCountry": ["country/CAN"], "sourceCountry": ["country/USA"]},
    ) as (batch_roles, batches):
        batch_table = pa.Table.from_batches(list(batches), schema=observation_schema(batch_roles))

    assert row_roles == batch_roles
    assert row_table.equals(batch_table)
