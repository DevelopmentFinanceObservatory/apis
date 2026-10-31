"""A stored frame artifact that does not read as a frame is a stored-artifact fault."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dfo_frame_grammar import Frame, canonical_frame_bytes

from assembler.testing import build_frame_doc, malaria_ethiopia_rows
from datasets_api.app import classify_datasets_exception
from datasets_api.store_errors import (
    StoredFrameUnreadable,
    read_stored_frame,
    stored_frame_rows,
)
from materialise.formats import CANONICAL_FRAME_MEDIA_TYPE, FRAME1_MEDIA_TYPE
from tests.materialise._canonical_frame import canonical_frame_parquet


def read_stored_frame1(data: bytes, *, digest: str):
    return read_stored_frame(data, media_type=FRAME1_MEDIA_TYPE, digest=digest)


def _malaria_document() -> tuple[dict[str, object], list[list[object]]]:
    """The malaria frame1 document and its rows."""
    document = build_frame_doc(role_keys=("recipient",), rows=malaria_ethiopia_rows())
    return document, cast("list[list[object]]", document["rows"])


def _null_role(rows: list[list[object]]) -> None:
    rows[0][1] = None


def _repeated_grain(rows: list[list[object]]) -> None:
    rows.insert(1, list(rows[1]))


def test_a_stored_frame1_the_rule_refuses_names_its_digest_and_not_the_parser():
    with pytest.raises(StoredFrameUnreadable) as excinfo:
        read_stored_frame1(b'{"frame_version": "frame0"}', digest="sha256:exemplar")

    assert "sha256:exemplar" in str(excinfo.value)
    assert "frame0" not in str(excinfo.value)


def _role(cell: object) -> Callable[[list[list[object]]], None]:
    def damage(rows: list[list[object]]) -> None:
        rows[0][1] = cell

    return damage


def _value(cell: object) -> Callable[[list[list[object]]], None]:
    def damage(rows: list[list[object]]) -> None:
        rows[0][3] = cell

    return damage


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(_null_role, id="a-null-role"),
        pytest.param(_repeated_grain, id="a-repeated-grain"),
        pytest.param(_value("1.5"), id="a-numeric-string-value"),
        pytest.param(_value("abc"), id="a-string-value"),
        pytest.param(_value([1]), id="a-list-value"),
        pytest.param(_value(10**400), id="a-value-past-the-double-range"),
        pytest.param(_value(True), id="a-boolean-value"),
        pytest.param(_role("country/\ud800"), id="a-role-with-a-lone-surrogate"),
    ],
)
def test_a_stored_frame1_that_is_no_grammar_frame_refuses_unreadable(
    damage: Callable[[list[list[object]]], None],
):
    document, rows = _malaria_document()
    damage(rows)

    with pytest.raises(StoredFrameUnreadable):
        read_stored_frame1(json.dumps(document).encode(), digest="sha256:exemplar")


def test_a_stored_frame1s_rows_read_in_grammar_order():
    document, rows = _malaria_document()
    data = json.dumps(document).encode()
    rows.reverse()

    reversed_frame = read_stored_frame1(json.dumps(document).encode(), digest="d")

    assert reversed_frame == read_stored_frame1(data, digest="d")


def _read_rows(data: bytes, *, media_type: str, digest: str) -> object:
    columns, rows = stored_frame_rows(data, media_type=media_type, digest=digest)
    return columns, list(rows)


_CANONICAL_CELLS: dict[str, pa.Array] = {
    "stat_var": pa.array(["dcid:X"]),
    "recipient": pa.array(["country/ETH"]),
    "time_period": pa.array(["2020"]),
    "value": pa.array([1.0]),
    "unit": pa.array([None], pa.string()),
    "measurementMethod": pa.array([None], pa.string()),
    "provenance": pa.array(["p"]),
}


def _parquet(cells: dict[str, object]) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table(cells), sink)
    return sink.getvalue().to_pybytes()


def _malaria_parquet() -> bytes:
    return canonical_frame_parquet(
        (Frame.build(role_keys=("recipient",), rows=malaria_ethiopia_rows()),)
    )


def _null_cell(column: str) -> Callable[[], bytes]:
    """The malaria canonical frame with every `column` cell null."""

    def damaged() -> bytes:
        table = pq.read_table(pa.BufferReader(_malaria_parquet()))
        index = table.column_names.index(column)
        nulls = pa.nulls(table.num_rows, table.schema.field(column).type)
        return _parquet(
            dict(
                zip(table.column_names, table.set_column(index, column, nulls).columns, strict=True)
            )
        )

    return damaged


def _flipped(offset: int) -> Callable[[], bytes]:
    """The malaria canonical frame with the byte at `offset` inverted. On this frame offset 4
    breaks a page header, 31 a page's compressed data and 1121 a dictionary's UTF-8."""

    def damaged() -> bytes:
        data = bytearray(_malaria_parquet())
        data[offset] ^= 0xFF
        return bytes(data)

    return damaged


@pytest.mark.parametrize("read", [read_stored_frame, _read_rows])
@pytest.mark.parametrize(
    "stored",
    [
        pytest.param(lambda: _parquet({"stat_var": ["dcid:X"], "value": [1.0]}), id="two-columns"),
        pytest.param(
            lambda: _parquet({k: v for k, v in _CANONICAL_CELLS.items() if k != "time_period"}),
            id="no-time-period",
        ),
        pytest.param(
            lambda: _parquet({**_CANONICAL_CELLS, "unit": pa.array([5])}), id="an-integer-unit"
        ),
        pytest.param(
            lambda: _parquet({**_CANONICAL_CELLS, "value": pa.array(["country/ETH"])}),
            id="a-string-value",
        ),
        pytest.param(_null_cell("stat_var"), id="a-null-stat-var"),
        pytest.param(_null_cell("recipient"), id="a-null-role"),
        pytest.param(_null_cell("time_period"), id="a-null-time-period"),
        pytest.param(_null_cell("value"), id="a-null-value"),
        pytest.param(_flipped(4), id="a-broken-page-header"),
        pytest.param(_flipped(31), id="broken-compressed-data"),
        pytest.param(_flipped(1121), id="a-dictionary-of-invalid-utf8"),
        pytest.param(lambda: _malaria_parquet()[:100], id="truncated"),
    ],
)
def test_a_stored_canonical_frame_that_does_not_decode_refuses_unreadable(
    read: Callable[..., object], stored: Callable[[], bytes]
):
    """Each reader of stored bytes refuses whatever stops the bytes decoding as a frame, the
    grammar or the Parquet reader, naming only the digest, so no cell reaches a client."""
    with pytest.raises(StoredFrameUnreadable) as excinfo:
        read(stored(), media_type=CANONICAL_FRAME_MEDIA_TYPE, digest="d")

    assert "country/ETH" not in str(excinfo.value)


@pytest.mark.parametrize("media_type", [FRAME1_MEDIA_TYPE, CANONICAL_FRAME_MEDIA_TYPE])
def test_a_stored_frames_rows_stream(media_type: str):
    """Either stored form's rows are built a slice at a time, never all up front."""
    frame = Frame.build(role_keys=("recipient",), rows=malaria_ethiopia_rows())
    data = (
        canonical_frame_bytes(frame)
        if media_type == FRAME1_MEDIA_TYPE
        else canonical_frame_parquet((frame,))
    )

    columns, rows = stored_frame_rows(data, media_type=media_type, digest="d")

    assert inspect.isgenerator(rows)
    assert [dict(zip(columns, row, strict=True))["value"] for row in rows] == [
        row["value"] for row in malaria_ethiopia_rows()
    ]


def test_no_datasets_door_classifies_an_unreadable_stored_frame():
    """It answers the platform's `internal_error`, as a stored canonical frame that does not
    decode does, so its detail carries nothing of the artifact."""
    exc = StoredFrameUnreadable(digest="sha256:exemplar")

    assert classify_datasets_exception(exc) is None
