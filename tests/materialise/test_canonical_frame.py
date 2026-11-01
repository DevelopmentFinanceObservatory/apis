"""The canonical frame: a frame's grammar table stored as Parquet. Its bytes depend on the rows
alone, whatever chunks they arrive in, and reading them back gives the frame, its rows, and the
frame1 document Presentation's assembler reads."""

from __future__ import annotations

import io
import json

import pyarrow.parquet as pq
import pytest
from dfo_frame_grammar import Frame, canonical_frame_bytes

import materialise.formats as materialise_formats
from materialise.formats import (
    CANONICAL_FRAME_MEDIA_TYPE,
    FRAME1_MEDIA_TYPE,
    canonical_frame_document,
    canonical_frame_rows,
    read_canonical_frame,
    read_canonical_frame_table,
    read_stored_frame,
    write_canonical_frame,
)
from materialise.frame_build import ParquetFrameSink, build_frame

_LARGE_VALUES = (9007199254740993.0, 1.2345678901234567e19, -1.2345678901234567e17, 1e21, -0.0)
"""Whole values at and above 2**53, where a double's exact integer and the digits JCS writes
for it differ, one in exponent form, and negative zero."""


def _value(index: int) -> float:
    """Whole and fractional values, as JCS writes them differently, after `_LARGE_VALUES`."""
    if index < len(_LARGE_VALUES):
        return _LARGE_VALUES[index]
    return float(index) if index % 2 else index + 0.25


def _frame(rows: int = 30) -> Frame:
    units: list[object] = [
        "USD",
        None,
        {"currency": "EUR", "prices": "constant", "base_year": 2024},
        {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person"}},
    ]
    return build_frame(
        role_keys=("provider", "recipient"),
        rows=[
            {
                "stat_var": "dcid:Amount_FinancialFlow",
                "provider": f"country/P{index % 3}",
                "recipient": f"country/R{index:03d}",
                "time_period": str(2000 + index % 5),
                "value": _value(index),
                "unit": units[index % len(units)],
                "measurementMethod": None if index % 3 else "GrantEquivalent",
                "provenance": "OECD_CRS",
            }
            for index in range(rows)
        ],
    )


def _chunks_of(frame: Frame, size: int) -> list[Frame]:
    return [
        Frame(columns=frame.columns, table=frame.table.slice(start, size))
        for start in range(0, frame.table.num_rows, size)
    ]


def _write(chunks) -> tuple[bytes, str]:
    sink = io.BytesIO()
    written = write_canonical_frame(chunks, sink=sink)
    assert written.media_type == CANONICAL_FRAME_MEDIA_TYPE == "application/vnd.apache.parquet"
    assert written.size_bytes == len(sink.getvalue())
    return sink.getvalue(), written.digest


def test_the_bytes_depend_on_the_rows_and_not_on_their_chunks():
    frame = _frame()

    whole = _write((frame,))
    chunked = _write(_chunks_of(frame, 7))

    assert whole == chunked


_RECIPE_UNIT = {"currency": "USD", "prices": "constant", "base_year": 2020, "recipe": "recipe-1"}


@pytest.mark.parametrize("later_unit", [_RECIPE_UNIT, "USD"], ids=["recipe", "plain"])
def test_a_mint_sink_reads_every_chunk_back_in_the_whole_frames_unit_layout(later_unit):
    """A chunk takes the recipe `unit` layout only when its own cells carry a recipe, so a
    mint's first chunk can lack the layout a later chunk needs. The canonical frame of what the
    sink reads back is the whole frame's."""
    rows = [
        {
            "stat_var": stat_var,
            "recipient": "country/KEN",
            "time_period": "2020",
            "value": 1.0,
            "unit": unit,
            "measurementMethod": None,
            "provenance": "OECD_CRS",
        }
        for stat_var, unit in (("dcid:A", "USD"), ("dcid:B", later_unit))
    ]
    chunks = [build_frame(role_keys=("recipient",), rows=[row]) for row in rows]
    sink = ParquetFrameSink()
    for chunk in chunks:
        sink.write(chunk.table)

    read = sink.finish(columns=chunks[0].columns)

    assert _write(read()) == _write((build_frame(role_keys=("recipient",), rows=rows),))


def test_reading_the_bytes_back_gives_the_frame():
    frame = _frame()

    data, _ = _write(_chunks_of(frame, 4))

    assert read_canonical_frame(data) == frame


def test_an_empty_frame_round_trips():
    frame = _frame(0)

    data, _ = _write((frame,))

    assert read_canonical_frame(data) == frame


def test_a_stored_frame_reads_by_its_media_type():
    frame = _frame()
    data, _ = _write((frame,))

    canonical = read_stored_frame(data, media_type=CANONICAL_FRAME_MEDIA_TYPE)
    frame1 = read_stored_frame(canonical_frame_bytes(frame), media_type=FRAME1_MEDIA_TYPE)

    assert canonical == frame
    assert frame1.columns == frame.columns
    assert len(frame1.rows) == frame.table.num_rows


def test_the_document_equals_the_parsed_frame1_document():
    frame = _frame()
    data, _ = _write((frame,))

    document = canonical_frame_document(read_canonical_frame_table(data))

    assert document == json.loads(canonical_frame_bytes(frame))
    # Key order and number types carry into an assembler's own JSON output.
    assert json.dumps(document) == json.dumps(json.loads(canonical_frame_bytes(frame)))


def test_a_projected_read_yields_the_frame_rows_in_the_named_columns():
    frame = _frame()
    data, _ = _write((frame,))
    wanted = [column for column in frame.columns if column != "status"]

    table = read_canonical_frame_table(data, columns=wanted)

    assert table.column_names == wanted
    positions = [frame.columns.index(column) for column in wanted]
    assert list(canonical_frame_rows(table)) == [
        tuple(row[position] for position in positions) for row in frame.rows
    ]


def test_row_groups_hold_a_fixed_count_of_rows_whatever_the_chunks(monkeypatch):
    monkeypatch.setattr(materialise_formats, "_CANONICAL_FRAME_ROW_GROUP_ROWS", 10)
    frame = _frame(25)

    whole = _write((frame,))
    chunked = _write(_chunks_of(frame, 7))

    assert whole == chunked
    metadata = pq.ParquetFile(io.BytesIO(whole[0])).metadata
    assert [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)] == [
        10,
        10,
        5,
    ]
