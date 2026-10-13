"""A canonical frame stored before the unit `request` struct carried `recipe` (frame grammar
kit 959f8bc) reads through every reader with the cells and frame hash its own release gave it.
The oracle is that release's reading (`_canonical_frame.stored_canonical_frame_959f8bc`); the
fresh frame is the same rows built by this release's grammar."""

from __future__ import annotations

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dfo_frame_grammar import STATUS_COLUMN, canonical_frame_bytes, frame_sha256

from materialise.formats import (
    CANONICAL_FRAME_MEDIA_TYPE,
    canonical_frame_document,
    canonical_frame_rows,
    read_canonical_frame,
    read_canonical_frame_table,
    read_stored_frame,
    write_canonical_frame,
)
from materialise.statistics import compute_series_statistics_table

from ._canonical_frame import frame_of_document, stored_canonical_frame_959f8bc

_DATA, _FRAME_SHA256, _DOCUMENT = stored_canonical_frame_959f8bc()


def test_the_fixture_holds_the_959f8bc_unit_layout_and_every_unit_kind():
    """The stored `request` struct has no `recipe`, and the rows hold a ref, converted
    requests, composed units and an empty unit."""
    unit = pq.read_schema(pa.BufferReader(_DATA)).field("unit").type
    assert unit.field("request").type.names == ["currency", "prices", "base_year"]
    units = [row[_DOCUMENT["columns"].index("unit")] for row in _DOCUMENT["rows"]]
    assert {
        "ref" if isinstance(cell, str) else "composed" if "numerator" in cell else "request"
        for cell in units
        if cell is not None
    } == {"ref", "request", "composed"}
    assert None in units


@pytest.mark.parametrize(
    "read",
    [
        read_canonical_frame,
        lambda data: read_stored_frame(data, media_type=CANONICAL_FRAME_MEDIA_TYPE),
    ],
    ids=["read_canonical_frame", "read_stored_frame"],
)
def test_the_frame_readers_give_the_frame_its_release_stored(read):
    frame = read(_DATA)

    assert frame_sha256(frame) == _FRAME_SHA256
    assert json.loads(canonical_frame_bytes(frame)) == _DOCUMENT
    assert frame.rows == frame_of_document(_DOCUMENT).rows


@pytest.mark.parametrize(
    "frame",
    [lambda: read_canonical_frame(_DATA), lambda: frame_of_document(_DOCUMENT)],
    ids=["read-back", "built-fresh"],
)
def test_a_frame_without_recipes_is_written_as_its_release_stored_it(frame):
    """A frame whose units carry no conversion recipe, read back or built from the same rows,
    is written in the layout 959f8bc stored, so either release stores the same object."""
    sink = io.BytesIO()
    write_canonical_frame((frame(),), sink=sink)

    assert sink.getvalue() == _DATA
    assert frame_sha256(read_canonical_frame(sink.getvalue())) == _FRAME_SHA256


def test_the_table_readers_give_its_document_and_rows():
    """Presentation's frame store reads the document, the observations door the projected
    rows."""
    fresh = frame_of_document(_DOCUMENT)
    projected = [name for name in fresh.columns if name != STATUS_COLUMN]
    status = fresh.columns.index(STATUS_COLUMN)

    assert canonical_frame_document(read_canonical_frame_table(_DATA)) == _DOCUMENT
    assert list(canonical_frame_rows(read_canonical_frame_table(_DATA))) == list(fresh.rows)
    assert list(canonical_frame_rows(read_canonical_frame_table(_DATA, columns=projected))) == [
        row[:status] + row[status + 1 :] for row in fresh.rows
    ]


def test_series_statistics_of_the_stored_frame_are_those_of_its_rows():
    stored = compute_series_statistics_table(read_canonical_frame(_DATA))

    assert stored.num_rows == 6
    assert stored == compute_series_statistics_table(frame_of_document(_DOCUMENT))
