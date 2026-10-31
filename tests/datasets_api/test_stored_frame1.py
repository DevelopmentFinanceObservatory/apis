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
from datasets_api.doors import observations
from datasets_api.store_errors import StoredFrameUnreadable, read_stored_frame
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


def test_a_stored_canonical_frame_outside_the_canonical_schema_refuses_unreadable():
    table = pa.table({"stat_var": ["dcid:X"], "value": [1.0]})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)

    with pytest.raises(StoredFrameUnreadable):
        read_stored_frame(
            sink.getvalue().to_pybytes(), media_type=CANONICAL_FRAME_MEDIA_TYPE, digest="d"
        )


@pytest.mark.parametrize("media_type", [FRAME1_MEDIA_TYPE, CANONICAL_FRAME_MEDIA_TYPE])
def test_the_observations_door_streams_a_stored_frames_rows(media_type: str):
    """Either stored form's rows are built a slice at a time, never all up front."""
    frame = Frame.build(role_keys=("recipient",), rows=malaria_ethiopia_rows())
    data = (
        canonical_frame_bytes(frame)
        if media_type == FRAME1_MEDIA_TYPE
        else canonical_frame_parquet((frame,))
    )

    columns, rows = observations._stored_frame_rows(data, media_type=media_type, digest="d")

    assert inspect.isgenerator(rows)
    assert [dict(zip(columns, row, strict=True))["value"] for row in rows] == [
        row["value"] for row in malaria_ethiopia_rows()
    ]


def test_no_datasets_door_classifies_an_unreadable_stored_frame():
    """It answers the platform's `internal_error`, as a stored canonical frame that does not
    decode does, so its detail carries nothing of the artifact."""
    exc = StoredFrameUnreadable(digest="sha256:exemplar")

    assert classify_datasets_exception(exc) is None
