"""Canonical frame bytes for tests: a frame's grammar table as Parquet."""

from __future__ import annotations

import io
import json
from collections.abc import Iterable

from dfo_frame_grammar import Frame

from materialise.formats import frame_entity_columns, write_canonical_frame
from materialise.frame_build import build_frame

CANONICAL_FRAME_RENDERING: dict[str, object] = {"format": "frame"}
"""The rendering a version's canonical frame is linked under."""


def canonical_frame_parquet(chunks: Iterable[Frame]) -> bytes:
    """The canonical frame of the rows of `chunks`, in order, as `write_canonical_frame` writes
    it."""
    sink = io.BytesIO()
    write_canonical_frame(chunks, sink=sink)
    return sink.getvalue()


def canonical_frame_of_frame1(frame1: bytes) -> bytes:
    """The canonical frame of the frame1 document `frame1`."""
    document = json.loads(frame1)
    columns = document["columns"]
    frame = build_frame(
        role_keys=frame_entity_columns(columns),
        rows=[
            {column: cell for column, cell in zip(columns, row, strict=True) if column != "status"}
            for row in document["rows"]
        ],
    )
    return canonical_frame_parquet((frame,))
