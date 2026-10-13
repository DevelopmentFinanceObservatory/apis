"""Canonical frame bytes for tests: a frame's grammar table as Parquet."""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from dfo_frame_grammar import Frame

from materialise.formats import frame_entity_columns, write_canonical_frame
from materialise.frame_build import build_frame

CANONICAL_FRAME_RENDERING: dict[str, object] = {"format": "frame"}
"""The rendering a version's canonical frame is linked under."""

_STORED_959F8BC = Path(__file__).parent / "fixtures" / "stored-canonical-frame-959f8bc"


def stored_canonical_frame_959f8bc() -> tuple[bytes, str, dict[str, object]]:
    """A canonical frame as dfo-apis stored it under frame grammar kit 959f8bc, whose `unit`
    struct's `request` has no `recipe`: its Parquet bytes, the `frame_sha256` that release
    computed for it, and its frame1 document as that release's canonical bytes parse
    (`fixtures/make_stored_canonical_frame.py`). Its rows hold a plain, a converted and a
    composed unit."""
    oracle = json.loads(_STORED_959F8BC.with_suffix(".json").read_text())
    return (
        _STORED_959F8BC.with_suffix(".parquet").read_bytes(),
        oracle["frame_sha256"],
        oracle["document"],
    )


def frame_of_document(document: dict[str, object]) -> Frame:
    """The frame the frame1 `document` holds, built by this release's grammar."""
    columns = cast("list[str]", document["columns"])
    return build_frame(
        role_keys=frame_entity_columns(columns),
        rows=[
            {column: cell for column, cell in zip(columns, row, strict=True) if column != "status"}
            for row in cast("list[list[object]]", document["rows"])
        ],
    )


def canonical_frame_parquet(chunks: Iterable[Frame]) -> bytes:
    """The canonical frame of the rows of `chunks`, in order, as `write_canonical_frame` writes
    it."""
    sink = io.BytesIO()
    write_canonical_frame(chunks, sink=sink)
    return sink.getvalue()


def canonical_frame_of_frame1(frame1: bytes) -> bytes:
    """The canonical frame of the frame1 document `frame1`."""
    return canonical_frame_parquet((frame_of_document(json.loads(frame1)),))
