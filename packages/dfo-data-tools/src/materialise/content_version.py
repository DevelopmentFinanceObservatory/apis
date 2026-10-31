"""Mints a materialisation's `content_version` from its canonical frame through
`dfo_frame_grammar.content_version`. The token hashes the frame alone, so identical content is
one version whatever facets, revisions, or conversion produced it. The trace and the
materialisation row record that lineage beside the token."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import pyarrow as pa
import rfc8785
from dfo_frame_grammar import (
    FRAME_VERSION,
    STATUS_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    Frame,
    FrameError,
    array_kernel,
    collation_sort_keys,
    mint_content_version,
    recompute_content_version,
    unit_cell_to_json,
    unit_cells,
    unit_codes,
)

from materialise.columnar import strictly_increasing
from materialise.recipe_bodies import RECIPE_BODIES, recipe_body_digest


def recipe_dependency_fingerprint(recipe_id: str) -> tuple[str, str] | None:
    """Returns `(recipe_id, recipe_body_digest)` for the current body, or `None` when
    `RECIPE_BODIES` has no entry. Materialisation and currency rechecks use this same rule."""
    recipe = RECIPE_BODIES.get(recipe_id)
    if recipe is None:
        return None
    return recipe_id, recipe_body_digest(recipe)


def mint_version(frame: Frame) -> str:
    """Single-frame convenience wrapper around `mint_version_from_chunks`."""
    return mint_version_from_chunks((frame,))


def mint_version_from_chunks(chunks: Iterable[Frame]) -> str:
    """Hashes validated grammar `Frame` chunks in iteration order, then mints their version.
    Accepts grammar `Frame` chunks built by `Frame.build`, which normalises each cell once. A
    single chunk hashes like `frame_sha256`. Multiple chunks hash like their concatenated rows
    while holding at most one chunk's rows in memory. Use `mint_version_from_digest` when the
    frame digest is already computed."""
    return mint_version_from_digest(stream_frame_sha256(chunks))


# Rows serialised per Arrow pass: about 13 MB of row text for a typical frame row.
_HASH_SLICE_ROWS = 65_536


def stream_frame_sha256(chunks: Iterable[Frame]) -> str:
    """The digest `dfo_frame_grammar.stream_frame_sha256` gives for `chunks`, computed per column.

    The grammar's function builds a dict and calls `rfc8785.dumps` for every row. This one calls
    `rfc8785.dumps` once per distinct cell of each column within a slice of rows (on the Python
    value `to_pylist` gives, as the grammar does), takes those bytes back to every row, and joins
    each row's cells with Arrow string kernels. The bytes hashed are the grammar's canonical
    document bytes, so every digest and `content_version` is the grammar's. It refuses as the
    grammar does: `schema_violation` for no chunks, `column_mismatch` for a chunk whose columns
    differ from the first, and `row_key_not_increasing` for a row whose `collation_sort_keys` key
    is not strictly greater than the previous row's, within or across chunks.
    """
    hasher = hashlib.sha256()
    columns: tuple[str, ...] | None = None
    last_key: pa.Table | None = None
    for frame in chunks:
        if columns is None:
            columns = frame.columns
            prefix = rfc8785.dumps(
                {"frame_version": FRAME_VERSION, "columns": list(columns), "rows": []}
            )
            hasher.update(prefix[:-2])
        elif frame.columns != columns:
            raise FrameError(
                "column_mismatch",
                f"chunk columns {list(frame.columns)!r} differ from the stream's columns "
                f"{list(columns)!r}",
            )
        grain = [column for column in columns if column not in (VALUE_COLUMN, STATUS_COLUMN)]
        table = frame.table
        for start in range(0, table.num_rows, _HASH_SLICE_ROWS):
            rows = table.slice(start, _HASH_SLICE_ROWS)
            first_of_stream = last_key is None
            last_key = _check_increasing(collation_sort_keys(rows, grain), last_key)
            _hash_rows(hasher, rows, columns, first_of_stream=first_of_stream)
    if columns is None:
        raise FrameError(
            "schema_violation",
            "stream_frame_sha256 requires at least one chunk to determine the frame's columns",
        )
    hasher.update(b"]}")
    return hasher.hexdigest()


def _check_increasing(keys: pa.Table, last_key: pa.Table | None) -> pa.Table:
    """Refuses unless each row of `keys` is lexicographically greater than the one before it,
    the first row compared with `last_key`. Returns the last row of `keys` as the next
    `last_key` (`columnar.strictly_increasing`, which orders text as the grammar's Python
    tuple comparison does)."""
    if last_key is not None:
        keys = pa.concat_tables([last_key, keys])
    count = keys.num_rows
    if not strictly_increasing([column.combine_chunks() for column in keys.columns]):
        raise FrameError(
            "row_key_not_increasing",
            "a row's row_sort_key is not strictly greater than the previous row's, "
            "within or across chunks",
        )
    return keys.slice(count - 1)


def _hash_rows(
    hasher: hashlib._Hash, rows: pa.Table, columns: Sequence[str], *, first_of_stream: bool
) -> None:
    """Feeds `hasher` the canonical bytes of `rows`: `[cell,...]` per row, each preceded by a
    comma unless it is the stream's first row."""
    if not rows.num_rows:
        return
    cells = [_cell_bytes(rows.column(column), unit=column == UNIT_COLUMN) for column in columns]
    joined = array_kernel("binary_join_element_wise", *cells, b",")
    framed = array_kernel("binary_join_element_wise", b",[", joined, b"]", b"")
    _, offsets_buffer, data = framed.buffers()
    assert offsets_buffer is not None and data is not None
    offsets = memoryview(offsets_buffer).cast("i")
    start = offsets[framed.offset] + (1 if first_of_stream else 0)
    end = offsets[framed.offset + len(framed)]
    hasher.update(memoryview(data)[start:end])


def _cell_bytes(values: pa.ChunkedArray, *, unit: bool) -> pa.Array:
    """Each row's canonical JCS cell bytes. `rfc8785.dumps` runs once per distinct value: the
    `unit_cell_to_json` form of a unit cell, else the value `to_pylist` reads (floats compare by
    bit pattern, so `0.0` and `-0.0` stay apart)."""
    codes = unit_codes(values)
    if unit:
        distinct = [unit_cell_to_json(cell) for cell in unit_cells(values, codes)]
    else:
        distinct = values.take(codes.first_rows).to_pylist()
    encoded = pa.array([rfc8785.dumps(value) for value in distinct], type=pa.binary())
    return encoded.take(codes.codes)


def mint_version_from_digest(digest: str) -> str:
    """Mints from an existing frame digest. Group generation hashes during its single fetch,
    and `_finish_group` uses this function to avoid rereading rows."""
    return mint_content_version(frame_sha256=digest)


def recompute_version(canonical_bytes: bytes) -> str:
    """Recomputes the content version from canonical frame bytes."""
    return recompute_content_version(canonical_bytes)
