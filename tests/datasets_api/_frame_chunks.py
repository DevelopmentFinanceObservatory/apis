"""Frames from rows written as tuples, for render tests that state a frame's rows by hand."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Sequence

from dfo_frame_grammar import STATUS_COLUMN, UNIT_COLUMN, Frame, unit_cell_to_json

from materialise.formats import frame_entity_columns


def frame_of_rows(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> Frame:
    """The frame of `rows`, each a tuple of cells in `columns` order as `Frame.rows` gives
    them, in grammar row order. A null cell is left out of its row, and `status` is the
    grammar's to fill."""
    return Frame.build(
        role_keys=frame_entity_columns(columns),
        rows=[
            {
                column: unit_cell_to_json(cell) if column == UNIT_COLUMN else cell
                for column, cell in zip(columns, row, strict=True)
                if cell is not None and column != STATUS_COLUMN
            }
            for row in rows
        ],
    )


def frame_chunks_of(
    columns: Sequence[str], chunks: Sequence[Sequence[Sequence[object]]]
) -> Callable[[], Iterator[Frame]]:
    """A fresh pass over the frame of every row of `chunks`, cut into chunks of their sizes, as
    a materialiser streams one frame in grammar row order."""
    frame = frame_of_rows(columns, [row for chunk in chunks for row in chunk])
    starts = [0]
    for chunk in chunks:
        starts.append(starts[-1] + len(chunk))
    return lambda: (
        Frame.from_table(columns=frame.columns, table=frame.table.slice(start, end - start))
        for start, end in itertools.pairwise(starts)
    )
