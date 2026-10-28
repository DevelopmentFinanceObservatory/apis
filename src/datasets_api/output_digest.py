"""The output digest of a derived frame (`output-digest` in the client contract's rules).

The digest names a frame's output alone: its members, periods, values and units. Its document
is the RFC 8785 (JCS) form of `{"digest_version": "out1", "columns", "rows"}`, where `columns`
are the frame's role columns in the frame's order, then `time_period`, `value` and `unit`, and
each row holds the frame row's cells in those columns. A unit cell is its JSON form with
`recipe` dropped from every unit-request object, in the cell itself and in either part of a
composed unit. The rows are sorted by the UTF-8 bytes of their own JCS form, and a repeated row
stays. The digest is `out_` and the first 32 hex digits of the SHA-256 of the document.

`stat_var`, the facet columns (`measurementMethod`, `provenance`) and `status` stay out, so an
operation minting its own variable, a source imported again or a conversion recipe revised
moves no digest while no number moves.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import cast

import rfc8785
from dfo_frame_grammar import Frame, unit_cell_to_json, unit_cells, unit_codes
from dfo_frame_grammar.grammar import TIME_PERIOD_COLUMN, UNIT_COLUMN, VALUE_COLUMN

DIGEST_VERSION = "out1"
DIGEST_PREFIX = "out_"
_DIGEST_HEX_LENGTH = 32
_RECIPE_KEY = "recipe"
_COMPOSED_PARTS = ("numerator", "denominator")


def _without_recipe(unit: object) -> object:
    """A unit cell's JSON form with `recipe` dropped from every unit-request object."""
    if not isinstance(unit, Mapping):
        return unit
    if any(part in unit for part in _COMPOSED_PARTS):
        return {key: _without_recipe(value) for key, value in unit.items()}
    return {key: value for key, value in unit.items() if key != _RECIPE_KEY}


def digest_document(*, role_columns: Sequence[str], rows: Sequence[Sequence[object]]) -> bytes:
    """The JCS bytes of the digest document over `rows`, each holding a frame row's cells in
    `role_columns`, then `time_period`, `value` and the unit cell in its JSON form."""
    cells = [[*row[:-1], _without_recipe(row[-1])] for row in rows]
    ordered = sorted(cells, key=rfc8785.dumps)
    return rfc8785.dumps(
        {
            "digest_version": DIGEST_VERSION,
            "columns": [*role_columns, TIME_PERIOD_COLUMN, VALUE_COLUMN, UNIT_COLUMN],
            "rows": ordered,
        }
    )


def output_digest(frame: Frame) -> str:
    """`out_` and 32 hex digits naming the output of `frame`. See the module docstring."""
    role_columns = frame.columns[1 : frame.columns.index(TIME_PERIOD_COLUMN)]
    table = frame.table
    unit_column = table.column(UNIT_COLUMN)
    codes = unit_codes(unit_column)
    units = [
        None if cell is None else unit_cell_to_json(cell) for cell in unit_cells(unit_column, codes)
    ]
    columns = [
        table.column(name).to_pylist() for name in (*role_columns, TIME_PERIOD_COLUMN, VALUE_COLUMN)
    ]
    rows = [
        [*(column[index] for column in columns), units[code]]
        for index, code in enumerate(cast("list[int]", codes.codes.to_pylist()))
    ]
    document = digest_document(role_columns=role_columns, rows=rows)
    return f"{DIGEST_PREFIX}{hashlib.sha256(document).hexdigest()[:_DIGEST_HEX_LENGTH]}"


__all__ = ["DIGEST_PREFIX", "DIGEST_VERSION", "digest_document", "output_digest"]
