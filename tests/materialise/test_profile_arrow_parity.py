"""Differential test: the Arrow profile against the row-loop oracle.

Seeded random frames cover null keys, the `unit` struct, single-chunk and multi-chunk input, and
`columns` values that collide only once stringified. `Frame` chunks take the Arrow coding path;
`Table` chunks, which can mix `int` and `str` cells, take the Python-row coding path.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence

import pyarrow as pa
import pytest
from dfo_frame_grammar import Frame
from dfo_frame_grammar.frame import canonical_frame_schema

from materialise import EnrichmentLookup
from materialise.profile import OutputProfile, compute_output_profile_from_chunks
from materialise.reshape import Table, TableLike

from ._profile_oracle import reference_output_profile_from_chunks
from .conftest import FakeGraphReader

_COLUMNS = (
    "stat_var",
    "r1",
    "r2",
    "time_period",
    "value",
    "unit",
    "measurementMethod",
    "provenance",
    "status",
)
_ROLES = ("r1", "r2")

# "None" collides with a null cell once stringified, and "r1" with the role column's own name.
_FRAME_ALPHABETS: dict[str, list[str | None]] = {
    "stat_var": ["sv1", "sv2", "sv3"],
    "r1": ["a", "b", "c", "None", None],
    "r2": ["x", "y", "r1", "None", None],
    "time_period": ["2020", "2021", "2022", "r2", None],
    "measurementMethod": ["m1", "m2", None],
    "provenance": ["p1", "p2"],
}
_UNIT_CELLS: list[dict[str, object] | None] = [
    {"ref": "USD", "request": None},
    {"ref": "EUR", "request": None},
    {"ref": None, "request": {"currency": "USD", "prices": "constant", "base_year": 2020}},
    {"ref": None, "request": {"currency": "USD", "prices": "constant", "base_year": None}},
    {"ref": None, "request": {"currency": "USD", "prices": "current", "base_year": None}},
    None,
]
_TABLE_ALPHABETS: dict[str, list[object]] = {
    "stat_var": ["sv1", "sv2"],
    "r1": ["a", "b", "c", None],
    "r2": ["x", "y", 1, "1", None],
    "time_period": ["2020", "2021", 2020, "r1"],
    "unit": ["USD", "EUR", None],
    "measurementMethod": ["m1", "m2", None],
    "provenance": ["p1", "p2"],
}
_ARCS = {
    "a": {"name": ("Alpha",), "countryAlpha3Code": ("AAA",)},
    "b": {"name": ("Beta",)},
    "x": {"countryAlpha3Code": ("XXX",)},
}


def _pick[T](rng: random.Random, choices: Sequence[T]) -> T:
    # A random alphabet prefix per column makes constant columns, functional
    # dependencies and duplicate keys all occur.
    return rng.choice(choices[: rng.randint(1, len(choices))])


def _random_frame(rng: random.Random, rows: int) -> Frame:
    widths = {name: rng.randint(1, len(cells)) for name, cells in _FRAME_ALPHABETS.items()}
    unit_width = rng.randint(1, len(_UNIT_CELLS))
    data: dict[str, list[object]] = {}
    for name in _COLUMNS:
        if name == "value":
            data[name] = [rng.random() for _ in range(rows)]
        elif name == "status":
            data[name] = [None] * rows
        elif name == "unit":
            data[name] = [rng.choice(_UNIT_CELLS[:unit_width]) for _ in range(rows)]
        else:
            data[name] = [rng.choice(_FRAME_ALPHABETS[name][: widths[name]]) for _ in range(rows)]
    table = pa.Table.from_pydict(data, schema=canonical_frame_schema(_COLUMNS))
    return Frame(columns=_COLUMNS, table=table)


def _random_table(rng: random.Random, rows: int) -> Table:
    widths = {name: rng.randint(1, len(cells)) for name, cells in _TABLE_ALPHABETS.items()}
    return Table(
        columns=_COLUMNS,
        rows=tuple(
            tuple(
                rng.random()
                if name == "value"
                else None
                if name == "status"
                else rng.choice(_TABLE_ALPHABETS[name][: widths[name]])
                for name in _COLUMNS
            )
            for _ in range(rows)
        ),
    )


def _split(rng: random.Random, whole: TableLike, *, multi: bool) -> list[TableLike]:
    """`whole` as one chunk, or cut at random row boundaries into several,
    some of them empty."""
    count = whole.table.num_rows if isinstance(whole, Frame) else len(whole.rows)
    if not multi:
        return [whole]
    cuts = sorted(rng.randint(0, count) for _ in range(rng.randint(1, 4)))
    bounds = list(zip([0, *cuts], [*cuts, count], strict=True))
    if isinstance(whole, Frame):
        return [
            Frame(columns=whole.columns, table=whole.table.slice(start, end - start))
            for start, end in bounds
        ]
    return [Table(columns=whole.columns, rows=whole.rows[start:end]) for start, end in bounds]


def _both(chunks: list[TableLike]) -> tuple[OutputProfile, OutputProfile]:
    def factory() -> Iterator[TableLike]:
        return iter(chunks)

    def run(compute: Callable[..., OutputProfile]) -> OutputProfile:
        return compute(
            factory,
            role_columns=_ROLES,
            enrichment=EnrichmentLookup(FakeGraphReader(arcs=_ARCS), parties={}),
        )

    return run(compute_output_profile_from_chunks), run(reference_output_profile_from_chunks)


@pytest.mark.parametrize("multi", [False, True], ids=["one_chunk", "chunked"])
@pytest.mark.parametrize("kind", ["frame", "table"])
def test_arrow_profile_matches_the_row_loop_oracle(kind: str, multi: bool) -> None:
    rng = random.Random(f"20261005-{kind}-{multi}")
    build = _random_frame if kind == "frame" else _random_table
    for case in range(120):
        rows = rng.choice((0, 1, 2, rng.randint(3, 40), rng.randint(40, 400)))
        chunks = _split(rng, build(rng, rows), multi=multi)
        arrow, oracle = _both(chunks)
        assert arrow == oracle, f"case {case}: {rows} rows in {len(chunks)} chunk(s)"
