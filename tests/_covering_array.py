"""Covering arrays: the cells of a grid that a pull request runs in place of its full product.

A grid is a product of factors, each a tuple of values, less the cells its `admits`
predicate refuses. A strength-t covering array is a set of admitted cells holding every
t-tuple of factor values that some admitted cell holds. `cover` builds one greedily: it
starts from the cells in product order and repeatedly takes the cell holding the most
t-tuples not yet held, the earliest in product order on a tie. It uses only integer
arithmetic and list order, so its output is the same on every run and Python version.

A grid test runs its committed cells (`tests/engine/_grid_cells.py`) by default and its
full product when `DFO_FULL_GRIDS=1` (`full_grids`). Regenerate the committed cells after
changing a grid's factors, its `admits` predicate or its strength:

    uv run python -m tests._covering_array
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

FULL_GRIDS_ENV = "DFO_FULL_GRIDS"
_CELLS_PATH = Path(__file__).parent / "engine" / "_grid_cells.py"

type Row = tuple[object, ...]


def full_grids() -> bool:
    """Whether the grid tests run their full product: `DFO_FULL_GRIDS=1`.

    Any value other than unset, `0` or `1` raises, so a mistyped switch cannot run the
    committed subset where the full product was asked for.
    """
    value = os.environ.get(FULL_GRIDS_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{FULL_GRIDS_ENV} must be 0 or 1, got {value!r}")
    return value == "1"


@dataclass(frozen=True)
class Grid[T]:
    """A grid: its factors, the strength its committed cells cover, how a cell is built
    from one value of each factor, which cells a request can form, and each cell's id.

    `name` is the constant `tests/engine/_grid_cells.py` holds the grid's cells under."""

    name: str
    factors: dict[str, tuple[object, ...]]
    strength: int
    cell: Callable[..., T]
    cell_id: Callable[[T], str]
    admits: Callable[[T], bool] = lambda _cell: True

    def cells(self) -> Iterator[tuple[Row, T]]:
        """Every admitted cell with its factor values, in product order."""
        for row in itertools.product(*self.factors.values()):
            if self.admits(cell := self.cell(*row)):
                yield row, cell

    def run(self, committed: Iterable[str]) -> list[T]:
        """The admitted cells a test runs, in product order: every one under
        `full_grids()`, otherwise those whose ids `committed` holds."""
        every, chosen = full_grids(), frozenset(committed)
        return [cell for _row, cell in self.cells() if every or self.cell_id(cell) in chosen]


def _tuples(grid: Grid, row: Row) -> Iterator[tuple[int, tuple[object, ...]]]:
    """The t-tuples `row` holds, each as the index of its factor combination and its
    values."""
    combos = itertools.combinations(range(len(grid.factors)), grid.strength)
    for index, combo in enumerate(combos):
        yield index, tuple(row[i] for i in combo)


def cover(grid: Grid) -> list[str]:
    """The ids of a strength-`grid.strength` covering array of `grid`'s admitted cells,
    sorted."""
    rows, cells = zip(*grid.cells(), strict=True)
    keys: dict[tuple[int, tuple[object, ...]], int] = {}
    held: list[list[int]] = []
    holders: list[list[int]] = []
    for position, row in enumerate(rows):
        own = []
        for key in _tuples(grid, row):
            if key not in keys:
                keys[key] = len(holders)
                holders.append([])
            holders[keys[key]].append(position)
            own.append(keys[key])
        held.append(own)
    gains = [len(own) for own in held]
    covered = [False] * len(holders)
    chosen: list[str] = []
    remaining = len(holders)
    while remaining:
        best = gains.index(max(gains))
        chosen.append(grid.cell_id(cells[best]))
        for key in held[best]:
            if covered[key]:
                continue
            covered[key] = True
            remaining -= 1
            for position in holders[key]:
                gains[position] -= 1
    return sorted(chosen)


def uncovered(grid: Grid, committed: Iterable[str]) -> list[str]:
    """Each committed id that names no admitted cell of `grid`, and each t-tuple some
    admitted cell holds and no committed cell does."""
    by_id = {grid.cell_id(cell): row for row, cell in grid.cells()}
    committed = list(committed)
    found = [f"{cell_id} is no admitted cell" for cell_id in committed if cell_id not in by_id]
    held = {
        key for cell_id in committed if cell_id in by_id for key in _tuples(grid, by_id[cell_id])
    }
    missing = {key for row in by_id.values() for key in _tuples(grid, row)} - held
    names = list(itertools.combinations(grid.factors, grid.strength))
    found += sorted(
        f"no committed cell holds {dict(zip(names[index], values, strict=True))}"
        for index, values in missing
    )
    return found


def render(grids: Sequence[Grid]) -> str:
    """The text of `tests/engine/_grid_cells.py` holding each grid's covering array."""
    lines = [
        '"""The cells each grid test runs on a pull request: a covering array of its factors',
        "(`tests/_covering_array.py`). Generated; regenerate with",
        "",
        "    uv run python -m tests._covering_array",
        '"""',
    ]
    for grid in grids:
        cells = cover(grid)
        lines += ["", f"# Strength {grid.strength}: {len(cells)} cells.", f"{grid.name} = ("]
        lines += [f'    "{cell_id}",' for cell_id in cells]
        lines.append(")")
    return "\n".join(lines) + "\n"


def main() -> None:
    from tests.engine import test_answer_invariants, test_read_plan_grid

    grids = [test_answer_invariants.GRID, *test_read_plan_grid.GRIDS]
    _CELLS_PATH.write_text(render(grids))
    print(f"wrote {_CELLS_PATH}")


if __name__ == "__main__":
    main()
