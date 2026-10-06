"""Regenerate the two fixtures `test_answer_invariants.i9_latency` reads.

`tests/fixtures/answer_invariants_requests_c669cc874.json` is a fixed record: what every door
of the grid measured at commit c669cc874 (`test_answer_invariants.measured`). It is
regenerated only from a copy of that tree, and only when the grid or the recorder changes.

`tests/fixtures/answer_invariants_requests_uncompared.json` is recorded from the current
tree: the doors that no door at c669cc874 with the same answer bounds and that have no
preview-off door with the same outcome to bound them. It may only shrink. `write` refuses a
list that adds a door, changes a door's answer or raises a door's requests or depth, unless
`--grow` names the reason.

Each command runs from the root of the tree it measures, with `DFO_SPEC_KIT_DIR` set.

Measure the current tree, then rebuild the uncompared list against the base fixture on disk:

    uv run python -m tests.engine._answer_invariants_record measure /tmp/branch.jsonl
    uv run python -m tests.engine._answer_invariants_record write --branch /tmp/branch.jsonl

Regenerate the base fixture as well. Make the copy with `git archive c669cc874 | tar -x -C
<copy>`, sync its own environment there, copy `test_answer_invariants.py`,
`_answer_invariants_world.py`, `_grid_cells.py` and this module into its `tests/engine/`,
and copy `tests/_covering_array.py` into its `tests/`:

    (cd <copy> && uv run python -m tests.engine._answer_invariants_record measure /tmp/base.jsonl)
    uv run python -m tests.engine._answer_invariants_record write \\
        --base /tmp/base.jsonl --branch /tmp/branch.jsonl

`measure` runs the grid's full product (`DFO_FULL_GRIDS=1`) under pytest with this module as
a plugin, which replaces the grid's checks with one that writes each door's measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from tests._covering_array import FULL_GRIDS_ENV

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_BASE_PATH = _FIXTURES / "answer_invariants_requests_c669cc874.json"
_UNCOMPARED_PATH = _FIXTURES / "answer_invariants_requests_uncompared.json"
_OUT_ENV = "ANSWER_INVARIANTS_RECORD_OUT"
_GRID = "tests/engine/test_answer_invariants.py"
_GRID_TEST = "test_every_cell_holds_every_invariant"
_FIELDS = ("outcome", "served", "requests", "depth")
# Recorded only for a door whose nearest-real relaxations, or whose alternates and
# re-election siblings, walked their members, or whose Specs built facet options.
_WALK_FIELDS = ("relaxation_walk", "alternate_walk", "facet_options")

type DoorKey = tuple[str, str, int]
type Doors = dict[DoorKey, dict[str, Any]]


@pytest.fixture(autouse=True, scope="session")
def _write_measurements() -> Iterator[None]:
    """Under `measure`, replace the grid's checks with one that writes every door's
    measurement to this worker's file. Without `measure` the fixture does nothing."""
    out = os.environ.get(_OUT_ENV)
    if out is None:
        yield
        return
    from tests.engine import test_answer_invariants as grid

    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    with open(f"{out}.{worker}", "w") as handle:

        def record(run) -> list[str]:
            positions: dict[str, int] = {}
            for door, answer in run.answers():
                index = positions.setdefault(door, 0)
                positions[door] += 1
                row = {
                    "cell": str(run.cell),
                    "door": grid._DOOR_KEYS[door],
                    "index": index,
                    **grid.measured(answer),
                }
                handle.write(json.dumps(row) + "\n")
            return []

        grid.CHECKS.clear()
        grid.CHECKS["record"] = record
        yield


def measure(out: Path, *, workers: int) -> None:
    """Run the grid on the tree in the working directory and write its doors to `out`."""
    for stale in out.parent.glob(f"{out.name}.*"):
        stale.unlink()
    env = {k: v for k, v in os.environ.items() if not k.startswith("SEARCH_GRAPH_")}
    env[_OUT_ENV] = str(out)
    env[FULL_GRIDS_ENV] = "1"
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    command += ["-p", "tests.engine._answer_invariants_record", "-n", str(workers)]
    command += [_GRID, "-k", _GRID_TEST]
    subprocess.run(command, env=env, check=True)
    parts = sorted(out.parent.glob(f"{out.name}.*"))
    lines = sorted(line for part in parts for line in part.read_text().splitlines())
    if not lines:
        raise SystemExit("measure: the grid wrote no door")
    out.write_text("\n".join(lines) + "\n")
    for part in parts:
        part.unlink()
    print(f"measure: {len(lines)} doors in {out}")


def _measured(path: Path) -> Doors:
    doors: Doors = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        doors[row["cell"], row["door"], row["index"]] = {
            f: row[f] for f in (*_FIELDS, *_WALK_FIELDS) if f in row
        }
    return doors


def _held(path: Path) -> Doors:
    """The doors a fixture holds, none when it does not exist."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {
        (cell, door, position): data["records"][index]
        for cell, doors in data["doors"].items()
        for door, indices in doors.items()
        for position, index in enumerate(indices)
    }


def _dump(doors: Doors) -> str:
    """`doors` as fixture text: each distinct record once, then each cell's doors as indices
    into the records, one cell a line."""
    records: list[str] = []
    index: dict[str, int] = {}
    cells: dict[str, dict[str, list[int]]] = {}
    for (cell, door, position), record in sorted(doors.items()):
        text = json.dumps(record, sort_keys=True)
        if text not in index:
            index[text] = len(records)
            records.append(text)
        held = cells.setdefault(cell, {}).setdefault(door, [])
        if len(held) != position:
            raise SystemExit(f"write: {cell} {door} skips position {len(held)}")
        held.append(index[text])
    lines = ["{", ' "records": [', ",\n".join("  " + r for r in records), " ],", ' "doors": {']
    lines.append(
        ",\n".join(f"  {json.dumps(c)}: {json.dumps(d)}" for c, d in sorted(cells.items()))
    )
    lines += [" }", "}"]
    return "\n".join(lines) + "\n"


def _same_answer(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return a["outcome"] == b["outcome"] and a["served"] == b["served"]


def uncompared(base: Doors, branch: Doors) -> Doors:
    """The branch doors `i9_latency` holds to neither a door at c669cc874 nor their own
    preview-off door."""
    doors: Doors = {}
    for key, now in branch.items():
        cell, door, position = key
        at_base = base.get(key)
        if at_base is not None and _same_answer(at_base, now):
            continue
        if "/preview_on/" in cell:
            off = branch.get((cell.replace("/preview_on/", "/preview_off/"), door, position))
            if off is not None and off["outcome"] == now["outcome"]:
                continue
        doors[key] = now
    return doors


def growth(held: Doors, now: Doors) -> list[str]:
    """Each way `now` grows on the list `held`: a door added, a door answering differently,
    or a door requesting more or running deeper."""
    grown: list[str] = []
    for key, record in sorted(now.items()):
        name = f"{key[0]} {key[1]} {key[2]}"
        old = held.get(key)
        if old is None:
            grown.append(f"{name}: added")
        elif not _same_answer(old, record):
            grown.append(f"{name}: answers {record['outcome']} against {old['outcome']}")
        else:
            over = {
                request: f"{count} against {old['requests'].get(request, 0)}"
                for request, count in record["requests"].items()
                if count > old["requests"].get(request, 0)
            }
            if over:
                grown.append(f"{name}: requests {over}")
            if record["depth"] > old["depth"]:
                grown.append(f"{name}: runs {record['depth']} steps against {old['depth']}")
    return grown


def write(*, base: Path | None, branch: Path, grow: str | None) -> None:
    """Write the uncompared list from `branch`, and the base fixture from `base` when given."""
    base_doors = _measured(base) if base is not None else _held(_BASE_PATH)
    if not base_doors:
        raise SystemExit(f"write: {_BASE_PATH.name} is missing; pass --base")
    listed = uncompared(base_doors, _measured(branch))
    grown = growth(_held(_UNCOMPARED_PATH), listed) if _UNCOMPARED_PATH.exists() else []
    if grown and grow is None:
        shown = "\n  ".join(grown[:20])
        raise SystemExit(
            f"write: refused, {_UNCOMPARED_PATH.name} would grow in {len(grown)} ways and "
            f"nothing was written. Pass --grow with the reason.\n  {shown}"
        )
    if grown:
        print(f"write: {_UNCOMPARED_PATH.name} grows in {len(grown)} ways: {grow}")
    if base is not None:
        _BASE_PATH.write_text(_dump(base_doors))
        print(f"write: {_BASE_PATH.name}, {len(base_doors)} doors")
    _UNCOMPARED_PATH.write_text(_dump(listed))
    print(f"write: {_UNCOMPARED_PATH.name}, {len(listed)} doors")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("measure", help="measure every door of the tree in the cwd")
    measuring.add_argument("out", type=Path)
    measuring.add_argument("--workers", type=int, default=4)
    writing = commands.add_parser("write", help="write the fixtures from measurements")
    writing.add_argument("--branch", type=Path, required=True, help="the current tree's doors")
    writing.add_argument("--base", type=Path, help="doors measured on a copy of c669cc874")
    writing.add_argument("--grow", metavar="REASON", help="why the uncompared list may grow")
    args = parser.parse_args()
    if args.command == "measure":
        measure(args.out, workers=args.workers)
    else:
        write(base=args.base, branch=args.branch, grow=args.grow)


if __name__ == "__main__":
    main()
