"""Reads a frame1 `unit` cell in its JSON form: `None`, a unit ref string, a converted unit
`{currency, prices, base_year?, deflation_basis?}`, or a composed unit
`{numerator, denominator}`.

Presentation groups, lists and selects a unit by `unit_key_text`. `shared_unit_key` finds the
one case in which two different cells share that text. `unit_cell_text` is the text a tabular
writer gives a `Frame` cell that holds a converted or composed unit. `unit_identity` is what
two series' units are compared by on a card, and `composed_unit_words` is a composed unit's two
sides as display text. `bind_unit_key` binds a unit key that names no deflation basis to the
cell of the frame being drawn that differs from it only by its basis.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import cast, get_args

from dfo_frame_grammar import DeflationBasis
from dfo_frame_grammar.grammar import ComposedUnit, UnitRequest
from dfo_spec_kit.identity import canonicalize

__all__ = [
    "UnitKeyAmbiguous",
    "bind_unit_key",
    "composed_unit_sides",
    "composed_unit_words",
    "shared_unit_key",
    "unit_cell_text",
    "unit_identity",
    "unit_key_text",
    "units_by_identity",
]


def composed_unit_sides(unit: object) -> tuple[object, object] | None:
    """Returns the numerator and denominator of a composed `unit` cell, `None` for any other
    cell."""
    if isinstance(unit, Mapping) and "numerator" in unit:
        composed = cast("Mapping[str, object]", unit)
        return composed["numerator"], composed["denominator"]
    return None


def composed_unit_words(unit: object) -> tuple[str, str] | None:
    """Returns the numerator's and the denominator's words for a composed unit in its payload
    form, `None` for any other unit.

    A string side reads as it is, a unit's label. A StatVar side reads as its `label`. A
    converted side reads as `unit_key_text` gives it,
    `currency/prices[/base_year][/deflation_basis]`. The words are for display, and a unit is
    matched and selected by its key text."""
    sides = composed_unit_sides(unit)
    if sides is None:
        return None
    numerator, denominator = sides
    return _side_words(numerator), _side_words(denominator)


def _side_words(side: object) -> str:
    if isinstance(side, str):
        return side
    part = cast("Mapping[str, object]", side)
    return cast("str", part["label"]) if "label" in part else unit_key_text(part)


def unit_cell_text(cell: UnitRequest | ComposedUnit) -> str:
    """Returns the JCS serialisation of the cell's `as_json`, the same object frame1 carries,
    for example `{"base_year":2024,"currency":"EUR","prices":"constant"}`."""
    return canonicalize(cell.as_json()).decode("utf-8")


def unit_key_text(unit: Mapping[str, object]) -> str:
    """Returns the text a converted or composed `unit` cell is grouped and displayed by.

    A converted unit renders as `currency/prices[/base_year][/deflation_basis]`, keeping
    converted units distinct, two deflation bases included. A composed unit renders as its JCS
    text, such as
    `{"denominator":{"stat_var":"dcid:Count_Person"},"numerator":"USDollar"}`, so two composed
    units share a key only when they are equal."""
    if composed_unit_sides(unit) is not None:
        return canonicalize(dict(unit)).decode("utf-8")
    parts = (unit["currency"], unit["prices"], unit.get("base_year"), unit.get("deflation_basis"))
    return "/".join(str(part) for part in parts if part is not None)


def shared_unit_key(*, cells: Sequence[object]) -> str | None:
    """Returns the key text a string `unit` cell shares with a converted or composed cell among
    `cells`, or `None` when every cell keys apart. `"USD/current"` beside
    `{"currency": "USD", "prices": "current"}` is such a pair.

    Grouping by key text would put the two cells in one series, so a frame's structural view
    refuses such a frame and the publish door does not store one."""
    refs = {cell for cell in cells if isinstance(cell, str)}
    if not refs:
        return None
    for cell in cells:
        if isinstance(cell, Mapping):
            key = unit_key_text(cast("Mapping[str, object]", cell))
            if key in refs:
                return key
    return None


_DEFLATION_BASES: frozenset[str] = frozenset(get_args(DeflationBasis))


class UnitKeyAmbiguous(ValueError):
    """A unit key naming no deflation basis meets a frame holding that unit under more than
    one basis. `cells` are those cells' key texts, sorted."""

    def __init__(self, key: str, *, cells: Sequence[str]) -> None:
        super().__init__(
            f"unit {key!r} names no deflation basis, and the frame holds it under "
            f"{len(cells)}: {', '.join(cells)}"
        )
        self.key = key
        self.cells = tuple(cells)


def bind_unit_key(key: str, *, held: Collection[str]) -> str:
    """Returns the key text of the frame cell a unit key binds, among `held`, the key texts of
    the frame's `unit` cells.

    A key `held` contains binds itself, so a key that names its basis matches exactly. A key
    naming a constant-price converted unit without its basis, as `USD/constant/2024`, binds
    the frame's cell for that currency, prices and base year when the frame holds it under
    exactly one basis, as `USD/constant/2024/donor`, and is returned as it is when the frame
    holds it under none. A composed unit's key binds by the same rule on each converted side,
    so a key whose sides name no basis binds the one composed cell that differs from it only
    by its sides' bases. This binds the one cell the frame holds for the fields a key names,
    as a stored select's facet binds a frame's one value (`assembler.rolling.StoredSelect`).
    It applies alike to select values and to stored series coordinates: select matching
    (`assembler.resolve`) and export coordinate matching
    (`assembler.pipeline.export_component`) both call it.

    Raises:
        UnitKeyAmbiguous: the frame holds the unit under more than one basis.
    """
    if key in held:
        return key
    cells = sorted(cell for cell in held if _without_basis(cell) == key)
    if len(cells) > 1:
        raise UnitKeyAmbiguous(key, cells=cells)
    return cells[0] if cells else key


def _without_basis(cell: str) -> str | None:
    """A frame `unit` cell's key text with every deflation basis dropped, `None` for a cell
    that names none. A composed cell's key is its JCS text, read back as JSON."""
    if cell.startswith("{"):
        unit = json.loads(cell)
        sides = composed_unit_sides(unit)
        if sides is None:
            return None
        numerator, denominator = (_side_without_basis(side) for side in sides)
        if (numerator, denominator) == sides:
            return None
        return canonicalize({**unit, "numerator": numerator, "denominator": denominator}).decode(
            "utf-8"
        )
    parts = cell.split("/")
    if len(parts) > 2 and parts[1] == "constant" and parts[-1] in _DEFLATION_BASES:
        return "/".join(parts[:-1])
    return None


def _side_without_basis(side: object) -> object:
    if isinstance(side, Mapping) and "deflation_basis" in side:
        return {name: value for name, value in side.items() if name != "deflation_basis"}
    return side


def unit_identity(
    *, series_id: object, payload_unit: object, frame_unit_keys: Mapping[str, object] | None
) -> tuple[str, object]:
    """Returns the value two series' units are compared by on a card.

    A series whose id `frame_unit_keys` holds compares by that key, the text its frame `unit`
    cell is grouped by. A payload unit is a label, and two units can share a graph name or the
    same first 200 characters, so equal labels do not make equal units. A series with no frame
    key, in a revision stored without series coordinates or a catalogue fixture, compares by
    the canonical JSON of its payload unit."""
    if frame_unit_keys is not None and series_id in frame_unit_keys:
        return "frame", frame_unit_keys[cast("str", series_id)]
    return "payload", json.dumps(payload_unit, sort_keys=True)


def units_by_identity(
    *,
    entries: Iterable[tuple[object, object]],
    frame_unit_keys: Mapping[str, object] | None,
) -> dict[tuple[str, object], object]:
    """Returns each distinct unit among `entries`, a `(series_id, payload_unit)` pair per
    series or row, keyed by `unit_identity`. More than one key means the units differ."""
    return {
        unit_identity(
            series_id=series_id, payload_unit=payload_unit, frame_unit_keys=frame_unit_keys
        ): payload_unit
        for series_id, payload_unit in entries
    }
