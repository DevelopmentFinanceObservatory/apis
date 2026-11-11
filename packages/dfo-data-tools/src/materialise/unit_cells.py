"""Reads a frame1 `unit` cell in its JSON form: `None`, a unit ref string, a converted unit
`{currency, prices, base_year?, deflation_basis?}`, or a composed unit
`{numerator, denominator}`.

Presentation groups, lists and selects a unit by `unit_key_text`. `shared_unit_key` finds the
one case in which two different cells share that text. `unit_cell_text` is the text a tabular
writer gives a `Frame` cell that holds a converted or composed unit. `unit_identity` is what
two series' units are compared by on a card, and `composed_unit_words` is a composed unit's two
sides as display text. `stored_unit_key` matches a unit key stored before converted units
named their deflation basis against the cells of the frame being drawn.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from typing import cast, get_args

from dfo_frame_grammar import DeflationBasis
from dfo_frame_grammar.grammar import ComposedUnit, UnitRequest
from dfo_spec_kit.identity import canonicalize

__all__ = [
    "StoredUnitAmbiguous",
    "composed_unit_sides",
    "composed_unit_words",
    "shared_unit_key",
    "stored_unit_key",
    "unit_cell_text",
    "unit_identity",
    "unit_key_text",
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


class StoredUnitAmbiguous(ValueError):
    """A stored unit key without a deflation basis meets a frame holding that unit under more
    than one basis. `cells` are those cells' key texts, sorted."""

    def __init__(self, key: str, *, cells: Sequence[str]) -> None:
        super().__init__(
            f"stored unit {key!r} names no deflation basis, and the frame holds it under "
            f"{len(cells)}: {', '.join(cells)}"
        )
        self.key = key
        self.cells = tuple(cells)


def stored_unit_key(key: str, *, held: Collection[str]) -> str:
    """Returns the key text of the frame cell a stored unit key binds, among `held`, the key
    texts of the frame's `unit` cells.

    A key `held` contains binds itself, so a key that names its basis matches exactly. A key
    stored before converted units named their deflation basis reads
    `currency/prices[/base_year]`. Where the frame holds that unit under exactly one basis,
    as `currency/prices[/base_year]/basis`, the key binds that cell, and where it holds none
    the key is returned as it is. Composed units are matched by their JCS text alone.

    New writes name the basis, so this rule applies only to stored select values and series
    coordinates. Select matching (`assembler.resolve`) and export coordinate matching
    (`assembler.pipeline.export_component`) both call it, so the two read a stored unit
    alike. It can be deleted once no stored revision holds a basis-less constant unit.

    Raises:
        StoredUnitAmbiguous: the frame holds the unit under more than one basis.
    """
    if key in held:
        return key
    prefix = f"{key}/"
    cells = sorted(
        cell
        for cell in held
        if cell.startswith(prefix) and cell.removeprefix(prefix) in _DEFLATION_BASES
    )
    if len(cells) > 1:
        raise StoredUnitAmbiguous(key, cells=cells)
    return cells[0] if cells else key


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
