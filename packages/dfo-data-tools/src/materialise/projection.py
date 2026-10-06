"""The column layout of long (unpivoted) output, applied at the writer boundary.

Reshape works in the grammar's own column names: decode adds `stat_var_<code_system>` label
columns, groupby collapses over them, and `options.columns`, `groupby.by` and `pivot` name them.
`LongProjection` maps that table to the published layout as the last step before the writer:

- `stat_var`, then `stat_var_label` when the table decodes no code system;
- each entity role in declared order, as the entity (`<role>`), its name (`<role>_label`) and
  any requested enrichment (`_iso2`, `_iso3`, `_type`);
- each decoded code system in property order, as `<code_system>` (the value's code dcid) and
  `<code_system>_label` (the decoded label);
- `time_period`, `value`;
- the unit split into `unit` (the unit node dcid), `unit_label`, `currency` (ISO 4217),
  `price_basis` (`current` or `constant`) and `base_year`;
- `measurementMethod` and `provenance`, each followed by its `_label`;
- any other column, in the order the table carries it.

`status` is dropped: the grammar reserves it and no source fills it. The projection reads the
table and leaves every frame alone, so frames, collapse frames and their content versions keep
the grammar names.

Names come from a `names` mapping (cell value to name) the caller reads for each table
(`LongProjection.refs_to_name`). A name cell left empty beside a non-empty code counts as
unavailable, and so does an axis code cell whose label two codes of its system share. `apply`
reports those counts per column so the caller can warn.

`measurementMethod_label` holds only a name that differs from the method's dcid, and its empty
cells are not counted as unavailable. Many methods carry their dcid as their graph name, and some
have no name, so an empty method name is normal.
"""

from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from operator import itemgetter
from typing import Literal

from dfo_frame_grammar.grammar import UnitRequest

from materialise.errors import MaterialisationUsageError
from materialise.reshape import Table, TableLike

type Row = tuple[object, ...]
type Names = Mapping[str, str]
type PriceBasis = Literal["current", "constant"]
type Cell = Callable[[Row, Names], object]
type Unavailable = Callable[[Row, Names], bool]

_STAT_VAR_COLUMN = "stat_var"
_STATUS_COLUMN = "status"
_UNIT_COLUMN = "unit"
_AXIS_SOURCE_PREFIX = f"{_STAT_VAR_COLUMN}_"
"""Decode names each code-system column `stat_var_<code_system>`
(`materialise.reshape.decode_columns`)."""

_LABEL_SUFFIX = "_label"

_ENRICHMENT_FIELD_ORDER = ("iso2", "iso3", "type")
"""The requested enrichment fields that follow an entity's name."""

_METHOD_COLUMN = "measurementMethod"
_PROVENANCE_COLUMN = "provenance"

_CURRENCY_DCID_PREFIX = "Currency"
"""`CurrencyUnit` nodes are `Currency<ISO>`."""


def axis_source_column(code_system: str) -> str:
    """The decoded column reshape names for `code_system`."""
    return f"{_AXIS_SOURCE_PREFIX}{code_system}"


@dataclass(frozen=True)
class UnitParts:
    """One `unit` cell split into the published unit columns.

    `unit` is the unit node dcid, the cell itself for a native unit and `Currency<ISO>` for a
    converted one. `currency` is the ISO 4217 code, `None` for a non-currency unit.
    """

    unit: str
    currency: str | None
    price_basis: PriceBasis | None
    base_year: int | None


@dataclass(frozen=True)
class NativeCurrency:
    """The currency of a native unit node: its ISO 4217 code, and the base year of a unit in
    constant prices of that year (`ChainedUSDollarBaseYr2017`), else `None`."""

    currency: str
    base_year: int | None = None


def split_unit(
    cell: object,
    *,
    currency_of: Callable[[str], NativeCurrency | None],
    native_price_basis: PriceBasis | None,
) -> UnitParts | None:
    """The unit columns of one `unit` cell, `None` for an empty cell.

    A converted cell (`UnitRequest`) names one concrete currency per row, a role-currency
    request included (`validate_unit_request`), and states its own prices and base year. A
    native unit is a currency when `currency_of` gives its currency. A unit with a base year
    is in constant prices of that year. Any other takes `native_price_basis`, the price basis
    of the row's StatVar.
    """
    if isinstance(cell, UnitRequest):
        return UnitParts(
            unit=f"{_CURRENCY_DCID_PREFIX}{cell.currency}",
            currency=cell.currency,
            price_basis="constant" if cell.prices == "constant" else "current",
            base_year=cell.base_year,
        )
    if not isinstance(cell, str) or not cell:
        return None
    native = currency_of(cell)
    if native is None:
        return UnitParts(unit=cell, currency=None, price_basis=None, base_year=None)
    if native.base_year is not None:
        return UnitParts(
            unit=cell, currency=native.currency, price_basis="constant", base_year=native.base_year
        )
    return UnitParts(
        unit=cell, currency=native.currency, price_basis=native_price_basis, base_year=None
    )


@dataclass(frozen=True)
class UnitNaming:
    """What the unit columns read beyond the cell. `currency_of` gives the currency of a
    native unit node that is one. `native_price_basis` gives the price basis of a row's
    native currency unit from its `stat_var` cell. `label` names a split unit, given the unit
    node's name when one was read."""

    currency_of: Callable[[str], NativeCurrency | None]
    native_price_basis: Callable[[object], PriceBasis | None]
    label: Callable[[UnitParts, str | None], str | None]


@dataclass(frozen=True)
class ProjectedTable:
    """A table in the long layout, with the number of rows whose cell in each name or code
    column is unavailable. A column with none is absent from `unavailable`."""

    table: Table
    unavailable: Mapping[str, int]


@dataclass(frozen=True)
class LongProjection:
    """One output layout: `columns` in order, each filled by the matching entry of `cells`
    from a row of a table whose columns are exactly `source_columns` and the names read for
    it. `unavailable` pairs a column with the test of a row whose cell there is unavailable.
    `named_positions` are the source positions whose cells want a name. `renamed` maps a
    source column whose values appear under another name to that name."""

    source_columns: tuple[str, ...]
    columns: tuple[str, ...]
    cells: tuple[Cell, ...]
    unavailable: tuple[tuple[str, Unavailable], ...]
    named_positions: tuple[int, ...]
    renamed: Mapping[str, str]

    def refs_to_name(self, table: TableLike) -> tuple[str, ...]:
        """The distinct non-empty string cells of `table` that a name column reads, in
        first-seen order."""
        refs: dict[str, None] = {}
        for row in table.rows:
            for position in self.named_positions:
                cell = row[position]
                if isinstance(cell, str) and cell:
                    refs[cell] = None
        return tuple(refs)

    def apply(self, table: TableLike, *, names: Names) -> ProjectedTable:
        """`table` in this layout, its name cells read from `names`. Raises
        `MaterialisationUsageError` when `table`'s columns differ from the ones the layout was
        planned for."""
        if tuple(table.columns) != self.source_columns:
            raise MaterialisationUsageError(
                f"a long projection planned for columns {self.source_columns!r} was applied to "
                f"a table with columns {tuple(table.columns)!r}"
            )
        source_rows = table.rows
        cells = self.cells
        rows = tuple(tuple(cell(row, names) for cell in cells) for row in source_rows)
        counts: Counter[str] = Counter()
        for column, is_unavailable in self.unavailable:
            missing = sum(1 for row in source_rows if is_unavailable(row, names))
            if missing:
                counts[column] = missing
        return ProjectedTable(table=Table(columns=self.columns, rows=rows), unavailable=counts)


def plan_long_projection(
    source_columns: Sequence[str],
    *,
    entity_columns: Sequence[str],
    axis_codes: Mapping[str, Mapping[str, str | None]],
    unit_naming: UnitNaming,
) -> LongProjection:
    """The canonical layout for a long table whose columns are `source_columns`.

    `entity_columns` are the frame's declared roles. `axis_codes` maps each code system the
    deployment declares to `{label: code dcid}`. A decoded cell holds a label, and after a
    groupby that rolls up levels or sums over StatVars it is the only fact the row keeps about
    its code, so the code column reads its value from the label. Groupby already treats a label
    as the identity of its code within one system, so the lookup gives the code the row was
    grouped under. A label two codes of one system share maps to `None`: its code cell stays
    empty and counts as unavailable. A label the mapping lacks (a declared absence's
    `absent_reading`, an empty decode gap) gives an empty code cell.
    """
    source = tuple(source_columns)
    position = {name: index for index, name in enumerate(source)}
    columns: list[str] = []
    cells: list[Cell] = []
    unavailable: list[tuple[str, Unavailable]] = []
    named: list[int] = []
    used: set[str] = {_STATUS_COLUMN}
    renamed: dict[str, str] = {}

    def add(name: str, cell: Cell, missing: Unavailable | None = None) -> None:
        columns.append(name)
        cells.append(cell)
        if missing is not None:
            unavailable.append((name, missing))

    def take(name: str) -> None:
        at = itemgetter(position[name])
        add(name, lambda row, _names: at(row))
        used.add(name)

    def take_named(name: str) -> None:
        take(name)
        at = position[name]
        named.append(at)
        add(
            f"{name}{_LABEL_SUFFIX}",
            lambda row, names: _name_of(row[at], names),
            lambda row, names: _name_missing(row[at], names),
        )

    def take_method() -> None:
        take(_METHOD_COLUMN)
        at = position[_METHOD_COLUMN]
        named.append(at)
        add(f"{_METHOD_COLUMN}{_LABEL_SUFFIX}", lambda row, names: _method_name(row[at], names))

    decodes_axes = any(axis_source_column(code_system) in position for code_system in axis_codes)
    if _STAT_VAR_COLUMN in position:
        if decodes_axes:
            take(_STAT_VAR_COLUMN)
        else:
            take_named(_STAT_VAR_COLUMN)

    for entity in entity_columns:
        if entity in position:
            take_named(entity)
        for field in _ENRICHMENT_FIELD_ORDER:
            if f"{entity}_{field}" in position:
                take(f"{entity}_{field}")

    for code_system in sorted(axis_codes):
        source_column = axis_source_column(code_system)
        if source_column not in position:
            continue
        label_at = itemgetter(position[source_column])
        codes = axis_codes[code_system]
        add(
            code_system,
            lambda row, _names, label_at=label_at, codes=codes: codes.get(label_at(row)),
            lambda row, _names, label_at=label_at, codes=codes: _shared_label(label_at(row), codes),
        )
        add(f"{code_system}{_LABEL_SUFFIX}", lambda row, _names, label_at=label_at: label_at(row))
        used.add(source_column)
        renamed[source_column] = f"{code_system}{_LABEL_SUFFIX}"

    for name in ("time_period", "value"):
        if name in position:
            take(name)

    if _UNIT_COLUMN in position:
        _add_unit_columns(
            add,
            unit_at=position[_UNIT_COLUMN],
            stat_var_at=position.get(_STAT_VAR_COLUMN),
            naming=unit_naming,
        )
        named.append(position[_UNIT_COLUMN])
        used.add(_UNIT_COLUMN)

    if _METHOD_COLUMN in position:
        take_method()
    if _PROVENANCE_COLUMN in position:
        take_named(_PROVENANCE_COLUMN)

    for name in source:
        if name not in used:
            take(name)

    return LongProjection(
        source_columns=source,
        columns=tuple(columns),
        cells=tuple(cells),
        unavailable=tuple(unavailable),
        named_positions=tuple(dict.fromkeys(named)),
        renamed=renamed,
    )


def _add_unit_columns(
    add: Callable[[str, Cell, Unavailable | None], None],
    *,
    unit_at: int,
    stat_var_at: int | None,
    naming: UnitNaming,
) -> None:
    """Adds `unit`, `unit_label`, `currency`, `price_basis` and `base_year`, each read from
    the row's split unit. Splits and labels repeat across rows, so both are memoised."""

    @functools.cache
    def split(cell: object, price_basis: PriceBasis | None) -> UnitParts | None:
        return split_unit(cell, currency_of=naming.currency_of, native_price_basis=price_basis)

    @functools.cache
    def label(parts: UnitParts, name: str | None) -> str | None:
        return naming.label(parts, name)

    def parts_of(row: Row) -> UnitParts | None:
        price_basis = (
            naming.native_price_basis(row[stat_var_at]) if stat_var_at is not None else None
        )
        return split(row[unit_at], price_basis)

    def label_of(row: Row, names: Names) -> str | None:
        parts = parts_of(row)
        if parts is None:
            return None
        return label(parts, names.get(parts.unit))

    def part(field: str) -> Cell:
        def cell(row: Row, _names: Names) -> object:
            parts = parts_of(row)
            return getattr(parts, field) if parts is not None else None

        return cell

    add(_UNIT_COLUMN, part("unit"), None)
    add(
        f"{_UNIT_COLUMN}{_LABEL_SUFFIX}",
        label_of,
        lambda row, names: parts_of(row) is not None and label_of(row, names) is None,
    )
    add("currency", part("currency"), None)
    add("price_basis", part("price_basis"), None)
    add("base_year", part("base_year"), None)


def _name_of(cell: object, names: Names) -> str | None:
    return names.get(cell) if isinstance(cell, str) else None


def _method_name(cell: object, names: Names) -> str | None:
    """A method's name when the graph gives one other than the method's dcid."""
    name = _name_of(cell, names)
    return name if name != cell else None


def _name_missing(cell: object, names: Names) -> bool:
    return isinstance(cell, str) and cell != "" and cell not in names


def _shared_label(label: object, codes: Mapping[str, str | None]) -> bool:
    return isinstance(label, str) and label in codes and codes[label] is None


def long_output_selection(
    select: Sequence[str] | None, *, axis_columns: Sequence[str]
) -> tuple[str, ...] | None:
    """The grammar-named columns a long output keeps for an `options.columns` selection.

    `axis_columns` are the decoded `stat_var_<code_system>` columns the table carries. They
    describe `stat_var`, which is grain and always kept, so a selection that names none of them
    keeps all of them, and one that names some keeps those. `None` (no selection) stays `None`.
    """
    if select is None:
        return None
    if any(column in select for column in axis_columns):
        return tuple(select)
    return (*select, *axis_columns)
