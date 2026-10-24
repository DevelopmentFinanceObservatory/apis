"""The column layout of long (unpivoted) output, applied at the writer boundary.

Reshape works in the grammar's own column names: decode adds `stat_var_<code_system>` label
columns, groupby collapses over them, and `options.columns`, `groupby.by` and `pivot` name them.
`LongProjection` maps that table to the published layout as the last step before the writer:

- `stat_var`, then `stat_var_label` when the table decodes no code system;
- each entity role in declared order, as the entity (`<role>`), its name (`<role>_label`),
  for a party role (`materialise.enrichment.is_party_role`) its party's kind in words
  (`<role>_kind`, `long_kind_cell`), and any requested enrichment (`_iso2`, `_iso3`, `_type`);
- each decoded code system in property order, as `<code_system>` (the value's code dcid) and
  `<code_system>_label` (the decoded label);
- `time_period` (an integer year for a frame of year points, `period_cell`), `value`;
- the unit split into `unit` (the unit node dcid), `unit_label`, `currency` (ISO 4217),
  `price_basis` (`current` or `constant`) and `base_year`;
- `measurementMethod` and `provenance`, each followed by its `_label`;
- any other column, in the order the table carries it.

`status` is dropped: the grammar reserves it and no source fills it. The projection reads the
table and leaves every frame alone, so frames, collapse frames and their content versions keep
the grammar names.

Names come from a `names` mapping (cell value to name) read by the caller for each table in one
lookup (`LongProjection.refs_to_name`). Kinds come from the code index's `parties` rows, read per
row: a `<role>_kind` cell is `long_kind_cell` of the party for the row's `provenance` and the
year of its `time_period`, so a whole total reads `whole total` only in the dataset and years
its marker holds (`PartyRow.whole`). A table with no `provenance` or `time_period` column, or a
row with no readable value in one, gives `total`. A name or kind cell left empty beside a
non-empty entity, and an axis code cell whose label two codes of its system share, count as
unavailable. `apply` reports those counts per column so the caller can warn. The machine party
cells (`materialise.enrichment.PARTY_FIELDS`) belong to the frame layout; the long layout's
`<role>_kind` reads all three.

`measurementMethod_label` holds only a name that differs from the method's dcid, and its empty
cells are not counted as unavailable. Many methods carry their dcid as their graph name, and some
have no name, so an empty method name is normal.

A row is a total when a role `totals_roles` names holds a total there, its `<role>_kind` cell
reading `total` or `whole total` (`split_totals`). The CSV and XLSX writers keep those rows out
of the main table.
"""

from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from operator import itemgetter
from typing import Literal

from dfo_frame_grammar import TIME_PERIOD_COLUMN, parse_time_period
from dfo_frame_grammar.grammar import ComposedUnit, UnitRequest

from code_index.types import PartyRow
from dfo_platform.roles import role_key_axis
from materialise.enrichment import is_party_role, is_whole_total, period_year
from materialise.errors import MaterialisationUsageError
from materialise.reshape import Table, TableLike
from materialise.unit_cells import unit_cell_text

type Row = tuple[object, ...]
type Names = Mapping[str, str]
type PriceBasis = Literal["current", "constant"]
type Cell = Callable[[Row, NodeReads], object]
type Unavailable = Callable[[Row, NodeReads], bool]

_STAT_VAR_COLUMN = "stat_var"
_STATUS_COLUMN = "status"
_UNIT_COLUMN = "unit"
_AXIS_SOURCE_PREFIX = f"{_STAT_VAR_COLUMN}_"
"""Decode names each code-system column `stat_var_<code_system>`
(`materialise.reshape.decode_columns`)."""

_LABEL_SUFFIX = "_label"

_ENRICHMENT_FIELD_ORDER = ("iso2", "iso3", "type")
"""The requested enrichment fields that follow an entity's name and kind."""

KIND_SUFFIX = "_kind"
"""The suffix of a party role's kind column in the long layout (`<role>_kind`)."""

TOTAL_CELL = "total"
WHOLE_TOTAL_CELL = "whole total"
RESIDUAL_CELL = "unallocated"
RECIPIENT_PARTY_CELL = "country or organisation"
"""The `<role>_kind` cell of an official party on the destination axis, where no class tells
recipients apart."""

_YEAR_GRAIN = "year"

_TOTAL_CELLS = frozenset({TOTAL_CELL, WHOLE_TOTAL_CELL})
"""The `<role>_kind` cells of a total."""

_METHOD_COLUMN = "measurementMethod"
_PROVENANCE_COLUMN = "provenance"

_CURRENCY_DCID_PREFIX = "Currency"
"""`CurrencyUnit` nodes are `Currency<ISO>`."""


def long_layout_default_enrichments(entity_columns: Sequence[str]) -> tuple[str, ...]:
    """The enrichment columns long output writes for `entity_columns` with no `columns` token,
    in layout order: each role's `<role>_label`, followed for a party role by its
    `<role>_kind`."""
    columns: list[str] = []
    for entity in entity_columns:
        columns.append(f"{entity}{_LABEL_SUFFIX}")
        if is_party_role(entity):
            columns.append(f"{entity}{KIND_SUFFIX}")
    return tuple(columns)


def long_kind_cell(
    ref: object,
    *,
    role: str,
    parties: Mapping[str, PartyRow],
    period: object,
    provenance: object,
) -> str | None:
    """The `<role>_kind` cell of a long-layout row whose `role` cell is `ref`, `time_period` cell
    `period` and `provenance` cell `provenance`, in words for a reader of the file.

    A total reads `whole total` when it is a whole total for the row's dataset and year
    (`materialise.enrichment.is_whole_total`) and `total` otherwise, and a residual reads
    `unallocated`. A party on the origin axis reads its class, `official` or `private`. A party
    on the destination axis reads `country or organisation`, or `private` for a private one.
    The kind and class are the party's in the row's dataset (`PartyRow.kind_in`,
    `PartyRow.class_in`). A row with no `provenance` reads the kind of every dataset. `None` for
    a ref the code index does not list, one that is not text, and a party that takes no one kind
    for the row (`PartyRow.kind_over`), a party whose kind differs between datasets on a row
    with no `provenance`."""
    party = parties.get(ref) if isinstance(ref, str) else None
    if party is None:
        return None
    dataset = provenance if isinstance(provenance, str) else None
    kind = party.kind_over(() if dataset is None else (dataset,))
    if kind is None:
        return None
    if kind == "total":
        whole = is_whole_total(party, year=period_year(period), provenance=dataset)
        return WHOLE_TOTAL_CELL if whole else TOTAL_CELL
    if kind == "residual":
        return RESIDUAL_CELL
    party_class = party.party_class if dataset is None else party.class_in(dataset)
    if party_class == "private" or role_key_axis(role) == "origin":
        return party_class
    return RECIPIENT_PARTY_CELL


def period_grain(periods: Iterable[object]) -> str | None:
    """The grain of a frame whose `time_period` cells are `periods` when every one is a point at
    that grain (`parse_time_period`), `None` when any is empty or a span, the points hold more
    than one grain, or there are none. Long output reads it once over the whole frame before its
    first row, so a frame mixing year points with year spans writes every period as text."""
    grains: set[str] = set()
    for cell in periods:
        if not isinstance(cell, str) or not cell:
            return None
        period = parse_time_period(cell)
        if period.start != period.end:
            return None
        grains.add(period.grain)
    return grains.pop() if len(grains) == 1 else None


def period_cell(grain: str | None, cell: object) -> object:
    """A `time_period` cell as long output and card exports write it: a year point as an `int`
    when `grain` is `year`, so spreadsheets and Parquet read a number, and any other cell
    unchanged. `grain` is the whole frame's grain from `period_grain`. A span stays text."""
    if grain == _YEAR_GRAIN and isinstance(cell, str) and len(cell) == 4 and cell.isdigit():
        return int(cell)
    return cell


def totals_axes(identity: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    """The flow axes whose total rows long CSV and XLSX output keeps apart, each mapped to the
    entities the identity binds on it: empty for an axis the identity leaves open
    (`open_roles`), the bound entities for one it binds to several. A role bound to one entity
    is what the definition asks for, a total included ("ODA from the UK to developing
    countries, total"), so it never makes a total row. Identity keys keep the caller's
    spelling, so they compare by axis (`dfo_platform.roles.role_key_axis`)."""
    open_roles = identity.get("open_roles")
    entities = identity.get("entities")
    axes: dict[str, tuple[str, ...]] = {}
    if isinstance(entities, Mapping):
        for role, bound in entities.items():
            axis = role_key_axis(str(role))
            if axis is not None and isinstance(bound, list | tuple) and len(bound) > 1:
                axes[axis] = tuple(str(entity) for entity in bound)
    if isinstance(open_roles, Sequence):
        for role in open_roles:
            if (axis := role_key_axis(str(role))) is not None:
                axes[axis] = ()
    return axes


def totals_roles(identity: Mapping[str, object], entity_columns: Sequence[str]) -> tuple[str, ...]:
    """The party roles of `entity_columns` (`is_party_role`) on an axis of `totals_axes`, whose
    total rows long CSV and XLSX output keeps apart."""
    axes = totals_axes(identity)
    return tuple(
        entity
        for entity in entity_columns
        if is_party_role(entity) and role_key_axis(entity) in axes
    )


def split_totals(table: Table, *, roles: Sequence[str]) -> tuple[Table, Table]:
    """`table`, a table in the long layout, split into the rows of parties and residuals and
    the rows where any of `roles` holds a total (its `<role>_kind` cell reads `total` or
    `whole total`). A role the table does not carry splits nothing."""
    positions = [
        list(table.columns).index(column)
        for role in roles
        if (column := f"{role}{KIND_SUFFIX}") in table.columns
    ]
    if not positions:
        return table, Table(columns=table.columns, rows=())
    parties: list[Row] = []
    totals: list[Row] = []
    for row in table.rows:
        (totals if any(row[at] in _TOTAL_CELLS for at in positions) else parties).append(row)
    return Table(columns=table.columns, rows=tuple(parties)), Table(
        columns=table.columns, rows=tuple(totals)
    )


def axis_source_column(code_system: str) -> str:
    """The decoded column reshape names for `code_system`."""
    return f"{_AXIS_SOURCE_PREFIX}{code_system}"


@dataclass(frozen=True)
class NodeReads:
    """What a long output's cells read beyond the row: `names` maps a cell value to its name,
    `parties` a party ref to its code index row, which gives its kind cell."""

    names: Names
    parties: Mapping[str, PartyRow]


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
    of the row's StatVar. A composed cell (`ComposedUnit`) is no one currency: `unit` holds
    the JCS text of its `as_json`, as the tabular writers render it, and the rest are `None`.
    """
    if isinstance(cell, ComposedUnit):
        return UnitParts(
            unit=unit_cell_text(cell),
            currency=None,
            price_basis=None,
            base_year=None,
        )
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

    def apply(
        self, table: TableLike, *, names: Names, parties: Mapping[str, PartyRow]
    ) -> ProjectedTable:
        """`table` in this layout, its name cells read from `names` and its kind cells from
        `parties`, each for the dataset and year of its row. Raises `MaterialisationUsageError`
        when `table`'s columns differ from the ones the layout was planned for."""
        if tuple(table.columns) != self.source_columns:
            raise MaterialisationUsageError(
                f"a long projection planned for columns {self.source_columns!r} was applied to "
                f"a table with columns {tuple(table.columns)!r}"
            )
        source_rows = table.rows
        cells = self.cells
        reads = NodeReads(names=names, parties=parties)
        rows = tuple(tuple(cell(row, reads) for cell in cells) for row in source_rows)
        counts: Counter[str] = Counter()
        for column, is_unavailable in self.unavailable:
            missing = sum(1 for row in source_rows if is_unavailable(row, reads))
            if missing:
                counts[column] = missing
        return ProjectedTable(table=Table(columns=self.columns, rows=rows), unavailable=counts)


def plan_long_projection(
    source_columns: Sequence[str],
    *,
    entity_columns: Sequence[str],
    axis_codes: Mapping[str, Mapping[str, str | None]],
    unit_naming: UnitNaming,
    grain: str | None,
) -> LongProjection:
    """The canonical layout for a long table whose columns are `source_columns`.

    `entity_columns` are the frame's declared roles. `grain` is the grain of the frame's
    periods (`period_grain`), which types its `time_period` cells (`period_cell`).
    `axis_codes` maps each code system the deployment declares to `{label: code dcid}`. A
    decoded cell holds a label, and after a groupby that rolls up levels or sums over StatVars
    it is the only fact the row keeps about its code, so the code column reads its value from the label. Groupby already treats a label
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
        add(name, lambda row, _reads: at(row))
        used.add(name)

    def take_named(name: str) -> None:
        take(name)
        at = position[name]
        named.append(at)
        add(
            f"{name}{_LABEL_SUFFIX}",
            lambda row, reads: _name_of(row[at], reads.names),
            lambda row, reads: _name_missing(row[at], reads.names),
        )

    def add_kind(entity: str) -> None:
        at = position[entity]
        period_at = position.get(TIME_PERIOD_COLUMN)
        provenance_at = position.get(_PROVENANCE_COLUMN)

        def kind(row: Row, reads: NodeReads) -> str | None:
            return long_kind_cell(
                row[at],
                role=entity,
                parties=reads.parties,
                period=None if period_at is None else row[period_at],
                provenance=None if provenance_at is None else row[provenance_at],
            )

        def missing(row: Row, reads: NodeReads) -> bool:
            cell = row[at]
            return isinstance(cell, str) and cell != "" and kind(row, reads) is None

        add(f"{entity}{KIND_SUFFIX}", kind, missing)

    def take_method() -> None:
        take(_METHOD_COLUMN)
        at = position[_METHOD_COLUMN]
        named.append(at)
        add(
            f"{_METHOD_COLUMN}{_LABEL_SUFFIX}",
            lambda row, reads: _method_name(row[at], reads.names),
        )

    decodes_axes = any(axis_source_column(code_system) in position for code_system in axis_codes)
    if _STAT_VAR_COLUMN in position:
        if decodes_axes:
            take(_STAT_VAR_COLUMN)
        else:
            take_named(_STAT_VAR_COLUMN)

    for entity in entity_columns:
        if entity in position:
            take_named(entity)
        if entity in position and is_party_role(entity):
            add_kind(entity)
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
            lambda row, _reads, label_at=label_at, codes=codes: codes.get(label_at(row)),
            lambda row, _reads, label_at=label_at, codes=codes: _shared_label(label_at(row), codes),
        )
        add(f"{code_system}{_LABEL_SUFFIX}", lambda row, _reads, label_at=label_at: label_at(row))
        used.add(source_column)
        renamed[source_column] = f"{code_system}{_LABEL_SUFFIX}"

    if TIME_PERIOD_COLUMN in position:
        period_at = position[TIME_PERIOD_COLUMN]
        add(TIME_PERIOD_COLUMN, lambda row, _reads: period_cell(grain, row[period_at]))
        used.add(TIME_PERIOD_COLUMN)
    if "value" in position:
        take("value")

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

    def label_of(row: Row, reads: NodeReads) -> str | None:
        parts = parts_of(row)
        if parts is None:
            return None
        return label(parts, reads.names.get(parts.unit))

    def part(field: str) -> Cell:
        def cell(row: Row, _reads: NodeReads) -> object:
            parts = parts_of(row)
            return getattr(parts, field) if parts is not None else None

        return cell

    add(_UNIT_COLUMN, part("unit"), None)
    add(
        f"{_UNIT_COLUMN}{_LABEL_SUFFIX}",
        label_of,
        lambda row, reads: parts_of(row) is not None and label_of(row, reads) is None,
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


def _name_missing(cell: object, names: Mapping[str, object]) -> bool:
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
