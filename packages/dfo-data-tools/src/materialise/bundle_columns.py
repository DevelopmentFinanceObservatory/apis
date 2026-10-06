"""Column meanings, the codes table and coverage facts for a bundle's data files.

`TableScan` reads a rendered table chunk by chunk as its writer streams it, keeping the row
count, which columns hold any cell, each coded column's distinct cells and the years
`time_period` spans. From one scan per data file:

- `column_docs` gives each column a title and a description, naming roles with the nouns of
  `materialise.roles` (a role's camelCase graph name stays out of prose).
- `bundle_codes` builds the codes table: every distinct cell of every coded column, and each
  heading of a pivot that spread a coded column, labelled.
- `summary_line` and `year_window` give the coverage facts a bundle heads with.

A column is coded when it is the variable, a role, a decoded axis's code column (`<code_system>` in
long output), the unit, the measurement method or the provenance, holds at least one cell, and every
cell is a string a spreadsheet would not read as a formula. Such a cell is written to the data file
unchanged, so the codes table can key it by the exact cell text, `dcid:` prefix included, and the
bundle's foreign keys hold. The name columns beside the codes (`<role>_label`,
`<code_system>_label`, `unit_label` and the rest) hold words and stay out of the coded set.

Nothing here reads the graph. `bundle_codes` takes the label read as a callable, so the caller
decides its client and deadline, and reports whether that read completed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
from dfo_frame_grammar import (
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    Frame,
)

from dc_graph import GraphReadError
from materialise.bundle import CODES_PATH, CodeEntry, ColumnDoc
from materialise.bundle_naming import years_text
from materialise.cell_safety import spreadsheet_safe
from materialise.formats import Rendering, written_header
from materialise.roles import axis_for_role, axis_noun

if TYPE_CHECKING:
    from materialise.formats import Table

logger = logging.getLogger(__name__)

MEASUREMENT_METHOD_COLUMN = "measurementMethod"
PROVENANCE_COLUMN = "provenance"
OBSERVATION_ABOUT = "observationAbout"

VARIABLE_KIND = "variable"
PLACE_KIND = "place"
ORGANISATION_KIND = "organisation"
UNIT_KIND = "unit"
METHOD_KIND = "method"
SOURCE_KIND = "source"

_ENTITY_KIND = "entity"
"""The kind of a role column's code until its graph types resolve it to `place` or
`organisation`; it stays when `bundle_codes` reads no types or the read fails."""

_ORGANISATION_TYPE = "Organization"

_UNIT_REQUEST = object()
"""Marks a Frame's converted-unit cell, which holds no code."""

_COLUMN_KINDS: Mapping[str, str] = {
    STAT_VAR_COLUMN: VARIABLE_KIND,
    UNIT_COLUMN: UNIT_KIND,
    MEASUREMENT_METHOD_COLUMN: METHOD_KIND,
    PROVENANCE_COLUMN: SOURCE_KIND,
}

_DCID_PREFIX = "dcid:"

_FIXED_COLUMNS: Mapping[str, tuple[str, str]] = {
    STAT_VAR_COLUMN: ("Variable", "The variable measured."),
    "stat_var_label": ("Variable name", "The name of the variable."),
    TIME_PERIOD_COLUMN: ("Period", "The period the value refers to."),
    UNIT_COLUMN: ("Unit", "The unit of `value`."),
    "unit_label": ("Unit name", 'The name of the unit, such as "Constant 2023 US dollars".'),
    "currency": ("Currency", "The ISO 4217 code of the currency `value` is in."),
    "price_basis": ("Price basis", "Whether `value` is in `current` or `constant` prices."),
    "base_year": ("Base year", "The year whose prices a `constant` value is in."),
    MEASUREMENT_METHOD_COLUMN: ("Measurement method", "How the source measured the value."),
    "measurementMethod_label": (
        "Measurement method name",
        "The name of the measurement method, empty when the graph holds no name beyond the "
        "method's code.",
    ),
    PROVENANCE_COLUMN: ("Source dataset", "The source dataset of the row."),
    "provenance_label": ("Source dataset name", "The name of the source dataset."),
    STATUS_COLUMN: ("Status", "The source's flag on the value, such as provisional."),
}

_ROLE_DESCRIPTIONS: Mapping[str, str] = {
    "provider": "The provider the value flows from.",
    "recipient": "The recipient the value flows to.",
    "place": "The place the value describes.",
    "entity": "An entity the value refers to.",
}

_ENRICHMENT_COLUMNS: Mapping[str, tuple[str, str]] = {
    "iso3": ("ISO3 code", "The ISO 3166-1 alpha-3 code of the {noun}."),
    "iso2": ("ISO2 code", "The ISO 3166-1 alpha-2 code of the {noun}."),
    "label": ("name", "The name of the {noun}."),
    "type": ("type", "The graph types of the {noun}, comma-separated."),
}

_LABEL_SUFFIX = "_label"

_MIDDLE_DOT = "\u00b7"

_PLURALS: Mapping[str, str] = {"entity": "entities"}


def role_noun(role: str) -> str:
    """The plain noun for a role column: `provider` or `recipient` for a role on a flow axis,
    `place` for `observationAbout`, else `entity`."""
    if role == OBSERVATION_ABOUT:
        return "place"
    axis = axis_for_role(role)
    return axis_noun(axis) if axis is not None else "entity"


class TableScan:
    """Facts about one rendered table, gathered from each chunk as the writer streams it.

    Holds one set of distinct cells per candidate code column and a year window, so its size
    follows the data's distinct values.

    `axis_kinds` maps each axis code column the table may carry (`<code_system>`) to the kind
    its codes take in the codes table, the code system's plain name, such as `sector`.

    `pivot_index` and `pivot_columns` name the pivot that shaped the table, if any: the table is
    then its `pivot_index` column followed by one column per distinct value of `pivot_columns`,
    each headed by that value."""

    def __init__(
        self,
        *,
        role_columns: Sequence[str],
        axis_kinds: Mapping[str, str] | None = None,
        pivot_index: str | None = None,
        pivot_columns: str | None = None,
    ) -> None:
        self.role_columns = tuple(role_columns)
        self.axis_kinds: Mapping[str, str] = dict(axis_kinds or {})
        self.pivot_index = pivot_index
        self.pivot_columns = pivot_columns
        self.columns: tuple[str, ...] = ()
        self.rows = 0
        self._candidates = (
            frozenset(_COLUMN_KINDS) | frozenset(self.role_columns) | frozenset(self.axis_kinds)
        )
        self._filled: set[str] = set()
        self._values: dict[str, set[str]] = {}
        self._uncoded: set[str] = set()
        self._start_year: int | None = None
        self._end_year: int | None = None

    def observe(self, chunks: Iterable[Table]) -> Iterator[Table]:
        """Yields `chunks` unchanged, scanning each before it passes on."""
        for chunk in chunks:
            self._scan(chunk)
            yield chunk

    def empty(self, column: str) -> bool:
        """Whether every cell of `column` is null or an empty string."""
        return column not in self._filled

    def coded(self, column: str) -> bool:
        """Whether `column`'s cells are codes the codes table names."""
        return column in self._candidates and column in self._filled and column not in self._uncoded

    def codes(self, column: str) -> frozenset[str]:
        """The distinct cells of a coded column, empty for any other column."""
        return frozenset(self._values.get(column, ())) if self.coded(column) else frozenset()

    @property
    def heading_kind(self) -> str | None:
        """The code kind of the pivot's headings. Set when the pivot spread a coded column (a
        role, the variable, the unit, the measurement method or the provenance) and every
        heading could key the codes table, `None` otherwise."""
        if self.pivot_columns is None:
            return None
        if self.pivot_columns in self.role_columns:
            kind = _ENTITY_KIND
        elif (kind := _COLUMN_KINDS.get(self.pivot_columns)) is None:
            return None
        headings = self.headings
        return kind if headings and all(_codable(heading) for heading in headings) else None

    @property
    def headings(self) -> tuple[str, ...]:
        """The columns a pivot spread its values into, in file order; empty without a pivot."""
        if self.pivot_columns is None:
            return ()
        return tuple(column for column in self.columns if column != self.pivot_index)

    @property
    def year_window(self) -> tuple[int, int] | None:
        """The first and last year `time_period` names, `None` when it names none."""
        if self._start_year is None or self._end_year is None:
            return None
        return self._start_year, self._end_year

    def _scan(self, chunk: Table) -> None:
        if not self.columns:
            self.columns = chunk.columns
        if isinstance(chunk, Frame):
            self.rows += chunk.table.num_rows
            for column in chunk.columns:
                self._add(column, _frame_distinct(chunk.table, column))
            return
        rows = chunk.rows
        self.rows += len(rows)
        for position, column in enumerate(chunk.columns):
            self._add(column, {row[position] for row in rows})

    def _add(self, column: str, distinct: Iterable[object]) -> None:
        for cell in distinct:
            if cell is None or cell == "":
                continue
            self._filled.add(column)
            if column == TIME_PERIOD_COLUMN:
                self._add_year(cell)
            if column not in self._candidates or column in self._uncoded:
                continue
            if isinstance(cell, str) and _codable(cell):
                self._values.setdefault(column, set()).add(cell)
            else:
                self._uncoded.add(column)
                self._values.pop(column, None)

    def _add_year(self, cell: object) -> None:
        text = str(cell)[:4]
        if len(text) < 4 or not text.isdigit():
            return
        year = int(text)
        self._start_year = year if self._start_year is None else min(self._start_year, year)
        self._end_year = year if self._end_year is None else max(self._end_year, year)


def _codable(text: str) -> bool:
    """Whether `text` can key the codes table: the CSV writer leaves it unchanged, since it
    escapes any cell a spreadsheet would read as a formula."""
    return spreadsheet_safe(text) == text


def _frame_distinct(table: pa.Table, column: str) -> list[object]:
    """A Frame column's distinct cells. The grammar stores `unit` as a struct holding either a
    unit `ref` or a conversion `request`; its refs are its codes, and a request, which carries
    no code, stands in as one non-string marker."""
    values = table.column(column)
    if not pa.types.is_struct(values.type):
        return pc.unique(values).to_pylist()
    refs = pc.struct_field(values, "ref")
    distinct: list[object] = pc.unique(refs).to_pylist()
    if refs.null_count > values.null_count:
        distinct.append(_UNIT_REQUEST)
    return distinct


_PLAIN_RENDERING = Rendering()


def column_docs(
    scan: TableScan,
    *,
    code_system_names: Mapping[str, str],
    rendering: Rendering = _PLAIN_RENDERING,
) -> tuple[ColumnDoc, ...]:
    """One `ColumnDoc` per column of the scanned table, in file order, named by the header
    `formats.written_header` gives it under `rendering`, the one the data file was written with,
    so a pivot heading such as `-5 to 0` is documented as the escaped `'-5 to 0` it heads.

    `code_system_names` maps a code system to its plain name (`CodeSystemDeclaration.plain_name`,
    such as `sector` for `crsPurpose`), which names the decoded axis columns: the long layout's
    `<code system>` code and `<code system>_label` name, and the `stat_var_<code system>` label
    column a pivot or groupby writes. Each pivot heading holds the values for the entry its heading
    names. Descriptions leave out "Names are in the codes table" and "Empty", which the README adds
    from `coded` and `empty`. A heading that is a code has no foreign key, because the code is the
    heading and no cell holds it, so its description points to the codes table itself."""
    headings = frozenset(scan.headings)
    docs: list[ColumnDoc] = []
    for column in scan.columns:
        if column in headings:
            title, description = column, _heading_description(scan)
        else:
            title, description = _column_meaning(
                column,
                role_columns=scan.role_columns,
                has_unit=UNIT_COLUMN in scan.columns,
                code_system_names=code_system_names,
            )
        docs.append(
            ColumnDoc(
                name=written_header(column, rendering=rendering),
                title=title,
                description=description,
                coded=scan.coded(column),
                empty=scan.empty(column),
            )
        )
    return tuple(docs)


def _column_meaning(
    column: str,
    *,
    role_columns: Sequence[str],
    has_unit: bool,
    code_system_names: Mapping[str, str],
) -> tuple[str, str]:
    if column in role_columns:
        noun = role_noun(column)
        return noun.capitalize(), _ROLE_DESCRIPTIONS[noun]
    if column == VALUE_COLUMN:
        if has_unit:
            return "Value", "The observed value, in the unit named in `unit`."
        return "Value", "The observed value."
    if column in _FIXED_COLUMNS:
        return _FIXED_COLUMNS[column]
    if (plain := code_system_names.get(column)) is not None:
        return plain.capitalize(), f"The {plain} of the variable, as a code."
    if (plain := code_system_names.get(column.removesuffix(_LABEL_SUFFIX))) is not None:
        return f"{plain.capitalize()} name", f"The {plain} of the variable, in words."
    if column.startswith(f"{STAT_VAR_COLUMN}_"):
        plain = code_system_names.get(column.removeprefix(f"{STAT_VAR_COLUMN}_"))
        if plain is not None:
            return plain.capitalize(), f"The {plain} of the variable, in words."
        return "Variable property", "A property of the variable, in words."
    for role in role_columns:
        field = column.removeprefix(f"{role}_")
        if column.startswith(f"{role}_") and field in _ENRICHMENT_COLUMNS:
            noun = role_noun(role)
            title, description = _ENRICHMENT_COLUMNS[field]
            return f"{noun.capitalize()} {title}", description.format(noun=noun)
    return column, "A column of the data."


def _heading_description(scan: TableScan) -> str:
    assert scan.pivot_columns is not None
    noun = _pivot_noun(scan.pivot_columns, role_columns=scan.role_columns)
    description = f"The value for the {noun} this column's heading names."
    if scan.heading_kind is not None:
        description += f" Names are in `{CODES_PATH}`."
    return description


def _pivot_noun(column: str, *, role_columns: Sequence[str]) -> str:
    if column in role_columns:
        return role_noun(column)
    if column == STAT_VAR_COLUMN:
        return "variable"
    if column == TIME_PERIOD_COLUMN:
        return "period"
    return "entry"


@dataclass(frozen=True)
class BundleCodes:
    """A bundle's codes table and whether the graph label read behind it completed."""

    entries: tuple[CodeEntry, ...]
    """One entry per code, sorted by code."""
    label_read_complete: bool
    """`True` when every label `bundle_codes` asked the graph for was answered, including codes
    the graph holds no name for, or when nothing was read. `False` when the read raised a
    `GraphReadError`, which leaves codes labelled with themselves and role codes with the kind
    `entity`, so the entries differ from a later render's."""


def bundle_codes(
    scans: Sequence[TableScan],
    *,
    known_labels: Mapping[str, str],
    lookup: Callable[[Sequence[str]], Mapping[str, Mapping[str, str]]] | None,
) -> BundleCodes:
    """The codes table for the scanned data files: one entry per distinct cell of every coded
    column and per heading of a pivot that spread a coded column, sorted by code.

    A code takes its label from `known_labels` first (a source's cited name, a collapsed
    family's label), else from `lookup`, which maps graph dcids to the fields
    `EnrichmentLookup.lookup` returns (`label`, `type`); a `dcid:` prefix is stripped for the
    read. A role column's code is an `organisation` when its graph types include
    Organization, else a `place`. With `lookup` `None` nothing is read, so codes take
    `known_labels` alone and a role column's code keeps the kind `entity`, since nothing tells
    a place from an organisation. A code neither labels is labelled with itself, so every
    foreign key holds. The function logs how many.

    A `GraphReadError` from `lookup`, deadline expiry included, is logged and leaves every read
    code labelled with itself and every role code with the kind `entity`, and the result's
    `label_read_complete` is `False`. Any other error, `GraphCapExceededError` among them,
    propagates. A code seen in two columns keeps the kind of the first."""
    kinds = _code_kinds(scans)
    if not kinds:
        return BundleCodes(entries=(), label_read_complete=True)
    to_read = {
        code: code.removeprefix(_DCID_PREFIX)
        for code, kind in kinds.items()
        if code not in known_labels and kind != SOURCE_KIND
    }
    fields: Mapping[str, Mapping[str, str]] = {}
    read_failed = False
    if to_read and lookup is not None:
        try:
            fields = lookup(tuple(dict.fromkeys(to_read.values())))
        except GraphReadError:
            read_failed = True
            logger.warning(
                "bundle code label read failed; %d codes are labelled with themselves",
                len(to_read),
                exc_info=True,
            )
    types_known = lookup is not None and not read_failed
    entries: list[CodeEntry] = []
    unlabelled = 0
    for code in sorted(kinds):
        found = fields.get(to_read[code], {}) if code in to_read else {}
        label = known_labels.get(code) or found.get("label")
        if label is None:
            unlabelled += 1
            label = code
        kind = kinds[code]
        if kind == _ENTITY_KIND and types_known:
            types = found.get("type", "").split(",")
            kind = ORGANISATION_KIND if _ORGANISATION_TYPE in types else PLACE_KIND
        entries.append(CodeEntry(code=code, label=label, kind=kind))
    if unlabelled and not read_failed:
        logger.info("bundle codes without a name: %d are labelled with themselves", unlabelled)
    return BundleCodes(entries=tuple(entries), label_read_complete=not read_failed)


def _code_kinds(scans: Sequence[TableScan]) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for scan in scans:
        for column in scan.columns:
            if column in scan.role_columns:
                kind = _ENTITY_KIND
            else:
                kind = scan.axis_kinds.get(column) or _COLUMN_KINDS.get(column)
            if kind is None:
                continue
            for code in sorted(scan.codes(column)):
                kinds.setdefault(code, kind)
        if (heading_kind := scan.heading_kind) is not None:
            for code in sorted(scan.headings):
                kinds.setdefault(code, heading_kind)
    return kinds


def year_window(scans: Sequence[TableScan]) -> tuple[int, int] | None:
    """The first and last year across the scanned files, `None` when none names a year."""
    windows = [window for scan in scans if (window := scan.year_window) is not None]
    if not windows:
        return None
    return min(start for start, _ in windows), max(end for _, end in windows)


def summary_line(scans: Sequence[TableScan], *, codes: Sequence[CodeEntry]) -> str:
    """One line of coverage facts, joined by ` · `: rows, variables, the distinct entities per
    role noun, the year window and the unit, such as
    "1,382 rows · 309 variables · 1 provider, 1 recipient · 1973\u20132024 · Unit: US dollar".

    The unit prints its label from `codes` when the files hold one unit, and a count when they
    hold several. A fact the files do not carry is left out."""
    parts = [_count(sum(scan.rows for scan in scans), "row")]
    variables: set[str] = set()
    by_noun: dict[str, set[str]] = {}
    units: set[str] = set()
    for scan in scans:
        variables |= scan.codes(STAT_VAR_COLUMN)
        units |= scan.codes(UNIT_COLUMN)
        for role in scan.role_columns:
            if role in scan.columns:
                by_noun.setdefault(role_noun(role), set()).update(scan.codes(role))
    if variables:
        parts.append(_count(len(variables), "variable"))
    roles = [_count(len(refs), noun) for noun, refs in by_noun.items() if refs]
    if roles:
        parts.append(", ".join(roles))
    if (window := year_window(scans)) is not None:
        parts.append(years_text(*window))
    if len(units) == 1:
        (unit,) = units
        labels = {entry.code: entry.label for entry in codes}
        parts.append(f"Unit: {labels.get(unit, unit)}")
    elif units:
        parts.append(_count(len(units), "unit"))
    return " · ".join(parts)


def _count(number: int, noun: str) -> str:
    plural = noun if number == 1 else _PLURALS.get(noun, f"{noun}s")
    return f"{number:,} {plural}"
