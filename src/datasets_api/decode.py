"""Build a decode plan from a frame and caller-supplied `CodeIndex`.

This pure function resolves code-system values from `StatVarRow.constraint_properties`: each
decoded `stat_var_<code_system>` cell holds the code's dcid (`CodeRow.dcid`), so a relabelled
code keeps its cell. Labels are read where output names a code: the long layout's
`<code_system>_label` (`materialise.projection`) and the presentation frame's labels.
`materialise.reshape.decode_columns` applies the plan. The index is already in memory, so
planning needs no I/O or cache.
"""

from __future__ import annotations

from collections.abc import Iterable

from code_index.query import CodeIndex
from code_index.types import StatVarRow
from dfo_platform.columns import PROVENANCE_COLUMN
from materialise.reshape import DecodePlan, TableLike

_STAT_VAR_COLUMN = "stat_var"
"""Mirrors `materialise.reshape._STAT_VAR_COLUMN` -- decode is always keyed
off the frame's `stat_var` column, never a role column (a code system's
value is a StatVar-level fact, not a role-level one)."""


_STAT_VAR_REF_PREFIX = "dcid:"
"""Mirrors `materialise.frame_build._STAT_VAR_REF_PREFIX`: every frame's own
`stat_var` column is canonicalised to this `dcid:`-prefixed spelling
(`_canonicalize_stat_var_ref`'s own docstring), a StatVar-ref convention
unrelated to any code system. The code index's own `StatVarRow
.dcid` is the graph's bare spelling instead -- the same one
`StatVarRow.constraint_properties`' own values and `CodeRow.dcid` use
throughout `code_index` -- so this module strips the prefix before matching
a table ref against `index.statvars`, the one place these two conventions
meet."""


def _bare_dcid(ref: str) -> str:
    return ref.removeprefix(_STAT_VAR_REF_PREFIX)


def build_decode_plan(
    table: TableLike, *, index: CodeIndex, extra_refs: Iterable[str] = ()
) -> DecodePlan | None:
    """Return the `(requests, lookup)` plan, or `None` when decoding would add no columns.

    `extra_refs` adds `dcid:` StatVars that may have no table rows, keeping zero-row members in
    the same decode pass as table values. Code systems are the union of StatVar constraint keys
    intersected with `index.declarations`. Requests are sorted for deterministic column order.

    A frame whose rows carry a composite provenance (`CodeIndex.composite`) leaves out the code
    systems its components' recipes bind (`ComponentRecipe.bind`). Each holds one value per
    component by declaration, so its column restates which component a row came from, which
    `stat_var` already names, and a sum over the composite would split on it.

    For each `(ref, code_system)`, `lookup` contains the code's dcid when the index holds a code row
    for the declared value. A declared absence stores the system's `absent_reading`. Missing
    StatVars or unresolved declared codes omit the lookup entry, so `decode_columns` emits an empty
    cell and `enrichment_unavailable` warning instead.
    """
    refs: set[str] = set(extra_refs)
    if _STAT_VAR_COLUMN in table.columns:
        stat_var_pos = table.columns.index(_STAT_VAR_COLUMN)
        for row in table.rows:
            cell = row[stat_var_pos]
            if isinstance(cell, str):
                refs.add(cell)
    if not refs:
        return None
    marked = _component_marker_axes(table, index=index)

    statvars = _index_rows(refs, index=index)
    code_systems = set(_code_systems(statvars.values(), index=index)) - marked
    if not code_systems:
        return None

    lookup: dict[str, dict[str, str]] = {}
    for ref in refs:
        statvar = statvars.get(ref)
        cells: dict[str, str] = {}
        for code_system in code_systems:
            if statvar is None:
                continue  # A genuine index gap for every code system on this row.
            value_dcid = statvar.constraint_properties.get(code_system)
            if value_dcid is None:
                cells[code_system] = index.declarations[code_system].absent_reading
                continue
            if index.code_by_dcid(value_dcid) is not None:
                cells[code_system] = value_dcid
        lookup[ref] = cells

    requests = tuple((_STAT_VAR_COLUMN, code_system) for code_system in sorted(code_systems))
    return DecodePlan(requests=requests, lookup=lookup)


def statvar_code_systems(refs: Iterable[str], *, index: CodeIndex) -> frozenset[str]:
    """The declared code systems (`index.declarations`) the index rows of `refs`, bare or
    `dcid:` StatVar refs, carry as constraint properties: the systems a frame of those
    StatVars decodes, before a composite leaves out its marker axes."""
    return _code_systems(_index_rows(refs, index=index).values(), index=index)


def _index_rows(refs: Iterable[str], *, index: CodeIndex) -> dict[str, StatVarRow]:
    """Each of `refs` the index enumerates (`CodeIndex.statvars`) mapped to its row, looked up
    by dcid. A listed StatVar (`CodeIndex.listed_statvars`), which belongs to no family, and a
    ref the index does not hold are left out."""
    rows: dict[str, StatVarRow] = {}
    for ref in refs:
        dcid = _bare_dcid(ref)
        if dcid in index.family_by_dcid and (row := index.statvar_by_dcid(dcid)) is not None:
            rows[ref] = row
    return rows


def _code_systems(rows: Iterable[StatVarRow], *, index: CodeIndex) -> frozenset[str]:
    """The declared code systems (`index.declarations`) `rows` carry as constraint
    properties."""
    return frozenset(
        code_system
        for row in rows
        for code_system in row.constraint_properties.keys() & index.declarations.keys()
    )


def _component_marker_axes(table: TableLike, *, index: CodeIndex) -> frozenset[str]:
    """The code systems the components of each composite `table`'s rows carry as their
    provenance bind (`build_decode_plan`)."""
    if PROVENANCE_COLUMN not in table.columns:
        return frozenset()
    position = table.columns.index(PROVENANCE_COLUMN)
    provenances = {cell for row in table.rows if isinstance(cell := row[position], str)}
    return frozenset(
        code_system
        for provenance in provenances
        if (composite := index.composite(provenance)) is not None
        for recipe in composite.components
        for code_system in recipe.bind
    )
