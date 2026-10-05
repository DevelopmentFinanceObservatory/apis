"""Build a decode plan from a frame and caller-supplied `CodeIndex`.

This pure function resolves code-system values from `StatVarRow.constraint_properties` and
`CodeRow.label`. `materialise.reshape.decode_columns` applies the plan. The index is already in
memory, so planning needs no I/O or cache.
"""

from __future__ import annotations

from collections.abc import Iterable

from code_index.query import CodeIndex
from materialise.reshape import DecodePlan, TableLike

_STAT_VAR_COLUMN = "stat_var"
"""Mirrors `materialise.reshape._STAT_VAR_COLUMN` -- decode is always keyed
off the frame's `stat_var` column, never a role column (a code system's
value is a StatVar-level fact, not a role-level one)."""

_STAT_VAR_REF_PREFIX = "dcid:"
"""Mirrors `materialise.frame_build._STAT_VAR_REF_PREFIX`: every frame's own
`stat_var` column is canonicalised to this `dcid:`-prefixed spelling
(`_canonicalize_stat_var_ref`'s own docstring), a StatVar-ref convention
unrelated to any code system. The packaged code index's own `StatVarRow
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

    For each `(ref, code_system)`, `lookup` contains the code label when the index resolves the
    declared value. A declared absence stores the system's `absent_reading`. Missing StatVars or
    unresolved declared codes omit the lookup entry, so `decode_columns` emits an empty cell and
    `enrichment_unavailable` warning instead.
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

    bare_refs = {_bare_dcid(ref) for ref in refs}
    statvar_by_dcid = {row.dcid: row for row in index.statvars if row.dcid in bare_refs}

    code_systems: set[str] = set()
    for statvar in statvar_by_dcid.values():
        code_systems.update(statvar.constraint_properties.keys() & index.declarations.keys())
    if not code_systems:
        return None

    lookup: dict[str, dict[str, str]] = {}
    for ref in refs:
        statvar = statvar_by_dcid.get(_bare_dcid(ref))
        cells: dict[str, str] = {}
        for code_system in code_systems:
            if statvar is None:
                continue  # A genuine index gap for every code system on this row.
            value_dcid = statvar.constraint_properties.get(code_system)
            if value_dcid is None:
                cells[code_system] = index.declarations[code_system].absent_reading
                continue
            code_row = index.code_by_dcid(value_dcid)
            if code_row is not None:
                cells[code_system] = code_row.label
        lookup[ref] = cells

    requests = tuple((_STAT_VAR_COLUMN, code_system) for code_system in sorted(code_systems))
    return DecodePlan(requests=requests, lookup=lookup)
