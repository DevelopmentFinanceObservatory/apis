"""Validate `groupby.by` and derive each frame's collapse axes.

The collapse set is the decoded, declared-summable axes minus `by`. Unknown
names refuse before per-frame checks. An undeclared axis absent from one frame
is a no-op. `stat_var` joins the collapse only when a CRS axis collapses. A
decoded axis not declared summable is never collapsed.

The guards refuse ambiguous measures, unknown axes, decode gaps, and mixed
provenance. They use code-index declarations and frame data. The reshape engine
performs the sum.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from code_index.query import CodeIndex
from code_index.types import CodeSystemDeclaration, summable_axes
from materialise.reshape import RollupNotSummableError, TableLike
from materialise.roles import KNOWN_ROLE_NAMES

_STAT_VAR_COLUMN = "stat_var"
_STAT_VAR_REF_PREFIX = "dcid:"
"""Mirrors `datasets_api.decode`'s own constants of the same name -- see
that module's docstring for why the prefix has to be stripped before a
table ref matches `index.statvars`."""

# Mirrors the fixed frame1 columns in `materialise.reshape`. Keep them local
# to avoid importing the grammar package. `value` stays valid in `by` as the
# summed column, which appears on every frame.
_ALWAYS_PRESENT_COLUMNS: frozenset[str] = frozenset(
    ("time_period", "value", "unit", "measurementMethod", "provenance", "status")
)


def _bare_dcid(ref: str) -> str:
    return ref.removeprefix(_STAT_VAR_REF_PREFIX)


def declared_axes_present(
    *,
    decoded_code_systems: Sequence[str],
    declarations: Mapping[str, CodeSystemDeclaration],
) -> frozenset[str]:
    """Return declared-summable code systems decoded in this frame."""
    return summable_axes(declarations) & frozenset(decoded_code_systems)


def collapse_axes(
    *,
    by: Sequence[str],
    decoded_code_systems: Sequence[str],
    declarations: Mapping[str, CodeSystemDeclaration],
) -> tuple[str, ...]:
    """Return sorted decoded summable axes absent from `by`.

    `stat_var` is added by the caller only when this tuple is non-empty.
    Unknown names are checked separately. An axis absent from this frame adds
    no collapse entry.
    """
    keep = frozenset(by)
    present = declared_axes_present(
        decoded_code_systems=decoded_code_systems, declarations=declarations
    )
    return tuple(sorted(present - keep))


def unsummed_axes(
    *,
    decoded_code_systems: Sequence[str],
    declarations: Mapping[str, CodeSystemDeclaration],
) -> tuple[str, ...]:
    """Return sorted decoded code systems not declared summable.

    A sum never collapses these, so each stays a grain column of the summed rows.
    """
    return tuple(sorted(frozenset(decoded_code_systems) - summable_axes(declarations)))


def is_full_collapse(
    *,
    by: Sequence[str],
    decoded_code_systems: Sequence[str],
    declarations: Mapping[str, CodeSystemDeclaration],
    lookup: Mapping[str, Mapping[str, str]],
) -> bool:
    """Return true when the sum leaves one series per entity tuple: `by` keeps none of this
    frame's decoded summable axes, and every unsummed axis holds one value across the
    resolved members in `lookup`.

    Only a full collapse yields a grammar frame, which has no column for a decoded axis. An
    unsummed axis holding one value describes the whole family, so its constant column is
    left out of that frame. One holding several values keeps its column and the rows stay a
    table grouped by it. An empty summable axis set is false.
    """
    present = declared_axes_present(
        decoded_code_systems=decoded_code_systems, declarations=declarations
    )
    if not present or not frozenset(by).isdisjoint(present):
        return False
    return all(
        len({cells.get(code_system) for cells in lookup.values()}) == 1
        for code_system in unsummed_axes(
            decoded_code_systems=decoded_code_systems, declarations=declarations
        )
    )


def contributing_dcids(table: TableLike) -> frozenset[str]:
    """Return distinct StatVars with rows, before `stat_var` collapses.

    Only members with rows contribute. The count drives collection refusal and
    the `stat_vars_summed` warning for full collapses.
    """
    if _STAT_VAR_COLUMN not in table.columns:
        return frozenset()
    pos = table.columns.index(_STAT_VAR_COLUMN)
    return frozenset(cast(str, row[pos]) for row in table.rows if isinstance(row[pos], str))


def refuse_stat_var_collision(
    table: TableLike, *, index: CodeIndex, decoded_code_systems: Sequence[str]
) -> None:
    """Raise `RollupNotSummableError` (`rollup_not_summable`,
    `stat_var_ambiguous`) when distinct StatVars share values across all decoded axes.

    Once `stat_var` collapses, decoded axes alone distinguish rows. Population
    type, measured property, and related identity fields sit outside the frame,
    so matching axes can merge distinct measures. Check every StatVar before
    collapsing, even when rows differ elsewhere.
    """
    if _STAT_VAR_COLUMN not in table.columns:
        return
    pos = table.columns.index(_STAT_VAR_COLUMN)
    bare_dcids: set[str] = set()
    for row in table.rows:
        cell = row[pos]
        if isinstance(cell, str):
            bare_dcids.add(_bare_dcid(cell))
    statvar_by_dcid = {row.dcid: row for row in index.statvars if row.dcid in bare_dcids}

    sorted_systems = sorted(decoded_code_systems)
    groups: dict[tuple[str | None, ...], set[str]] = {}
    for row in statvar_by_dcid.values():
        key = tuple(row.constraint_properties.get(code_system) for code_system in sorted_systems)
        groups.setdefault(key, set()).add(row.dcid)

    for dcids in groups.values():
        if len(dcids) > 1:
            named = ", ".join(sorted(dcids))
            raise RollupNotSummableError(
                f"groupby axis 'stat_var' refused: StatVar(s) {named} decode to the same "
                f"values across {tuple(sorted_systems)!r} but are distinct measures; "
                "collapsing stat_var would merge them with no discriminating column left",
                axis=_STAT_VAR_COLUMN,
                precondition="stat_var_ambiguous",
            )


def refuse_unknown_by_axes(
    *, by: Sequence[str], declarations: Mapping[str, CodeSystemDeclaration]
) -> None:
    """Raise `RollupNotSummableError` (`rollup_not_summable`,
    `not_declared_summable`) for the first `by` name outside declared
    summable axes, known roles, `stat_var`, and always-present columns.

    This check is deployment-wide. A heterogeneous frame can omit a known axis.
    That name is valid across the deployment.
    """
    known = (
        summable_axes(declarations)
        | KNOWN_ROLE_NAMES
        | {_STAT_VAR_COLUMN}
        | _ALWAYS_PRESENT_COLUMNS
    )
    for name in by:
        if name not in known:
            raise RollupNotSummableError(
                f"groupby.by names {name!r}, which this deployment declares neither a "
                "summable code system, a role name, nor a column every frame carries",
                axis=name,
                precondition="not_declared_summable",
            )


def refuse_decode_gaps(
    *, lookup: Mapping[str, Mapping[str, str]], refs: Iterable[str], code_systems: Sequence[str]
) -> None:
    """Raise `RollupNotSummableError` (`rollup_not_summable`, `index_gap`) if a
    ref lacks a decoded value for a requested code system.

    A declared absence has a value. A lookup miss becomes an empty cell and
    could merge distinct StatVars after collapse. Check every ref, including
    zero-row members, across every decoded code system.
    """
    missing: list[str] = []
    for ref in sorted(set(refs)):
        cells = lookup.get(ref, {})
        missing.extend(
            f"{ref}/{code_system}" for code_system in code_systems if code_system not in cells
        )
    if missing:
        raise RollupNotSummableError(
            f"groupby axis 'stat_var' refused: {', '.join(missing)} have no decoded value "
            "(a genuine code-index gap rather than a declared absence), and merging an "
            "undecoded StatVar into a rollup could silently combine distinct measures",
            axis=_STAT_VAR_COLUMN,
            precondition="index_gap",
        )


def refuse_provenance_disagreement(*, provenance_refs: Sequence[str]) -> None:
    """Raise `RollupNotSummableError` (`rollup_not_summable`,
    `provenance_disagreement`) when the frame spans multiple sources.

    Each provenance declares whether absent observations count as zero or stay
    unknown. Mixed sources can disagree, so refuse before selecting a rule. A materialised group
    carries one provenance, so this guards other callers.
    """
    distinct = sorted(set(provenance_refs))
    if len(distinct) > 1:
        raise RollupNotSummableError(
            f"groupby refused: this frame's rows come from {len(distinct)} distinct "
            f"provenances ({', '.join(distinct)}), and whether an absent member counts as "
            "zero is a per-source declaration -- a summed group spanning sources that could "
            "disagree on it cannot be resolved to one answer",
            axis="provenance",
            precondition="provenance_disagreement",
        )
