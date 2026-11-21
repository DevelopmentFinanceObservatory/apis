"""The rows of a computed version past the last complete year of their source.

A dataset version records which of its rows reach a period their source's import holds only
part of (`ManifestFacts.incomplete_after`, `datasets_api.manifest.incomplete_after`), keyed by
the rows' own `provenance` and `stat_var` cells. A computed (`ops_`) version keeps or replaces
those cells, so `derived_incomplete_after` gives it a record of its own, keyed the same way,
which the publish door stores with its lineage (`dataset_store.snapshot.FrameLineage`):

- an output cell an input's entry names takes that entry;
- an output cell an input's rows carry, which no entry names, is complete;
- any other output cell, which the computation made, takes the earliest last complete year
  among the inputs' entries, since a figure computed from a partial input is partial unless
  its key shows otherwise.

A dataset input's rows carry each provenance its trace selects with each variable its
definition resolved. A computed version's record lists every cell of its rows: those past a
last complete year as a dataset's entries do, and the rest in entries whose `source` and
`last_complete_year` are `None`, which no reader flags, so a computation over it knows its
cells. A row with no `provenance` cell is kept when an input carries its variable under any
provenance; it takes that variable's entries when every provenance carrying it has one, and is
complete when none has.

The record is unknown, `None`, when it cannot be told: no inputs, an input the store does not
hold, an input whose record is unknown (recorded before versions stored it, or minted where its
catalogue could not judge it), and a row with no `provenance` cell whose variable some carrying
provenances mark partial and others do not.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from dfo_frame_grammar import STAT_VAR_COLUMN, Frame

from dataset_store.snapshot import ManifestFacts, SnapshotReader, StoredVersion
from datasets_api.manifest import provenance_refs_of
from description.facts import InputVersion
from dfo_platform.columns import PROVENANCE_COLUMN
from materialise.generation import bare_stat_var_ref

__all__ = ["derived_incomplete_after", "frame_cells"]

type Cell = tuple[str | None, str]
"""A row's `provenance` cell, `None` for a frame with none, and its `stat_var` cell. Cells
are compared with the variable bare (`bare_stat_var_ref`)."""

type Source = tuple[str | None, int | None]
"""The source an output cell's rows come from and its last complete year, both `None` for a
complete cell."""

_COMPLETE: Source = (None, None)


class _Unknown(Exception):
    """The record cannot be told."""


def frame_cells(frame: Frame) -> set[Cell]:
    """The distinct `(provenance, stat_var)` cells of `frame`'s rows."""
    table = frame.table
    if STAT_VAR_COLUMN not in table.column_names:
        return set()
    stat_vars = table.column(STAT_VAR_COLUMN).to_pylist()
    provenances = (
        table.column(PROVENANCE_COLUMN).to_pylist()
        if PROVENANCE_COLUMN in table.column_names
        else [None] * len(stat_vars)
    )
    return {
        (provenance, stat_var)
        for stat_var, provenance in zip(stat_vars, provenances, strict=True)
        if isinstance(stat_var, str)
    }


def derived_incomplete_after(
    inputs: Sequence[InputVersion], *, reader: SnapshotReader, frame: Frame
) -> list[dict[str, object]] | None:
    """The `incomplete_after` record of a computed version over `inputs` whose rows are
    `frame`'s, read through `reader` in one batch: one entry per output `provenance` cell,
    source and last complete year, holding the variables of its cells, the complete cells
    under a `None` source and year. `None` when it cannot be told."""
    if not inputs:
        return None
    held = reader.read_versions_of([one.content_version for one in inputs])
    partial: dict[Cell, set[tuple[str, int]]] = {}
    carried: set[Cell] = set()
    for one in inputs:
        versions = [
            version
            for version in held
            if version.content_version == one.content_version
            and (one.id is None or version.subject_id == one.id)
        ]
        if not versions:
            return None
        for version in versions:
            recorded = _recorded(version)
            if recorded is None:
                return None
            for entry in recorded:
                source, year = entry.get("source"), entry.get("last_complete_year")
                for cell in _cells(entry):
                    carried.add(cell)
                    if isinstance(source, str) and isinstance(year, int):
                        partial.setdefault(cell, set()).add((source, year))
            carried.update(_dataset_cells(version))
    earliest = min(
        (source for named in partial.values() for source in named),
        key=lambda source: source[1],
        default=None,
    )
    grouped: dict[tuple[str | None, str | None, int | None], list[str]] = {}
    try:
        for provenance, stat_var in sorted(
            frame_cells(frame), key=lambda cell: (cell[0] or "", cell[1])
        ):
            cell = (provenance, bare_stat_var_ref(stat_var))
            for source in _sources(cell, partial=partial, carried=carried, earliest=earliest):
                stat_vars = grouped.setdefault((provenance, *source), [])
                if stat_var not in stat_vars:
                    stat_vars.append(stat_var)
    except _Unknown:
        return None
    return [
        {
            "provenance": provenance,
            "stat_vars": stat_vars,
            "source": source,
            "last_complete_year": year,
        }
        for (provenance, source, year), stat_vars in grouped.items()
    ]


def _sources(
    cell: Cell,
    *,
    partial: Mapping[Cell, set[tuple[str, int]]],
    carried: set[Cell],
    earliest: tuple[str, int] | None,
) -> list[Source]:
    """The sources the output `cell` takes (module docstring): its inputs' entries for a kept
    cell, `_COMPLETE` for a kept cell none names, the earliest partial source for a cell the
    computation made, and `_COMPLETE` for one when no input is partial.

    Raises:
        _Unknown: `cell` has no `provenance`, and some provenances carrying its variable are
            partial and others are not.
    """
    provenance, variable = cell
    kept = (
        {held for held in carried if held[1] == variable}
        if provenance is None
        else {cell} & carried
    )
    if kept:
        named = [held for held in kept if held in partial]
        if not named:
            return [_COMPLETE]
        if len(named) < len(kept):
            raise _Unknown
        return sorted({source for held in named for source in partial[held]})
    return [_COMPLETE] if earliest is None else [earliest]


def _recorded(version: StoredVersion) -> list[Mapping[str, object]] | None:
    """The `incomplete_after` entries `version` stores, `None` when it stores none."""
    recorded = version.facts.incomplete_after
    if recorded is None:
        return None
    return list(cast("Iterable[Mapping[str, object]]", recorded))


def _cells(entry: Mapping[str, object]) -> list[Cell]:
    """The cells `entry` names, each variable bare."""
    provenance = cast("str | None", entry.get("provenance"))
    stat_vars = cast("Sequence[str]", entry.get("stat_vars") or ())
    return [(provenance, bare_stat_var_ref(stat_var)) for stat_var in stat_vars]


def _dataset_cells(version: StoredVersion) -> set[Cell]:
    """The cells a dataset version's rows carry: each provenance its rows carry
    (`provenance_refs_of`) with each variable its definition resolved. A computed version's
    record names its own (`_cells`)."""
    facts = version.facts
    if not isinstance(facts, ManifestFacts):
        return set()
    trace = facts.trace
    refs = provenance_refs_of(cast("Sequence[Mapping[str, object]]", trace.get("facets") or ()))
    variables = [
        bare_stat_var_ref(cast(str, entry["ref"]))
        for entry in cast("Sequence[Mapping[str, object]]", trace.get("resolved_stat_vars") or ())
    ]
    return {(ref, variable) for ref in refs for variable in variables}
