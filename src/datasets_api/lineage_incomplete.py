"""How complete the rows of a computed version are.

A dataset version records which of its rows reach a period their source's import holds only
part of, and which its catalogue could not judge (`ManifestFacts.incomplete_after`,
`datasets_api.manifest.incomplete_after`, `manifest_models.IncompleteAfter`), keyed by the rows'
own `provenance` and `stat_var` cells. A computed (`ops_`) version keeps or replaces those
cells, so `derived_incomplete_after` gives it a record of its own, keyed the same way, which
the publish door stores with its lineage (`dataset_store.snapshot.FrameLineage`):

- an output cell an input's entries name takes those entries' states;
- an output cell an input's rows carry, which no entry names, is complete;
- any other output cell, which the computation made, takes the earliest partial period among
  the inputs' entries, since a figure computed from a partial input is partial unless its key
  shows otherwise; with no partial input it is unknown when an input holds an unknown cell,
  and complete otherwise.

A dataset input's rows carry each provenance its trace selects with each variable its
definition resolved. A computed version's record lists every cell of its rows, its complete
cells as `complete` entries, so a computation over it knows its cells. A row with no
`provenance` cell is kept when an input carries its variable under any provenance; it takes
the states of those cells when they agree, and is unknown when some are partial and others
are not.

The record is unknown as a whole, `None`, with no inputs, an input the store does not hold,
or an input whose record is unknown (recorded before versions stored one).
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
from materialise.partial_period import start_of

__all__ = ["derived_incomplete_after", "frame_cells"]

type Cell = tuple[str | None, str]
"""A row's `provenance` cell, `None` for a frame with none, and its `stat_var` cell. Cells
are compared with the variable bare (`bare_stat_var_ref`)."""

type State = tuple[str, str | None, str | None]
"""An entry's `status`, `source` and `partial_period`."""

_COMPLETE: State = ("complete", None, None)
_UNKNOWN: State = ("unknown", None, None)


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
    `frame`'s, read through `reader` in one batch: one entry per output `provenance` cell and
    state, holding the variables of its cells, every cell listed. `None` when unknown as a
    whole."""
    if not inputs:
        return None
    held = reader.read_versions_of([one.content_version for one in inputs])
    named: dict[Cell, set[State]] = {}
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
                state = _state(entry)
                for cell in _cells(entry):
                    carried.add(cell)
                    if state != _COMPLETE:
                        named.setdefault(cell, set()).add(state)
            carried.update(_dataset_cells(version))
    states = {state for held_states in named.values() for state in held_states}
    partial = [state for state in states if state[0] == "partial"]
    if partial:
        # The earliest mark, ties broken by source, so equal inputs give one record.
        made = min(partial, key=_order_by_mark)
    elif any(state[0] == "unknown" for state in states):
        made = _UNKNOWN
    else:
        made = _COMPLETE
    grouped: dict[tuple[str | None, State], list[str]] = {}
    for provenance, stat_var in sorted(frame_cells(frame), key=lambda c: (c[0] or "", c[1])):
        cell = (provenance, bare_stat_var_ref(stat_var))
        for state in _states(cell, named=named, carried=carried, made=made):
            stat_vars = grouped.setdefault((provenance, state), [])
            if stat_var not in stat_vars:
                stat_vars.append(stat_var)
    return [
        {
            "provenance": provenance,
            "stat_vars": stat_vars,
            "source": source,
            "status": status,
            "partial_period": mark,
        }
        for (provenance, (status, source, mark)), stat_vars in grouped.items()
    ]


def _states(
    cell: Cell, *, named: Mapping[Cell, set[State]], carried: set[Cell], made: State
) -> list[State]:
    """The states the output `cell` takes (module docstring), `made` for a cell the
    computation made."""
    provenance, variable = cell
    kept = (
        {held for held in carried if held[1] == variable}
        if provenance is None
        else {cell} & carried
    )
    if not kept:
        return [made]
    if provenance is not None:
        return sorted(named.get(cell, {_COMPLETE}), key=_order)
    per_cell = [frozenset(named.get(held, {_COMPLETE})) for held in kept]
    if len(set(per_cell)) == 1:
        return sorted(per_cell[0], key=_order)
    return [_UNKNOWN]


def _order(state: State) -> tuple[str, str, str]:
    return state[0], state[1] or "", state[2] or ""


def _order_by_mark(state: State) -> tuple[str, str]:
    return start_of(state[2] or ""), state[1] or ""


def _state(entry: Mapping[str, object]) -> State:
    status = cast(str, entry.get("status"))
    if status == "partial":
        return status, cast(str, entry["source"]), cast(str, entry["partial_period"])
    if status == "unknown":
        return status, cast("str | None", entry.get("source")), None
    return _COMPLETE


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
