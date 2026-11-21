"""The rows of a computed version past the last complete year of their source.

A dataset version records which of its rows reach a period their source's import holds only
part of (`coverage.incomplete_after`, `datasets_api.manifest.incomplete_after`), keyed by the
rows' own `provenance` and `stat_var` cells. A computed (`ops_`) version keeps or replaces
those cells, so `derived_incomplete_after` gives it a record of its own, keyed the same way,
which the publish door stores with its lineage (`dataset_store.snapshot.FrameLineage`):

- an output cell an input's entry names takes that entry;
- an output cell an input dataset's rows carry, which no entry names, is complete;
- any other output cell, which the computation made, takes the earliest last complete year
  among the inputs' entries, since a figure computed from a partial input is partial unless
  its key shows otherwise.

The inputs are those `lineage_continuations.lineage_inputs` names. An input whose record is
unknown (a version recorded before versions stored it) or that the store does not hold makes
the whole record unknown, `None`.
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
"""A row's `provenance` cell, `None` for a frame with none, and its `stat_var` cell."""


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
    """The `incomplete_after` entries of a computed version over `inputs` whose rows are
    `frame`'s, read through `reader` in one batch: one entry per output `provenance` cell,
    source and last complete year, `[]` when no output row is past one, and `None` when an
    input's record is unknown."""
    held = reader.read_versions_of([one.content_version for one in inputs])
    entries: list[Mapping[str, object]] = []
    carried: set[tuple[str | None, str]] = set()
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
            entries.extend(recorded)
            carried.update(_carried_cells(version))
    earliest = min(entries, key=_last_complete_year, default=None)
    grouped: dict[tuple[str | None, str, int], list[str]] = {}
    for provenance, stat_var in sorted(
        frame_cells(frame), key=lambda cell: (cell[0] or "", cell[1])
    ):
        named = [entry for entry in entries if _names(entry, provenance, stat_var)]
        if (
            not named
            and (provenance, bare_stat_var_ref(stat_var)) not in carried
            and earliest is not None
        ):
            named = [earliest]
        for entry in named:
            key = (provenance, cast(str, entry["source"]), _last_complete_year(entry))
            grouped.setdefault(key, []).append(stat_var)
    return [
        {
            "provenance": provenance,
            "stat_vars": stat_vars,
            "source": source,
            "last_complete_year": year,
        }
        for (provenance, source, year), stat_vars in grouped.items()
    ]


def _recorded(version: StoredVersion) -> list[Mapping[str, object]] | None:
    """The `incomplete_after` entries `version` stores, `None` when it stores none."""
    facts = version.facts
    if isinstance(facts, ManifestFacts):
        recorded = facts.coverage.get("incomplete_after")
    else:
        recorded = facts.incomplete_after
    if recorded is None:
        return None
    return list(cast("Iterable[Mapping[str, object]]", recorded))


def _carried_cells(version: StoredVersion) -> set[tuple[str, str]]:
    """The cells a dataset version's rows carry, each variable bare: each provenance its rows
    carry (`provenance_refs_of`) with each variable its definition resolved. A computed
    version's cells are not stored, so it gives none."""
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


def _names(entry: Mapping[str, object], provenance: str | None, stat_var: str) -> bool:
    """Whether `entry` names the cell `(provenance, stat_var)`, a `stat_var` with or without
    its `dcid:` prefix naming the same variable."""
    stat_vars = cast("Sequence[str]", entry.get("stat_vars") or ())
    return entry.get("provenance") == provenance and bare_stat_var_ref(stat_var) in {
        bare_stat_var_ref(one) for one in stat_vars
    }


def _last_complete_year(entry: Mapping[str, object]) -> int:
    return cast(int, entry["last_complete_year"])
