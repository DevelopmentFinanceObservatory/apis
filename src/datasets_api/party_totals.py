"""`options.without_party_totals`: a flow frame without the group totals of the party roles its
definition leaves open.

Flow datasets such as OECD DAC2A list group totals (the DAC total, the G7, DAC EU members) as
parties beside their members, so a frame for an open party side mixes members with sums of
members. A party role is a role on a flow axis (`materialise.enrichment.is_party_role`). It is
bound when the served identity's `entities` name an entity on its axis
(`dfo_platform.roles.role_key_axis`), which a request's `scope.entities` replaces whole
(`datasets_api.surgery.apply_scope_surgery`); every other party role of the frame is open.

For each open party role, in column-name order, the rows whose party the code index declares a
`total` in the row's dataset (`materialise.enrichment.party_kind_of` for the row's `provenance`)
can be left out, each role reading the rows the roles before it kept. A party can be a total in
one dataset and a party in another (`code_index.types.PartyRow.dataset_kinds`), so the cut keys
on the party and the dataset together. A row with no provenance reads the kind of every dataset,
and keeps a party whose kind differs between datasets. Parties of kind `party` or `residual`,
and parties the index declares no kind for, stay. A bound role is left alone whatever its
parties' kinds.

A role decides each cell of one full `time_period` value and one provenance on its own. A cell
in which the role holds another party loses its totals. A cell in which it holds only totals
keeps the totals whose row reads as a whole total (`materialise.enrichment.row_party_cell` gives
`party_whole` `true`, the cell the analysis engine reads as `{role}_party_whole`), since whole
totals hold disjoint populations, and loses the rest. A totals-only cell with no whole total
loses its totals when the role holds another party anywhere in the frame, and keeps them all
when it holds none, since the totals are then the only parties there are (the G7's flows to
Kenya). DAC2B publishes India's 1968 receipts only as totals (DAC countries, the G7 and official
donors), so the cut keeps official donors, the one whole total, for that year, whether or not
later years list members.

The cell is the time period and the provenance alone, so in a frame with another open role a
totals-only series is cut like the rest in a period where another series of the frame holds
members.

The cut reads the frame before any reshape, so a `groupby` over it sums members alone. Each pass
holds one chunk at a time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc
from dfo_frame_grammar import TIME_PERIOD_COLUMN, Frame

from code_index.types import PartyRow
from datasets_api.manifest_models import PartyTotalsLeftOutModel
from dfo_platform.columns import PROVENANCE_COLUMN
from dfo_platform.roles import role_key_axis
from materialise.content_version import stream_frame_sha256
from materialise.enrichment import (
    PARTY_WHOLE_FIELD,
    is_party_role,
    party_kind_of,
    row_party_cell,
)

_TOTAL_KIND = "total"

YEARS_DIMENSION = "years"
"""The coverage dimension counting a frame's distinct leading years
(`materialise.columnar.coverage`)."""

TRACE_KEY = "party_totals_left_out"
"""The trace key of the totals a version left out (`PartyTotalsCut.trace_entries`)."""


def open_party_roles(identity: object, entity_columns: Sequence[str]) -> tuple[str, ...]:
    """The party roles among `entity_columns` whose axis `identity`'s `entities` bind no entity
    on, in column-name order."""
    entities = identity.get("entities") if isinstance(identity, Mapping) else None
    bound_axes = (
        {
            role_key_axis(str(role))
            for role, refs in entities.items()
            if isinstance(refs, list | tuple) and refs
        }
        if isinstance(entities, Mapping)
        else set()
    )
    return tuple(
        sorted(
            role
            for role in entity_columns
            if is_party_role(role) and role_key_axis(role) not in bound_axes
        )
    )


class _EveryProvenance(Enum):
    """The provenance key of refs left out of the rows of every provenance, which a trace entry
    recorded without a `provenance` names (`PartyTotalsCut.of_trace_entries`)."""

    EVERY = "every"


EVERY_PROVENANCE = _EveryProvenance.EVERY

type _Provenance = str | None | _EveryProvenance


type _LeftOut = Mapping[str, Mapping[_Provenance, Mapping[str, tuple[str, ...]]]]
"""Each open party role that lost rows, mapped to each provenance it lost rows in, mapped to
each ref it lost there and the sorted `time_period` values in which that ref's rows stayed,
empty when they left every period."""


@dataclass(frozen=True)
class PartyTotalsCut:
    """The totals a cut leaves out of a frame (`_LeftOut`). The provenance is `None` for the
    rows of a frame without a `provenance` column and the rows whose cell is null, and
    `EVERY_PROVENANCE` for the rows of any provenance, which only a cut rebuilt from a trace
    entry without a `provenance` holds."""

    left_out: _LeftOut

    @classmethod
    def of_trace_entries(cls, entries: Iterable[Mapping[str, object]]) -> PartyTotalsCut:
        """The cut whose `trace_entries` are `entries`, as a version's trace recorded them,
        whatever the code index now declares of their parties. An entry without a `provenance`,
        as versions recorded before the cut keyed totals on the dataset, leaves its refs out of
        the rows of every provenance, as the cut that minted them did. An entry without
        `kept_periods`, as every version recorded before the cut kept whole totals, leaves its
        refs out of every period."""
        left_out: dict[str, dict[_Provenance, dict[str, tuple[str, ...]]]] = {}
        for entry in entries:
            recorded = entry.get("provenance", EVERY_PROVENANCE)
            provenance = (
                recorded if recorded is None or recorded is EVERY_PROVENANCE else str(recorded)
            )
            kept = tuple(
                str(period) for period in cast("list[object]", entry.get("kept_periods", []))
            )
            refs = left_out.setdefault(str(entry["role"]), {}).setdefault(provenance, {})
            for ref in cast("list[object]", entry["entities"]):
                refs[str(ref)] = kept
        return cls(left_out=left_out)

    def chunks(self, frame_chunks: Callable[[], Iterator[Frame]]) -> Callable[[], Iterator[Frame]]:
        """A fresh pass over `frame_chunks` less the rows `left_out` names, on each call. A
        chunk the cut empties is skipped."""

        def cut() -> Iterator[Frame]:
            for chunk in frame_chunks():
                kept = _without(chunk, self.left_out)
                if kept.table.num_rows:
                    yield kept

        return cut

    def trace_entries(self) -> list[PartyTotalsLeftOutModel]:
        """The trace's `party_totals_left_out` entries: one `{role, provenance, entities}` per
        role, provenance and set of kept periods, in role order, then provenance order (`None`
        first), then kept periods (the refs left out of every period first, without
        `kept_periods`), its entities in ref order. A party left out of one dataset's rows is
        named for that dataset alone, so a frame that keeps its rows from another dataset says
        so. An entry's `kept_periods` lists the `time_period` values in which its refs stayed,
        so refs that stayed in different periods take an entry each. Refs left out of every
        provenance keep the `{role, entities}` entry they were rebuilt from, after the role's
        other entries."""
        entries: list[PartyTotalsLeftOutModel] = []
        for role, by_dataset in sorted(self.left_out.items()):
            for dataset, by_ref in sorted(by_dataset.items(), key=lambda item: _order(item[0])):
                for kept, refs in sorted(_refs_by_kept(by_ref).items()):
                    entry: PartyTotalsLeftOutModel = (
                        {"role": role, "entities": refs}
                        if dataset is EVERY_PROVENANCE
                        else {"role": role, "provenance": dataset, "entities": refs}
                    )
                    if kept:
                        entry["kept_periods"] = list(kept)
                    entries.append(entry)
        return entries


@dataclass(frozen=True)
class CutFrame:
    """The frame a cut keeps. `frame_chunks` returns a fresh pass over it, `frame_sha256` is its
    hash (`dfo_frame_grammar.frame_sha256`), `rows` counts its rows, and `distinct` counts the
    distinct non-null values of each of its role columns and its leading years
    (`YEARS_DIMENSION`), as a materialisation's coverage counts them
    (`materialise.columnar.coverage`).
    `left_out` holds the cut's trace entries (`PartyTotalsCut.trace_entries`)."""

    frame_chunks: Callable[[], Iterator[Frame]]
    frame_sha256: str
    rows: int
    distinct: Mapping[str, int]
    left_out: tuple[PartyTotalsLeftOutModel, ...]


def cut_frame(
    frame_chunks: Callable[[], Iterator[Frame]],
    cut: PartyTotalsCut,
    *,
    role_columns: Sequence[str],
) -> CutFrame:
    """The frame `frame_chunks` passes over, less what `cut` leaves out, hashed and counted in
    one pass."""
    kept = cut.chunks(frame_chunks)
    rows = 0
    seen: dict[str, set[object]] = {name: set() for name in (*role_columns, YEARS_DIMENSION)}

    def tallied() -> Iterator[Frame]:
        nonlocal rows
        for chunk in kept():
            rows += chunk.table.num_rows
            for name in role_columns:
                seen[name].update(_distinct(chunk.table.column(name)))
            years = pc.utf8_slice_codeunits(chunk.table.column(TIME_PERIOD_COLUMN), 0, 4)
            seen[YEARS_DIMENSION].update(_distinct(years))
            yield chunk

    digest = stream_frame_sha256(tallied())
    return CutFrame(
        frame_chunks=kept,
        frame_sha256=digest,
        rows=rows,
        distinct={name: len(values) for name, values in seen.items()},
        left_out=tuple(cut.trace_entries()),
    )


def cut_party_totals(
    frame_chunks: Callable[[], Iterator[Frame]],
    *,
    roles: Sequence[str],
    parties: Mapping[str, PartyRow],
) -> PartyTotalsCut:
    """The cut of the frame `frame_chunks` passes over, on its open party `roles`, with kinds from
    the code index's `parties` on each role's axis for each row's dataset. Reads the frame once per
    role, each pass after the cuts of the roles before it."""
    left_out: dict[str, dict[_Provenance, dict[str, tuple[str, ...]]]] = {}
    for role in roles:
        if (axis := role_key_axis(role)) is None:
            continue  # A role on no flow axis holds no party kinds, so no totals to cut.
        totals: defaultdict[tuple[object, str | None], set[str]] = defaultdict(set)
        mixed: set[tuple[object, str | None]] = set()
        is_total: dict[tuple[str, str | None], bool] = {}
        for chunk in frame_chunks():
            if role not in chunk.columns:
                break
            kept = _without(chunk, left_out)
            for ref, period, dataset in _parties_by_cell(kept, role):
                if not isinstance(ref, str):
                    mixed.add((period, dataset))
                    continue
                if (key := (ref, dataset)) not in is_total:
                    datasets = () if dataset is None else (dataset,)
                    is_total[key] = (
                        party_kind_of(ref, axis=axis, parties=parties, datasets=datasets)
                        == _TOTAL_KIND
                    )
                if is_total[key]:
                    totals[period, dataset].add(ref)
                else:
                    mixed.add((period, dataset))
        if cell_cut := _cell_cut(totals, mixed, role=role, parties=parties):
            left_out[role] = cell_cut
    return PartyTotalsCut(left_out=left_out)


def _cell_cut(
    totals: Mapping[tuple[object, str | None], set[str]],
    mixed: set[tuple[object, str | None]],
    *,
    role: str,
    parties: Mapping[str, PartyRow],
) -> dict[_Provenance, dict[str, tuple[str, ...]]]:
    """The refs `role` loses in each provenance and the periods each keeps there, from the
    `totals` of each `(time_period, provenance)` cell. A cell in `mixed`, which holds another
    party, loses every total. Any other cell keeps its whole totals and loses the rest; when it
    holds no whole total, it loses every total if `mixed` names any cell, the role then holding
    another party somewhere, and keeps them all if not. A ref no cell loses is not named, so a
    cut that loses nothing is empty."""
    holds_members = bool(mixed)
    dropped: defaultdict[str | None, set[str]] = defaultdict(set)
    kept: defaultdict[tuple[str | None, str], set[str]] = defaultdict(set)
    for (period, dataset), refs in totals.items():
        whole = (
            set()
            if (period, dataset) in mixed or not isinstance(period, str)
            else {
                ref
                for ref in refs
                if row_party_cell(
                    PARTY_WHOLE_FIELD,
                    ref,
                    role=role,
                    parties=parties,
                    period=period,
                    provenance=dataset,
                )
                == "true"
            }
        )
        stays = whole if whole or holds_members else refs
        for ref in refs:
            if ref not in stays:
                dropped[dataset].add(ref)
            elif isinstance(period, str):
                # A period that is not text names no kept period, so the ref's rows there
                # leave with its rows of any cell that loses it.
                kept[dataset, ref].add(period)
    return {
        dataset: {ref: tuple(sorted(kept[dataset, ref])) for ref in sorted(refs)}
        for dataset, refs in dropped.items()
    }


def _parties_by_cell(chunk: Frame, role: str) -> list[tuple[object, object, str | None]]:
    """The distinct `(ref, time_period, provenance)` triples of `chunk`'s rows, the provenance
    `None` for a frame without a `provenance` column and for a null cell."""
    table = chunk.table
    if PROVENANCE_COLUMN not in table.column_names:
        pairs = table.select([role, TIME_PERIOD_COLUMN]).group_by([role, TIME_PERIOD_COLUMN])
        found = pairs.aggregate([])
        return [
            (ref, period, None)
            for ref, period in zip(
                found.column(role).to_pylist(),
                found.column(TIME_PERIOD_COLUMN).to_pylist(),
                strict=True,
            )
        ]
    keys = [role, TIME_PERIOD_COLUMN, PROVENANCE_COLUMN]
    found = table.select(keys).group_by(keys).aggregate([])
    return list(
        zip(
            found.column(role).to_pylist(),
            found.column(TIME_PERIOD_COLUMN).to_pylist(),
            found.column(PROVENANCE_COLUMN).to_pylist(),
            strict=True,
        )
    )


def _refs_by_kept(by_ref: Mapping[str, tuple[str, ...]]) -> dict[tuple[str, ...], list[str]]:
    """`by_ref`'s refs grouped by the periods they kept, each group in `by_ref`'s order, which
    is ref order for a cut and the recorded order for one rebuilt from a trace."""
    grouped: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)
    for ref, kept in by_ref.items():
        grouped[kept].append(ref)
    return grouped


def _distinct(values: pa.Array | pa.ChunkedArray) -> list[object]:
    """The distinct non-null values of `values`."""
    return pc.unique(values).drop_null().to_pylist()


def _order(provenance: _Provenance) -> tuple[int, str]:
    """The order of `provenance` among a role's trace entries: `None`, the named provenances,
    then `EVERY_PROVENANCE`."""
    if provenance is None:
        return 0, ""
    if provenance is EVERY_PROVENANCE:
        return 2, ""
    return 1, provenance


def _without(chunk: Frame, left_out: _LeftOut) -> Frame:
    """`chunk` less each row whose party on a role of `left_out` is one of that role's refs,
    unless the row's `time_period` is one the ref kept. A null cell matches no ref. A ref left
    out in a provenance leaves out its rows of that provenance alone, one left out for `None`
    its rows with no provenance, and one left out for `EVERY_PROVENANCE` its rows of any
    provenance."""
    table = chunk.table
    left: pa.ChunkedArray | None = None
    for role, by_dataset in left_out.items():
        for dataset, by_ref in by_dataset.items():
            for kept, refs in _refs_by_kept(by_ref).items():
                match = _left_out_rows(table, role, dataset, refs, kept)
                if match is not None:
                    left = match if left is None else cast("pa.ChunkedArray", pc.or_(left, match))
    if left is None:
        return chunk
    # Two column operands give a column. The stubs widen it to a scalar too, here and above.
    return Frame(
        columns=chunk.columns, table=table.filter(cast("pa.ChunkedArray", pc.invert(left)))
    )


def _left_out_rows(
    table: pa.Table, role: str, dataset: _Provenance, refs: Sequence[str], kept: Sequence[str]
) -> pa.ChunkedArray | None:
    """The mask of `table`'s rows of `dataset` whose `role` party is one of `refs` and whose
    `time_period` is not one of `kept` (`_without`), or `None` when `table` holds no row of
    `dataset`."""
    column = table.column(role)
    match = pc.is_in(column, value_set=pa.array(refs, type=column.type))
    if kept:
        period = table.column(TIME_PERIOD_COLUMN)
        match = pc.and_(
            match, pc.invert(pc.is_in(period, value_set=pa.array(kept, type=period.type)))
        )
    if dataset is not EVERY_PROVENANCE and PROVENANCE_COLUMN in table.column_names:
        # A `None` dataset matches the null cells, and a named one no null cell.
        provenance = table.column(PROVENANCE_COLUMN)
        in_dataset = pc.is_in(provenance, value_set=pa.array([dataset], type=provenance.type))
        match = pc.and_(match, in_dataset)
    elif dataset is not None and dataset is not EVERY_PROVENANCE:
        return None
    # Two column operands give a column; the stubs widen it to a scalar too.
    return cast("pa.ChunkedArray", match)


__all__ = [
    "EVERY_PROVENANCE",
    "TRACE_KEY",
    "YEARS_DIMENSION",
    "CutFrame",
    "PartyTotalsCut",
    "cut_frame",
    "cut_party_totals",
    "open_party_roles",
]
