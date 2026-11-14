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

A role decides each cell on its own. A cell is one series of the role, one full `time_period`
value and one provenance. A series is the value of every other column of the frame's grain, which
no two rows share (every column but `value` and `status`, `dfo_frame_grammar.columnar`): the
StatVar, the frame's other role columns, bound or open, `unit` and `measurementMethod`. So Kenya's
receipts and Ghana's form two series of the provider role, and each keeps its own totals.

A cell in which the role holds another party loses its totals. A cell in which it holds only
totals keeps the totals whose row reads as a whole total (`materialise.enrichment.row_party_cell`
gives `party_whole` `true`, the cell the analysis engine reads as `{role}_party_whole`), since
whole totals hold disjoint populations, and loses the rest. A totals-only cell with no whole total
loses its totals when the role holds another party in that series in any period or provenance,
and keeps them all when it holds none, since the totals are then the only parties the series has
(the G7's flows to Kenya). DAC2B publishes India's 1968 receipts only as totals (DAC countries,
the G7 and official donors), so the cut keeps official donors, the one whole total, for that
year, whether or not later years list members.

The cut reads the frame before any reshape, so a `groupby` over it sums members alone. Each pass
holds one chunk at a time. A role's decision holds the cells of its total rows and the cells in
which it holds another party.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc
import rfc8785
from dfo_frame_grammar import (
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    Frame,
    UnitCell,
    unit_cell_from_json,
    unit_cell_to_json,
    unit_cells,
    unit_codes,
)
from pydantic import JsonValue

from code_index.types import PartyRow
from datasets_api.manifest_models import PartyTotalsKeptCellModel, PartyTotalsLeftOutModel
from dfo_platform.columns import PROVENANCE_COLUMN
from dfo_platform.roles import RoleAxis, role_key_axis
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

_NOT_SERIES = frozenset({TIME_PERIOD_COLUMN, PROVENANCE_COLUMN, VALUE_COLUMN, STATUS_COLUMN})
"""The columns of a frame that name no series: `value` and `status`, which the frame's grain
leaves out, and the `time_period` and provenance of a cell."""


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

type _Series = tuple[tuple[str, str | None], ...]
"""A series of a role: each column of the frame's grain but the role, `time_period` and
`provenance`, with its cell as text (`_cell_text`), in column-name order."""

type _Cell = tuple[str, _Series]
"""A cell of a role: a `time_period` value and a series."""

type _LeftOut = Mapping[str, Mapping[_Provenance, Mapping[str, frozenset[_Cell]]]]
"""Each open party role that lost rows, mapped to each provenance it lost rows in, mapped to
each ref it lost there and the cells in which that ref's rows stayed, empty when they left
every cell."""


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
        `kept_in`, as every version recorded before the cut kept whole totals, leaves its refs
        out of every cell."""
        left_out: dict[str, dict[_Provenance, dict[str, frozenset[_Cell]]]] = {}
        for entry in entries:
            recorded = entry.get("provenance", EVERY_PROVENANCE)
            provenance = (
                recorded if recorded is None or recorded is EVERY_PROVENANCE else str(recorded)
            )
            kept = frozenset(
                _recorded_cell(cast("Mapping[str, object]", cell))
                for cell in cast("list[object]", entry.get("kept_in", []))
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
        role, provenance and set of kept cells, in role order, then provenance order (`None`
        first), then kept cells (the refs left out of every cell first, without `kept_in`, then
        by their sorted `kept_in` lists, compared cell by cell in `kept_in` order), its entities
        in ref order. A party left out of one dataset's rows is named for that dataset alone, so
        a frame that keeps its rows from another dataset says so. An entry's `kept_in` lists
        the cells in which its refs stayed (`_kept_cell`), so refs that stayed in different
        cells take an entry each. Refs left out of every provenance keep the `{role, entities}`
        entry they were rebuilt from, after the role's other entries."""
        entries: list[PartyTotalsLeftOutModel] = []
        for role, by_dataset in sorted(self.left_out.items()):
            for dataset, by_ref in sorted(by_dataset.items(), key=lambda item: _order(item[0])):
                for kept, refs in sorted(
                    _refs_by_kept(by_ref).items(), key=lambda item: _kept_order(item[0])
                ):
                    entry: PartyTotalsLeftOutModel = (
                        {"role": role, "entities": refs}
                        if dataset is EVERY_PROVENANCE
                        else {"role": role, "provenance": dataset, "entities": refs}
                    )
                    if kept:
                        entry["kept_in"] = [
                            _kept_cell(cell) for cell in sorted(kept, key=_cell_order)
                        ]
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
    left_out: dict[str, dict[_Provenance, dict[str, frozenset[_Cell]]]] = {}
    for role in roles:
        if (axis := role_key_axis(role)) is None:
            continue  # A role on no flow axis holds no party kinds, so no totals to cut.
        cells = _RoleCells()
        is_total: dict[tuple[str, str | None], bool] = {}
        for chunk in frame_chunks():
            if role not in chunk.columns:
                break
            kept = _without(chunk, left_out)
            total = _total_rows(kept.table, role, axis=axis, parties=parties, is_total=is_total)
            cells.read(kept, role, total=total)
        if cell_cut := _cell_cut(cells, role=role, parties=parties):
            left_out[role] = cell_cut
    return PartyTotalsCut(left_out=left_out)


type _CellKey = tuple[object, str | None, int]
"""A cell of one provenance as `_RoleCells` holds it: its `time_period`, its provenance and the
index of its series in `_RoleCells.series`."""


@dataclass
class _RoleCells:
    """What one role's pass read of a frame: `totals`, the total refs of each cell that holds
    any, and `others`, each cell in which the role holds another party, a null cell included.
    `series` lists each series a cell names, which `series_index` indexes."""

    totals: defaultdict[_CellKey, set[str]] = field(default_factory=lambda: defaultdict(set))
    others: set[_CellKey] = field(default_factory=set)
    series: list[_Series] = field(default_factory=list)
    series_index: dict[tuple[str | None, ...], int] = field(default_factory=dict)

    def read(self, chunk: Frame, role: str, *, total: pa.ChunkedArray | None) -> None:
        """Add `chunk`'s cells of `role`, whose rows `total` marks as total rows (`None`
        marking none)."""
        table = chunk.table
        columns = _series_columns(chunk.columns, role)
        keys = pa.table(
            {
                "ref": table.column(role),
                "period": table.column(TIME_PERIOD_COLUMN),
                "provenance": (
                    table.column(PROVENANCE_COLUMN)
                    if PROVENANCE_COLUMN in table.column_names
                    else pa.nulls(table.num_rows, pa.string())
                ),
                **{f"series_{at}": _text_column(table, name) for at, name in enumerate(columns)},
            }
        )
        cell_names = keys.column_names[1:]
        others = keys if total is None else keys.filter(pc.invert(total))
        found = others.select(cell_names).group_by(cell_names).aggregate([])
        for period, provenance, *values in _rows(found):
            self.others.add(self._key(period, provenance, values, columns))
        if total is None:
            return
        for ref, period, provenance, *values in _rows(keys.filter(total)):
            self.totals[self._key(period, provenance, values, columns)].add(cast("str", ref))

    def _key(
        self,
        period: object,
        provenance: object,
        values: Sequence[object],
        columns: Sequence[str],
    ) -> _CellKey:
        series = tuple(cast("str | None", value) for value in values)
        if (index := self.series_index.get(series)) is None:
            index = self.series_index[series] = len(self.series)
            self.series.append(tuple(sorted(zip(columns, series, strict=True))))
        return period, cast("str | None", provenance), index


def _cell_cut(
    cells: _RoleCells,
    *,
    role: str,
    parties: Mapping[str, PartyRow],
) -> dict[_Provenance, dict[str, frozenset[_Cell]]]:
    """The refs `role` loses in each provenance and the cells each keeps there, from `cells`.
    A cell holding another party loses every total. Any other cell keeps its whole totals and
    loses the rest; when it holds no whole total, it loses every total if the role holds
    another party in its series in some cell, and keeps them all if not. A ref no cell loses
    is not named, so a cut that loses nothing is empty."""
    with_others = {series for _, _, series in cells.others}
    whole: dict[tuple[str, str, str | None], bool] = {}
    dropped: defaultdict[str | None, set[str]] = defaultdict(set)
    kept: defaultdict[tuple[str | None, str], set[_Cell]] = defaultdict(set)
    for key, refs in cells.totals.items():
        period, dataset, series = key
        stays: set[str] = set()
        if key not in cells.others and isinstance(period, str):
            for ref in refs:
                if (whole_key := (ref, period, dataset)) not in whole:
                    whole[whole_key] = (
                        row_party_cell(
                            PARTY_WHOLE_FIELD,
                            ref,
                            role=role,
                            parties=parties,
                            period=period,
                            provenance=dataset,
                        )
                        == "true"
                    )
                if whole[whole_key]:
                    stays.add(ref)
        if not stays and series not in with_others:
            stays = refs
        for ref in refs:
            if ref not in stays:
                dropped[dataset].add(ref)
            elif isinstance(period, str):
                # A period that is not text names no kept cell, so the ref's rows there leave
                # with its rows of any cell that loses it.
                kept[dataset, ref].add((period, cells.series[series]))
    return {
        dataset: {ref: frozenset(kept[dataset, ref]) for ref in sorted(refs)}
        for dataset, refs in dropped.items()
    }


def _series_columns(columns: Sequence[str], role: str) -> tuple[str, ...]:
    """The columns of a frame of `columns` that name a series of `role`: its grain less `role`,
    `time_period` and `provenance`."""
    return tuple(name for name in columns if name != role and name not in _NOT_SERIES)


def _total_rows(
    table: pa.Table,
    role: str,
    *,
    axis: RoleAxis,
    parties: Mapping[str, PartyRow],
    is_total: dict[tuple[str, str | None], bool],
) -> pa.ChunkedArray | None:
    """The mask of `table`'s rows whose `role` party the code index's `parties` declares a
    total on `axis` in the row's dataset, `None` when no row's is. `is_total` caches each
    `(ref, provenance)` kind across chunks."""
    with_provenance = PROVENANCE_COLUMN in table.column_names
    keys = [role, PROVENANCE_COLUMN] if with_provenance else [role]
    found = table.select(keys).group_by(keys).aggregate([])
    refs = found.column(role).to_pylist()
    datasets = (
        found.column(PROVENANCE_COLUMN).to_pylist() if with_provenance else [None] * len(refs)
    )
    totals: defaultdict[str | None, list[str]] = defaultdict(list)
    for ref, dataset in zip(refs, datasets, strict=True):
        if not isinstance(ref, str):
            continue  # A null cell holds no total, so it counts as another party.
        if (key := (ref, dataset)) not in is_total:
            is_total[key] = (
                party_kind_of(
                    ref, axis=axis, parties=parties, datasets=() if dataset is None else (dataset,)
                )
                == _TOTAL_KIND
            )
        if is_total[key]:
            totals[dataset].append(ref)
    mask: pa.ChunkedArray | None = None
    for dataset, total_refs in totals.items():
        if (match := _party_rows(table, role, dataset, total_refs)) is not None:
            mask = match if mask is None else cast("pa.ChunkedArray", pc.or_(mask, match))
    return mask


def _rows(table: pa.Table) -> Iterator[tuple[object, ...]]:
    """`table`'s rows as tuples, in column order."""
    return zip(*(column.to_pylist() for column in table.columns), strict=True)


def _text_column(table: pa.Table, name: str) -> pa.ChunkedArray | pa.Array:
    """`table`'s column `name` as text: a `unit` cell as the JCS text of its JSON form
    (`_unit_text`), any other column as it is, every one a string column."""
    column = table.column(name)
    if name != UNIT_COLUMN:
        return column
    codes = unit_codes(column)
    texts = pa.array([_unit_text(cell) for cell in unit_cells(column, codes)], pa.string())
    return texts.take(codes.codes)


def _unit_text(cell: UnitCell | None) -> str | None:
    """The JCS text of the JSON form of the `unit` cell `cell` (`unit_cell_to_json`)."""
    if cell is None:
        return None
    return rfc8785.dumps(unit_cell_to_json(cell)).decode()


def _recorded_cell(recorded: Mapping[str, object]) -> _Cell:
    """The cell a trace entry's `kept_in` item records (`_kept_cell`)."""
    series = cast("Mapping[str, object]", recorded["series"])
    return (
        str(recorded["time_period"]),
        tuple(
            sorted(
                (
                    str(name),
                    (
                        None
                        if value is None
                        else _unit_text(unit_cell_from_json(value))
                        if name == UNIT_COLUMN
                        else str(value)
                    ),
                )
                for name, value in series.items()
            )
        ),
    )


def _kept_cell(cell: _Cell) -> PartyTotalsKeptCellModel:
    """`cell` as a `kept_in` item: its `time_period` and its `series`, each column mapped to
    its cell, a `unit` cell in its JSON form and a null cell as null."""
    period, series = cell
    return {
        "time_period": period,
        "series": {name: _series_value(name, text) for name, text in series},
    }


def _series_value(name: str, text: str | None) -> str | dict[str, JsonValue] | None:
    """The JSON value of a series cell held as text (`_text_column`): a unit cell's JSON form,
    a ref or an object, and any other cell as it is."""
    if text is None or name != UNIT_COLUMN:
        return text
    return cast("str | dict[str, JsonValue]", json.loads(text))


def _cell_order(cell: _Cell) -> tuple[object, ...]:
    """The order of `cell` among an entry's `kept_in` items: by `time_period`, then by its
    series cells in column-name order, each null first, then a string by its code points, then
    an object by the code points of its JCS text."""
    period, series = cell
    return period, tuple((name, _value_order(name, text)) for name, text in series)


def _value_order(name: str, text: str | None) -> tuple[int, str]:
    """The order of one series cell (`_cell_order`)."""
    value = _series_value(name, text)
    if value is None:
        return 0, ""
    if isinstance(value, str):
        return 1, value
    return 2, cast("str", text)


def _kept_order(kept: frozenset[_Cell]) -> tuple[object, ...]:
    """The order of a group of refs kept in `kept` among a role's entries of one provenance:
    the refs kept in no cell first, then by the sorted cells, cell by cell."""
    return tuple(_cell_order(cell) for cell in sorted(kept, key=_cell_order))


def _refs_by_kept(by_ref: Mapping[str, frozenset[_Cell]]) -> dict[frozenset[_Cell], list[str]]:
    """`by_ref`'s refs grouped by the cells they kept, each group in `by_ref`'s order, which
    is ref order for a cut and the recorded order for one rebuilt from a trace."""
    grouped: defaultdict[frozenset[_Cell], list[str]] = defaultdict(list)
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
    unless the row lies in a cell the ref kept (`_in_cells`). A null cell matches no ref. A
    ref left out in a provenance leaves out its rows of that provenance alone, one left out for
    `None` its rows with no provenance, and one left out for `EVERY_PROVENANCE` its rows of any
    provenance. The rows leave in one filter."""
    table = chunk.table
    keyed: dict[tuple[str, ...], pa.Table] = {}
    left: pa.ChunkedArray | None = None
    for role, by_dataset in left_out.items():
        for dataset, by_ref in by_dataset.items():
            for kept, refs in _refs_by_kept(by_ref).items():
                match = _party_rows(table, role, dataset, refs)
                if match is None:
                    continue
                if kept:
                    match = cast(
                        "pa.ChunkedArray",
                        pc.and_(match, pc.invert(_in_cells(table, kept, keyed=keyed))),
                    )
                left = match if left is None else cast("pa.ChunkedArray", pc.or_(left, match))
    if left is None:
        return chunk
    # Two column operands give a column. The stubs widen it to a scalar too, here and above.
    return Frame(
        columns=chunk.columns, table=table.filter(cast("pa.ChunkedArray", pc.invert(left)))
    )


def _party_rows(
    table: pa.Table, role: str, dataset: _Provenance, refs: Sequence[str]
) -> pa.ChunkedArray | None:
    """The mask of `table`'s rows of `dataset` whose `role` party is one of `refs`, or `None`
    when `table` holds no row of `dataset`."""
    column = table.column(role)
    match = pc.is_in(column, value_set=pa.array(refs, type=column.type))
    if dataset is not EVERY_PROVENANCE and PROVENANCE_COLUMN in table.column_names:
        # A `None` dataset matches the null cells, and a named one no null cell.
        provenance = table.column(PROVENANCE_COLUMN)
        in_dataset = pc.is_in(provenance, value_set=pa.array([dataset], type=provenance.type))
        match = pc.and_(match, in_dataset)
    elif dataset is not None and dataset is not EVERY_PROVENANCE:
        return None
    # Two column operands give a column; the stubs widen it to a scalar too.
    return cast("pa.ChunkedArray", match)


def _in_cells(
    table: pa.Table, cells: Iterable[_Cell], *, keyed: dict[tuple[str, ...], pa.Table]
) -> pa.Array:
    """The mask of `table`'s rows that lie in one of `cells`: whose `time_period` and each
    column a cell's series names hold that cell's values, a null cell matching null. A cell
    naming a column `table` lacks holds no row. `keyed` holds `table`'s join keys over each
    set of columns read so far, with each row's index as `row`."""
    by_columns: defaultdict[tuple[str, ...], list[_Cell]] = defaultdict(list)
    for cell in cells:
        by_columns[tuple(name for name, _ in cell[1])].append(cell)
    rows = pa.array(range(table.num_rows), pa.int64())
    found: list[pa.Array] = []
    for columns, group in by_columns.items():
        if not set(columns) <= set(table.column_names):
            continue
        if (frame_keys := keyed.get(columns)) is None:
            frame_keys = keyed[columns] = _join_keys(
                [
                    table.column(TIME_PERIOD_COLUMN),
                    *(_text_column(table, name) for name in columns),
                ]
            ).append_column("row", rows)
        cell_keys = _join_keys(
            [
                pa.array([period for period, _ in group], pa.string()),
                *(
                    pa.array([series[at][1] for _, series in group], pa.string())
                    for at in range(len(columns))
                ),
            ]
        )
        matched = frame_keys.join(cell_keys, keys=cell_keys.column_names, join_type="left semi")
        found.extend(matched.column("row").chunks)
    hits = pa.chunked_array(found, pa.int64()).combine_chunks()
    return cast("pa.Array", pc.is_in(rows, value_set=hits))


def _join_keys(columns: Sequence[pa.Array | pa.ChunkedArray]) -> pa.Table:
    """Join keys over `columns`: each a column marking its nulls and the column with its nulls
    as `""`, since a join matches no null key."""
    keys: dict[str, object] = {}
    for at, column in enumerate(columns):
        keys[f"null_{at}"] = pc.is_null(column)
        keys[f"text_{at}"] = pc.fill_null(column, pa.scalar(""))
    return pa.table(keys)


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
