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

The cut reads the frame before any reshape. A series whose cells hold only totals, none of them
whole, keeps them all, so a sum over that role can count a total beside the totals nested in it.

Each pass holds one chunk at a time. A role with no total row reads the frame once and decides
nothing. A role with one reads again the chunks before its first total row, and holds, as Arrow
tables, the distinct cells of its total rows and the distinct cells in which it holds another
party, which it compares once every chunk is read.
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

import numpy as np
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
from datasets_api.manifest_models import PartyTotalsKeptSeriesModel, PartyTotalsLeftOutModel
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
`provenance`, with its cell as text (`_text_column`), in column-name order."""

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
                cell
                for item in cast("list[object]", entry.get("kept_in", []))
                for cell in _recorded_cells(cast("Mapping[str, object]", item))
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
        by their `kept_in` lists compared item by item), its entities in ref order. A party left
        out of one dataset's rows is named for that dataset alone, so a frame that keeps its
        rows from another dataset says so. An entry's `kept_in` lists each series in which its
        refs stayed with the `time_period` values where they stayed (`_kept_in`), so refs that
        stayed in different cells take an entry each. Refs left out of every provenance keep
        the `{role, entities}` entry they were rebuilt from, after the role's other entries."""
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
                        entry["kept_in"] = _kept_in(kept)
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
    role, each pass after the cuts of the roles before it, and a role with total rows reads again
    the chunks before its first (`_role_cells`)."""
    left_out: dict[str, dict[_Provenance, dict[str, frozenset[_Cell]]]] = {}
    for role in roles:
        if (axis := role_key_axis(role)) is None:
            continue  # A role on no flow axis holds no party kinds, so no totals to cut.
        cells = _role_cells(frame_chunks, role, axis=axis, parties=parties, left_out=left_out)
        if cells is not None and (cell_cut := _cell_cut(cells, role=role, parties=parties)):
            left_out[role] = cell_cut
    return PartyTotalsCut(left_out=left_out)


def _role_cells(
    frame_chunks: Callable[[], Iterator[Frame]],
    role: str,
    *,
    axis: RoleAxis,
    parties: Mapping[str, PartyRow],
    left_out: _LeftOut,
) -> _RoleCells | None:
    """The cells of `role` in the frame `frame_chunks` passes over, less what `left_out` leaves
    out, or `None` when no row's party is a total, which leaves the role nothing to decide.
    Cells are read from the chunk of the first total row on; the chunks before it, which hold
    no total, are then read again for their cells."""
    is_total: dict[tuple[str, str | None], bool] = {}
    cells: _RoleCells | None = None
    for at, chunk in enumerate(frame_chunks()):
        if role not in chunk.columns:
            break
        kept = _without(chunk, left_out)
        total = _total_rows(kept.table, role, axis=axis, parties=parties, is_total=is_total)
        if cells is None:
            if total is None:
                continue
            cells = _RoleCells(role=role, columns=_series_columns(chunk.columns, role))
            for earlier in itertools.islice(frame_chunks(), at):
                cells.read(_without(earlier, left_out), total=None)
        cells.read(kept, total=total)
    return cells


_PERIOD_KEY = "period"
_DATASET_KEY = "dataset"
_REF_KEY = "ref"


@dataclass
class _RoleCells:
    """What one role's pass read of a frame whose series columns are `columns`, one Arrow
    table per chunk, each column text (`_cell_keys`): `totals`, the distinct cells of the total
    rows, each with its total ref, and `others`, the distinct cells in which the role holds
    another party, a null cell included."""

    role: str
    columns: tuple[str, ...]
    totals: list[pa.Table] = field(default_factory=list)
    others: list[pa.Table] = field(default_factory=list)

    def read(self, chunk: Frame, *, total: pa.ChunkedArray | None) -> None:
        """Add `chunk`'s cells, whose rows `total` marks as total rows (`None` marking none)."""
        keys = _cell_keys(chunk.table, self.columns)
        others = keys if total is None else keys.filter(pc.invert(total))
        self.others.append(_distinct_rows(others))
        if total is not None:
            refs = keys.append_column(_REF_KEY, _text(chunk.table.column(self.role)))
            self.totals.append(_distinct_rows(refs.filter(total)))


def _cell_keys(table: pa.Table, columns: Sequence[str]) -> pa.Table:
    """The cell of each row of `table`, as text: its `time_period` (null when not text), its
    provenance (null for a table without the column), and each series column of `columns`
    (`_text_column`), named by its index."""
    return pa.table(
        {
            _PERIOD_KEY: _period_text(table.column(TIME_PERIOD_COLUMN)),
            _DATASET_KEY: (
                _text(table.column(PROVENANCE_COLUMN))
                if PROVENANCE_COLUMN in table.column_names
                else pa.nulls(table.num_rows, pa.string())
            ),
            **{_series_key(at): _text_column(table, name) for at, name in enumerate(columns)},
        }
    )


def _series_key(at: int) -> str:
    """The name of a cell's series column `at` in `_cell_keys`."""
    return f"series_{at}"


def _distinct_rows(table: pa.Table) -> pa.Table:
    """`table`'s distinct rows, every column as `pa.string()`, so tables of chunks whose
    columns differ in dictionary concatenate."""
    distinct = table.group_by(table.column_names).aggregate([])
    return pa.table({name: _text(distinct.column(name)) for name in distinct.column_names})


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
    is not named, so a cut that loses nothing is empty.

    The cells are compared in Arrow. Python reads each distinct (ref, `time_period`,
    provenance) that may be a whole total, the refs lost and the cells a lost ref kept."""
    totals = _distinct_rows(pa.concat_tables(cells.totals))
    others = pa.concat_tables(cells.others)
    series_names = [_series_key(at) for at in range(len(cells.columns))]
    cell_names = [_PERIOD_KEY, _DATASET_KEY, *series_names]
    in_cell = _matched(totals, others, keys=cell_names)
    in_series = _matched(totals, others, keys=series_names)
    # A period that is not text is null here: it reads as no whole total and names no cell.
    text_period = totals.column(_PERIOD_KEY).is_valid().to_numpy(zero_copy_only=False)

    may_be_whole = totals.filter(pa.array(text_period & ~in_cell))
    candidates = _distinct_rows(may_be_whole.select([_REF_KEY, _PERIOD_KEY, _DATASET_KEY]))
    whole = [
        row_party_cell(
            PARTY_WHOLE_FIELD, ref, role=role, parties=parties, period=period, provenance=dataset
        )
        == "true"
        for ref, period, dataset in _rows(candidates)
    ]
    keys = [_REF_KEY, _PERIOD_KEY, _DATASET_KEY]
    is_whole = (
        text_period
        & ~in_cell
        & _matched(totals, candidates.filter(pa.array(whole, pa.bool_())), keys=keys)
    )
    with_whole = _matched(totals, totals.filter(pa.array(is_whole)), keys=cell_names)
    stays = np.where(with_whole, is_whole, ~in_series)

    lost = _distinct_rows(totals.filter(pa.array(~stays)).select([_DATASET_KEY, _REF_KEY]))
    of_lost = _matched(totals, lost, keys=[_DATASET_KEY, _REF_KEY])
    kept_rows = totals.filter(pa.array(stays & text_period & of_lost))
    kept: defaultdict[tuple[str | None, str], set[_Cell]] = defaultdict(set)
    series: dict[tuple[tuple[str, ...], tuple[str | None, ...]], _Series] = {}
    for period, dataset, ref, *values in _rows(
        kept_rows.select([_PERIOD_KEY, _DATASET_KEY, _REF_KEY, *series_names])
    ):
        if (series_key := (cells.columns, tuple(values))) not in series:
            series[series_key] = tuple(sorted(zip(cells.columns, values, strict=True)))
        kept[dataset, ref].add((period, series[series_key]))
    dropped: defaultdict[str | None, set[str]] = defaultdict(set)
    for dataset, ref in _rows(lost):
        dropped[dataset].add(ref)
    return {
        dataset: {ref: frozenset(kept[dataset, ref]) for ref in sorted(refs)}
        for dataset, refs in dropped.items()
    }


def _rows(table: pa.Table) -> Iterator[tuple[Any, ...]]:
    """`table`'s rows as tuples, in column order."""
    return zip(*(column.to_pylist() for column in table.columns), strict=True)


def _matched(left: pa.Table, right: pa.Table, *, keys: Sequence[str]) -> np.ndarray:
    """Whether each row of `left` holds the cells of some row of `right` in `keys`, a null
    cell matching null (`_join_keys`)."""
    left_keys = _join_keys([left.column(name) for name in keys]).append_column(
        "row", pa.array(np.arange(left.num_rows, dtype=np.int64))
    )
    right_keys = _join_keys([right.column(name) for name in keys])
    # A right semi join builds its table on `left`, the total cells, and probes with `right`.
    found = right_keys.join(left_keys, keys=right_keys.column_names, join_type="right semi")
    matched = np.zeros(left.num_rows, dtype=bool)
    matched[found.column("row").to_numpy()] = True
    return matched


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
    found = pa.table({name: _text(table.column(name)) for name in keys})
    found = found.group_by(keys).aggregate([])
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


def _text(column: pa.ChunkedArray | pa.Array) -> pa.ChunkedArray | pa.Array:
    """`column` as `pa.string()`, from a dictionary, large or view string column."""
    return column if column.type == pa.string() else pc.cast(column, pa.string())


def _period_text(column: pa.ChunkedArray) -> pa.ChunkedArray | pa.Array:
    """A `time_period` column as `pa.string()`, null in every row when its cells are not text,
    since such a period names no kept cell and reads as no whole total."""
    value_type = (
        column.type.value_type if isinstance(column.type, pa.DictionaryType) else column.type
    )
    if (
        pa.types.is_string(value_type)
        or pa.types.is_large_string(value_type)
        or pa.types.is_string_view(value_type)
    ):
        return _text(column)
    return pa.nulls(len(column), pa.string())


def _text_column(table: pa.Table, name: str) -> pa.ChunkedArray | pa.Array:
    """`table`'s column `name` as text: a `unit` cell as the JCS text of its JSON form
    (`_unit_text`), encoded as a dictionary of the distinct texts, any other column as
    `pa.string()`."""
    column = table.column(name)
    if name != UNIT_COLUMN:
        return _text(column)
    codes = unit_codes(column)
    texts = pa.array([_unit_text(cell) for cell in unit_cells(column, codes)], pa.string())
    return pa.DictionaryArray.from_arrays(codes.codes, texts)


def _unit_text(cell: UnitCell | None) -> str | None:
    """The JCS text of the JSON form of the `unit` cell `cell` (`unit_cell_to_json`)."""
    if cell is None:
        return None
    return rfc8785.dumps(unit_cell_to_json(cell)).decode()


def _recorded_cells(recorded: Mapping[str, object]) -> Iterator[_Cell]:
    """The cells a trace entry's `kept_in` item records (`_kept_in`): each of its
    `time_periods` in its `series`."""
    series = tuple(
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
            for name, value in cast("Mapping[str, object]", recorded["series"]).items()
        )
    )
    for period in cast("list[object]", recorded["time_periods"]):
        yield str(period), series


def _kept_in(kept: frozenset[_Cell]) -> list[PartyTotalsKeptSeriesModel]:
    """`kept` as an entry's `kept_in` items: one per series, with its `series`, each column
    mapped to its cell, a `unit` cell in its JSON form and a null cell as null, and the
    `time_periods` in which the entry's refs stayed in it, in `_kept_series` order."""
    return [
        {
            "series": {name: _series_value(name, text) for name, text in series},
            "time_periods": periods,
        }
        for series, periods in _kept_series(kept)
    ]


def _kept_series(kept: frozenset[_Cell]) -> list[tuple[_Series, list[str]]]:
    """Each series of `kept` with its sorted `time_period` values, sorted by series
    (`_series_order`)."""
    periods: defaultdict[_Series, set[str]] = defaultdict(set)
    for period, series in kept:
        periods[series].add(period)
    return [(series, sorted(periods[series])) for series in sorted(periods, key=_series_order)]


def _series_value(name: str, text: str | None) -> str | dict[str, JsonValue] | None:
    """The JSON value of a series cell held as text (`_text_column`): a unit cell's JSON form,
    a ref or an object, and any other cell as it is."""
    if text is None or name != UNIT_COLUMN:
        return text
    return cast("str | dict[str, JsonValue]", json.loads(text))


def _series_order(series: _Series) -> tuple[object, ...]:
    """The order of `series` among an entry's `kept_in` items: by its cells in column-name
    order, each null first, then a string by its code points, then an object by the code
    points of its JCS text."""
    return tuple((name, _value_order(name, text)) for name, text in series)


def _value_order(name: str, text: str | None) -> tuple[int, str]:
    """The order of one series cell (`_series_order`)."""
    value = _series_value(name, text)
    if value is None:
        return 0, ""
    if isinstance(value, str):
        return 1, value
    return 2, cast("str", text)


def _kept_order(kept: frozenset[_Cell]) -> tuple[object, ...]:
    """The order of a group of refs kept in `kept` among a role's entries of one provenance:
    the refs kept in no cell first, then by their `kept_in` items, item by item, each by its
    series and then its `time_periods`."""
    return tuple((_series_order(series), periods) for series, periods in _kept_series(kept))


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
    unless the row lies in a cell the ref kept (`_kept_rows`). A null cell matches no ref. A
    ref left out in a provenance leaves out its rows of that provenance alone, one left out for
    `None` its rows with no provenance, and one left out for `EVERY_PROVENANCE` its rows of any
    provenance. The rows leave in one filter."""
    table = chunk.table
    left: np.ndarray | None = None
    for role, by_dataset in left_out.items():
        for dataset, by_ref in by_dataset.items():
            match = _party_rows(table, role, dataset, list(by_ref))
            if match is None:
                continue
            rows = np.asarray(match.to_numpy(), dtype=bool)
            if any(by_ref.values()) and rows.any():
                rows[_kept_rows(table, role, by_ref, rows=rows)] = False
            left = rows if left is None else left | rows
    if left is None or not left.any():
        return chunk
    return Frame(columns=chunk.columns, table=table.filter(pa.array(~left)))


def _kept_rows(
    table: pa.Table, role: str, by_ref: Mapping[str, frozenset[_Cell]], *, rows: np.ndarray
) -> np.ndarray:
    """The indices of the rows of `table` among `rows` whose `role` party lies in a cell
    `by_ref` names it kept in: its `time_period` and each column the cell's series names hold
    that cell's values, a null cell matching null. One semi join per set of series columns,
    of which a cut's cells name one, matches the rows against every (ref, cell) pair. A cell
    naming a column `table` lacks holds no row."""
    by_columns: defaultdict[tuple[str, ...], list[tuple[str | None, ...]]] = defaultdict(list)
    for ref, cells in by_ref.items():
        for period, series in cells:
            by_columns[tuple(name for name, _ in series)].append(
                (ref, period, *(text for _, text in series))
            )
    indices = np.flatnonzero(rows)
    found: list[np.ndarray] = [np.empty(0, dtype=np.int64)]
    for columns, cells in by_columns.items():
        if not set(columns) <= set(table.column_names):
            continue
        # The text columns are cast before the take, which reads no string view.
        candidates = pa.table(
            {
                name: (
                    column
                    if name == UNIT_COLUMN
                    else _period_text(column)
                    if name == TIME_PERIOD_COLUMN
                    else _text(column)
                )
                for name in (role, TIME_PERIOD_COLUMN, *columns)
                for column in (table.column(name),)
            }
        ).take(indices)
        row_keys = _join_keys(
            [
                candidates.column(role),
                candidates.column(TIME_PERIOD_COLUMN),
                *(_text_column(candidates, name) for name in columns),
            ]
        ).append_column("row", pa.array(indices))
        cell_keys = _join_keys(
            [pa.array(values, pa.string()) for values in zip(*cells, strict=True)]
        )
        matched = cell_keys.join(row_keys, keys=cell_keys.column_names, join_type="right semi")
        found.append(matched.column("row").to_numpy())
    return np.concatenate(found)


def _party_rows(
    table: pa.Table, role: str, dataset: _Provenance, refs: Sequence[str]
) -> pa.ChunkedArray | None:
    """The mask of `table`'s rows of `dataset` whose `role` party is one of `refs`, or `None`
    when `table` holds no row of `dataset`."""
    match = pc.is_in(_text(table.column(role)), value_set=pa.array(refs, pa.string()))
    if dataset is not EVERY_PROVENANCE and PROVENANCE_COLUMN in table.column_names:
        # A `None` dataset matches the null cells, and a named one no null cell.
        in_dataset = pc.is_in(
            _text(table.column(PROVENANCE_COLUMN)), value_set=pa.array([dataset], pa.string())
        )
        match = pc.and_(match, in_dataset)
    elif dataset is not None and dataset is not EVERY_PROVENANCE:
        return None
    # Two column operands give a column; the stubs widen it to a scalar too.
    return cast("pa.ChunkedArray", match)


def _join_keys(columns: Sequence[pa.Array | pa.ChunkedArray]) -> pa.Table:
    """Join keys over `columns`, each text: each a column marking its nulls and the column as
    `pa.string()` with its nulls as `""`, since a join matches no null key."""
    keys: dict[str, object] = {}
    for at, column in enumerate(columns):
        # A dictionary marks a null cell by its value, which `is_null` reads only once cast.
        text = _text(column)
        keys[f"null_{at}"] = pc.is_null(text)
        keys[f"text_{at}"] = pc.fill_null(text, pa.scalar(""))
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
