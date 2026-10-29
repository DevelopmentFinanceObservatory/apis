"""The latest period of a recipe's root frames: the period every rolling
op-spec of the recipe counts from.

`latest_period` takes the populations the rolling op-specs carry (one list,
`op_spec.Populations`) and the root frames their periods are taken over
(`rolling.rolling_roots`). It narrows each root frame by each population of its
dataset with the `filter` engine (a binding as a `category_match`, a level as a
`place_level`), and reads the narrowed rows as series: one for each distinct
combination of every column but `time_period`, `value` and `status`, mapped to
the periods it has a row at. `dfo_frame_grammar.latest_anchor` then gives the
newest period at least half of the series of every population hold, and the
series that lack it.

A pairing population (`op_spec.PairingPopulation`) stands for the series a
keyed verb pairs key by key, so half of each side holding a period can be two
different halves. It is counted as pseudo-series: one for each key value both
sides hold, whose periods are those at which, on each side, at least half of
that key's series hold the period. A side of several flat populations, or of
one population over several root frames, holds a key that each of them holds,
at the periods each holds it. A pairing on one side of another is read by the
outer keys too: an outer key that is an inner key under its left name on the
inner left side and under the column it pairs with on the inner right, and any
other outer key on each inner side that carries it, matched where both do.
`latest_anchor` then takes the pseudo-series of each pairing as one population
beside the flat ones, and `left_out` names each key that lacks the period on
one side or the other, its coordinates the pairing's left key columns.

A population stated several times is read once: each distinct flat population
narrows and groups each root frame of its dataset once, and each distinct
pairing is paired once for each set of columns it is read by. More than
`MAX_DISTINCT_POPULATIONS` distinct populations, flat and pairing, refuse before
any frame is read.

A population names its dataset and no frame, so it is counted over every root
frame of its `spec_id`. Every population needs at least one such frame and every
root frame at least one such population. Every refusal is
`rolling_anchor_unresolved`, its `fix["detail"]` naming the check, except
periods other than calendar years, which refuse `rolling_grain_unsupported`.

The series are those dfo-apis' assembler groups a frame's rows into, with the
same three columns left out.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pyarrow as pa
from dfo_frame_grammar import (
    ANCHOR_RULE,
    ANCHOR_SHARE,
    Anchor,
    RollingAnchorUnresolved,
    RollingGrainUnsupported,
    latest_anchor,
)
from dfo_frame_grammar.grammar import (
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    ComposedUnit,
    UnitCell,
    UnitRequest,
)
from dfo_spec_kit.identity import canonicalize

from dfo_analysis_engine.calls import Input
from dfo_analysis_engine.engine.columnar import group_rows
from dfo_analysis_engine.engine.common import Units
from dfo_analysis_engine.engine.filter import (
    FilterPlaceLevelUnknownError,
    FilterPredicateTypeMismatchError,
    FilterPredicateValueTypeMismatchError,
    FilterUnitConversionUnsupportedError,
    FilterUnknownColumnError,
    run_filter,
)
from dfo_analysis_engine.op_spec import (
    CategoryMatchPredicate,
    PairingPopulation,
    PlaceLevelPredicate,
    Population,
    RootPopulation,
)
from dfo_analysis_engine.refusal import AnalysisRefused, Refusal
from dfo_analysis_engine.rolling import NON_SERIES_COLUMNS
from dfo_analysis_engine.time_grain import is_interval, time_grain

__all__ = ["MAX_DISTINCT_POPULATIONS", "LeftOutSeries", "ResolvedPeriods", "latest_period"]

# How many periods a `rolling_grain_unsupported` refusal names.
_NAMED_PERIODS = 3

# The most distinct populations, flat and pairing, one recipe's latest period
# is taken over. Each is one reading of the root frames of its dataset.
MAX_DISTINCT_POPULATIONS = 64


@dataclass(frozen=True, kw_only=True)
class LeftOutSeries:
    """A series counted towards the latest period that holds no value at it:
    `root` indexes the populations, and `coordinates` are the series' value in
    each of its columns."""

    root: int
    coordinates: Mapping[str, str]


@dataclass(frozen=True, kw_only=True)
class ResolvedPeriods:
    """The latest period a recipe's rolling op-specs count from, the rule that
    chose it (`dfo_frame_grammar.ANCHOR_RULE`), and the series it leaves out."""

    period: str
    rule: str
    left_out: tuple[LeftOutSeries, ...]

    def as_json(self) -> dict[str, Any]:
        """`{period, rule, left_out}`, the recipe's `anchor` member."""
        return {
            "period": self.period,
            "rule": self.rule,
            "left_out": [
                {"root": series.root, "coordinates": dict(series.coordinates)}
                for series in self.left_out
            ],
        }


@dataclass(frozen=True, kw_only=True)
class _Root:
    """One root frame the latest period is taken over, by its dataset's `spec_id`."""

    spec_id: str
    frame: Input


def latest_period(
    populations: Sequence[Population], roots: Sequence[tuple[str, Input]]
) -> ResolvedPeriods:
    """The latest period of `roots` over `populations`. See the module docstring.

    Args:
        populations: The canonical populations every rolling op-spec carries.
        roots: Each distinct root frame the rolling periods are taken over, by
            its dataset's `spec_id`.

    Raises:
        AnalysisRefused: `rolling_anchor_unresolved` or `rolling_grain_unsupported`.
    """
    frames = [_Root(spec_id=spec_id, frame=frame) for spec_id, frame in roots]
    _too_many(populations)
    _unmatched(populations, frames)
    reading = _Reading(frames)
    counted, owners, coordinates = _counted(populations, reading)
    anchor = _latest(counted, read=reading.held)
    return ResolvedPeriods(
        period=anchor.period,
        rule=ANCHOR_RULE,
        left_out=tuple(
            LeftOutSeries(root=owners[position], coordinates=coordinates[key])
            for position, key in anchor.left_out
        ),
    )


def _unresolved(detail: str, reason: str, **context: object) -> AnalysisRefused:
    return AnalysisRefused(
        Refusal.rolling_anchor_unresolved(detail=detail, reason=reason, context=context)
    )


def _flat(populations: Iterable[Population]) -> Iterator[RootPopulation]:
    """Every flat population of `populations`, those on each side of a pairing included."""
    for population in populations:
        if isinstance(population, PairingPopulation):
            for side in population.pair:
                yield from _flat(side)
        else:
            yield population


def _content(population: Population) -> str:
    """`population`'s identity: its JSON text, one for each content since a
    population is held in its one spelling."""
    return population.model_dump_json()


def _too_many(populations: Sequence[Population]) -> None:
    """Refuse populations of more than `MAX_DISTINCT_POPULATIONS` distinct
    entries. A repeated entry is walked once.

    Raises:
        AnalysisRefused: `rolling_anchor_unresolved`, detail `populations_too_many`.
    """
    seen: set[str] = set()
    pending: list[Population] = list(populations)
    while pending:
        population = pending.pop()
        if (key := _content(population)) in seen:
            continue
        seen.add(key)
        if len(seen) > MAX_DISTINCT_POPULATIONS:
            raise _unresolved(
                "populations_too_many",
                f"the populations hold more than {MAX_DISTINCT_POPULATIONS} distinct entries",
                ceiling=MAX_DISTINCT_POPULATIONS,
            )
        if isinstance(population, PairingPopulation):
            for side in population.pair:
                pending.extend(side)
    return None


def _unmatched(populations: Sequence[Population], roots: Sequence[_Root]) -> None:
    """Refuse a flat population or a root frame with no counterpart.

    Raises:
        AnalysisRefused: `rolling_anchor_unresolved`, detail `population_without_root`
            or `root_without_population`.
    """
    flat = list(_flat(populations))
    for population in flat:
        if all(root.spec_id != population.spec_id for root in roots):
            raise _unresolved(
                "population_without_root",
                f"no root frame was given for the population over {population.spec_id}",
                spec_id=population.spec_id,
            )
    for root in roots:
        if all(root.spec_id != population.spec_id for population in flat):
            raise _unresolved(
                "root_without_population",
                f"the root frame {root.spec_id} matches the spec_id of no population",
                spec_id=root.spec_id,
            )


def _narrowed(population: RootPopulation, admitted: Input) -> pa.Table:
    """The rows of `admitted` that `population`'s bindings keep."""
    frame = admitted.frame
    narrowing = {*population.bindings, *population.levels}
    unknown = sorted(column for column in narrowing if column not in frame.columns)
    if unknown:
        raise _unresolved(
            "binding_column_unknown",
            f"the root frame of {population.spec_id} has no column named {unknown[0]!r}",
            spec_id=population.spec_id,
            columns=unknown,
            available=list(frame.columns),
        )
    if not narrowing:
        return frame.table
    if off_series := sorted(set(population.bindings) & NON_SERIES_COLUMNS):
        raise _unresolved(
            "binding_unusable",
            f"{off_series[0]!r} is no part of a series, so no population binds it",
            spec_id=population.spec_id,
            columns=off_series,
        )
    binding_predicates = [
        CategoryMatchPredicate(column=column, values=values)
        for column, values in population.bindings.items()
    ]
    level_predicates = [
        PlaceLevelPredicate(column=column, level=level, strict=column in population.strict_levels)
        for column, level in population.levels.items()
    ]
    try:
        narrowed = frame
        if binding_predicates:
            narrowed = run_filter(narrowed, predicates=binding_predicates, span=admitted.span).frame
        if level_predicates:
            narrowed = run_filter(narrowed, predicates=level_predicates, span=admitted.span).frame
        return narrowed.table
    except (
        FilterUnknownColumnError,
        FilterPredicateTypeMismatchError,
        FilterPredicateValueTypeMismatchError,
        FilterPlaceLevelUnknownError,
        FilterUnitConversionUnsupportedError,
    ) as exc:
        raise _unresolved(
            "binding_unusable",
            f"a binding of the population over {population.spec_id} cannot narrow its "
            f"root frame: {exc}",
            spec_id=population.spec_id,
            columns=sorted(narrowing),
        ) from exc


def coordinate_text(cell: str | UnitCell | None) -> str | None:
    """One cell of a series' coordinates as text, in the form `apis`' assembler
    groups it under: a converted unit as `currency/prices[/base_year]`, a composed
    unit as the RFC 8785 text of its JSON object, and any other cell as itself."""
    if isinstance(cell, UnitRequest):
        text = f"{cell.currency}/{cell.prices}"
        return text if cell.base_year is None else f"{text}/{cell.base_year}"
    if isinstance(cell, ComposedUnit):
        return canonicalize(cell.as_json()).decode()
    return cell


def _series(table: pa.Table) -> Iterator[tuple[dict[str, str], set[str]]]:
    """Each series of `table`, in order of its first row: its coordinates, a
    column it holds no value in left out, and the periods it has a row at."""
    dimensions = [column for column in table.column_names if column not in NON_SERIES_COLUMNS]
    groups = group_rows(table, dimensions)
    periods = cast("list[str]", table.column(TIME_PERIOD_COLUMN).to_pylist())
    indices = cast("list[int]", groups.indices.to_pylist())
    plain = [column for column in dimensions if column != UNIT_COLUMN]
    keys = cast("list[dict[str, str | None]]", groups.keys.select(plain).to_pylist())
    units = Units.of(groups.keys) if UNIT_COLUMN in dimensions else None
    unit_codes = cast("list[int]", units.codes.to_pylist()) if units is not None else []
    for group in range(groups.count):
        cells: dict[str, str | UnitCell | None] = dict(keys[group]) if plain else {}
        if units is not None:
            cells[UNIT_COLUMN] = units.cells[unit_codes[group]]
        coordinates = {
            column: text
            for column in dimensions
            if (text := coordinate_text(cells[column])) is not None
        }
        rows = indices[groups.offsets[group] : groups.offsets[group + 1]]
        yield coordinates, {periods[row] for row in rows}


# The series a population is counted over, keyed by their place in the call,
# with the periods each holds: one `latest_anchor` population.
type _Counted = dict[tuple[int, int], set[str]]
# The series of one side of a pairing that one flat population over one root
# frame gives, or a nested pairing's pseudo-series, by their values in the
# pairing's key columns, each series as the periods it holds.
type _Grouped = dict[tuple[str, ...], list[set[str]]]


def _half_held(series: Sequence[set[str]]) -> set[str]:
    """The periods at least `ANCHOR_SHARE` of `series` hold."""
    held = Counter(period for periods in series for period in periods)
    return {period for period, count in held.items() if count >= ANCHOR_SHARE * len(series)}


def _held_by_key(units: Sequence[_Grouped]) -> dict[tuple[str, ...], set[str]]:
    """Each key every one of a side's `units` holds, with the periods at which
    at least half of each unit's series under it hold them."""
    keys = set.intersection(*(set(unit) for unit in units))
    return {key: set.intersection(*(_half_held(unit[key]) for unit in units)) for key in keys}


type _Frames = list[tuple[frozenset[str], list[tuple[dict[str, str], set[str]]]]]
type _Pseudo = tuple[tuple[str, ...], dict[tuple[str, ...], set[str]]]


class _Reading:
    """The series `resolve_anchor` counts, read from the admitted root frames.
    `held` gathers every period read, which must all be calendar years. Each
    reading is kept by the content of its population, so a population stated
    on several branches is read once; the kept values are shared, never changed."""

    def __init__(
        self,
        roots: Sequence[_Root],
    ) -> None:
        self.roots = roots
        self.held: set[str] = set()
        self._frames: dict[str, _Frames] = {}
        self._carried: dict[str, frozenset[str]] = {}
        self._pseudo: dict[tuple[str, tuple[str, ...]], _Pseudo] = {}

    def frames(self, population: RootPopulation) -> _Frames:
        """The series `population` keeps of each root frame of its dataset, in
        the call's root order, beside that frame's column names. Read once."""
        key = _content(population)
        if (read := self._frames.get(key)) is not None:
            return read
        found: _Frames = []
        for root in self.roots:
            if root.spec_id != population.spec_id:
                continue
            table = _narrowed(population, root.frame)
            series = list(_series(table))
            self.held.update(period for _, periods in series for period in periods)
            found.append((frozenset(table.column_names), series))
        self._frames[key] = found
        return found

    def carried(self, populations: Sequence[Population]) -> frozenset[str]:
        """The columns every unit of one side carries: each root frame of a flat
        population, and a pairing's frame, which holds each key under its left
        name and every other column a side of it carries."""
        found: frozenset[str] | None = None
        for population in populations:
            if isinstance(population, PairingPopulation):
                key = _content(population)
                if (pairing_columns := self._carried.get(key)) is None:
                    left = self.carried(population.pair[0])
                    right = self.carried(population.pair[1])
                    left_name = {paired: name for name, paired in population.keys.items()}
                    pairing_columns = left | {left_name.get(column, column) for column in right}
                    self._carried[key] = pairing_columns
                units = [pairing_columns]
            else:
                frames = self.frames(population)
                units = [columns for columns, _ in frames]
            for columns in units:
                found = columns if found is None else found & columns
        return found or frozenset()

    def grouped(
        self, populations: Sequence[Population], columns: tuple[str, ...]
    ) -> list[_Grouped]:
        """One `_Grouped` for each flat population of one side over each root
        frame of its dataset, and one for each pairing on it, by `columns`. A
        series with no value in one of `columns` pairs with nothing."""
        units: list[_Grouped] = []
        for population in populations:
            if isinstance(population, PairingPopulation):
                pseudo = self.pseudo_series(population, columns)
                wanted, by_key = pseudo
                at = [wanted.index(column) for column in columns]
                unit: _Grouped = {}
                for key, periods in by_key.items():
                    unit.setdefault(tuple(key[i] for i in at), []).append(periods)
                units.append(unit)
                continue
            frames = self.frames(population)
            for frame_columns, series in frames:
                if missing := sorted(set(columns) - frame_columns):
                    raise _unresolved(
                        "pairing_key_column_unknown",
                        f"the root frame of {population.spec_id} has no key column "
                        f"named {missing[0]!r}",
                        spec_id=population.spec_id,
                        columns=missing,
                        available=sorted(frame_columns),
                    )
                unit = {}
                for coordinates, periods in series:
                    if all(column in coordinates for column in columns):
                        key = tuple(coordinates[column] for column in columns)
                        unit.setdefault(key, []).append(periods)
                units.append(unit)
        return units

    def pseudo_series(self, pairing: PairingPopulation, columns: tuple[str, ...] = ()) -> _Pseudo:
        """The pseudo-series of `pairing`: one for each key both sides hold,
        over the pairing's left key columns and then each of `columns` it does
        not key on, with the periods at which at least half of that key's
        series hold them on each side. Returned beside those column names.
        Paired once for each `columns`.

        The pairing's frame keeps the left name of each key and any other column
        of either side under its own name, so such a column is read on each side
        that carries it, and matched where both do.
        """
        key = (_content(pairing), columns)
        if (read := self._pseudo.get(key)) is not None:
            return read
        pseudo = self._paired(pairing, columns)
        self._pseudo[key] = pseudo
        return pseudo

    def _paired(self, pairing: PairingPopulation, columns: tuple[str, ...]) -> _Pseudo:
        """`pseudo_series`, computed."""
        keys = tuple(pairing.keys)
        extra = tuple(column for column in columns if column not in pairing.keys)
        left_side, right_side = pairing.pair
        left_carried = self.carried(left_side)
        right_carried = self.carried(right_side)
        right_carried -= set(pairing.keys.values())
        if missing := [c for c in extra if c not in left_carried | right_carried]:
            raise _unresolved(
                "pairing_key_column_unknown",
                f"neither side of the pairing on {list(keys)!r} has a key column named "
                f"{missing[0]!r}",
                columns=missing,
            )
        on_left = tuple(column for column in extra if column in left_carried)
        on_right = tuple(column for column in extra if column in right_carried)
        left = self.grouped(left_side, (*keys, *on_left))
        right = self.grouped(right_side, (*pairing.keys.values(), *on_right))
        shared = [column for column in on_left if column in on_right]

        def matched(key: tuple[str, ...], side: tuple[str, ...]) -> tuple[str, ...]:
            """`key`'s pairing keys and its values in the columns both sides carry."""
            values = dict(zip(side, key[len(keys) :], strict=True))
            return (*key[: len(keys)], *(values[column] for column in shared))

        by_match: dict[tuple[str, ...], list[tuple[tuple[str, ...], set[str]]]] = {}
        for key, periods in _held_by_key(right).items():
            by_match.setdefault(matched(key, on_right), []).append((key, periods))
        pseudo: dict[tuple[str, ...], set[str]] = {}
        for key, periods in sorted(_held_by_key(left).items()):
            for other, other_periods in by_match.get(matched(key, on_left), ()):
                values = dict(zip(on_left, key[len(keys) :], strict=True)) | dict(
                    zip(on_right, other[len(keys) :], strict=True)
                )
                pseudo[(*key[: len(keys)], *(values[column] for column in extra))] = (
                    periods & other_periods
                )
        return (*keys, *extra), pseudo


def _counted(
    populations: Sequence[Population], reading: _Reading
) -> tuple[list[_Counted], list[int], dict[tuple[int, int], dict[str, str]]]:
    """The `latest_anchor` populations of `populations`, the index in
    populations each stands for, and every series' coordinates by its key.

    A flat population gives one for each root frame of its dataset, so two
    series with one set of coordinates, in two frames, stay apart. A pairing
    gives one of its pseudo-series, each with its key as its coordinates.
    """
    counted: list[_Counted] = []
    owners: list[int] = []
    coordinates: dict[tuple[int, int], dict[str, str]] = {}

    def count(index: int, series: Iterable[tuple[dict[str, str], set[str]]]) -> None:
        """Add one `latest_anchor` population of `series`, standing for `index`."""
        position = len(counted)
        counted.append({})
        owners.append(index)
        for number, (cells, periods) in enumerate(series):
            counted[position][position, number] = periods
            coordinates[position, number] = cells

    for index, population in enumerate(populations):
        if isinstance(population, PairingPopulation):
            pseudo = reading.pseudo_series(population)
            wanted, by_key = pseudo
            if not by_key:
                raise _unresolved(
                    "pairing_shares_no_key",
                    f"the two sides of the pairing on {list(population.keys)!r} name no "
                    "entity in common",
                    keys=dict(population.keys),
                )
            count(
                index,
                ((dict(zip(wanted, key, strict=True)), periods) for key, periods in by_key.items()),
            )
            continue
        frames = reading.frames(population)
        for _, series in frames:
            count(index, series)
    return counted, owners, coordinates


def _latest(counted: Sequence[_Counted], *, read: set[str]) -> Anchor[tuple[int, int]]:
    """`latest_anchor` over `counted`. Every period `read` must be a calendar year."""
    held = sorted(read)
    off_grain = [period for period in held if is_interval(period) or time_grain(period) != "year"]
    if off_grain:
        raise AnalysisRefused(
            Refusal.rolling_grain_unsupported(
                reason="a root frame holds periods that are no calendar year",
                periods=off_grain[:_NAMED_PERIODS],
            )
        )
    try:
        return latest_anchor(counted)
    except RollingAnchorUnresolved as exc:
        raise _unresolved("no_qualifying_period", str(exc)) from exc
    except RollingGrainUnsupported as exc:
        raise AnalysisRefused(
            Refusal.rolling_grain_unsupported(reason=str(exc), periods=held[:_NAMED_PERIODS])
        ) from exc
