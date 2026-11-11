"""Content facts for materialised frames, before reshape or output-option decisions.

`compute_output_profile` reports column distinct counts, pivot feasibility,
and enrichment coverage. `datasets_api.output_options` combines this profile
with the capability document to decide availability.

`compute_output_profile_from_chunks` produces the same exact profile with two
passes when duplicate pivot keys need per-column variation. Both passes run on
Arrow: each chunk's cells become dense integer codes shared across chunks, and
grouping runs over those codes. State follows distinct values and pivot keys.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, cast

import pyarrow as pa
import pyarrow.compute as pc
from dfo_frame_grammar import Frame

from code_index.types import PartyRow
from dfo_platform.columns import PROVENANCE_COLUMN
from materialise.enrichment import (
    ENRICHMENT_FIELDS,
    PARTY_KIND_FIELD,
    EnrichmentLookup,
    pivot_party_cell,
    row_party_cell,
)
from materialise.errors import MaterialisationUsageError
from materialise.reshape import (
    SUPPORTED_ENRICHMENTS,
    ColumnClassification,
    TableLike,
    classify_columns,
    enrichment_qualifies,
)

_SCHEMA_VERSION = 1

_VALUE_COLUMN = "value"
_STATUS_COLUMN = "status"
_UNIT_COLUMN = "unit"

_MAX_NAMED_MISSING_REFS = 10
"""Cap on how many refs an `EnrichmentCoverage.missing` list names, the same
capped-list convention `materialise.facets._MAX_ALTERNATIVES_PER_ISSUE` and
`reshape._MAX_NAMED_DUPLICATE_KEYS` use."""


@dataclass(frozen=True)
class ColumnProfile:
    """One frame column's shape. `kind` is one of `"grain"`, `"role"`,
    `"value"`, `"unit"`, `"facet"`, or `"status"`. `droppable` comes from
    `reshape.classify_columns`. `distinct` is the column's frame-wide
    distinct value count, `None` for `value` and `status`. `value` varies
    by construction, and `status` carries no pivot or enrichment meaning.
    """

    name: str
    kind: str
    droppable: bool
    distinct: int | None


@dataclass(frozen=True)
class EnrichmentCoverage:
    """Coverage of one `(column, field)` enrichment pair over the frame's
    distinct refs in `column`. `entities` is the distinct ref count.
    `covered` is how many of those refs have `field`: for `party_kind`, a kind
    from the code index on the column's axis wherever every layout fills it, a
    long download in each row's own dataset and a pivot over the datasets the
    frame serves the ref from; the graph lookup for every other field.
    `missing` lists the refs without `field`, each paired with its label
    from the same lookup call, empty when the graph carries no label for
    that ref either. It is capped at `_MAX_NAMED_MISSING_REFS`, with
    `truncated` set when the cap cut the list.
    """

    column: str
    field: str
    entities: int
    covered: int
    missing: tuple[tuple[str, str], ...]
    truncated: bool


@dataclass(frozen=True)
class PivotFeasibility:
    """Pivot facts for one `(index, columns)` pair of grain or role columns.
    `duplicate_keys` flags repeated key pairs; `unique` also fails when
    distinct column values stringify to the same output name. `homogeneous`
    requires dropped non-droppable columns to be constant within each index.
    `width` is one plus the distinct pivot-value count; `cells` is width times
    the distinct index count. `varying` lists other columns, excluding
    `value` and `status`, that differ inside duplicate-key groups, with their
    frame-wide distinct counts.
    """

    index: str
    columns: str
    duplicate_keys: bool
    homogeneous: bool
    unique: bool
    width: int
    cells: int
    varying: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class OutputProfile:
    """Content facts about one materialised frame. `pivots` covers every
    ordered pair of grain and role columns whose distinct count exceeds
    one. A column needs more than one distinct value to be a pivot
    candidate, since pivoting on a constant produces nothing useful.
    `enrichments` carries one entry per `(role column, supported field)`
    pair with `covered > 0`. A pair with zero coverage is omitted. A
    zero-row frame reports its columns with `rows: 0` and empty `pivots`
    and `enrichments`.
    """

    schema_version: int
    rows: int
    columns: tuple[ColumnProfile, ...]
    pivots: tuple[PivotFeasibility, ...]
    enrichments: tuple[EnrichmentCoverage, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, body: Mapping[str, Any]) -> OutputProfile:
        return cls(
            schema_version=body["schema_version"],
            rows=body["rows"],
            columns=tuple(
                ColumnProfile(
                    name=c["name"], kind=c["kind"], droppable=c["droppable"], distinct=c["distinct"]
                )
                for c in body["columns"]
            ),
            pivots=tuple(_pivot_feasibility_from_json(p) for p in body["pivots"]),
            enrichments=tuple(_enrichment_coverage_from_json(e) for e in body["enrichments"]),
        )


def _pivot_feasibility_from_json(body: Mapping[str, Any]) -> PivotFeasibility:
    return PivotFeasibility(
        index=body["index"],
        columns=body["columns"],
        duplicate_keys=body["duplicate_keys"],
        homogeneous=body["homogeneous"],
        unique=body["unique"],
        width=body["width"],
        cells=body["cells"],
        varying=tuple((entry[0], entry[1]) for entry in body["varying"]),
    )


def _enrichment_coverage_from_json(body: Mapping[str, Any]) -> EnrichmentCoverage:
    return EnrichmentCoverage(
        column=body["column"],
        field=body["field"],
        entities=body["entities"],
        covered=body["covered"],
        missing=tuple((entry[0], entry[1]) for entry in body["missing"]),
        truncated=body["truncated"],
    )


def compute_output_profile(
    frame: TableLike,
    *,
    role_columns: Sequence[str],
    enrichment: EnrichmentLookup,
    parties: Mapping[str, PartyRow],
) -> OutputProfile:
    """Wrap `frame` as a one-chunk stream and delegate to the chunked implementation."""
    return compute_output_profile_from_chunks(
        lambda: iter((frame,)), role_columns=role_columns, enrichment=enrichment, parties=parties
    )


def compute_output_profile_from_chunks(
    frame_chunks: Callable[[], Iterator[TableLike]],
    *,
    role_columns: Sequence[str],
    enrichment: EnrichmentLookup,
    parties: Mapping[str, PartyRow],
) -> OutputProfile:
    """Compute the exact profile from a fresh chunk iterator on each pass.
    Pass one gathers distinct counts, functional dependencies, and duplicate
    keys (`OutputProfileFold.add`); only duplicate keys trigger pass two, which
    finds columns that vary within those groups (`OutputProfileFold.finish`).
    A column stops being tracked for a dependency once a chunk shows it varying.

    State is bounded by distinct column values, per-chunk group summaries,
    and the distinct pivot keys of a multi-chunk frame. Cells compare by Python equality, as `Frame.rows` and
    `Table.rows` present them, so a `unit` struct compares as its unit cell.

    `role_columns` must exist in the first chunk, and every chunk must share
    its column order. The iterator must yield at least one chunk, including
    for an empty result; pivot pairs follow that shared column order.

    `enrichment` reads the role columns' graph fields and `parties`, the code index's party
    rows (`CodeIndex.parties`), their party kinds.
    """
    fold = OutputProfileFold(role_columns=role_columns, parties=parties)
    for chunk in frame_chunks():
        fold.add(chunk)
    return fold.finish(frame_chunks, enrichment=enrichment)


class OutputProfileFold:
    """The first pass of `compute_output_profile_from_chunks`, fed one chunk at a time
    (`add`), so a caller already passing over the chunks folds the profile in that pass.
    `finish` runs the second pass, when one is needed, over `frame_chunks`, which yields the
    chunks `add` was given in the same order. `party_kind_coverage` counts the `party_kind`
    coverage alone, which reads no graph; a fold made by `of_party_kinds` keeps only what it
    reads, the role columns' refs and their datasets, and has no `finish`."""

    def __init__(self, *, role_columns: Sequence[str], parties: Mapping[str, PartyRow]) -> None:
        self._role_columns = tuple(role_columns)
        self._parties = parties
        self._pivots = True
        # A party's kind depends on the dataset only where its row names dataset kinds.
        self._kinds_vary = any(party.dataset_kinds for party in parties.values())
        self._columns: tuple[str, ...] | None = None
        self._total_rows = 0

    def _start(self, first_chunk: TableLike) -> None:
        role_columns = self._role_columns
        columns = first_chunk.columns
        unknown_roles = [name for name in role_columns if name not in columns]
        if unknown_roles:
            raise MaterialisationUsageError(
                "compute_output_profile() requires every role_columns entry to be present "
                f"on the frame; frame columns are {columns!r}, got unknown "
                f"role column(s) {unknown_roles!r}"
            )

        self._columns = tuple(columns)
        self._classification = classify_columns(columns)
        self._role_column_set = frozenset(role_columns)
        self._distinct_columns = tuple(
            name
            for name in columns
            if name not in (_VALUE_COLUMN, _STATUS_COLUMN)
            and (self._pivots or name in self._role_column_set or name == PROVENANCE_COLUMN)
        )
        self._books: dict[str, dict[object, int]] = {name: {} for name in self._distinct_columns}
        # Coded tables name columns by position, so aggregate outputs cannot clash.
        self._alias = {name: f"c{i}" for i, name in enumerate(self._distinct_columns)}
        # The distinct (ref, provenance) codes of each role column, which `party_kind`
        # coverage reads where a party's kind can depend on its dataset.
        self._role_datasets: dict[str, set[tuple[int, int]]] | None = (
            {name: set() for name in role_columns}
            if self._kinds_vary and PROVENANCE_COLUMN in self._distinct_columns
            else None
        )

        # Candidate sources come from column kinds before distinct counts are known;
        # pivot output uses candidates with multiple distinct values.
        self._candidate_sources = tuple(
            name
            for name in columns
            if self._pivots
            and _column_kind(
                name, role_columns=self._role_column_set, classification=self._classification
            )
            in ("grain", "role")
        )
        alias = self._alias
        self._dependency_trackers = {
            a: _ConstantWithinGroups(
                keys=(alias[a],),
                targets=tuple(alias[c] for c in self._distinct_columns if c != a),
            )
            for a in self._candidate_sources
        }
        candidate_sources = self._candidate_sources
        pair_names: tuple[tuple[str, str], ...] = tuple(
            (a, b) for i, a in enumerate(candidate_sources) for b in candidate_sources[i + 1 :]
        )
        self._repeat_trackers = {
            (a, b): _KeyRepeats(keys=(alias[a], alias[b])) for a, b in pair_names
        }

    def add(self, chunk: TableLike) -> None:
        """Fold one chunk into the first pass. The first chunk sets the columns."""
        if self._columns is None:
            self._start(chunk)
        coded = _coded_chunk(
            chunk, names=self._distinct_columns, books=self._books, alias=self._alias
        )
        self._total_rows += coded.num_rows
        if self._role_datasets is not None:
            provenance = self._alias[PROVENANCE_COLUMN]
            for name, pairs in self._role_datasets.items():
                grouped = coded.group_by([self._alias[name], provenance]).aggregate([])
                # Codes are never null: every cell, a null one included, has a code.
                pairs.update(
                    zip(
                        cast("list[int]", grouped.column(self._alias[name]).to_pylist()),
                        cast("list[int]", grouped.column(provenance).to_pylist()),
                        strict=True,
                    )
                )
        for tracker in self._dependency_trackers.values():
            tracker.add(coded)
        for repeats in self._repeat_trackers.values():
            repeats.add(coded)

    def finish(
        self, frame_chunks: Callable[[], Iterator[TableLike]], *, enrichment: EnrichmentLookup
    ) -> OutputProfile:
        """The profile of the chunks `add` was given, its role columns' graph fields read
        through `enrichment`. Raises `MaterialisationUsageError` when it was given none or the
        fold counts party kinds alone (`of_party_kinds`)."""
        if not self._pivots:
            raise MaterialisationUsageError(
                "OutputProfileFold.finish() needs a fold made for a whole profile, not by "
                "of_party_kinds()"
            )
        self._check_started("OutputProfileFold.finish()")
        assert self._columns is not None
        columns = self._columns
        role_columns = self._role_columns
        classification = self._classification
        role_column_set = self._role_column_set
        distinct_columns = self._distinct_columns
        books = self._books
        alias = self._alias
        candidate_sources = self._candidate_sources
        total_rows = self._total_rows

        dependencies: dict[tuple[str, str], bool] = {}
        for a, tracker in self._dependency_trackers.items():
            varied = tracker.finish()
            for c in distinct_columns:
                if c != a:
                    dependencies[(a, c)] = alias[c] not in varied
        dup_found = {pair: repeats.finish() for pair, repeats in self._repeat_trackers.items()}

        column_profiles = tuple(
            ColumnProfile(
                name=name,
                kind=_column_kind(
                    name, role_columns=role_column_set, classification=classification
                ),
                droppable=name not in classification.non_droppable_columns,
                distinct=None if name in (_VALUE_COLUMN, _STATUS_COLUMN) else len(books[name]),
            )
            for name in columns
        )

        grain_columns = tuple(name for name in candidate_sources if len(books[name]) > 1)

        def _canonical_dup_pair(index: str, other: str) -> tuple[str, str]:
            return (index, other) if (index, other) in dup_found else (other, index)

        # The candidate set is symmetric in the pair, so one scan serves both orders.
        varying_candidates_by_pair: dict[tuple[str, str], tuple[str, ...]] = {}
        for index in grain_columns:
            for other in grain_columns:
                pair = _canonical_dup_pair(index, other)
                if other == index or pair in varying_candidates_by_pair or not dup_found[pair]:
                    continue
                candidates = tuple(
                    c
                    for c in columns
                    if c not in (index, other, _VALUE_COLUMN, _STATUS_COLUMN)
                    and len(books[c]) > 1
                    and not dependencies[(index, c)]
                    and not dependencies[(other, c)]
                )
                if candidates:
                    varying_candidates_by_pair[pair] = candidates

        varying_by_pair = (
            _second_pass_varying(
                frame_chunks,
                books=books,
                alias=alias,
                candidates_by_pair=varying_candidates_by_pair,
            )
            if varying_candidates_by_pair
            else {}
        )

        pivots = tuple(
            _pivot_feasibility(
                index=a,
                columns=b,
                distinct_by_column=books,
                dependencies=dependencies,
                non_droppable_columns=classification.non_droppable_columns,
                duplicate_keys=dup_found[_canonical_dup_pair(a, b)],
                varying_columns=varying_by_pair.get(_canonical_dup_pair(a, b), frozenset()),
            )
            for a in grain_columns
            for b in grain_columns
            if a != b
        )

        role_refs = sorted(
            {
                ref
                for role_column in role_columns
                for ref in books[role_column]
                if isinstance(ref, str)
            }
        )
        resolved = enrichment.lookup(role_refs) if role_refs else {}
        enrichments = tuple(
            coverage
            for role_column in role_columns
            for coverage in self._coverage(role_column, resolved, ENRICHMENT_FIELDS)
        )

        return OutputProfile(
            schema_version=_SCHEMA_VERSION,
            rows=total_rows,
            columns=column_profiles,
            pivots=pivots,
            enrichments=enrichments,
        )

    def party_kind_coverage(
        self, labels: Mapping[str, str]
    ) -> dict[str, EnrichmentCoverage | None]:
        """Each role column's `party_kind` coverage of the chunks `add` was given, as
        `finish` counts it, `None` for a column whose parties take no kind. A missing ref is
        named by its label in `labels`, else by none. Raises `MaterialisationUsageError` when
        `add` was given no chunk."""
        self._check_started("OutputProfileFold.party_kind_coverage()")
        resolved = {ref: {"label": label} for ref, label in labels.items()}
        return {
            role_column: next(
                iter(self._coverage(role_column, resolved, (PARTY_KIND_FIELD,))), None
            )
            for role_column in self._role_columns
        }

    @classmethod
    def of_party_kinds(
        cls, *, role_columns: Sequence[str], parties: Mapping[str, PartyRow]
    ) -> OutputProfileFold:
        """A fold that counts `party_kind_coverage` alone, keeping each role column's refs
        and the datasets serving them and none of the pivot facts `finish` reads."""
        fold = cls(role_columns=role_columns, parties=parties)
        fold._pivots = False
        return fold

    def _check_started(self, caller: str) -> None:
        if self._columns is None:
            raise MaterialisationUsageError(
                f"{caller} requires the fold to have been given at least one chunk"
            )

    def _coverage(
        self,
        role_column: str,
        resolved: Mapping[str, Mapping[str, str]],
        fields: Sequence[str],
    ) -> tuple[EnrichmentCoverage, ...]:
        return _enrichment_coverage_for_column(
            role_column,
            self._books[role_column],
            resolved,
            fields=fields,
            parties=self._parties,
            datasets=self._datasets_of(role_column),
        )

    def _datasets_of(self, role_column: str) -> Mapping[object, frozenset[object]] | None:
        """The `provenance` cells each ref of `role_column` shares a row with, `None` for a
        frame without a `provenance` column or parties whose kind depends on no dataset."""
        if self._role_datasets is None:
            return None
        refs = {code: ref for ref, code in self._books[role_column].items()}
        provenances = {code: cell for cell, code in self._books[PROVENANCE_COLUMN].items()}
        datasets: dict[object, set[object]] = {}
        for ref_code, provenance_code in self._role_datasets[role_column]:
            datasets.setdefault(refs[ref_code], set()).add(provenances[provenance_code])
        return {ref: frozenset(cells) for ref, cells in datasets.items()}


def _column_kind(
    name: str, *, role_columns: frozenset[str], classification: ColumnClassification
) -> str:
    if name == _VALUE_COLUMN:
        return "value"
    if name == _STATUS_COLUMN:
        return "status"
    if name in classification.facet_axis_columns:
        return "unit" if name == _UNIT_COLUMN else "facet"
    if name in role_columns:
        return "role"
    return "grain"


def _coded_chunk(
    chunk: TableLike,
    *,
    names: Sequence[str],
    books: Mapping[str, dict[object, int]],
    alias: Mapping[str, str],
) -> pa.Table:
    """`chunk`'s `names` columns as int64 codes, each under `alias[name]`.
    `books[name]` maps each distinct cell to its code, so equal cells share a
    code across chunks; a cell seen for the first time is added to its book.
    The table keeps `chunk`'s row count when `names` is empty.
    """
    if isinstance(chunk, Frame):
        arrays = [_frame_column_codes(chunk, name=name, book=books[name]) for name in names]
        row_count = chunk.table.num_rows
    else:
        rows = chunk.rows
        arrays = []
        for name in names:
            position = chunk.columns.index(name)
            book = books[name]
            arrays.append(
                pa.array([book.setdefault(row[position], len(book)) for row in rows], pa.int64())
            )
        row_count = len(rows)
    if not arrays:
        return pa.table({"rows": pa.nulls(row_count)})
    return pa.table(arrays, names=[alias[name] for name in names])


def _frame_column_codes(frame: Frame, *, name: str, book: dict[object, int]) -> pa.Array:
    """Codes a `Frame` column without building its rows. Only the column's
    distinct cells become Python objects, read the way `Frame.rows` reads
    them, so a struct column's cells come from one sample row per distinct value.
    """
    column = frame.table.column(name).combine_chunks()
    if pa.types.is_struct(column.type):
        local, distinct_count = _struct_local_codes(column)
        # `index_in` answers each code's first position in `local`.
        first_rows = pc.index_in(pa.array(range(distinct_count), pa.int64()), value_set=local)
        position = frame.columns.index(name)
        sample = Frame(columns=frame.columns, table=frame.table.take(first_rows)).rows
        cells = [row[position] for row in sample]
    else:
        distinct = pc.unique(column)
        local = pc.index_in(column, value_set=distinct, skip_nulls=False)
        cells = distinct.to_pylist()
    lookup = pa.array([book.setdefault(cell, len(book)) for cell in cells], pa.int64())
    return pc.take(lookup, local)


def _struct_local_codes(column: pa.Array) -> tuple[pa.Array, int]:
    """Dense codes for a struct column, equal for rows with equal leaf values
    and validity, and the number of distinct codes. Leaves combine one at a
    time and re-densify, so a combined code stays below the squared row count.
    """
    codes: pa.Array | None = None
    count = 0
    for leaf in _struct_leaves(column):
        distinct = pc.unique(leaf)
        leaf_codes = pc.cast(pc.index_in(leaf, value_set=distinct, skip_nulls=False), pa.int64())
        if codes is None:
            codes, count = leaf_codes, len(distinct)
            continue
        scaled = cast(pa.Array, pc.multiply(codes, pa.scalar(len(distinct), pa.int64())))
        combined = cast(pa.Array, pc.add(scaled, leaf_codes))
        combined_distinct = pc.unique(combined)
        codes = pc.cast(pc.index_in(combined, value_set=combined_distinct), pa.int64())
        count = len(combined_distinct)
    assert codes is not None
    return codes, count


def _struct_leaves(array: pa.Array) -> Iterator[pa.Array]:
    """The struct's validity, then every non-struct leaf in field order. A
    leaf read through `pc.struct_field` is null wherever a parent is null.
    """
    yield pc.is_valid(array)
    for position in range(array.type.num_fields):
        child = pc.struct_field(array, [position])
        if pa.types.is_struct(child.type):
            yield from _struct_leaves(child)
        else:
            yield child


def _min_max(columns: Sequence[str]) -> list[tuple[str, Literal["min", "max"]]]:
    return [*((c, "min") for c in columns), *((c, "max") for c in columns)]


@dataclass
class _ConstantWithinGroups:
    """Finds the `targets` whose code varies inside some group of rows with
    equal `keys`, over every coded chunk added. Each chunk reduces to its
    groups' minimum and maximum code per target; a target varies when those
    differ in one chunk or, at `finish`, across chunks. A target that varies
    stops being tracked.
    """

    keys: tuple[str, ...]
    targets: tuple[str, ...]
    varied: set[str] = field(default_factory=set)
    _parts: list[pa.Table] = field(default_factory=list)

    def add(self, coded: pa.Table) -> None:
        live = self._live()
        if not live:
            return
        grouped = coded.group_by(list(self.keys)).aggregate(_min_max(live))
        self._record(grouped, live, low="min", high="max")
        self._parts.append(grouped)

    def finish(self) -> set[str]:
        live = self._live()
        if len(self._parts) > 1 and live:
            lows = [f"{t}_min" for t in live]
            highs = [f"{t}_max" for t in live]
            merged = pa.concat_tables(
                [part.select([*self.keys, *lows, *highs]) for part in self._parts]
            )
            aggregations: list[tuple[str, Literal["min", "max"]]] = [
                *((c, "min") for c in lows),
                *((c, "max") for c in highs),
            ]
            grouped = merged.group_by(list(self.keys)).aggregate(aggregations)
            self._record(grouped, live, low="min_min", high="max_max")
        self._parts.clear()
        return self.varied

    def _live(self) -> list[str]:
        return [t for t in self.targets if t not in self.varied]

    def _record(self, grouped: pa.Table, live: Sequence[str], *, low: str, high: str) -> None:
        for target in live:
            if not grouped.column(f"{target}_{low}").equals(grouped.column(f"{target}_{high}")):
                self.varied.add(target)


@dataclass
class _KeyRepeats:
    """Whether a combination of `keys` codes occurs twice over every coded
    chunk added. A chunk with a repeat settles it; otherwise each chunk keeps
    its distinct combinations until `finish` checks them across chunks."""

    keys: tuple[str, ...]
    found: bool = False
    _parts: list[pa.Table] = field(default_factory=list)

    def add(self, coded: pa.Table) -> None:
        if self.found:
            return
        distinct = coded.group_by(list(self.keys)).aggregate([])
        if distinct.num_rows < coded.num_rows:
            self.found = True
            self._parts.clear()
            return
        self._parts.append(distinct)

    def finish(self) -> bool:
        if not self.found and len(self._parts) > 1:
            merged = pa.concat_tables(self._parts)
            self.found = merged.group_by(list(self.keys)).aggregate([]).num_rows < merged.num_rows
        self._parts.clear()
        return self.found


def _second_pass_varying(
    frame_chunks: Callable[[], Iterator[TableLike]],
    *,
    books: Mapping[str, dict[object, int]],
    alias: Mapping[str, str],
    candidates_by_pair: Mapping[tuple[str, str], Sequence[str]],
) -> dict[tuple[str, str], frozenset[str]]:
    """Scan a fresh chunk iterator only for pairs with duplicate keys and find,
    per pair, the candidate columns that differ inside a group of rows sharing
    the pair's key. Pass one filled `books`, so this pass only reads codes.
    """
    trackers = {
        (a, b): _ConstantWithinGroups(
            keys=(alias[a], alias[b]), targets=tuple(alias[c] for c in candidates)
        )
        for (a, b), candidates in candidates_by_pair.items()
    }
    names = tuple(
        dict.fromkeys(
            name for pair, candidates in candidates_by_pair.items() for name in (*pair, *candidates)
        )
    )
    for chunk in frame_chunks():
        coded = _coded_chunk(chunk, names=names, books=books, alias=alias)
        for tracker in trackers.values():
            tracker.add(coded)
    varying: dict[tuple[str, str], frozenset[str]] = {}
    for pair, tracker in trackers.items():
        varied = tracker.finish()
        varying[pair] = frozenset(c for c in candidates_by_pair[pair] if alias[c] in varied)
    return varying


def _stringification_collides(distinct_b: Collection[object], *, index: str) -> bool:
    """`True` when two distinct `columns` values stringify to the same
    output column name, or a `columns` value's string equals `index`'s
    own column name. Either way `pivot` would produce a `Table` with a
    duplicate column name. Runs over the distinct `columns` value set
    alone, order-independent, since only the existence of a collision
    matters here.
    """
    seen_names: set[str] = set()
    for value in distinct_b:
        name = value if isinstance(value, str) else str(value)
        if name == index or name in seen_names:
            return True
        seen_names.add(name)
    return False


def _pivot_feasibility(
    *,
    index: str,
    columns: str,
    distinct_by_column: Mapping[str, Collection[object]],
    dependencies: Mapping[tuple[str, str], bool],
    non_droppable_columns: frozenset[str],
    duplicate_keys: bool,
    varying_columns: frozenset[str],
) -> PivotFeasibility:
    distinct_columns_values = distinct_by_column[columns]

    width = 1 + len(distinct_columns_values)
    cells = len(distinct_by_column[index]) * width

    unique = not duplicate_keys and not _stringification_collides(
        distinct_columns_values, index=index
    )
    homogeneous = all(
        dependencies[(index, c)] for c in non_droppable_columns if c not in (index, columns)
    )

    varying: tuple[tuple[str, int], ...] = tuple(
        (c, len(distinct_by_column[c])) for c in distinct_by_column if c in varying_columns
    )

    return PivotFeasibility(
        index=index,
        columns=columns,
        duplicate_keys=duplicate_keys,
        homogeneous=homogeneous,
        unique=unique,
        width=width,
        cells=cells,
        varying=varying,
    )


def with_party_kind_coverage(
    profile: OutputProfile,
    frame_chunks: Callable[[], Iterator[TableLike]],
    *,
    parties: Mapping[str, PartyRow],
    declared_roles: Sequence[str],
) -> OutputProfile:
    """`profile`, the profile of the frame `frame_chunks` yields, with each role column's
    `party_kind` coverage counted again from `parties` by the mint's own fold
    (`OutputProfileFold.of_party_kinds`), every other entry kept. A role column is one whose
    `ColumnProfile.kind` is `role`, and `declared_roles` are the roles the mint declared, in
    its order, which a mint writes the entries in. A recounted entry takes the place of the
    one it replaces; a new one follows its column's graph fields, where a mint writes it, and a
    column whose parties take no kind has none. A missing ref is named by the label another of
    the profile's entries names it by, else by none: the graph is not read. Raises
    `MaterialisationUsageError` for a role column `declared_roles` does not name."""
    profiled = {column.name for column in profile.columns if column.kind == "role"}
    if undeclared := sorted(profiled.difference(declared_roles)):
        raise MaterialisationUsageError(
            f"with_party_kind_coverage() needs every role column among declared_roles; "
            f"{undeclared!r} are not"
        )
    role_columns = [role for role in declared_roles if role in profiled]
    fold = OutputProfileFold.of_party_kinds(role_columns=role_columns, parties=parties)
    for chunk in frame_chunks():
        fold.add(chunk)
    labels = {ref: label for entry in profile.enrichments for ref, label in entry.missing if label}
    kinds = fold.party_kind_coverage(labels)

    enrichments: list[EnrichmentCoverage] = []
    placed: set[str] = set()
    for entry in profile.enrichments:
        if entry.field == PARTY_KIND_FIELD and entry.column in kinds:
            placed.add(entry.column)
            if (recounted := kinds[entry.column]) is not None:
                enrichments.append(recounted)
            continue
        enrichments.append(entry)
    for column, recounted in kinds.items():
        if column in placed or recounted is None:
            continue
        enrichments.insert(_new_entry_place(enrichments, column, role_columns), recounted)
    return replace(profile, enrichments=tuple(enrichments))


def _new_entry_place(
    enrichments: Sequence[EnrichmentCoverage], column: str, role_columns: Sequence[str]
) -> int:
    """Where a new `party_kind` entry of `column` goes among `enrichments`: after the
    column's last entry, `party_kind` being the last of `ENRICHMENT_FIELDS`, or, for a
    column with none, before the first entry of a role column after it in `role_columns`."""
    held = [at for at, entry in enumerate(enrichments) if entry.column == column]
    if held:
        return held[-1] + 1
    later = set(role_columns[role_columns.index(column) + 1 :])
    return next(
        (at for at, entry in enumerate(enrichments) if entry.column in later), len(enrichments)
    )


def _enrichment_coverage_for_column(
    role_column: str,
    refs_seen: Collection[object],
    resolved: Mapping[str, Mapping[str, str]],
    *,
    fields: Sequence[str],
    parties: Mapping[str, PartyRow],
    datasets: Mapping[object, Collection[object]] | None,
) -> tuple[EnrichmentCoverage, ...]:
    """The coverage of each of `fields` (`ENRICHMENT_FIELDS`) on `role_column`, whose cells
    hold `refs_seen`: a graph field from `resolved`, the lookup shared across role columns,
    and `party_kind` from `parties` on the column's own axis. A ref has `party_kind` where
    every layout fills its cells: a long download, which reads each row for its own dataset
    (`row_party_cell`), and a pivot, which reads the ref once over the datasets the frame
    serves it from (`pivot_party_cell`). `datasets` gives each ref's `provenance` cells,
    `None` where the frame has no such column or no party's kind depends on its dataset; each
    row then reads no dataset."""
    refs = sorted({ref for ref in refs_seen if isinstance(ref, str)})
    if not refs:
        return ()

    def kind_filled(ref: str) -> bool:
        served = datasets[ref] if datasets is not None else (None,)
        pivoted = pivot_party_cell(
            PARTY_KIND_FIELD,
            ref,
            role=role_column,
            parties=parties,
            datasets=[cell for cell in served if isinstance(cell, str)],
        )
        return pivoted is not None and all(
            row_party_cell(
                PARTY_KIND_FIELD,
                ref,
                role=role_column,
                parties=parties,
                period=None,
                provenance=provenance,
            )
            is not None
            for provenance in served
        )

    with_kind = {ref for ref in refs if kind_filled(ref)}

    def has(ref: str, field_name: str) -> bool:
        if field_name == PARTY_KIND_FIELD:
            return ref in with_kind
        return field_name in resolved.get(ref, {})

    coverage: list[EnrichmentCoverage] = []
    for field_name in fields:
        if field_name not in SUPPORTED_ENRICHMENTS or not enrichment_qualifies(
            role_column, field_name
        ):
            continue
        covered_refs = [ref for ref in refs if has(ref, field_name)]
        if not covered_refs:
            continue
        missing_refs = [ref for ref in refs if not has(ref, field_name)]
        truncated = len(missing_refs) > _MAX_NAMED_MISSING_REFS
        missing = tuple(
            (ref, resolved.get(ref, {}).get("label", ""))
            for ref in missing_refs[:_MAX_NAMED_MISSING_REFS]
        )
        coverage.append(
            EnrichmentCoverage(
                column=role_column,
                field=field_name,
                entities=len(refs),
                covered=len(covered_refs),
                missing=missing,
                truncated=truncated,
            )
        )
    return tuple(coverage)
