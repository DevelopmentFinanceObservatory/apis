"""Row-loop oracle for `materialise.profile.compute_output_profile_from_chunks`.

This module is a permanent copy of the Python row-loop profile that the Arrow
implementation replaced. It shares only the profile dataclasses and column
classification with production, so the differential test in
`test_profile_arrow_parity.py` keeps comparing against the row semantics. A
deliberate change to a profile rule changes both, or deletes that test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

from dfo_frame_grammar import Frame

from materialise.enrichment import ENRICHMENT_PROPERTIES, EnrichmentLookup
from materialise.errors import MaterialisationUsageError
from materialise.profile import (
    ColumnProfile,
    EnrichmentCoverage,
    OutputProfile,
    PivotFeasibility,
)
from materialise.reshape import (
    SUPPORTED_ENRICHMENTS,
    ColumnClassification,
    TableLike,
    classify_columns,
)

_SCHEMA_VERSION = 1
_VALUE_COLUMN = "value"
_STATUS_COLUMN = "status"
_UNIT_COLUMN = "unit"
_MAX_NAMED_MISSING_REFS = 10

_ROWS_SLICE = 8192


def _iter_rows(table: TableLike) -> Iterator[tuple[object, ...]]:
    """Read `Frame` rows in `_ROWS_SLICE`-row slices; other tables expose stored rows."""
    if not isinstance(table, Frame):
        yield from table.rows
        return
    whole = table.table
    for start in range(0, whole.num_rows, _ROWS_SLICE):
        row_slice = whole.slice(start, _ROWS_SLICE)
        yield from Frame(columns=table.columns, table=row_slice).rows


def reference_output_profile_from_chunks(
    frame_chunks: Callable[[], Iterator[TableLike]],
    *,
    role_columns: Sequence[str],
    enrichment: EnrichmentLookup,
) -> OutputProfile:
    """Compute the exact profile from a fresh chunk iterator on each pass.
    State is bounded by distinct column values and candidate pivot keys,
    rather than row count. Pass one gathers distinct counts, functional
    dependencies, and duplicate keys; only duplicate keys trigger pass two,
    which finds columns that vary within those groups. Each candidate stops
    accumulating state once its result is known.

    `role_columns` must exist in the first chunk, and every chunk must share
    its column order. The iterator must yield at least one chunk, including
    for an empty result; pivot pairs follow that shared column order.
    """
    chunks = frame_chunks()
    try:
        first_chunk = next(chunks)
    except StopIteration:
        raise MaterialisationUsageError(
            "compute_output_profile_from_chunks() requires frame_chunks() to yield at "
            "least one chunk"
        ) from None

    columns = first_chunk.columns
    unknown_roles = [name for name in role_columns if name not in columns]
    if unknown_roles:
        raise MaterialisationUsageError(
            "compute_output_profile() requires every role_columns entry to be present "
            f"on the frame; frame columns are {columns!r}, got unknown "
            f"role column(s) {unknown_roles!r}"
        )

    classification = classify_columns(columns)
    role_column_set = frozenset(role_columns)
    positions = {name: i for i, name in enumerate(columns)}

    distinct_columns = tuple(
        name for name in columns if name not in (_VALUE_COLUMN, _STATUS_COLUMN)
    )
    distinct_by_column: dict[str, set[object]] = {name: set() for name in distinct_columns}

    # Candidate sources come from column kinds before distinct counts are known.
    # Tracking stops when a dependency fails or a key repeats; pivot output
    # uses candidates with multiple distinct values.
    candidate_sources = tuple(
        name
        for name in columns
        if _column_kind(name, role_columns=role_column_set, classification=classification)
        in ("grain", "role")
    )
    fd_seen: dict[tuple[str, str], dict[object, object]] = {
        (a, c): {} for a in candidate_sources for c in distinct_columns if c != a
    }
    fd_violated: set[tuple[str, str]] = set()

    pair_names: tuple[tuple[str, str], ...] = tuple(
        (a, b) for i, a in enumerate(candidate_sources) for b in candidate_sources[i + 1 :]
    )
    dup_seen: dict[tuple[str, str], set[tuple[object, object]]] = {
        pair: set() for pair in pair_names
    }
    dup_found: dict[tuple[str, str], bool] = dict.fromkeys(pair_names, False)

    total_rows = 0
    for chunk in _chain_first(first_chunk, chunks):
        for row in _iter_rows(chunk):
            total_rows += 1
            for name in distinct_columns:
                distinct_by_column[name].add(row[positions[name]])
            for pair, seen in fd_seen.items():
                if pair in fd_violated:
                    continue
                a, c = pair
                a_value = row[positions[a]]
                c_value = row[positions[c]]
                if a_value in seen:
                    if seen[a_value] != c_value:
                        fd_violated.add(pair)
                        seen.clear()
                else:
                    seen[a_value] = c_value
            for pair in pair_names:
                if dup_found[pair]:
                    continue
                a, b = pair
                key = (row[positions[a]], row[positions[b]])
                seen_keys = dup_seen[pair]
                if key in seen_keys:
                    dup_found[pair] = True
                    seen_keys.clear()
                else:
                    seen_keys.add(key)

    dependencies: dict[tuple[str, str], bool] = {pair: pair not in fd_violated for pair in fd_seen}

    column_profiles = tuple(
        ColumnProfile(
            name=name,
            kind=_column_kind(name, role_columns=role_column_set, classification=classification),
            droppable=name not in classification.non_droppable_columns,
            distinct=(
                None if name in (_VALUE_COLUMN, _STATUS_COLUMN) else len(distinct_by_column[name])
            ),
        )
        for name in columns
    )

    grain_columns = tuple(name for name in candidate_sources if len(distinct_by_column[name]) > 1)

    def _canonical_dup_pair(index: str, other: str) -> tuple[str, str]:
        return (index, other) if (index, other) in dup_found else (other, index)

    varying_candidates_by_pair: dict[tuple[str, str], tuple[str, ...]] = {}
    for index in grain_columns:
        for other in grain_columns:
            if other == index:
                continue
            if not dup_found[_canonical_dup_pair(index, other)]:
                continue
            candidates = tuple(
                c
                for c in columns
                if c not in (index, other, _VALUE_COLUMN, _STATUS_COLUMN)
                and len(distinct_by_column[c]) > 1
                and not dependencies[(index, c)]
                and not dependencies[(other, c)]
            )
            if candidates:
                varying_candidates_by_pair[(index, other)] = candidates

    varying_by_pair = (
        _second_pass_varying(
            frame_chunks, positions=positions, candidates_by_pair=varying_candidates_by_pair
        )
        if varying_candidates_by_pair
        else {}
    )

    pivots = tuple(
        _pivot_feasibility(
            index=a,
            columns=b,
            distinct_by_column=distinct_by_column,
            dependencies=dependencies,
            non_droppable_columns=classification.non_droppable_columns,
            duplicate_keys=dup_found[_canonical_dup_pair(a, b)],
            varying_columns=varying_by_pair.get((a, b), frozenset()),
        )
        for a in grain_columns
        for b in grain_columns
        if a != b
    )

    role_refs = sorted(
        {
            ref
            for role_column in role_columns
            for ref in distinct_by_column[role_column]
            if isinstance(ref, str)
        }
    )
    resolved = enrichment.lookup(role_refs) if role_refs else {}
    enrichments = tuple(
        coverage
        for role_column in role_columns
        for coverage in _enrichment_coverage_for_column(
            role_column, distinct_by_column[role_column], resolved
        )
    )

    return OutputProfile(
        schema_version=_SCHEMA_VERSION,
        rows=total_rows,
        columns=column_profiles,
        pivots=pivots,
        enrichments=enrichments,
    )


def _chain_first(first: TableLike, rest: Iterator[TableLike]) -> Iterator[TableLike]:
    """Yield the peeked first chunk, then the rest in order."""
    yield first
    yield from rest


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


def _second_pass_varying(
    frame_chunks: Callable[[], Iterator[TableLike]],
    *,
    positions: Mapping[str, int],
    candidates_by_pair: Mapping[tuple[str, str], Sequence[str]],
) -> dict[tuple[str, str], frozenset[str]]:
    """Scan a fresh chunk iterator only for pairs with duplicate keys.
    Pass one releases its seen-key set at the first duplicate; this pass
    records the first value per key for candidate columns and drops a column
    once it varies. All affected pairs share this pass.
    """
    first_seen: dict[tuple[str, str], dict[tuple[object, object], dict[str, object]]] = {
        pair: {} for pair in candidates_by_pair
    }
    active: dict[tuple[str, str], dict[str, int]] = {
        pair: {name: positions[name] for name in candidates} if candidates else {}
        for pair, candidates in candidates_by_pair.items()
    }
    varying: dict[tuple[str, str], set[str]] = {pair: set() for pair in candidates_by_pair}

    for chunk in frame_chunks():
        for row in _iter_rows(chunk):
            for pair, live_candidates in active.items():
                if not live_candidates:
                    continue
                a_pos = positions[pair[0]]
                b_pos = positions[pair[1]]
                key = (row[a_pos], row[b_pos])
                key_seen = first_seen[pair].setdefault(key, {})
                for name in list(live_candidates):
                    value = row[live_candidates[name]]
                    if name in key_seen:
                        if key_seen[name] != value:
                            varying[pair].add(name)
                            del live_candidates[name]
                    else:
                        key_seen[name] = value

    return {pair: frozenset(names) for pair, names in varying.items()}


def _stringification_collides(distinct_b: set[object], *, index: str) -> bool:
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
    distinct_by_column: Mapping[str, set[object]],
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


def _enrichment_coverage_for_column(
    role_column: str, refs_seen: set[object], resolved: Mapping[str, Mapping[str, str]]
) -> tuple[EnrichmentCoverage, ...]:
    # Role columns contain string dcids. `resolved` is the shared lookup
    # across all role columns; this function reads its own refs from it.
    refs = sorted({ref for ref in refs_seen if isinstance(ref, str)})
    if not refs:
        return ()

    coverage: list[EnrichmentCoverage] = []
    for field_name in ENRICHMENT_PROPERTIES:
        if field_name not in SUPPORTED_ENRICHMENTS:
            continue
        covered_refs = [ref for ref in refs if field_name in resolved.get(ref, {})]
        if not covered_refs:
            continue
        missing_refs = [ref for ref in refs if field_name not in resolved.get(ref, {})]
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
