"""Checks coverage rows before they become a `CoverageIndex`.

`checked_index` refuses rows whose year-mask popcount disagrees with `n_obs`, rows whose facet
has no metadata, rows whose entities disagree with their variable's role names, and a table
`CoverageIndex` refuses. `graph_release.views` builds the release's coverage view through it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from series_coverage._bits import popcount64
from series_coverage.index import CoverageIndex
from series_coverage.schema import (
    ABSENT_ENTITY,
    DONOR_COLUMN,
    FACET_COLUMN,
    N_OBS_COLUMN,
    RECIPIENT_COLUMN,
    VARIABLE_COLUMN,
    FacetMetadata,
    RoleNames,
    YearHorizon,
)


class CoverageArtifactError(Exception):
    """Raised when coverage rows fail a `checked_index` check: mask popcounts, unique series
    keys, complete facet metadata, or entities matching their variable's role names."""


def checked_index(
    table: pa.Table,
    role_names: Mapping[str, RoleNames],
    facet_metadata: Mapping[str, FacetMetadata],
    *,
    horizon: YearHorizon,
    build_year: int | None,
    label: str,
) -> CoverageIndex:
    """The `CoverageIndex` over `table`, once every row's mask popcount equals its `n_obs`,
    every facet has metadata and every row's entities match its variable's role names.
    `table`'s `donor` and `recipient` columns are dictionary-encoded. `label` names the source
    in errors.

    Raises `CoverageArtifactError` for a failed check or a table `CoverageIndex` refuses.
    """
    _check_popcount_invariant(table, horizon, label=label)
    _check_facet_metadata_completeness(table, facet_metadata, label=label)
    _check_role_slots(table, role_names, label=label)
    try:
        return CoverageIndex(
            table, role_names, facet_metadata, horizon=horizon, build_year=build_year
        )
    except ValueError as exc:
        raise CoverageArtifactError(f"{label}: {exc}") from exc


def _check_facet_metadata_completeness(
    table: pa.Table, facet_metadata: Mapping[str, FacetMetadata], *, label: str
) -> None:
    """Requires metadata for every `facet_id` in `table`.

    Missing entries raise `CoverageArtifactError` before candidate ranking.
    """
    if table.num_rows == 0:
        return
    table_facet_ids = set(cast("list[str]", pc.unique(table[FACET_COLUMN]).to_pylist()))
    missing = sorted(table_facet_ids - facet_metadata.keys())
    if missing:
        raise CoverageArtifactError(
            f"{label} has {len(missing)} facet_id(s) with no facet metadata, e.g. {missing[:10]!r}"
        )


def _variables_by_absent_entity(table: pa.Table, column: str) -> tuple[set[str], set[str]]:
    """The variables with at least one row holding `ABSENT_ENTITY` in `column`, and those
    with at least one row holding another value, tested once per dictionary entry."""
    absent_rows = pa.chunked_array(
        [
            pc.take(
                pc.equal(chunk.dictionary, pa.scalar(ABSENT_ENTITY, type=pa.string())),
                chunk.indices,
            )
            for chunk in cast("list[pa.DictionaryArray]", table[column].chunks)
        ],
        type=pa.bool_(),
    )

    def variables(rows: pa.ChunkedArray) -> set[str]:
        found = pc.unique(table[VARIABLE_COLUMN].filter(rows))
        return set(cast("list[str]", found.cast(pa.string()).to_pylist()))

    return variables(absent_rows), variables(pc.invert(absent_rows))


def _check_role_slots(table: pa.Table, role_names: Mapping[str, RoleNames], *, label: str) -> None:
    """Requires every row's entities to match its variable's `RoleNames`: every row of a
    variable with no donor role holds an empty `donor`, and no row of any other variable
    does, and the same for `recipient`."""
    if table.num_rows == 0:
        return
    covered = set(
        cast("list[str]", pc.unique(table[VARIABLE_COLUMN]).cast(pa.string()).to_pylist())
    )
    named = [(variable, names) for variable in covered if (names := role_names.get(variable))]
    for role, column, lacking in (
        ("donor", DONOR_COLUMN, {v for v, n in named if n.donor is None}),
        ("recipient", RECIPIENT_COLUMN, {v for v, n in named if n.recipient is None}),
    ):
        with_absent, with_entity = _variables_by_absent_entity(table, column)
        mismatched = sorted((lacking & with_entity) | (with_absent - lacking))
        if mismatched:
            raise CoverageArtifactError(
                f"{label} has {len(mismatched)} variable(s) whose rows' {role} column "
                f"disagrees with its role names (every row of a variable with a null {role} "
                f"role holds an empty {role}, and no other row does), e.g. {mismatched[:10]!r}"
            )


def _check_popcount_invariant(table: pa.Table, horizon: YearHorizon, *, label: str) -> None:
    """Checks each row's `n_obs` against the total set bits in its mask words."""
    if table.num_rows == 0:
        return
    n_obs = table[N_OBS_COLUMN].to_numpy(zero_copy_only=False).astype(np.uint64)
    bits = np.zeros(table.num_rows, dtype=np.uint64)
    for name in horizon.mask_columns:
        bits += popcount64(table[name].to_numpy(zero_copy_only=False).astype(np.uint64))
    mismatched = n_obs != bits
    count = int(mismatched.sum())
    if count:
        first_index = int(np.flatnonzero(mismatched)[0])
        raise CoverageArtifactError(
            f"{label} has {count} row(s) where "
            f"n_obs != the year-mask bit count, e.g. row {first_index}"
        )


__all__ = ["CoverageArtifactError", "checked_index"]
