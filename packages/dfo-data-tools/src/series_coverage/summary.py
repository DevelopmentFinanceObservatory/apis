"""A per-variable facet summary of a coverage build, built once when the index loads.

`CoverageIndex` builds a `CoverageSummary` in its constructor, so a reload swaps the
summary with the rows it folds. A variable card reads one variable's facets from it
(`CoverageSummary.variable`) with no scan over coverage rows: each facet's metadata,
first and last years, observation count, and the parties it covers in each role.

The summary covers every row of the build. Source admission and DFO precedence are a
request's concern (`CoverageIndex.filter_candidates`), and the summary applies neither.

Parties are kept per facet as dictionary codes, sorted, in one array per role with an
offset per facet, and decode to dcids only for the variable a reader asks for. A
party's kind comes from the code index (`CodeIndex.parties`), which a coverage build
does not carry, so `VariableSummary.party_counts` takes it at read time.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import pyarrow as pa

from code_index.types import PartyKind, PartyRow
from series_coverage.coverage_types import Role
from series_coverage.schema import (
    ABSENT_ENTITY,
    DONOR_COLUMN,
    FACET_COLUMN,
    FIRST_YEAR_COLUMN,
    LAST_YEAR_COLUMN,
    N_OBS_COLUMN,
    RECIPIENT_COLUMN,
    VARIABLE_COLUMN,
    FacetMetadata,
)

_NO_METADATA = FacetMetadata()
_KIND_PRECEDENCE: tuple[PartyKind, ...] = ("party", "residual", "total")
"""The kind a party counts under when it takes several across datasets: a kind with figures of
its own before one holding other parties' figures."""
_ROLE_COLUMNS: dict[Role, str] = {"donor": DONOR_COLUMN, "recipient": RECIPIENT_COLUMN}


@dataclass(frozen=True, slots=True)
class PartyCounts:
    """The distinct parties in one role, split by their kind in the code index.

    `parties` counts parties with figures of their own (a country, an organisation),
    `private` those of them whose class is private (a foundation), a subset of `parties`,
    `totals` parties that hold other parties' figures ("Africa, Total"; a whole total such as
    "Official Donors, Total" is one), and `residual` parties holding figures no other party
    does ("Africa, unallocated by country"). `unclassified` counts parties the code index
    does not list. Each party counts once, under the first of `party`, `residual` and `total`
    it takes in the datasets of the facets that hold it (`PartyRow.kinds_over`, every dataset
    that publishes it for a facet with no provenance), so the kinds sum to the distinct parties.
    One that is a total in DAC2A and a party in the CRS has figures of its own in the CRS and
    counts in `parties` across both.
    """

    parties: int = 0
    private: int = 0
    totals: int = 0
    residual: int = 0
    unclassified: int = 0


@dataclass(frozen=True, slots=True)
class FacetSummary:
    """One facet of one variable across the build.

    `metadata` holds the facet's provenance, measurement method, unit, scaling factor
    and observation period, all `None` for a facet the build gives no metadata.
    `first_year` and `last_year` bound its years with data, a union over its series that
    claims no continuity at any one party. `observations` sums its series' observations.
    `parties` maps each role the variable declares to the dcids the facet's series carry
    in it; a role the variable lacks has no entry.
    """

    facet_id: str
    metadata: FacetMetadata
    first_year: int
    last_year: int
    observations: int
    parties: Mapping[Role, frozenset[str]]


@dataclass(frozen=True, slots=True)
class VariableSummary:
    """One variable's facets across the build, ordered by facet id."""

    variable: str
    facets: tuple[FacetSummary, ...]

    @property
    def first_year(self) -> int:
        """The first year any facet holds data."""
        return min(facet.first_year for facet in self.facets)

    @property
    def last_year(self) -> int:
        """The last year any facet holds data."""
        return max(facet.last_year for facet in self.facets)

    @property
    def observations(self) -> int:
        """The observations every facet holds, summed."""
        return sum(facet.observations for facet in self.facets)

    @property
    def provenances(self) -> frozenset[str]:
        """The provenances the facets carry; a facet without one adds none."""
        return frozenset(
            facet.metadata.provenance_id
            for facet in self.facets
            if facet.metadata.provenance_id is not None
        )

    def party_counts(
        self, *, parties: Mapping[str, PartyRow], facet_ids: Collection[str] | None = None
    ) -> dict[Role, PartyCounts]:
        """The distinct parties of each declared role across `facet_ids` (every facet when
        `None`), split by kind through `parties`, the code index's party rows
        (`CodeIndex.parties`), each read for the provenances of the facets that hold it. A party
        that sits in several facets counts once, under the kind `PartyCounts` names for it.
        """
        selected = [
            facet for facet in self.facets if facet_ids is None or facet.facet_id in facet_ids
        ]
        roles = sorted({role for facet in selected for role in facet.parties})
        held: dict[Role, dict[str, set[str]]] = {role: defaultdict(set) for role in roles}
        for facet in selected:
            provenance = facet.metadata.provenance_id
            for role, dcids in facet.parties.items():
                for dcid in dcids:
                    datasets = held[role][dcid]
                    if provenance is not None:
                        datasets.add(provenance)
        return {role: _split_by_kind(held[role], parties) for role in roles}


def _split_by_kind(
    held: Mapping[str, Collection[str]], parties: Mapping[str, PartyRow]
) -> PartyCounts:
    """`held`'s parties, each mapped to the provenances of the facets holding it, counted once
    under the first kind of `_KIND_PRECEDENCE` it takes there. A party counted as a `party` is
    private when that is its own kind and class, since a private provider takes no dataset kind
    (`PartyRow.class_in`)."""
    kinds: Counter[PartyKind | None] = Counter()
    private = 0
    for dcid, datasets in held.items():
        row = parties.get(dcid)
        if row is None:
            kinds[None] += 1
            continue
        taken = row.kinds_over(datasets)
        kind = next(kind for kind in _KIND_PRECEDENCE if kind in taken)
        kinds[kind] += 1
        if kind == row.kind == "party" and row.party_class == "private":
            private += 1
    return PartyCounts(
        parties=kinds["party"],
        private=private,
        totals=kinds["total"],
        residual=kinds["residual"],
        unclassified=kinds[None],
    )


@dataclass(frozen=True, slots=True)
class _RoleParties:
    """One role's parties per facet block: block `b` holds `codes[offsets[b]:offsets[b+1]]`,
    sorted codes into `names`."""

    names: tuple[str, ...]
    offsets: np.ndarray
    codes: np.ndarray


class CoverageSummary:
    """Every variable's facets across one coverage build, from one pass at load.

    Holds one entry per `(variable, facet_id)` block of the index's sorted rows in numpy
    arrays. For the 4.2 million rows of build `20261008T090650Z-7bcfc92c` that is 24,253
    blocks and 1,010,900 party entries in 2.8 MiB, built in 0.04 s.
    """

    def __init__(
        self,
        table: pa.Table,
        *,
        variable_codes: Mapping[str, int],
        facet_metadata: Mapping[str, FacetMetadata],
    ) -> None:
        """Folds `table`, the index's rows: one chunk, key columns dictionary-encoded, rows
        grouped by `(variable, facet_id)` (`series_coverage.index._group_rows`).
        `variable_codes` maps each variable to its code in the variable column."""
        self._variable_codes = variable_codes
        self._facet_metadata = facet_metadata
        variables = _codes(table, VARIABLE_COLUMN)
        facets = _codes(table, FACET_COLUMN)
        changed = (variables[1:] != variables[:-1]) | (facets[1:] != facets[:-1])
        starts = np.flatnonzero(np.r_[True, changed]) if table.num_rows else np.empty(0, np.int64)
        self._block_variables = variables[starts]
        self._block_facets = facets[starts]
        self._facet_names = _dictionary(table, FACET_COLUMN)
        if table.num_rows:
            self._observations = np.add.reduceat(
                table[N_OBS_COLUMN].to_numpy().astype(np.int64), starts
            )
            self._first_years = np.minimum.reduceat(table[FIRST_YEAR_COLUMN].to_numpy(), starts)
            self._last_years = np.maximum.reduceat(table[LAST_YEAR_COLUMN].to_numpy(), starts)
        else:
            self._observations = self._first_years = self._last_years = np.empty(0, np.int64)
        self._parties = {
            role: _role_parties(table, column, starts) for role, column in _ROLE_COLUMNS.items()
        }

    def variable(self, variable: str) -> VariableSummary | None:
        """`variable`'s facets, or `None` when the build holds no row of it."""
        code = self._variable_codes.get(variable)
        if code is None:
            return None
        needle = self._block_variables.dtype.type(code)
        start = int(np.searchsorted(self._block_variables, needle, side="left"))
        stop = int(np.searchsorted(self._block_variables, needle, side="right"))
        if start == stop:
            return None
        facets = [self._facet(block) for block in range(start, stop)]
        return VariableSummary(
            variable=variable, facets=tuple(sorted(facets, key=lambda f: f.facet_id))
        )

    def _facet(self, block: int) -> FacetSummary:
        facet_id = self._facet_names[int(self._block_facets[block])]
        parties: dict[Role, frozenset[str]] = {}
        for role, found in self._parties.items():
            codes = found.codes[found.offsets[block] : found.offsets[block + 1]]
            if codes.size:
                parties[role] = frozenset(found.names[code] for code in codes.tolist())
        return FacetSummary(
            facet_id=facet_id,
            metadata=self._facet_metadata.get(facet_id, _NO_METADATA),
            first_year=int(self._first_years[block]),
            last_year=int(self._last_years[block]),
            observations=int(self._observations[block]),
            parties=parties,
        )


def _codes(table: pa.Table, column: str) -> np.ndarray:
    if table.num_rows == 0:
        return np.empty(0, np.int32)
    return cast("pa.DictionaryArray", table[column].chunk(0)).indices.to_numpy()


def _dictionary(table: pa.Table, column: str) -> tuple[str, ...]:
    if table[column].num_chunks == 0:
        return ()
    dictionary = cast("pa.DictionaryArray", table[column].chunk(0)).dictionary
    return tuple(cast("list[str]", dictionary.to_pylist()))


def _role_parties(table: pa.Table, column: str, starts: np.ndarray) -> _RoleParties:
    """Each block's distinct parties in `column`, without `ABSENT_ENTITY`.

    Marks `block * parties + code` in one boolean array of blocks times parties, whose
    set positions read back in block order, then code order. For the 24,253 blocks and
    340 donors of build `20261008T090650Z-7bcfc92c` that array takes 8 MiB while it
    builds, and `np.unique` over the 4.2 million keys would take 0.15 s per role.
    """
    names = _dictionary(table, column)
    width = max(len(names), 1)
    blocks = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, table.num_rows]))
    present = np.zeros((starts.size, width), dtype=bool)
    present[blocks, _codes(table, column)] = True
    if ABSENT_ENTITY in names:
        present[:, names.index(ABSENT_ENTITY)] = False
    keys = np.flatnonzero(present)
    return _RoleParties(
        names=names,
        offsets=np.r_[0, np.cumsum(present.sum(axis=1))],
        codes=(keys % width).astype(_code_dtype(len(names))),
    )


def _code_dtype(size: int) -> type[np.unsignedinteger]:
    return np.uint16 if size <= np.iinfo(np.uint16).max + 1 else np.uint32
