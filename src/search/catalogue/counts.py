"""Counts what DFO holds, globally and at each country, from one origin
catalogue build and one coverage index build.

- Publishers: distinct `ProvenanceRow.source.ref` over the admitted
  provenances. At a country, the sources of the base provenances holding a
  variable there (`CatalogueTables.provenance_places`) and of the coverage
  facets with a row naming it.
- Countries: `country/` dcids with a base variable's place bit set, or named
  as donor or recipient on a coverage row.
- Aid: donors, recipients and years over the coverage rows of variables
  with a CRS facet. At a country, the donors to it, the recipients of its
  own aid, and the years of either.
- Variables (per country only): base variables with data there, plus
  coverage variables with a row naming it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import pyarrow as pa

from origin_catalogue.catalogue import CatalogueSnapshot
from search.catalogue.artifact import AidCounts, Counts, PlaceCounts, YearRange
from series_coverage.coverage_types import Role
from series_coverage.index import CoverageIndex
from series_coverage.schema import FACET_COLUMN, FIRST_YEAR_COLUMN, LAST_YEAR_COLUMN

COUNTRY_PREFIX: Final[str] = "country/"

CRS_PROVENANCE: Final[str] = "provenance/OECDCreditorReportingSystem"
"""The provenance of the coverage index's aid rows. Its other rows are World
Bank debt series."""

_ROLES: Final[tuple[Role, ...]] = ("donor", "recipient")


@dataclass(frozen=True, slots=True)
class CatalogueCounts:
    """The global counts and the counts at each country `counts.countries`
    counts, keyed by dcid."""

    counts: Counts
    places: Mapping[str, PlaceCounts]


@dataclass(frozen=True, slots=True)
class _BaseByCountry:
    variables: Mapping[str, int]
    sources: Mapping[str, frozenset[str]]
    index: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _CoverageByCountry:
    """What the coverage rows naming each country, as donor or recipient,
    hold. `overlap` counts the variables that are also base variables with
    data there. `aid_donors` counts the donors of aid to the country and
    `aid_recipients` the recipients of its own aid."""

    variables: Mapping[str, int]
    overlap: Mapping[str, int]
    sources: Mapping[str, frozenset[str]]
    years: Mapping[str, YearRange]
    aid_donors: Mapping[str, int]
    aid_recipients: Mapping[str, int]
    aid: AidCounts


def catalogue_counts(*, catalogue: CatalogueSnapshot, coverage: CoverageIndex) -> CatalogueCounts:
    """Counts over `catalogue`'s admitted provenances and base variable
    table and every row of `coverage`."""
    source_of = {row.id: row.source.ref for row in catalogue.provenance_rows()}
    base = _base_by_country(catalogue, source_of)
    held = _coverage_by_country(coverage, catalogue=catalogue, base=base, source_of=source_of)
    countries = sorted(set(base.variables) | set(held.variables))
    no_sources: frozenset[str] = frozenset()
    places = {
        country: PlaceCounts(
            publishers=len(
                base.sources.get(country, no_sources) | held.sources.get(country, no_sources)
            ),
            variables=base.variables.get(country, 0)
            + held.variables.get(country, 0)
            - held.overlap.get(country, 0),
            aid=AidCounts(
                donors=held.aid_donors.get(country, 0),
                recipients=held.aid_recipients.get(country, 0),
                years=held.years.get(country),
            ),
        )
        for country in countries
    }
    counts = Counts(publishers=len(set(source_of.values())), countries=len(countries), aid=held.aid)
    return CatalogueCounts(counts=counts, places=places)


def _coverage_by_country(
    coverage: CoverageIndex,
    *,
    catalogue: CatalogueSnapshot,
    base: _BaseByCountry,
    source_of: Mapping[str, str],
) -> _CoverageByCountry:
    """Every country's coverage summary from six grouped passes over the
    index: each role by facet and by variable, the variables by facet, and
    the aid variables' donor and recipient pairs."""
    provenance_of = {
        facet_id: meta.provenance_id for facet_id, meta in coverage.facet_metadata.items()
    }
    crs_facets = frozenset(
        facet_id for facet_id, provenance in provenance_of.items() if provenance == CRS_PROVENANCE
    )

    sources: dict[str, set[str]] = defaultdict(set)
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    crs_spans: list[tuple[int, int]] = []
    for role in _ROLES:
        by_facet = coverage.grouped(by=(role, "facet_id"))
        for place, facet, first, last in zip(
            _strings(by_facet[role]),
            _strings(by_facet[FACET_COLUMN]),
            _ints(by_facet[FIRST_YEAR_COLUMN]),
            _ints(by_facet[LAST_YEAR_COLUMN]),
            strict=True,
        ):
            if facet in crs_facets:
                crs_spans.append((first, last))
            if not place.startswith(COUNTRY_PREFIX):
                continue
            provenance = provenance_of.get(facet)
            if provenance is not None and provenance in source_of:
                sources[place].add(source_of[provenance])
            if facet in crs_facets:
                spans[place].append((first, last))

    by_variable_facet = coverage.grouped(by=("variable", "facet_id"))
    aid = sorted(
        {
            variable
            for variable, facet in zip(
                _strings(by_variable_facet["variable"]),
                _strings(by_variable_facet[FACET_COLUMN]),
                strict=True,
            )
            if facet in crs_facets
        }
    )
    pairs = coverage.grouped(by=_ROLES, variables=aid)
    donors, recipients = _strings(pairs["donor"]), _strings(pairs["recipient"])

    variables, overlap = _variables_by_country(coverage, catalogue=catalogue, base=base)
    return _CoverageByCountry(
        variables=variables,
        overlap=overlap,
        sources={place: frozenset(named) for place, named in sources.items()},
        years={place: years for place, held in spans.items() if (years := _years(held))},
        aid_donors=Counter(recipients),
        aid_recipients=Counter(donors),
        aid=AidCounts(
            donors=len(set(donors)), recipients=len(set(recipients)), years=_years(crs_spans)
        ),
    )


def _variables_by_country(
    coverage: CoverageIndex, *, catalogue: CatalogueSnapshot, base: _BaseByCountry
) -> tuple[dict[str, int], dict[str, int]]:
    """Each country's count of coverage variables with a row naming it in
    either role, and how many of those are base variables with data there."""
    place_ids: dict[str, int] = {}
    variable_ids: dict[str, int] = {}
    held: list[tuple[np.ndarray, np.ndarray]] = []
    for role in _ROLES:
        by_variable = coverage.grouped(by=(role, "variable"))
        held.append(
            (_ids(by_variable[role], place_ids), _ids(by_variable["variable"], variable_ids))
        )
    place_names = list(place_ids)
    countries = np.array(
        [place_id for place, place_id in place_ids.items() if place.startswith(COUNTRY_PREFIX)],
        dtype=np.int64,
    )
    width = max(len(variable_ids), 1)
    keys = np.unique(
        np.concatenate(
            [
                places[np.isin(places, countries)] * width + variables[np.isin(places, countries)]
                for places, variables in held
            ]
        )
    )
    counts = np.bincount(keys // width, minlength=len(place_names))
    variables = {place_names[i]: int(count) for i, count in enumerate(counts) if count}

    base_variables = catalogue.tables.base_variables
    names = list(variable_ids)
    shared = np.array(
        [variable_id for name, variable_id in variable_ids.items() if name in base_variables],
        dtype=np.int64,
    )
    overlap: Counter[str] = Counter()
    for key in keys[np.isin(keys % width, shared)].tolist():
        place = place_names[key // width]
        bit = base.index.get(place)
        if bit is not None and base_variables[names[key % width]].places >> bit & 1:
            overlap[place] += 1
    return variables, overlap


def _base_by_country(catalogue: CatalogueSnapshot, source_of: Mapping[str, str]) -> _BaseByCountry:
    """Each country's base variable count, from one pass over the variable
    table, and the sources of the base provenances holding a variable
    there, from the provenance place bitsets."""
    tables = catalogue.tables
    index = {
        place: bit
        for bit, place in enumerate(tables.scan_places)
        if place.startswith(COUNTRY_PREFIX)
    }
    place_at = {bit: place for place, bit in index.items()}
    country_mask = sum(1 << bit for bit in place_at)
    variables: Counter[str] = Counter()
    for entry in tables.base_variables.values():
        variables.update(place_at[bit] for bit in _set_bits(entry.places & country_mask))
    sources: dict[str, set[str]] = defaultdict(set)
    for provenance, places in tables.provenance_places.items():
        source = source_of.get(provenance)
        if source is None:
            continue
        for bit in _set_bits(places & country_mask):
            sources[place_at[bit]].add(source)
    return _BaseByCountry(
        variables=variables,
        sources={place: frozenset(named) for place, named in sources.items()},
        index=index,
    )


def _set_bits(bits: int) -> Iterator[int]:
    while bits:
        lowest = bits & -bits
        yield lowest.bit_length() - 1
        bits ^= lowest


def _strings(column: pa.ChunkedArray) -> list[str]:
    return cast("list[str]", column.cast(pa.string()).to_pylist())


def _ints(column: pa.ChunkedArray) -> list[int]:
    return cast("list[int]", column.to_pylist())


def _ids(column: pa.ChunkedArray, ids: dict[str, int]) -> np.ndarray:
    """Each row's id in `ids`, which gains an id for every value of the
    dictionary-encoded `column` it lacks. Rows are read as dictionary codes,
    so no string is built per row."""
    array = cast("pa.DictionaryArray", column.combine_chunks())
    lookup = np.array(
        [
            ids.setdefault(value, len(ids))
            for value in cast("list[str]", array.dictionary.to_pylist())
        ],
        dtype=np.int64,
    )
    return lookup[array.indices.to_numpy()]


def _years(spans: Iterable[tuple[int, int]]) -> YearRange | None:
    spans = list(spans)
    if not spans:
        return None
    return YearRange(first=min(first for first, _ in spans), last=max(last for _, last in spans))


__all__ = ["COUNTRY_PREFIX", "CRS_PROVENANCE", "CatalogueCounts", "catalogue_counts"]
