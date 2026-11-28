"""Counts what DFO holds, globally and at each country, from one origin
catalogue build and one coverage index build.

- Indicators (global only): the StatVars search can serve, meaning the
  origin catalogue's base variables, its DFO variables, the coverage
  variables it does not withhold, and the UN variables (none with the UN
  origin off).
- Datasets (global only): distinct source-index entries (`entry_ref`) over
  the admitted provenances.
- Publishers: distinct `ProvenanceRow.source.ref` over the admitted
  provenances. At a country, the sources of the base provenances holding a
  variable there (`CatalogueTables.provenance_places`) and of the coverage
  facets with a row naming it.
- Countries: `country/` dcids with a base variable's place bit set, or named
  as donor or recipient on a coverage row.
- Aid: donors, recipients and years over the coverage rows on a facet of a
  declared dataset (`CodeIndex.datasets`). A variable that declares one
  role adds no entity on the role it lacks. A party the code index declares
  a total of other parties (`CodeIndex.parties`) is no donor or recipient of
  its own. At a country, the donors to it, the recipients of its own aid,
  and the years of either.
- Origin split (`Counts.by_origin`, global only): one group per origin, and
  a provenance belongs to the group of its `ProvenanceRow.data_commons`. The
  dfo group (`DFO`) holds the DFO variables, the provenances of that origin
  and the coverage variables search serves on a facet of one of them. The
  data_commons group (`BASE`) holds the base variables, the base provenances
  and the coverage variables search serves on a facet of one of them. The un
  group (`UN_SYSTEM`) holds the UN variables of the origin catalogue's
  `un_variables.parquet`, the UN provenances and the coverage variables
  search serves on a facet of one of them; with the UN origin off it admits
  no UN provenance and holds nothing. A coverage variable therefore follows
  the origin of its facet's provenance. Something more than one group holds,
  such as a publisher with a dfo and a base provenance, counts in each, so
  the split can exceed the global figure. The provenance ids of each group
  are kept beside the counts (`CatalogueCounts.provenances`). The global
  `indicators` is the union of the groups' indicators, so a StatVar in two
  groups counts once there. A coverage facet with no provenance, or one the
  catalogue does not admit, adds to no group.
- Variables (per country only): base variables with data there, plus
  coverage variables with a row naming it.
"""

from __future__ import annotations

import functools
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np
import pyarrow as pa

from code_index.multilaterals import COUNTRY_PREFIX
from code_index.shared import dataset_of
from code_index.types import PartyRow
from origin_catalogue.catalogue import CatalogueSnapshot, ProvenanceRow, entry_ref
from origin_catalogue.origins import BASE, DFO, UN_SYSTEM, OriginId
from search.catalogue.artifact import (
    AidCounts,
    Counts,
    OriginCounts,
    OriginProvenances,
    OriginSplit,
    PlaceCounts,
    YearRange,
)
from series_coverage.coverage_types import COVERAGE_ROLE_AXES, Role
from series_coverage.index import CoverageIndex
from series_coverage.schema import (
    ABSENT_ENTITY,
    FACET_COLUMN,
    FIRST_YEAR_COLUMN,
    LAST_YEAR_COLUMN,
)

_ROLES: Final[tuple[Role, ...]] = ("donor", "recipient")

type _Group = Literal["dfo", "data_commons", "un"]

_GROUP_OF: Final[dict[OriginId, _Group]] = {
    DFO.id: "dfo",
    BASE.id: "data_commons",
    UN_SYSTEM.id: "un",
}
"""The `OriginCounts` group of each origin."""


@dataclass(frozen=True, slots=True)
class CatalogueCounts:
    """The global counts and the counts at each country `counts.countries`
    counts, keyed by dcid."""

    counts: Counts
    provenances: OriginProvenances
    places: Mapping[str, PlaceCounts]


@dataclass(frozen=True, slots=True)
class _BaseByCountry:
    variables: Mapping[str, int]
    sources: Mapping[str, frozenset[str]]
    index: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _CoverageByCountry:
    """What the coverage rows naming each country, as donor or recipient,
    hold. `covered` holds, by origin group, the variables with a coverage
    row on a facet of an admitted provenance of that group.
    `overlap` counts the variables that are also base variables with data
    there.
    `aid_donors` counts the donors of aid to the country and
    `aid_recipients` the recipients of its own aid."""

    covered: Mapping[_Group, frozenset[str]]
    variables: Mapping[str, int]
    overlap: Mapping[str, int]
    sources: Mapping[str, frozenset[str]]
    years: Mapping[str, YearRange]
    aid_donors: Mapping[str, int]
    aid_recipients: Mapping[str, int]
    aid: AidCounts


def catalogue_counts(
    *,
    catalogue: CatalogueSnapshot,
    coverage: CoverageIndex,
    datasets: Collection[str],
    parties: Mapping[str, PartyRow],
) -> CatalogueCounts:
    """Counts over `catalogue`'s admitted provenances and base variable
    table and every row of `coverage`. `datasets` holds the ids of the
    declared datasets (`CodeIndex.datasets`). Their coverage rows are the
    aid rows, a member provenance's rows read as its dataset's in the bound
    release's code index (`code_index.shared.dataset_of`). The aid counts
    leave out each aid row's party that totals other parties on its role's
    axis in the row's dataset (`parties`, `PartyRow.kind_over` of `total` on
    `COVERAGE_ROLE_AXES` for the facet's dataset), so the IMF trust funds, a
    total as a donor in some datasets, count as a donor in the others."""
    provenances = catalogue.provenance_rows()
    source_of = {row.id: row.source.ref for row in provenances}
    base = _base_by_country(catalogue, source_of)
    held = _coverage_by_country(
        coverage,
        catalogue=catalogue,
        base=base,
        source_of=source_of,
        datasets=datasets,
        parties=parties,
    )
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
    tables = catalogue.tables
    un_variables = catalogue.un.mirror.variables.keys() if catalogue.un_system_enabled else set()
    held_variables: dict[_Group, Collection[str]] = {
        "dfo": tables.dfo_variables,
        "data_commons": tables.base_variables.keys(),
        "un": un_variables,
    }
    group_indicators = {
        group: set(held_variables[group]) | (held.covered[group] - tables.withheld_variables)
        for group in _GROUP_OF.values()
    }
    group_rows: dict[_Group, list[ProvenanceRow]] = {group: [] for group in _GROUP_OF.values()}
    for row in provenances:
        group_rows[_GROUP_OF[row.data_commons.id]].append(row)
    counts = Counts(
        indicators=len(set().union(*group_indicators.values())),
        datasets=len({entry_ref(row) for row in provenances}),
        publishers=len(set(source_of.values())),
        countries=len(countries),
        aid=held.aid,
        by_origin=OriginCounts(
            dfo=_origin_split(group_indicators["dfo"], group_rows["dfo"]),
            data_commons=_origin_split(
                group_indicators["data_commons"], group_rows["data_commons"]
            ),
            un=_origin_split(group_indicators["un"], group_rows["un"]),
        ),
    )
    grouped = OriginProvenances(
        dfo=sorted(row.id for row in group_rows["dfo"]),
        data_commons=sorted(row.id for row in group_rows["data_commons"]),
        un=sorted(row.id for row in group_rows["un"]),
    )
    return CatalogueCounts(counts=counts, provenances=grouped, places=places)


def _origin_split(indicators: Collection[str], rows: Collection[ProvenanceRow]) -> OriginSplit:
    """`indicators` counted, with the distinct datasets and publishers of `rows`."""
    return OriginSplit(
        indicators=len(indicators),
        datasets=len({entry_ref(row) for row in rows}),
        publishers=len({row.source.ref for row in rows}),
    )


def _coverage_by_country(
    coverage: CoverageIndex,
    *,
    catalogue: CatalogueSnapshot,
    base: _BaseByCountry,
    source_of: Mapping[str, str],
    datasets: Collection[str],
    parties: Mapping[str, PartyRow],
) -> _CoverageByCountry:
    """Every country's coverage summary from six grouped passes over the
    index: each role by facet and by variable, the variables by facet, and
    the donor and recipient pairs by facet."""
    provenance_of = {
        facet_id: meta.provenance_id for facet_id, meta in coverage.facet_metadata.items()
    }
    # A declared dataset's facets carry member provenances; its rows count under its id.
    dataset_by_facet = {
        facet_id: dataset_of(provenance)
        for facet_id, provenance in provenance_of.items()
        if provenance is not None
    }
    aid_facets = frozenset(
        facet_id for facet_id, dataset in dataset_by_facet.items() if dataset in datasets
    )

    sources: dict[str, set[str]] = defaultdict(set)
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    aid_spans: list[tuple[int, int]] = []
    for role in _ROLES:
        by_facet = coverage.grouped(by=(role, "facet_id"))
        for place, facet, first, last in zip(
            _strings(by_facet[role]),
            _strings(by_facet[FACET_COLUMN]),
            _ints(by_facet[FIRST_YEAR_COLUMN]),
            _ints(by_facet[LAST_YEAR_COLUMN]),
            strict=True,
        ):
            if facet in aid_facets:
                aid_spans.append((first, last))
            if not place.startswith(COUNTRY_PREFIX):
                continue
            provenance = provenance_of.get(facet)
            if provenance is not None and provenance in source_of:
                sources[place].add(source_of[provenance])
            if facet in aid_facets:
                spans[place].append((first, last))

    # Pairs repeat a party across many rows, so each (party, role, dataset) is judged once.
    @functools.cache
    def counted(dcid: str, role: Role, dataset: str) -> bool:
        # A one-role variable's rows hold `ABSENT_ENTITY` on the role it lacks.
        if dcid == ABSENT_ENTITY:
            return False
        party = parties.get(dcid)
        return party is None or party.kind_over(COVERAGE_ROLE_AXES[role], (dataset,)) != "total"

    by_pair_facet = coverage.grouped(by=(*_ROLES, "facet_id"))
    donors: set[str] = set()
    recipients: set[str] = set()
    flows: set[tuple[str, str]] = set()
    for donor, recipient, facet in zip(
        _strings(by_pair_facet["donor"]),
        _strings(by_pair_facet["recipient"]),
        _strings(by_pair_facet[FACET_COLUMN]),
        strict=True,
    ):
        if facet not in aid_facets:
            continue
        dataset = dataset_by_facet[facet]
        donor_counted = counted(donor, "donor", dataset)
        recipient_counted = counted(recipient, "recipient", dataset)
        if donor_counted:
            donors.add(donor)
        if recipient_counted:
            recipients.add(recipient)
        if donor_counted and recipient_counted:
            flows.add((donor, recipient))

    covered: dict[_Group, set[str]] = defaultdict(set)
    by_variable_facet = coverage.grouped(by=("variable", "facet_id"))
    for variable, facet in zip(
        _strings(by_variable_facet["variable"]),
        _strings(by_variable_facet[FACET_COLUMN]),
        strict=True,
    ):
        provenance = provenance_of.get(facet)
        if provenance is None or not catalogue.is_admitted(provenance):
            continue
        origin = catalogue.tables.provenances[provenance].data_commons
        covered[_GROUP_OF[origin.id]].add(variable)

    variables, overlap = _variables_by_country(coverage, catalogue=catalogue, base=base)
    return _CoverageByCountry(
        covered={group: frozenset(covered[group]) for group in _GROUP_OF.values()},
        variables=variables,
        overlap=overlap,
        sources={place: frozenset(named) for place, named in sources.items()},
        years={place: years for place, held in spans.items() if (years := _years(held))},
        aid_donors=Counter(recipient for _, recipient in flows),
        aid_recipients=Counter(donor for donor, _ in flows),
        aid=AidCounts(donors=len(donors), recipients=len(recipients), years=_years(aid_spans)),
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


__all__ = ["COUNTRY_PREFIX", "CatalogueCounts", "catalogue_counts"]
