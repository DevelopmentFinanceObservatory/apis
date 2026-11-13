"""How each Provenance uses the model, from the code index and the coverage index alone.

`build_provenances` makes one `OntologyProvenance` per key of
`CodeIndex.variables_by_provenance`, so a Provenance appears when at least one StatVar row of
the index names it. Its `label` and `description` come from the Provenance's
`DatasetDeclaration` in `CodeIndex.datasets`.

A pattern is a distinct combination, over the Provenance's StatVar rows, of the five shape
fields, the sorted names of the constraint properties and the sorted observation properties.
A StatVar that two Provenances publish counts under both.

`facets` reads the coverage index's `facet_metadata` entries whose `provenance_id` is the
Provenance, and `currencies` reads `provenance_currencies`. A Provenance with no coverage
entry gets empty lists.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import fields
from typing import NamedTuple, Protocol

from code_index import CodeIndex, StatVarRow
from code_index.query import row_five_tuple
from graph_api.ontology_models import (
    OntologyConstraintValues,
    OntologyFacets,
    OntologyPattern,
    OntologyProvenance,
    OntologyProvenances,
    OntologyValueCount,
)
from series_coverage.schema import FacetMetadata
from statvar_members import FiveTuple

EXAMPLES_PER_PATTERN = 3
"""How many StatVar dcids a pattern lists as examples."""


class ProvenanceCoverage(Protocol):
    """The two coverage-index reads the build uses, as `series_coverage.CoverageIndex` serves
    them."""

    @property
    def facet_metadata(self) -> Mapping[str, FacetMetadata]: ...

    @property
    def provenance_currencies(self) -> Mapping[str, frozenset[str]]: ...


SHAPE_FIELDS: tuple[str, ...] = tuple(
    field.name for field in fields(FiveTuple) if field.name != "denominator_unresolved"
)
"""The StatVar shape fields in `FiveTuple`'s order, each a `StatVarRow` field of the same name
and a field of `OntologyPattern`."""


class _PatternKey(NamedTuple):
    shape: FiveTuple
    constraint_properties: tuple[str, ...]
    observation_properties: tuple[str, ...]


def _pattern_key(row: StatVarRow) -> _PatternKey:
    return _PatternKey(
        shape=row_five_tuple(row),
        constraint_properties=tuple(sorted(row.constraint_properties)),
        observation_properties=tuple(sorted(row.observation_properties)),
    )


def _pattern_order(entry: tuple[_PatternKey, list[str]]) -> tuple[object, ...]:
    """Count descending, then the fields, with a missing field before any value."""
    key, dcids = entry
    return (
        -len(dcids),
        *(getattr(key.shape, name) or "" for name in SHAPE_FIELDS),
        key.constraint_properties,
        key.observation_properties,
    )


def _patterns(rows: Iterable[StatVarRow]) -> list[OntologyPattern]:
    """The patterns of `rows`, which arrive sorted by dcid, so each pattern's dcids are too."""
    members: dict[_PatternKey, list[str]] = defaultdict(list)
    for row in rows:
        members[_pattern_key(row)].append(row.dcid)
    return [
        OntologyPattern(
            **{name: getattr(key.shape, name) for name in SHAPE_FIELDS},
            constraint_properties=list(key.constraint_properties),
            observation_properties=list(key.observation_properties),
            stat_var_count=len(dcids),
            examples=dcids[:EXAMPLES_PER_PATTERN],
        )
        for key, dcids in sorted(members.items(), key=_pattern_order)
    ]


def _constraint_values(rows: Iterable[StatVarRow]) -> list[OntologyConstraintValues]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for prop, value in row.constraint_properties.items():
            counts[prop][value] += 1
    return [
        OntologyConstraintValues(
            property=prop,
            values=[
                OntologyValueCount(value=value, stat_var_count=count)
                for value, count in sorted(counts[prop].items())
            ],
        )
        for prop in sorted(counts)
    ]


def _facets_by_provenance(coverage: ProvenanceCoverage) -> dict[str, OntologyFacets]:
    units: dict[str, set[str]] = defaultdict(set)
    scaling: dict[str, set[str]] = defaultdict(set)
    methods: dict[str, set[str]] = defaultdict(set)
    periods: dict[str, set[str]] = defaultdict(set)
    held: set[str] = set(coverage.provenance_currencies)
    for metadata in coverage.facet_metadata.values():
        provenance = metadata.provenance_id
        if provenance is None:
            continue
        held.add(provenance)
        for field, values in (
            (metadata.unit, units),
            (metadata.scaling_factor, scaling),
            (metadata.measurement_method, methods),
            (metadata.observation_period, periods),
        ):
            if field is not None:
                values[provenance].add(field)
    currencies = coverage.provenance_currencies
    return {
        provenance: OntologyFacets(
            units=sorted(units[provenance]),
            scaling_factors=sorted(scaling[provenance]),
            measurement_methods=sorted(methods[provenance]),
            observation_periods=sorted(periods[provenance]),
            currencies=sorted(currencies.get(provenance, ())),
        )
        for provenance in held
    }


def _no_facets() -> OntologyFacets:
    return OntologyFacets(
        units=[], scaling_factors=[], measurement_methods=[], observation_periods=[], currencies=[]
    )


def build_provenances(
    *, code_index: CodeIndex, coverage: ProvenanceCoverage
) -> OntologyProvenances:
    """How each Provenance of `code_index` uses the model, with facet facts from `coverage`."""
    facets = _facets_by_provenance(coverage)
    provenances: list[OntologyProvenance] = []
    for provenance, dcids in sorted(code_index.variables_by_provenance.items()):
        rows = [row for dcid in dcids if (row := code_index.statvar_by_dcid(dcid)) is not None]
        declaration = code_index.datasets.get(provenance)
        provenances.append(
            OntologyProvenance(
                dcid=provenance,
                label=declaration.label if declaration is not None else None,
                description=declaration.description if declaration is not None else None,
                stat_var_count=len(rows),
                patterns=_patterns(rows),
                constraint_values=_constraint_values(rows),
                facets=facets.get(provenance) or _no_facets(),
            )
        )
    return OntologyProvenances(provenances=provenances)


__all__ = ["EXAMPLES_PER_PATTERN", "SHAPE_FIELDS", "ProvenanceCoverage", "build_provenances"]
