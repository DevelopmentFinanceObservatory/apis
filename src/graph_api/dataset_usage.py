"""How each Provenance uses the model, from the code index, the coverage index and the
Provenance's graph node.

`build_usage` makes one `ProvenanceUsage` per key of
`CodeIndex.variables_by_provenance`, so a Provenance appears when at least one StatVar row of
the index names it. Its `label` and `description` come from the Provenance's graph node, read
for `PROVENANCE_NODE_PROPERTIES`: the label is the node's display name
(`dc_graph.display_name`) and the description the least of its `description` values, as the
schema document labels its nodes. A Provenance the graph does not hold, or whose node carries no
name or no description, gets null for that field.

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
from collections.abc import Set as AbstractSet
from dataclasses import replace
from typing import NamedTuple, Protocol

from code_index import CodeIndex, StatVarRow
from code_index.query import row_five_tuple
from dc_graph import display_name
from graph_api.schema_models import (
    DatasetUsage,
    ProvenanceUsage,
    SchemaConstraintValues,
    SchemaFacets,
    SchemaPattern,
    SchemaValueCount,
    ShapeField,
)
from series_coverage.schema import FacetMetadata
from statvar_members import FiveTuple

EXAMPLES_PER_PATTERN = 3
"""How many StatVar dcids a pattern lists as examples."""

PROVENANCE_NODE_PROPERTIES: tuple[str, ...] = ("name", "description")
"""The arcs of each Provenance node `build_usage` reads its label and description from."""

type ProvenanceNodes = Mapping[str, Mapping[str, tuple[str, ...]] | None]
"""Each Provenance's `PROVENANCE_NODE_PROPERTIES` values, `None` for a node the graph does not
hold, as `GraphNodeReader.node_arcs_many` answers them."""


class ProvenanceCoverage(Protocol):
    """The two coverage-index reads the build uses, as `series_coverage.CoverageIndex` serves
    them."""

    @property
    def facet_metadata(self) -> Mapping[str, FacetMetadata]: ...

    @property
    def provenance_currencies(self) -> Mapping[str, frozenset[str]]: ...


SHAPE_FIELDS: tuple[ShapeField, ...] = (
    "population_type",
    "measured_property",
    "stat_type",
    "measurement_qualifier",
    "measurement_denominator",
)
"""The StatVar shape fields in `FiveTuple`'s order, each a `FiveTuple` field, a `StatVarRow`
field of the same name and a field of `SchemaPattern`."""


class _PatternKey(NamedTuple):
    shape: FiveTuple
    constraint_properties: tuple[str, ...]
    observation_properties: tuple[str, ...]


def _pattern_order(entry: tuple[_PatternKey, list[str]]) -> tuple[object, ...]:
    """Count descending, then the fields, with a missing field before any value and a resolved
    denominator before an unresolved one."""
    key, dcids = entry
    return (
        -len(dcids),
        *(getattr(key.shape, name) or "" for name in SHAPE_FIELDS),
        key.shape.denominator_unresolved,
        key.constraint_properties,
        key.observation_properties,
    )


def group_patterns(
    rows: Iterable[StatVarRow], *, unresolved_denominators: AbstractSet[str] | None = None
) -> list[SchemaPattern]:
    """The patterns of `rows`, which arrive sorted by dcid, so each pattern's dcids are too.

    `unresolved_denominators` holds the dcids of the rows whose `measurementDenominator` arc
    is unresolved, which keys their patterns apart from those of the rows it leaves out. When
    it is `None`, as for code-index rows, which hold resolved denominators alone, each pattern's
    `measurement_denominator_unresolved` is `None` and the body leaves it out."""
    members: dict[_PatternKey, list[str]] = defaultdict(list)
    for row in rows:
        shape = row_five_tuple(row)
        if unresolved_denominators is not None and row.dcid in unresolved_denominators:
            shape = replace(shape, denominator_unresolved=True)
        key = _PatternKey(
            shape=shape,
            constraint_properties=tuple(sorted(row.constraint_properties)),
            observation_properties=tuple(sorted(row.observation_properties)),
        )
        members[key].append(row.dcid)
    return [
        SchemaPattern(
            **{name: getattr(key.shape, name) for name in SHAPE_FIELDS},
            measurement_denominator_unresolved=(
                None if unresolved_denominators is None else key.shape.denominator_unresolved
            ),
            constraint_properties=list(key.constraint_properties),
            observation_properties=list(key.observation_properties),
            stat_var_count=len(dcids),
            examples=dcids[:EXAMPLES_PER_PATTERN],
        )
        for key, dcids in sorted(members.items(), key=_pattern_order)
    ]


def count_constraint_values(rows: Iterable[StatVarRow]) -> list[SchemaConstraintValues]:
    """Each constraint property `rows` set, with how many rows take each of its values."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for prop, value in row.constraint_properties.items():
            counts[prop][value] += 1
    return [
        SchemaConstraintValues(
            property=prop,
            values=[
                SchemaValueCount(value=value, stat_var_count=count)
                for value, count in sorted(counts[prop].items())
            ],
        )
        for prop in sorted(counts)
    ]


def _facets_by_provenance(coverage: ProvenanceCoverage) -> dict[str, SchemaFacets]:
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
        provenance: SchemaFacets(
            units=sorted(units[provenance]),
            scaling_factors=sorted(scaling[provenance]),
            measurement_methods=sorted(methods[provenance]),
            observation_periods=sorted(periods[provenance]),
            currencies=sorted(currencies.get(provenance, ())),
        )
        for provenance in held
    }


def empty_facets() -> SchemaFacets:
    """Facets with every list empty, for a Provenance coverage holds nothing for."""
    return SchemaFacets(
        units=[], scaling_factors=[], measurement_methods=[], observation_periods=[], currencies=[]
    )


def build_usage(
    *, code_index: CodeIndex, coverage: ProvenanceCoverage, nodes: ProvenanceNodes
) -> DatasetUsage:
    """How each Provenance of `code_index` uses the model, with facet facts from `coverage` and
    the label and description of its node in `nodes`."""
    facets = _facets_by_provenance(coverage)
    provenances: list[ProvenanceUsage] = []
    for provenance, dcids in sorted(code_index.variables_by_provenance.items()):
        rows = [row for dcid in dcids if (row := code_index.statvar_by_dcid(dcid)) is not None]
        node = nodes.get(provenance) or {}
        provenances.append(
            ProvenanceUsage(
                ref=provenance,
                label=display_name(node.get("name", ()), provenance),
                description=min(node.get("description", ()), default=None),
                stat_var_count=len(rows),
                patterns=group_patterns(rows),
                constraint_values=count_constraint_values(rows),
                facets=facets.get(provenance) or empty_facets(),
            )
        )
    return DatasetUsage(provenances=provenances)


__all__ = [
    "EXAMPLES_PER_PATTERN",
    "PROVENANCE_NODE_PROPERTIES",
    "SHAPE_FIELDS",
    "ProvenanceCoverage",
    "ProvenanceNodes",
    "build_usage",
    "count_constraint_values",
    "empty_facets",
    "group_patterns",
]
