"""How each base and UN System Provenance uses the model, from the resident origin catalogue.

`build_origin_usage` makes the `DatasetUsage` body of one outside origin with no graph read, in
the shape and with the grouping `graph_api.dataset_usage.build_usage` gives the DFO body. It
reads the unscoped catalogue: a request's source scope changes none of its answers.

Admission follows the catalogue's rules:

- base: every variable of `CatalogueTables.base_variables`, under each of its provenances
  `Catalogue.is_admitted` admits. The table holds only the provenance rows the registry leaves
  in, so an excluded provenance has no entry, and no superseded or withheld variable, which
  `CatalogueTables` keeps in `withheld_variables`.
- UN System: every variable of the UN tables, under its agency's provenance rows
  (`Catalogue.variable_provenances`). The tables hold no variable of a series the registry
  excludes.

A variable's StatVar row reads its arcs (`ArcTables.arcs` for base, `UnVariable.arcs` for UN)
through `origin_catalogue.families.VariableArcs`, as a family key does: a property's first
literal, else its first node. A `measurementQualifier` whose every target is a
`ProvisionalNode` reads as none, and a missing `measuredProperty` as the empty string. A
`measurementDenominator` with targets but no value, or whose every target is a
`ProvisionalNode`, is unresolved: the pattern keeps the target it names, null for none, and
sets `measurement_denominator_unresolved`, so it stays apart from a pattern with no
denominator. The constraint properties are the `constraintProperties` targets less the
`import_added` ones. A declared constraint whose value the arcs do not resolve counts in its
pattern's `constraint_properties` and adds no row to `constraint_values`. A base variable the
arc tables do not hold has no row.

`label` is the provenance row's name and `description` the `CardFacts.description` of the row's
source-index entry (`entry_ref`). UN rows carry no card facts, so a UN description is null.
`facets` is empty for every provenance: the coverage index covers DFO series alone.

`terms` lists every node the patterns and constraint values name, with the display name
(`dc_graph.display_name`) and the types the variables' arcs to it carry. UN series and the base
terms the DFO schema document leaves out have no other label source.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from code_index import StatVarRow
from dc_graph import display_name
from graph_api.dataset_usage import count_constraint_values, empty_facets, group_patterns
from graph_api.schema_models import DatasetUsage, ProvenanceUsage, UsageTerm
from graph_generation import GenerationStamp, RegistrationHandle
from graph_generation.registry import CacheRegistry
from origin_catalogue import CatalogueSnapshot, current_catalogue, entry_ref
from origin_catalogue.families import VariableArcs
from origin_catalogue.origins import UN_SYSTEM
from origin_catalogue.un_mirror import UnArc

type UsageOrigin = Literal["base", "un_system"]
"""An origin with its own usage document."""

_TERM_PROPERTIES = (
    "populationType",
    "measuredProperty",
    "statType",
    "measurementQualifier",
    "measurementDenominator",
    "constraintProperties",
    "observationProperties",
)
"""The arcs whose targets a pattern names, beside each declared constraint's value arc."""


class CatalogueGeneration:
    """A generation provider whose stamp is the bound origin catalogue's build, so a cache on
    it keeps an entry until a request bound to another build looks it up or the stamp's
    maximum age passes. `stamp` raises `CatalogueNotBound` when no catalogue is bound."""

    def __init__(self) -> None:
        self._registry = CacheRegistry()

    def stamp(self) -> GenerationStamp:
        return GenerationStamp(token=current_catalogue().build_id)

    def register(
        self,
        name: str,
        *,
        stats: Callable[[], Mapping[str, int]],
        clear: Callable[[], None],
    ) -> RegistrationHandle:
        return self._registry.register(name, stats=stats, clear=clear)

    async def refresh_if_stale(self, *, timeout_s: float = 1.0) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _Read:
    """One variable as the document counts it: `pattern` is its StatVar row, `values` the
    same row with only the constraints whose value the arcs resolve, which is the row the
    constraint values count, and `targets` the arcs naming its pattern's nodes."""

    pattern: StatVarRow
    values: StatVarRow
    denominator_unresolved: bool
    targets: tuple[UnArc, ...]


def _read(dcid: str, arcs: tuple[UnArc, ...]) -> _Read:
    read = VariableArcs(arcs)
    declared = sorted(set(read.declared_constraints()))
    values = {prop: read.value(prop) for prop in declared}
    resolved = {prop: found for prop, found in values.items() if found is not None}
    row = StatVarRow(
        dcid=dcid,
        population_type=read.value("populationType"),
        measured_property=read.value("measuredProperty") or "",
        stat_type=read.value("statType"),
        measurement_qualifier=read.qualifier(),
        measurement_denominator=read.value("measurementDenominator"),
        observation_properties=tuple(
            sorted(arc.dcid for arc in read.rows("observationProperties") if arc.dcid is not None)
        ),
        # The pattern reads only the names, so an unresolved constraint keeps its place there.
        constraint_properties={prop: found or "" for prop, found in values.items()},
    )
    return _Read(
        pattern=row,
        values=row
        if len(resolved) == len(values)
        else replace(row, constraint_properties=resolved),
        denominator_unresolved=read.denominator_unresolved(),
        targets=tuple(
            arc
            for prop in (*_TERM_PROPERTIES, *declared)
            for arc in read.rows(prop)
            if arc.dcid is not None
        ),
    )


type _Grouped = dict[str, list[_Read]]


def _group_base(catalogue: CatalogueSnapshot) -> _Grouped:
    grouped: _Grouped = defaultdict(list)
    for dcid, variable in sorted(catalogue.tables.base_variables.items()):
        provenances = [p for p in variable.provenances if catalogue.is_admitted(p)]
        arcs = catalogue.arcs.arcs(dcid) if provenances else None
        if not arcs:
            continue
        read = _read(dcid, arcs)
        for provenance in provenances:
            grouped[provenance].append(read)
    return grouped


def _group_un(catalogue: CatalogueSnapshot) -> _Grouped:
    grouped: _Grouped = defaultdict(list)
    for dcid, variable in sorted(catalogue.un.mirror.variables.items()):
        provenances = [
            row.id for row in catalogue.variable_provenances(dcid) if catalogue.is_admitted(row.id)
        ]
        if not provenances:
            continue
        read = _read(dcid, variable.arcs)
        for provenance in provenances:
            grouped[provenance].append(read)
    return grouped


@dataclass(slots=True)
class _Term:
    names: list[str] = field(default_factory=list)
    types: set[str] = field(default_factory=set)


def _named_refs(usage: ProvenanceUsage) -> Iterable[str | None]:
    for pattern in usage.patterns:
        yield pattern.population_type
        yield pattern.measured_property
        yield pattern.stat_type
        yield pattern.measurement_qualifier
        yield pattern.measurement_denominator
        yield from pattern.constraint_properties
        yield from pattern.observation_properties
    for entry in usage.constraint_values:
        yield entry.property
        yield from (value.value for value in entry.values)


def _terms(provenances: list[ProvenanceUsage], reads: Iterable[_Read]) -> list[UsageTerm]:
    """Each node `provenances` name that an arc of `reads` targets, labelled from those arcs."""
    named = {ref for usage in provenances for ref in _named_refs(usage) if ref}
    found: dict[str, _Term] = defaultdict(_Term)
    for read in reads:
        for arc in read.targets:
            if arc.dcid in named:
                term = found[arc.dcid]
                if arc.name:
                    term.names.append(arc.name)
                term.types.update(arc.types)
    return [
        UsageTerm(ref=ref, label=display_name(term.names, ref), types=sorted(term.types))
        for ref, term in sorted(found.items())
    ]


def build_origin_usage(catalogue: CatalogueSnapshot, origin: UsageOrigin) -> DatasetUsage:
    """How each admitted provenance of `origin` in `catalogue` uses the model, one entry per
    provenance holding at least one admitted variable, sorted by ref, and the terms they name."""
    grouped = _group_base(catalogue) if origin == "base" else _group_un(catalogue)
    provenances: list[ProvenanceUsage] = []
    for ref, reads in sorted(grouped.items()):
        row = catalogue.provenance(ref)
        facts = (
            catalogue.dataset_facts(entry_ref(row))
            if row is not None and row.data_commons != UN_SYSTEM
            else None
        )
        provenances.append(
            ProvenanceUsage(
                ref=ref,
                label=row.name or None if row is not None else None,
                description=facts.description if facts is not None else None,
                stat_var_count=len(reads),
                patterns=group_patterns(
                    (read.pattern for read in reads),
                    unresolved_denominators={
                        read.pattern.dcid for read in reads if read.denominator_unresolved
                    },
                ),
                constraint_values=count_constraint_values(read.values for read in reads),
                facets=empty_facets(),
            )
        )
    reads = (read for group in grouped.values() for read in group)
    return DatasetUsage(provenances=provenances, terms=_terms(provenances, reads))


__all__ = ["CatalogueGeneration", "UsageOrigin", "build_origin_usage"]
