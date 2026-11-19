"""How each base and UN System Provenance uses the model, from the resident origin catalogue.

`build_origin_usage` makes the `DatasetUsage` body of one outside origin with no graph read, in
the shape and with the grouping `graph_api.dataset_usage.build_usage` gives the DFO body.

Admission follows the catalogue's rules:

- base: every variable of `CatalogueTables.base_variables` that `Catalogue.is_superseded` does
  not withhold, under each of its provenances `Catalogue.admits_in_scope` admits. The table
  holds only the provenance rows the registry leaves in, so an excluded provenance has no entry.
- UN System: every variable of the UN tables that `Catalogue.un_variable` returns and
  `Catalogue.admit_candidate` admits as UN, under its agency's provenance rows
  (`Catalogue.variable_provenances`). The tables hold no variable of a series the registry
  excludes.

A variable's StatVar row reads its arcs (`ArcTables.arcs` for base, `UnVariable.arcs` for UN)
as `origin_catalogue.families` reads a five-tuple: a property's first literal, else its first
node. A `measurementQualifier` or `measurementDenominator` whose every target is a
`ProvisionalNode` reads as none, and a missing `measuredProperty` as the empty string. The
constraint properties are the `constraintProperties` targets less the `import_added` ones. A
declared constraint whose value the arcs do not resolve counts in its pattern's
`constraint_properties` and adds no row to `constraint_values`. A base variable the arc tables
do not hold has no row.

`label` is the provenance row's name and `description` the `CardFacts.description` of the row's
source-index entry (`entry_ref`). UN rows carry no card facts, so a UN description is null.
`facets` is empty for every provenance: the coverage index covers DFO series alone.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Final, Literal

from code_index import StatVarRow
from graph_api.dataset_usage import _constraint_values, _no_facets, _patterns
from graph_api.schema_models import DatasetUsage, ProvenanceUsage
from graph_generation import GenerationStamp, RegistrationHandle
from graph_generation.registry import CacheRegistry
from origin_catalogue import CatalogueSnapshot, current_catalogue, entry_ref
from origin_catalogue.origins import UN_SYSTEM
from origin_catalogue.un_mirror import UnArc, import_added

type UsageOrigin = Literal["base", "un_system"]
"""An origin with its own usage document."""

_PROVISIONAL_TYPES: Final[tuple[str, ...]] = ("ProvisionalNode",)


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


def _rows(dcid: str, arcs: tuple[UnArc, ...]) -> tuple[StatVarRow, StatVarRow]:
    """`dcid`'s StatVar row read from `arcs`, and the same row with only the constraints
    whose value the arcs resolve, which is the row the constraint values count."""
    by_property: dict[str, list[UnArc]] = defaultdict(list)
    for arc in arcs:
        by_property[arc.property].append(arc)

    def value(prop: str) -> str | None:
        rows = by_property.get(prop, ())
        found = next((arc for arc in rows if arc.value is not None), None) or next(
            (arc for arc in rows if arc.dcid is not None), None
        )
        if found is None:
            return None
        return found.value if found.value is not None else found.dcid

    def unless_provisional(prop: str) -> str | None:
        rows = by_property.get(prop, ())
        if rows and all(arc.types == _PROVISIONAL_TYPES for arc in rows):
            return None
        return value(prop)

    declared = sorted(
        {
            arc.dcid
            for arc in by_property.get("constraintProperties", ())
            if arc.dcid is not None and not import_added(arc)
        }
    )
    values = {prop: value(prop) for prop in declared}
    resolved = {prop: found for prop, found in values.items() if found is not None}
    row = StatVarRow(
        dcid=dcid,
        population_type=value("populationType"),
        measured_property=value("measuredProperty") or "",
        stat_type=value("statType"),
        measurement_qualifier=unless_provisional("measurementQualifier"),
        measurement_denominator=unless_provisional("measurementDenominator"),
        observation_properties=tuple(
            sorted(
                arc.dcid
                for arc in by_property.get("observationProperties", ())
                if arc.dcid is not None
            )
        ),
        # The pattern reads only the names, so an unresolved constraint keeps its place there.
        constraint_properties={prop: found or "" for prop, found in values.items()},
    )
    if len(resolved) == len(values):
        return row, row
    return row, replace(row, constraint_properties=resolved)


type _Grouped = dict[str, list[tuple[StatVarRow, StatVarRow]]]


def _group_base(catalogue: CatalogueSnapshot) -> _Grouped:
    grouped: _Grouped = defaultdict(list)
    for dcid, variable in sorted(catalogue.tables.base_variables.items()):
        if catalogue.is_superseded(dcid):
            continue
        provenances = [p for p in variable.provenances if catalogue.admits_in_scope(p)]
        arcs = catalogue.arcs.arcs(dcid) if provenances else None
        if not arcs:
            continue
        rows = _rows(dcid, arcs)
        for provenance in provenances:
            grouped[provenance].append(rows)
    return grouped


def _group_un(catalogue: CatalogueSnapshot) -> _Grouped:
    grouped: _Grouped = defaultdict(list)
    for dcid in sorted(catalogue.un.mirror.variables):
        variable = catalogue.un_variable(dcid)
        if variable is None or catalogue.admit_candidate(dcid) != UN_SYSTEM:
            continue
        provenances = [
            row.id
            for row in catalogue.variable_provenances(dcid)
            if catalogue.admits_in_scope(row.id)
        ]
        if not provenances:
            continue
        rows = _rows(dcid, variable.arcs)
        for provenance in provenances:
            grouped[provenance].append(rows)
    return grouped


def build_origin_usage(catalogue: CatalogueSnapshot, origin: UsageOrigin) -> DatasetUsage:
    """How each admitted provenance of `origin` in `catalogue` uses the model, one entry per
    provenance holding at least one admitted variable, sorted by ref."""
    grouped = _group_base(catalogue) if origin == "base" else _group_un(catalogue)
    provenances: list[ProvenanceUsage] = []
    for ref, pairs in sorted(grouped.items()):
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
                stat_var_count=len(pairs),
                patterns=_patterns(pattern_row for pattern_row, _ in pairs),
                constraint_values=_constraint_values(value_row for _, value_row in pairs),
                facets=_no_facets(),
            )
        )
    return DatasetUsage(provenances=provenances)


__all__ = ["CatalogueGeneration", "UsageOrigin", "build_origin_usage"]
