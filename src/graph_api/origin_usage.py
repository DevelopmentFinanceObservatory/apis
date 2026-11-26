"""How each base and UN System Provenance uses the model, from the resident origin catalogue.

`build_origin_usage` makes the `DatasetUsage` body of one outside origin with no graph read, in
the shape and with the grouping `graph_api.dataset_usage.build_usage` gives the DFO body. It
reads the unscoped catalogue, so a request's source scope changes none of its answers.

Admission follows the catalogue's rules:

- base: every variable of `CatalogueTables.base_variables`, under each of its provenances
  `Catalogue.is_admitted` admits. The table holds only the provenance rows the registry leaves
  in, so an excluded provenance has no entry, and no superseded or withheld variable, which
  `CatalogueTables` keeps in `withheld_variables`.
- UN System: every variable of the UN tables, under its agency's provenance rows
  (`Catalogue.variable_provenances`), read once per agency id segment. The tables hold no
  variable of a series the registry excludes.

A base variable's arcs are `Catalogue.held_arcs`: the bound graph release's for a variable the
release holds as DFO, else the arc tables', less the build's own DFO rows, which give way to the
release (`origin_catalogue.catalogue.release_held_arcs`). A UN variable's are
`UnVariable.arcs`. A StatVar row reads them through `origin_catalogue.families.VariableArcs`, as
a family key does. A property reads as its first literal, else its first node. A
`measurementQualifier` whose every target is a `ProvisionalNode` reads as none, and a missing
`measuredProperty` as the empty string. A `measurementDenominator` with targets but no value,
or whose every target is a `ProvisionalNode`, is unresolved. The pattern keeps the target it names, null for none, and
sets `measurement_denominator_unresolved`, so it stays apart from a pattern with no
denominator. The constraint properties are the `constraintProperties` targets less the
`import_added` ones. A declared constraint whose value the arcs do not resolve counts in its
pattern's `constraint_properties` and adds no row to `constraint_values`. A base variable
neither the release nor the arc tables hold has no row.

`label` is the provenance row's name and `description` the `CardFacts.description` of the row's
source-index entry (`entry_ref`), null where the entry's graph states none.
`facets` is empty for every provenance, since the coverage index covers DFO series alone.

`terms` lists every node the patterns and constraint values name, with the types the
variables' arcs to it carry and the label `GET /api/graph/node/{dcid}` gives it, the
`dc_graph.display_name` of its names (`_term_label`): a node the catalogue holds reads its own
`name` arcs, a node an arc of the arc tables names reads `Catalogue.arc_label`, and any other
node, such as a UN series, reads the names the variables' arcs carry. A name equal to the dcid
is no label.

The body depends on the catalogue build and, through `held_arcs`, on the bound release, so the
door keys it on both (`OriginUsageKey`, `CatalogueGeneration`).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, NamedTuple

from code_index import StatVarRow
from dc_graph import display_name
from graph_api.dataset_usage import count_constraint_values, empty_facets, group_patterns
from graph_api.schema_models import DatasetUsage, ProvenanceUsage, UsageTerm
from graph_generation import GenerationStamp, RegistrationHandle
from graph_generation.registry import CacheRegistry
from origin_catalogue import CatalogueSnapshot, current_catalogue, entry_ref
from origin_catalogue.families import VariableArcs
from origin_catalogue.origins import UN_ID_PREFIX
from origin_catalogue.un_mirror import UnArc

type UsageOrigin = Literal["base", "un_system"]
"""An origin with its own usage document."""


class OriginUsageKey(NamedTuple):
    """The origin of a usage body and the id of the graph release it read
    (`graph_release.context.served_origin_key`). The cache stamps it with the catalogue
    build."""

    origin: UsageOrigin
    release: str


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
    """A generation provider whose stamp is the bound origin catalogue's build, so a request
    bound to another build misses every entry stored under this one. Each entry lives until the
    stamp's maximum age passes or the cache's LRU bound evicts it. `stamp` raises
    `CatalogueNotBound` when no catalogue is bound."""

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
        arcs = catalogue.held_arcs(dcid) if provenances else None
        if not arcs:
            continue
        read = _read(dcid, arcs)
        for provenance in provenances:
            grouped[provenance].append(read)
    return grouped


def _group_un(catalogue: CatalogueSnapshot) -> _Grouped:
    """The UN variables under their agency's rows. `variable_provenances` scans every
    provenance row and answers alike for every variable of one id segment, so it is read once
    per segment."""
    grouped: _Grouped = defaultdict(list)
    by_segment: dict[str, list[str]] = {}
    for dcid, variable in sorted(catalogue.un.mirror.variables.items()):
        segment = dcid.removeprefix(UN_ID_PREFIX).split("/", 1)[0]
        provenances = by_segment.get(segment)
        if provenances is None:
            provenances = by_segment[segment] = [
                row.id
                for row in catalogue.variable_provenances(dcid)
                if catalogue.is_admitted(row.id)
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


def _term_label(catalogue: CatalogueSnapshot, ref: str, arc_names: Iterable[str]) -> str | None:
    """`ref`'s `display_name`, as `GET /api/graph/node/{dcid}` labels it: from its own `name`
    arcs where the catalogue holds the node, from `arc_label` where an arc of the arc tables
    names it, and from `arc_names`, the names the variables' arcs to it carry, otherwise.
    `arc_label` is `GraphClient.node_label`'s answer, `label_or_self_name`, so its self-name
    reads as none."""
    variable = catalogue.un_variable(ref)
    own = variable.arcs if variable is not None else catalogue.held_arcs(ref)
    if own is not None:
        return display_name((arc.value for arc in own if arc.property == "name" and arc.value), ref)
    if catalogue.arc_target(ref) is not None:
        label = catalogue.arc_label(ref)
        return None if label == ref else label
    return display_name(arc_names, ref)


def _terms(
    catalogue: CatalogueSnapshot, provenances: list[ProvenanceUsage], reads: Iterable[_Read]
) -> list[UsageTerm]:
    """Each node `provenances` name that an arc of `reads` targets, with the types those arcs
    carry and its `_term_label`."""
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
        UsageTerm(ref=ref, label=_term_label(catalogue, ref, term.names), types=sorted(term.types))
        for ref, term in sorted(found.items())
    ]


def build_origin_usage(catalogue: CatalogueSnapshot, origin: UsageOrigin) -> DatasetUsage:
    """How each admitted provenance of `origin` in `catalogue` uses the model, one entry per
    provenance holding at least one admitted variable, sorted by ref, and the terms they name.
    Reads the bound graph release, and raises `graph_release.context.ReleaseNotBound` with none
    bound."""
    grouped = _group_base(catalogue) if origin == "base" else _group_un(catalogue)
    provenances: list[ProvenanceUsage] = []
    for ref, reads in sorted(grouped.items()):
        row = catalogue.provenance(ref)
        facts = catalogue.dataset_facts(entry_ref(row)) if row is not None else None
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
    return DatasetUsage(provenances=provenances, terms=_terms(catalogue, provenances, reads))


__all__ = ["CatalogueGeneration", "OriginUsageKey", "UsageOrigin", "build_origin_usage"]
