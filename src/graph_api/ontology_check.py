"""Check candidate StatisticalVariable definitions against the instance's ontology.

`check_rows` answers each node a row names from the ontology document, and reads the nodes
the document does not hold in one batched `node_arcs_many` per level of `subClassOf`
ancestry, so an id costs at most one read however many rows name it. A node the read answers
`None` for is not held; a read error propagates, so no node is reported absent because a read
failed.

Every row gets existence and kind checks. A constraint value must name a resolved node, by
`code_index.constraint_values.constraint_value_fault`, the test the code index export refuses
a release on. Range and domain rules apply only to a property the document declares
`declared_by == "instance"`: a value is inside the range when one of its types reaches a
`rangeIncludes` class by `subClassOf`, and a property is inside its domain when the row's
population type reaches a `domainIncludes` class the same way. A property with no
`rangeIncludes` or no `domainIncludes` admits anything. The domain rule covers the measured
property and the constraint properties, and is skipped when the row names no population type.
A row whose `dcid` the code index serves is compared with the served definition field by field.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from code_index import CodeIndex, StatVarRow
from code_index.constraint_values import constraint_value_fault
from graph_api.ontology_document import Ontology
from graph_api.ontology_models import (
    ConstraintValue,
    FindingCode,
    OntologyCheckRow,
    OntologyFinding,
    OntologyNode,
)
from graph_api.reader import GraphNodeReader

_CLASS: Final[str] = "Class"
_PROPERTY: Final[str] = "Property"
_READ_PROPERTIES: Final[tuple[str, ...]] = ("typeOf", "subClassOf")
_SHAPE_FIELDS: Final[tuple[str, ...]] = (
    "population_type",
    "measured_property",
    "stat_type",
    "measurement_qualifier",
    "measurement_denominator",
)
_CONSTRAINT: Final[str] = "constraint_properties"
_OBSERVATION: Final[str] = "observation_properties"


@dataclass(frozen=True)
class _Node:
    types: frozenset[str]
    parents: tuple[str, ...]
    instance: bool = False
    domain: tuple[str, ...] = ()
    range: tuple[str, ...] = ()


class _Nodes:
    """The nodes the rows name: document nodes free, the rest read once each."""

    def __init__(
        self, document: Iterable[OntologyNode], *, reader: GraphNodeReader, deadline: float
    ) -> None:
        self._known: dict[str, _Node | None] = {
            node.dcid: _Node(
                types=frozenset(node.types),
                parents=tuple(node.sub_class_of),
                instance=node.declared_by == "instance",
                domain=tuple(node.domain_includes),
                range=tuple(node.range_includes),
            )
            for node in document
        }
        self._reader = reader
        self._deadline = deadline

    def read(self, dcids: Iterable[str]) -> None:
        """Read in one batch every one of `dcids` not yet known."""
        unknown = sorted({dcid for dcid in dcids if dcid not in self._known})
        if not unknown:
            return
        if "" in unknown:
            self._known[""] = None
            unknown.remove("")
            if not unknown:
                return
        arcs = self._reader.node_arcs_many(
            unknown, properties=_READ_PROPERTIES, deadline=self._deadline
        )
        for dcid in unknown:
            entry = arcs.get(dcid)
            self._known[dcid] = (
                None
                if entry is None
                else _Node(
                    types=frozenset(entry.get("typeOf", ())),
                    parents=tuple(entry.get("subClassOf", ())),
                )
            )

    def read_ancestry(self, roots: Collection[str]) -> None:
        """Read `roots` and their `subClassOf` ancestors, one batch per level."""
        seen: set[str] = set()
        level = set(roots)
        while level:
            self.read(level)
            seen |= level
            level = {
                parent
                for dcid in level
                if (node := self._known[dcid]) is not None
                for parent in node.parents
            } - seen

    def get(self, dcid: str) -> _Node | None:
        return self._known[dcid]

    def ancestry(self, dcids: Iterable[str]) -> set[str]:
        """`dcids` and every `subClassOf` ancestor of them, over the nodes read."""
        seen: set[str] = set()
        stack = list(dcids)
        while stack:
            dcid = stack.pop()
            if dcid in seen:
                continue
            seen.add(dcid)
            if (node := self._known.get(dcid)) is not None:
                stack.extend(node.parents)
        return seen


def check_rows(
    rows: Sequence[OntologyCheckRow],
    *,
    ontology: Ontology,
    reader: GraphNodeReader,
    code_index: CodeIndex,
    deadline: float,
) -> list[OntologyFinding]:
    """Find the faults in candidate StatVar definitions.

    Args:
        rows: the candidate definitions.
        ontology: the built ontology, whose document answers most lookups with no read.
        reader: the graph reads for nodes the document does not hold.
        code_index: the bound code index, whose StatVar rows the rows' dcids are compared with.
        deadline: the absolute monotonic deadline passed to every read.

    Returns:
        One finding per fault, in row order. Within a row: the shape fields in the order
        `population_type`, `measured_property`, `stat_type`, `measurement_qualifier`,
        `measurement_denominator`; the constraint properties sorted, each property's finding
        before its value's; the observation properties in row order; then `definition_differs`
        findings in the same field order.

    Raises:
        GraphReadError: a read failed.
    """
    nodes = _Nodes(ontology.document.nodes, reader=reader, deadline=deadline)
    nodes.read(dcid for row in rows for dcid in _named(row))
    roots: set[str] = set()
    for row in rows:
        if row.population_type is not None:
            roots.add(row.population_type)
        for prop, value in row.constraint_properties.items():
            if isinstance(value, str) and _ranged(nodes.get(prop)):
                roots.update(_types(nodes.get(value)))
    nodes.read_ancestry(roots)

    findings: list[OntologyFinding] = []
    for row in rows:
        findings.extend(_check_row(row, nodes))
        served = code_index.statvar_by_dcid(row.dcid) if row.dcid is not None else None
        if served is not None:
            findings.extend(_differences(row, served))
    return findings


def _named(row: OntologyCheckRow) -> list[str]:
    """Every node id the row names."""
    named = [value for field in _SHAPE_FIELDS if (value := getattr(row, field)) is not None]
    for prop, value in row.constraint_properties.items():
        named.append(prop)
        if isinstance(value, str):
            named.append(value)
    named.extend(row.observation_properties)
    return named


def _types(node: _Node | None) -> frozenset[str]:
    return frozenset() if node is None else node.types


def _ranged(node: _Node | None) -> bool:
    return node is not None and node.instance and _PROPERTY in node.types and bool(node.range)


def _check_row(row: OntologyCheckRow, nodes: _Nodes) -> list[OntologyFinding]:
    findings: list[OntologyFinding] = []

    def find(field: str, value: ConstraintValue, code: FindingCode) -> None:
        severity = "warning" if code == "property_outside_domain" else "error"
        findings.append(
            OntologyFinding(ref=row.ref, field=field, value=value, code=code, severity=severity)
        )

    population: set[str] | None = None
    if row.population_type is not None:
        held = nodes.get(row.population_type)
        if held is None:
            find("population_type", row.population_type, "node_not_held")
        elif _CLASS not in held.types:
            find("population_type", row.population_type, "not_a_class")
        else:
            population = nodes.ancestry([row.population_type])

    def check_property(field: str, prop: str) -> None:
        held = nodes.get(prop)
        if held is None:
            find(field, prop, "node_not_held")
        elif _PROPERTY not in held.types:
            find(field, prop, "not_a_property")
        elif (
            population is not None
            and held.instance
            and held.domain
            and population.isdisjoint(held.domain)
        ):
            find(field, prop, "property_outside_domain")

    check_property("measured_property", row.measured_property)
    for field in ("stat_type", "measurement_qualifier", "measurement_denominator"):
        value = getattr(row, field)
        if value is not None and nodes.get(value) is None:
            find(field, value, "node_not_held")

    for prop, value in sorted(row.constraint_properties.items()):
        field = f"{_CONSTRAINT}.{prop}"
        check_property(field, prop)
        if value is None:
            find(field, value, "value_missing")
            continue
        if not isinstance(value, str):
            find(field, value, "node_not_held")
            continue
        held_value = nodes.get(value)
        fault = constraint_value_fault(None if held_value is None else held_value.types)
        if fault == "not_held":
            find(field, value, "node_not_held")
        elif fault == "untyped":
            find(field, value, "node_untyped")
        elif fault == "provisional":
            find(field, value, "node_provisional")
        elif (
            (prop_node := nodes.get(prop)) is not None
            and _ranged(prop_node)
            and nodes.ancestry(_types(held_value)).isdisjoint(prop_node.range)
        ):
            find(field, value, "value_outside_range")

    for prop in row.observation_properties:
        held = nodes.get(prop)
        if held is None:
            find(_OBSERVATION, prop, "node_not_held")
        elif _PROPERTY not in held.types:
            find(_OBSERVATION, prop, "not_a_property")
    return findings


def _differences(row: OntologyCheckRow, served: StatVarRow) -> list[OntologyFinding]:
    """One `definition_differs` warning per field where the row departs from `served`."""
    differs: list[tuple[str, ConstraintValue | list[str]]] = [
        (field, getattr(served, field))
        for field in _SHAPE_FIELDS
        if getattr(row, field) != getattr(served, field)
    ]
    differs.extend(_constraint_differences(row.constraint_properties, served.constraint_properties))
    if set(row.observation_properties) != set(served.observation_properties):
        differs.append((_OBSERVATION, sorted(served.observation_properties)))
    return [
        OntologyFinding(
            ref=row.ref, field=field, value=value, code="definition_differs", severity="warning"
        )
        for field, value in differs
    ]


def _constraint_differences(
    sent: Mapping[str, ConstraintValue], served: Mapping[str, str]
) -> list[tuple[str, ConstraintValue | list[str]]]:
    return [
        (f"{_CONSTRAINT}.{prop}", served.get(prop))
        for prop in sorted(sent.keys() | served.keys())
        if prop not in sent or prop not in served or sent[prop] != served[prop]
    ]
