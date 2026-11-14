"""Check candidate StatisticalVariable definitions against the instance's ontology.

`OntologyLookup` holds the ontology document's nodes as the check reads them, built once per
document. `plan_reads` decides, before any read, which ids outside the document a request
reads. It refuses the request when they pass `CHECK_MAX_READ_IDS` or include an id of an
origin the deployment does not serve. `check_rows` answers each node a row names from the
document and asks every other named id once of the reader's `node_arcs_many`, for its `typeOf`
and `name`, with no probe budget. The served reader, `origin_catalogue.graph.OriginGraphClient`,
answers an id the origin catalogue's arc tables hold from those tables, a node they hold with no
arc as absent, and reads every other id from the graph. There an id a batch answers empty is
read again with the other empties, then alone, so `None` is the graph's answer for that id
read alone. A node with neither a type nor a name reads as absent, and a node with a name and
no `typeOf` is untyped. A read error propagates, so no node is reported absent because a read
failed. The check reads no ancestry.

The reads come in two rounds. The first reads the shape fields, the constraint properties and
the observation properties the document does not hold, and the values of constraint properties
the document holds as Properties. The second reads the values of constraint properties the
first round found held as Properties. A constraint value whose property is not a held Property
is not read and gets no finding of its own, because the property's finding marks the cell.

A request reads at most `CHECK_MAX_READ_IDS` (41) ids, each once, because the second round
skips an id the first read. It reads fewer from the graph when the arc tables answer some.
Each round is one `node_arcs_many` call, which sends, for each backend its ids go to (the
mixer, and the UN deployment where the deployment serves the UN origin), one POST of its ids
(a chunk holds 50), one POST of the group retry of the empty answers, and one POST per id
still empty. A check of k ids therefore sends at most k + 4 POSTs with one backend, 45 for 41
ids, and k + 8 with two, 49. A front-door refusal splits a POST of several ids into two
halves, so a batch or group retry of j ids costs at most 2j - 1 POSTs, and a check at most
5k - 2, 203 for 41 ids, on either kind of deployment. A continuation page adds one POST. Each
POST is tried up to 3 times on a 5xx answer or a transport error. Every read carries the
check's deadline, `GRAPH_ONTOLOGY_CHECK_TIMEOUT_SECONDS` (30 s) from the request's start.

Some constraint values need no read. A value that is a number, a boolean, or a string outside
dcid syntax names no node, so it gets `node_not_held`. A value outside the document, on an
instance property whose range the document decides (`OntologyLookup.decides`), is no member of
the range and gets `value_outside_range`. Every other id outside the document is read.
`plan_reads` counts every value under a property outside the document, read or not. One row
names at most `CHECK_MAX_READ_IDS` ids, so a single row always fits.

The findings stop at `CHECK_MAX_FINDINGS`. `check_rows` raises `OntologyCheckTooManyFindings`
at the row that passes it. One row yields at most 79 findings (`CHECK_MAX_FINDINGS` counts
them), so a single row always fits.

Every row gets existence and kind checks. A constraint value, `stat_type`,
`measurement_qualifier` and `measurement_denominator` must each name a resolved node, by
`code_index.constraint_values.constraint_value_fault`, the test the code index export refuses
a release on for a constraint value. Range and domain rules apply only to a property the
document declares `declared_by == "instance"`, and judge only through the document, which
holds every `subClassOf` ancestor of each class it holds. A value is outside the range when
every one of its types is a document node and none reaches a `rangeIncludes` class by
`subClassOf`. A value with any type outside the document gets no range finding. A property is
outside its domain
when the row's population type is a document node that reaches no `domainIncludes` class by
`subClassOf`; a population type outside the document gets no domain finding. A property with
no `rangeIncludes` or no `domainIncludes` admits anything. The domain rule covers the measured
property and the constraint properties, and is skipped when the row names no population type.
A row whose `dcid` the code index serves is compared with the served definition field by field.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from code_index import CodeIndex, StatVarRow
from code_index.constraint_values import ConstraintValueFault, constraint_value_fault
from dc_graph import UNRESOLVED_NODE
from graph_api.ontology_models import (
    CHECK_MAX_FINDINGS,
    CHECK_MAX_READ_IDS,
    ConstraintValue,
    FindingCode,
    OntologyCheckRow,
    OntologyDocument,
    OntologyFinding,
)
from graph_api.ontology_provenances import SHAPE_FIELDS
from graph_api.reader import GraphNodeReader
from graph_api.routing import DcidConvertor
from origin_catalogue.graph import is_un_dcid

_CLASS: Final[str] = "Class"
_PROPERTY: Final[str] = "Property"
_READ_PROPERTIES: Final[tuple[str, ...]] = ("typeOf", "name")
_DCID: Final[re.Pattern[str]] = re.compile(rf"{DcidConvertor.regex}")
_CONSTRAINT: Final[str] = "constraint_properties"
_OBSERVATION: Final[str] = "observation_properties"


class OntologyCheckTooManyReads(Exception):
    """The rows name more ids to read from the graph than `CHECK_MAX_READ_IDS`."""

    def __init__(self, count: int) -> None:
        super().__init__(
            f"the rows name {count} distinct ids the ontology document does not hold, past "
            f"the cap of {CHECK_MAX_READ_IDS} one check reads from the graph; send fewer rows"
        )
        self.count = count


class OntologyCheckUnservedId(Exception):
    """The rows name `dcid`, an id of an origin this deployment does not serve, so no read can
    establish whether the graph holds it."""

    def __init__(self, dcid: str) -> None:
        super().__init__(
            f"{dcid!r} belongs to an origin this deployment does not serve, so the check "
            "cannot read it"
        )
        self.dcid = dcid


class OntologyCheckTooManyFindings(Exception):
    """The rows have more findings than `CHECK_MAX_FINDINGS`."""

    def __init__(self) -> None:
        super().__init__(
            f"the rows have more than {CHECK_MAX_FINDINGS} findings, the most one check "
            "answers; send fewer rows"
        )


@dataclass(frozen=True)
class _Node:
    types: frozenset[str]
    parents: tuple[str, ...] = ()
    instance: bool = False
    domain: tuple[str, ...] = ()
    range: tuple[str, ...] = ()


@dataclass(frozen=True)
class OntologyLookup:
    """The ontology document's nodes as the check reads them, and the classes whose instances
    the document lists in full: enumerations whose `complete` member list is every member the
    graph holds, and whose every enumeration below them in the document by `subClassOf` is
    complete too."""

    nodes: Mapping[str, _Node]
    listed_in_full: frozenset[str]

    @classmethod
    def of(cls, document: OntologyDocument) -> OntologyLookup:
        nodes = {
            node.dcid: _Node(
                types=frozenset(node.types),
                parents=tuple(node.sub_class_of),
                instance=node.declared_by == "instance",
                domain=tuple(node.domain_includes),
                range=tuple(node.range_includes),
            )
            for node in document.nodes
        }
        enumerations = [node for node in document.nodes if node.enumeration is not None]
        full = {
            node.dcid
            for node in enumerations
            if node.enumeration is not None and node.enumeration.complete
        }
        for node in enumerations:
            if node.dcid not in full:
                full -= _ancestry(nodes, [node.dcid])
        return cls(nodes=MappingProxyType(nodes), listed_in_full=frozenset(full))

    def decides(self, prop: str) -> bool:
        """Whether a value outside the document is outside `prop`'s range with no read: `prop`
        is an instance property whose every `rangeIncludes` class the document lists in full,
        with every enumeration below it."""
        node = self.nodes.get(prop)
        return (
            node is not None
            and _ranged(node)
            and all(target in self.listed_in_full for target in node.range)
        )

    def ancestry(self, dcids: Iterable[str]) -> set[str]:
        """`dcids` and every `subClassOf` ancestor of them the document holds."""
        return _ancestry(self.nodes, dcids)


def _ancestry(nodes: Mapping[str, _Node], dcids: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    stack = list(dcids)
    while stack:
        dcid = stack.pop()
        if dcid in seen:
            continue
        seen.add(dcid)
        if (node := nodes.get(dcid)) is not None:
            stack.extend(node.parents)
    return seen


@dataclass(frozen=True)
class CheckReads:
    """The ids outside the document one check reads: `first`, read in the first round, and
    `pending`, each constraint property outside the document with its values outside the
    document, read in the second round under a property the first found a held Property."""

    first: frozenset[str]
    pending: Mapping[str, frozenset[str]]

    @property
    def ids(self) -> frozenset[str]:
        """Every id the check may read."""
        return self.first.union(*self.pending.values())


def plan_reads(
    rows: Sequence[OntologyCheckRow], *, lookup: OntologyLookup, un_served: bool
) -> CheckReads:
    """The ids outside the document `rows` name that the check reads, decided with no read.

    Args:
        rows: the candidate definitions.
        lookup: the ontology document's lookup.
        un_served: whether the deployment serves the UN origin, whose ids
            `origin_catalogue.graph.is_un_dcid` names.

    Raises:
        OntologyCheckTooManyReads: the rows name more than `CHECK_MAX_READ_IDS` ids to read.
        OntologyCheckUnservedId: an id to read belongs to the UN origin and `un_served` is
            false. The smallest such id is named.
    """
    first: set[str] = set()
    pending: dict[str, set[str]] = {}
    for row in rows:
        first.update(dcid for dcid in _named(row) if dcid not in lookup.nodes)
        for prop, value in row.constraint_properties.items():
            read = _read_value(prop, value, lookup)
            if read is None or read in lookup.nodes:
                continue
            if prop in lookup.nodes:
                first.add(read)
            else:
                pending.setdefault(prop, set()).add(read)
    reads = CheckReads(
        first=frozenset(first),
        pending=MappingProxyType({prop: frozenset(values) for prop, values in pending.items()}),
    )
    ids = reads.ids
    if len(ids) > CHECK_MAX_READ_IDS:
        raise OntologyCheckTooManyReads(len(ids))
    if not un_served and (unserved := sorted(dcid for dcid in ids if is_un_dcid(dcid))):
        raise OntologyCheckUnservedId(unserved[0])
    return reads


class _Nodes:
    """The nodes the rows name: document nodes free, the rest read once each."""

    def __init__(self, lookup: OntologyLookup, *, reader: GraphNodeReader, deadline: float) -> None:
        self._lookup = lookup
        self._read: dict[str, _Node | None] = {}
        self._reader = reader
        self._deadline = deadline

    def read(self, dcids: Iterable[str]) -> None:
        """Read every one of `dcids` not read before in one `node_arcs_many` call that confirms
        each empty answer alone, so a check asks each id once."""
        unknown = sorted({dcid for dcid in dcids if dcid not in self._read})
        if not unknown:
            return
        arcs = self._reader.node_arcs_many(
            unknown,
            properties=_READ_PROPERTIES,
            individual_probe_budget=None,
            deadline=self._deadline,
        )
        for dcid in unknown:
            self._read[dcid] = _held(arcs.get(dcid))

    def get(self, dcid: str) -> _Node | None:
        if (node := self._lookup.nodes.get(dcid)) is not None:
            return node
        return self._read[dcid]


def _held(arcs: Mapping[str, tuple[str, ...]] | None) -> _Node | None:
    """The node `node_arcs_many` answered with `arcs`, or `None` for a node with neither a
    type nor a name."""
    if arcs is None:
        return None
    types = frozenset(value for value in arcs["typeOf"] if value != UNRESOLVED_NODE)
    if not types and all(value == UNRESOLVED_NODE for value in arcs["name"]):
        return None
    return _Node(types=types)


def check_rows(
    rows: Sequence[OntologyCheckRow],
    *,
    reads: CheckReads,
    lookup: OntologyLookup,
    reader: GraphNodeReader,
    code_index: CodeIndex,
    deadline: float,
) -> list[OntologyFinding]:
    """Find the faults in candidate StatVar definitions.

    Args:
        rows: the candidate definitions.
        reads: `plan_reads` of `rows` and `lookup`.
        lookup: the ontology document's lookup, which answers most ids with no read.
        reader: the graph reads for nodes the document does not hold, called only when
            `reads` names an id.
        code_index: the code index whose StatVar rows the rows' dcids are compared with.
        deadline: the absolute monotonic deadline passed to every read.

    Returns:
        One finding per fault, in row order. Within a row: the shape fields in the order
        `population_type`, `measured_property`, `stat_type`, `measurement_qualifier`,
        `measurement_denominator`; the constraint properties sorted, each property's finding
        before its value's; the observation properties in row order; then `definition_differs`
        findings in the same field order.

    Raises:
        OntologyCheckTooManyFindings: the findings passed `CHECK_MAX_FINDINGS`, raised at the
            row that passed it.
        GraphReadError: a read failed.
    """
    nodes = _Nodes(lookup, reader=reader, deadline=deadline)
    nodes.read(reads.first)
    nodes.read(
        value
        for prop, values in reads.pending.items()
        if _is_property(nodes.get(prop))
        for value in values
    )

    findings: list[OntologyFinding] = []
    for row in rows:
        findings.extend(_check_row(row, nodes, lookup))
        served = code_index.statvar_by_dcid(row.dcid) if row.dcid is not None else None
        if served is not None:
            findings.extend(_differences(row, served))
        if len(findings) > CHECK_MAX_FINDINGS:
            raise OntologyCheckTooManyFindings()
    return findings


def _is_dcid(value: ConstraintValue) -> bool:
    return isinstance(value, str) and _DCID.fullmatch(value) is not None


def _read_value(prop: str, value: ConstraintValue, lookup: OntologyLookup) -> str | None:
    """`value` when the check answers it from a node, `None` when the value or the property
    alone decides: a literal, a value under a document node that is no Property, or a value
    outside the document whose range the document decides."""
    if not isinstance(value, str) or not _is_dcid(value):
        return None
    if prop in lookup.nodes and not _is_property(lookup.nodes[prop]):
        return None
    if value not in lookup.nodes and lookup.decides(prop):
        return None
    return value


def _named(row: OntologyCheckRow) -> list[str]:
    """Every node id the row names other than its constraint values."""
    named = [value for field in SHAPE_FIELDS if (value := getattr(row, field)) is not None]
    named.extend(row.constraint_properties)
    named.extend(row.observation_properties)
    return named


def _is_property(node: _Node | None) -> bool:
    return node is not None and _PROPERTY in node.types


def _ranged(node: _Node | None) -> bool:
    return node is not None and node.instance and _is_property(node) and bool(node.range)


def _outside_range(value: _Node, prop: _Node, lookup: OntologyLookup) -> bool:
    """Every type of `value` is a document node and none reaches `prop`'s range."""
    if not value.types or any(dcid not in lookup.nodes for dcid in value.types):
        return False
    return lookup.ancestry(value.types).isdisjoint(prop.range)


_NODE_FAULTS: Final[Mapping[ConstraintValueFault, FindingCode]] = MappingProxyType(
    {"not_held": "node_not_held", "untyped": "node_untyped", "provisional": "node_provisional"}
)


def _node_fault(node: _Node | None) -> FindingCode | None:
    """Why `node` is no resolved node, by `constraint_value_fault`, or `None` when it is."""
    fault = constraint_value_fault(None if node is None else node.types)
    return None if fault is None else _NODE_FAULTS[fault]


def _check_row(
    row: OntologyCheckRow, nodes: _Nodes, lookup: OntologyLookup
) -> list[OntologyFinding]:
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
        elif row.population_type in lookup.nodes:
            population = lookup.ancestry([row.population_type])

    def check_property(field: str, prop: str) -> bool:
        """Find the property's faults and return whether it is a held Property."""
        held = nodes.get(prop)
        if held is None:
            find(field, prop, "node_not_held")
            return False
        if not _is_property(held):
            find(field, prop, "not_a_property")
            return False
        if (
            population is not None
            and held.instance
            and held.domain
            and population.isdisjoint(held.domain)
        ):
            find(field, prop, "property_outside_domain")
        return True

    check_property("measured_property", row.measured_property)
    for field in ("stat_type", "measurement_qualifier", "measurement_denominator"):
        if (value := getattr(row, field)) is not None and (fault := _node_fault(nodes.get(value))):
            find(field, value, fault)

    for prop, value in sorted(row.constraint_properties.items()):
        field = f"{_CONSTRAINT}.{prop}"
        if not check_property(field, prop):
            continue
        if value is None:
            find(field, value, "value_missing")
            continue
        if not _is_dcid(value):
            find(field, value, "node_not_held")
            continue
        read = _read_value(prop, value, lookup)
        if read is None:
            find(field, value, "value_outside_range")
            continue
        held_value = nodes.get(read)
        if fault := _node_fault(held_value):
            find(field, value, fault)
        elif (
            held_value is not None
            and _ranged(prop_node := nodes.get(prop))
            and prop_node is not None
            and _outside_range(held_value, prop_node, lookup)
        ):
            find(field, value, "value_outside_range")

    for prop in row.observation_properties:
        held = nodes.get(prop)
        if held is None:
            find(_OBSERVATION, prop, "node_not_held")
        elif not _is_property(held):
            find(_OBSERVATION, prop, "not_a_property")
    return findings


def _differences(row: OntologyCheckRow, served: StatVarRow) -> list[OntologyFinding]:
    """One `definition_differs` warning per field where the row departs from `served`."""
    differs: list[tuple[str, ConstraintValue | list[str]]] = [
        (field, getattr(served, field))
        for field in SHAPE_FIELDS
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
