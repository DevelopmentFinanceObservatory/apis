"""Check candidate StatisticalVariable definitions against the instance's ontology.

`OntologyLookup` holds the ontology document's nodes as the check reads them, built once per
document. `plan_reads` decides, before any read, how many rows from the first the check takes
and which ids outside the document they read: it adds rows while their distinct ids stay within
`CHECK_MAX_READ_IDS`. `check_rows` answers each node a row names from the document and asks
every other named id once of the reader's `node_arcs_many`, for its `typeOf` and `name`, with
no probe budget. The served reader, `origin_catalogue.graph.OriginGraphClient`, answers an id
the origin catalogue's arc tables hold from those tables, a node they hold with no arc as
absent, and reads every other id from the graph. There an id a batch answers empty is read
again with the other empties, then alone, so `None` is the graph's answer for that id read
alone. A node with neither a type nor a name reads as absent, and a node with a name and no
`typeOf` is untyped. A read error propagates, so no node is reported absent because a read
failed. The check reads no ancestry.

The check reads every planned id in one `node_arcs_many` call: the shape fields, constraint
properties, observation properties and constraint values the document does not hold, less the
cells no read can decide and the values under a document node that is no Property. A
constraint value whose property is not a held Property gets no finding of its own, because the
property's finding marks the cell, so a value under a property outside dcid syntax or outside
the served origins is not read.

A check reads at most `CHECK_MAX_READ_IDS` (41) ids, each once. It reads fewer from the
graph when the arc tables answer some. The `node_arcs_many` call sends, for each backend its ids
go to (the mixer, and the UN deployment where the deployment serves the UN origin), one POST of
its ids (a chunk holds 50), one POST of the group retry of the empty answers, and one POST per
id still empty. A check of k ids therefore sends at most k + 2 POSTs with one backend, 43 for
41 ids, and k + 4 with two, 45. A front-door refusal splits a POST of several ids into two
halves, so a batch or group retry of j ids costs at most 2j - 1 POSTs, and a check at most
5k - 2, 203 for 41 ids, on either kind of deployment. A continuation page adds one POST. Each
POST is tried up to 3 times on a 5xx answer or a transport error. Every read carries the
check's deadline, `GRAPH_ONTOLOGY_CHECK_TIMEOUT_SECONDS` (30 s) from the request's start.

No read can decide three kinds of cell, so each gets its finding with no read: a string outside
dcid syntax, a number or a boolean gets `node_not_held`, and an id of the UN origin on a
deployment that does not serve it (`origin_catalogue.graph.is_un_dcid`) gets
`origin_not_served`. Every other id outside the document is read, whatever the range of its
property: the document holds the ancestors of the classes it holds and none of their
descendants, so no member listing proves that a value it lacks lies outside a range. One row
names at most `CHECK_MAX_READ_IDS` ids, so the first row always fits.

`check_rows` stops before the row whose findings would take the total past
`CHECK_MAX_FINDINGS`. One row yields at most 79 findings (`CHECK_MAX_FINDINGS` counts them), so
the first row always fits.

Every row gets existence and kind checks. A row with no measured property gets `value_missing`.
A constraint value, `stat_type`, `measurement_qualifier` and `measurement_denominator` must each
name a resolved node, by `code_index.constraint_values.constraint_value_fault`, the test the
code index export refuses a release on for a constraint value. Range and domain rules apply
only to a property the document declares `declared_by == "instance"`, and judge only through
the document, which holds every `subClassOf` ancestor of each class it holds. A value is
outside the range when every one of its types is a document node and none reaches a
`rangeIncludes` class by `subClassOf`. A value with any type outside the document gets no
range finding. A property is outside its domain when the row's population type is a document
node that reaches no `domainIncludes` class by `subClassOf`; a population type outside the
document gets no domain finding. A property with no `rangeIncludes` or no `domainIncludes`
admits anything. The domain rule covers the measured property and the constraint properties,
and is skipped when the row names no population type. A row's `dcid` outside dcid syntax gets
`node_not_held`. A row whose `dcid` the code index serves is compared with the served
definition cell by cell.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from code_index import CodeIndex, StatVarRow
from code_index.constraint_values import ConstraintValueFault, constraint_value_fault
from dc_graph import UNRESOLVED_NODE
from graph_api.dataset_usage import SHAPE_FIELDS
from graph_api.ontology_models import (
    CHECK_MAX_FINDINGS,
    CHECK_MAX_READ_IDS,
    ConstraintValue,
    FindingCode,
    FindingField,
    OntologyCheckRow,
    OntologyDocument,
    OntologyFinding,
)
from graph_api.reader import GraphNodeReader
from graph_api.routing import DcidConvertor
from origin_catalogue.graph import is_un_dcid

_CLASS: Final[str] = "Class"
_PROPERTY: Final[str] = "Property"
_READ_PROPERTIES: Final[tuple[str, ...]] = ("typeOf", "name")
_DCID: Final[re.Pattern[str]] = re.compile(rf"{DcidConvertor.regex}")
_CONSTRAINT: Final = "constraint_properties"
_OBSERVATION: Final = "observation_properties"


@dataclass(frozen=True)
class _Node:
    types: frozenset[str]
    parents: tuple[str, ...] = ()
    instance: bool = False
    domain: tuple[str, ...] = ()
    range: tuple[str, ...] = ()


@dataclass(frozen=True)
class OntologyLookup:
    """The ontology document's nodes as the check reads them."""

    nodes: Mapping[str, _Node]

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
        return cls(nodes=MappingProxyType(nodes))

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
class _Unread:
    """Decides, with no read, the cells no read can answer."""

    lookup: OntologyLookup
    un_served: bool

    def fault(self, dcid: str) -> FindingCode | None:
        """`node_not_held` for `dcid` outside dcid syntax, `origin_not_served` for an id of the
        UN origin outside the document while the deployment does not serve it, else `None`."""
        if _DCID.fullmatch(dcid) is None:
            return "node_not_held"
        if not self.un_served and dcid not in self.lookup.nodes and is_un_dcid(dcid):
            return "origin_not_served"
        return None


@dataclass(frozen=True)
class ReadPlan:
    """The rows from the first that one check takes, and the ids outside the document they
    read."""

    row_count: int
    ids: frozenset[str]


def plan_reads(
    rows: Sequence[OntologyCheckRow], *, lookup: OntologyLookup, un_served: bool
) -> ReadPlan:
    """Take rows from the first while the distinct ids they read stay within
    `CHECK_MAX_READ_IDS`, decided with no read. A row reads every shape field, constraint
    property and observation property outside the document, and every constraint value outside
    it that is a dcid, less the cells no read can decide and the values under a property no read
    can decide or under a document node that is no Property. The first row is always taken.

    Args:
        rows: the candidate definitions, at least one.
        lookup: the ontology document's lookup.
        un_served: whether the deployment serves the UN origin, whose ids
            `origin_catalogue.graph.is_un_dcid` names.
    """
    unread = _Unread(lookup, un_served)
    ids: set[str] = set()
    for count, row in enumerate(rows):
        taken = ids | _row_reads(row, unread)
        if count and len(taken) > CHECK_MAX_READ_IDS:
            return ReadPlan(row_count=count, ids=frozenset(ids))
        ids = taken
    return ReadPlan(row_count=len(rows), ids=frozenset(ids))


def _row_reads(row: OntologyCheckRow, unread: _Unread) -> set[str]:
    named = [value for field in SHAPE_FIELDS if (value := getattr(row, field)) is not None]
    named.extend(row.observation_properties)
    for prop, value in row.constraint_properties.items():
        named.append(prop)
        if unread.fault(prop) is None and (read := _read_value(prop, value, unread)) is not None:
            named.append(read)
    return {
        dcid for dcid in named if dcid not in unread.lookup.nodes and unread.fault(dcid) is None
    }


class _Nodes:
    """The nodes the rows name: document nodes free, the rest read once each."""

    def __init__(self, lookup: OntologyLookup, *, reader: GraphNodeReader, deadline: float) -> None:
        self._lookup = lookup
        self._read: dict[str, _Node | None] = {}
        self._reader = reader
        self._deadline = deadline

    def read(self, dcids: Iterable[str]) -> None:
        """Read `dcids` in one `node_arcs_many` call that confirms each empty answer alone."""
        unknown = sorted(set(dcids))
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


@dataclass(frozen=True)
class CheckedRows:
    """The rows one check took, from the first, and their findings."""

    row_count: int
    findings: list[OntologyFinding]


def check_rows(
    rows: Sequence[OntologyCheckRow],
    *,
    reads: frozenset[str],
    lookup: OntologyLookup,
    reader: GraphNodeReader,
    code_index: CodeIndex,
    un_served: bool,
    deadline: float,
) -> CheckedRows:
    """Find the faults in candidate StatVar definitions, from the first row, stopping before
    the row whose findings would take the total past `CHECK_MAX_FINDINGS`.

    Args:
        rows: the rows `plan_reads` took.
        reads: the ids `plan_reads` planned for `rows`.
        lookup: the ontology document's lookup, which answers most ids with no read.
        reader: the graph reads for nodes the document does not hold, called only when
            `reads` names an id.
        code_index: the code index whose StatVar rows the rows' dcids are compared with.
        un_served: whether the deployment serves the UN origin, as `plan_reads` was told.
        deadline: the absolute monotonic deadline passed to every read.

    Returns:
        The count of rows checked, at least one, and one finding per fault, in row order.
        Within a row: `dcid`; the shape fields in the order `population_type`,
        `measured_property`, `stat_type`, `measurement_qualifier`, `measurement_denominator`;
        the constraint properties sorted, each property's finding before its value's; the
        observation properties in row order; then `definition_differs` findings in the same
        field order.

    Raises:
        GraphReadError: a read failed.
    """
    nodes = _Nodes(lookup, reader=reader, deadline=deadline)
    nodes.read(reads)
    unread = _Unread(lookup, un_served)

    findings: list[OntologyFinding] = []
    for index, row in enumerate(rows):
        found = _check_row(index, row, nodes, unread)
        if row.dcid is not None and (served := code_index.statvar_by_dcid(row.dcid)) is not None:
            found.extend(_differences(index, row, served))
        if index and len(findings) + len(found) > CHECK_MAX_FINDINGS:
            return CheckedRows(row_count=index, findings=findings)
        findings.extend(found)
    return CheckedRows(row_count=len(rows), findings=findings)


def _is_dcid(value: ConstraintValue) -> bool:
    return isinstance(value, str) and _DCID.fullmatch(value) is not None


def _read_value(prop: str, value: ConstraintValue, unread: _Unread) -> str | None:
    """`value` when the check answers it from a node, `None` when the value or the property
    alone decides: a literal, or a value under a document node that is no Property."""
    if not isinstance(value, str) or not _is_dcid(value):
        return None
    nodes = unread.lookup.nodes
    if prop in nodes and not _is_property(nodes[prop]):
        return None
    return value


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


_WARNINGS: Final[frozenset[FindingCode]] = frozenset(
    {"property_outside_domain", "definition_differs"}
)


def _finding(
    index: int,
    row: OntologyCheckRow,
    field: FindingField,
    value: ConstraintValue | list[str],
    code: FindingCode,
    *,
    prop: str | None = None,
    subject: Literal["property", "value"] = "value",
    served: ConstraintValue | list[str] = None,
) -> OntologyFinding:
    return OntologyFinding(
        row=index,
        ref=row.ref,
        field=field,
        property=prop,
        subject=subject,
        value=value,
        served=served,
        code=code,
        severity="warning" if code in _WARNINGS else "error",
    )


def _check_row(
    index: int, row: OntologyCheckRow, nodes: _Nodes, unread: _Unread
) -> list[OntologyFinding]:
    findings: list[OntologyFinding] = []
    lookup = unread.lookup

    def find(
        field: FindingField,
        value: ConstraintValue,
        code: FindingCode,
        *,
        prop: str | None = None,
        subject: Literal["property", "value"] = "value",
    ) -> None:
        findings.append(_finding(index, row, field, value, code, prop=prop, subject=subject))

    if row.dcid is not None and _DCID.fullmatch(row.dcid) is None:
        find("dcid", row.dcid, "node_not_held")

    population: set[str] | None = None
    if row.population_type is not None:
        if fault := unread.fault(row.population_type):
            find("population_type", row.population_type, fault)
        elif (held := nodes.get(row.population_type)) is None:
            find("population_type", row.population_type, "node_not_held")
        elif _CLASS not in held.types:
            find("population_type", row.population_type, "not_a_class")
        elif row.population_type in lookup.nodes:
            population = lookup.ancestry([row.population_type])

    def check_property(field: FindingField, prop: str, *, cell: bool) -> bool:
        """Find the property's faults and return whether it is a held Property. `cell` is true
        when `prop` is a constraint property, which the finding then names as its subject."""

        def find_property(code: FindingCode) -> None:
            if cell:
                find(field, prop, code, prop=prop, subject="property")
            else:
                find(field, prop, code)

        if fault := unread.fault(prop):
            find_property(fault)
            return False
        held = nodes.get(prop)
        if held is None:
            find_property("node_not_held")
            return False
        if not _is_property(held):
            find_property("not_a_property")
            return False
        if (
            population is not None
            and held.instance
            and held.domain
            and population.isdisjoint(held.domain)
        ):
            find_property("property_outside_domain")
        return True

    if row.measured_property is None:
        find("measured_property", None, "value_missing")
    else:
        check_property("measured_property", row.measured_property, cell=False)
    for field in ("stat_type", "measurement_qualifier", "measurement_denominator"):
        value = getattr(row, field)
        if value is None:
            continue
        if fault := unread.fault(value) or _node_fault(nodes.get(value)):
            find(field, value, fault)

    for prop, value in sorted(row.constraint_properties.items()):
        if not check_property(_CONSTRAINT, prop, cell=True):
            continue
        if value is None:
            find(_CONSTRAINT, value, "value_missing", prop=prop)
            continue
        if not isinstance(value, str):
            find(_CONSTRAINT, value, "node_not_held", prop=prop)
            continue
        if fault := unread.fault(value):
            find(_CONSTRAINT, value, fault, prop=prop)
            continue
        held_value = nodes.get(value)
        if fault := _node_fault(held_value):
            find(_CONSTRAINT, value, fault, prop=prop)
        elif (
            held_value is not None
            and _ranged(prop_node := nodes.get(prop))
            and prop_node is not None
            and _outside_range(held_value, prop_node, lookup)
        ):
            find(_CONSTRAINT, value, "value_outside_range", prop=prop)

    for prop in row.observation_properties:
        if (fault := unread.fault(prop)) is None:
            held = nodes.get(prop)
            if held is None:
                fault = "node_not_held"
            elif not _is_property(held):
                fault = "not_a_property"
        if fault is not None:
            find(_OBSERVATION, prop, fault, prop=prop, subject="property")
    return findings


def _differences(index: int, row: OntologyCheckRow, served: StatVarRow) -> list[OntologyFinding]:
    """One `definition_differs` warning per cell where the row departs from `served`, carrying
    what the row sent and what the code index serves."""
    differs = [
        _finding(index, row, field, sent, "definition_differs", served=getattr(served, field))
        for field in SHAPE_FIELDS
        if (sent := getattr(row, field)) != getattr(served, field)
    ]
    sent_constraints = row.constraint_properties
    served_constraints = served.constraint_properties
    differs.extend(
        _finding(
            index,
            row,
            _CONSTRAINT,
            sent_constraints.get(prop),
            "definition_differs",
            prop=prop,
            served=served_constraints.get(prop),
        )
        for prop in sorted(sent_constraints.keys() | served_constraints.keys())
        if prop not in sent_constraints
        or prop not in served_constraints
        or sent_constraints[prop] != served_constraints[prop]
    )
    if set(row.observation_properties) != set(served.observation_properties):
        differs.append(
            _finding(
                index,
                row,
                _OBSERVATION,
                sorted(row.observation_properties),
                "definition_differs",
                served=sorted(served.observation_properties),
            )
        )
    return differs
