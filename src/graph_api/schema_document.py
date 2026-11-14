"""Build the instance's schema document and the terms list from live graph reads.

`build_schema` reads `Class <-typeOf` and `Property <-typeOf` once each, and every subject is a
term. The document's seed is every term the instance declares plus every node a StatVar row of
the code index names. One `node_all_arc_targets_many` reads every outgoing property of the
seed's nodes with the provenances that declare each value, and each further level reads the
nodes not yet read that the instance's terms name in `subClassOf`, `subPropertyOf`,
`domainIncludes` or `rangeIncludes`, the `typeOf` targets of every node read that is neither a
Class nor a Property, and the `subClassOf` and `subPropertyOf` parents, so the class of every
value is present and every ancestor is too. One `node_arc_subjects_many` lists the members of
every instance-declared enumeration, and each enumeration that listing omits or answers with no
member is listed again alone with `node_arc_subjects`, whose answer stands. The members not yet
read join the document with the same closure over their types and ancestors. Each node read
probes every empty answer: it reads the empties together once more, then each remaining empty
alone, and a dcid empty after that is absent from the document. The node reads are uncached,
since no other caller reads every property of these nodes. So the number of reads grows with
the depth of the class and property hierarchy, plus a group read per level with an empty answer,
one read per dcid the graph does not hold, and one listing per instance enumeration the batched
listing leaves empty. Each node read sends one request per chunk of the client's `->*` chunk
size, and halves a chunk the mixer answers with a continuation.

Each node's values and provenances come from that one read of the node. The ten properties
`SchemaNode` names are its named members, and every other property the read answers is in
`other_properties`, each target a node reference or a literal, less any literal past
`OTHER_LITERAL_MAX_CHARS` and the values past the first `OTHER_VALUES_MAX` of a property, which
one warning per build counts. Its provenances are
those its `typeOf` targets carry, and `declared_by` follows from them. A node with no `typeOf`
arc, or whose `typeOf` targets carry no provenance, is served with none and declared by
neither. A term's provenances come from its listing entries by the same rule. One warning per
build counts the terms served with none and the nodes with a `typeOf` target that carries none,
and names the first ten of each. A build whose document holds no node the instance declares
refuses with `SchemaNoInstanceDeclarationError`.

An instance-declared enumeration's members are its `<-typeOf` listing, `complete` when the
listing read every subject. A base enumeration lists the members StatVar rows use and is
always `complete: false`.

Every node, member and list of values is sorted, so two builds from the same graph serialise
to the same bytes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from code_index import CodeIndex
from dc_graph import ArcSubject, ArcTarget, GraphReadError, display_name
from graph_api.dataset_usage import SHAPE_FIELDS
from graph_api.reader import GraphNodeReader
from graph_api.schema_models import (
    DeclaredBy,
    SchemaDocument,
    SchemaEnumeration,
    SchemaLiteral,
    SchemaNode,
    SchemaNodeReference,
    SchemaTerm,
    SchemaTerms,
)

logger = logging.getLogger(__name__)

TERM_LISTING_CAP: Final[int] = 20_000
"""The most subjects one `<-typeOf` listing of `Class` or `Property` reads. A listing that
reaches it raises `SchemaListingTruncatedError`."""

ENUMERATION_MEMBER_CAP: Final[int] = 5_000
"""The most members read for one instance-declared enumeration. An enumeration with more is
served with the members read and `complete: false`."""

OTHER_LITERAL_MAX_CHARS: Final[int] = 2_000
"""The longest literal `other_properties` holds. A longer one, such as a geometry, is left out."""

OTHER_VALUES_MAX: Final[int] = 100
"""The most values `other_properties` holds for one property: the first of them sorted."""

_LOGGED_DCIDS: Final[int] = 10
_BASE_PREFIX: Final[str] = "dc/base/"
_CLASS: Final = "Class"
_PROPERTY: Final = "Property"
_ENUMERATION: Final[str] = "Enumeration"

_TYPE_OF: Final[str] = "typeOf"
_ANCESTOR_ARCS: Final[tuple[str, ...]] = ("subClassOf", "subPropertyOf")
_SCHEMA_ARCS: Final[tuple[str, ...]] = (*_ANCESTOR_ARCS, "domainIncludes", "rangeIncludes")
_NODE_PROPERTIES: Final[tuple[str, ...]] = (
    _TYPE_OF,
    "name",
    "description",
    "alternateName",
    "subClassOf",
    "subPropertyOf",
    "specializationOf",
    "domainIncludes",
    "rangeIncludes",
    "classificationCode",
)

type _Arcs = dict[str, tuple[str, ...]]
type _OtherProperties = dict[str, list[SchemaNodeReference | SchemaLiteral]]
type _TermKind = Literal["Class", "Property"]


class SchemaListingTruncatedError(GraphReadError):
    """A `Class` or `Property` listing reached `TERM_LISTING_CAP`, so the terms are incomplete."""


class SchemaNoInstanceDeclarationError(GraphReadError):
    """No node of the built document is declared by the instance, so the graph named no
    instance provenance for any of them and the check would judge no range or domain."""

    def __init__(self, node_count: int) -> None:
        super().__init__(
            f"none of the {node_count} nodes of the schema document is declared by the instance"
        )
        self.node_count = node_count


@dataclass(frozen=True)
class Schema:
    """The schema document and the terms list from one build."""

    document: SchemaDocument
    terms: SchemaTerms


@dataclass
class _ListedTerm:
    names: set[str] = field(default_factory=set)
    kinds: set[_TermKind] = field(default_factory=set)
    provenances: set[str] = field(default_factory=set)


def declared_by(provenances: Collection[str]) -> DeclaredBy:
    """`"base"` when every declaring provenance starts `dc/base/`, `"instance"` otherwise.

    Raises:
        ValueError: `provenances` is empty, which labels nothing.
    """
    if not provenances:
        raise ValueError("a term with no declaring provenance is neither base nor instance")
    if all(provenance.startswith(_BASE_PREFIX) for provenance in provenances):
        return "base"
    return "instance"


def build_schema(
    *,
    reader: GraphNodeReader,
    code_index: CodeIndex,
    generation: str | None,
    deadline: float,
) -> Schema:
    """Build the schema document and the terms list.

    Args:
        reader: the graph reads.
        code_index: the bound code index, whose StatVar rows name nodes the document holds.
        generation: the graph generation token both bodies carry, `None` where it is unknown.
        deadline: the absolute monotonic deadline passed to every read.

    Raises:
        SchemaListingTruncatedError: a `Class` or `Property` listing reached its cap.
        SchemaNoInstanceDeclarationError: no node of the document is declared by the
            instance.
        GraphReadError: a read failed.
    """
    listed = _read_terms(reader, deadline=deadline)
    instance_terms = {
        dcid
        for dcid, term in listed.items()
        if term.provenances and declared_by(term.provenances) == "instance"
    }
    index_named = _index_named(code_index)

    arcs: dict[str, _Arcs] = {}
    others: dict[str, _OtherProperties] = {}
    # The values left out of `others`, by node and property.
    left_out: dict[tuple[str, str], int] = {}
    # Each node's provenances, read with its arcs.
    provenances: dict[str, frozenset[str]] = {}
    absent: set[str] = set()
    unsourced: set[str] = set()
    read = _NodeReads(
        reader,
        listed=listed,
        arcs=arcs,
        others=others,
        left_out=left_out,
        provenances=provenances,
        absent=absent,
        unsourced=unsourced,
        deadline=deadline,
    )
    read.nodes(instance_terms | index_named)
    read.close(_named(arcs, instance_terms, _SCHEMA_ARCS) | _next_level(arcs, arcs.keys(), listed))

    instance_enumerations = sorted(
        dcid
        for dcid in instance_terms
        if dcid in arcs and _CLASS in listed[dcid].kinds and _reaches_enumeration(dcid, arcs)
    )
    members: dict[str, SchemaEnumeration] = {}
    listed_members: set[str] = set()
    for enumeration, (subjects, complete) in _list_members(
        reader, instance_enumerations, deadline=deadline
    ).items():
        listed_members.update(subject.dcid for subject in subjects)
        members[enumeration] = SchemaEnumeration(
            members=sorted({subject.dcid for subject in subjects}), complete=complete
        )
    if unread_members := listed_members - arcs.keys() - absent:
        read.close(_next_level(arcs, read.nodes(unread_members), listed))

    index_named_by_type: dict[str, list[str]] = defaultdict(list)
    for dcid in index_named:
        for type_ in arcs.get(dcid, {}).get(_TYPE_OF, ()):
            index_named_by_type[type_].append(dcid)
    enumerations = (
        dcid
        for dcid, term in listed.items()
        if dcid in arcs and _CLASS in term.kinds and _reaches_enumeration(dcid, arcs)
    )
    for enumeration in enumerations:
        if enumeration not in members:
            members[enumeration] = SchemaEnumeration(
                members=sorted(index_named_by_type.get(enumeration, ())), complete=False
            )

    unsourced_terms = sorted(dcid for dcid, term in listed.items() if not term.provenances)
    if unsourced_terms or unsourced:
        logger.warning(
            "schema build read %d terms with no listed provenance, served with declared_by null, "
            "among them %s, and %d nodes with a typeOf target that carries no provenance, among "
            "them %s",
            len(unsourced_terms),
            unsourced_terms[:_LOGGED_DCIDS],
            len(unsourced),
            sorted(unsourced)[:_LOGGED_DCIDS],
        )
    if left_out:
        logger.warning(
            "schema build left %d values out of other_properties, literals past %d characters "
            "or values past %d a property, among them on %s",
            sum(left_out.values()),
            OTHER_LITERAL_MAX_CHARS,
            OTHER_VALUES_MAX,
            sorted(left_out)[:_LOGGED_DCIDS],
        )
    nodes = [
        _node(dcid, arcs[dcid], others[dcid], provenances[dcid], members.get(dcid))
        for dcid in sorted(arcs)
    ]
    if not any(node.declared_by == "instance" for node in nodes):
        raise SchemaNoInstanceDeclarationError(len(nodes))
    terms = [
        SchemaTerm(
            dcid=dcid,
            name=display_name(term.names, dcid),
            types=sorted(term.kinds),
            provenances=sorted(term.provenances),
            declared_by=declared_by(term.provenances) if term.provenances else None,
        )
        for dcid, term in sorted(listed.items())
    ]
    return Schema(
        document=SchemaDocument(generation=generation, nodes=nodes),
        terms=SchemaTerms(generation=generation, terms=terms),
    )


def _list_members(
    reader: GraphNodeReader, enumerations: Sequence[str], *, deadline: float
) -> dict[str, tuple[tuple[ArcSubject, ...], bool]]:
    """Each of `enumerations`' `<-typeOf` subjects, and whether the listing read every one. One
    batched listing reads them all, and an enumeration the batch omits or answers with no
    subject is listed again alone, whose answer stands, as a forward read alone does."""
    if not enumerations:
        return {}
    batched = reader.node_arc_subjects_many(
        enumerations, property=_TYPE_OF, max_results=ENUMERATION_MEMBER_CAP, deadline=deadline
    )
    listed: dict[str, tuple[tuple[ArcSubject, ...], bool]] = {}
    for enumeration in enumerations:
        subjects, coverage = batched.get(enumeration, ((), "complete"))
        if subjects:
            listed[enumeration] = (subjects, coverage == "complete")
            continue
        page = reader.node_arc_subjects(
            enumeration, property=_TYPE_OF, max_results=ENUMERATION_MEMBER_CAP, deadline=deadline
        )
        listed[enumeration] = (page.subjects, not page.truncated)
    return listed


def _read_terms(reader: GraphNodeReader, *, deadline: float) -> dict[str, _ListedTerm]:
    """Every subject of `Class <-typeOf` and `Property <-typeOf`, merged by dcid."""
    listed: dict[str, _ListedTerm] = {}
    kinds: tuple[_TermKind, ...] = (_CLASS, _PROPERTY)
    for kind in kinds:
        page = reader.node_arc_subjects(
            kind, property=_TYPE_OF, max_results=TERM_LISTING_CAP, deadline=deadline
        )
        if page.truncated:
            raise SchemaListingTruncatedError(
                f"{kind} <-typeOf lists more than {TERM_LISTING_CAP} subjects"
            )
        for subject in page.subjects:
            term = listed.setdefault(subject.dcid, _ListedTerm())
            term.names.update(subject.names)
            term.kinds.add(kind)
            term.provenances.update(subject.provenances)
    return listed


def _index_named(code_index: CodeIndex) -> set[str]:
    """Every node a StatVar row names in its five fields, constraints or observation roles."""
    named: set[str] = set()
    for row in (*code_index.statvars, *code_index.listed_statvars):
        named.update(value for name in SHAPE_FIELDS if (value := getattr(row, name)))
        named.update(row.constraint_properties.keys())
        named.update(row.constraint_properties.values())
        named.update(row.observation_properties)
    return named


@dataclass(frozen=True, slots=True)
class _NodeReads:
    """The build's node reads, each filling `arcs`, `others` and `provenances` or `absent`,
    `left_out` with the count of values `others` leaves out by node and property, and
    `unsourced` with the nodes that have a `typeOf` target with no provenance. `listed` is every
    listed term, whose `typeOf` targets the closure does not follow."""

    reader: GraphNodeReader
    listed: Collection[str]
    arcs: dict[str, _Arcs]
    others: dict[str, _OtherProperties]
    left_out: dict[tuple[str, str], int]
    provenances: dict[str, frozenset[str]]
    absent: set[str]
    unsourced: set[str]
    deadline: float

    def nodes(self, dcids: Collection[str]) -> set[str]:
        """Read every outgoing property of `dcids` and the provenances declaring each value in
        one `node_all_arc_targets_many`, which probes every empty answer, so an answer of none is
        the graph's answer for the dcid read alone and the dcid joins `absent`. A dcid's
        provenances are those its `typeOf` targets carry, and a dcid with a target that carries
        none joins `unsourced`. Returns the dcids held.
        """
        read = self.reader.node_all_arc_targets_many(sorted(dcids), deadline=self.deadline)
        held: set[str] = set()
        for dcid in dcids:
            entry = read.get(dcid)
            if entry is None:
                self.absent.add(dcid)
                continue
            types = entry.get(_TYPE_OF, ())
            self.arcs[dcid] = {
                prop: tuple(target.value for target in entry.get(prop, ()))
                for prop in _NODE_PROPERTIES
            }
            self.others[dcid] = _other_properties(dcid, entry, self.left_out)
            if any(not target.provenances for target in types):
                self.unsourced.add(dcid)
            self.provenances[dcid] = frozenset(
                provenance for target in types for provenance in target.provenances
            )
            held.add(dcid)
        return held

    def close(self, frontier: set[str]) -> None:
        """Read `frontier` and each `_next_level` after it until no node is left unread."""
        while frontier := frontier - self.arcs.keys() - self.absent:
            frontier = _next_level(self.arcs, self.nodes(frontier), self.listed)


def _named(arcs: Mapping[str, _Arcs], dcids: Iterable[str], properties: Iterable[str]) -> set[str]:
    """The values `dcids` hold in `properties`, over the dcids read."""
    properties = tuple(properties)
    return {
        value for dcid in dcids if dcid in arcs for prop in properties for value in arcs[dcid][prop]
    }


def _next_level(
    arcs: Mapping[str, _Arcs], read: Collection[str], listed: Collection[str]
) -> set[str]:
    """The nodes the closure reads after `read`: their `subClassOf` and `subPropertyOf`
    parents, and the `typeOf` targets of those that are neither a Class nor a Property."""
    values = [dcid for dcid in read if dcid not in listed]
    return _named(arcs, read, _ANCESTOR_ARCS) | _named(arcs, values, (_TYPE_OF,))


def _reaches_enumeration(dcid: str, arcs: Mapping[str, _Arcs]) -> bool:
    """Whether `dcid`'s `subClassOf` chain, over the nodes read, reaches `Enumeration`."""
    seen: set[str] = set()
    stack = [dcid]
    while stack:
        current = stack.pop()
        if current in seen or current not in arcs:
            continue
        seen.add(current)
        parents = arcs[current]["subClassOf"]
        if _ENUMERATION in parents:
            return True
        stack.extend(parents)
    return False


def _other_properties(
    dcid: str, entry: Mapping[str, Sequence[ArcTarget]], left_out: dict[tuple[str, str], int]
) -> _OtherProperties:
    """Every property of `entry` outside `_NODE_PROPERTIES`, each target a node reference or a
    literal, properties and targets sorted. A literal past `OTHER_LITERAL_MAX_CHARS` and the
    values past the first `OTHER_VALUES_MAX` are left out and counted in `left_out` under
    `(dcid, property)`, and a property left with no value is left out."""
    others: _OtherProperties = {}
    for prop, targets in sorted(entry.items()):
        if prop in _NODE_PROPERTIES:
            continue
        kept = sorted(
            (
                target
                for target in targets
                if not target.literal or len(target.value) <= OTHER_LITERAL_MAX_CHARS
            ),
            key=lambda target: (target.value, target.literal),
        )
        if dropped := len(targets) - min(len(kept), OTHER_VALUES_MAX):
            left_out[(dcid, prop)] = dropped
        if kept:
            others[prop] = [
                SchemaLiteral(value=target.value)
                if target.literal
                else SchemaNodeReference(dcid=target.value)
                for target in kept[:OTHER_VALUES_MAX]
            ]
    return others


def _node(
    dcid: str,
    arcs: _Arcs,
    other_properties: _OtherProperties,
    provenances: Collection[str],
    enumeration: SchemaEnumeration | None,
) -> SchemaNode:
    return SchemaNode(
        dcid=dcid,
        name=display_name(arcs["name"], dcid),
        description=min(arcs["description"], default=None),
        types=sorted(arcs[_TYPE_OF]),
        alternate_name=sorted(arcs["alternateName"]),
        sub_class_of=sorted(arcs["subClassOf"]),
        sub_property_of=sorted(arcs["subPropertyOf"]),
        specialization_of=sorted(arcs["specializationOf"]),
        domain_includes=sorted(arcs["domainIncludes"]),
        range_includes=sorted(arcs["rangeIncludes"]),
        classification_code=sorted(arcs["classificationCode"]),
        other_properties=other_properties,
        provenances=sorted(provenances),
        declared_by=declared_by(provenances) if provenances else None,
        enumeration=enumeration,
    )
