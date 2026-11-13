"""Build the instance's ontology document and the terms list from live graph reads.

`build_ontology` reads `Class <-typeOf` and `Property <-typeOf` once each; every subject is a
term. The document's seed is every term the instance declares plus every node a StatVar row
of the code index names. One `node_arc_targets_many` with `confirm_empty` reads the seed's node
properties with the provenances that declare each value, and each further level reads the
nodes not yet read that the instance's terms name in `subClassOf`,
`subPropertyOf`, `domainIncludes` or `rangeIncludes`, the `typeOf` targets of every node
read that is neither a Class nor a Property, and the `subClassOf` and `subPropertyOf`
parents, so the class of every value is present and every ancestor is too. One
`node_arc_subjects_many` lists the members of every instance-declared enumeration, and the
members not yet read join the document with the same closure over their types and ancestors.
`confirm_empty` confirms an empty answer before returning it: one more read of the empties
together, then one read of each still empty alone, and a dcid still empty is absent from the
document. So the number of reads grows with the depth of the class and property hierarchy,
plus a group read per level with an empty answer and one read per dcid the graph does not hold.

Each node's values and provenances come from that one read of the node. Its provenances are
those declaring its `typeOf` arcs: every typed node carries at least one, and a node held with
no `typeOf` arc carries none and is declared by neither. A term's provenances come from its
listing entries. A listing subject with none, or a `typeOf` arc read with none, refuses the
build with `OntologyProvenanceUnreadError`.

An instance-declared enumeration's members are its `<-typeOf` listing, `complete` when the
listing read every subject. A base enumeration lists the members StatVar rows use and is never
`complete`.

Every node, member and list of values is sorted, so two builds from the same graph serialise
to the same bytes.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

from code_index import CodeIndex
from dc_graph import GraphReadError, display_name
from graph_api.ontology_models import (
    DeclaredBy,
    OntologyDocument,
    OntologyEnumeration,
    OntologyNode,
    OntologyTerm,
    OntologyTerms,
)
from graph_api.ontology_provenances import SHAPE_FIELDS
from graph_api.reader import GraphNodeReader

TERM_LISTING_CAP: Final[int] = 20_000
"""The most subjects one `<-typeOf` listing of `Class` or `Property` reads. A listing that
reaches it raises `OntologyListingTruncatedError`."""

ENUMERATION_MEMBER_CAP: Final[int] = 5_000
"""The most members read for one instance-declared enumeration. An enumeration with more is
served with the members read and `complete: false`."""

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
type _TermKind = Literal["Class", "Property"]


class OntologyListingTruncatedError(GraphReadError):
    """A `Class` or `Property` listing reached `TERM_LISTING_CAP`, so the terms are incomplete."""


class OntologyProvenanceUnreadError(GraphReadError):
    """`dcid`, a listing subject or a node with a `typeOf` arc, came back with no declaring
    provenance on any entry, so it cannot be labelled base or instance."""

    def __init__(self, dcid: str) -> None:
        super().__init__(f"no declaring provenance read for {dcid!r}")
        self.dcid = dcid


@dataclass(frozen=True)
class Ontology:
    """The ontology document and the terms list from one build."""

    document: OntologyDocument
    terms: OntologyTerms


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


def build_ontology(
    *,
    reader: GraphNodeReader,
    code_index: CodeIndex,
    generation: str | None,
    deadline: float,
) -> Ontology:
    """Build the ontology document and the terms list.

    Args:
        reader: the graph reads.
        code_index: the bound code index, whose StatVar rows name nodes the document holds.
        generation: the graph generation token both bodies carry, `None` where it is unknown.
        deadline: the absolute monotonic deadline passed to every read.

    Raises:
        OntologyListingTruncatedError: a `Class` or `Property` listing reached its cap.
        OntologyProvenanceUnreadError: a term or node has no declaring provenance.
        GraphReadError: a read failed.
    """
    listed = _read_terms(reader, deadline=deadline)
    instance_terms = {
        dcid for dcid, term in listed.items() if declared_by(term.provenances) == "instance"
    }
    index_named = _index_named(code_index)

    arcs: dict[str, _Arcs] = {}
    # Each node's provenances, read with its arcs.
    provenances: dict[str, frozenset[str]] = {}
    absent: set[str] = set()
    read = _NodeReads(
        reader,
        listed=listed,
        arcs=arcs,
        provenances=provenances,
        absent=absent,
        deadline=deadline,
    )
    read.nodes(instance_terms | index_named)
    read.close(_named(arcs, instance_terms, _SCHEMA_ARCS) | _next_level(arcs, arcs.keys(), listed))

    instance_enumerations = sorted(
        dcid
        for dcid in instance_terms
        if dcid in arcs and _CLASS in listed[dcid].kinds and _reaches_enumeration(dcid, arcs)
    )
    members: dict[str, OntologyEnumeration] = {}
    listed_members: set[str] = set()
    if instance_enumerations:
        listings = reader.node_arc_subjects_many(
            instance_enumerations,
            property=_TYPE_OF,
            max_results=ENUMERATION_MEMBER_CAP,
            deadline=deadline,
        )
        for enumeration in instance_enumerations:
            subjects, coverage = listings[enumeration]
            listed_members.update(subject.dcid for subject in subjects)
            members[enumeration] = OntologyEnumeration(
                members=sorted({subject.dcid for subject in subjects}),
                complete=coverage == "complete",
            )
    if unread_members := listed_members - arcs.keys() - absent:
        read.close(_next_level(arcs, read.nodes(unread_members), listed))

    enumerations = (
        dcid
        for dcid, term in listed.items()
        if dcid in arcs and _CLASS in term.kinds and _reaches_enumeration(dcid, arcs)
    )
    for enumeration in enumerations:
        if enumeration not in members:
            members[enumeration] = OntologyEnumeration(
                members=sorted(
                    dcid
                    for dcid in index_named
                    if dcid in arcs and enumeration in arcs[dcid][_TYPE_OF]
                ),
                complete=False,
            )

    nodes = [_node(dcid, arcs[dcid], provenances[dcid], members.get(dcid)) for dcid in sorted(arcs)]
    terms = [
        OntologyTerm(
            dcid=dcid,
            name=display_name(term.names, dcid),
            types=sorted(term.kinds),
            provenances=sorted(term.provenances),
            declared_by=declared_by(term.provenances),
        )
        for dcid, term in sorted(listed.items())
    ]
    return Ontology(
        document=OntologyDocument(generation=generation, nodes=nodes),
        terms=OntologyTerms(generation=generation, terms=terms),
    )


def _read_terms(reader: GraphNodeReader, *, deadline: float) -> dict[str, _ListedTerm]:
    """Every subject of `Class <-typeOf` and `Property <-typeOf`, merged by dcid."""
    listed: dict[str, _ListedTerm] = {}
    kinds: tuple[_TermKind, ...] = (_CLASS, _PROPERTY)
    for kind in kinds:
        page = reader.node_arc_subjects(
            kind, property=_TYPE_OF, max_results=TERM_LISTING_CAP, deadline=deadline
        )
        if page.truncated:
            raise OntologyListingTruncatedError(
                f"{kind} <-typeOf lists more than {TERM_LISTING_CAP} subjects"
            )
        for subject in page.subjects:
            term = listed.setdefault(subject.dcid, _ListedTerm())
            term.names.update(subject.names)
            term.kinds.add(kind)
            term.provenances.update(subject.provenances)
    if unsourced := sorted(dcid for dcid, term in listed.items() if not term.provenances):
        raise OntologyProvenanceUnreadError(unsourced[0])
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
    """The build's node reads, each filling `arcs` and `provenances` or `absent`. `listed` is
    every listed term, whose `typeOf` targets the closure does not follow."""

    reader: GraphNodeReader
    listed: Collection[str]
    arcs: dict[str, _Arcs]
    provenances: dict[str, frozenset[str]]
    absent: set[str]
    deadline: float

    def nodes(self, dcids: Collection[str]) -> set[str]:
        """Read `dcids`' node properties and the provenances declaring each value in one
        `node_arc_targets_many` with `confirm_empty`, so an answer of none is the graph's
        answer for the dcid read alone and the dcid joins `absent`. Returns the dcids held.

        Raises:
            OntologyProvenanceUnreadError: a `typeOf` target of a dcid carries no provenance.
        """
        read = self.reader.node_arc_targets_many(
            sorted(dcids), properties=_NODE_PROPERTIES, confirm_empty=True, deadline=self.deadline
        )
        held: set[str] = set()
        for dcid in dcids:
            entry = read.get(dcid)
            if entry is None:
                self.absent.add(dcid)
                continue
            types = entry.get(_TYPE_OF, ())
            if any(not target.provenances for target in types):
                raise OntologyProvenanceUnreadError(dcid)
            self.arcs[dcid] = {
                prop: tuple(target.value for target in entry.get(prop, ()))
                for prop in _NODE_PROPERTIES
            }
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


def _node(
    dcid: str,
    arcs: _Arcs,
    provenances: Collection[str],
    enumeration: OntologyEnumeration | None,
) -> OntologyNode:
    return OntologyNode(
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
        provenances=sorted(provenances),
        declared_by=declared_by(provenances) if provenances else None,
        enumeration=enumeration,
    )
