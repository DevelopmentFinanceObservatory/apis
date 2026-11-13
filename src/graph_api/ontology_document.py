"""Build the instance's ontology document and the terms list from live graph reads.

`build_ontology` reads `Class <-typeOf` and `Property <-typeOf` once each; every subject is a
term. The document's seed is every term the instance declares plus every node a StatVar row
of the code index names. One `node_arcs_many` reads the seed's arcs, and each further level
reads the nodes not yet read that the instance's terms name in `subClassOf`,
`subPropertyOf`, `domainIncludes` or `rangeIncludes`, the `typeOf` targets of every node
read that is neither a Class nor a Property, and the `subClassOf` and `subPropertyOf`
parents, so the class of every value is present and every ancestor is too. One `node_arc_subjects_many` lists the members of every
instance-declared enumeration, one `node_arcs_many` reads the members not yet read, and one
`node_arc_targets_many` reads the declaring provenances of every node outside the listings.
The number of reads grows with the depth of the class and property hierarchy, never with the
number of nodes.

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
from graph_api.reader import GraphNodeReader

TERM_LISTING_CAP: Final[int] = 20_000
"""The most subjects one `<-typeOf` listing of `Class` or `Property` reads. A listing that
reaches it raises `OntologyListingTruncatedError`."""

ENUMERATION_MEMBER_CAP: Final[int] = 5_000
"""The most members read for one instance-declared enumeration. An enumeration with more is
served with the members read and `truncated: true`."""

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
    """`"base"` when there is at least one declaring provenance and every one starts
    `dc/base/`; `"instance"` otherwise."""
    if provenances and all(provenance.startswith(_BASE_PREFIX) for provenance in provenances):
        return "base"
    return "instance"


def build_ontology(
    *,
    reader: GraphNodeReader,
    code_index: CodeIndex,
    generation: str,
    deadline: float,
) -> Ontology:
    """Build the ontology document and the terms list.

    Args:
        reader: the graph reads.
        code_index: the bound code index, whose StatVar rows name nodes the document holds.
        generation: the graph generation token both bodies carry.
        deadline: the absolute monotonic deadline passed to every read.

    Raises:
        OntologyListingTruncatedError: a `Class` or `Property` listing reached its cap.
        GraphReadError: a read failed.
    """
    listed = _read_terms(reader, deadline=deadline)
    instance_terms = {
        dcid for dcid, term in listed.items() if declared_by(term.provenances) == "instance"
    }
    index_named = _index_named(code_index)

    arcs: dict[str, _Arcs] = {}
    absent: set[str] = set()
    _read_nodes(reader, instance_terms | index_named, arcs=arcs, absent=absent, deadline=deadline)
    frontier = _named(arcs, instance_terms, _SCHEMA_ARCS) | _next_level(arcs, arcs.keys(), listed)
    while frontier := frontier - arcs.keys() - absent:
        read = _read_nodes(reader, frontier, arcs=arcs, absent=absent, deadline=deadline)
        frontier = _next_level(arcs, read, listed)

    enumerations = sorted(
        dcid
        for dcid, term in listed.items()
        if dcid in arcs and _CLASS in term.kinds and _reaches_enumeration(dcid, arcs)
    )
    instance_enumerations = [dcid for dcid in enumerations if dcid in instance_terms]
    members: dict[str, OntologyEnumeration] = {}
    member_provenances: dict[str, set[str]] = {}
    if instance_enumerations:
        listings = reader.node_arc_subjects_many(
            instance_enumerations,
            property=_TYPE_OF,
            max_results=ENUMERATION_MEMBER_CAP,
            deadline=deadline,
        )
        for enumeration in instance_enumerations:
            subjects, coverage = listings[enumeration]
            for subject in subjects:
                member_provenances.setdefault(subject.dcid, set()).update(subject.provenances)
            members[enumeration] = OntologyEnumeration(
                members=sorted({subject.dcid for subject in subjects}),
                truncated=coverage != "complete",
            )
    for enumeration in enumerations:
        if enumeration not in members:
            members[enumeration] = OntologyEnumeration(
                members=sorted(
                    dcid
                    for dcid in index_named
                    if dcid in arcs and enumeration in arcs[dcid][_TYPE_OF]
                ),
                truncated=False,
            )
    if unread_members := member_provenances.keys() - arcs.keys() - absent:
        _read_nodes(reader, unread_members, arcs=arcs, absent=absent, deadline=deadline)

    provenances = {dcid: term.provenances for dcid, term in listed.items()} | member_provenances
    if unsourced := sorted(arcs.keys() - provenances.keys()):
        targets = reader.node_arc_targets_many(unsourced, properties=(_TYPE_OF,), deadline=deadline)
        for dcid in unsourced:
            by_property = targets.get(dcid) or {}
            provenances[dcid] = {
                provenance
                for target in by_property.get(_TYPE_OF, ())
                for provenance in target.provenances
            }

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
            if subject.name is not None:
                term.names.add(subject.name)
            term.kinds.add(kind)
            term.provenances.update(subject.provenances)
    return listed


def _index_named(code_index: CodeIndex) -> set[str]:
    """Every node a StatVar row names in its five fields, constraints or observation roles."""
    named: set[str] = set()
    for row in (*code_index.statvars, *code_index.listed_statvars):
        named.update(
            value
            for value in (
                row.population_type,
                row.measured_property,
                row.stat_type,
                row.measurement_qualifier,
                row.measurement_denominator,
            )
            if value
        )
        named.update(row.constraint_properties.keys())
        named.update(row.constraint_properties.values())
        named.update(row.observation_properties)
    return named


def _read_nodes(
    reader: GraphNodeReader,
    dcids: Collection[str],
    *,
    arcs: dict[str, _Arcs],
    absent: set[str],
    deadline: float,
) -> set[str]:
    """Read `dcids`' node properties in one batch into `arcs`, or `absent` where the graph
    holds no node. Returns the dcids held."""
    read = reader.node_arcs_many(sorted(dcids), properties=_NODE_PROPERTIES, deadline=deadline)
    held: set[str] = set()
    for dcid in dcids:
        entry = read.get(dcid)
        if entry is None:
            absent.add(dcid)
            continue
        arcs[dcid] = {prop: entry.get(prop, ()) for prop in _NODE_PROPERTIES}
        held.add(dcid)
    return held


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
        declared_by=declared_by(provenances),
        enumeration=enumeration,
    )
