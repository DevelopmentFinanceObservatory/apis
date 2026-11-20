"""Build the deployment's schema document and the terms list from live graph reads.

`build_schema` reads `Class <-typeOf` and `Property <-typeOf` once each, and every subject is a
term. The document's seed is every term DFO declares plus every node a StatVar row of the code
index names. `node_all_arc_targets_many` reads every outgoing property of each node with the
provenances that declare each value, and the closure reads the nodes not yet read that DFO's
terms name in `subClassOf`, `subPropertyOf`, `domainIncludes` or `rangeIncludes`, the `typeOf`
targets of every node read that is neither a Class nor a Property, and the `subClassOf` and
`subPropertyOf` parents, so the class of every value is present and every ancestor is too. One
`node_arc_subjects` listing per DFO-declared Class reads its instances, and the instances of
each that is a value list join the document with the same closure over their types and
ancestors. Each DFO value list's descendants are read too: the `<-subClassOf` subjects of
each DFO value list are listed, then those of the subclasses a listing finds as soon as that
listing lands, until no new class appears. Every subclass found joins the document with the
same closure. `node_arc_subjects_many` lists the classes that become ready together in
listings of at most 12 classes, each class listed once.

A range class is a class the `rangeIncludes` of a DFO-declared Property names. A DFO range is a
value list, so its descendants are listed and join as above. A base range's descendants are
listed the same way, and they join the document with the same closure only when the listings
cover its whole closure, none is truncated and it holds at most `RANGE_CLOSURE_CAP`
descendants. A base range past the cap lists no more once the classes found pass it and adds
none of its listed classes. Each class is listed once, whichever closures it is in.
`Schema.closed_ranges` records the range classes whose whole descendant tree the document
holds.

The reads run as a pipeline, at most `BUILD_CONCURRENCY` at once: the index-named nodes are read
while the term listings run, each node read of at most 12 dcids is expanded the moment it lands,
every DFO Class's instances are listed as soon as the terms are known, a value list's instances and
subclasses are listed and read once the nodes read make it one, and a base range's descendants
are listed once a DFO term naming it is read. The nodes read and the closed ranges follow from
the graph alone, so the order the answers arrive in changes nothing in the document. Each node
read probes every empty answer: it reads each empty dcid again alone, and a dcid empty after
that is absent from the document. The node reads are uncached, since no other
caller reads every property of these nodes. A read that fails ends the build with its error at
once, while the reads already in flight finish on their own threads and land nowhere. One
log line per build gives each phase's read count and when its first read started and last read
ended: the term listings, the seed, each closure level, the instance listings, the
instances, the subclass listings, the subclasses, the range listings and the range classes.

Each node's values and provenances come from that one read of the node. The ten properties
`SchemaNode` names are its named members, and every other property the read answers is in
`other_properties`, each target a node reference or a literal, less any literal past
`OTHER_LITERAL_MAX_CHARS` and the values past the first `OTHER_VALUES_MAX` of a property, which
one warning per build counts. Its provenances are
those its `typeOf` targets carry, and `declared_by` follows from them by the origin each
belongs to: the origin catalogue's provenance table names it where it lists the provenance, and
the id's prefix decides otherwise, `dc/base/` for base Data Commons, `undata/` for the UN
origin and any other for DFO. The table lists the data provenances the catalogue admits and none
of the ontology's, such as `provenance/DFOOntology`, so those take the prefix rule. A node
declared by any DFO provenance is `"dfo"`, one whose provenances are all base is `"base"`, and
any other is `"un_system"`. A node with no `typeOf` arc, or whose `typeOf` targets carry no
provenance, is served with none and `declared_by` null. A term's provenances come from its listing entries by the same rule. One warning per
build counts the terms served with none and the nodes with a `typeOf` target that carries none,
and names the first ten of each. A build whose document holds no node DFO declares refuses with
`SchemaNoDfoDeclarationError`.

A value list is a Class whose instances the document lists. A DFO-declared Class is one when
its `subClassOf` chain reaches `Enumeration`, or when it or an ancestor is a DFO-declared Class
that the `rangeIncludes` of a DFO-declared Property names, so a DFO subclass of a DFO range
lists too. A base Class is one only when its chain reaches `Enumeration`, so neither
`Enumeration` itself nor a base range such as `Thing` or `Country` is. A DFO list covers itself
and every descendant its `<-subClassOf` listings find. It holds the subjects of the `<-typeOf`
listings of the DFO-declared classes it covers and the nodes StatVar rows use typed by a base
one. It is `complete` when it covers no base class and every listing, of instances or of
subclasses, read every subject. A base list holds
the nodes StatVar rows use typed by it or any subclass and is always `complete: false`. A list
that holds no node and is not complete is left out.

Every node, instance and list of values is sorted, so two builds from the same graph serialise
to the same bytes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
from dataclasses import dataclass, field
from functools import partial
from typing import Final, Literal

from code_index import CodeIndex
from dc_graph import (
    UNRESOLVED_NODE,
    ArcSubject,
    ArcSubjectPage,
    ArcTarget,
    GraphReadError,
    SubjectsCoverageFlag,
    display_name,
)
from dfo_platform.executor import ContextCarryingExecutor
from graph_api.dataset_usage import SHAPE_FIELDS
from graph_api.reader import GraphNodeReader
from graph_api.schema_models import (
    DeclaredBy,
    SchemaDocument,
    SchemaInstances,
    SchemaLiteral,
    SchemaNode,
    SchemaNodeReference,
    SchemaTerm,
    SchemaTerms,
)
from origin_catalogue import Catalogue, OriginId
from origin_catalogue.origins import UN_ID_PREFIX

logger = logging.getLogger(__name__)

TERM_LISTING_CAP: Final[int] = 20_000
"""The most subjects one `<-typeOf` listing of `Class` or `Property` reads. A listing that
reaches it raises `SchemaListingTruncatedError`."""

INSTANCE_LISTING_CAP: Final[int] = 5_000
"""The most subjects one `<-typeOf` listing of a DFO-declared Class reads. A value list whose
class or subclass has more is served with the subjects read and `complete: false`."""

OTHER_LITERAL_MAX_CHARS: Final[int] = 2_000
"""The longest literal `other_properties` holds. A longer one, such as a geometry, is left out."""

OTHER_VALUES_MAX: Final[int] = 100
"""The most values `other_properties` holds for one property: the first of them sorted."""

BUILD_CONCURRENCY: Final[int] = 24
"""The most reads one build runs at once. Each node read is one request, or one at a time when
the mixer answers a continuation, and each listing is one, so the build has at most this many
requests in flight beside the confirmation probes, which the process-wide pool of 16 probe
threads bounds (`dc_graph.client._PROBES`)."""

_NODE_READ_CHUNK: Final[int] = 12
"""The most dcids one node read names: `dc_graph.GraphClient`'s default `->*` chunk, so each
read is one request. A chunk of 12 schema nodes carries 65 to 124 arc values, well inside the
mixer's page of about 500."""

RANGE_CLOSURE_CAP: Final[int] = 500
"""The most descendant classes a base range class's closure joins the document with. A base
range with more adds none of its listed classes, its listings stop once the classes found pass
the cap, and it is no closed range. On the dev graph `Organization` has 176 descendants, `Class`,
`Number`, `Text`, `Country` and `Provenance` 0 to 2 each, and `Thing` 1,957."""

SUBCLASS_LISTING_CAP: Final[int] = 1_000
"""The most `<-subClassOf` subjects a listing reads for one class, in a `node_arc_subjects_many`
request of at most `_NODE_READ_CHUNK` classes. A value list covering a class with more is served
with the subclasses read and `complete: false`, and no range covering it is closed.

A build lists each class in the descendant closures of the DFO value lists and the base
ranges once, and the build's deadline bounds how many it lists."""

_TERM_LISTINGS: Final[str] = "term listings"
_SEED: Final[str] = "seed"
_INSTANCE_LISTINGS: Final[str] = "instance listings"
_INSTANCES: Final[str] = "instances"
_SUBCLASS_LISTINGS: Final[str] = "subclass listings"
_SUBCLASSES: Final[str] = "subclasses"
_RANGE_LISTINGS: Final[str] = "range listings"
_RANGE_CLASSES: Final[str] = "range classes"

_LOGGED_DCIDS: Final[int] = 10
_BASE_PREFIX: Final[str] = "dc/base/"
_CLASS: Final = "Class"
_PROPERTY: Final = "Property"
_ENUMERATION: Final[str] = "Enumeration"

_TYPE_OF: Final[str] = "typeOf"
_SUB_CLASS_OF: Final[str] = "subClassOf"
_RANGE_INCLUDES: Final[str] = "rangeIncludes"
_DOMAIN_INCLUDES: Final[str] = "domainIncludes"
_ANCESTOR_ARCS: Final[tuple[str, ...]] = ("subClassOf", "subPropertyOf")
_SCHEMA_ARCS: Final[tuple[str, ...]] = (*_ANCESTOR_ARCS, "domainIncludes", "rangeIncludes")
NODE_PROPERTIES: Final[tuple[str, ...]] = (
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
"""The properties a `SchemaNode` holds in fields of its own. Every other is in
`other_properties`."""

type _Arcs = dict[str, tuple[str, ...]]
type _OtherProperties = dict[str, list[SchemaNodeReference | SchemaLiteral]]
type TermKind = Literal["Class", "Property"]


class SchemaListingTruncatedError(GraphReadError):
    """A `Class` or `Property` listing reached `TERM_LISTING_CAP`, so the terms are incomplete."""


class SchemaNoDfoDeclarationError(GraphReadError):
    """No node of the built document is declared by DFO, so the graph named no DFO provenance
    for any of them and the check would judge no range or domain."""

    def __init__(self, node_count: int) -> None:
        super().__init__(
            f"none of the {node_count} nodes of the schema document is declared by DFO"
        )
        self.node_count = node_count


@dataclass(frozen=True)
class Schema:
    """The schema document and the terms list from one build, with what the build read that
    neither body serves: the nodes whose read answered a `typeOf` target (`unresolved_types`),
    a `subClassOf` target (`unresolved_parents`), a `rangeIncludes` target
    (`unresolved_ranges`) or a `domainIncludes` target (`unresolved_domains`) the graph cannot
    resolve, the ids the graph answered with no node and no unresolved target (`absent`), and
    the range classes whose whole descendant tree the document holds (`closed_ranges`).

    A range class is a class the `rangeIncludes` of a DFO-declared Property names. It is closed
    when the build listed the `<-subClassOf` subjects of it and of every descendant, no listing
    was truncated, and the document holds it and every descendant: a DFO range by the
    descendant listing of its value list, a base range by its own listings within
    `RANGE_CLOSURE_CAP`. A value typed by no class of a closed range's tree is outside it."""

    document: SchemaDocument
    terms: SchemaTerms
    unresolved_types: frozenset[str]
    unresolved_parents: frozenset[str]
    unresolved_ranges: frozenset[str]
    unresolved_domains: frozenset[str]
    absent: frozenset[str]
    closed_ranges: frozenset[str]


@dataclass
class ListedTerm:
    """One subject of the `Class` and `Property` listings, with the names, kinds and
    provenances of its entries."""

    names: set[str] = field(default_factory=set)
    kinds: set[TermKind] = field(default_factory=set)
    provenances: set[str] = field(default_factory=set)


def fold_listings(
    listings: Iterable[tuple[TermKind, Iterable[ArcSubject]]],
) -> dict[str, ListedTerm]:
    """Fold `listings`, pairs of a kind and the `<-typeOf` subjects of that kind, into one
    `ListedTerm` per subject, with every name and provenance its entries carry and every kind
    that lists it."""
    listed: dict[str, ListedTerm] = {}
    for kind, subjects in listings:
        for subject in subjects:
            term = listed.setdefault(subject.dcid, ListedTerm())
            term.names.update(subject.names)
            term.kinds.add(kind)
            term.provenances.update(subject.provenances)
    return listed


def schema_term(dcid: str, term: ListedTerm, *, declared_by: DeclaredBy | None) -> SchemaTerm:
    """`dcid` as a `SchemaTerm` of the terms list, labelled by `dc_graph.display_name`."""
    return SchemaTerm(
        ref=dcid,
        label=display_name(term.names, dcid),
        types=sorted(term.kinds),
        provenances=sorted(term.provenances),
        declared_by=declared_by,
    )


def provenance_origin(provenance: str, catalogue: Catalogue) -> OriginId:
    """The origin `provenance` belongs to: the one `catalogue`'s provenance table names, else
    `"base"` for an id under `dc/base/`, `"un_system"` for one under `undata/` and `"dfo"` for
    any other."""
    if (row := catalogue.provenance(provenance)) is not None:
        return row.data_commons.id
    if provenance.startswith(_BASE_PREFIX):
        return "base"
    if provenance.startswith(UN_ID_PREFIX):
        return "un_system"
    return "dfo"


def declared_by(provenances: Collection[str], *, catalogue: Catalogue) -> DeclaredBy:
    """`"dfo"` when any declaring provenance is DFO's, `"base"` when every one is base Data
    Commons', `"un_system"` otherwise, each by `provenance_origin`.

    Raises:
        ValueError: `provenances` is empty, which names no origin.
    """
    if not provenances:
        raise ValueError("a term with no declaring provenance has no origin")
    origins = {provenance_origin(provenance, catalogue) for provenance in provenances}
    if "dfo" in origins:
        return "dfo"
    if origins == {"base"}:
        return "base"
    return "un_system"


def build_schema(
    *,
    reader: GraphNodeReader,
    code_index: CodeIndex,
    catalogue: Catalogue,
    generation: str | None,
    deadline: float,
) -> Schema:
    """Build the schema document and the terms list.

    Args:
        reader: the graph reads.
        code_index: the bound code index, whose StatVar rows name nodes the document holds.
        catalogue: the origin catalogue whose provenance table names the origin of the
            provenances it lists.
        generation: the graph generation token both bodies carry, `None` where it is unknown.
        deadline: the absolute monotonic deadline passed to every read.

    Raises:
        SchemaListingTruncatedError: a `Class` or `Property` listing reached its cap.
        SchemaNoDfoDeclarationError: no node of the document is declared by DFO.
        GraphReadError: a read failed.
    """
    started = time.monotonic()
    index_named = _index_named(code_index)
    executor = ContextCarryingExecutor(
        max_workers=BUILD_CONCURRENCY, thread_name_prefix="schema-build"
    )
    reads = _Reads(reader, executor=executor, catalogue=catalogue, deadline=deadline)
    try:
        reads.run(index_named)
    finally:
        # A failed read raises at once: the reads still in flight finish on their own threads,
        # and their answers land nowhere, since only `run` lands them.
        executor.shutdown(wait=False, cancel_futures=True)
    read_at = time.monotonic()
    listed, arcs = reads.listed, reads.arcs

    instances = _instances(reads, index_named)

    unsourced_terms = sorted(dcid for dcid, term in listed.items() if not term.provenances)
    if unsourced_terms or reads.unsourced:
        logger.warning(
            "schema build read %d terms with no listed provenance, served with declared_by null, "
            "among them %s, and %d nodes with a typeOf target that carries no provenance, among "
            "them %s",
            len(unsourced_terms),
            unsourced_terms[:_LOGGED_DCIDS],
            len(reads.unsourced),
            sorted(reads.unsourced)[:_LOGGED_DCIDS],
        )
    if reads.left_out:
        logger.warning(
            "schema build left %d values out of other_properties, literals past %d characters "
            "or values past %d a property, among them on %s",
            sum(reads.left_out.values()),
            OTHER_LITERAL_MAX_CHARS,
            OTHER_VALUES_MAX,
            sorted(reads.left_out)[:_LOGGED_DCIDS],
        )
    nodes = [
        schema_node(
            dcid,
            arcs[dcid],
            reads.others[dcid],
            reads.provenances[dcid],
            instances.get(dcid),
            catalogue,
        )
        for dcid in sorted(arcs)
    ]
    if not any(node.declared_by == "dfo" for node in nodes):
        raise SchemaNoDfoDeclarationError(len(nodes))
    terms = [
        schema_term(
            dcid,
            term,
            declared_by=(
                declared_by(term.provenances, catalogue=catalogue) if term.provenances else None
            ),
        )
        for dcid, term in sorted(listed.items())
    ]
    schema = Schema(
        document=SchemaDocument(generation=generation, nodes=nodes),
        terms=SchemaTerms(generation=generation, terms=terms),
        unresolved_types=frozenset(reads.unresolved[_TYPE_OF]),
        unresolved_parents=frozenset(reads.unresolved[_SUB_CLASS_OF]),
        unresolved_ranges=frozenset(reads.unresolved[_RANGE_INCLUDES]),
        unresolved_domains=frozenset(reads.unresolved[_DOMAIN_INCLUDES]),
        absent=frozenset(reads.absent),
        closed_ranges=reads.closed_ranges(),
    )
    reads.log_timings(started=started, read_at=read_at, ended=time.monotonic())
    return schema


def _index_named(code_index: CodeIndex) -> set[str]:
    """Every node a StatVar row names in its five fields, constraints or observation roles."""
    named: set[str] = set()
    for row in (*code_index.statvars, *code_index.listed_statvars):
        named.update(value for name in SHAPE_FIELDS if (value := getattr(row, name)))
        named.update(row.constraint_properties.keys())
        named.update(row.constraint_properties.values())
        named.update(row.observation_properties)
    return named


type _NodeEntries = Mapping[str, Mapping[str, Sequence[ArcTarget]] | None]
type _Landing = Callable[[], None]


@dataclass(slots=True)
class _PhaseTiming:
    reads: int = 0
    first_start: float = float("inf")
    last_end: float = 0.0
    longest: float = 0.0

    def add(self, started: float, ended: float) -> None:
        self.reads += 1
        self.first_start = min(self.first_start, started)
        self.last_end = max(self.last_end, ended)
        self.longest = max(self.longest, ended - started)


class _Reads:
    """The build's graph reads, each started as soon as what it depends on has landed, at most
    `BUILD_CONCURRENCY` at once on `executor`.

    The `Class` and `Property` listings and the reads of the index-named nodes start together.
    Once both listings land, the DFO terms are read and every DFO Class's instances are listed.
    Each node read is expanded as it lands: its `subClassOf` and `subPropertyOf` parents, the
    `typeOf` targets of a node that is no listed term, and the four schema arcs of a DFO term
    are read next, unless already read or under way. A node read that lands before the listings
    is expanded when they land. A DFO Class's instances are read, and expanded the same way, once
    its listing has landed and the nodes read make it a value list (`lists_in_full`), a test
    run again after every landing. Once a DFO Class is a value list, its `<-subClassOf`
    subjects are listed; each subject is read and expanded the same way, and its own
    `<-subClassOf` subjects listed, until no new class appears. The classes whose listing a
    landing makes due are listed together by `node_arc_subjects_many`, at most
    `_NODE_READ_CHUNK` a request, and a class whose coverage is not `"complete"` counts as a
    truncated listing. Each base range class, one the
    `rangeIncludes` of a DFO term names that DFO does not declare, has its descendants listed
    the same way, each listing as soon as the classes found so far are known, but its classes
    are read only once its whole closure is listed: when no listing in it was truncated and it
    holds at most `RANGE_CLOSURE_CAP` descendants, every class in it is read and expanded the
    same way. Once a listing is truncated or the classes found pass the cap, the range lists
    no more and reads none of them. Each class is listed once, whichever closures it is in,
    and each node is read once, in reads of at most `_NODE_READ_CHUNK` dcids; the reads end
    when none is left to start.

    The successor rule depends only on a node's own read and the listings, `lists_in_full`
    only grows as nodes land, and a subclass listing's subjects depend only on the class
    listed, whichever request lists it with which others. A base range's classes join only
    when its whole closure is known, and whether it
    is known within the cap follows from the graph alone: a closure within the cap with no
    truncated listing is listed in full whatever the order, and any other passes the cap or
    meets a truncated listing before it is. The listings of a range past the cap that land
    before it stops depend on the order, and they read no node. So every rule that reads a
    node only adds reads as answers land, and the build stops at the one set they all close
    over: the nodes read and `closed_ranges` are the same whatever order the answers arrive
    in. The first read that raises cancels the reads not yet started and its error propagates
    at once, with no wait for the reads in flight.
    """

    def __init__(
        self,
        reader: GraphNodeReader,
        *,
        executor: Executor,
        catalogue: Catalogue,
        deadline: float,
    ) -> None:
        self._reader = reader
        self._executor = executor
        self._catalogue = catalogue
        self._deadline = deadline
        self._term_pages: dict[str, ArcSubjectPage] = {}
        self._listed: dict[str, ListedTerm] | None = None
        self.dfo_terms: set[str] = set()
        self.arcs: dict[str, _Arcs] = {}
        self.others: dict[str, _OtherProperties] = {}
        # The values left out of `others`, by node and property.
        self.left_out: dict[tuple[str, str], int] = {}
        # Each node's provenances, read with its arcs.
        self.provenances: dict[str, frozenset[str]] = {}
        self.absent: set[str] = set()
        # The nodes whose `typeOf`, `subClassOf`, `rangeIncludes` or `domainIncludes` held a
        # target the graph cannot resolve, by property.
        self.unresolved: dict[str, set[str]] = {
            prop: set() for prop in (_TYPE_OF, _SUB_CLASS_OF, _RANGE_INCLUDES, _DOMAIN_INCLUDES)
        }
        self.unsourced: set[str] = set()
        # Each DFO Class's `<-typeOf` subjects, and whether the listing read every one.
        self.instance_pages: dict[str, tuple[tuple[ArcSubject, ...], bool]] = {}
        self._instances_read: set[str] = set()
        # Each listed class's `<-subClassOf` subjects, and whether the listing read every one.
        self.subclass_pages: dict[str, tuple[tuple[str, ...], bool]] = {}
        self._subclasses_listed: set[str] = set()
        # The DFO value lists and every descendant their listings found, whose subclasses the
        # build reads.
        self._dfo_closure: set[str] = set()
        # Every class the `rangeIncludes` of a DFO term read names.
        self.ranges: set[str] = set()
        # The base ranges whose listing is over, and those of them whose closure joined.
        self._ranges_settled: set[str] = set()
        self._ranges_joined: set[str] = set()
        self._ranges_changed = False
        self._submitted: set[str] = set()
        self._unexpanded: list[tuple[set[str], str]] = []
        # Each read under way, whose answer is the landing the build runs on its own thread.
        self._pending: set[Future[_Landing]] = set()
        self._phases: dict[str, _PhaseTiming] = {}
        self._phases_lock = threading.Lock()

    @property
    def listed(self) -> dict[str, ListedTerm]:
        """Every listed term, once both listings have landed."""
        if self._listed is None:
            raise RuntimeError("the term listings have not landed")
        return self._listed

    def run(self, index_named: Collection[str]) -> None:
        """Start the listings and the index-named reads, and land every read until none is left.

        Raises:
            SchemaListingTruncatedError: a `Class` or `Property` listing reached its cap.
            GraphReadError: a read failed.
        """
        for kind in (_CLASS, _PROPERTY):
            self._list(kind, cap=TERM_LISTING_CAP, phase=_TERM_LISTINGS)
        self._read_nodes(index_named, phase=_SEED)
        try:
            while self._pending:
                done, self._pending = wait(self._pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()()
                if self._listed is not None:
                    self._read_value_lists()
                    self._list_base_ranges()
        except BaseException:
            for future in self._pending:
                future.cancel()
            raise

    def log_timings(self, *, started: float, read_at: float, ended: float) -> None:
        """Log how long each phase's reads took, by when its first read started and its last
        ended, in milliseconds from the start of the build."""

        def ms(at: float) -> int:
            return round((at - started) * 1_000)

        phases = sorted(self._phases.items(), key=lambda item: item[1].first_start)
        logger.info(
            "schema build took %d ms, reads ended at %d ms: %s; assembly %d ms",
            ms(ended),
            ms(read_at),
            "; ".join(
                f"{name} {timing.reads} reads {ms(timing.first_start)}-{ms(timing.last_end)} ms "
                f"longest {round(timing.longest * 1_000)} ms"
                for name, timing in phases
            ),
            round((ended - read_at) * 1_000),
        )

    def _list(self, dcid: str, *, cap: int, phase: str) -> None:
        def read() -> _Landing:
            started = time.monotonic()
            page = self._reader.node_arc_subjects(
                dcid, property=_TYPE_OF, max_results=cap, deadline=self._deadline
            )
            self._record(phase, started)
            return partial(self._land_listing, page, phase=phase, dcid=dcid)

        self._pending.add(self._executor.submit(read))

    def _read_nodes(self, dcids: Iterable[str], *, phase: str) -> None:
        fresh = sorted(set(dcids) - self._submitted)
        self._submitted.update(fresh)
        for start in range(0, len(fresh), _NODE_READ_CHUNK):
            chunk = tuple(fresh[start : start + _NODE_READ_CHUNK])

            def read(chunk: tuple[str, ...] = chunk) -> _Landing:
                started = time.monotonic()
                entries = self._reader.node_all_arc_targets_many(chunk, deadline=self._deadline)
                self._record(phase, started)
                return partial(self._land_nodes, entries, dcids=chunk, phase=phase)

            self._pending.add(self._executor.submit(read))

    def _list_subclasses(self, dcids: Iterable[str], *, phase: str) -> None:
        """List the `<-subClassOf` subjects of each of `dcids` not yet listed, in listings of
        at most `_NODE_READ_CHUNK` classes."""
        fresh = sorted(set(dcids) - self._subclasses_listed)
        self._subclasses_listed.update(fresh)
        for start in range(0, len(fresh), _NODE_READ_CHUNK):
            chunk = tuple(fresh[start : start + _NODE_READ_CHUNK])

            def read(chunk: tuple[str, ...] = chunk) -> _Landing:
                started = time.monotonic()
                pages = self._reader.node_arc_subjects_many(
                    chunk,
                    property=_SUB_CLASS_OF,
                    max_results=SUBCLASS_LISTING_CAP,
                    deadline=self._deadline,
                )
                self._record(phase, started)
                return partial(self._land_subclasses, pages, dcids=chunk)

            self._pending.add(self._executor.submit(read))

    def _record(self, phase: str, started: float) -> None:
        with self._phases_lock:
            self._phases.setdefault(phase, _PhaseTiming()).add(started, time.monotonic())

    def _land_listing(self, page: ArcSubjectPage, *, phase: str, dcid: str) -> None:
        if phase == _INSTANCE_LISTINGS:
            self.instance_pages[dcid] = (page.subjects, not page.truncated)
            return
        if page.truncated:
            raise SchemaListingTruncatedError(
                f"{dcid} <-typeOf lists more than {TERM_LISTING_CAP} subjects"
            )
        self._term_pages[dcid] = page
        if len(self._term_pages) == 2:
            self._land_terms()

    def _land_subclasses(
        self,
        pages: Mapping[str, tuple[Sequence[ArcSubject], SubjectsCoverageFlag]],
        *,
        dcids: tuple[str, ...],
    ) -> None:
        """Keep each class's subclasses, read in full when its coverage is `"complete"`, and
        read and join to the DFO closure the subclasses of each class in it."""
        found: set[str] = set()
        for dcid in dcids:
            subjects, coverage = pages[dcid]
            subclasses = tuple(sorted({subject.dcid for subject in subjects}))
            self.subclass_pages[dcid] = (subclasses, coverage == "complete")
            if dcid in self._dfo_closure:
                found.update(subclasses)
        self._ranges_changed = True
        self._read_nodes(found, phase=_SUBCLASSES)
        self._join_dfo_closure(found)

    def _land_terms(self) -> None:
        listed = fold_listings(
            (kind, self._term_pages[kind].subjects) for kind in (_CLASS, _PROPERTY)
        )
        self._listed = listed
        self.dfo_terms = {
            dcid
            for dcid, term in listed.items()
            if term.provenances
            and declared_by(term.provenances, catalogue=self._catalogue) == "dfo"
        }
        self._read_nodes(self.dfo_terms, phase=_SEED)
        for dcid in sorted(self.dfo_terms):
            if _CLASS in listed[dcid].kinds:
                self._list(dcid, cap=INSTANCE_LISTING_CAP, phase=_INSTANCE_LISTINGS)
        for held, phase in self._unexpanded:
            self._expand(held, phase=phase)
        self._unexpanded.clear()

    def _land_nodes(self, entries: _NodeEntries, *, dcids: tuple[str, ...], phase: str) -> None:
        """Keep each node's arcs, other properties and provenances, or mark it absent: the read
        confirms every empty answer alone, so an answer of none is the graph's answer for the
        dcid read alone. A node's provenances are those its `typeOf` targets carry, and a node
        with a target that carries none joins `unsourced`. A node's arcs keep each distinct
        target text once, in first-occurrence order, where a node reference and a literal share
        it. A target the graph cannot resolve (`UNRESOLVED_NODE`) is served nowhere: it marks
        the node in `unresolved` under its property when that is `typeOf`, `subClassOf`,
        `rangeIncludes` or `domainIncludes`. A
        node answered with nothing is `absent`; one answered only with unresolved targets is
        neither held nor absent."""
        held: set[str] = set()
        for dcid in dcids:
            read = entries.get(dcid)
            if read is None:
                self.absent.add(dcid)
                continue
            for prop, marked in self.unresolved.items():
                if any(target.value == UNRESOLVED_NODE for target in read.get(prop, ())):
                    marked.add(dcid)
            if (entry := resolved_entry(read)) is None:
                continue
            types = entry.get(_TYPE_OF, ())
            self.arcs[dcid] = {
                prop: tuple(dict.fromkeys(target.value for target in entry.get(prop, ())))
                for prop in NODE_PROPERTIES
            }
            self.others[dcid] = node_other_properties(dcid, entry, self.left_out)
            if any(not target.provenances for target in types):
                self.unsourced.add(dcid)
            self.provenances[dcid] = frozenset(
                provenance for target in types for provenance in target.provenances
            )
            held.add(dcid)
        if self._listed is None:
            self._unexpanded.append((held, phase))
        else:
            self._expand(held, phase=phase)

    def _expand(self, held: set[str], *, phase: str) -> None:
        if ranges := _named(self.arcs, held & self.dfo_terms, ("rangeIncludes",)) - self.ranges:
            self.ranges |= ranges
            self._ranges_changed = True
        successors = _next_level(self.arcs, held, self.listed) | _named(
            self.arcs, held & self.dfo_terms, _SCHEMA_ARCS
        )
        self._read_nodes(successors, phase=_next_phase(phase))

    def lists_in_full(self, dcid: str) -> bool:
        """Whether DFO Class `dcid` is a value list read in full, over the nodes read: its
        `subClassOf` chain reaches `Enumeration`, or it or an ancestor is a DFO Class the
        `rangeIncludes` of a DFO term names."""
        if _reaches_enumeration(dcid, self.arcs):
            return True
        return any(
            ancestor in self.ranges and ancestor in self.dfo_terms
            for ancestor in _ancestry(dcid, self.arcs)
        )

    def _join_dfo_closure(self, dcids: Iterable[str]) -> None:
        """Add `dcids` to the DFO value lists' descendant closure: list the subclasses of those
        whose listing has not landed, together, and read and add those of each whose listing
        has."""
        unlisted: set[str] = set()
        found: set[str] = set()
        stack = list(dcids)
        while stack:
            dcid = stack.pop()
            if dcid in self._dfo_closure:
                continue
            self._dfo_closure.add(dcid)
            if (page := self.subclass_pages.get(dcid)) is None:
                unlisted.add(dcid)
            else:
                found.update(page[0])
                stack.extend(page[0])
        self._read_nodes(found, phase=_SUBCLASSES)
        self._list_subclasses(unlisted, phase=_SUBCLASS_LISTINGS)

    def _closure(self, dcid: str) -> tuple[set[str], set[str], bool]:
        """`dcid` and every descendant the landed `<-subClassOf` listings found, those of them
        whose listing has not landed, and whether any landed listing in it was truncated."""
        members: set[str] = set()
        unlisted: set[str] = set()
        truncated = False
        stack = [dcid]
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            if (page := self.subclass_pages.get(current)) is None:
                unlisted.add(current)
                continue
            subclasses, read_all = page
            truncated = truncated or not read_all
            stack.extend(subclasses)
        return members, unlisted, truncated

    def _list_base_ranges(self) -> None:
        """Advance each base range not yet settled: settle it unjoined once a listing in its
        closure is truncated or its descendants pass `RANGE_CLOSURE_CAP`, settle it and read
        every class of its closure once every one is listed, and otherwise list each class of
        it whose subclasses are not listed, every range's together."""
        if not self._ranges_changed:
            return
        self._ranges_changed = False
        to_list: set[str] = set()
        for dcid in sorted(self.ranges - self.dfo_terms - self._ranges_settled):
            members, unlisted, truncated = self._closure(dcid)
            if truncated or len(members) - 1 > RANGE_CLOSURE_CAP:
                self._ranges_settled.add(dcid)
            elif not unlisted:
                self._ranges_settled.add(dcid)
                self._ranges_joined.add(dcid)
                self._read_nodes(members, phase=_RANGE_CLASSES)
            else:
                to_list |= unlisted
        self._list_subclasses(to_list, phase=_RANGE_LISTINGS)

    def closed_ranges(self) -> frozenset[str]:
        """The range classes whose whole descendant closure the build listed with no truncated
        listing, and whose every class the document holds: each DFO range in the value lists'
        closure, and each base range whose closure joined the document."""
        closed: set[str] = set()
        for dcid in self.ranges:
            if dcid not in self._dfo_closure and dcid not in self._ranges_joined:
                continue
            members, unlisted, truncated = self._closure(dcid)
            if not unlisted and not truncated and members <= self.arcs.keys():
                closed.add(dcid)
        return frozenset(closed)

    def _read_value_lists(self) -> None:
        """Join each DFO Class read that is now a value list read in full to the descendant
        closure, and read the instances of each whose instance listing has landed."""
        self._join_dfo_closure(
            dcid
            for dcid in self.dfo_terms - self._dfo_closure
            if _CLASS in self.listed[dcid].kinds and dcid in self.arcs and self.lists_in_full(dcid)
        )
        for dcid, (subjects, _) in sorted(self.instance_pages.items()):
            if dcid in self._instances_read or dcid not in self.arcs:
                continue
            if self.lists_in_full(dcid):
                self._instances_read.add(dcid)
                self._read_nodes((subject.dcid for subject in subjects), phase=_INSTANCES)


def resolved_entry(
    entry: Mapping[str, Sequence[ArcTarget]] | None,
) -> dict[str, tuple[ArcTarget, ...]] | None:
    """`entry` less every `UNRESOLVED_NODE` target and every property left with none, `None`
    when none is left."""
    if entry is None:
        return None
    resolved = {
        prop: kept
        for prop, targets in entry.items()
        if (kept := tuple(target for target in targets if target.value != UNRESOLVED_NODE))
    }
    return resolved or None


def _next_phase(phase: str) -> str:
    """The phase a read's successors are timed under: the seed's are level 2, a level's the
    next level's, and the instances', the subclasses' and the range classes' their own."""
    if phase in (_INSTANCES, _SUBCLASSES, _RANGE_CLASSES):
        return phase
    if phase == _SEED:
        return "level 2"
    return f"level {int(phase.removeprefix('level ')) + 1}"


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


def _ancestry(dcid: str, arcs: Mapping[str, _Arcs]) -> set[str]:
    """`dcid` and every `subClassOf` ancestor of it over the nodes read."""
    seen: set[str] = set()
    stack = [dcid]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in arcs:
            stack.extend(arcs[current]["subClassOf"])
    return seen


def _reaches_enumeration(dcid: str, arcs: Mapping[str, _Arcs]) -> bool:
    """Whether `dcid`'s `subClassOf` chain, over the nodes read, reaches `Enumeration`."""
    return _ENUMERATION in _ancestry(dcid, arcs) - {dcid}


def _instances(reads: _Reads, index_named: Collection[str]) -> dict[str, SchemaInstances]:
    """Each value list's instances, by class.

    The value lists are the DFO Classes `_Reads.lists_in_full` names, and every base Class whose
    chain reaches `Enumeration`. A DFO list covers itself, each read class whose ancestry
    reaches it, and each descendant its `<-subClassOf` listings found, and holds, for each
    class it covers, the subjects of a DFO Class's listing and the index-named nodes typed by
    any other class. A base list holds the index-named nodes typed by it or any read subclass.
    A list is `complete` when it is DFO, every class it covers is a listed DFO Class whose
    listing read every subject, and every class it covers has a subclass listing that read
    every subject. A list with no node that is not complete is left out.
    """
    arcs, listed = reads.arcs, reads.listed
    classes = [dcid for dcid in sorted(arcs) if dcid in listed and _CLASS in listed[dcid].kinds]
    ancestry = {dcid: _ancestry(dcid, arcs) for dcid in classes}
    used_by_type: dict[str, set[str]] = defaultdict(set)
    for dcid in index_named:
        for type_ in arcs.get(dcid, {}).get(_TYPE_OF, ()):
            used_by_type[type_].add(dcid)

    instances: dict[str, SchemaInstances] = {}
    for dcid in classes:
        dfo = dcid in reads.dfo_terms
        if dfo and not reads.lists_in_full(dcid):
            continue
        if not dfo and not _reaches_enumeration(dcid, arcs):
            continue
        refs: set[str] = set()
        complete = dfo
        covered_classes = {sub for sub in classes if dcid in ancestry[sub]}
        if dfo:
            covered_classes |= _descendants(dcid, reads.subclass_pages)
        for covered in sorted(covered_classes):
            refs |= used_by_type[covered]
            if dfo and not reads.subclass_pages.get(covered, ((), False))[1]:
                complete = False
            if dfo and (page := reads.instance_pages.get(covered)) is not None:
                subjects, read_all = page
                refs.update(subject.dcid for subject in subjects)
                complete = complete and read_all
            else:
                complete = False
        if refs or complete:
            instances[dcid] = SchemaInstances(refs=sorted(refs), complete=complete)
    return instances


def _descendants(dcid: str, subclass_pages: Mapping[str, tuple[tuple[str, ...], bool]]) -> set[str]:
    """`dcid` and every class the `<-subClassOf` listings found below it."""
    seen: set[str] = set()
    stack = [dcid]
    while stack:
        current = stack.pop()
        if current not in seen:
            seen.add(current)
            stack.extend(subclass_pages.get(current, ((), True))[0])
    return seen


def node_other_properties(
    dcid: str, entry: Mapping[str, Sequence[ArcTarget]], left_out: dict[tuple[str, str], int]
) -> _OtherProperties:
    """Every property of `entry` outside `NODE_PROPERTIES`, each target a node reference or a
    literal, properties and targets sorted. A literal past `OTHER_LITERAL_MAX_CHARS` and the
    values past the first `OTHER_VALUES_MAX` are left out and counted in `left_out` under
    `(dcid, property)`, and a property left with no value is left out."""
    others: _OtherProperties = {}
    for prop, targets in sorted(entry.items()):
        if prop in NODE_PROPERTIES:
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
                else SchemaNodeReference(ref=target.value)
                for target in kept[:OTHER_VALUES_MAX]
            ]
    return others


def schema_node(
    dcid: str,
    arcs: _Arcs,
    other_properties: _OtherProperties,
    provenances: Collection[str],
    instances: SchemaInstances | None,
    catalogue: Catalogue,
) -> SchemaNode:
    """The `SchemaNode` of `dcid` from its `NODE_PROPERTIES` values `arcs`, every value sorted,
    with `declared_by` from `provenances`, null when there is none."""
    return SchemaNode(
        ref=dcid,
        label=display_name(arcs["name"], dcid),
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
        declared_by=declared_by(provenances, catalogue=catalogue) if provenances else None,
        instances=instances,
    )
