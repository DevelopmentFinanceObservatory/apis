"""The origin catalogue in memory, and the admission rules that read it.

`CatalogueTables` holds one build's tables. `CatalogueSnapshot` pairs them
with the build's manifest and the UN origin setting, and answers every
admission question from them with no graph read. Request code reads the
`Catalogue` protocol, which `CatalogueSnapshot` and test doubles implement.

Admission works on provenance ids:

- a DFO provenance is always admitted.
- a base provenance is admitted when the provenance table lists it. The job
  lists each registered source's provenances minus DFO's and the entry's
  exclusions.
- a UN agency provenance is admitted when the table lists it and the UN
  origin is enabled.

A UN candidate is admitted when the UN tables hold it: a variable of a
registered agency whose series the registry does not exclude, or a topic
reaching one. `UnTables` holds those tables with the lookups request code
reads them through, so a UN variable's facts cost no read of the UN
deployment.

A base variable held only under a provenance the registry excludes as
`superseded_by_un_system` is superseded: it is withheld, and where the UN
tables hold a variable standing for it, `Catalogue.successor` names that
variable. With the UN origin on, a leaf with a successor counts toward a
base Topic's admission as an admitted leaf does, so search can expand the
Topic and replace its leaves.

Where a DFO facet and a base facet are both present for a variable and
place, both are admitted; the facet election picks one origin for the whole
request (`materialise.facets.dfo_precedence`).

A search candidate in `withheld_variables` is refused unless DFO holds it:
its data at the scan places is known to sit under provenances the catalogue
does not admit. A base variable
the base variable table lists, or one that neither the withheld set nor a
topic names, passes, since every facet read still admits facet by facet, and
`data_commons` names base for it.

With `GRAPH_RELEASE_ORIGIN` on, `CatalogueSnapshot` reads its DFO half from the request's
graph release (`graph_release.context.served_origin`): a DFO variable's arcs come from the
release's origin view, and so do the DFO members of each family, joined to the build's base
summaries (`families.ReleaseFamilies`), which leave out a base variable the release holds. The build's base and schema arcs, and the nodes it
holds empty, still answer, and a DFO variable the release does not hold is read from the graph.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Final, Protocol
from urllib.parse import urlsplit

from artifact_publication.jobs import Ingestion
from graph_release.context import served_origin
from origin_catalogue.arc_tables import EMPTY_ARC_TABLES, ArcTables
from origin_catalogue.families import (
    EMPTY_FAMILY_TABLES,
    FamilyKey,
    FamilySummary,
    FamilyTables,
    ReleaseFamilies,
)
from origin_catalogue.origins import BASE, DFO, UN_ID_PREFIX, UN_SYSTEM, DataCommons
from origin_catalogue.schema import CatalogueManifest, ScanLevel
from origin_catalogue.topic_walk import TOPIC_EXPAND_MAX_MEMBERS
from origin_catalogue.un_mirror import EMPTY_UN_MIRROR, UnArc, UnMirror, UnVariable

STALE_AFTER: Final[timedelta] = timedelta(days=14)
"""Answers carry a staleness warning once the loaded build is older than this."""


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A provenance's source: the `dc/s/*` node its `source` arc names on
    base Data Commons, the agency provenance itself on the UN deployment."""

    ref: str
    label: str


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """The `dc/d/*` Dataset a provenance's `isPartOf` arc names, labelled
    with the Dataset's `name`, else its dcid."""

    ref: str
    label: str


@dataclass(frozen=True, slots=True)
class Licence:
    """A provenance's licence as its own arcs state it
    (`origin_catalogue.card_facts.licence_of`). `type` is the dcid of its one
    `licenseType`, `label` that node's `name`, else its dcid, and `url` its
    one `license` terms page. Each is `None` where the provenance states none
    or more than one."""

    type: str | None = None
    label: str | None = None
    url: str | None = None


@dataclass(frozen=True, slots=True)
class ProvenanceRow:
    """One admitted provenance. `segment` is the UN agency's id segment and
    is set on UN rows only. `dataset` is the provenance's parent Dataset,
    `None` on UN rows and on rows with no `isPartOf` arc. `licence` is empty
    on UN rows, which our graph does not hold."""

    id: str
    data_commons: DataCommons
    source: SourceRef
    name: str
    url: str
    segment: str | None = None
    dataset: DatasetRef | None = None
    licence: Licence = Licence()


def entry_ref(row: ProvenanceRow) -> str:
    """The source-index entry `row` belongs to: its parent Dataset, else the
    provenance itself. `CatalogueTables.dataset_facts` is keyed by it."""
    return row.dataset.ref if row.dataset is not None else row.id


@dataclass(frozen=True, slots=True)
class CardFacts:
    """The facts a source-index entry and its documentation page share, each
    `None` when the graph gives none (`origin_catalogue.card_facts`).
    `license_type` is the licence's label, the `name` of its `licenseType`
    node. `update_cadence` is the raw ISO 8601 `sourceReleaseFrequency`
    token, which readers map to a label."""

    description: str | None = None
    license_type: str | None = None
    update_cadence: str | None = None
    years_covered: str | None = None
    source_url: str | None = None


@dataclass(frozen=True, slots=True)
class SourceDocumentation:
    """The facts of a source-index entry's documentation beyond its
    `CardFacts`, each `None` when the graph gives none
    (`origin_catalogue.card_facts.source_documentation_of`).

    `provenances` are the Provenances the documentation covers: the rows
    typed `Provenance` that name the entry's Dataset in `isPartOf`, the
    orphan itself when it is typed `Provenance`, else none. `publisher` is
    the `name` of the Dataset's `isPartOf` Source, or of an orphan's `source`.
    `source_data_url` (`sourceDataUrl`), `citation` (`licenseAttribution`)
    and `license_url` (`license`) are the value every child agrees on.
    `last_update` is the latest child `lastDataRefreshDate` and
    `next_expected_update` the earliest `nextDataRefreshDate`. `documents`
    maps each of the four document properties a sole child carries to its
    distinct stored values, MCF-escaped Markdown; it is empty for an entry
    with several children."""

    provenances: tuple[str, ...] = ()
    publisher: str | None = None
    source_data_url: str | None = None
    citation: str | None = None
    license_url: str | None = None
    last_update: str | None = None
    next_expected_update: str | None = None
    documents: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BaseVariable:
    """A variable an admitted base provenance holds data for. Bit `i` of
    `places` is set when an admitted provenance has data at scan place `i`.
    Which of `provenances` has data at which place is not kept per variable;
    `CatalogueTables.provenance_places` keeps it per provenance."""

    provenances: tuple[str, ...]
    places: int


@dataclass(frozen=True, slots=True)
class SeriesAdmission:
    """The provenances admitted and withheld for one variable at one place,
    each in first-occurrence order."""

    admitted: tuple[str, ...]
    withheld: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TopicLeaves:
    """How a topic's leaves stand: `base` as admitted base variables,
    `dfo_only` as variables DFO alone holds, and `superseded` as superseded
    variables, `with_successor` of them with a UN variable standing for
    them. A superseded leaf was a base leaf before its provenance was
    superseded, so `admitted_before_supersession` reads the topic as the
    bound read it then."""

    base: int
    dfo_only: int
    superseded: int
    with_successor: int

    @property
    def admitted_before_supersession(self) -> bool:
        return 0 < self.base + self.dfo_only + self.superseded <= TOPIC_EXPAND_MAX_MEMBERS


class CatalogueTableError(ValueError):
    """The tables break an invariant the admission rules rely on."""


@dataclass(frozen=True, slots=True)
class BaseTables:
    """The tables the catalogue job enumerates from base Data Commons: the
    base variable table, each admitted base provenance's places, the
    withheld variable set, the topic table, and the scan places and levels
    the bitsets index. Fields mean what `CatalogueTables`' fields of the
    same names mean."""

    base_variables: Mapping[str, BaseVariable]
    provenance_places: Mapping[str, int]
    withheld_variables: frozenset[str]
    topics: Mapping[str, tuple[str, ...]]
    scan_places: tuple[str, ...]
    scan_levels: Mapping[str, tuple[ScanLevel, ...]]


@dataclass(frozen=True, slots=True)
class CatalogueTables:
    """One build's tables.

    `provenances` maps every admitted provenance id to its row.
    `provenance_places` maps each base provenance holding a base variable to
    a bitset over `scan_places`: bit `i` is set when one of its variables
    has a facet under it at scan place `i`. `withheld_variables` holds each variable our mixer has a facet for at a
    scan place under a provenance that is neither an admitted base
    provenance nor DFO's, and none under an admitted base provenance. `topics` maps each Topic on our mixer to
    every leaf variable search's topic walk reaches from it
    (`topic_walk.walk_topics`, uncapped). `scan_places` orders the places the
    variable bitsets index. `scan_levels` records, per source ref, the
    levels the job scanned beyond every country, Earth and the regional
    aggregates. `dataset_facts` maps the `entry_ref` of every DFO and base
    row to its entry's `CardFacts`; UN entries carry none.
    `source_documentation` maps the same entry refs to each entry's
    `SourceDocumentation`. `superseded` maps
    every withheld variable of a base provenance a UN agency supersedes to
    the UN variable that stands for it, or to `None` where none does
    (`origin_catalogue.successors`).

    `topic_leaves` counts each topic's leaves by how they stand, and
    `admit_topic` answers admission from it. A leaf is admitted when
    `dfo_variables` holds it or `withheld_variables` does not. A topic
    admitted on an admitted leaf that is not DFO's alone maps to base, and
    one admitted on DFO-only leaves to DFO. The cap is
    `TOPIC_EXPAND_MAX_MEMBERS`, the cap search applies after the same
    admission. The topic table depends on no DFO data, so a build after an
    ingestion carries it forward and admission follows the new DFO variable
    set. The member bound counts every
    superseded leaf as the base leaf it was, so supersession admits no topic
    the bound refused before it. A topic within the bound is admitted when a
    leaf stands: an admitted leaf, or, with `successors` set, a leaf with a
    successor. A topic admitted on such leaves alone maps to the UN System.

    Construction refuses a base variable naming a provenance that is not a
    base row, a bitset reaching past `scan_places`, a base variable with a
    place none of its provenances' places holds, a provenance place no base
    variable naming it holds, a variable both in
    `base_variables` and `withheld_variables`, a UN row with no segment
    or a non-UN row with one, a UN topic or a topic with no leaves,
    `dataset_facts` or `source_documentation` keys other than the DFO and
    base rows' entry refs, and
    a superseded variable that is not withheld or that DFO holds, and a
    successor that is not a UN id.
    """

    provenances: Mapping[str, ProvenanceRow]
    dfo_variables: frozenset[str]
    base_variables: Mapping[str, BaseVariable]
    provenance_places: Mapping[str, int]
    withheld_variables: frozenset[str]
    topics: Mapping[str, tuple[str, ...]]
    scan_places: tuple[str, ...]
    scan_levels: Mapping[str, tuple[ScanLevel, ...]]
    dataset_facts: Mapping[str, CardFacts]
    source_documentation: Mapping[str, SourceDocumentation]
    un_blocked_places: frozenset[str] = frozenset()
    """Places the UN front door refused to have named in an observation GET
    when the job probed it: UN reads leave them out and answer them empty."""
    superseded: Mapping[str, str | None] = field(default_factory=dict)
    topic_leaves: Mapping[str, TopicLeaves] = field(init=False)
    dfo_provenances: frozenset[str] = field(init=False)
    un_segments: frozenset[str] = field(init=False)
    mixer_domains: tuple[str, ...] = field(init=False)
    """The URL hostnames of the admitted DFO and base provenances, sorted."""
    scan_place_set: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        for provenance_id, row in self.provenances.items():
            if row.id != provenance_id:
                raise CatalogueTableError(
                    f"provenance row {row.id!r} is keyed as {provenance_id!r}"
                )
            if (row.data_commons == UN_SYSTEM) != (row.segment is not None):
                raise CatalogueTableError(
                    f"provenance {provenance_id!r} ({row.data_commons.id}) has segment "
                    f"{row.segment!r}; UN rows carry a segment and no others do"
                )
        entry_refs = {
            entry_ref(row) for row in self.provenances.values() if row.data_commons != UN_SYSTEM
        }
        for table, keyed in (
            ("dataset facts", self.dataset_facts),
            ("source documentation", self.source_documentation),
        ):
            if entry_refs != keyed.keys():
                missing = sorted(entry_refs - keyed.keys())
                extra = sorted(keyed.keys() - entry_refs)
                raise CatalogueTableError(
                    f"{table} miss {len(missing)} entry ref(s), e.g. {missing[:3]!r}, and "
                    f"hold {len(extra)} unknown one(s), e.g. {extra[:3]!r}"
                )
        base_ids = {pid for pid, row in self.provenances.items() if row.data_commons == BASE}
        limit = 1 << len(self.scan_places)
        held_by: dict[str, int] = {}
        for variable, entry in self.base_variables.items():
            unknown = [pid for pid in entry.provenances if pid not in base_ids]
            if unknown or not entry.provenances:
                raise CatalogueTableError(
                    f"base variable {variable!r} names {list(entry.provenances)!r}; each must be "
                    "a base provenance row and there must be at least one"
                )
            if not 0 <= entry.places < limit:
                raise CatalogueTableError(
                    f"base variable {variable!r} has place bits past the "
                    f"{len(self.scan_places)} scan places"
                )
            covered = 0
            for pid in entry.provenances:
                covered |= self.provenance_places.get(pid, 0)
                held_by[pid] = held_by.get(pid, 0) | entry.places
            if entry.places & ~covered:
                raise CatalogueTableError(
                    f"base variable {variable!r} has places none of its provenances "
                    f"{list(entry.provenances)!r} has"
                )
        for pid, places in self.provenance_places.items():
            if places & ~held_by.get(pid, 0) or not places:
                raise CatalogueTableError(
                    f"provenance {pid!r} has places no base variable naming it has, or none"
                )
        both = sorted(self.withheld_variables.intersection(self.base_variables))
        if both:
            raise CatalogueTableError(
                f"{len(both)} variable(s) are both admitted and withheld, e.g. {both[:3]!r}"
            )
        for topic, leaves in self.topics.items():
            if topic.startswith(UN_ID_PREFIX) or not leaves:
                raise CatalogueTableError(
                    f"topic {topic!r} has leaves {list(leaves)!r}; the topic table holds "
                    "our mixer's topics, each with at least one leaf"
                )
        for variable, successor in self.superseded.items():
            if (
                variable not in self.withheld_variables
                or variable in self.dfo_variables
                or not (successor is None or successor.startswith(UN_ID_PREFIX))
            ):
                raise CatalogueTableError(
                    f"{variable!r} is superseded with successor {successor!r}; a superseded "
                    "variable is withheld and not DFO's, and its successor is a UN variable"
                )
        object.__setattr__(self, "topic_leaves", self._count_topic_leaves())
        object.__setattr__(
            self,
            "dfo_provenances",
            frozenset(pid for pid, row in self.provenances.items() if row.data_commons == DFO),
        )
        object.__setattr__(
            self,
            "un_segments",
            frozenset(row.segment for row in self.provenances.values() if row.segment is not None),
        )
        hostnames = (
            urlsplit(row.url).hostname
            for row in self.provenances.values()
            if row.data_commons != UN_SYSTEM
        )
        object.__setattr__(
            self, "mixer_domains", tuple(sorted({host for host in hostnames if host}))
        )
        object.__setattr__(self, "scan_place_set", frozenset(self.scan_places))

    def _count_topic_leaves(self) -> dict[str, TopicLeaves]:
        counts: dict[str, TopicLeaves] = {}
        for topic, leaves in self.topics.items():
            base = dfo_only = superseded = with_successor = 0
            for leaf in leaves:
                if leaf in self.base_variables or not (
                    leaf in self.dfo_variables or leaf in self.withheld_variables
                ):
                    base += 1
                elif leaf in self.dfo_variables:
                    dfo_only += 1
                elif leaf in self.superseded:
                    superseded += 1
                    with_successor += self.superseded[leaf] is not None
            counts[topic] = TopicLeaves(
                base=base, dfo_only=dfo_only, superseded=superseded, with_successor=with_successor
            )
        return counts

    def admit_topic(self, topic: str, *, successors: bool) -> DataCommons | None:
        """The origin that admits `topic`, or `None`: a topic the bound
        admitted before supersession (`TopicLeaves.admitted_before_supersession`)
        that has at least one standing leaf. A leaf with a successor stands
        only when `successors` is set."""
        leaves = self.topic_leaves.get(topic)
        if leaves is None or not leaves.admitted_before_supersession:
            return None
        if leaves.base + leaves.dfo_only + (leaves.with_successor if successors else 0) == 0:
            return None
        return BASE if leaves.base else DFO if leaves.dfo_only else UN_SYSTEM


@dataclass(frozen=True, slots=True)
class UnTables:
    """One build's UN tables and the lookups derived from them.

    `series_members` maps each admitted series to its member variables,
    sorted. `nodes` maps each node a variable's arc names (a constraint
    property, a constraint value, a series) to the first arc row naming it,
    which carries the node's name and types.
    """

    mirror: UnMirror = EMPTY_UN_MIRROR
    series_members: Mapping[str, tuple[str, ...]] = field(init=False)
    nodes: Mapping[str, UnArc] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "series_members", self.mirror.series_members())
        nodes: dict[str, UnArc] = {}
        # Variables share arc rows, so each row is read once.
        seen: set[int] = set()
        for variable in self.mirror.variables.values():
            for arc in variable.arcs:
                if id(arc) in seen:
                    continue
                seen.add(id(arc))
                if arc.dcid is not None:
                    nodes.setdefault(arc.dcid, arc)
        object.__setattr__(self, "nodes", nodes)

    def is_candidate(self, dcid: str) -> bool:
        """Whether `dcid` is an admitted variable or a topic reaching one."""
        return dcid in self.mirror.variables or dcid in self.mirror.topics

    def superseded_by(self, variable: str) -> str | None:
        series = self.mirror.excluded_variables.get(variable)
        return None if series is None else self.mirror.exclusions[series]


EMPTY_UN_TABLES: Final = UnTables()
"""The UN tables of a build whose registry lists no UN agency."""


class Catalogue(Protocol):
    """What request code reads from one catalogue version."""

    @property
    def build_id(self) -> str: ...

    @property
    def built_at(self) -> datetime: ...

    @property
    def ingestion(self) -> Ingestion | None:
        """The `IngestionHistory` row the build's DFO tables read, which a coverage build
        records too, or `None` for a build made outside the job."""
        ...

    @property
    def un_system_enabled(self) -> bool:
        """Whether the UN origin is served. With it off, UN dcids are read nowhere."""
        ...

    @property
    def mixer_domains(self) -> tuple[str, ...] | None:
        """The hostnames sent as `filter.domains` on facet reads to our mixer,
        or `None` to send no filter."""
        ...

    @property
    def un_blocked_places(self) -> frozenset[str]:
        """Places the UN front door refuses to have named in an observation
        GET, as the catalogue job found them."""
        ...

    def is_stale(self, *, now: datetime | None = None) -> bool:
        """Whether the build is older than `STALE_AFTER`."""
        ...

    def is_scan_place(self, place: str) -> bool:
        """Whether the variable table's place bitsets cover `place`."""
        ...

    def source_documentation(self, ref: str) -> SourceDocumentation | None:
        """The documentation facts of the source-index entry `ref` (an
        `entry_ref`), or `None` for a UN entry or one the build does not list."""
        ...

    def provenance(self, provenance_id: str) -> ProvenanceRow | None:
        """The row for an admitted provenance id: origin, source, name, URL and licence."""
        ...

    def provenance_rows(self) -> tuple[ProvenanceRow, ...]:
        """Every admitted provenance's row, in table order."""
        ...

    def dataset_facts(self, ref: str) -> CardFacts | None:
        """The facts of the source-index entry `ref` (an `entry_ref`), or
        `None` for a UN entry or one the build does not list."""
        ...

    @property
    def scope(self) -> frozenset[str] | None:
        """The provenances a request allows (a source scope), or `None` for every one.
        `admit_series` drops a provenance outside it before admission.
        Every other answer of the catalogue ignores it."""
        ...

    def scoped(self, allowed: frozenset[str]) -> Catalogue:
        """This catalogue with its scope narrowed to `allowed`: the scope is `allowed`, or
        its intersection with the scope already set. Every other answer is unchanged."""
        ...

    def is_admitted(self, provenance_id: str) -> bool: ...

    def admit_series(self, provenance_ids: Iterable[str]) -> SeriesAdmission:
        """Admission for the facets of one variable at one place: every admitted provenance
        (`is_admitted`) in `scope`, whichever origin it comes from. The facet election chooses
        between DFO and base (`materialise.facets.dfo_precedence`). A provenance outside
        `scope` is left out of both `admitted` and `withheld`."""
        ...

    def admit_candidate(self, dcid: str) -> DataCommons | None:
        """The origin that admits a search candidate, or `None` when none does.
        A base candidate in `withheld_variables` and outside `dfo_variables`
        is refused: its data at the scan places sits under provenances the
        catalogue does not admit."""
        ...

    def data_commons(self, variable: str) -> tuple[DataCommons, ...]:
        """Every origin holding admitted data for `variable`, in wire order. It
        names base for every variable `admit_candidate` admits as base."""
        ...

    def base_places(self, variable: str) -> tuple[str, ...] | None:
        """The scan places where admitted base provenances hold `variable`,
        or `None` when the variable table does not list it."""
        ...

    def variable_provenances(self, variable: str) -> tuple[ProvenanceRow, ...]:
        """The admitted provenance rows known to hold `variable`, in
        first-occurrence order. Empty for DFO variables and anything the
        catalogue does not list."""
        ...

    def un_variable(self, dcid: str) -> UnVariable | None:
        """The series and arcs of a UN variable the UN tables hold, or `None`
        for any other dcid, an excluded variable included, and for every
        dcid with the UN origin off."""
        ...

    def un_series_members(self, series: str) -> tuple[str, ...]:
        """Every admitted member variable of a UN series, sorted. Empty for a
        series the UN tables do not hold and with the UN origin off."""
        ...

    def un_topic_leaves(self, topic: str) -> tuple[str, ...] | None:
        """The admitted variables the topic walk reaches from a UN topic, in
        walk order and uncapped, or `None` for a dcid the UN topic table does
        not hold and with the UN origin off."""
        ...

    def un_node(self, dcid: str) -> UnArc | None:
        """An arc row naming `dcid`, which carries the name and types of a
        node some UN variable's arc points at: a constraint property, a
        constraint value or a series. `None` for any other dcid and with the
        UN origin off."""
        ...

    def held_arcs(self, dcid: str) -> tuple[UnArc, ...] | None:
        """The `->*` arcs of a DFO variable, a base variable or a schema node
        the arc tables hold (`arc_tables.ArcTables.arcs`): empty for a node
        held and empty, which the graph does not have, and `None` for a dcid
        the tables do not hold, which is read from the graph."""
        ...

    def held_family(self, key: FamilyKey) -> FamilySummary | None:
        """The summary of every variable of family `key` the arc tables or, with the UN
        origin on, the UN tables hold (`families.FamilyTables.family`), as the catalogue
        job computed it. `None` when they hold none."""
        ...

    def arc_target(self, dcid: str) -> UnArc | None:
        """An arc row naming `dcid` that some node the arc tables hold has,
        which carries the name and types the graph gave `dcid` on that arc
        (`arc_tables.ArcTables.targets`). `None` for any other dcid."""
        ...

    def arc_label(self, dcid: str) -> str | None:
        """The label `GraphClient.node_label` gives a node an arc the arc
        tables hold names (`arc_tables.ArcTables.labels`). `None` for such a
        node with no name and for any other dcid."""
        ...

    def mixer_topic_leaves(self, topic: str) -> tuple[str, ...] | None:
        """Every leaf variable the topic walk reaches from a Topic of our
        mixer, in walk order and uncapped, admitted or not
        (`CatalogueTables.topics`). `None` for a dcid the topic table does
        not hold."""
        ...

    def superseded_by(self, variable: str) -> str | None:
        """The DFO provenance holding the data a UN variable is derived
        from, when the registry excludes the variable's series, else
        `None`. Such a variable is admitted nowhere."""
        ...

    def successor(self, dcid: str) -> str | None:
        """The UN variable that stands for a withheld variable of a base
        provenance a UN agency supersedes: an admitted variable of the UN
        tables. `None` for a variable with no successor, for any other
        dcid, and with the UN origin off."""
        ...

    def is_superseded(self, dcid: str) -> bool:
        """Whether `dcid` is a variable held only under a base provenance a
        UN agency supersedes, with a successor or without one, or a base
        Topic that the member bound admitted before supersession and that
        has no standing leaf since: a Topic refused for its size is not
        superseded. A superseded variable is withheld with the UN origin on
        or off."""
        ...

    def un_excluded_series_name(self, series: str) -> str | None:
        """The name of a UN series the registry excludes, which the UN tables
        hold though they keep no node for it. `None` for any other dcid, for
        a series the registry excludes and the build predates, and with the
        UN origin off."""
        ...


@dataclass(frozen=True, slots=True)
class CatalogueSnapshot:
    """One loaded catalogue version. `un` holds the build's UN tables, `arcs`
    its arc tables and `families` the family summaries of both. `base_families` summarises
    the `base` arc table alone (`families.summarise_table`), which `held_family` joins to a
    graph release's DFO summaries with `GRAPH_RELEASE_ORIGIN` on, and is `None` for a build
    loaded with it off.
    `un_system_enabled` gates UN admission; with it off, no UN provenance or
    candidate is admitted and the UN tables answer nothing.
    `scope` narrows `admit_series` to a request's allowed provenances (`scoped`)."""

    manifest: CatalogueManifest
    tables: CatalogueTables
    un: UnTables = EMPTY_UN_TABLES
    arcs: ArcTables = EMPTY_ARC_TABLES
    families: FamilyTables = EMPTY_FAMILY_TABLES
    un_system_enabled: bool = True
    scope: frozenset[str] | None = None
    base_families: Mapping[FamilyKey, FamilySummary] | None = None

    @property
    def build_id(self) -> str:
        return self.manifest.build_id

    @property
    def built_at(self) -> datetime:
        return self.manifest.built_at

    @property
    def ingestion(self) -> Ingestion | None:
        return self.manifest.ingestion

    @property
    def mixer_domains(self) -> tuple[str, ...]:
        return self.tables.mixer_domains

    @property
    def un_blocked_places(self) -> frozenset[str]:
        return self.tables.un_blocked_places

    def is_stale(self, *, now: datetime | None = None) -> bool:
        return (now or datetime.now(UTC)) - self.manifest.built_at > STALE_AFTER

    def is_scan_place(self, place: str) -> bool:
        return place in self.tables.scan_place_set

    def source_documentation(self, ref: str) -> SourceDocumentation | None:
        return self.tables.source_documentation.get(ref)

    def provenance(self, provenance_id: str) -> ProvenanceRow | None:
        return self.tables.provenances.get(provenance_id)

    def provenance_rows(self) -> tuple[ProvenanceRow, ...]:
        """UN rows are left out when the UN origin is off, as they are not admitted."""
        return tuple(row for row in self.tables.provenances.values() if self.is_admitted(row.id))

    def dataset_facts(self, ref: str) -> CardFacts | None:
        return self.tables.dataset_facts.get(ref)

    def is_admitted(self, provenance_id: str) -> bool:
        row = self.tables.provenances.get(provenance_id)
        if row is None:
            return False
        return row.data_commons != UN_SYSTEM or self.un_system_enabled

    def scoped(self, allowed: frozenset[str]) -> CatalogueSnapshot:
        scope = allowed if self.scope is None else self.scope & allowed
        return replace(self, scope=scope)

    def admit_series(self, provenance_ids: Iterable[str]) -> SeriesAdmission:
        present = tuple(dict.fromkeys(provenance_ids))
        if self.scope is not None:
            present = tuple(pid for pid in present if pid in self.scope)
        admitted = [pid for pid in present if self.is_admitted(pid)]
        kept = set(admitted)
        return SeriesAdmission(
            admitted=tuple(admitted), withheld=tuple(pid for pid in present if pid not in kept)
        )

    def admit_candidate(self, dcid: str) -> DataCommons | None:
        if dcid.startswith(UN_ID_PREFIX):
            return UN_SYSTEM if self._is_admitted_un_candidate(dcid) else None
        if dcid in self.tables.dfo_variables:
            return DFO
        if self._is_admitted_base(dcid):
            return BASE
        return self.tables.admit_topic(dcid, successors=self.un_system_enabled)

    def data_commons(self, variable: str) -> tuple[DataCommons, ...]:
        if variable.startswith(UN_ID_PREFIX):
            return (UN_SYSTEM,) if self._is_admitted_un_candidate(variable) else ()
        held = (
            (DFO, variable in self.tables.dfo_variables),
            (BASE, self._is_admitted_base(variable)),
        )
        return tuple(origin for origin, holds in held if holds)

    def base_places(self, variable: str) -> tuple[str, ...] | None:
        entry = self.tables.base_variables.get(variable)
        if entry is None:
            return None
        places = self.tables.scan_places
        bits = entry.places
        return tuple(places[i] for i in range(bits.bit_length()) if bits >> i & 1)

    def variable_provenances(self, variable: str) -> tuple[ProvenanceRow, ...]:
        """Base variables name their rows in the variable table; an admitted
        `undata/<segment>/...` variable maps to the UN rows of its segment."""
        entry = self.tables.base_variables.get(variable)
        if entry is not None:
            return tuple(self.tables.provenances[pid] for pid in entry.provenances)
        if not variable.startswith(UN_ID_PREFIX) or not self._is_admitted_un_candidate(variable):
            return ()
        segment = _un_segment(variable)
        return tuple(row for row in self.tables.provenances.values() if row.segment == segment)

    def un_variable(self, dcid: str) -> UnVariable | None:
        return self.un.mirror.variables.get(dcid) if self.un_system_enabled else None

    def un_series_members(self, series: str) -> tuple[str, ...]:
        return self.un.series_members.get(series, ()) if self.un_system_enabled else ()

    def un_topic_leaves(self, topic: str) -> tuple[str, ...] | None:
        return self.un.mirror.topics.get(topic) if self.un_system_enabled else None

    def un_node(self, dcid: str) -> UnArc | None:
        return self.un.nodes.get(dcid) if self.un_system_enabled else None

    def held_arcs(self, dcid: str) -> tuple[UnArc, ...] | None:
        origin = served_origin()
        if origin is None:
            return self.arcs.arcs(dcid)
        released = origin.arcs.get(dcid)
        if released is not None:
            return released
        # The build's own DFO rows give way to the release's, which do not hold this one.
        held = self.arcs.arcs(dcid)
        return None if held and dcid in self.arcs.dfo else held

    def held_family(self, key: FamilyKey) -> FamilySummary | None:
        origin = served_origin()
        if origin is None:
            return self.families.family(key, un_system_enabled=self.un_system_enabled)
        if self.base_families is None:
            raise ValueError(
                f"catalogue build {self.build_id!r} was loaded without base family summaries, "
                "which GRAPH_RELEASE_ORIGIN joins to the release's"
            )
        return self.families.family(
            key,
            un_system_enabled=self.un_system_enabled,
            release=ReleaseFamilies(
                dfo=origin.families,
                base=self.base_families,
                base_without_dfo=origin.base_families_without_dfo(
                    self.build_id, base=self.arcs.base, families=self.base_families
                ),
            ),
        )

    def arc_target(self, dcid: str) -> UnArc | None:
        return self.arcs.targets.get(dcid)

    def arc_label(self, dcid: str) -> str | None:
        return self.arcs.labels.get(dcid)

    def mixer_topic_leaves(self, topic: str) -> tuple[str, ...] | None:
        return self.tables.topics.get(topic)

    def superseded_by(self, variable: str) -> str | None:
        return self.un.superseded_by(variable)

    def successor(self, dcid: str) -> str | None:
        return self.tables.superseded.get(dcid) if self.un_system_enabled else None

    def is_superseded(self, dcid: str) -> bool:
        tables = self.tables
        if dcid in tables.superseded:
            return True
        leaves = tables.topic_leaves.get(dcid)
        return (
            leaves is not None
            and leaves.superseded > 0
            and leaves.admitted_before_supersession
            and tables.admit_topic(dcid, successors=self.un_system_enabled) is None
        )

    def un_excluded_series_name(self, series: str) -> str | None:
        if not self.un_system_enabled:
            return None
        return self.un.mirror.excluded_series_names.get(series)

    def _is_admitted_base(self, dcid: str) -> bool:
        """A variable the base table lists, or one no table places: not DFO's,
        not withheld and not a topic."""
        tables = self.tables
        return dcid in tables.base_variables or not (
            dcid in tables.dfo_variables
            or dcid in tables.withheld_variables
            or dcid in tables.topics
        )

    def _is_admitted_un_candidate(self, dcid: str) -> bool:
        """A variable or topic the UN tables hold. They hold no variable of
        an excluded series and no dcid outside the registered agencies."""
        return self.un_system_enabled and self.un.is_candidate(dcid)


def _un_segment(variable: str) -> str:
    """The id segment of a UN variable, `undata/<segment>/...`, which its
    agency's provenance row carries."""
    return variable[len(UN_ID_PREFIX) :].split("/")[0]


__all__ = [
    "EMPTY_UN_TABLES",
    "STALE_AFTER",
    "BaseTables",
    "BaseVariable",
    "CardFacts",
    "Catalogue",
    "CatalogueSnapshot",
    "CatalogueTableError",
    "CatalogueTables",
    "DatasetRef",
    "ProvenanceRow",
    "SeriesAdmission",
    "SourceRef",
    "UnTables",
    "entry_ref",
]
