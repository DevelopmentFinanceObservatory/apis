"""Copies the UN System Data Commons facts search reads for a UN candidate.

Search derives a shape from a recalled candidate's node facts. For a UN
candidate the catalogue job copies those facts into the build, so derivation
sends the UN deployment nothing. `build_un_mirror` reads, for every registered
agency:

1. every member variable, by walking the agency's StatVarGroup tree from its
   root group (`un_root_group`): `<-specializationOf` finds each level's
   groups and `<-memberOf` each group's variables. Groups are small, where a
   series' own `<-populationType` list repeats its page token past 1,000
   members;
2. each variable's `->*` arcs, kept as `UnArc` rows. Its `populationType` arc
   names its series;
3. the UN deployment's own count of the variables under the root group and
   under each of the root's child groups (`UnGroupCounts`), which the
   deployment computes without paging. The walk must find the same child
   groups and the same count under each.

It then walks the UN topic tree from `undata/topic/Root` and keeps, for every
topic, the admitted variables `topic_walk.walk_topics` reaches from it.

A variable whose series a registry entry excludes is listed in
`UnMirror.excluded_variables` and carries no arcs, and is a leaf of no topic.
The tables keep the name of each excluded series
(`UnMirror.excluded_series_names`), read from the `populationType` arc of the
series' variables before those arcs are set aside.

The table is complete or absent. A read that stops advancing, a count that
differs, a variable naming no series, a topic or peer group the walk names
that answers no arcs, a topic tree yielding no topic, a variable under an
agency's groups that does not carry its id prefix (as one listed under two
agencies' groups does for one of them), tables breaking `UnMirror`'s invariants, an excluded series
with no member and an excluded series no variable names each return an
`UnMirrorFault`, and a build with a fault has no mirror. All but the last two
say the UN deployment's answer is incomplete (`UnMirrorFault.incomplete`),
which another run may read whole. The last two are a registry entry naming a
series the agency does not hold or the UN deployment does not name. A read the UN
deployment fails raises `dc_graph.GraphReadError`. A read it answers with HTTP
429 is repeated after a wait, three times at most.

Every `/v2/node` read follows `origin_catalogue.node_reads`' rules: a read
the UN front door refuses is read again in halves, a node a read naming
several leaves out or answers empty is read again alone, and a page
repeating a token is requested again. A read those rules cannot answer
whole is an `un_read_refused`, `un_node_omitted` or `un_enumeration_stalled`
fault. The count check decides whether the tables are whole, whatever the
pages did.

At most `UN_MIRROR_CONCURRENCY` requests are in flight against the UN
deployment.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Final, Literal, Protocol

import httpx

from artifact_publication.jobs import emit
from dc_graph import IMPORT_ADDED_CONSTRAINT_PROPERTIES, GraphReadError
from origin_catalogue.node_reads import NodeReadFault, NodeRun, chunks, read_arcs, read_subjects
from origin_catalogue.origins import UN_ID_PREFIX
from origin_catalogue.registry import Registry, UnAgency
from origin_catalogue.schema import UnAgencyCounts
from origin_catalogue.topic_walk import TOPIC_EXPAND_MAX_DEPTH, is_container_child, walk_topics

UN_MIRROR_CONCURRENCY: Final[int] = 4
"""The most requests the enumeration keeps in flight against the UN deployment."""

UN_SITE_SUFFIX: Final[str] = "/core/api"
"""The path of the UN deployment's API root under its site root. The group
counts are served by the site, beside the API."""

_GROUP_INFO_PATH: Final[str] = "/api/variable-group/info"
_ROOT_GROUP_PREFIX: Final[str] = f"{UN_ID_PREFIX}g/"
_TOPIC_ROOT: Final[str] = f"{UN_ID_PREFIX}topic/Root"
_TOPIC_TYPE: Final[str] = "Topic"
_SERIES_ARC: Final[str] = "populationType"
_TOPIC_MEMBER_ARCS: Final[tuple[str, ...]] = ("relevantVariable", "member")

_TOPIC_ROOTS_PER_WALK: Final[int] = 500
"""Topics walked together. A walk holds every root's visited nodes at once,
and the UN deployment has about 32,000 topics."""

_RATE_LIMITED: Final[int] = 429
_SITE_BACKOFF_SECONDS: Final[tuple[float, ...]] = (1.0, 4.0)
"""The waits between the three attempts a group count read makes."""

type UnMirrorFaultKind = Literal[
    "un_enumeration_stalled",
    "un_excluded_series_missing",
    "un_excluded_series_unnamed",
    "un_node_omitted",
    "un_read_refused",
    "un_tables_inconsistent",
    "un_topic_unanswered",
    "un_topics_empty",
    "un_variable_count_mismatch",
    "un_variable_outside_agency",
    "un_variable_without_series",
]

_INCOMPLETE_FAULT_KINDS: Final[frozenset[UnMirrorFaultKind]] = frozenset(
    {
        "un_enumeration_stalled",
        "un_node_omitted",
        "un_read_refused",
        "un_tables_inconsistent",
        "un_topic_unanswered",
        "un_topics_empty",
        "un_variable_count_mismatch",
        "un_variable_outside_agency",
        "un_variable_without_series",
    }
)
"""The faults the UN deployment's answer causes. No registry entry decides whether a read advances
or the front door admits it, a count agrees, a variable names its series, a topic answers its arcs,
a variable sits under one agency or carries its agency's id prefix, where an excluded series with
no member or no name is an entry naming the wrong series."""


def un_root_group(agency: UnAgency) -> str:
    """The StatVarGroup every group of `agency` specialises: `undata/g/<segment>`."""
    return f"{_ROOT_GROUP_PREFIX}{agency.segment}"


# --- Tables ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnArc:
    """One target of one arc of a node, as `/v2/node` answers `->*`:
    `property` is the arc, `dcid` the target node or `value` the literal,
    `name` the target's name and `types` its types. The answer's
    `provenanceId` is dropped. The UN tables and the arc tables
    (`origin_catalogue.arc_tables`) both hold their arcs as these rows."""

    property: str
    dcid: str | None = None
    value: str | None = None
    name: str | None = None
    types: tuple[str, ...] = ()


def wire_arcs(arcs: Iterable[UnArc]) -> dict[str, dict[str, list[dict]]]:
    """`arcs` as `dc_graph.GraphClient.node_arcs_batch` returns them: each
    property's `nodes`, in the order given, each node carrying the keys it has."""
    wire: dict[str, dict[str, list[dict]]] = {}
    for arc in arcs:
        node: dict = {}
        if arc.dcid is not None:
            node["dcid"] = arc.dcid
        if arc.value is not None:
            node["value"] = arc.value
        if arc.name is not None:
            node["name"] = arc.name
        if arc.types:
            node["types"] = list(arc.types)
        wire.setdefault(arc.property, {"nodes": []})["nodes"].append(node)
    return wire


def import_added(arc: UnArc) -> bool:
    """Whether `arc` is a `constraintProperties` target naming an
    `IMPORT_ADDED_CONSTRAINT_PROPERTIES` property, which every served read leaves out
    (`dc_graph.GraphClient.node_arcs_batch` live, `origin_catalogue.graph` from the tables)."""
    return arc.property == "constraintProperties" and arc.dcid in IMPORT_ADDED_CONSTRAINT_PROPERTIES


@dataclass(frozen=True, slots=True)
class UnVariable:
    """One admitted UN variable: its series, the `populationType` target, and
    every arc it has. Arcs are ordered by property, and targets of one
    property keep the order the UN deployment answered them in."""

    series: str
    arcs: tuple[UnArc, ...]

    def wire_arcs(self) -> dict[str, dict[str, list[dict]]]:
        """The arcs as `dc_graph.GraphClient.node_arcs_batch` returns them."""
        return wire_arcs(self.arcs)


class UnMirrorTableError(ValueError):
    """The UN tables break an invariant their readers rely on. `subject` is
    a variable, series, topic or agency segment at fault."""

    def __init__(self, message: str, *, subject: str) -> None:
        super().__init__(message)
        self.subject = subject


@dataclass(frozen=True, slots=True)
class UnMirror:
    """The UN tables of one build.

    `variables` maps every admitted variable of every registered agency to
    its `UnVariable`. `excluded_variables` maps every variable of an excluded
    series to that series. `topics` maps each UN topic to the admitted
    variables the topic walk reaches from it, in walk order. `exclusions`
    maps each excluded series to the DFO provenance its registry entry
    names, and `excluded_series_names` each excluded series to its name.
    `agencies` maps each agency's id segment to its counts.

    Construction refuses a variable both admitted and excluded, an admitted
    variable of an excluded series, an excluded variable whose series
    `exclusions` does not list, names for any series but the excluded ones, a topic with no leaf or with a leaf outside
    `variables`, and agency counts that do not sum to the tables.
    """

    variables: Mapping[str, UnVariable]
    excluded_variables: Mapping[str, str]
    topics: Mapping[str, tuple[str, ...]]
    exclusions: Mapping[str, str]
    excluded_series_names: Mapping[str, str]
    agencies: Mapping[str, UnAgencyCounts]

    def __post_init__(self) -> None:
        both = sorted(self.variables.keys() & self.excluded_variables.keys())
        if both:
            raise UnMirrorTableError(
                f"{len(both)} UN variable(s) are both admitted and excluded, e.g. {both[:3]!r}",
                subject=both[0],
            )
        admitted_excluded = sorted(
            dcid for dcid, variable in self.variables.items() if variable.series in self.exclusions
        )
        if admitted_excluded:
            raise UnMirrorTableError(
                f"{len(admitted_excluded)} admitted UN variable(s) belong to an excluded series, "
                f"e.g. {admitted_excluded[:3]!r}",
                subject=admitted_excluded[0],
            )
        unlisted = sorted(set(self.excluded_variables.values()) - self.exclusions.keys())
        if unlisted:
            raise UnMirrorTableError(
                f"excluded UN variables name series the exclusions do not list: {unlisted[:3]!r}",
                subject=unlisted[0],
            )
        unnamed = sorted(self.excluded_series_names.keys() ^ self.exclusions.keys())
        if unnamed:
            raise UnMirrorTableError(
                f"the excluded series and the names held for them differ: {unnamed[:3]!r}",
                subject=unnamed[0],
            )
        for topic, leaves in self.topics.items():
            if not leaves or any(leaf not in self.variables for leaf in leaves):
                raise UnMirrorTableError(
                    f"UN topic {topic!r} has no leaf, or a leaf that is not an admitted variable",
                    subject=topic,
                )
        counted = sum(counts.variables for counts in self.agencies.values())
        excluded = sum(counts.excluded_variables for counts in self.agencies.values())
        if counted != len(self.variables) + len(self.excluded_variables) or excluded != len(
            self.excluded_variables
        ):
            raise UnMirrorTableError(
                f"agency counts name {counted} variable(s), {excluded} excluded; the tables hold "
                f"{len(self.variables)} admitted and {len(self.excluded_variables)} excluded",
                subject=min(self.agencies, default=""),
            )

    def series_members(self) -> dict[str, tuple[str, ...]]:
        """Each admitted series' member variables, sorted."""
        members: dict[str, list[str]] = {}
        for dcid in sorted(self.variables):
            members.setdefault(self.variables[dcid].series, []).append(dcid)
        return {series: tuple(dcids) for series, dcids in members.items()}


EMPTY_UN_MIRROR: Final = UnMirror(
    variables={},
    excluded_variables={},
    topics={},
    exclusions={},
    excluded_series_names={},
    agencies={},
)
"""The UN tables of a build whose registry lists no UN agency."""


def foreign_variables(mirror: UnMirror) -> list[str]:
    """The admitted and excluded variables of `mirror` whose id starts with
    no agency's `undata/<segment>/`, sorted. Admission reads a UN variable's
    agency from that prefix, so the tables hold none."""
    prefixes = tuple(f"{UN_ID_PREFIX}{segment}/" for segment in mirror.agencies)
    return sorted(
        dcid
        for dcid in (*mirror.variables, *mirror.excluded_variables)
        if not dcid.startswith(prefixes)
    )


def registry_exclusions(registry: Registry) -> dict[str, str]:
    """Each UN series the registry excludes, mapped to the DFO provenance
    its entry names."""
    return {
        exclusion.series: exclusion.dfo_provenance
        for agency in registry.un_system
        for exclusion in agency.exclusions
    }


# --- Collaborators -----------------------------------------------------------


class UnMirrorGraph(Protocol):
    """The UN deployment's `/v2/node` POST, which `dc_graph.GraphClient` provides."""

    def post(self, path: str, payload: dict, *, deadline: float | None = None) -> dict: ...


class UnGroupCounts(Protocol):
    """The UN deployment's own count of the distinct variables under a
    StatVarGroup and under each of its child groups, keyed by group. A group
    the deployment gives no count for is absent."""

    def __call__(self, group: str) -> Mapping[str, int]: ...


class UnSiteClient:
    """`UnGroupCounts` over the UN site's `POST /api/variable-group/info`,
    which answers a group's `descendentStatVarCount` and its child groups'
    from the deployment's group index in one unpaged response. A 5xx, a 429
    or a transport error is retried, and a read that still fails raises
    `GraphReadError`."""

    def __init__(
        self,
        *,
        site_url: str,
        timeout: float,
        user_agent: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=site_url.rstrip("/"),
            timeout=timeout,
            headers={"User-Agent": user_agent},
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __call__(self, group: str) -> dict[str, int]:
        body = self._read(group)
        counts = {
            child["id"]: child["descendentStatVarCount"]
            for child in body.get("childStatVarGroups", [])
            if "descendentStatVarCount" in child
        }
        if "descendentStatVarCount" in body:
            counts[group] = body["descendentStatVarCount"]
        return counts

    def _read(self, group: str) -> dict:
        payload = {"dcid": group, "entities": [], "numEntitiesExistence": 0}
        failure = GraphReadError(f"UN group count read for {group!r} was not attempted")
        for backoff in (*_SITE_BACKOFF_SECONDS, None):
            try:
                response = self._client.post(_GROUP_INFO_PATH, json=payload)
            except httpx.TransportError as exc:
                failure = GraphReadError(f"UN group count read for {group!r} failed: {exc!r}")
            else:
                if response.status_code < 500 and response.status_code != _RATE_LIMITED:
                    return _group_info_body(group, response)
                failure = GraphReadError(
                    f"UN group count read for {group!r} returned HTTP {response.status_code}",
                    upstream_status=response.status_code,
                )
            if backoff is not None:
                time.sleep(backoff)
        raise failure


def _group_info_body(group: str, response: httpx.Response) -> dict:
    if response.status_code != 200:
        raise GraphReadError(
            f"UN group count read for {group!r} returned HTTP {response.status_code}",
            upstream_status=response.status_code,
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise GraphReadError(f"UN group count read for {group!r} returned no JSON") from exc
    if not isinstance(body, dict):
        raise GraphReadError(f"UN group count read for {group!r} returned {type(body).__name__}")
    return body


def un_site_url(un_base_url: str) -> str | None:
    """The UN site root for an API root ending in `UN_SITE_SUFFIX`, else `None`."""
    base = un_base_url.rstrip("/")
    return base.removesuffix(UN_SITE_SUFFIX) if base.endswith(UN_SITE_SUFFIX) else None


# --- Result ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnMirrorFault:
    """One reason the UN tables are not complete. `subject` is the dcid at fault."""

    kind: UnMirrorFaultKind
    subject: str
    detail: str

    @property
    def incomplete(self) -> bool:
        """Whether the UN deployment's answer is at fault and the registry is not."""
        return self.kind in _INCOMPLETE_FAULT_KINDS


@dataclass(frozen=True, slots=True)
class UnAgencyStats:
    """What one agency's enumeration read. `variables` counts every member
    variable, excluded ones included, and `counted` is the UN deployment's
    own count for the agency's root group. `page_retries` counts the page
    requests sent again because the answer repeated a token."""

    segment: str
    series: int
    variables: int
    counted: int | None
    groups: int
    requests: int
    page_retries: int
    seconds: float


@dataclass(frozen=True, slots=True)
class UnMirrorBuild:
    """`build_un_mirror`'s result: the tables, or `None` with the faults that
    kept them from being complete, and what each agency's enumeration read."""

    mirror: UnMirror | None
    faults: tuple[UnMirrorFault, ...]
    agencies: tuple[UnAgencyStats, ...]


class _Faulted(Exception):
    """An enumeration step found the tables cannot be complete."""

    def __init__(self, fault: UnMirrorFault) -> None:
        super().__init__(fault.detail)
        self.fault = fault


def _fault(exc: _Faulted | NodeReadFault) -> UnMirrorFault:
    """The fault an enumeration step or one of its reads raised."""
    if isinstance(exc, _Faulted):
        return exc.fault
    return UnMirrorFault(kind=exc.kind, subject=exc.subject, detail=exc.detail)


@dataclass
class _Run(NodeRun):
    """One enumeration's reads (`node_reads.NodeRun`), with the arcs its tables share."""

    _arcs: dict[UnArc, UnArc] = field(default_factory=dict)

    def arc(self, property: str, node: Mapping) -> UnArc:
        """The shared `UnArc` for one answered arc target."""
        dcid, value, name = node.get("dcid"), node.get("value"), node.get("name")
        arc = UnArc(
            property=self.string(property),
            dcid=self.string(dcid) if dcid else None,
            value=self.string(value) if value else None,
            name=self.string(name) if name else None,
            types=tuple(self.string(t) for t in node.get("types") or ()),
        )
        return self._arcs.setdefault(arc, arc)


# --- Reads -------------------------------------------------------------------


def _compact(run: _Run, arcs: Mapping[str, list[dict]]) -> tuple[UnArc, ...]:
    """The arcs as shared `UnArc` rows, ordered by property. A target another
    provenance repeats is kept once, as `dc_graph` keeps it."""
    compact: dict[UnArc, None] = {}
    for property in sorted(arcs):
        seen: set[str | None] = set()
        for node in arcs[property]:
            key = node.get("dcid") or node.get("value")
            if key not in seen:
                seen.add(key)
                compact[run.arc(property, node)] = None
    return tuple(compact)


# --- Agencies ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Agency:
    """One agency's member variables: each admitted one's facts, each
    excluded one's series, each excluded series' name, and its counts."""

    variables: dict[str, UnVariable]
    excluded: dict[str, str]
    excluded_names: dict[str, str]
    counts: UnAgencyCounts
    stats: UnAgencyStats


def _walk_groups(run: _Run, root: str) -> dict[str, tuple[str, ...]]:
    """Every group under `root`, itself included, mapped to its child groups."""
    children: dict[str, tuple[str, ...]] = {}
    frontier = [root]
    while frontier:
        found = read_subjects(run, frontier, "specializationOf")
        children.update(found)
        frontier = sorted({child for kids in found.values() for child in kids} - children.keys())
    return children


def _descendants(
    group: str, children: Mapping[str, tuple[str, ...]], members: Mapping[str, tuple[str, ...]]
) -> set[str]:
    """The distinct variables of `group` and every group under it."""
    variables: set[str] = set()
    seen = {group}
    stack = [group]
    while stack:
        current = stack.pop()
        variables.update(members[current])
        for child in children[current]:
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return variables


def _check_counts(
    agency: UnAgency,
    *,
    counted: Mapping[str, int],
    children: Mapping[str, tuple[str, ...]],
    members: Mapping[str, tuple[str, ...]],
) -> None:
    """Refuses a walk whose variable count under the root group, or under
    any of the root's child groups, differs from the UN deployment's own."""
    root = un_root_group(agency)
    walked = {
        group: len(_descendants(group, children, members)) for group in (root, *children[root])
    }
    differing = sorted(
        group for group in walked.keys() | counted.keys() if walked.get(group) != counted.get(group)
    )
    if differing:
        examples = ", ".join(
            f"{group} walked {walked.get(group)} counted {counted.get(group)}"
            for group in differing[:5]
        )
        raise _Faulted(
            UnMirrorFault(
                kind="un_variable_count_mismatch",
                subject=agency.provenance,
                detail=(
                    f"the walk from {root} and the UN deployment's own count differ on "
                    f"{len(differing)} group(s): {examples}"
                ),
            )
        )


def _read_variables(run: _Run, agency: UnAgency, dcids: Sequence[str]) -> dict[str, UnVariable]:
    """Each variable's series and arcs. A variable with no `populationType`
    target is a fault, since its series decides its admission."""
    variables: dict[str, UnVariable] = {}
    for read in run.map(lambda chunk: read_arcs(run, chunk), chunks(dcids)):
        for dcid, arcs in read.items():
            series = next(
                (node["dcid"] for node in arcs.get(_SERIES_ARC, []) if node.get("dcid")), None
            )
            if series is None:
                raise _Faulted(
                    UnMirrorFault(
                        kind="un_variable_without_series",
                        subject=dcid,
                        detail=f"{agency.provenance}'s variable {dcid} names no {_SERIES_ARC}",
                    )
                )
            variables[dcid] = UnVariable(series=run.string(series), arcs=_compact(run, arcs))
    return variables


def _series_names(read: Mapping[str, UnVariable], excluded: Mapping[str, str]) -> dict[str, str]:
    """Each excluded series' name: the first `name` its excluded variables'
    `populationType` arcs carry. A series none names is left out."""
    names: dict[str, str] = {}
    for dcid, series in excluded.items():
        if series in names:
            continue
        name = next(
            (
                arc.name
                for arc in read[dcid].arcs
                if arc.property == _SERIES_ARC and arc.dcid == series and arc.name
            ),
            None,
        )
        if name is not None:
            names[series] = name
    return names


def _read_agency(run: _Run, agency: UnAgency, group_counts: UnGroupCounts) -> _Agency:
    started = time.monotonic()
    requests, page_retries = run.requests(), run.page_retries()
    root = un_root_group(agency)
    children = _walk_groups(run, root)
    members = read_subjects(run, sorted(children), "memberOf")
    counted = group_counts(root)
    _check_counts(agency, counted=counted, children=children, members=members)

    dcids = sorted({dcid for group in members.values() for dcid in group})
    del members
    foreign = [dcid for dcid in dcids if not dcid.startswith(agency.id_prefix)]
    if foreign:
        raise _Faulted(
            UnMirrorFault(
                kind="un_variable_outside_agency",
                subject=foreign[0],
                detail=(
                    f"{len(foreign)} variable(s) under {root} do not start with "
                    f"{agency.id_prefix}, e.g. {foreign[:3]!r}"
                ),
            )
        )
    read = _read_variables(run, agency, dcids)
    excluded_series = {exclusion.series for exclusion in agency.exclusions}
    held = {variable.series for variable in read.values()}
    missing = sorted(excluded_series - held)
    if missing:
        raise _Faulted(
            UnMirrorFault(
                kind="un_excluded_series_missing",
                subject=missing[0],
                detail=(
                    f"{agency.provenance} excludes {missing!r}, which no variable under {root} "
                    f"names as its {_SERIES_ARC}"
                ),
            )
        )
    excluded = {
        dcid: variable.series
        for dcid, variable in read.items()
        if variable.series in excluded_series
    }
    excluded_names = _series_names(read, excluded)
    unnamed = sorted(excluded_series - excluded_names.keys())
    if unnamed:
        raise _Faulted(
            UnMirrorFault(
                kind="un_excluded_series_unnamed",
                subject=unnamed[0],
                detail=(
                    f"{agency.provenance} excludes {unnamed!r}, which no variable under {root} "
                    f"names with a name in its {_SERIES_ARC} arc"
                ),
            )
        )
    for dcid in excluded:
        del read[dcid]
    return _Agency(
        variables=read,
        excluded=excluded,
        excluded_names=excluded_names,
        counts=UnAgencyCounts(
            series=len(held),
            variables=len(dcids),
            excluded_series=len(excluded_series),
            excluded_variables=len(excluded),
        ),
        stats=UnAgencyStats(
            segment=agency.segment,
            series=len(held),
            variables=len(dcids),
            counted=counted.get(root),
            groups=len(children),
            requests=run.requests() - requests,
            page_retries=run.page_retries() - page_retries,
            seconds=round(time.monotonic() - started, 1),
        ),
    )


# --- Topics ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _TopicArcs:
    """`topic_walk.ArcsReader` over the member arcs read from the UN topic tree."""

    members: Mapping[str, Mapping[str, tuple[UnArc, ...]]]

    def node_arcs_batch(self, dcids: list[str]) -> dict[str, dict | None]:
        return {dcid: self._wire(dcid) for dcid in dcids}

    def _wire(self, dcid: str) -> dict | None:
        arcs = self.members.get(dcid)
        if arcs is None:
            return None
        return {
            property: {"nodes": [{"dcid": arc.dcid, "types": list(arc.types)} for arc in targets]}
            for property, targets in arcs.items()
        }


def _read_topics(run: _Run, admitted: Mapping[str, UnVariable]) -> dict[str, tuple[str, ...]]:
    """Every topic under `undata/topic/Root`, mapped to the admitted
    variables the topic walk reaches from it. A topic reaching none is left
    out, as the topic table of our mixer leaves it out. A node the walk
    names that answers no arcs is a fault, since every node has a type and
    its subtree would otherwise go missing unseen, and so is a tree that
    yields no topic while `admitted` holds variables."""
    members: dict[str, dict[str, tuple[UnArc, ...]]] = {}
    topics: list[str] = []
    frontier = [_TOPIC_ROOT]
    while frontier:
        level: dict[str, dict[str, list[dict]]] = {}
        for read in run.map(lambda chunk: read_arcs(run, chunk), chunks(frontier)):
            level.update(read)
        found: set[str] = set()
        for container, arcs in level.items():
            if not arcs:
                raise _Faulted(
                    UnMirrorFault(
                        kind="un_topic_unanswered",
                        subject=container,
                        detail=f"the UN deployment answered no arcs for {container}",
                    )
                )
            if any(node.get("dcid") == _TOPIC_TYPE for node in arcs.get("typeOf", [])):
                topics.append(container)
            kept = {
                property: tuple(
                    run.arc(property, node) for node in arcs.get(property, []) if node.get("dcid")
                )
                for property in _TOPIC_MEMBER_ARCS
            }
            members[container] = kept
            found.update(
                arc.dcid
                for targets in kept.values()
                for arc in targets
                if arc.dcid is not None and is_container_child(arc.dcid, list(arc.types))
            )
        frontier = sorted(found - members.keys())
    reader = _TopicArcs(members)
    table: dict[str, tuple[str, ...]] = {}
    for roots in chunks(sorted(topics), _TOPIC_ROOTS_PER_WALK):
        walked = walk_topics(
            list(roots),
            graph=reader,
            max_depth=TOPIC_EXPAND_MAX_DEPTH,
            max_members=None,
            admit=admitted.__contains__,
        )
        table.update({topic: tuple(leaves) for topic, leaves in walked.items() if leaves})
    if admitted and not table:
        raise _Faulted(
            UnMirrorFault(
                kind="un_topics_empty",
                subject=_TOPIC_ROOT,
                detail=f"the walk from {_TOPIC_ROOT} reached no topic holding an admitted variable",
            )
        )
    return table


# --- Build -------------------------------------------------------------------


def build_un_mirror(
    registry: Registry,
    *,
    graph: UnMirrorGraph,
    group_counts: UnGroupCounts,
    deadline: float | None = None,
) -> UnMirrorBuild:
    """Enumerates every registered agency and the UN topic tree, and emits
    each agency's `UnAgencyStats` as `origin_catalogue_un_mirror_enumerated`.

    Agencies are read one after another, each with at most
    `UN_MIRROR_CONCURRENCY` requests in flight. An agency with a fault is
    named in `faults` and the others are still read, so one run names every
    agency at fault. `deadline` is an absolute `time.monotonic()` value
    bounding every read. A read the UN front door refuses is read again in
    halves, and a node it refuses on its own is an `un_read_refused` fault.

    Raises `GraphReadError` when a UN read fails or the deadline passes.
    """
    started = time.monotonic()
    faults: list[UnMirrorFault] = []
    agencies: dict[str, _Agency] = {}
    with ThreadPoolExecutor(max_workers=UN_MIRROR_CONCURRENCY) as pool:
        run = _Run(graph=graph, pool=pool, deadline=deadline)
        for agency in registry.un_system:
            try:
                agencies[agency.segment] = _read_agency(run, agency, group_counts)
            except (_Faulted, NodeReadFault) as exc:
                faults.append(_fault(exc))
        stats = tuple(agency.stats for agency in agencies.values())
        mirror = None
        topic_requests, topic_page_retries = run.requests(), run.page_retries()
        if not faults:
            variables = {
                dcid: variable
                for agency in agencies.values()
                for dcid, variable in agency.variables.items()
            }
            try:
                mirror = UnMirror(
                    variables=variables,
                    excluded_variables={
                        dcid: series
                        for agency in agencies.values()
                        for dcid, series in agency.excluded.items()
                    },
                    topics=_read_topics(run, variables),
                    exclusions=registry_exclusions(registry),
                    excluded_series_names={
                        series: name
                        for agency in agencies.values()
                        for series, name in agency.excluded_names.items()
                    },
                    agencies={segment: agency.counts for segment, agency in agencies.items()},
                )
            except (_Faulted, NodeReadFault) as exc:
                faults.append(_fault(exc))
            except UnMirrorTableError as exc:
                faults.append(
                    UnMirrorFault(
                        kind="un_tables_inconsistent", subject=exc.subject, detail=str(exc)
                    )
                )
        emit(
            "origin_catalogue_un_mirror_enumerated",
            agencies=[asdict(agency) for agency in stats],
            topics=len(mirror.topics) if mirror is not None else None,
            topic_requests=run.requests() - topic_requests,
            topic_page_retries=run.page_retries() - topic_page_retries,
            requests=run.requests(),
            seconds=round(time.monotonic() - started, 1),
            faults=[asdict(fault) for fault in faults],
        )
    return UnMirrorBuild(mirror=mirror, faults=tuple(faults), agencies=stats)


__all__ = [
    "EMPTY_UN_MIRROR",
    "UN_MIRROR_CONCURRENCY",
    "UN_SITE_SUFFIX",
    "UnAgencyStats",
    "UnArc",
    "UnGroupCounts",
    "UnMirror",
    "UnMirrorBuild",
    "UnMirrorFault",
    "UnMirrorFaultKind",
    "UnMirrorGraph",
    "UnMirrorTableError",
    "UnSiteClient",
    "UnVariable",
    "build_un_mirror",
    "foreign_variables",
    "registry_exclusions",
    "un_root_group",
    "un_site_url",
    "wire_arcs",
]
