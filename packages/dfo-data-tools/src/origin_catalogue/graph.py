"""A graph client over two Data Commons: our mixer and the UN System Data Commons.

`OriginGraphClient` holds our `dc_graph.GraphClient`, which serves DFO and
base Data Commons, and a UN one. It routes each read by dcid: a dcid goes to
the UN when it starts with `undata/`, or is a UN property (`undataAge`).
Everything else goes to our mixer. Batch methods split their dcids, read
both halves, and merge in first-occurrence input order. Methods that carry
no UN dcids (place containment, resolve, currency units) go to our mixer
whole.

A read of a node's schema facts (its arcs, names, label, type and
observation properties, and whether it exists) is answered from the
request's catalogue where its tables hold the node, and sends no backend
anything. The UN tables hold each admitted UN variable, the nodes its arcs
name, each UN topic's admitted leaves, and the variables of each series
(the subjects of a reverse read naming the series as `populationType`). A
variable of a series the registry excludes reads as an absent node, and
such a series has no subjects. The arc tables hold every DFO and base
variable the catalogue lists and the schema nodes their arcs depend on: a
denominator, a role property's `rangeIncludes`, and the `subClassOf` chain
of each type in that range (`origin_catalogue.arc_tables`). A node they
hold empty reads as absent. The topic table holds our Topics' leaves.

A dcid the tables do not hold is out of their scope, not absent: it is read
from its backend, as an unlisted base variable, an entity node or a UN unit
is. So is a reverse read that names no series the UN tables hold: our
graph holds variables the arc tables do not list, so they cannot say which
variables hold an arc. Table
answers pass through no backend cache, which keeps only what the graph
answered. An observation read names only the UN variables the catalogue
admits; the others read as holding no data there.

Every facet it returns has passed admission: a facet whose provenance the
request's catalogue does not admit is dropped (`admit_facets`), and DFO's and
base facets for one variable and place are kept alike, so the election
(`materialise.facets.dfo_precedence`) chooses between them. A catalogue under
a source scope (`Catalogue.scope`) drops every provenance outside it first, so
the election runs among the provenances the request allows. Facet reads to
our mixer send the catalogue's admitted hostnames as `filter.domains`, which
only shrinks the response. Admission decides what serves. With the UN
origin off, UN dcids read as absent and go to neither backend.

An observation read naming UN and other variables serves the other part
when the UN read fails, if the caller records UN unavailability
(`recording_un_blocked_places`). The failure joins that record as an
`UnReadFailure`. Without a record, and for a read naming only UN variables,
the failure raises.

Each method reads the request's catalogue once, at entry, on the caller's
thread. A method that reads both backends at once runs the UN half on a
worker thread under a copy of the caller's context, and admits the merged
result on the caller's thread with the catalogue it read at entry.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any, Final, Protocol, Self, TypedDict, Unpack

import httpx

from dc_graph import (
    UNRESOLVED_NODE,
    CapErrorFactory,
    Facet,
    FacetDefaultPolicy,
    GraphClient,
    GraphReadError,
    ReadErrorFactory,
    SubjectPage,
    SubjectsCoverageFlag,
    parse_arcs,
)
from graph_generation import (
    UNKNOWN_GENERATION,
    GenerationProvider,
    GenerationStamp,
    RegistrationHandle,
)
from graph_generation.registry import CacheRegistry
from origin_catalogue.catalogue import Catalogue
from origin_catalogue.context import (
    current_catalogue,
    note_un_blocked_places,
    record_un_read_failure,
)
from origin_catalogue.origins import DFO, UN_ID_PREFIX, UN_SYSTEM
from origin_catalogue.un_mirror import UnArc, import_added, wire_arcs

UN_BASE_URL: Final[str] = "https://unsd-datacommons.gcp.un-icc.cloud/core/api"
"""The UN System Data Commons API root. It needs no token."""

UN_BASE_URL_ENV: Final[str] = "UN_GRAPH_BASE_URL"
"""The one environment variable that overrides `UN_BASE_URL`, read by the
catalogue job and every serving surface. The job records UN rows and the
places the UN front door refuses from the deployment it read, so serving
must read the same one."""

UN_OBSERVATION_GET_URL_BUDGET_BYTES: Final[int] = 7_000
"""Query-string budget for UN observation GETs. The UN front door answers
HTTP 414 once the request line passes 8 KiB. On `/core/api/v2/observation`,
a query string of 8,151 bytes passes and one of 8,153 bytes fails."""

_UN_PROPERTY_PREFIX: Final[str] = "undata"

_RESOLVE_PATH: Final[str] = "/v2/resolve"


def is_un_dcid(dcid: str) -> bool:
    """Whether `dcid` lives on the UN deployment: a node under `undata/`, or
    a UN property such as `undataAge`, which base Data Commons does not know."""
    return dcid.startswith(UN_ID_PREFIX) or (
        dcid.startswith(_UN_PROPERTY_PREFIX) and dcid[len(_UN_PROPERTY_PREFIX) :][:1].isupper()
    )


class _Unmirrored(Enum):
    """What a table read answers for a dcid the catalogue's tables hold no fact for."""

    LIVE = "live"
    """The tables do not hold the dcid, so its backend is read."""
    OMITTED = "omitted"
    """The tables hold the dcid and it has no such fact, so the result leaves it out."""


_LIVE: Final = _Unmirrored.LIVE
_OMITTED: Final = _Unmirrored.OMITTED

_CONSTRAINT_PROPERTIES: Final[str] = "constraintProperties"
_SERIES_ARC: Final[str] = "populationType"
_NAME_PROPERTIES: Final[tuple[str, str]] = ("name", "alternateName")
_TOPIC_MEMBER_ARC: Final[str] = "relevantVariable"
_TOPIC_TYPE: Final[str] = "Topic"


def _wire(arcs: Iterable[UnArc]) -> dict:
    """Table arc rows as `GraphClient.node_arcs_batch` returns them, without the
    `import_added` targets, as the graph client leaves them out. The client keeps the
    `constraintProperties` entry when it leaves out every target, and so does this."""
    rows = tuple(arcs)
    wire = wire_arcs(arc for arc in rows if not import_added(arc))
    if any(arc.property == _CONSTRAINT_PROPERTIES for arc in rows):
        wire.setdefault(_CONSTRAINT_PROPERTIES, {"nodes": []})
    return wire


def _topic_arcs(leaves: Iterable[tuple[str, list[str]]]) -> dict:
    """A topic's arcs as the tables answer them: its type and its leaves, each with its
    types, as `relevantVariable` targets. The topic walk reaches the same leaves in one
    level."""
    return {
        "typeOf": {"nodes": [{"dcid": _TOPIC_TYPE}]},
        _TOPIC_MEMBER_ARC: {"nodes": [{"dcid": leaf, "types": types} for leaf, types in leaves]},
    }


def _types(arcs: Iterable[UnArc]) -> list[str]:
    return [arc.dcid for arc in arcs if arc.property == "typeOf" and arc.dcid is not None]


def _table_arcs(catalogue: Catalogue, dcid: str) -> dict | None | _Unmirrored:
    """A dcid's `->*` arcs from the catalogue's tables, as
    `GraphClient.node_arcs_batch` returns them.

    A UN variable's arcs are the ones the catalogue job copied, and a variable of an
    excluded series is absent, `None`. A node the arc tables hold has the arcs they hold, and
    one held empty is absent: every DFO variable, base variable and schema node, and every node
    a resolver index names, among them our index's subjects the graph dropped. A topic, UN or ours,
    has its type and its leaves as `relevantVariable` targets, and nothing else: a UN
    topic its admitted leaves, ours every leaf of the topic table. `_LIVE` for a dcid the
    tables do not hold."""
    if is_un_dcid(dcid):
        variable = catalogue.un_variable(dcid)
        if variable is not None:
            return _wire(variable.arcs)
        un_leaves = catalogue.un_topic_leaves(dcid)
        if un_leaves is not None:
            return _topic_arcs(
                (leaf, _types(member.arcs))
                for leaf in un_leaves
                if (member := catalogue.un_variable(leaf)) is not None
            )
        return None if catalogue.superseded_by(dcid) is not None else _LIVE
    held = catalogue.held_arcs(dcid)
    if held is not None:
        return _wire(held) or None
    leaves = catalogue.mixer_topic_leaves(dcid)
    if leaves is not None:
        return _topic_arcs((leaf, _types(catalogue.held_arcs(leaf) or ())) for leaf in leaves)
    return _LIVE


def arcs_mirrored(dcid: str, *, catalogue: Catalogue) -> bool:
    """Whether `catalogue`'s tables answer `dcid`'s `->*` arcs, so
    `OriginGraphClient.node_arcs_batch` reads it from neither backend."""
    return _table_arcs(catalogue, dcid) is not _LIVE


def mirrored_arcs(dcid: str, *, catalogue: Catalogue) -> dict | None:
    """`dcid`'s `->*` arcs as `catalogue`'s tables answer them, in the shape
    `GraphClient.node_arcs_batch` returns, or None for a dcid the tables hold absent or do
    not hold. Reads no backend."""
    arcs = _table_arcs(catalogue, dcid)
    return None if isinstance(arcs, _Unmirrored) else arcs


def _table_exists(catalogue: Catalogue, dcid: str) -> bool | _Unmirrored:
    arcs = _table_arcs(catalogue, dcid)
    return arcs if isinstance(arcs, _Unmirrored) else arcs is not None


def _table_arc_values(
    catalogue: Catalogue, dcid: str, properties: Sequence[str]
) -> dict[str, tuple[str, ...]] | None | _Unmirrored:
    """The named arcs of a dcid, as `GraphClient.node_arcs_many` returns them."""
    arcs = _table_arcs(catalogue, dcid)
    if arcs is None or isinstance(arcs, _Unmirrored):
        return arcs
    return parse_arcs(arcs, properties, mark_unresolved=True)


def _table_observation_properties(catalogue: Catalogue, dcid: str) -> tuple[str, ...] | _Unmirrored:
    arcs = _table_arc_values(catalogue, dcid, ("observationProperties",))
    if isinstance(arcs, _Unmirrored):
        return arcs
    if arcs is None:
        return ()
    return tuple(v for v in arcs["observationProperties"] if v != UNRESOLVED_NODE)


def _node_arcs(catalogue: Catalogue, dcid: str) -> tuple[UnArc, ...] | None:
    """The arc rows the tables hold for `dcid` as a node: a UN variable's or an arc
    table's. `None` for a dcid neither holds."""
    if is_un_dcid(dcid):
        variable = catalogue.un_variable(dcid)
        return None if variable is None else variable.arcs
    return catalogue.held_arcs(dcid)


def _absent(catalogue: Catalogue, dcid: str) -> bool:
    """Whether the tables say `dcid` has no node to read: a UN variable of an
    excluded series."""
    return is_un_dcid(dcid) and catalogue.superseded_by(dcid) is not None


def _table_names(catalogue: Catalogue, dcid: str) -> tuple[str, ...] | _Unmirrored:
    """A held node's `name` then `alternateName` values. The tables hold one name of
    a node an arc names, so its names are read live."""
    arcs = _node_arcs(catalogue, dcid)
    if arcs is None:
        return () if _absent(catalogue, dcid) else _LIVE
    names = parse_arcs(wire_arcs(arcs), _NAME_PROPERTIES)
    return tuple(dict.fromkeys(name for prop in _NAME_PROPERTIES for name in names[prop]))


def _table_label(catalogue: Catalogue, dcid: str) -> str | _Unmirrored:
    """A dcid's label as `GraphClient.node_label` picks it: a held node's last `name`
    value, the label the arc tables read for a node an arc of theirs names, or the name
    a UN arc naming the node carries."""
    arcs = _node_arcs(catalogue, dcid)
    if arcs is not None:
        names = [arc.value for arc in arcs if arc.property == "name" and arc.value]
        return names[-1] if names else _OMITTED
    if _absent(catalogue, dcid):
        return _OMITTED
    if is_un_dcid(dcid):
        node = catalogue.un_node(dcid)
        return node.name if node is not None and node.name else _LIVE
    if catalogue.arc_target(dcid) is None:
        return _LIVE
    return catalogue.arc_label(dcid) or _OMITTED


def _table_type(catalogue: Catalogue, dcid: str) -> str | _Unmirrored:
    """A dcid's first type, as `GraphClient.node_type` picks it: a held node's first
    `typeOf` target, `Topic` for a topic, or the type an arc naming the node carries.
    An arc of ours naming a node of several types carries them in no order the
    graph's `typeOf` read is known to share, so that node's type is read live."""
    arcs = _node_arcs(catalogue, dcid)
    if arcs is not None:
        types = _types(arcs)
        return types[0] if types else _OMITTED
    if is_un_dcid(dcid):
        if catalogue.un_topic_leaves(dcid) is not None:
            return _TOPIC_TYPE
        if catalogue.superseded_by(dcid) is not None:
            return _OMITTED
        node = catalogue.un_node(dcid)
        return node.types[0] if node is not None and node.types else _LIVE
    if catalogue.mixer_topic_leaves(dcid) is not None:
        return _TOPIC_TYPE
    target = catalogue.arc_target(dcid)
    return target.types[0] if target is not None and len(target.types) == 1 else _LIVE


def _table_subjects(
    catalogue: Catalogue, arcs: Sequence[tuple[str, str]], *, max_results: int
) -> SubjectPage | _Unmirrored:
    """The variables holding every arc in `arcs`, pairs of property and target dcid, from
    the catalogue's UN tables, as `GraphClient.node_subjects` pages them.

    A `populationType` arc naming a series the tables hold decides the read: every
    variable of that series is in the tables with all its arcs, so the subjects are the
    series' members holding the other arcs too, in the tables' sorted order. More than
    `max_results` of them is a truncated page, at any series size. A series the registry
    excludes has no subjects. `_LIVE` when no arc names a series the tables know."""
    members: tuple[str, ...] = ()
    for prop, target in arcs:
        if prop != _SERIES_ARC:
            continue
        members = catalogue.un_series_members(target)
        if members:
            break
        if catalogue.un_excluded_series_name(target) is not None:
            return SubjectPage(dcids=(), truncated=False)
    if not members:
        return _LIVE
    wanted = set(arcs)
    held = [
        member
        for member in members
        if (variable := catalogue.un_variable(member)) is not None
        and wanted <= {(arc.property, arc.dcid) for arc in variable.arcs}
    ]
    return SubjectPage(dcids=tuple(held[:max_results]), truncated=len(held) > max_results)


class UnGeneration:
    """The UN backend's cache generation: the build id of the catalogue bound
    to the calling request, and the UTC date. The UN deployment exposes no
    serving revision, so a UN read is served from cache for at most the rest
    of its UTC day, and a new catalogue build starts afresh. A request bound
    to one build reads and writes only that build's entries, so one still
    running when a new build loads cannot seed its refused places into the
    new build's cache. `stamp` raises `CatalogueNotBound` when no catalogue
    is bound."""

    def __init__(self, *, today: Callable[[], date] | None = None) -> None:
        self._today = today if today is not None else lambda: datetime.now(UTC).date()
        self._registry = CacheRegistry()

    def stamp(self) -> GenerationStamp:
        build_id = current_catalogue().build_id
        return GenerationStamp(token=f"{build_id}+{self._today().isoformat()}")

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


class UnAvailabilityClient(Protocol):
    """The UN deployment's SDMX `/availability` read: a `dc_sdmx.SdmxClient` on the UN API
    root, typed here so this module stays free of `dc_sdmx`'s pyarrow dependency."""

    def availability_values(
        self,
        *,
        variables: Sequence[str],
        component: str,
        filters: Mapping[str, Any] | None = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
        read_seconds: float | None = None,
    ) -> list[str]: ...

    def source_key(self) -> str: ...

    def close(self) -> None: ...


class OriginGraphClient:
    """Routes reads between our mixer (`ours`) and the UN deployment (`un`)
    and admits every facet against the request's catalogue. Each backend
    owns its caches under its own generation. Satisfies `GraphReader` and
    every method `search`'s `LiveGraphClient` delegates.

    `un_availability` reads the UN deployment's SDMX `/availability`, which places a UN
    variable for a request that names no place (`materialise.place_scope`). A client built
    without it serves no such request for a UN variable.

    Reads raise `CatalogueNotBound` when no catalogue is bound to the
    caller's context. Where both backends are read in one call, `max_bytes`
    caps each backend's read separately, and `node_arcs_many` gives each
    backend the whole `individual_probe_budget`.
    """

    def __init__(
        self,
        *,
        ours: GraphClient,
        un: GraphClient,
        un_availability: UnAvailabilityClient | None = None,
    ) -> None:
        self._ours = ours
        self._un = un
        self._un_availability = un_availability

    def close(self) -> None:
        """Close both backends and the UN availability client. Safe to call more than once."""
        try:
            self._ours.close()
        finally:
            try:
                self._un.close()
            finally:
                if self._un_availability is not None:
                    self._un_availability.close()

    def generation_stamp(self) -> GenerationStamp:
        """Our mixer's generation stamp, read without I/O. The UN backend's caches key on
        `UnGeneration` instead."""
        return self._ours.generation_stamp()

    def source_key(self) -> str:
        """Both backends' API roots, ours first."""
        return f"{self._ours.source_key()}+{self._un.source_key()}"

    def un_availability(self) -> UnAvailabilityClient | None:
        """The UN deployment's `/availability` read, or `None` when this client was built
        without one. With the UN origin off, its places hold no admitted facet."""
        return self._un_availability

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def post(self, path: str, payload: dict, *, deadline: float | None = None) -> dict:
        """POST to our mixer."""
        return self._ours.post(path, payload, deadline=deadline)

    # -- node reads --

    def node_label(self, dcid: str) -> str | None:
        return self._route_one(
            current_catalogue(), dcid, _table_label, lambda backend: backend.node_label(dcid)
        )

    def node_type(self, dcid: str) -> str | None:
        return self._route_one(
            current_catalogue(), dcid, _table_type, lambda backend: backend.node_type(dcid)
        )

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        return self._route_many(
            current_catalogue(),
            dcids,
            lambda backend, part: backend.node_labels_batch(part),
            tables=_table_label,
        )

    def node_types_batch(self, dcids: list[str]) -> dict[str, str]:
        return self._route_many(
            current_catalogue(),
            dcids,
            lambda backend, part: backend.node_types_batch(part),
            tables=_table_type,
        )

    def node_arcs(self, dcid: str) -> dict | None:
        return self._route_one(
            current_catalogue(), dcid, _table_arcs, lambda backend: backend.node_arcs(dcid)
        )

    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        return self._route_many(
            current_catalogue(),
            dcids,
            lambda backend, part: backend.node_arcs_batch(part, deadline=deadline),
            disabled=lambda: None,
            tables=_table_arcs,
        )

    def exists(self, dcid: str) -> bool:
        return bool(
            self._route_one(
                current_catalogue(), dcid, _table_exists, lambda backend: backend.exists(dcid)
            )
        )

    def exists_batch(self, dcids: list[str]) -> set[str]:
        catalogue = current_catalogue()
        ours, un = self._split(catalogue, dcids)
        held = {dcid: _table_exists(catalogue, dcid) for dcid in (*ours, *un)}
        ours = [dcid for dcid in ours if held[dcid] is _LIVE]
        un = [dcid for dcid in un if held[dcid] is _LIVE]
        found_ours, found_un = self._read_both(
            (lambda: self._ours.exists_batch(ours)) if ours else None,
            (lambda: self._un.exists_batch(un)) if un else None,
        )
        mirrored = {dcid for dcid, answer in held.items() if answer is True}
        return (found_ours or set()) | (found_un or set()) | mirrored

    def node_arc_values(
        self, dcid: str, *, properties: Sequence[str]
    ) -> dict[str, tuple[str, ...]]:
        def tables(catalogue: Catalogue, dcid: str) -> dict[str, tuple[str, ...]] | _Unmirrored:
            arcs = _table_arcs(catalogue, dcid)
            if isinstance(arcs, _Unmirrored):
                return arcs
            return parse_arcs(arcs or {}, properties)

        found = self._route_one(
            current_catalogue(),
            dcid,
            tables,
            lambda backend: backend.node_arc_values(dcid, properties=properties),
        )
        return dict.fromkeys(properties, ()) if found is None else found

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        return self._route_many(
            current_catalogue(),
            dcids,
            lambda backend, part: backend.node_arcs_many(
                part,
                properties=properties,
                individual_probe_budget=individual_probe_budget,
                deadline=deadline,
            ),
            disabled=lambda: None,
            tables=lambda catalogue, dcid: _table_arc_values(catalogue, dcid, properties),
        )

    def node_names_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, tuple[str, ...]]:
        return self._route_many(
            current_catalogue(),
            dcids,
            lambda backend, part: backend.node_names_many(part, deadline=deadline),
            disabled=tuple,
            tables=_table_names,
        )

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        """A read naming a series the catalogue's UN tables hold, as the lead under
        `populationType` or as that filter's value, is answered from the tables
        (`_table_subjects`) and sends the UN deployment nothing.

        Any other read routes on every name it carries: the lead, the property, and each filter's
        property and value. A subject holding an arc to a UN node lives only on the UN
        deployment, so one UN name anywhere sends the read there, in any filter order. The
        UN deployment also holds the base vocabulary a UN StatVar points at (`value`,
        `measuredValue`), so a UN population type filtered beside those matches there. A read
        that also names a node only our graph holds matches no subject on either graph and
        returns the UN deployment's empty page."""
        catalogue = current_catalogue()
        arcs = [(property, dcid), *(filters or {}).items()]
        held = _table_subjects(catalogue, arcs, max_results=max_results)
        if not isinstance(held, _Unmirrored):
            return held
        backend = self._backend(catalogue, *(name for arc in arcs for name in arc))
        if backend is None:
            return SubjectPage(dcids=(), truncated=False)
        return backend.node_subjects(
            dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
        )

    def node_subjects_many(
        self,
        dcids: Sequence[str],
        *,
        property: str,
        max_results: int,
        deadline: float | None = None,
    ) -> dict[str, tuple[tuple[str, ...], SubjectsCoverageFlag]]:
        """A UN `property` sends every lead to the UN. Any other property routes leads by
        dcid, and under `populationType` a lead the catalogue's UN tables know as a series is
        answered from them (`_table_subjects`)."""
        catalogue = current_catalogue()

        def read(
            backend: GraphClient, part: list[str]
        ) -> dict[str, tuple[tuple[str, ...], SubjectsCoverageFlag]]:
            return backend.node_subjects_many(
                part, property=property, max_results=max_results, deadline=deadline
            )

        def disabled() -> tuple[tuple[str, ...], SubjectsCoverageFlag]:
            return ((), "complete")

        def tables(
            catalogue: Catalogue, dcid: str
        ) -> tuple[tuple[str, ...], SubjectsCoverageFlag] | _Unmirrored:
            page = _table_subjects(catalogue, [(property, dcid)], max_results=max_results)
            if isinstance(page, _Unmirrored):
                return page
            return (page.dcids, "capped" if page.truncated else "complete")

        if is_un_dcid(property):
            unique = list(dict.fromkeys(dcids))
            if not catalogue.un_system_enabled:
                return {dcid: disabled() for dcid in unique}
            return read(self._un, unique) if unique else {}
        return self._route_many(
            catalogue,
            dcids,
            read,
            disabled=disabled,
            tables=tables if property == _SERIES_ARC else None,
        )

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        return self._route_many(
            current_catalogue(),
            dcids,
            lambda backend, part: backend.node_observation_properties_batch(part),
            disabled=tuple,
            tables=_table_observation_properties,
        )

    def enum_members(self, property_dcid: str) -> list[str]:
        """Routes on the property: UN properties enumerate on the UN deployment."""
        backend = self._backend(current_catalogue(), property_dcid)
        return [] if backend is None else backend.enum_members(property_dcid)

    def resolve_by_arc(
        self, names: Sequence[str], *, arc: str, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return self._ours.resolve_by_arc(names, arc=arc, deadline=deadline)

    def resolve_indicators(
        self, query: str, *, target: str, un: bool = False, deadline: float | None = None
    ) -> dict:
        """POST one `/v2/resolve` with the indicator resolver and `target` for `query`, to our
        mixer or, with `un`, to the UN deployment, and return the body. With the UN origin
        off, a UN resolve sends nothing and returns no entities."""
        backend = self._backend(current_catalogue(), UN_ID_PREFIX) if un else self._ours
        if backend is None:
            return {"entities": []}
        return backend.post(
            _RESOLVE_PATH,
            {"nodes": [query], "resolver": "indicator", "target": target},
            deadline=deadline,
        )

    def child_places_batch(
        self, parent_dcids: Sequence[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return self._ours.child_places_batch(parent_dcids, child_type, deadline=deadline)

    def child_dcids(self, parent_dcid: str) -> list[str]:
        return self._ours.child_dcids(parent_dcid)

    def currency_units(self, *, deadline: float | None = None) -> dict[str, str]:
        return self._ours.currency_units(deadline=deadline)

    def declares_facet_defaults(self) -> bool:
        return self._ours.declares_facet_defaults()

    def default_facet_policy(self, *, stat_var: str, axis: str) -> FacetDefaultPolicy | None:
        return self._ours.default_facet_policy(stat_var=stat_var, axis=axis)

    # -- facet reads, admitted --

    def observation_facets(
        self, *, stat_var: str, entity: str, needs_dates: bool = False
    ) -> list[Facet]:
        return (
            self.observation_facets_batch([stat_var], [entity], needs_dates)
            .get(stat_var, {})
            .get(entity, [])
        )

    def observation_facets_batch(
        self,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        """Admitted facets for every pair, through each backend's observation cache."""
        return self._admitted_facets(
            current_catalogue(),
            stat_vars,
            entities,
            lambda backend, part, domains: backend.observation_facets_batch(
                part, entities, needs_dates, max_bytes=max_bytes, domains=domains, deadline=deadline
            ),
        )

    def observation_facets_batch_many(
        self,
        *,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        deadline: float | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        return self._admitted_facets(
            current_catalogue(),
            stat_vars,
            entities,
            lambda backend, part, domains: backend.observation_facets_batch_many(
                stat_vars=part,
                entities=entities,
                deadline=deadline,
                max_bytes=max_bytes,
                domains=domains,
            ),
        )

    def variable_facets_batch(
        self, *, stat_var: str, entities: Sequence[str], deadline: float | None = None
    ) -> dict[str, list[Facet]]:
        return self.observation_facets_batch_many(
            stat_vars=[stat_var], entities=entities, deadline=deadline
        ).get(stat_var, {})

    def un_observation_reader(self) -> Self:
        """This client: its `observation_facets_batch` reads UN rows from the UN deployment."""
        return self

    def withheld_at(
        self,
        variables: Sequence[str],
        entities: Sequence[str],
        *,
        deadline: float | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Per entity, in input order, the provenances holding data for a pair of one of
        `variables` and that entity where the catalogue admits none of the pair's facets, in
        first-occurrence order over variables. An entity where every pair with data has an
        admitted facet is absent: its data serves, so nothing is withheld there.

        One read per backend with no domain filter, bounded by `deadline` and `max_bytes`
        (each backend's read separately). UN variables read the UN deployment, and read
        nothing with the UN origin off. A UN variable of a series the registry excludes is
        admitted under no provenance, so every provenance holding data for it is named. A UN
        variable the catalogue's UN tables do not hold is not read. Under a source scope
        (`Catalogue.scope`) a provenance outside it is not named, since the request left it
        out and it withholds nothing."""
        catalogue = current_catalogue()
        if not entities:
            return {}
        ours, un = self._split_variables(catalogue, variables)
        excluded = [
            variable
            for variable in self._split(catalogue, variables)[1]
            if catalogue.superseded_by(variable) is not None
        ]
        un = [*un, *excluded]

        def read(reader: GraphClient, part: list[str]) -> dict[str, dict[str, list[Facet]]]:
            return reader.observation_facets_batch(
                part, entities, max_bytes=max_bytes, deadline=deadline
            )

        from_ours, from_un = self._read_both(
            (lambda: read(self._ours, ours)) if ours else None,
            (
                lambda: self._read_un_observations(
                    catalogue, un, entities, lambda backend: read(backend, un), alongside=bool(ours)
                )
            )
            if un
            else None,
        )
        facets = {**(from_ours or {}), **(from_un or {})}
        scope = catalogue.scope
        withheld: dict[str, tuple[str, ...]] = {}
        for entity in dict.fromkeys(entities):
            held: dict[str, None] = {}
            for variable in dict.fromkeys(variables):
                provenances = [
                    provenance_id
                    for facet in facets.get(variable, {}).get(entity, [])
                    if (provenance_id := facet.provenance_id) is not None
                    and (scope is None or provenance_id in scope)
                ]
                if variable in excluded or not catalogue.admit_series(provenances).admitted:
                    held.update(dict.fromkeys(provenances))
            if held:
                withheld[entity] = tuple(held)
        return withheld

    def withheld_provenances(
        self,
        variables: Sequence[str],
        entities: Sequence[str],
        *,
        deadline: float | None = None,
        max_bytes: int | None = None,
    ) -> tuple[str, ...]:
        """`withheld_at`'s provenances over every entity, in first-occurrence order."""
        by_entity = self.withheld_at(variables, entities, deadline=deadline, max_bytes=max_bytes)
        return tuple(
            dict.fromkeys(provenance for held in by_entity.values() for provenance in held)
        )

    # -- presence --

    def entities_with_data(
        self, variable_dcids: list[str], entity_dcids: list[str], *, max_bytes: int | None = None
    ) -> list[str]:
        """Entities where any of the variables has admitted data, in input order.

        A base variable answers places in the catalogue's scan set from the
        variable table and reads the rest live. A DFO variable answers the
        places the table lists for it, when base Data Commons also holds it,
        and reads every other place live, since the table lists only base
        data. Both live reads select admitted facets, so a place whose only
        data a non-admitted provenance holds does not count. Under a source
        scope (`Catalogue.scope`) every place is read live, since the variable
        table does not say which provenance holds a variable where. A UN
        variable reads from the UN. Reads run cheapest first and stop once
        every entity has data. `max_bytes` caps each read.
        """
        catalogue = current_catalogue()
        if not variable_dcids or not entity_dcids:
            return []
        dfo, base, un = self._by_origin(catalogue, variable_dcids)
        wanted = list(dict.fromkeys(entity_dcids))
        live = [entity for entity in wanted if not _reads_scan_tables(catalogue, entity)]
        present = set(wanted).intersection(
            place for variable in (*base, *dfo) for place in _scan_places(catalogue, variable)
        )

        def missing(entities: list[str]) -> list[str]:
            return [entity for entity in entities if entity not in present]

        def admitted(variables: list[str], entities: list[str]) -> None:
            by_variable = self._facet_presence(
                catalogue, self._ours, variables, entities, catalogue.mixer_domains, max_bytes
            )
            present.update(entity for entities in by_variable.values() for entity in entities)

        if dfo and (rest := missing(wanted)):
            admitted(dfo, rest)
        if base and (rest := missing(live)):
            admitted(base, rest)
        if un and (rest := missing(wanted)):
            by_variable = self._read_un_observations(
                catalogue,
                un,
                rest,
                lambda backend: self._facet_presence(catalogue, backend, un, rest, None, max_bytes),
                alongside=bool(dfo or base),
            )
            present.update(entity for entities in by_variable.values() for entity in entities)
        return [entity for entity in entity_dcids if entity in present]

    def entities_with_data_by_variable(
        self, variable_dcids: list[str], entity_dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        """Entities with admitted data per variable, in input order, by the
        rules of `entities_with_data`. Every read takes `deadline`."""
        catalogue = current_catalogue()
        if not variable_dcids or not entity_dcids:
            return {}
        dfo, base, un = self._by_origin(catalogue, variable_dcids)
        present: dict[str, set[str]] = {variable: set() for variable in variable_dcids}
        for variable in (*base, *dfo):
            present[variable].update(_scan_places(catalogue, variable))
        wanted = list(dict.fromkeys(entity_dcids))
        live = [entity for entity in wanted if not _reads_scan_tables(catalogue, entity)]
        reads: list[Mapping[str, Sequence[str]]] = []
        if dfo:
            reads.append(
                self._facet_presence(
                    catalogue, self._ours, dfo, wanted, catalogue.mixer_domains, None, deadline
                )
            )
        if base and live:
            reads.append(
                self._facet_presence(
                    catalogue, self._ours, base, live, catalogue.mixer_domains, None, deadline
                )
            )
        if un:
            reads.append(
                self._read_un_observations(
                    catalogue,
                    un,
                    wanted,
                    lambda backend: self._facet_presence(
                        catalogue, backend, un, wanted, None, None, deadline
                    ),
                    alongside=bool(dfo or base),
                )
            )
        for read in reads:
            for variable, entities in read.items():
                present[variable].update(entities)
        return {
            variable: [entity for entity in entity_dcids if entity in present[variable]]
            for variable in variable_dcids
        }

    def un_blocked_places(self) -> frozenset[str]:
        """Places the UN front door refuses to have named in an observation GET: those the
        request's catalogue build found, and those UN reads have met since under the UN
        backend's current generation. UN reads leave them out and answer them empty, so an
        answer whose UN reads named one should say the UN could not serve it there."""
        return self._un_observations(current_catalogue()).blocked_observation_entities()

    # -- caches --

    def stats(self) -> Mapping[str, int]:
        """Our backend's counters, and the UN backend's prefixed `un_`."""
        return {**self._ours.stats(), **{f"un_{k}": v for k, v in self._un.stats().items()}}

    def clear(self) -> None:
        self._ours.clear()
        self._un.clear()

    # -- routing --

    def _backend(self, catalogue: Catalogue, *dcids: str) -> GraphClient | None:
        """The backend for a read naming `dcids`, or `None` for a UN read with the UN off."""
        if not any(is_un_dcid(dcid) for dcid in dcids):
            return self._ours
        return self._un if catalogue.un_system_enabled else None

    def _un_observations(self, catalogue: Catalogue) -> GraphClient:
        """The UN backend for an observation read, seeded with the places the catalogue build
        found the UN front door refuses, so no read pays to find them again. A place refused
        since the build is found by the backend's own bisection."""
        self._un.seed_blocked_observation_entities(catalogue.un_blocked_places)
        return self._un

    def _read_un_observations[V](
        self,
        catalogue: Catalogue,
        variables: Sequence[str],
        entities: Sequence[str],
        read: Callable[[GraphClient], Mapping[str, V]],
        *,
        alongside: bool,
    ) -> Mapping[str, V]:
        """Run one UN observation read of `variables` at `entities`, and note those the UN
        front door refuses for the caller's `recording_un_blocked_places` record.

        `alongside` says the caller's read also names our variables. Then a `GraphReadError`
        (a deadline included, a byte cap excluded) joins the caller's record as an
        `UnReadFailure` and the UN part reads as empty, so our part still serves. With no
        record bound, or with `alongside` false, it raises."""
        backend = self._un_observations(catalogue)
        try:
            result = read(backend)
        except GraphReadError as exc:
            if not record_un_read_failure(
                exc, variables=variables, places=entities, alongside_ours=alongside
            ):
                raise
            result = {}
        blocked = backend.blocked_observation_entities()
        note_un_blocked_places(entity for entity in entities if entity in blocked)
        return result

    def _split(self, catalogue: Catalogue, dcids: Sequence[str]) -> tuple[list[str], list[str]]:
        """Distinct dcids for our mixer and for the UN, in input order. UN
        dcids are left out when the UN origin is off."""
        ours: list[str] = []
        un: list[str] = []
        for dcid in dict.fromkeys(dcids):
            (un if is_un_dcid(dcid) else ours).append(dcid)
        return ours, un if catalogue.un_system_enabled else []

    def _split_variables(
        self, catalogue: Catalogue, variables: Sequence[str]
    ) -> tuple[list[str], list[str]]:
        """Distinct variables whose observations our mixer and the UN serve,
        in input order. A UN variable the catalogue does not admit is left
        out: one of an excluded series, one the UN tables do not hold, and
        every one with the UN origin off."""
        ours, un = self._split(catalogue, variables)
        return ours, [v for v in un if UN_SYSTEM in catalogue.data_commons(v)]

    def _route_one[V](
        self,
        catalogue: Catalogue,
        dcid: str,
        tables: Callable[[Catalogue, str], V | _Unmirrored],
        read: Callable[[GraphClient], V],
    ) -> V | None:
        """One dcid's fact: from the catalogue's tables where they hold it,
        else from its backend. `None` for a UN dcid with the UN off and for a
        fact the tables say the node lacks."""
        backend = self._backend(catalogue, dcid)
        if backend is None:
            return None
        answer = tables(catalogue, dcid)
        if answer is _OMITTED:
            return None
        if not isinstance(answer, _Unmirrored):
            return answer
        return read(backend)

    def _route_many[V](
        self,
        catalogue: Catalogue,
        dcids: Sequence[str],
        read: Callable[[GraphClient, list[str]], Mapping[str, V]],
        *,
        disabled: Callable[[], V] | None = None,
        tables: Callable[[Catalogue, str], V | _Unmirrored] | None = None,
    ) -> dict[str, V]:
        """Read each backend's dcids and merge in first-occurrence input
        order. A dcid `tables` answers is read from neither backend, and one
        they answer `_OMITTED` is left out. With the UN off, UN dcids map to
        `disabled()`, or are left out when `disabled` is `None`, as the
        backend leaves out a dcid it does not hold."""
        ours, un = self._split(catalogue, dcids)
        mirrored: dict[str, V] = {}
        if tables is not None:
            answers = {dcid: tables(catalogue, dcid) for dcid in (*ours, *un)}
            mirrored = {
                dcid: answer
                for dcid, answer in answers.items()
                if not isinstance(answer, _Unmirrored)
            }
            ours = [dcid for dcid in ours if answers[dcid] is _LIVE]
            un = [dcid for dcid in un if answers[dcid] is _LIVE]
        from_ours, from_un = self._read_both(
            (lambda: read(self._ours, ours)) if ours else None,
            (lambda: read(self._un, un)) if un else None,
        )
        merged = {**(from_ours or {}), **(from_un or {}), **mirrored}
        result: dict[str, V] = {}
        for dcid in dict.fromkeys(dcids):
            if dcid in merged:
                result[dcid] = merged[dcid]
            elif disabled is not None and is_un_dcid(dcid) and not catalogue.un_system_enabled:
                result[dcid] = disabled()
        return result

    def _read_both[A, B](
        self, ours: Callable[[], A] | None, un: Callable[[], B] | None
    ) -> tuple[A | None, B | None]:
        """Run each present half. When both are, the UN half runs on a worker
        thread under a copy of the caller's context while ours runs here."""
        if ours is None or un is None:
            return (None if ours is None else ours(), None if un is None else un())
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="origin-un") as pool:
            context = contextvars.copy_context()
            future: Future[B] = pool.submit(lambda: context.run(un))
            from_ours = ours()
            return from_ours, future.result()

    def _admitted_facets(
        self,
        catalogue: Catalogue,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        read: Callable[
            [GraphClient, list[str], tuple[str, ...] | None],
            Mapping[str, Mapping[str, list[Facet]]],
        ],
    ) -> dict[str, dict[str, list[Facet]]]:
        """Read facets from each backend, our mixer under the catalogue's
        domain filter and the UN under none, and admit each pair's facets.
        Every requested pair maps to a list, an empty one for a UN variable
        the catalogue does not admit, which is not read."""
        if not stat_vars or not entities:
            return {}
        ours, un = self._split_variables(catalogue, stat_vars)
        from_ours, from_un = self._read_both(
            (lambda: read(self._ours, ours, catalogue.mixer_domains)) if ours else None,
            (
                lambda: self._read_un_observations(
                    catalogue,
                    un,
                    entities,
                    lambda backend: read(backend, un, None),
                    alongside=bool(ours),
                )
            )
            if un
            else None,
        )
        merged = {**(from_ours or {}), **(from_un or {})}
        return {
            stat_var: {
                entity: admit_facets(catalogue, merged.get(stat_var, {}).get(entity, []))
                for entity in entities
            }
            for stat_var in stat_vars
        }

    def _facet_presence(
        self,
        catalogue: Catalogue,
        backend: GraphClient,
        variables: Sequence[str],
        entities: Sequence[str],
        domains: tuple[str, ...] | None,
        max_bytes: int | None,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        """Entities per variable where an admitted facet holds data, from one
        uncached facet read under `deadline`."""
        facets = backend.observation_facets_batch_many(
            stat_vars=variables,
            entities=entities,
            max_bytes=max_bytes,
            domains=domains,
            deadline=deadline,
        )
        return {
            variable: [
                entity
                for entity in entities
                if admit_facets(catalogue, facets.get(variable, {}).get(entity, []))
            ]
            for variable in variables
        }

    @staticmethod
    def _by_origin(
        catalogue: Catalogue, variables: Sequence[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """Distinct variables split into DFO, base and UN, in input order.
        A variable the DFO variable set holds is DFO. A UN variable the
        catalogue does not admit is left out, as every one is with the UN
        origin off."""
        dfo: list[str] = []
        base: list[str] = []
        un: list[str] = []
        for variable in dict.fromkeys(variables):
            origins = catalogue.data_commons(variable)
            if is_un_dcid(variable):
                if catalogue.un_system_enabled and UN_SYSTEM in origins:
                    un.append(variable)
            elif DFO in origins:
                dfo.append(variable)
            else:
                base.append(variable)
        return dfo, base, un


def _reads_scan_tables(catalogue: Catalogue, place: str) -> bool:
    """Whether presence at `place` comes from the variable table. It knows no provenance,
    so a catalogue under a source scope reads every place live."""
    return catalogue.scope is None and catalogue.is_scan_place(place)


def _scan_places(catalogue: Catalogue, variable: str) -> tuple[str, ...]:
    """The scan places the variable table holds `variable` at, none under a source scope
    (`_reads_scan_tables`)."""
    return (catalogue.base_places(variable) or ()) if catalogue.scope is None else ()


def admit_facets(catalogue: Catalogue, facets: Sequence[Facet]) -> list[Facet]:
    """The facets of one variable at one place that `catalogue` admits, in input
    order, DFO's and base's alike. A facet with no provenance is dropped."""
    admitted = set(
        catalogue.admit_series(
            facet.provenance_id for facet in facets if facet.provenance_id is not None
        ).admitted
    )
    return [facet for facet in facets if facet.provenance_id in admitted]


LABEL_TYPE_CACHE_SIZE: Final[int] = 50_000
"""The label entries, and the type entries, an origin client keeps, enough for the deployment's
referenced dcids. Search and the Datasets API size their clients from these four constants."""

ARCS_CACHE_SIZE: Final[int] = 4096
"""`node_arcs` entries, keyed by dcid."""

OBS_PROPERTIES_CACHE_SIZE: Final[int] = 5000
"""`node_observation_properties_batch` entries, kept apart from the full-arc cache."""

OBS_CACHE_SIZE: Final[int] = 512
"""Observation-facet entries, keyed by (stat_var, entity, needs_dates)."""


class GraphClientOptions(TypedDict, total=False):
    """`dc_graph.GraphClient` settings shared by both backends. An omitted
    key keeps `GraphClient`'s default."""

    timeout: float
    transport: httpx.BaseTransport
    user_agent: str
    max_attempts: int
    timer: Callable[[], float]
    node_arc_call_concurrency: int
    node_arc_total_concurrency: int
    label_cache_size: int
    type_cache_size: int
    arcs_cache_size: int
    obs_properties_cache_size: int
    observation_properties_flight_timeout_s: float
    obs_cache_size: int
    single_value_cache_enabled: bool
    read_error_cls: ReadErrorFactory
    deadline_error_cls: ReadErrorFactory
    cap_error_cls: CapErrorFactory


def build_graph_client(
    *,
    base_url: str,
    token: str | None,
    generation: GenerationProvider = UNKNOWN_GENERATION,
    un_base_url: str = UN_BASE_URL,
    un_observation_get_url_budget_bytes: int = UN_OBSERVATION_GET_URL_BUDGET_BYTES,
    un_availability: UnAvailabilityClient | None = None,
    **options: Unpack[GraphClientOptions],
) -> OriginGraphClient:
    """Build the origin client. `base_url` and `token` address our mixer,
    whose caches key on `generation`. The UN backend sends no token, reads
    observations by GET under `un_observation_get_url_budget_bytes`, and
    keys its caches on `UnGeneration`, the request's bound build and the UTC
    date. `un_availability` is the UN deployment's SDMX client, which the
    client closes with its backends."""
    ours = GraphClient(base_url=base_url, token=token, generation=generation, **options)
    try:
        un = GraphClient(
            base_url=un_base_url,
            generation=UnGeneration(),
            observation_get_url_budget_bytes=un_observation_get_url_budget_bytes,
            **options,
        )
    except BaseException:
        ours.close()
        raise
    return OriginGraphClient(ours=ours, un=un, un_availability=un_availability)


__all__ = [
    "ARCS_CACHE_SIZE",
    "LABEL_TYPE_CACHE_SIZE",
    "OBS_CACHE_SIZE",
    "OBS_PROPERTIES_CACHE_SIZE",
    "UN_BASE_URL",
    "UN_BASE_URL_ENV",
    "UN_OBSERVATION_GET_URL_BUDGET_BYTES",
    "GraphClientOptions",
    "OriginGraphClient",
    "UnAvailabilityClient",
    "UnGeneration",
    "admit_facets",
    "arcs_mirrored",
    "build_graph_client",
    "is_un_dcid",
    "mirrored_arcs",
]
