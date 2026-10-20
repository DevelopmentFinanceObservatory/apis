"""Serve `GET /api/graph/node/{dcid}` and `POST /api/graph/nodes` as one envelope per node.

Both routes make at most one job on the shared DCP-read executor: the allowlisted arc read,
then the classification of every held node (`graph_api.kinds.classify`, over the code index and
the membership policy), then one batched name read for the label targets the kinds show, skipped
when none has a target. `GRAPH_NODE_TIMEOUT_SECONDS` bounds the client wait and the deadline
both reads share; every worker busy refuses the read at once. The GET route checks
`DCID_MAX_LENGTH` before reading and returns 400 for longer ids; the batch request model applies
the same ceiling and caps the list at `NODES_MAX_DCIDS`. The allowlist excludes multi-kilobyte
`geoJson` arcs.

The single read is a batch of one: a missing node answers 404 `node_not_held`, and in a batch
it is `{ref, status: "not_held"}`. A group or region the membership policy holds opens with or
without a graph node, since the membership authority vouches for it until the graph carries it,
as the members route and resolve hold it.

Labels skip names equal to the target dcid. A Provenance's label prefers its parent Dataset's
name. A group, and a region the membership policy holds, take the membership authority's
canonical name, as resolve labels them. The code index fills the code system and code blocks, the membership policy the group
block and a region's membership, `unit_words` the unit block, and the origin catalogue a
source family's `data_commons`. An organisation's or aggregate's roles come from the role
ranges the graph declares (`entity_resolution.admission.Admission`), over the role properties
the code index's variables declare, under the deadline the availability read takes; that read
failing, refused by a busy or shut DCP-read executor, or due after the deadline leaves `roles`
null with a transient gap.

A variable's block holds its `definition`, built in the same job from the code index, the
measure index and at most two more batched reads (`graph_api.card_definition`), and its
`availability` (`graph_api.availability`). After the job, every variable in the request takes
one facet profile read awaited on the event loop, then one read of the names and place types its
folds need and, once per graph generation, the national currencies (`graph_api.card_reads`),
waiting until just before the request's deadline while those reads run on and fill their
caches. A failed profile read, or a coverage index that failed to load, leaves `availability`
null with a transient gap; a surface with no profile cache or no coverage index configured
leaves it null with the permanent gap `stat_var.availability.not_served`.

`as_of` on the GET route dates a group's or region's membership. On an undated membership it
refuses 400 `request_invalid` at `/as_of`; any other kind ignores it.

The GET route answers `public, max-age=300` with an `ETag` when every gap is permanent, and
`no-store` when a gap comes from a failed read (`gapped_public_json`). The batch is a POST and
is not cached over HTTP.

The batch keeps each envelope it builds with no transient gap in `GraphDeps.node_entries`
(`graph_api.node_cache`), keyed by the generation stamp, the code index, origin catalogue,
coverage and membership builds, and the dcid, and reads only the dcids it misses there, so a
batch whose every node is held makes no graph read. The GET route reads and fills the same
entries when it carries no `as_of`; a GET with `as_of` dates a membership and always reads.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import date

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, TypeAdapter

from code_index.query import CodeIndex
from code_index.shared import get_index, get_measure_index, per_index
from code_index.types import MeasureIndex
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution
from dfo_platform.refusal import TIMEOUT_DEFAULT
from dfo_platform.state import StateNotSet, StateSlot
from dfo_platform.static_response import gapped_public_json
from entity_resolution.admission import Admission
from entity_resolution.membership import (
    AsOfNotRecorded,
    Membership,
    MembershipPolicy,
    NotRecorded,
    Projection,
)
from graph_api import routing
from graph_api.availability import NOT_SERVED, UNAVAILABLE, AvailabilityRead
from graph_api.card_definition import CardDefinition, build_definitions
from graph_api.card_reads import read_availability, read_profiles
from graph_api.config import GRAPH_NODE_TIMEOUT_SECONDS
from graph_api.deps import (
    DcpExecutorSaturated,
    DcpExecutorShutDown,
    GraphDeps,
    GraphDepsDep,
    run_dcp_read,
)
from graph_api.kinds import PLACE_KINDS, Kind, classify, code_system_of
from graph_api.node_cache import NodeKey
from graph_api.node_models import (
    Authority,
    BuiltFromTable,
    CodeBlock,
    CodeSystemBlock,
    CodeSystemDeclarations,
    Freshness,
    LinkRelation,
    MembershipBlock,
    NodeResponse,
    NodesEntry,
    NodesRequest,
    NodesResponse,
    NotHeldEntry,
    PartyBlock,
    PlaceBlock,
    PriceBasisWords,
    ProjectionCount,
    SourceBlock,
    StatVarBlock,
    UnitBlock,
    members_link,
)
from graph_api.origin_models import DataCommonsRef
from graph_api.reader import GraphNodeReader, NodeArcsByDcid
from graph_api.refs import DOCUMENTATION_PATH, NodeRef, first_name, first_value
from materialise.cadence import cadence_label
from materialise.graph import GraphDeadlineExceededError, GraphReadError
from origin_catalogue import current_catalogue
from unit_words.currency_nouns import currency_noun
from unit_words.price_basis import price_basis
from unit_words.unit_class import classify_unit

logger = logging.getLogger(__name__)

_ALLOWLIST: tuple[str, ...] = (
    "typeOf",
    "name",
    "description",
    "definition",
    "populationType",
    "measuredProperty",
    "observationProperties",
    "constraintProperties",
    "sourceIndicatorCode",
    "license",
    "licenseType",
    "source",
    "sourceReleaseFrequency",
    "automaticRefreshType",
    "url",
    "sourceDataUrl",
    "descriptionUrl",
    "dataTransformationLogic",
    "lastDataRefreshDate",
    "nextDataRefreshDate",
    "earliestObservationDate",
    "latestObservationDate",
    "isPartOf",
    "provenanceCategory",
    "containedInPlace",
    "iso4217AlphaCode",
)

_LABEL_PROPERTY = "name"
_SOURCE_FAMILY: frozenset[Kind] = frozenset({"source", "dataset", "provenance"})
# The base Data Commons names a `Currency` member by its ISO 4217 code (`INR`) and gives it no
# `iso4217AlphaCode` arc, as `dc_graph.GraphClient.currency_units` reads it.
_CURRENCY_TYPE = "Currency"

_ADMISSION_STATE = "graph_node_admission"
# A card stops waiting for its availability and role reads this long before the door's own
# timeout, so it answers with their gaps instead of timing out.
_ANSWER_MARGIN_SECONDS = 0.5
_NODE_ADAPTER: TypeAdapter[NodeResponse] = TypeAdapter(NodeResponse)

type Arcs = dict[str, tuple[str, ...]]

NODE_MEMBERSHIP_SLOT: StateSlot[MembershipPolicy] = StateSlot("graph_node_membership")
"""The membership policy the node routes classify groups and regions by. The surface's lifespan
sets it; while it is unset no node is a group or region."""


class NodeNotHeld(Exception):
    """This deployment's graph has no node for the dcid. The door maps it to 404 `node_not_held`."""

    def __init__(self, dcid: str) -> None:
        self.dcid = dcid
        super().__init__(f"graph does not hold {dcid!r}")


class DcidTooLong(Exception):
    """The dcid exceeded the path limit. Raised before a DCP read. Stores only its length
    so the 400 detail omits the oversized identifier.
    """

    def __init__(self, length: int) -> None:
        self.length = length
        super().__init__(
            f"dcid length {length} exceeds the {routing.DCID_MAX_LENGTH}-character ceiling"
        )


def _label(arcs: Arcs, dcid: str) -> str | None:
    return first_name(arcs.get(_LABEL_PROPERTY, ()), dcid)


@dataclass(frozen=True, slots=True)
class _Tables:
    """The in-process tables one read classifies and builds against, captured once."""

    code_index: CodeIndex
    measure_index: MeasureIndex
    membership: MembershipPolicy | None

    def held_membership(self, dcid: str) -> Membership | None:
        if self.membership is None or dcid not in self.membership:
            return None
        return self.membership.get(dcid)


@dataclass(frozen=True, slots=True)
class _Read:
    """The arcs of every node the read holds, its kind, the names of its label targets, and
    each variable's definition. A group or region the membership policy holds without a graph
    node has empty arcs."""

    arcs: dict[str, Arcs]
    kinds: dict[str, Kind]
    labels: NodeArcsByDcid
    definitions: dict[str, CardDefinition]

    @property
    def stat_vars(self) -> list[str]:
        return [dcid for dcid, kind in self.kinds.items() if kind == "stat_var"]


def _label_targets(kind: Kind, dcid: str, arcs: Arcs, tables: _Tables) -> tuple[str, ...]:
    """The second-hop dcids whose names a node's block shows: a source family's publisher and
    parent, a place's containing places, or a code's system node."""
    if kind in _SOURCE_FAMILY:
        return tuple(v[0] for prop in ("source", "isPartOf") if (v := arcs.get(prop, ())))
    if kind in PLACE_KINDS:
        return arcs.get("containedInPlace", ())
    if kind == "code":
        row = tables.code_index.code_by_dcid(dcid)
        declaration = tables.code_index.declarations.get(row.code_system) if row else None
        if declaration is not None and declaration.enum_dcid is not None:
            return (declaration.enum_dcid,)
    return ()


def _read_nodes_sync(
    graph: GraphNodeReader, dcids: list[str], tables: _Tables, *, deadline: float
) -> _Read:
    """Read the allowlisted arcs of `dcids`, classify every held node, read the names of their
    label targets in one batched second read, skipped when no node has a target, then build
    every variable's definition (`card_definition.build_definitions`). Runs on the DCP
    executor, so every read shares one worker and one deadline."""
    confirmed = graph.node_arcs_many(dcids, properties=_ALLOWLIST, deadline=deadline)
    arcs: dict[str, Arcs] = {}
    kinds: dict[str, Kind] = {}
    for dcid in dcids:
        node_arcs = confirmed.get(dcid)
        if node_arcs is None:
            if tables.held_membership(dcid) is None:
                continue
            node_arcs = {}
        arcs[dcid] = node_arcs
        kinds[dcid] = classify(
            dcid=dcid,
            types=node_arcs.get("typeOf", ()),
            code_index=tables.code_index,
            membership=tables.membership,
        )
    targets = dict.fromkeys(
        target
        for dcid, kind in kinds.items()
        for target in _label_targets(kind, dcid, arcs[dcid], tables)
    )
    labels = (
        graph.node_arcs_many(list(targets), properties=(_LABEL_PROPERTY,), deadline=deadline)
        if targets
        else {}
    )
    variables = {dcid: arcs[dcid] for dcid, kind in kinds.items() if kind == "stat_var"}
    definitions = (
        build_definitions(
            variables,
            graph=graph,
            code_index=tables.code_index,
            measure_index=tables.measure_index,
            deadline=deadline,
        )
        if variables
        else {}
    )
    return _Read(arcs=arcs, kinds=kinds, labels=labels, definitions=definitions)


@dataclass(slots=True)
class _Built:
    """What a block builder adds to the envelope beside its block."""

    links: dict[LinkRelation, str] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    built_from: dict[BuiltFromTable, str] = field(default_factory=dict)
    transient: bool = False


@dataclass(frozen=True, slots=True)
class _Availability:
    """Each variable's availability in one response, and the builds it read. A variable with
    no entry has none (`availability.UNAVAILABLE`)."""

    reads: dict[str, AvailabilityRead] = field(default_factory=dict)
    built_from: dict[BuiltFromTable, str] = field(default_factory=dict)


class _NodeBuilder:
    """Builds envelopes for one request from its read, its tables and its `as_of` date."""

    def __init__(
        self,
        *,
        read: _Read,
        tables: _Tables,
        admission: Admission,
        role_properties: tuple[str, ...],
        as_of: date | None,
        availability: _Availability,
        deadline: float,
    ) -> None:
        self._read = read
        self._tables = tables
        self._admission = admission
        self._role_properties = role_properties
        self._as_of = as_of
        self._availability = availability
        self._deadline = deadline

    def _ref(self, dcid: str) -> NodeRef:
        return NodeRef(ref=dcid, label=_label(self._read.labels.get(dcid) or {}, dcid))

    async def build(self, dcid: str) -> tuple[NodeResponse, bool]:
        """The envelope for a held `dcid`, and whether any of its gaps is transient."""
        arcs = self._read.arcs[dcid]
        kind = self._read.kinds[dcid]
        built = _Built(built_from={"code_index": self._tables.code_index.stamp.content_hash})
        label = _label(arcs, dcid)
        block: BaseModel | None
        match kind:
            case "stat_var":
                built.links["facets"] = members_link(dcid, "facets")
                block = self._stat_var(dcid, built)
            case "code_system":
                built.links["codes"] = members_link(dcid, "codes")
                block = self._code_system(dcid)
            case "code":
                block = self._code(dcid, built)
            case "group":
                membership = self._membership(dcid)
                built.links["members"] = members_link(dcid, "members", as_of=self._as_of)
                block = self._membership_block(membership, "group", built)
                label = membership.label
            case "country" | "continent" | "subregion" | "continental_union" | "place_other":
                block = self._place(kind, dcid, arcs, built)
                region = self._tables.held_membership(dcid)
                if region is not None and region.kind == "region":
                    label = region.label
            case "organisation" | "aggregate":
                block = await self._party(kind, arcs, built)
            case "unit":
                block = self._unit(dcid, arcs, label)
            case "source" | "dataset" | "provenance":
                block = self._source_family(kind, dcid, arcs, built)
                label = block.parent.label if block.parent is not None else None
                label = label if label is not None else _label(arcs, dcid)
            case "other":
                block = None
        if label is None:
            built.gaps.insert(0, "label")
        fields: dict[str, object] = {
            "ref": dcid,
            "label": label,
            "kind": kind,
            "types": list(arcs.get("typeOf", ())),
            "description": first_value(arcs, "description"),
            "links": built.links,
            "gaps": built.gaps,
            "built_from": built.built_from,
        }
        if block is not None:
            fields[kind] = block
        return _NODE_ADAPTER.validate_python(fields), built.transient

    def _stat_var(self, dcid: str, built: _Built) -> StatVarBlock:
        definition = self._read.definitions[dcid]
        availability = self._availability.reads.get(dcid, UNAVAILABLE)
        built.gaps.extend((*definition.gaps, *availability.gaps))
        built.transient = built.transient or definition.transient or availability.transient
        built.built_from.update(self._availability.built_from)
        return StatVarBlock(
            definition=definition.definition, availability=availability.availability
        )

    def _code_system(self, dcid: str) -> CodeSystemBlock:
        index = self._tables.code_index
        declaration = code_system_of(index, dcid)
        assert declaration is not None, "classify names code_system only for a declared system"
        return CodeSystemBlock(
            property=declaration.code_system,
            plain_name=declaration.plain_name,
            declarations=CodeSystemDeclarations(
                hierarchical=declaration.hierarchical,
                disjoint_from_children=declaration.disjoint_from_children,
                summable=declaration.summable,
                absent_reading=declaration.absent_reading,
            ),
            code_count=len(index.codes_by_system.get(declaration.code_system, ())),
        )

    def _code(self, dcid: str, built: _Built) -> CodeBlock:
        index = self._tables.code_index
        row = index.code_by_dcid(dcid)
        assert row is not None, "classify names code only for a code row"
        declaration = index.declarations.get(row.code_system)
        enum_dcid = declaration.enum_dcid if declaration is not None else None
        if enum_dcid is None:
            built.gaps.append("code.code_system")
        if index.has_children(dcid):
            built.links["children"] = members_link(dcid, "children")
        parent = index.code_by_dcid(row.parent) if row.parent is not None else None
        return CodeBlock(
            code_system=self._ref(enum_dcid) if enum_dcid is not None else None,
            property=row.code_system,
            code=row.code,
            parent=(
                NodeRef(ref=row.parent, label=parent.label if parent is not None else None)
                if row.parent is not None
                else None
            ),
            synonyms=list(row.synonyms),
        )

    def _membership(self, dcid: str) -> Membership:
        membership = self._tables.held_membership(dcid)
        assert membership is not None, "classify names group or region only for a held ref"
        return membership

    def _membership_block(
        self, membership: Membership, path: str, built: _Built
    ) -> MembershipBlock:
        """The membership at the read's `as_of`. Raises `AsOfNotRecorded` for `as_of` on an
        undated membership."""
        effective_date = membership.effective_date(self._as_of)
        if effective_date is None:
            built.gaps.append(f"{path}.effective_date")
        projections: list[ProjectionCount] = []
        for projection in Projection:
            found = membership.member_count(projection, self._as_of)
            count = None if isinstance(found, NotRecorded) else found
            if count is None:
                built.gaps.append(f"{path}.projections.{projection.value}")
            projections.append(
                ProjectionCount(
                    projection=projection,
                    count=count,
                    complete=count is not None and membership.gap(projection) is None,
                )
            )
        built.built_from["membership"] = membership.authority.version
        return MembershipBlock(
            authority=Authority(
                name=membership.authority.name, version=membership.authority.version
            ),
            as_of=self._as_of,
            effective_date=effective_date,
            projections=projections,
        )

    def _place(self, kind: Kind, dcid: str, arcs: Arcs, built: _Built) -> PlaceBlock:
        region = self._tables.held_membership(dcid)
        membership = None
        if region is not None and region.kind == "region":
            built.links["countries"] = members_link(dcid, "countries", as_of=self._as_of)
            membership = self._membership_block(region, f"{kind}.membership", built)
        return PlaceBlock(
            contained_in=[self._ref(target) for target in arcs.get("containedInPlace", ())],
            membership=membership,
        )

    async def _party(self, kind: Kind, arcs: Arcs, built: _Built) -> PartyBlock:
        """The role properties any of the node's types can fill. A range read that fails, is
        refused by the DCP-read executor, or is due after the deadline leaves `roles` null
        with a transient gap."""
        types = arcs.get("typeOf", ())
        roles: list[str] = []
        try:
            for role in self._role_properties:
                for graph_type in types:
                    if await self._admission.admits(role, graph_type, deadline=self._deadline):
                        roles.append(role)
                        break
        except (GraphReadError, DcpExecutorSaturated, DcpExecutorShutDown):
            logger.warning("Graph node door's role-range read failed", exc_info=True)
            built.gaps.append(f"{kind}.roles")
            built.transient = True
            return PartyBlock(roles=None)
        return PartyBlock(roles=roles)

    def _unit(self, dcid: str, arcs: Arcs, label: str | None) -> UnitBlock:
        iso = first_value(arcs, "iso4217AlphaCode")
        if iso is None and _CURRENCY_TYPE in arcs.get("typeOf", ()) and _is_iso_code(dcid):
            iso = dcid
        unit_class = classify_unit(
            unit=dcid, denominator=None, currency_units={dcid: iso} if iso else {}
        )
        basis = price_basis(requested=None, unit_class=unit_class, measurement_qualifier=None)
        return UnitBlock(
            unit_class=unit_class.kind,
            currency=unit_class.currency,
            currency_noun=(
                currency_noun(unit_class.currency, label=label)
                if unit_class.currency is not None
                else None
            ),
            price_basis=(
                PriceBasisWords(prices=basis.prices, base_year=basis.base_year)
                if basis is not None
                else None
            ),
            changes_magnitude=unit_class.changes_magnitude,
        )

    def _source_family(self, kind: Kind, dcid: str, arcs: Arcs, built: _Built) -> SourceBlock:
        """`publisher` (the `source` arc's target) and `parent` (the `isPartOf` arc's target)
        take their labels from the second-hop name read."""
        publisher_dcid = first_value(arcs, "source")
        parent_dcid = first_value(arcs, "isPartOf")
        catalogue = current_catalogue()
        built.built_from["origin_catalogue"] = catalogue.build_id
        row = catalogue.provenance(dcid) if catalogue.is_admitted(dcid) else None
        release_frequency = first_value(arcs, "sourceReleaseFrequency")
        if kind == "source":
            built.links["datasets"] = members_link(dcid, "datasets")
        else:
            built.links["variables"] = members_link(dcid, "variables")
        return SourceBlock(
            publisher=self._ref(publisher_dcid) if publisher_dcid is not None else None,
            parent=self._ref(parent_dcid) if parent_dcid is not None else None,
            license=first_value(arcs, "license"),
            license_type=first_value(arcs, "licenseType"),
            url=first_value(arcs, "url"),
            source_data_url=first_value(arcs, "sourceDataUrl"),
            description_url=first_value(arcs, "descriptionUrl"),
            transformation_logic_url=first_value(arcs, "dataTransformationLogic"),
            last_data_refresh_date=first_value(arcs, "lastDataRefreshDate"),
            next_data_refresh_date=first_value(arcs, "nextDataRefreshDate"),
            earliest_observation_date=first_value(arcs, "earliestObservationDate"),
            latest_observation_date=first_value(arcs, "latestObservationDate"),
            freshness=Freshness(
                release_frequency=release_frequency,
                release_frequency_label=cadence_label(release_frequency),
                refresh_type=first_value(arcs, "automaticRefreshType"),
            ),
            data_commons=DataCommonsRef.of(row.data_commons) if row is not None else None,
            documentation=DOCUMENTATION_PATH.format(ref=dcid),
        )


def _is_iso_code(dcid: str) -> bool:
    return len(dcid) == 3 and dcid.isascii() and dcid.isalpha() and dcid.isupper()


@per_index
def _role_properties(code_index: CodeIndex) -> tuple[str, ...]:
    """The role properties the code index's variables declare, sorted."""
    return tuple(
        sorted(
            {
                role
                for row in (*code_index.statvars, *code_index.listed_statvars)
                for role in row.observation_properties
            }
        )
    )


def published_membership(request: Request) -> MembershipPolicy | None:
    """The membership policy the surface's lifespan published, or `None` where none is."""
    try:
        return NODE_MEMBERSHIP_SLOT(request)
    except StateNotSet:
        return None


def _admission(request: Request, deps: GraphDeps) -> Admission:
    """One `Admission` per surface lifespan, kept on the app beside the deps it reads through,
    so its generation-keyed range cache outlives a request."""
    held: tuple[GraphDeps, Admission] | None = getattr(request.app.state, _ADMISSION_STATE, None)
    if held is None or held[0] is not deps:
        held = (
            deps,
            Admission(deps.graph, graph_executor=deps.dcp_executor, generation=deps.generation),
        )
        setattr(request.app.state, _ADMISSION_STATE, held)
    return held[1]


def _tables(request: Request) -> _Tables:
    return _Tables(
        code_index=get_index(),
        measure_index=get_measure_index(),
        membership=published_membership(request),
    )


async def _read_held(deps: GraphDeps, dcids: list[str], tables: _Tables) -> tuple[_Read, float]:
    """Run `_read_nodes_sync` as one job on the shared DCP-read executor, and return the read
    and its deadline.

    Compute the deadline before submission so queue time counts against it."""
    deadline = time.monotonic() + GRAPH_NODE_TIMEOUT_SECONDS
    read = await run_dcp_read(
        deps.dcp_executor, _read_nodes_sync, deps.graph, dcids, tables, deadline=deadline
    )
    return read, deadline


async def holds_node(request: Request, deps: GraphDeps, dcid: str) -> bool:
    """Whether the deployment holds `dcid`: a node the graph holds, or a group or region the
    membership policy holds with or without a graph node."""
    membership = published_membership(request)
    if membership is not None and dcid in membership:
        return True
    read, _ = await _read_held(deps, [dcid], _tables(request))
    return dcid in read.kinds


async def _read(
    request: Request, deps: GraphDeps, dcids: list[str], tables: _Tables, *, as_of: date | None
) -> tuple[_Read, _NodeBuilder]:
    """The batch's hold read (`_read_held`) and a builder over it."""
    read, deadline = await _read_held(deps, dcids, tables)
    code_index = tables.code_index
    answer_deadline = deadline - _ANSWER_MARGIN_SECONDS
    builder = _NodeBuilder(
        read=read,
        tables=tables,
        admission=_admission(request, deps),
        role_properties=_role_properties(code_index),
        as_of=as_of,
        availability=await _availability(deps, read.stat_vars, tables, deadline=answer_deadline),
        deadline=answer_deadline,
    )
    return read, builder


async def _availability(
    deps: GraphDeps, stat_vars: list[str], tables: _Tables, *, deadline: float
) -> _Availability:
    """Every variable's availability in one profile read and one read of its inputs."""
    if not stat_vars:
        return _Availability()
    profiles = await read_profiles(deps, stat_vars, deadline=deadline)
    if profiles is None:
        # A coverage index that failed to load can load after a restart; none configured cannot.
        if deps.coverage_load_failed:
            return _Availability()
        return _Availability(reads=dict.fromkeys(stat_vars, NOT_SERVED))
    reads = await read_availability(
        profiles, code_index=tables.code_index, membership=tables.membership, deadline=deadline
    )
    return _Availability(
        reads=reads,
        built_from={
            "origin_catalogue": profiles.catalogue.build_id,
            "coverage_index": profiles.coverage_build,
        },
    )


def _lookup_keys(deps: GraphDeps, tables: _Tables, dcids: list[str]) -> dict[str, NodeKey]:
    """The key each dcid's envelope is looked up under, from the builds the request sees
    before its read."""
    stamp = deps.generation.stamp().key
    code_index = tables.code_index.stamp.content_hash
    catalogue = current_catalogue().build_id
    coverage = deps.coverage.snapshot().manifest.build_id if deps.coverage is not None else None
    keys: dict[str, NodeKey] = {}
    for dcid in dcids:
        membership = tables.held_membership(dcid)
        keys[dcid] = NodeKey(
            stamp=stamp,
            code_index=code_index,
            origin_catalogue=catalogue,
            coverage_index=coverage,
            membership=membership.authority.version if membership is not None else None,
            dcid=dcid,
        )
    return keys


async def _build_response(
    request: Request, deps: GraphDeps, dcid: str, as_of: date | None
) -> tuple[NodeResponse, bool]:
    """Build `dcid`'s envelope and whether it has a transient gap. Without `as_of` the read
    goes through `deps.node_entries` as the batch's does; a dated read always reads."""
    tables = _tables(request)
    key = _lookup_keys(deps, tables, [dcid])[dcid] if as_of is None else None
    if key is not None and (held := deps.node_entries.get(key)) is not None:
        return held, False
    read, builder = await _read(request, deps, [dcid], tables, as_of=as_of)
    if dcid not in read.kinds:
        raise NodeNotHeld(dcid)
    node, transient = await builder.build(dcid)
    if key is not None and not transient:
        deps.node_entries.put(key, node)
    return node, transient


async def _build_batch_response(
    request: Request, deps: GraphDeps, dcids: list[str]
) -> NodesResponse:
    """Serve each envelope `deps.node_entries` holds, read only the rest, and store each one
    built with no transient gap. The batch takes no `as_of`, so every envelope it builds is
    one the cache can hold."""
    unique = list(dict.fromkeys(dcids))
    tables = _tables(request)
    keys = _lookup_keys(deps, tables, unique)
    entries: dict[str, NodesEntry] = {}
    for dcid, key in keys.items():
        node = deps.node_entries.get(key)
        if node is not None:
            entries[dcid] = node
    misses = [dcid for dcid in unique if dcid not in entries]
    if misses:
        read, builder = await _read(request, deps, misses, tables, as_of=None)
        for dcid in misses:
            if dcid not in read.kinds:
                entries[dcid] = NotHeldEntry(ref=dcid)
                continue
            node, transient = await builder.build(dcid)
            entries[dcid] = node
            if not transient:
                deps.node_entries.put(keys[dcid], node)
    return NodesResponse(nodes=[entries[dcid] for dcid in dcids])


def classify_node_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, DcidTooLong):
        return ProblemSpec(
            status=400,
            code="request_invalid",
            detail=(
                f"dcid exceeds the {routing.DCID_MAX_LENGTH}-character ceiling "
                f"(received {exc.length} characters)"
            ),
            pointer="/dcid",
            retryable=False,
        )
    if isinstance(exc, AsOfNotRecorded):
        return ProblemSpec(
            status=400, code="request_invalid", detail=str(exc), pointer="/as_of", retryable=False
        )
    if isinstance(exc, NodeNotHeld):
        return ProblemSpec(
            status=404, code="node_not_held", detail=str(exc), pointer="/dcid", retryable=False
        )
    if isinstance(exc, GraphDeadlineExceededError):
        # This subclass maps to 504, so it must precede `GraphReadError`.
        return ProblemSpec(
            status=TIMEOUT_DEFAULT.status,
            code=TIMEOUT_DEFAULT.code,
            detail="the request's deadline expired during the graph node read; retry",
            retryable=TIMEOUT_DEFAULT.retryable,
        )
    if isinstance(exc, GraphReadError):
        logger.error("Graph node door's DCP read failed: %s", exc, exc_info=exc)
        return graph_read_problem(exc)
    return None


def graph_read_problem(exc: GraphReadError) -> ProblemSpec:
    """Refuse a failed graph read short of its deadline: 502 `upstream_error` where the graph
    answered 4xx, 503 `service_unavailable` otherwise."""
    upstream_status = exc.upstream_status
    if isinstance(upstream_status, int) and 400 <= upstream_status < 500:
        return ProblemSpec(status=502, code="upstream_error", retryable=True)
    return ProblemSpec(status=503, code="service_unavailable", retryable=True)


async def get_node(
    dcid: str, deps: GraphDepsDep, request: Request, as_of: date | None = None
) -> Response:
    if len(dcid) > routing.DCID_MAX_LENGTH:
        raise DcidTooLong(len(dcid))
    node, transient_gap = await asyncio.wait_for(
        _build_response(request, deps, dcid, as_of), timeout=GRAPH_NODE_TIMEOUT_SECONDS
    )
    return gapped_public_json(node.model_dump(mode="json"), request, transient_gap=transient_gap)


async def get_nodes(body: NodesRequest, deps: GraphDepsDep, request: Request) -> JSONResponse:
    response = await asyncio.wait_for(
        _build_batch_response(request, deps, body.dcids), timeout=GRAPH_NODE_TIMEOUT_SECONDS
    )
    return JSONResponse(response.model_dump(mode="json"))


def build_node_door() -> DoorContribution:
    # Load the code index now, so the first node request does not parse it.
    _role_properties(get_index())
    router = APIRouter()
    # `response_model` documents the union in OpenAPI. The handlers' responses pass through.
    router.add_api_route(
        "/api/graph/node/{dcid:dcid}",
        get_node,
        methods=["GET"],
        response_model=NodeResponse,
    )
    router.add_api_route(
        "/api/graph/nodes",
        get_nodes,
        methods=["POST"],
        response_model=NodesResponse,
    )
    return DoorContribution(
        router=router,
        registry_endpoints={
            "/api/graph/node/{dcid:dcid}": "graph.node",
            "/api/graph/nodes": "graph.nodes",
        },
        classify=classify_node_exception,
        exception_types=(
            DcidTooLong,
            AsOfNotRecorded,
            NodeNotHeld,
            GraphDeadlineExceededError,
            GraphReadError,
        ),
        exception_exemplars=(
            DcidTooLong(routing.DCID_MAX_LENGTH + 1),
            AsOfNotRecorded("exemplar/dcid"),
            NodeNotHeld("exemplar/dcid"),
            GraphDeadlineExceededError("exemplar"),
            GraphReadError("exemplar", upstream_status=502),
        ),
    )
