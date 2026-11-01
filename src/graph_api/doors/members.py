"""Serve `GET /api/graph/node/{dcid}/members` and `GET /api/graph/roots`.

The members route lists what belongs to a node under one `relation`, each with one owner, read
in process:

- `codes`: a code system's codes, from the code index (`CodeIndex.codes_by_system`), ordered by
  code. The node is the system's graph id (`CodeSystemDeclaration.enum_dcid`).
- `children`: a code's direct children, from the code index (`CodeIndex.children`), ordered by
  code. A leaf answers an empty list.
- `datasets`: a source's upstream Datasets, from the origin catalogue through the source index
  (`build_source_catalogue`), ordered by dcid. A provenance with no Dataset is its own entry, as
  the source index lists it.
- `variables`: the variables DFO indexes under a Dataset or a Provenance, from the code index's
  provenance map (`CodeIndex.variables_by_provenance`) over the catalogue's map from Dataset to
  Provenance, ordered by dcid. A Provenance the code index does not associate adds no items, and
  the answer says `complete: false` with the gap `variables_not_indexed`.
- `members`: a group's members under a projection and an `as_of` date, from the membership
  policy, ordered by dcid.
- `countries`: a region's countries under its policy and a projection, ordered by dcid.
- `facets`: a variable's facets, from its facet profile as its card's availability reads them
  (`graph_api.card_reads`), ordered by facet id, which hashes the facet's metadata. The
  `source`, `unit` and `measurement_method` query values filter them; an empty
  `measurement_method` selects the facets that state no method. A profile that could not be
  read lists nothing with the gap `facets_not_read`, and names not read by the deadline leave
  the unit and method labels null with `labels_not_read`; both answer `no-store`. A surface
  that reads no profiles, because it has no profile cache, lists nothing with the permanent gap `facets_not_served` and answers `public`. A profile
  whose default scope fell back to the countries with data carries the card's permanent gap
  `stat_var.availability.countries_only`, since each facet's `places` then counts countries
  only. Every other relation refuses a filter at its pointer.

A membership answer carries the authority's gap (`membership_incomplete`) where it holds the
projection only in part, and `projection_not_recorded` with no items where it records none. Both
answer `complete: false`. Items carry the owner's label: the code index's for codes, the
catalogue's for Datasets, and the membership authority's canonical name for members. A
`variables` page takes its labels from one batched graph name read on the DCP-read executor,
under `GRAPH_NODE_TIMEOUT_SECONDS`; a read that fails, runs past it, or that the executor
refuses leaves them `null` with the gap `labels_not_read`. A label is `null` where its owner holds no name.

A page holds `MEMBERS_PAGE_DEFAULT` items unless `limit` asks for up to `MEMBERS_PAGE_MAX`. The
cursor (`graph_api.cursor`) is bound to the node, relation, projection, `as_of` date and order,
and for `facets` to the filter values, and continues after the last item's sort key, with the dcid breaking ties. `complete` is true
when the page reaches the end of a whole list, and `total` is the list's length, `null` where
the list is partial. `built_from` names the build of each table the list read.

Refusals are 400 `request_invalid` with a pointer: `/cursor` for a cursor that does not decode
or belongs to another walk, `/relation` for a relation a held node lacks, `/projection` and
`/as_of` for a parameter the relation does not take, and `/as_of` for a date asked of an
undated membership. A dcid the deployment does not hold answers 404 `node_not_held` in place
of the `/relation` refusal, as the node route does; one node read (`holds_node`) tells the two
apart, and only a refusal makes it. A group or region the membership policy holds is held
with or without a graph node. A membership policy the surface has not published refuses 503, as does a
`facets` request whose node's types were not read by the deadline. A failed node read refuses
as the node route's does: 502 `upstream_error` where the graph answered 4xx, 504 `timeout` at
the deadline, and 503 `service_unavailable` otherwise.

`GET /api/graph/roots` names where to start: each code system with its code count, each group
and region with its member count under the default projection, and each source with its Datasets.

Both routes answer `public, max-age=300` with an `ETag` (`gapped_public_json`), except a page
whose label, facet or profile read failed, which answers `no-store`. The members route must be mounted ahead of
`GET /api/graph/node/{dcid}`, whose dcid convertor accepts slashes and would otherwise take
`<dcid>/members` as one dcid.
"""

from __future__ import annotations

import asyncio
import bisect
import functools
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, Query, Request, Response
from pydantic import ConfigDict, Field

from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import CodeRow
from dc_graph import display_name
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution, path_registry_endpoints
from dfo_platform.static_response import gapped_public_json
from dfo_platform.wire import ResponseModel
from entity_resolution.membership import (
    DEFAULT_PROJECTION,
    Membership,
    MembershipKind,
    MembershipPolicy,
    NotRecorded,
    Projection,
)
from graph_api import cursor, routing
from graph_api.availability import NO_METHOD, FacetFilter, FacetItem
from graph_api.card_reads import read_facets, read_profiles
from graph_api.config import GRAPH_NODE_TIMEOUT_SECONDS
from graph_api.deps import (
    DcpExecutorSaturated,
    DcpExecutorShutDown,
    GraphDeps,
    GraphDepsDep,
    run_dcp_read,
)
from graph_api.doors.node import (
    DcidTooLong,
    NodeNotHeld,
    classify_node_exception,
    graph_read_problem,
    holds_node,
    published_membership,
)
from graph_api.doors.sources import build_source_catalogue
from graph_api.kinds import code_system_of
from graph_api.node_models import BuiltFromTable
from graph_api.reader import GraphNodeReader
from materialise.graph import GraphDeadlineExceededError, GraphReadError
from origin_catalogue import current_catalogue, load_registry

logger = logging.getLogger(__name__)

MEMBERS_PAGE_MAX = 1000
MEMBERS_PAGE_DEFAULT = 200

MembersRelation = Literal[
    "codes", "children", "datasets", "variables", "members", "countries", "facets"
]
"""The relations the members route lists."""

_MEMBERSHIP_KINDS: dict[MembersRelation, MembershipKind] = {
    "members": "group",
    "countries": "region",
}
_VARIABLES_NOT_INDEXED = "variables_not_indexed"
_PROJECTION_NOT_RECORDED = "projection_not_recorded"
_LABELS_NOT_READ = "labels_not_read"
_FACETS_NOT_READ = "facets_not_read"
_FACETS_NOT_SERVED = "facets_not_served"
_STAT_VAR_TYPE = "StatisticalVariable"

type SortKey = tuple[str, str]


class MemberItem(ResponseModel):
    """One item of a list: its dcid and the owner's label for it, `null` where the owner holds
    no name."""

    model_config = ConfigDict(title="GraphMemberItem")

    ref: str
    label: str | None


class CodeItem(MemberItem):
    """A code in a `codes` or `children` list, with its code and its parent's dcid."""

    model_config = ConfigDict(title="GraphCodeItem")

    code: str
    parent: str | None = Field(description="the parent code's dcid; null for a top-level code")


class MembersPage(ResponseModel):
    """One page of `GET /api/graph/node/{dcid}/members`."""

    model_config = ConfigDict(title="GraphMembersPage")

    ref: str = Field(description="the node whose list this is")
    relation: MembersRelation
    projection: Projection | None = Field(
        description="the membership projection listed; null for a relation that takes none"
    )
    as_of: date | None = Field(description="the date membership was read at; null for today")
    items: list[CodeItem | FacetItem | MemberItem]
    next_cursor: str | None = Field(
        description="the cursor for the next page; null when this page ends the list"
    )
    complete: bool = Field(
        description="whether the walk reached the end of a list the owner holds in full"
    )
    total: int | None = Field(description="the list's length; null where the list is partial")
    gaps: list[str] = Field(description="the code of each way the list is partial")
    built_from: dict[BuiltFromTable, str] = Field(
        description="the build of each table the list read"
    )


class RootEntry(ResponseModel):
    """A place to start listing from, with the length of its list."""

    model_config = ConfigDict(title="GraphRootEntry")

    ref: str
    label: str | None
    count: int = Field(
        description=(
            "codes for a code system, members under the default projection for a group or "
            "region, Datasets for a source"
        )
    )


class Roots(ResponseModel):
    """`GET /api/graph/roots`: where to start listing. Each list is ordered by dcid."""

    model_config = ConfigDict(title="GraphRoots")

    code_systems: list[RootEntry]
    groups: list[RootEntry]
    regions: list[RootEntry]
    sources: list[RootEntry]
    built_from: dict[BuiltFromTable, str] = Field(
        description="the build of each table the document read"
    )


class ListRequestInvalid(Exception):
    """A list request the door refuses 400 `request_invalid` at `pointer`."""

    def __init__(self, pointer: str, detail: str) -> None:
        super().__init__(detail)
        self.pointer = pointer


class RelationLacked(ListRequestInvalid):
    """The node has no list of the requested relation. The door refuses it 400 at `/relation`
    where the node is held, and 404 `node_not_held` where it is not."""

    def __init__(self, dcid: str, relation: MembersRelation) -> None:
        super().__init__("/relation", f"{dcid!r} has no {relation} list")


class NodeTypesUnread(Exception):
    """The node's types were not read by the deadline, so the door cannot tell whether it is a
    variable. The door refuses 503."""

    def __init__(self, dcid: str) -> None:
        super().__init__(f"the types of {dcid!r} could not be read; retry")


class MembershipUnavailable(Exception):
    """The surface has not published a membership policy. The door refuses 503."""

    def __init__(self) -> None:
        super().__init__("the membership policy is not loaded")


@dataclass(frozen=True, slots=True)
class _Listing[R]:
    """A whole list in its order, and what a page of it says about the list."""

    rows: Sequence[R]
    sort_key: Callable[[R], SortKey]
    item: Callable[[R], MemberItem | FacetItem]
    order: Literal["code", "dcid", "facet_id"]
    built_from: dict[BuiltFromTable, str]
    gaps: tuple[str, ...] = ()
    whole: bool = True
    # The page's labels come from one graph name read, as no in-process table names its items.
    graph_labels: bool = False
    # A gap comes from a failed read, so the page is not cached.
    transient: bool = False


@dataclass(frozen=True, slots=True)
class _CatalogueLists:
    """The origin catalogue's lists for one build: each source's label and Datasets, and each
    Dataset's provenances."""

    sources: dict[str, str]
    datasets_by_source: dict[str, tuple[MemberItem, ...]]
    provenances_by_dataset: dict[str, tuple[str, ...]]


@functools.lru_cache(maxsize=1)
def _catalogue_lists(build_id: str) -> _CatalogueLists:
    """The lists of the bound catalogue, built once per catalogue build from the source index,
    so they hold what `GET /api/graph/sources` holds. `build_id` keys the cache and must be the
    bound catalogue's own."""
    index = build_source_catalogue(registry=load_registry(), catalogue=current_catalogue())
    sources: dict[str, str] = {}
    datasets: dict[str, dict[str, MemberItem]] = {}
    provenances: dict[str, tuple[str, ...]] = {}
    for entry in index.sources:
        sources.setdefault(entry.source.ref, entry.source.label)
        held = datasets.setdefault(entry.source.ref, {})
        for dataset in entry.datasets:
            held[dataset.ref] = MemberItem(ref=dataset.ref, label=dataset.label)
            provenances[dataset.ref] = tuple(p.ref for p in dataset.provenances)
    return _CatalogueLists(
        sources=sources,
        datasets_by_source={
            ref: tuple(held[key] for key in sorted(held)) for ref, held in datasets.items()
        },
        provenances_by_dataset=provenances,
    )


def _by_dcid(dcid: str) -> SortKey:
    return (dcid, dcid)


def _by_code(row: CodeRow) -> SortKey:
    return (row.code, row.dcid)


def _code_item(row: CodeRow) -> CodeItem:
    return CodeItem(ref=row.dcid, label=row.label, code=row.code, parent=row.parent)


def _unlabelled(dcid: str) -> MemberItem:
    return MemberItem(ref=dcid, label=None)


def _labelled_by(labels: Mapping[str, str]) -> Callable[[str], MemberItem]:
    def item(dcid: str) -> MemberItem:
        return MemberItem(ref=dcid, label=labels.get(dcid))

    return item


def _same[I: MemberItem | FacetItem](item: I) -> I:
    return item


def _by_facet_id(item: FacetItem) -> SortKey:
    return (item.ref, item.ref)


def _codes(dcid: str, index: CodeIndex) -> _Listing[CodeRow]:
    declaration = code_system_of(index, dcid)
    if declaration is None:
        raise RelationLacked(dcid, "codes")
    return _Listing(
        rows=index.codes_by_system.get(declaration.code_system, ()),
        sort_key=_by_code,
        item=_code_item,
        order="code",
        built_from={"code_index": index.stamp.content_hash},
    )


def _children(dcid: str, index: CodeIndex) -> _Listing[CodeRow]:
    if index.code_by_dcid(dcid) is None:
        raise RelationLacked(dcid, "children")
    return _Listing(
        rows=index.children(dcid),
        sort_key=_by_code,
        item=_code_item,
        order="code",
        built_from={"code_index": index.stamp.content_hash},
    )


def _datasets(dcid: str) -> _Listing[MemberItem]:
    catalogue = current_catalogue()
    datasets = _catalogue_lists(catalogue.build_id).datasets_by_source.get(dcid)
    if datasets is None:
        raise RelationLacked(dcid, "datasets")
    return _Listing(
        rows=datasets,
        sort_key=lambda item: _by_dcid(item.ref),
        item=_same,
        order="dcid",
        built_from={"origin_catalogue": catalogue.build_id},
    )


def _variables(dcid: str, index: CodeIndex) -> _Listing[str]:
    """A Dataset's variables are the union of its provenances' variables. A dcid the catalogue
    lists as no Dataset is read as a Provenance, if either table holds it."""
    catalogue = current_catalogue()
    provenances = _catalogue_lists(catalogue.build_id).provenances_by_dataset.get(dcid)
    if provenances is None:
        if dcid not in index.variables_by_provenance and catalogue.provenance(dcid) is None:
            raise RelationLacked(dcid, "variables")
        provenances = (dcid,)
    held = [
        index.variables_by_provenance[p] for p in provenances if p in index.variables_by_provenance
    ]
    rows = held[0] if len(held) == 1 else tuple(sorted({v for dcids in held for v in dcids}))
    whole = len(held) == len(provenances)
    return _Listing(
        rows=rows,
        sort_key=_by_dcid,
        item=_unlabelled,
        order="dcid",
        built_from={
            "code_index": index.stamp.content_hash,
            "origin_catalogue": catalogue.build_id,
        },
        gaps=() if whole else (_VARIABLES_NOT_INDEXED,),
        whole=whole,
        graph_labels=True,
    )


def _policy(request: Request) -> MembershipPolicy:
    if (policy := published_membership(request)) is None:
        raise MembershipUnavailable()
    return policy


def _membership(
    dcid: str,
    relation: MembersRelation,
    policy: MembershipPolicy,
    *,
    projection: Projection,
    as_of: date | None,
) -> _Listing[str]:
    membership = policy.get(dcid) if dcid in policy else None
    if membership is None or membership.kind != _MEMBERSHIP_KINDS[relation]:
        raise RelationLacked(dcid, relation)
    if as_of is not None and not membership.dated:
        raise ListRequestInvalid("/as_of", f"{dcid!r} has an undated membership")
    members = membership.members(projection, as_of)
    rows: tuple[str, ...] = ()
    gaps: tuple[str, ...] = (_PROJECTION_NOT_RECORDED,)
    if not isinstance(members, NotRecorded):
        gap = membership.gap(projection)
        rows, gaps = members, (gap.code,) if gap is not None else ()
    return _Listing(
        rows=rows,
        sort_key=_by_dcid,
        item=_labelled_by(membership.member_labels),
        order="dcid",
        built_from={"membership": membership.authority.version},
        gaps=gaps,
        whole=not gaps,
    )


async def _facets(deps: GraphDeps, dcid: str, where: FacetFilter) -> _Listing[FacetItem]:
    """A variable's facets that `where` keeps, from its profile, under the node deadline. A
    profile that could not be read lists nothing, with the transient gap `facets_not_read`. A
    surface that reads no profiles lists nothing, with the permanent gap `facets_not_served`."""
    deadline = time.monotonic() + GRAPH_NODE_TIMEOUT_SECONDS
    profiles = None
    if deps.card_reads is not None:
        facts = await deps.card_reads.node_facts_of(dcid, deadline=deadline)
        if facts is None:
            raise NodeTypesUnread(dcid)
        if _STAT_VAR_TYPE not in facts.types:
            raise RelationLacked(dcid, "facets")
        profiles = await read_profiles(deps, [dcid], deadline=deadline)
    built_from: dict[BuiltFromTable, str] = {"origin_catalogue": current_catalogue().build_id}
    facets = None
    if profiles is not None:
        built_from["coverage_index"] = profiles.coverage_build
        facets = await read_facets(profiles, dcid, where=where, deadline=deadline)
    if facets is None or facets.items is None:
        transient = profiles is not None
        return _Listing(
            rows=(),
            sort_key=_by_facet_id,
            item=_same,
            order="facet_id",
            built_from=built_from,
            gaps=(_FACETS_NOT_READ if transient else _FACETS_NOT_SERVED,),
            whole=False,
            transient=transient,
        )
    return _Listing(
        rows=facets.items,
        sort_key=_by_facet_id,
        item=_same,
        order="facet_id",
        built_from=built_from,
        gaps=(*facets.gaps, *((_LABELS_NOT_READ,) if facets.labels_failed else ())),
        transient=facets.labels_failed,
    )


def _bound_filters(where: FacetFilter) -> tuple[tuple[str, str | None], ...]:
    """The filter values a `facets` cursor binds: each named axis and its value, with `None`
    for "no stated method", so an absent filter and an empty one bind apart."""
    bound: list[tuple[str, str | None]] = []
    if where.source is not None:
        bound.append(("source", where.source))
    if where.unit is not None:
        bound.append(("unit", where.unit))
    if (method := where.measurement_method) is not None:
        bound.append(("measurement_method", None if method is NO_METHOD else method))
    return tuple(bound)


def _facet_filter(
    source: str | None, unit: str | None, measurement_method: str | None
) -> FacetFilter:
    """The filter the query names. An empty `measurement_method` selects facets that state no
    method."""
    return FacetFilter(
        source=source,
        unit=unit,
        measurement_method=NO_METHOD if measurement_method == "" else measurement_method,
    )


async def _listing(
    request: Request,
    deps: GraphDeps,
    dcid: str,
    relation: MembersRelation,
    *,
    projection: Projection | None,
    as_of: date | None,
    where: FacetFilter,
) -> _Listing:
    match relation:
        case "codes":
            return _codes(dcid, get_index())
        case "children":
            return _children(dcid, get_index())
        case "datasets":
            return _datasets(dcid)
        case "variables":
            return _variables(dcid, get_index())
        case "members" | "countries":
            assert projection is not None, "a membership relation always names its projection"
            policy = _policy(request)
            return _membership(dcid, relation, policy, projection=projection, as_of=as_of)
        case "facets":
            return await _facets(deps, dcid, where)


def _page[R](
    listing: _Listing[R], binding: cursor.Binding, after: str | None, limit: int
) -> tuple[list[MemberItem | FacetItem], str | None]:
    """The page of `listing` that follows the cursor `after`, and the cursor past it."""
    start = 0
    if after is not None:
        try:
            position = cursor.decode(after, binding)
        except cursor.CursorInvalid as exc:
            raise ListRequestInvalid("/cursor", str(exc)) from None
        start = bisect.bisect_right(
            listing.rows, (position.key, position.dcid), key=listing.sort_key
        )
    rows = listing.rows[start : start + limit]
    more = start + limit < len(listing.rows)
    next_cursor = None
    if more:
        key, dcid = listing.sort_key(rows[-1])
        next_cursor = cursor.encode(binding, cursor.Position(key=key, dcid=dcid))
    return [listing.item(row) for row in rows], next_cursor


def _names_sync(graph: GraphNodeReader, dcids: list[str], *, deadline: float) -> dict[str, str]:
    """Each dcid's `display_name`, read from its `name` arc alone, as the node door labels it;
    a dcid with none is left out. An `alternateName` never becomes a label."""
    arcs = graph.node_arcs_many(dcids, properties=("name",), deadline=deadline)
    return {
        dcid: label
        for dcid, held in arcs.items()
        if held is not None and (label := display_name(held.get("name", ()), dcid)) is not None
    }


async def _graph_labels(
    deps: GraphDeps, items: list[MemberItem | FacetItem]
) -> tuple[list[MemberItem | FacetItem], bool]:
    """`items` labelled from one batched name read on the DCP executor, under the node
    deadline, and whether the read failed. A failed or refused read leaves every label null."""
    if not items:
        return items, False
    deadline = time.monotonic() + GRAPH_NODE_TIMEOUT_SECONDS
    try:
        labels = await asyncio.wait_for(
            run_dcp_read(
                deps.dcp_executor,
                _names_sync,
                deps.graph,
                [item.ref for item in items],
                deadline=deadline,
            ),
            timeout=GRAPH_NODE_TIMEOUT_SECONDS,
        )
    except (GraphReadError, TimeoutError, DcpExecutorSaturated, DcpExecutorShutDown):
        logger.warning("Graph members door's label read failed", exc_info=True)
        return items, True
    labelled: list[MemberItem | FacetItem] = [
        MemberItem(ref=item.ref, label=labels.get(item.ref)) for item in items
    ]
    return labelled, False


async def list_members(
    dcid: str,
    request: Request,
    deps: GraphDepsDep,
    relation: MembersRelation,
    limit: int = MEMBERS_PAGE_DEFAULT,
    cursor_token: Annotated[str | None, Query(alias="cursor")] = None,
    projection: Projection | None = None,
    as_of: date | None = None,
    source: str | None = None,
    unit: str | None = None,
    measurement_method: str | None = None,
) -> Response:
    if len(dcid) > routing.DCID_MAX_LENGTH:
        raise DcidTooLong(len(dcid))
    # Checked here, as a query-parameter bound would refuse without a pointer.
    if not 1 <= limit <= MEMBERS_PAGE_MAX:
        raise ListRequestInvalid("/limit", f"limit must be between 1 and {MEMBERS_PAGE_MAX}")
    if relation in _MEMBERSHIP_KINDS:
        projection = projection or DEFAULT_PROJECTION
    elif projection is not None:
        raise ListRequestInvalid("/projection", f"the {relation} list takes no projection")
    elif as_of is not None:
        raise ListRequestInvalid("/as_of", f"the {relation} list takes no as_of date")
    filters = {"source": source, "unit": unit, "measurement_method": measurement_method}
    if relation != "facets" and (
        named := next((name for name, value in filters.items() if value is not None), None)
    ):
        raise ListRequestInvalid(f"/{named}", f"the {relation} list takes no {named} filter")
    where = _facet_filter(source, unit, measurement_method)
    try:
        listing = await _listing(
            request,
            deps,
            dcid,
            relation,
            projection=projection,
            as_of=as_of,
            where=where,
        )
    except RelationLacked:
        if not await holds_node(request, deps, dcid):
            raise NodeNotHeld(dcid) from None
        raise
    binding = cursor.Binding(
        node=dcid,
        relation=relation,
        projection=projection.value if projection is not None else None,
        as_of=as_of,
        order=listing.order,
        filters=_bound_filters(where) if relation == "facets" else None,
    )
    items, next_cursor = _page(listing, binding, cursor_token, limit)
    unread = False
    if listing.graph_labels:
        items, unread = await _graph_labels(deps, items)
    page = MembersPage(
        ref=dcid,
        relation=relation,
        projection=projection,
        as_of=as_of,
        items=items,
        next_cursor=next_cursor,
        complete=next_cursor is None and listing.whole,
        total=len(listing.rows) if listing.whole else None,
        gaps=[*listing.gaps, *((_LABELS_NOT_READ,) if unread else ())],
        built_from=listing.built_from,
    )
    return gapped_public_json(
        page.model_dump(mode="json"), request, transient_gap=unread or listing.transient
    )


def _membership_roots(policy: MembershipPolicy, kind: MembershipKind) -> list[RootEntry]:
    def count(membership: Membership) -> int:
        members = membership.member_count(DEFAULT_PROJECTION)
        return 0 if isinstance(members, NotRecorded) else members

    return sorted(
        (RootEntry(ref=m.ref, label=m.label, count=count(m)) for m in policy if m.kind == kind),
        key=lambda entry: entry.ref,
    )


async def list_roots(request: Request) -> Response:
    index = get_index()
    catalogue = current_catalogue()
    lists = _catalogue_lists(catalogue.build_id)
    policy = _policy(request)
    code_systems = sorted(
        (
            RootEntry(
                ref=declaration.enum_dcid,
                label=declaration.plain_name,
                count=len(index.codes_by_system.get(declaration.code_system, ())),
            )
            for declaration in index.declarations.values()
            if declaration.enum_dcid is not None
        ),
        key=lambda entry: entry.ref,
    )
    roots = Roots(
        code_systems=code_systems,
        groups=_membership_roots(policy, "group"),
        regions=_membership_roots(policy, "region"),
        sources=[
            RootEntry(ref=ref, label=label, count=len(lists.datasets_by_source[ref]))
            for ref, label in sorted(lists.sources.items())
        ],
        built_from={
            "code_index": index.stamp.content_hash,
            "origin_catalogue": catalogue.build_id,
            "membership": "+".join(sorted({m.authority.version for m in policy})),
        },
    )
    return gapped_public_json(roots.model_dump(mode="json"), request, transient_gap=False)


def classify_members_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, ListRequestInvalid):
        return ProblemSpec(
            status=400,
            code="request_invalid",
            detail=str(exc),
            pointer=exc.pointer,
            retryable=False,
        )
    if isinstance(exc, MembershipUnavailable | NodeTypesUnread):
        return ProblemSpec(status=503, code="service_unavailable", detail=str(exc), retryable=True)
    if isinstance(exc, DcidTooLong | NodeNotHeld | GraphDeadlineExceededError):
        # The node read that tells a held node from an unheld one refuses as the node route does.
        return classify_node_exception(exc)
    if isinstance(exc, GraphReadError):
        return graph_read_problem(exc)._replace(
            log=f"Graph members door's DCP read failed: {exc}", log_traceback=True
        )
    return None


def build_members_door() -> DoorContribution:
    """The members and roots routes. Mount this door ahead of the node door."""
    router = APIRouter()
    # `response_model` documents the response body in OpenAPI. The handlers' responses pass
    # through.
    router.add_api_route(
        "/api/graph/node/{dcid:dcid}/members",
        list_members,
        methods=["GET"],
        response_model=MembersPage,
    )
    router.add_api_route("/api/graph/roots", list_roots, methods=["GET"], response_model=Roots)
    return DoorContribution(
        router=router,
        registry_endpoints=path_registry_endpoints(
            router,
            {
                "/api/graph/node/{dcid:dcid}/members": "graph.members",
                "/api/graph/roots": "graph.roots",
            },
        ),
        classify=classify_members_exception,
        exception_types=(
            ListRequestInvalid,
            RelationLacked,
            MembershipUnavailable,
            NodeTypesUnread,
            DcidTooLong,
            NodeNotHeld,
            GraphDeadlineExceededError,
            GraphReadError,
        ),
        exception_exemplars=(
            ListRequestInvalid("/cursor", "exemplar"),
            RelationLacked("exemplar/dcid", "codes"),
            MembershipUnavailable(),
            NodeTypesUnread("exemplar/dcid"),
            DcidTooLong(routing.DCID_MAX_LENGTH + 1),
            NodeNotHeld("exemplar/dcid"),
            GraphDeadlineExceededError("exemplar"),
            GraphReadError("exemplar", upstream_status=404),
        ),
    )
