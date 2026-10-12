"""Serve `GET /api/graph/node/{dcid}` and `POST /api/graph/nodes` from an allowlisted DCP read.

Both routes make one job on the shared DCP-read executor: the allowlisted
arc read, then one batched name read for every held node's label targets,
skipped when none has a target. A busy executor queues the job.
`GRAPH_NODE_TIMEOUT_SECONDS` bounds the client wait, queue time included,
and the deadline both reads share; a job still queued when the door times
out is cancelled before it reads. The GET route checks `DCID_MAX_LENGTH`
before reading and returns 400 for longer ids; the batch request model
applies the same ceiling and caps the list at `NODES_MAX_DCIDS`. A missing
node returns 404 on the GET route and a `not_held` entry in a batch. The
allowlist excludes multi-kilobyte `geoJson` arcs.

Shape precedence is StatVar, source, place-family entity, then generic
node. Labels skip names equal to the target dcid. Source labels prefer the
parent Dataset. A StatVar's and a source's `data_commons` come from the
request's origin catalogue, with no graph read. Entity containment and
source publisher/Dataset labels come from the name read. The StatVar
`facets`, `sources`, and `processing_level` fields remain null.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution
from dfo_platform.refusal import TIMEOUT_DEFAULT
from graph_api import routing
from graph_api.config import GRAPH_NODE_TIMEOUT_SECONDS
from graph_api.deps import GraphDeps, GraphDepsDep, run_dcp_read
from graph_api.node_models import (
    ContainedInPlace,
    EntityNode,
    Freshness,
    GenericNode,
    NodeResponse,
    NodesEntry,
    NodesRequest,
    NodesResponse,
    NotHeldNode,
    ParentDataset,
    Publisher,
    SourceNode,
    StatVarNode,
)
from graph_api.origin_models import DataCommonsRef
from graph_api.reader import GraphNodeReader, NodeArcsByDcid
from materialise.cadence import cadence_label
from materialise.graph import GraphDeadlineExceededError, GraphReadError
from materialise.roles import DegenerateObservationProperties, declared_roles
from origin_catalogue import current_catalogue

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
)

# `types` (the `node` shape's own field) covers `typeOf`, and `label`
# (every shape's own field) covers `name`, so neither belongs in `node`'s
# own `properties` map too.
_DEDICATED_FIELDS = frozenset({"typeOf", "name"})

_STAT_VAR_TYPE = "StatisticalVariable"
_SOURCE_TYPES = frozenset({"Provenance", "Source"})
_PLACE_FAMILY_TYPES = frozenset(
    {
        "Country",
        "Continent",
        "ContinentalUnion",
        "UNGeoRegion",
        "State",
        "County",
        "City",
        "Town",
        "Village",
    }
)

_LABEL_PROPERTY = "name"


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


def _label(arcs: dict[str, tuple[str, ...]], dcid: str) -> str | None:
    """Return the first name unequal to `dcid`. Some imports add a self-referential arc."""
    for value in arcs.get(_LABEL_PROPERTY, ()):
        if value != dcid:
            return value
    return None


def _classify_shape(type_of: tuple[str, ...]) -> str:
    if _STAT_VAR_TYPE in type_of:
        return "stat_var"
    if any(t in _SOURCE_TYPES for t in type_of):
        return "source"
    if any(t in _PLACE_FAMILY_TYPES for t in type_of):
        return "entity"
    return "node"


def _label_targets(arcs: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Return the second-hop dcids whose names a node's shape shows: a source's publisher and
    parent Dataset, or an entity's `containedInPlace` targets. Other shapes need none."""
    shape = _classify_shape(arcs.get("typeOf", ()))
    if shape == "source":
        return tuple(values[0] for prop in ("source", "isPartOf") if (values := arcs.get(prop, ())))
    if shape == "entity":
        return arcs.get("containedInPlace", ())
    return ()


def _read_nodes_sync(
    graph: GraphNodeReader, dcids: list[str], *, deadline: float
) -> tuple[NodeArcsByDcid, NodeArcsByDcid]:
    """Read the allowlisted arcs of `dcids`, then the names of every held node's label
    targets in one batched second read, skipped when no node has a target. Runs on the DCP
    executor, so both reads share one worker and one deadline."""
    confirmed = graph.node_arcs_many(dcids, properties=_ALLOWLIST, deadline=deadline)
    targets = dict.fromkeys(
        target for arcs in confirmed.values() if arcs is not None for target in _label_targets(arcs)
    )
    if not targets:
        return confirmed, {}
    labels = graph.node_arcs_many(list(targets), properties=(_LABEL_PROPERTY,), deadline=deadline)
    return confirmed, labels


async def _read_nodes(deps: GraphDeps, dcids: list[str]) -> tuple[NodeArcsByDcid, NodeArcsByDcid]:
    """Run `_read_nodes_sync` as one job on the shared DCP-read executor.

    Compute the deadline before submission so queue time counts against it.
    """
    deadline = time.monotonic() + GRAPH_NODE_TIMEOUT_SECONDS
    return await run_dcp_read(
        deps.dcp_executor, _read_nodes_sync, deps.graph, dcids, deadline=deadline
    )


def _stat_var_shape(dcid: str, arcs: dict[str, tuple[str, ...]]) -> StatVarNode:
    definition_values = arcs.get("definition", ())
    description_values = arcs.get("description", ())
    observation_properties = arcs.get("observationProperties", ())
    try:
        roles = list(declared_roles(observation_properties))
    except DegenerateObservationProperties:
        # Report the graph declaration as-is. The mint path rejects it.
        roles = list(observation_properties)
    population_values = arcs.get("populationType", ())
    population_type = population_values[0] if population_values else None
    return StatVarNode(
        dcid=dcid,
        label=_label(arcs, dcid),
        definition=definition_values[0] if definition_values else None,
        description=description_values[0] if description_values else None,
        roles=roles,
        population_type=population_type,
        source_indicator_codes=list(arcs.get("sourceIndicatorCode", ())),
        data_commons=[
            DataCommonsRef.of(origin) for origin in current_catalogue().data_commons(dcid)
        ],
    )


def _source_shape(
    dcid: str, arcs: dict[str, tuple[str, ...]], labels: NodeArcsByDcid
) -> SourceNode:
    """`publisher` (the `source` arc's target) and `dataset` (the `isPartOf` arc's target)
    take their labels from `labels`, the batched second-hop name read `_read_nodes_sync`
    makes. `label` prefers `dataset`'s own label, falling back to this node's own `name`
    arc when there is no parent Dataset, or that Dataset carries no name of its own (see
    `SourceNode`'s own docstring).
    """

    def first(prop: str) -> str | None:
        values = arcs.get(prop, ())
        return values[0] if values else None

    source_dcid = first("source")
    dataset_dcid = first("isPartOf")

    publisher = (
        Publisher(dcid=source_dcid, label=_label(labels.get(source_dcid) or {}, source_dcid))
        if source_dcid is not None
        else None
    )

    dataset_label = (
        _label(labels.get(dataset_dcid) or {}, dataset_dcid) if dataset_dcid is not None else None
    )
    dataset = (
        ParentDataset(dcid=dataset_dcid, label=dataset_label) if dataset_dcid is not None else None
    )

    label = dataset_label if dataset_label is not None else _label(arcs, dcid)
    release_frequency = first("sourceReleaseFrequency")
    catalogue = current_catalogue()
    row = catalogue.provenance(dcid) if catalogue.is_admitted(dcid) else None

    return SourceNode(
        dcid=dcid,
        label=label,
        license=first("license"),
        license_type=first("licenseType"),
        description=first("description"),
        url=first("url"),
        source_data_url=first("sourceDataUrl"),
        description_url=first("descriptionUrl"),
        transformation_logic_url=first("dataTransformationLogic"),
        last_data_refresh_date=first("lastDataRefreshDate"),
        next_data_refresh_date=first("nextDataRefreshDate"),
        earliest_observation_date=first("earliestObservationDate"),
        latest_observation_date=first("latestObservationDate"),
        publisher=publisher,
        dataset=dataset,
        freshness=Freshness(
            release_frequency=release_frequency,
            release_frequency_label=cadence_label(release_frequency),
            refresh_type=first("automaticRefreshType"),
        ),
        data_commons=DataCommonsRef.of(row.data_commons) if row is not None else None,
    )


def _entity_shape(
    dcid: str, arcs: dict[str, tuple[str, ...]], labels: NodeArcsByDcid
) -> EntityNode:
    type_of = arcs.get("typeOf", ())
    entity_type = next((t for t in type_of if t in _PLACE_FAMILY_TYPES), None)
    contained_in_dcids = arcs.get("containedInPlace", ())
    contained_in = [
        ContainedInPlace(dcid=target, label=_label(labels.get(target) or {}, target))
        for target in contained_in_dcids
    ]
    return EntityNode(
        dcid=dcid, label=_label(arcs, dcid), entity_type=entity_type, contained_in=contained_in
    )


def _generic_shape(dcid: str, arcs: dict[str, tuple[str, ...]]) -> GenericNode:
    types = list(arcs.get("typeOf", ()))
    properties = {
        prop: list(values)
        for prop, values in arcs.items()
        if prop not in _DEDICATED_FIELDS and values
    }
    return GenericNode(dcid=dcid, label=_label(arcs, dcid), types=types, properties=properties)


def _node_shape(
    dcid: str, arcs: dict[str, tuple[str, ...]], labels: NodeArcsByDcid
) -> NodeResponse:
    shape = _classify_shape(arcs.get("typeOf", ()))
    if shape == "stat_var":
        return _stat_var_shape(dcid, arcs)
    if shape == "source":
        return _source_shape(dcid, arcs, labels)
    if shape == "entity":
        return _entity_shape(dcid, arcs, labels)
    return _generic_shape(dcid, arcs)


async def _build_response(deps: GraphDeps, dcid: str) -> NodeResponse:
    confirmed, labels = await _read_nodes(deps, [dcid])
    arcs = confirmed.get(dcid)
    if arcs is None:
        raise NodeNotHeld(dcid)
    return _node_shape(dcid, arcs, labels)


async def _build_batch_response(deps: GraphDeps, dcids: list[str]) -> NodesResponse:
    confirmed, labels = await _read_nodes(deps, list(dict.fromkeys(dcids)))
    nodes: list[NodesEntry] = []
    for dcid in dcids:
        arcs = confirmed.get(dcid)
        nodes.append(NotHeldNode(dcid=dcid) if arcs is None else _node_shape(dcid, arcs, labels))
    return NodesResponse(nodes=nodes)


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
        upstream_status = exc.upstream_status
        if isinstance(upstream_status, int) and 400 <= upstream_status < 500:
            return ProblemSpec(status=502, code="upstream_error", retryable=True)
        return ProblemSpec(status=503, code="service_unavailable", retryable=True)
    return None


async def node_endpoint(dcid: str, deps: GraphDepsDep) -> JSONResponse:
    if len(dcid) > routing.DCID_MAX_LENGTH:
        raise DcidTooLong(len(dcid))
    response = await asyncio.wait_for(
        _build_response(deps, dcid), timeout=GRAPH_NODE_TIMEOUT_SECONDS
    )
    return JSONResponse(response.model_dump(mode="json"))


async def nodes_endpoint(body: NodesRequest, deps: GraphDepsDep) -> JSONResponse:
    response = await asyncio.wait_for(
        _build_batch_response(deps, body.dcids), timeout=GRAPH_NODE_TIMEOUT_SECONDS
    )
    return JSONResponse(response.model_dump(mode="json"))


def build_node_door() -> DoorContribution:
    router = APIRouter()
    # `response_model` documents the union in OpenAPI. JSONResponse passes through unchanged.
    router.add_api_route(
        "/api/graph/node/{dcid:dcid}",
        node_endpoint,
        methods=["GET"],
        response_model=NodeResponse,
    )
    router.add_api_route(
        "/api/graph/nodes",
        nodes_endpoint,
        methods=["POST"],
        response_model=NodesResponse,
    )
    return DoorContribution(
        router=router,
        classify=classify_node_exception,
        exception_types=(
            DcidTooLong,
            NodeNotHeld,
            GraphDeadlineExceededError,
            GraphReadError,
        ),
        exception_exemplars=(
            DcidTooLong(routing.DCID_MAX_LENGTH + 1),
            NodeNotHeld("exemplar/dcid"),
            GraphDeadlineExceededError("exemplar"),
            GraphReadError("exemplar", upstream_status=502),
        ),
    )
