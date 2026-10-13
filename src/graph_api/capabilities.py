"""Build `GET /api/graph/capabilities` from enforced limits and live deployment facts.

The four published limits use the same constants as their guards. The
`resolve`, `node`, `members`, `roots`, `sources`, and `schema` doors are live. `node_shape_gaps` and
`source_document_gaps` publish the member paths each door always gaps: on the node route, the
measure's definition of every variable with a measure
(`graph_api.card_definition.MEASURE_DEFINITION_GAP`). Every other source-document member is
filled where the graph carries it and is not listed as a permanent gap.

`source_catalogue_curated` is true because `GET /api/graph/sources` serves the
origin registry (`origin_catalogue.registry`), the reviewed list of sources
this deployment admits. `capability_version` and `build` come from callers
because `graph_api.version` hashes this module's schema.

`kinds` is the closed kind list resolve and the node route classify by, and `resolve_kinds` the
part of it a resolve `kinds` filter accepts.

`data_builds` names the graph generation and the builds of the origin
catalogue, the coverage index and the code index this process serves. They
change without a deploy, so the capability token hashes none of them: the
token's exemplar document omits them, which publishes every one as null.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from graph_api.card_definition import MEASURE_DEFINITION_GAP
from graph_api.config import (
    GRAPH_NODE_POOL_SIZE,
    GRAPH_NODE_TIMEOUT_SECONDS,
    GRAPH_RESOLVE_TIMEOUT_SECONDS,
)
from graph_api.kinds import KINDS, Kind
from graph_api.node_models import NODES_MAX_DCIDS
from graph_api.resolve_models import RESOLVE_KINDS
from graph_api.routing import DCID_MAX_LENGTH
from materialise.provenance_doc import UNCONDITIONAL_GAPS

# The non-health doors this surface serves, and their liveness.
_DOORS: tuple[str, ...] = ("resolve", "node", "members", "roots", "sources", "schema")


class CapabilityCaps(BaseModel):
    """Published input limits, one field per enforced constant."""

    model_config = ConfigDict(title="GraphCapabilityCaps")

    node_pool_size: int = Field(
        description="the bounded DCP-read executor's own worker count, shared by resolve confirmation reads and every node read"
    )
    resolve_timeout_seconds: int = Field(
        description="ceiling on how long a client waits for POST /api/graph/resolve before a timeout refusal"
    )
    node_timeout_seconds: int = Field(
        description="ceiling on how long a client waits for GET /api/graph/node/{dcid}, POST /api/graph/nodes, or GET /api/graph/documentation/{dcid} before a timeout refusal; GET /api/graph/node/{dcid}/members bounds its graph reads by it, and a label or profile read it cuts short leaves a gap"
    )
    max_dcid_chars: int = Field(
        description="ceiling on the length of the dcid path parameter of GET /api/graph/node/{dcid} and GET /api/graph/documentation/{dcid}, and of each dcid POST /api/graph/nodes takes; a longer value is refused 400 request_invalid"
    )
    max_node_batch_dcids: int = Field(
        description="the most dcids one POST /api/graph/nodes request names; a longer list is refused 400 request_invalid"
    )


class CapabilityDoor(BaseModel):
    """One door's liveness in this deployment."""

    name: str = Field(description="resolve, node, members, roots, sources, or schema")
    live: bool = Field(description="whether this deployment serves this door")


class DataBuilds(BaseModel):
    """The data builds this process serves, each null where it is unknown or absent."""

    model_config = ConfigDict(title="GraphDataBuilds")

    graph_generation: str | None = Field(
        default=None,
        description=(
            "the graph generation token: the graph service's Cloud Run revisions last read, "
            "joined with +; null while unknown"
        ),
    )
    origin_catalogue: str | None = Field(
        default=None, description="build id of the origin catalogue this request is bound to"
    )
    coverage_index: str | None = Field(
        default=None,
        description="build id of the series coverage index; null when none is loaded",
    )
    code_index: str | None = Field(
        default=None, description="content hash of the packaged code index export"
    )


class Capabilities(BaseModel):
    """The full `GET /api/graph/capabilities` document."""

    model_config = ConfigDict(title="GraphCapabilities")

    capability_version: str = Field(
        description=(
            "deterministic token identifying this capability set; distinct from "
            "schema_version and shared verbatim with problem-details bodies"
        )
    )
    build: str = Field(description="the commit sha the running image was built from")
    data_builds: DataBuilds = Field(
        description=(
            "the data builds this process serves; they change without a deploy and leave "
            "capability_version unchanged"
        )
    )
    doors: list[CapabilityDoor] = Field(description="per-door liveness")
    kinds: list[Kind] = Field(
        description="the closed kind list resolve and the node route classify entities by"
    )
    resolve_kinds: list[Kind] = Field(
        description="the kinds POST /api/graph/resolve accepts in its kinds filter"
    )
    recipe_classes: list[str] = Field(
        description="live unit-conversion recipe classes; empty, no conversion machinery"
    )
    node_shape_gaps: list[str] = Field(
        description=(
            "member paths the node door names in gaps on every node that carries the member, "
            "whatever the node"
        )
    )
    source_document_gaps: list[str] = Field(
        description=(
            "member paths the sources door's own documentation document always "
            "empties, whatever the source -- materialise.provenance_doc.UNCONDITIONAL_GAPS"
        )
    )
    source_catalogue_curated: bool = Field(
        description=(
            "whether GET /api/graph/sources serves the real curated allowlist "
            "rather than a sample standing in for one"
        )
    )
    caps: CapabilityCaps = Field(description="published input limits")


def build_capabilities(
    *, capability_version: str, build: str, data_builds: DataBuilds | None = None
) -> Capabilities:
    """Assemble the capabilities document. `capability_version` and `build`
    are always caller-supplied (never read from `graph_api.version`/
    `graph_api.config` directly here) -- see this module's own docstring
    for why. The served document passes `data_builds`; the token's exemplar
    omits it, so every build serves null there.
    """
    return Capabilities(
        capability_version=capability_version,
        build=build,
        data_builds=data_builds if data_builds is not None else DataBuilds(),
        doors=[CapabilityDoor(name=door, live=True) for door in _DOORS],
        kinds=list(KINDS),
        resolve_kinds=list(RESOLVE_KINDS),
        recipe_classes=[],
        node_shape_gaps=[MEASURE_DEFINITION_GAP],
        source_document_gaps=list(UNCONDITIONAL_GAPS),
        source_catalogue_curated=True,
        caps=CapabilityCaps(
            node_pool_size=GRAPH_NODE_POOL_SIZE,
            resolve_timeout_seconds=GRAPH_RESOLVE_TIMEOUT_SECONDS,
            node_timeout_seconds=GRAPH_NODE_TIMEOUT_SECONDS,
            max_dcid_chars=DCID_MAX_LENGTH,
            max_node_batch_dcids=NODES_MAX_DCIDS,
        ),
    )
