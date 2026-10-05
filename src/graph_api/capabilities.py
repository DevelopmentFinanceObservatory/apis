"""Build `GET /api/graph/capabilities` from enforced limits and live deployment facts.

The four published limits use the same constants as their guards. The
`resolve`, `node`, and `sources` doors are live. `node_shape_gaps` and
`source_document_gaps` publish their permanent null fields. Every other
source-document member is filled where the graph carries it and is not
listed as a permanent gap.

`source_catalogue_curated` is true because `GET /api/graph/sources` serves the
origin registry (`origin_catalogue.registry`), the reviewed list of sources
this deployment admits. `capability_version` and `build` come from callers
because `graph_api.version` hashes this module's schema.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from graph_api.config import (
    GRAPH_NODE_POOL_SIZE,
    GRAPH_NODE_TIMEOUT_SECONDS,
    GRAPH_RESOLVE_TIMEOUT_SECONDS,
)
from graph_api.node_models import NODES_MAX_DCIDS
from graph_api.routing import DCID_MAX_LENGTH
from materialise.provenance_doc import UNCONDITIONAL_GAPS

# The three non-health doors this surface serves, and their liveness.
_DOORS: tuple[str, ...] = ("resolve", "node", "sources")


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
        description="ceiling on how long a client waits for GET /api/graph/node/{dcid}, POST /api/graph/nodes, or GET /api/graph/documentation/{dcid} before a timeout refusal"
    )
    max_dcid_chars: int = Field(
        description="ceiling on the length of the dcid path parameter of GET /api/graph/node/{dcid} and GET /api/graph/documentation/{dcid}, and of each dcid POST /api/graph/nodes takes; a longer value is refused 400 request_invalid"
    )
    max_node_batch_dcids: int = Field(
        description="the most dcids one POST /api/graph/nodes request names; a longer list is refused 400 request_invalid"
    )


class CapabilityDoor(BaseModel):
    """One door's liveness in this deployment."""

    name: str = Field(description="resolve, node, or sources")
    live: bool = Field(description="whether this deployment serves this door")


class NodeShapeGaps(BaseModel):
    """The node door's own recorded permanent-null gaps -- see
    `graph_api.node_models`'s own docstring for why each is a gap rather
    than an oversight.
    """

    facets: bool = Field(description="whether stat_var.facets is live (currently always null)")
    sources: bool = Field(description="whether stat_var.sources is live (currently always null)")
    processing_level: bool = Field(
        description="whether stat_var.processing_level is live (currently always null)"
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
    doors: list[CapabilityDoor] = Field(description="per-door liveness")
    recipe_classes: list[str] = Field(
        description="live unit-conversion recipe classes; empty, no conversion machinery"
    )
    node_shape_gaps: NodeShapeGaps = Field(
        description="the node door's own recorded permanent-null gaps"
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


def build_capabilities(*, capability_version: str, build: str) -> Capabilities:
    """Assemble the capabilities document. `capability_version` and `build`
    are always caller-supplied (never read from `graph_api.version`/
    `graph_api.config` directly here) -- see this module's own docstring
    for why.
    """
    return Capabilities(
        capability_version=capability_version,
        build=build,
        doors=[CapabilityDoor(name=door, live=True) for door in _DOORS],
        recipe_classes=[],
        node_shape_gaps=NodeShapeGaps(facets=False, sources=False, processing_level=False),
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
