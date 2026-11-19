"""Wire model for one node's view of the graph: the node and its schema neighbours.

`SchemaNodeView` holds `generation`, the graph generation token the view was read from, null
where it is unknown; `centre`, the node as a full `SchemaNode`; `nodes`, its neighbours, each a
`SchemaNode` carrying its `label`, its `types` and the arcs that tie it to the centre or to the
centre's `subClassOf` chain, sorted by `ref`, with the provenances and origin of the arcs read
from it; and `capped`, each in-arc listing that stopped at its cap, as a `CappedListing` naming
the node listed and the property, sorted. `{generation, nodes}` has the shape of a
`SchemaDocument`. `graph_api.node_view.build_node_view` builds it.
"""

from __future__ import annotations

from pydantic import Field

from dfo_platform.wire import ResponseModel
from graph_api.schema_models import SchemaNode


class CappedListing(ResponseModel):
    """One in-arc listing a view read that stopped at its cap of 1,000 subjects."""

    ref: str = Field(description="the node whose in-arcs were listed")
    property: str = Field(description="the property the listed subjects point through")


class SchemaNodeView(ResponseModel):
    """One node of the graph and its schema neighbours."""

    generation: str | None = Field(
        description="the graph generation the view was read from; null when unknown"
    )
    centre: SchemaNode = Field(description="the node, with every outgoing arc the graph holds")
    nodes: list[SchemaNode] = Field(
        description="the centre's neighbours, sorted by ref. Each carries its label and types "
        "and only the arcs the graph answered that tie it to the centre or to the centre's "
        "subClassOf chain. provenances are those of the arcs read from the neighbour (a "
        "member's typeOf, an in-arc subject's arc, a chain class's subClassOf edges) and "
        "declared_by follows from them; both are empty and null for a neighbour no arc was "
        "read from, such as a target of the centre's arcs"
    )
    capped: list[CappedListing] = Field(
        description="the in-arc listings that stopped at 1,000 subjects, sorted by ref and "
        "property: the centre's typeOf for a value list's members, any schema arc of the "
        "centre, and a value's type class under rangeIncludes"
    )
