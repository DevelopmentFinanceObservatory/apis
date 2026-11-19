"""Wire model for one node's view of the graph: the node and its schema neighbours.

`SchemaNodeView` holds `generation`, the graph generation token the view was read from, null
where it is unknown; `centre`, the node as a full `SchemaNode`; `nodes`, its neighbours, each a
`SchemaNode` carrying its `label`, its `types` and the arcs that tie it to the centre or to the
centre's `subClassOf` chain, sorted by `ref`; and `capped`, the in-arc properties whose listing
stopped at `graph_api.node_view.IN_ARC_CAP`, sorted. `{generation, nodes}` has the shape of a
`SchemaDocument`. `graph_api.node_view.build_node_view` builds it.
"""

from __future__ import annotations

from pydantic import Field

from dfo_platform.wire import ResponseModel
from graph_api.schema_models import SchemaNode


class SchemaNodeView(ResponseModel):
    """One node of the graph and its schema neighbours."""

    generation: str | None = Field(
        description="the graph generation the view was read from; null when unknown"
    )
    centre: SchemaNode = Field(description="the node, with every outgoing arc the graph holds")
    nodes: list[SchemaNode] = Field(
        description="the centre's neighbours, sorted by ref. Each carries its label and types "
        "and only the arcs the graph answered that tie it to the centre or to the centre's "
        "subClassOf chain; provenances is empty and declared_by null, since a neighbour's "
        "typeOf is not read"
    )
    capped: list[str] = Field(
        description="the in-arc properties whose listing stopped at 1,000 subjects, sorted: "
        "typeOf for a value list's members, rangeIncludes for a value's list, and any schema "
        "arc of the centre"
    )
