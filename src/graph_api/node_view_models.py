"""Wire model for one node's view of the graph: the node and its schema neighbours.

`SchemaNodeView` holds `generation`, the graph generation token the view was read from, null
where it is unknown; `centre`, the node as a full `SchemaNode`; `nodes`, its neighbours, each a
`SchemaNode` carrying its `label`, its `types` and the arcs that tie it to the centre, to the
centre's `subClassOf` chain or to the classes below the centre, sorted by `ref`, with the
provenances and origin of the arcs read from it; and `capped`, each listing the view cut, as a
`CappedListing` naming the node listed and the property, sorted. `{generation, nodes}` has the shape of a
`SchemaDocument`. `graph_api.node_view.build_node_view` builds it.
"""

from __future__ import annotations

from pydantic import Field

from dfo_platform.wire import ResponseModel
from graph_api.schema_models import SchemaNode


class CappedListing(ResponseModel):
    """One listing a view cut: an in-arc listing at its cap of 1,000 subjects or past the 2 MiB
    byte cap, a property of the centre past 100 values or with a literal past 2,000 characters,
    the classes below a Class centre (`subClassOf+`) at 1,000 classes or 12 levels with a class
    below them unread, a Property centre's range classes past the third
    (`rangeIncludes/subClassOf+`), or a value centre's type classes past the third
    (`typeOf/rangeIncludes`)."""

    ref: str = Field(description="the node whose listing was cut")
    property: str = Field(
        description="the property listed: an in-arc's subjects point through it, the centre's "
        "own values sit under it, subClassOf+ for the classes below the centre, "
        "rangeIncludes/subClassOf+ for a property's range classes read for a value list, or "
        "typeOf/rangeIncludes for a value's type classes read for their rangeIncludes subjects "
        "and value list"
    )


class SchemaNodeView(ResponseModel):
    """One node of the graph and its schema neighbours."""

    generation: str | None = Field(
        description="the graph generation the view was read from; null when unknown"
    )
    centre: SchemaNode = Field(description="the node, with every outgoing arc the graph holds")
    nodes: list[SchemaNode] = Field(
        description="the centre's neighbours, sorted by ref. Each carries its label and types "
        "and only the arcs the graph answered that tie it to the centre, to the centre's "
        "subClassOf chain or, for a Class centre, to the classes below it, so a walk down "
        "sub_class_of from the centre reaches every class the view lists below it. provenances are those of the arcs read from the neighbour (a "
        "member's typeOf, an in-arc subject's arc, a chain class's subClassOf edges) and "
        "declared_by follows from them; both are empty and null for a neighbour no arc was "
        "read from, such as a target of the centre's arcs"
    )
    capped: list[CappedListing] = Field(
        description="the listings the view cut, sorted by ref and property: an in-arc listing "
        "at 1,000 subjects or past 2 MiB (the centre's typeOf for a value list's members, any "
        "schema arc of the centre, and a value's type class under rangeIncludes), a property "
        "of the centre's other_properties past 100 values or with a literal past 2,000 "
        "characters, subClassOf+ on a Class centre whose classes below it stopped at 1,000 or "
        "at 12 levels with a class below them unread, rangeIncludes/subClassOf+ on a Property "
        "centre with more than three range classes, and typeOf/rangeIncludes on a value "
        "centre with more than three type classes"
    )
