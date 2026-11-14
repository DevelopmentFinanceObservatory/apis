"""Whether a StatVar's constraint value names a resolved graph node.

`code_index.export.export_statvars` refuses a release whose StatVar binds a constraint
property to a value that fails this test, and `graph_api.ontology_check` reports a candidate
row that would meet the same refusal.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Final, Literal

PROVISIONAL_NODE: Final[str] = "ProvisionalNode"
"""The type the graph gives a node it holds only as a placeholder for a dangling reference."""

type ConstraintValueFault = Literal["not_held", "untyped", "provisional"]


def constraint_value_fault(types: Collection[str] | None) -> ConstraintValueFault | None:
    """Why a constraint value's node is not a resolved node, or `None` when it is.

    Args:
        types: the value node's `typeOf` values as a batched read answers them, or `None`
            when the graph holds no node. `dc_graph.UNRESOLVED_NODE`, a target the graph
            could not resolve, counts as no type, so a caller may pass the values filtered
            or not.

    Returns:
        `"not_held"` when the graph holds no node, `"untyped"` when the node carries no
        resolved `typeOf`, `"provisional"` when its only type is `ProvisionalNode`, and `None`
        otherwise. A `ProvisionalNode` type beside a real type is resolved.
    """
    from dc_graph import UNRESOLVED_NODE

    if types is None:
        return "not_held"
    distinct = set(types) - {UNRESOLVED_NODE}
    if not distinct:
        return "untyped"
    if distinct == {PROVISIONAL_NODE}:
        return "provisional"
    return None
