"""The `{ref, label}` reference every Graph API response uses for a graph node, and the
readers of a node's arcs that every door shares."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import ConfigDict

from dfo_platform.wire import ResponseModel

DOCUMENTATION_PATH = "/api/graph/documentation/{ref}"
"""The path of a source node's documentation; format it with the node's `ref`."""


class NodeRef(ResponseModel):
    """A graph reference. `label` is `null` when the target is not held, carries no `name`, or
    carries only a `name` equal to its own dcid."""

    model_config = ConfigDict(title="GraphNodeRef")

    ref: str
    label: str | None


def first_name(names: Iterable[str], dcid: str) -> str | None:
    """The first of a node's `name` values that differs from its `dcid`. Some imports add a
    self-referential arc, which is no name."""
    return next((name for name in names if name != dcid), None)


def first_value(arcs: Mapping[str, tuple[str, ...]], prop: str) -> str | None:
    """The first value of `prop` among a node's arcs, `None` where it has none."""
    values = arcs.get(prop, ())
    return values[0] if values else None
