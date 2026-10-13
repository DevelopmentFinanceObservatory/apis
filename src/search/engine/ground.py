"""Confirm graph nodes before minting `GraphRef` values.

Absent nodes raise `GroundingMiss`; transport and non-2xx failures propagate
as `EngineInfraError`. Callers either drop missing dcids or return no data.
"""

from __future__ import annotations

import contextlib

from search.engine.clients.graph import EngineGraphClient
from search.engine.errors import GroundingMiss
from search.models import GraphRef


def _ref_from_label(dcid: str, label: str | None) -> GraphRef:
    """Return a `GraphRef`, or raise `GroundingMiss` when `label` is absent."""
    if label is None:
        raise GroundingMiss(dcid)
    return GraphRef(ref=dcid, label=label)


def graphref(dcid: str, *, graph: EngineGraphClient) -> GraphRef:
    """Return the graph label for `dcid`.

    Raises `GroundingMiss` for an absent node and `EngineInfraError` for a
    transport or non-2xx failure.
    """
    return _ref_from_label(dcid, graph.node_label(dcid))


def graphrefs(dcids: list[str], *, graph: EngineGraphClient) -> list[GraphRef]:
    """Confirm each dcid, dropping absent nodes and preserving input order.

    `GroundingMiss` is ignored; `GraphInfraError` propagates as
    `EngineInfraError`.
    """
    result: list[GraphRef] = []
    for dcid in dcids:
        with contextlib.suppress(GroundingMiss):
            result.append(graphref(dcid, graph=graph))
    return result
