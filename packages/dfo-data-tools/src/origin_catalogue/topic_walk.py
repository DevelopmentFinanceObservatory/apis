"""The walk from Topic roots to their member variables.

Search expands a recalled topic with this walk at query time
(`search.engine.retrieval.topics.expand_topics`), and the catalogue job runs
it over every topic to build the topic table, so both see the same members.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final, Protocol

TOPIC_EXPAND_MAX_DEPTH: Final[int] = 4
"""Arc levels read per walk. Four cover the deepest observed SDG chain:
indicator Topic, series Topic, StatVarPeerGroup, then the variable."""

TOPIC_EXPAND_MAX_MEMBERS: Final[int] = 50
"""A root with more members than this is dropped. Topic and StatVarPeerGroup
nodes can hold 1,700 to 5,200 members."""

_MEMBER_ARCS: Final[tuple[str, ...]] = ("relevantVariable", "member")

# SDG taxonomy containers use relevantVariable and member arcs at different levels.
_CONTAINER_TYPES: Final[frozenset[str]] = frozenset({"Topic", "StatVarPeerGroup"})


class ArcsReader(Protocol):
    """Reads each dcid's `->*` arcs. An absent node maps to `None`."""

    def node_arcs_batch(self, dcids: list[str]) -> Mapping[str, dict | None]: ...


def is_container_child(dcid: str, types: list[str] | None) -> bool:
    """True when an arc child node is itself a container to descend into.

    Prefers the inline `types` the graph returns on each arc child. It falls
    back to the "/topic/" and "/svpg/" dcid substrings when type info is
    absent (offline fixtures that only set "dcid").
    """
    if types:
        return any(t in _CONTAINER_TYPES for t in types)
    return "/topic/" in dcid or "/svpg/" in dcid


def walk_topics(
    root_dcids: list[str],
    *,
    graph: ArcsReader,
    max_depth: int = TOPIC_EXPAND_MAX_DEPTH,
    max_members: int | None = TOPIC_EXPAND_MAX_MEMBERS,
    admit: Callable[[str], bool] | None = None,
) -> dict[str, list[str]]:
    """Breadth-first walk from each root to its leaf variables through
    `relevantVariable` and `member` arcs, batching each level across roots.

    Arc reads stay at `max_depth` whatever the root count, and duplicate
    frontier dcids share a read. Each root keeps its own visited set, leaf
    deduplication and `max_members` cap, and a root over the cap is dropped
    without affecting the others. `None` walks with no cap. With `admit`, a
    leaf it rejects is neither kept nor counted toward the cap. Containers
    are walked whatever `admit` says.

    Returns each surviving root's leaves in discovery order, roots in input
    order. A dropped root is absent and a root with no leaves maps to `[]`.
    """
    root_order = list(dict.fromkeys(root_dcids))
    visited: dict[str, set[str]] = {root: {root} for root in root_order}
    frontier: dict[str, list[str]] = {root: [root] for root in root_order}
    seen_leaves: dict[str, set[str]] = {root: set() for root in root_order}
    leaves: dict[str, list[str]] = {root: [] for root in root_order}
    declined: set[str] = set()

    for _ in range(max_depth):
        active_roots = [r for r in root_order if r not in declined and frontier[r]]
        if not active_roots:
            break
        merged_frontier = list(dict.fromkeys(dcid for r in active_roots for dcid in frontier[r]))
        arcs_by_dcid = graph.node_arcs_batch(merged_frontier)
        for r in active_roots:
            next_frontier: list[str] = []
            if _walk_level(
                frontier[r],
                arcs_by_dcid,
                visited=visited[r],
                seen_leaves=seen_leaves[r],
                leaves=leaves[r],
                next_frontier=next_frontier,
                max_members=max_members,
                admit=admit,
            ):
                declined.add(r)
                frontier[r] = []
            else:
                frontier[r] = next_frontier

    return {r: leaves[r] for r in root_order if r not in declined}


def _walk_level(
    frontier: list[str],
    arcs_by_dcid: Mapping[str, dict | None],
    *,
    visited: set[str],
    seen_leaves: set[str],
    leaves: list[str],
    next_frontier: list[str],
    max_members: int | None,
    admit: Callable[[str], bool] | None,
) -> bool:
    """Reads one root's frontier level into `leaves` and `next_frontier`.
    Returns whether the root passed `max_members` and is declined."""
    for dcid in frontier:
        arcs = arcs_by_dcid.get(dcid)
        if arcs is None:
            continue
        for arc_name in _MEMBER_ARCS:
            for node in arcs.get(arc_name, {}).get("nodes", []):
                child = node.get("dcid")
                if child is None or child in visited or child in seen_leaves:
                    continue
                if is_container_child(child, node.get("types")):
                    visited.add(child)
                    next_frontier.append(child)
                    continue
                seen_leaves.add(child)
                if admit is not None and not admit(child):
                    continue
                leaves.append(child)
                if max_members is not None and len(leaves) > max_members:
                    return True
    return False


__all__ = [
    "TOPIC_EXPAND_MAX_DEPTH",
    "TOPIC_EXPAND_MAX_MEMBERS",
    "ArcsReader",
    "is_container_child",
    "walk_topics",
]
