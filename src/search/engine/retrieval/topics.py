"""Identify Topic candidates and expand them to member StatVars."""

from __future__ import annotations

from origin_catalogue import current_catalogue
from origin_catalogue.topic_walk import walk_topics
from search.engine.clients.graph import EngineGraphClient
from search.engine.config import settings


def is_topic_candidate(sv_dcid: str, sv_type: str | None) -> bool:
    """True when sv_dcid is a Topic node.

    Prefers the authoritative typeOf value; falls back to the "/topic/"
    dcid-substring heuristic when sv_type is None (offline fixtures/fallback
    paths with no recall-side type info).
    """
    if sv_type is not None:
        return sv_type == "Topic"
    return "/topic/" in sv_dcid


def expand_topics(
    root_dcids: list[str],
    *,
    graph: EngineGraphClient,
    max_depth: int = settings().topic_expand_max_depth,
    max_members: int = settings().topic_expand_max_members,
) -> dict[str, str]:
    """Expand roots into admitted leaf StatVars with `walk_topics`, which the
    origin catalogue job also runs to build its topic table.

    SDG indicator topics reach leaves through three container hops. Batched arc
    reads keep graph round trips at ``max_depth`` regardless of root count. A
    leaf passes the bound catalogue's candidate admission, as a recalled
    candidate does, before it counts toward its root's ``max_members`` cap. An
    oversized root is dropped without affecting others, and a root that keeps
    no leaf contributes none. Shared containers are attributed to each root
    that reaches them.

    Returns ``{leaf_dcid: root_dcid}``; the first root in ``root_dcids`` wins
    when roots share a leaf.
    """
    catalogue = current_catalogue()
    leaf_to_root: dict[str, str] = {}
    for root, leaves in walk_topics(
        root_dcids,
        graph=graph,
        max_depth=max_depth,
        max_members=max_members,
        admit=lambda dcid: catalogue.admit_candidate(dcid) is not None,
    ).items():
        # DCIDs encode added constraints as ``__`` segments; fewer segments
        # puts general members first when the bind-aware tie-break scores tie.
        for leaf in sorted(leaves, key=lambda dcid: dcid.count("__")):
            leaf_to_root.setdefault(leaf, root)
    return leaf_to_root
