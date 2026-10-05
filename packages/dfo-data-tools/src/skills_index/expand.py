"""Expand skill attachments breadth-first through graph hierarchy arcs.

Depth or node-limit breaches exclude the whole skill, never a partial reach.
"""

from __future__ import annotations

from collections.abc import Sequence

from skills_index.config import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_NODES,
    DEFAULT_MAX_RESULTS_PER_LEAD,
    EXCLUDED_ROOTS,
    HIERARCHY_PROPERTIES,
)
from skills_index.graph import SkillsGraph, SubjectsCoverage
from skills_index.types import ReachedNode, SkillIssue, SkillReach


def expand_skill(
    skill_id: str,
    attachment_dcids: Sequence[str],
    *,
    graph: SkillsGraph,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_results_per_lead: int = DEFAULT_MAX_RESULTS_PER_LEAD,
) -> tuple[SkillReach, tuple[SkillIssue, ...]]:
    """Expand one skill's attachments and return its reach and issues.

    Duplicate dcids appear once. Sorted hierarchy properties determine the first
    `via_property`.

    `max_nodes` includes attachments. `max_depth` probes one level past its limit.
    Either breach returns an empty reach with a rejection issue. Capped pages warn
    with the lead and property and mark the reach `capped`, but keep the skill. Budget-exhausted pages exclude
    the skill because their dcids are only a lower bound.
    """
    issues: list[SkillIssue] = []
    capped = False
    seen: set[str] = set()
    nodes: list[ReachedNode] = []

    frontier: list[str] = []
    for dcid in attachment_dcids:
        if dcid in EXCLUDED_ROOTS or dcid in seen:
            continue
        seen.add(dcid)
        frontier.append(dcid)
        nodes.append(ReachedNode(dcid=dcid, hop=0, via_property=None))

    excluded = _breaches_node_bound(nodes, max_nodes, skill_id, issues)

    hop = 0
    while not excluded and frontier and hop < max_depth + 1:
        hop += 1
        next_frontier: list[str] = []
        for hierarchy_property in sorted(HIERARCHY_PROPERTIES):
            pages = graph.node_subjects_many(
                frontier, property=hierarchy_property, max_results=max_results_per_lead
            )
            for lead, page in pages.items():
                if page.coverage == SubjectsCoverage.budget_exhausted:
                    issues.append(
                        SkillIssue(
                            skill_id=skill_id,
                            code="skill_build_unconfirmed",
                            message=(
                                f"{lead}'s <-{hierarchy_property} read stopped at its page "
                                "budget before completing, so the skill is excluded from "
                                "matching on this build"
                            ),
                        )
                    )
                    excluded = True
                    continue
                if page.coverage == SubjectsCoverage.capped:
                    capped = True
                    issues.append(
                        SkillIssue(
                            skill_id=skill_id,
                            code="skill_reach_read_truncated",
                            message=(
                                f"{lead}'s <-{hierarchy_property} read truncated at "
                                f"{max_results_per_lead} results"
                            ),
                        )
                    )
                for child in page.dcids:
                    if child in EXCLUDED_ROOTS or child in seen:
                        continue
                    seen.add(child)
                    next_frontier.append(child)
                    nodes.append(ReachedNode(dcid=child, hop=hop, via_property=hierarchy_property))
            if excluded:
                break

        if excluded:
            break
        if _breaches_node_bound(nodes, max_nodes, skill_id, issues):
            excluded = True
            break
        if hop == max_depth + 1 and next_frontier:
            issues.append(
                SkillIssue(
                    skill_id=skill_id,
                    code="skill_reach_too_deep",
                    message=f"reach exceeds the {max_depth}-level depth bound",
                )
            )
            excluded = True
            break
        frontier = next_frontier

    if excluded:
        return SkillReach(skill_id=skill_id, nodes=(), excluded=True), tuple(issues)
    return (
        SkillReach(skill_id=skill_id, nodes=tuple(nodes), excluded=False, capped=capped),
        tuple(issues),
    )


def _breaches_node_bound(
    nodes: list[ReachedNode], max_nodes: int, skill_id: str, issues: list[SkillIssue]
) -> bool:
    if len(nodes) <= max_nodes:
        return False
    issues.append(
        SkillIssue(
            skill_id=skill_id,
            code="skill_reach_too_large",
            message=f"reach exceeds the {max_nodes}-node bound",
        )
    )
    return True
