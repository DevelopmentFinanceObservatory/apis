"""`expand_skill` against a `FakeSkillsGraph`: descent depth and hop
distances, which hierarchy property carried each step, excluded roots,
the depth and node bounds, a property attachment's dead end, a capped
page surfacing as a warning rather than being swallowed, and a
budget-exhausted page excluding the whole skill.
"""

from __future__ import annotations

from skills_index.expand import expand_skill
from skills_index.types import ReachedNode

from .conftest import FakeSkillsGraph


def _nodes_by_dcid(nodes: tuple[ReachedNode, ...]) -> dict[str, ReachedNode]:
    return {node.dcid: node for node in nodes}


def test_three_level_descent_reaches_every_level():
    graph = FakeSkillsGraph(
        subjects={
            ("A", "specializationOf"): ["B"],
            ("B", "specializationOf"): ["C"],
            ("C", "specializationOf"): ["D"],
        }
    )

    reach, issues = expand_skill("skill-1", ["A"], graph=graph)

    assert issues == ()
    assert reach.excluded is False
    assert {node.dcid for node in reach.nodes} == {"A", "B", "C", "D"}


def test_hierarchy_property_recorded_per_step():
    graph = FakeSkillsGraph(
        subjects={
            ("A", "specializationOf"): ["B"],
            ("B", "specializationOf"): ["C"],
        }
    )

    reach, _issues = expand_skill("skill-1", ["A"], graph=graph)

    by_dcid = _nodes_by_dcid(reach.nodes)
    assert by_dcid["A"].via_property is None
    assert by_dcid["B"].via_property == "specializationOf"
    assert by_dcid["C"].via_property == "specializationOf"


def test_hop_distances_count_steps_from_the_attachment():
    graph = FakeSkillsGraph(
        subjects={
            ("A", "specializationOf"): ["B"],
            ("B", "specializationOf"): ["C"],
        }
    )

    reach, _issues = expand_skill("skill-1", ["A"], graph=graph)

    by_dcid = _nodes_by_dcid(reach.nodes)
    assert by_dcid["A"].hop == 0
    assert by_dcid["B"].hop == 1
    assert by_dcid["C"].hop == 2


def test_mixed_hierarchy_descends_by_different_properties_per_level():
    graph = FakeSkillsGraph(
        subjects={
            ("A", "specializationOf"): ["B"],
            ("B", "subClassOf"): ["C"],
        }
    )

    reach, _issues = expand_skill("skill-1", ["A"], graph=graph)

    by_dcid = _nodes_by_dcid(reach.nodes)
    assert by_dcid["B"].via_property == "specializationOf"
    assert by_dcid["C"].via_property == "subClassOf"


def test_excluded_root_is_dropped_and_never_expanded_through():
    graph = FakeSkillsGraph(
        subjects={
            ("A", "specializationOf"): ["Thing"],
            # If the walk expanded through "Thing" it would reach "Z" too.
            ("Thing", "specializationOf"): ["Z"],
        }
    )

    reach, _issues = expand_skill("skill-1", ["A"], graph=graph)

    dcids = {node.dcid for node in reach.nodes}
    assert "Thing" not in dcids
    assert "Z" not in dcids
    assert dcids == {"A"}


def test_depth_bound_excludes_the_whole_skill():
    graph = FakeSkillsGraph(
        subjects={
            ("A", "specializationOf"): ["B"],
            ("B", "specializationOf"): ["C"],
        }
    )

    reach, issues = expand_skill("skill-1", ["A"], graph=graph, max_depth=1)

    assert reach.excluded is True
    assert reach.nodes == ()
    assert any(issue.severity == "rejection" and "depth" in issue.message for issue in issues)


def test_node_bound_excludes_the_whole_skill():
    graph = FakeSkillsGraph(
        subjects={
            ("A", "specializationOf"): ["B", "C", "D"],
        }
    )

    reach, issues = expand_skill("skill-1", ["A"], graph=graph, max_nodes=2)

    assert reach.excluded is True
    assert reach.nodes == ()
    assert any(issue.severity == "rejection" and "node" in issue.message for issue in issues)


def test_property_attachment_expands_no_further():
    graph = FakeSkillsGraph(subjects={})

    reach, issues = expand_skill("skill-1", ["crsPurpose"], graph=graph)

    assert issues == ()
    assert reach.excluded is False
    assert reach.nodes == (ReachedNode(dcid="crsPurpose", hop=0, via_property=None),)


def test_capped_page_surfaces_as_a_warning():
    graph = FakeSkillsGraph(
        subjects={("A", "specializationOf"): ["B"]},
        capped={("A", "specializationOf")},
    )

    reach, issues = expand_skill("skill-1", ["A"], graph=graph)

    assert reach.excluded is False
    assert reach.capped is True
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "A" in issues[0].message
    assert "specializationOf" in issues[0].message


def test_budget_exhausted_page_excludes_the_whole_skill():
    graph = FakeSkillsGraph(
        subjects={("A", "specializationOf"): ["B"]},
        budget_exhausted={("A", "specializationOf")},
    )

    reach, issues = expand_skill("skill-1", ["A"], graph=graph)

    assert reach.excluded is True
    assert reach.nodes == ()
    assert any(
        issue.code == "skill_build_unconfirmed"
        and "A" in issue.message
        and "specializationOf" in issue.message
        for issue in issues
    )
