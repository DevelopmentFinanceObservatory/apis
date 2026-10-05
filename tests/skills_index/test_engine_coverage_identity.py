"""Tests coverage handling with the enum returned by the engine's real graph client.

Cross-enum tests catch identity comparisons. Complete and incomplete pages cover both outcomes.
"""

from __future__ import annotations

from search.engine.clients.graph import SubjectsCoverage as EngineSubjectsCoverage
from skills_index.build import build_index
from skills_index.types import TupleMemberAttachment, UnconfirmedAttachment

from .conftest import FakeSkillsGraph, skill_node
from .test_build import _roster


def test_roster_page_from_the_engine_enum_reporting_complete_raises_no_truncation_issue():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ()}},
        coverage_cls=EngineSubjectsCoverage,
    )

    index = build_index(graph=graph)

    assert not any("roster read truncated" in issue.message for issue in index.issues)


def test_budget_exhausted_breadth_page_from_the_engine_enum_is_acted_on():
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("Person", "populationType"): ["sv-1"]},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
        budget_exhausted={("Person", "populationType")},
        coverage_cls=EngineSubjectsCoverage,
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.statvar_count_truncated is True
    assert any(
        issue.severity == "warning" and "page budget" in issue.message
        for issue in index.issues
        if issue.skill_id == "skill-a"
    )


def test_capped_breadth_page_from_the_engine_enum_is_acted_on():
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("Person", "populationType"): ["sv-1"]},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
        capped={("Person", "populationType")},
        coverage_cls=EngineSubjectsCoverage,
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.statvar_count_truncated is True


def test_complete_breadth_page_from_the_engine_enum_is_not_truncated():
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("Person", "populationType"): ["sv-1"]},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
        coverage_cls=EngineSubjectsCoverage,
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.statvar_count_truncated is False


def test_budget_exhausted_probe_from_the_engine_enum_leaves_the_kind_unconfirmed():
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a"),
            ("ambiguous-target", "populationType"): ["sv-1"],
        },
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("ambiguous-target",)}},
        budget_exhausted={("ambiguous-target", "measuredProperty")},
        coverage_cls=EngineSubjectsCoverage,
    )

    index = build_index(graph=graph)

    kind = index.skills["skill-a"].attachments[0].kind
    assert isinstance(kind, UnconfirmedAttachment)
    assert not isinstance(kind, TupleMemberAttachment)
