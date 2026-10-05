"""Tests index construction, attachment kinds, graph limits, and content handling."""

from __future__ import annotations

import json

import pytest

import skills_index.build as build_module
import skills_index.frontmatter as frontmatter_module
from skills_index.build import build_index
from skills_index.config import (
    HIERARCHY_PROPERTIES,
    REFERENCE_TYPE,
    SKILL_BODY_MAX_BYTES,
    SKILL_REFERENCE_MAX_BYTES,
    SKILL_REFERENCES_MAX_ENTRIES,
    SKILLS_APPLIES_TO_MAX_TARGETS,
    SKILLS_BREADTH_COUNT_MAX_RESULTS,
    SKILLS_BREADTH_WARN_STATVARS,
    SKILLS_ROSTER_MAX_RESULTS,
)
from skills_index.types import (
    ConstraintPropertyAttachment,
    ConstraintValueAttachment,
    ProvenanceAttachment,
    ReachedNode,
    RejectedAttachment,
    SkillFile,
    TupleMemberAttachment,
    UnconfirmedAttachment,
)

from .conftest import FakeSkillsGraph, skill_markdown, skill_node, stored


def _roster_dcids(*dcids: str) -> dict[tuple[str, str], list[str]]:
    return {("AgentSkill", "typeOf"): list(dcids)}


def _roster(*skill_ids: str) -> dict[tuple[str, str], list[str]]:
    return _roster_dcids(*(f"skill/{skill_id}" for skill_id in skill_ids))


def test_roster_reads_every_agentskill_subject_and_its_front_matter_from_text():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b"),
        arcs={
            "skill/skill-a": skill_node("skill-a", description="Does A things.")
            | {"name": ("Graph Name",), "description": ("Graph description.",)},
            "skill/skill-b": skill_node("skill-b", description="Does B things."),
        },
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-a", "skill-b"}
    entry = index.skills["skill-a"]
    assert entry.front_matter.name == "skill-a"
    assert entry.front_matter.description == "Does A things."
    assert entry.frontmatter == {"name": "skill-a", "description": "Does A things."}
    assert entry.content_text == skill_markdown("skill-a", description="Does A things.")
    assert all(
        "name" not in properties and "description" not in properties
        for requested, properties in graph.arc_calls
        if "skill/skill-a" in requested
    )


def test_frontmatter_keeps_every_key_with_every_scalar_a_string():
    text = (
        "---\nname: skill-a\ndescription: Does things.\nversion: 1.10\nlicense: MIT\n"
        "metadata:\n  count: 3\n  tags: [a, 2]\n---\nBody.\n"
    )
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"), arcs={"skill/skill-a": {"text": (stored(text),)}}
    )

    index = build_index(graph=graph)

    assert index.skills["skill-a"].frontmatter == {
        "name": "skill-a",
        "description": "Does things.",
        "version": "1.10",
        "license": "MIT",
        "metadata": {"count": "3", "tags": ["a", "2"]},
    }


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Body without front matter.\n", "does not open"),
        ("---\nname: skill-a\ndescription: Does things.\n", "does not open"),
        ("\n---\nname: skill-a\ndescription: Does things.\n---\n", "does not open"),
        ("---\nname: [skill-a\n---\n", "does not parse"),
        ("---\n- skill-a\n---\n", "not a mapping"),
        ("---\n---\nBody.\n", "not a mapping"),
        ("---\ndescription: Does things.\n---\n", "no string 'name'"),
        ("---\nname: [skill-a]\ndescription: Does things.\n---\n", "no string 'name'"),
        ("---\nname: skill-a\n---\n", "no string 'description'"),
        ("---\nname: skill-a\ndescription:\n  a: b\n---\n", "no string 'description'"),
        ("---\nname: skill-b\ndescription: Does things.\n---\n", "differs from the skill id"),
        (
            "---\nname: skill-a\ndescription: &d Does things.\nsummary: *d\n---\n",
            "anchor or alias",
        ),
        ("---\nname: skill-a\ndescription: &d Does things.\n---\n", "anchor or alias"),
        # About 4 KB of nested flow sequences, deep enough to exhaust the
        # interpreter's recursion limit if values were built from it.
        pytest.param(
            f"---\nname: skill-a\ndescription: Does things.\nx: {'[' * 2000}{']' * 2000}\n---\n",
            "nests deeper than",
            id="sequences-nested-2000-deep",
        ),
        pytest.param(
            f"---\nname: skill-a\ndescription: Does things.\nx: {'{a: ' * 40}b{'}' * 40}\n---\n",
            "nests deeper than",
            id="mappings-nested-past-the-bound",
        ),
        # A YAML escape yields a lone surrogate, which no JSON response can carry.
        pytest.param(
            '---\nname: skill-a\ndescription: "\\ud800"\n---\n',
            "Skills API wire model",
            id="lone-surrogate",
        ),
    ],
)
def test_skill_with_invalid_front_matter_is_left_out_with_a_corpus_warning(text, reason):
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-c"),
        arcs={
            "skill/skill-a": {"text": (stored(text),), "skillAppliesTo": ("Person",)},
            "skill/skill-c": skill_node("skill-c"),
        },
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-c"}
    assert [(issue.code, issue.skill_id) for issue in index.issues] == [
        ("skill_frontmatter_invalid", "skill/skill-a")
    ]
    assert reason in index.issues[0].message
    assert "skill/skill-a" in index.issues[0].message
    assert index.may_omit_published_skills is False
    assert index.node_to_skills == {}


def test_front_matter_nested_at_the_depth_bound_is_admitted():
    depth = frontmatter_module._MAX_NESTING_DEPTH
    # The root mapping is one level, so `depth - 1` sequences reach the bound.
    nested = "[" * (depth - 1) + "]" * (depth - 1)
    text = f"---\nname: skill-a\ndescription: Does things.\nx: {nested}\n---\n"
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"), arcs={"skill/skill-a": {"text": (stored(text),)}}
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-a"}


def test_a_parser_failure_outside_yaml_errors_still_rejects_only_that_skill(monkeypatch):
    # With the depth bound lifted, composing the nested value exhausts the
    # recursion limit, and that failure still leaves only skill-a out.
    monkeypatch.setattr(frontmatter_module, "_MAX_NESTING_DEPTH", 10**6)
    text = f"---\nname: skill-a\ndescription: Does things.\nx: {'[' * 2000}{']' * 2000}\n---\n"
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-c"),
        arcs={"skill/skill-a": {"text": (stored(text),)}, "skill/skill-c": skill_node("skill-c")},
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-c"}
    assert [(issue.code, issue.skill_id) for issue in index.issues] == [
        ("skill_frontmatter_invalid", "skill/skill-a")
    ]


def test_front_matter_closing_line_may_carry_trailing_blanks_and_crlf():
    text = "---\r\nname: skill-a\r\ndescription: Does things.\r\n--- \r\nBody.\r\n"
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"), arcs={"skill/skill-a": {"text": (stored(text),)}}
    )

    index = build_index(graph=graph)

    assert index.skills["skill-a"].front_matter.description == "Does things."


def test_url_is_read_from_the_skill_node():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b"),
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"url": ("https://example.org/skill-a",)},
            "skill/skill-b": skill_node("skill-b"),
        },
    )

    index = build_index(graph=graph)

    assert index.skills["skill-a"].url == "https://example.org/skill-a"
    assert index.skills["skill-b"].url is None


def test_slash_bearing_skill_dcid_is_rejected_and_absent_from_the_catalogue():
    graph = FakeSkillsGraph(
        subjects=_roster_dcids("dc/skill/donor-coordination", "skill/skill-b"),
        arcs={"skill/skill-b": skill_node("skill-b")},
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-b"}
    assert any(
        issue.severity == "rejection"
        and issue.skill_id == "dc/skill/donor-coordination"
        and "dc/skill/donor-coordination" in issue.message
        for issue in index.issues
    )


def test_reserved_platform_name_skill_dcid_is_rejected_and_absent_from_the_catalogue():
    graph = FakeSkillsGraph(
        subjects=_roster_dcids("skill/healthz", "skill/skill-b"),
        arcs={"skill/skill-b": skill_node("skill-b")},
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-b"}
    assert any(
        issue.severity == "rejection"
        and issue.code == "skills_roster_entry_unaddressable"
        and issue.skill_id == "skill/healthz"
        for issue in index.issues
    )


@pytest.mark.parametrize(
    "dcid",
    [
        "skill-b",
        "Skill/skill-b",
        "skill/Skill-B",
        "skill/skill_b",
        "skill/skill--b",
        "skill/-skill-b",
        "skill/skill-b/",
        "skill/" + "a" * 65,
    ],
)
def test_dcid_that_is_not_skill_plus_a_valid_id_is_unaddressable(dcid):
    graph = FakeSkillsGraph(
        subjects=_roster_dcids(dcid, "skill/skill-a"),
        arcs={"skill/skill-a": skill_node("skill-a"), dcid: {"text": ("Other.",)}},
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-a"}
    assert [
        (issue.code, issue.severity, issue.skill_id)
        for issue in index.issues
        if issue.code == "skills_roster_entry_unaddressable"
    ] == [("skills_roster_entry_unaddressable", "rejection", dcid)]
    assert all(dcid not in requested for requested, _ in graph.arc_calls)


def test_skill_id_at_the_grammar_length_limit_is_admitted():
    skill_id = "a" * 64
    graph = FakeSkillsGraph(
        subjects=_roster(skill_id), arcs={f"skill/{skill_id}": skill_node(skill_id)}
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {skill_id}
    assert index.skills[skill_id].front_matter.id == skill_id


def test_slug_shaped_skill_dcid_is_admitted():
    graph = FakeSkillsGraph(
        subjects=_roster("donor-coordination"),
        arcs={"skill/donor-coordination": skill_node("donor-coordination")},
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"donor-coordination"}
    assert not any(issue.skill_id == "donor-coordination" for issue in index.issues)


def test_skill_with_no_skillappliesto_has_empty_reach_and_matches_nothing():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"), arcs={"skill/skill-a": skill_node("skill-a")}
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.attachments == ()
    assert entry.reach.nodes == ()
    assert entry.reach.excluded is False
    assert entry.statvar_count == 0
    assert index.node_to_skills == {}


def test_target_absent_from_graph_rejects_cleanly():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("healthCareFunction",)}},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert len(entry.attachments) == 1
    assert isinstance(entry.attachments[0].kind, RejectedAttachment)
    assert entry.reach.nodes == ()
    assert any(
        issue.severity == "rejection" and "healthCareFunction" in issue.message
        for issue in index.issues
    )


def test_classifies_via_population_type_probe():
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("Person", "populationType"): ["sv-1"]},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
    )

    index = build_index(graph=graph)

    kind = index.skills["skill-a"].attachments[0].kind
    assert kind == TupleMemberAttachment(tuple_property="populationType")


def test_classifies_via_measured_property_probe():
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("count", "measuredProperty"): ["sv-1"]},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("count",)}},
    )

    index = build_index(graph=graph)

    kind = index.skills["skill-a"].attachments[0].kind
    assert kind == TupleMemberAttachment(tuple_property="measuredProperty")


def test_classifies_via_stat_type_probe():
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("measuredValue", "statType"): ["sv-1"]},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("measuredValue",)}},
    )

    index = build_index(graph=graph)

    kind = index.skills["skill-a"].attachments[0].kind
    assert kind == TupleMemberAttachment(tuple_property="statType")


def test_classifies_via_measurement_qualifier_probe():
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("Nominal", "measurementQualifier"): ["sv-1"]},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Nominal",)}},
    )

    index = build_index(graph=graph)

    kind = index.skills["skill-a"].attachments[0].kind
    assert kind == TupleMemberAttachment(tuple_property="measurementQualifier")


def test_classifies_via_measurement_denominator_probe():
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("PerCapita", "measurementDenominator"): ["sv-1"]},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("PerCapita",)}},
    )

    index = build_index(graph=graph)

    kind = index.skills["skill-a"].attachments[0].kind
    assert kind == TupleMemberAttachment(tuple_property="measurementDenominator")


def test_classifies_via_constraint_properties_probe():
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("crsPurpose", "constraintProperties"): ["SomeClass"]},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("crsPurpose",)}},
    )

    index = build_index(graph=graph)

    kind = index.skills["skill-a"].attachments[0].kind
    assert kind == ConstraintPropertyAttachment()


def test_classifies_via_type_then_range_walk():
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a"),
            ("CRSPurposeCodeEnum", "rangeIncludes"): ["crsPurpose"],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("CRSPurposeCode120",)},
            "CRSPurposeCode120": {"typeOf": ("CRSPurposeCodeEnum",)},
        },
    )

    index = build_index(graph=graph)

    kind = index.skills["skill-a"].attachments[0].kind
    assert kind == ConstraintValueAttachment(property="crsPurpose")


def test_statvar_count_deduplicates_across_reached_nodes():
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a"),
            ("Person", "populationType"): ["sv-1"],
            ("Person", "specializationOf"): ["Adult"],
            ("Adult", "populationType"): ["sv-1", "sv-2"],
        },
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert {node.dcid for node in entry.reach.nodes} == {"Person", "Adult"}
    assert entry.statvar_count == 2
    assert entry.statvar_count_truncated is False


def test_statvar_count_stops_and_warns_past_the_threshold():
    many_statvars = [f"sv-{i}" for i in range(SKILLS_BREADTH_COUNT_MAX_RESULTS)]
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("Person", "populationType"): many_statvars},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.statvar_count == SKILLS_BREADTH_COUNT_MAX_RESULTS
    assert entry.statvar_count_truncated is True
    assert any(
        issue.severity == "warning" and str(SKILLS_BREADTH_WARN_STATVARS) in issue.message
        for issue in index.issues
        if issue.skill_id == "skill-a"
    )


def test_statvar_count_under_the_threshold_does_not_warn():
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("Person", "populationType"): ["sv-1", "sv-2"]},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.statvar_count == 2
    assert entry.statvar_count_truncated is False
    # The fixture also raises `skill_content_absent`. This isolates the breadth warning.
    assert not any(
        issue.code == "skill_breadth_high" for issue in index.issues if issue.skill_id == "skill-a"
    )


def test_node_to_skills_inverts_across_multiple_skills():
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a", "skill-b"),
            ("Person", "populationType"): ["sv-1"],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)},
            "skill/skill-b": skill_node("skill-b") | {"skillAppliesTo": ("Person",)},
        },
    )

    index = build_index(graph=graph)

    matches = index.node_to_skills["Person"]
    assert {match.skill_id for match in matches} == {"skill-a", "skill-b"}
    assert all(match.hop == 0 for match in matches)


def test_budget_exhausted_breadth_page_truncates_below_the_warn_threshold():
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("Person", "populationType"): ["sv-1"]},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
        budget_exhausted={("Person", "populationType")},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.statvar_count == 1
    assert entry.statvar_count_truncated is True
    # A count floor omits nothing search can match.
    assert index.may_omit_published_skills is False
    assert any(
        issue.severity == "warning" and "page budget" in issue.message
        for issue in index.issues
        if issue.skill_id == "skill-a"
    )


def test_budget_exhausted_classification_probe_records_an_issue_not_absence():
    # An exhausted empty page leaves the kind uncertain because another accepted arc may exist.
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("mystery-target",)}},
        budget_exhausted={("mystery-target", "populationType")},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert isinstance(entry.attachments[0].kind, UnconfirmedAttachment)
    assert entry.reach.nodes == ()
    assert entry.statvar_count == 0
    assert any(
        issue.severity == "warning"
        and "mystery-target" in issue.message
        and "stopped at its page budget before settling its attachment kind" in issue.message
        for issue in index.issues
        if issue.skill_id == "skill-a"
    )
    assert index.may_omit_published_skills is True


def test_constraint_property_attachment_counts_through_admissible_values():
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a"),
            ("gender", "constraintProperties"): ["SomeClass"],
            ("GenderType", "typeOf"): ["Male", "Female"],
            ("Male", "gender"): ["sv-1"],
            ("Female", "gender"): ["sv-2", "sv-3"],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("gender",)},
            "gender": {"rangeIncludes": ("GenderType",)},
        },
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.attachments[0].kind == ConstraintPropertyAttachment()
    assert entry.statvar_count == 3
    assert entry.statvar_count_truncated is False


def test_admissible_values_budget_exhausted_truncates_the_count():
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a"),
            ("gender", "constraintProperties"): ["SomeClass"],
            ("GenderType", "typeOf"): ["Male"],
            ("Male", "gender"): ["sv-1"],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("gender",)},
            "gender": {"rangeIncludes": ("GenderType",)},
        },
        budget_exhausted={("GenderType", "typeOf")},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.statvar_count == 1
    assert entry.statvar_count_truncated is True
    # A count floor omits nothing search can match.
    assert index.may_omit_published_skills is False
    assert any(
        issue.severity == "warning" and "page budget" in issue.message
        for issue in index.issues
        if issue.skill_id == "skill-a"
    )


def test_capped_breadth_page_truncates_the_count_too():
    # `capped` and `budget_exhausted` return partial pages. Only `complete` proves the count.
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("Person", "populationType"): ["sv-1"]},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
        capped={("Person", "populationType")},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.statvar_count == 1
    assert entry.statvar_count_truncated is True
    # A count floor omits nothing search can match.
    assert index.may_omit_published_skills is False


def test_capped_hierarchy_page_understates_reach_and_sets_the_flag():
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a"),
            ("Person", "populationType"): ["sv-1"],
            ("Person", "specializationOf"): ["Child"],
        },
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
        capped={("Person", "specializationOf")},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.attached is True
    assert entry.reach.capped is True
    assert [issue.code for issue in index.issues] == ["skill_reach_read_truncated"]
    assert index.may_omit_published_skills is True


def test_capped_admissible_values_page_truncates_the_count_too():
    # A `capped` `<-typeOf` page also gives only a lower bound on admissible values.
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a"),
            ("gender", "constraintProperties"): ["SomeClass"],
            ("GenderType", "typeOf"): ["Male"],
            ("Male", "gender"): ["sv-1"],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("gender",)},
            "gender": {"rangeIncludes": ("GenderType",)},
        },
        capped={("GenderType", "typeOf")},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.statvar_count == 1
    assert entry.statvar_count_truncated is True
    # A count floor omits nothing search can match.
    assert index.may_omit_published_skills is False


def test_range_walk_budget_exhausted_flags_kind_as_unconfirmed_even_when_resolved():
    # An exhausted `rangeIncludes` page may hide a property that makes the target ambiguous.
    # The attachment stays unconfirmed.
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a"),
            ("SomeType", "rangeIncludes"): ["crsPurpose"],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("mystery-target",)},
            "mystery-target": {"typeOf": ("SomeType",)},
        },
        budget_exhausted={("SomeType", "rangeIncludes")},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert isinstance(entry.attachments[0].kind, UnconfirmedAttachment)
    assert entry.reach.nodes == ()
    assert any(
        issue.severity == "warning"
        and "mystery-target" in issue.message
        and "stopped at its page budget before settling its attachment kind" in issue.message
        for issue in index.issues
        if issue.skill_id == "skill-a"
    )
    assert index.may_omit_published_skills is True


def test_roster_read_uses_its_own_cap_not_the_per_lead_one():
    graph = FakeSkillsGraph(subjects=_roster("skill-a"))

    build_index(graph=graph)

    assert graph.calls[0] == (("AgentSkill",), "typeOf", SKILLS_ROSTER_MAX_RESULTS)
    assert SKILLS_ROSTER_MAX_RESULTS != build_module.DEFAULT_MAX_RESULTS_PER_LEAD


def _markdown_of_bytes(skill_id: str, size: int) -> str:
    """Returns a valid `SKILL.md` for `skill_id` that is exactly `size` bytes."""
    padding = size - len(skill_markdown(skill_id, body="").encode("utf-8"))
    return skill_markdown(skill_id, body="x" * padding)


def test_text_past_the_byte_ceiling_is_left_out_with_a_corpus_warning():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b"),
        arcs={
            "skill/skill-a": {
                "text": (stored(_markdown_of_bytes("skill-a", SKILL_BODY_MAX_BYTES + 1)),)
            },
            "skill/skill-b": skill_node("skill-b"),
        },
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-b"}
    assert [(issue.code, issue.severity, issue.skill_id) for issue in index.issues] == [
        ("skill_content_too_large", "warning", "skill/skill-a")
    ]
    assert f"{SKILL_BODY_MAX_BYTES + 1} bytes" in index.issues[0].message


def test_text_at_the_byte_ceiling_is_served_in_full():
    exact_text = _markdown_of_bytes("skill-a", SKILL_BODY_MAX_BYTES)
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"), arcs={"skill/skill-a": {"text": (stored(exact_text),)}}
    )

    index = build_index(graph=graph)

    assert index.skills["skill-a"].content_text == exact_text
    assert index.issues == ()


def test_skill_node_without_text_is_left_out_with_a_corpus_warning():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b", "skill-c"),
        arcs={
            "skill/skill-a": {"url": ("https://example.org/skill-a",)},
            "skill/skill-c": skill_node("skill-c"),
        },
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-c"}
    assert [(issue.code, issue.skill_id) for issue in index.issues] == [
        ("skill_content_absent", "skill/skill-a"),
        ("skill_content_absent", "skill/skill-b"),
    ]


def test_a_failing_chunk_leaves_its_skills_unknown_and_marks_the_index_incomplete(monkeypatch):
    monkeypatch.setattr(build_module, "_FRONT_MATTER_CHUNK_SIZE", 1)
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b"),
        arcs={
            "skill/skill-a": skill_node("skill-a"),
            "skill/skill-b": skill_node("skill-b"),
        },
        failing_arc_dcids=frozenset({"skill/skill-a"}),
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-b"}
    assert [(issue.code, issue.skill_id) for issue in index.issues] == [
        ("skill_build_unconfirmed", "skill/skill-a")
    ]
    assert index.may_omit_published_skills is True
    assert len(graph.arc_calls) == 2


def test_every_chunk_failing_reraises_instead_of_returning_an_empty_catalogue(monkeypatch):
    monkeypatch.setattr(build_module, "_FRONT_MATTER_CHUNK_SIZE", 1)
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b"),
        arcs={
            "skill/skill-a": skill_node("skill-a"),
            "skill/skill-b": skill_node("skill-b"),
        },
        failing_arc_dcids=frozenset({"skill/skill-a", "skill/skill-b"}),
    )

    with pytest.raises(RuntimeError):
        build_index(graph=graph)


def test_single_chunk_corpus_whose_only_chunk_fails_reraises():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b"),
        arcs={
            "skill/skill-a": skill_node("skill-a"),
            "skill/skill-b": skill_node("skill-b"),
        },
        failing_arc_dcids=frozenset({"skill/skill-a"}),
    )

    with pytest.raises(RuntimeError):
        build_index(graph=graph)


def test_incomplete_probe_evidence_does_not_confirm_a_single_tuple_property():
    # An exhausted measuredProperty page leaves the target ambiguous.
    # Unconfirmed attachments contribute no reach.
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a"),
            ("ambiguous-target", "populationType"): ["sv-1"],
        },
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("ambiguous-target",)}},
        budget_exhausted={("ambiguous-target", "measuredProperty")},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    kind = entry.attachments[0].kind
    assert isinstance(kind, UnconfirmedAttachment)
    assert not isinstance(kind, TupleMemberAttachment)
    assert entry.reach.nodes == ()
    assert entry.reach.excluded is False
    assert entry.statvar_count == 0
    assert index.node_to_skills == {}
    assert any(
        issue.severity == "warning" and "ambiguous-target" in issue.message
        for issue in index.issues
        if issue.skill_id == "skill-a"
    )


@pytest.mark.parametrize(
    "provenance_count",
    [
        pytest.param(0, id="all-on-skillAppliesTo"),
        pytest.param(1, id="one-on-skillAppliesToProvenance"),
    ],
)
def test_skillappliesto_past_the_target_cap_rejects_the_whole_skill(provenance_count):
    # The bound counts targets across both attachment arcs.
    too_many_targets = tuple(
        f"target-{i}" for i in range(SKILLS_APPLIES_TO_MAX_TARGETS + 1 - provenance_count)
    )
    provenance_targets = tuple(f"provenance/P{i}" for i in range(provenance_count))
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={
            "skill/skill-a": skill_node("skill-a")
            | {"skillAppliesTo": too_many_targets, "skillAppliesToProvenance": provenance_targets}
        },
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.attachments == ()
    assert entry.reach.excluded is True
    assert entry.statvar_count == 0
    assert any(
        issue.severity == "rejection" and "skillAppliesTo" in issue.message
        for issue in index.issues
        if issue.skill_id == "skill-a"
    )
    # The cap applies before graph reads for excess targets.
    assert not any(call[0] == too_many_targets for call in graph.calls)


def test_node_budget_applies_once_across_every_counting_property_group():
    # Each attachment reaches one node, under max_nodes=1, but their shared total is 2.
    # A per-group check would admit both. The skill-wide bound excludes the skill.
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a"),
            ("TypeA", "rangeIncludes"): ["prop-a"],
            ("TypeB", "rangeIncludes"): ["prop-b"],
            ("v-a-1", "prop-a"): ["sv-1"],
            ("v-a-2", "prop-a"): ["sv-2"],
            ("v-a-3", "prop-a"): ["sv-3"],
            ("v-b-1", "prop-b"): ["sv-4"],
            ("v-b-2", "prop-b"): ["sv-5"],
            ("v-b-3", "prop-b"): ["sv-6"],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("value-a", "value-b")},
            "value-a": {"typeOf": ("TypeA",)},
            "value-b": {"typeOf": ("TypeB",)},
        },
    )
    index = build_index(graph=graph, max_nodes=1)

    entry = index.skills["skill-a"]
    assert entry.reach.excluded is True
    assert entry.reach.nodes == ()
    assert any(
        issue.severity == "rejection" and "node bound" in issue.message
        for issue in index.issues
        if issue.skill_id == "skill-a"
    )


def test_unsafe_constraint_property_token_is_rejected_not_raised():
    # Reject unsafe property names before `_count_statvars` raises `ValueError`.
    unsafe_dcid = "bad:property"
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a"),
            (unsafe_dcid, "constraintProperties"): ["SomeClass"],
        },
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": (unsafe_dcid,)}},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert isinstance(entry.attachments[0].kind, RejectedAttachment)
    assert entry.reach.nodes == ()
    assert entry.statvar_count == 0
    assert any(
        issue.severity == "rejection" and unsafe_dcid in issue.message
        for issue in index.issues
        if issue.skill_id == "skill-a"
    )


def test_unsafe_constraint_value_property_token_is_rejected_not_raised():
    # A graph-derived `rangeIncludes` subject can also name an unsafe property.
    unsafe_property = "bad property"
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a"),
            ("SomeType", "rangeIncludes"): [unsafe_property],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("mystery-target",)},
            "mystery-target": {"typeOf": ("SomeType",)},
        },
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert isinstance(entry.attachments[0].kind, RejectedAttachment)
    assert entry.reach.nodes == ()


@pytest.mark.parametrize("unsafe_dcid", ["foo+", "foo*", ""])
def test_graph_derived_transitive_or_empty_property_token_is_rejected_not_raised(unsafe_dcid):
    # `+` and `*` broaden graph reads. An empty name makes a malformed expression.
    # Reject these tokens before a graph call.
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a"),
            (unsafe_dcid, "constraintProperties"): ["SomeClass"],
        },
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": (unsafe_dcid,)}},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert isinstance(entry.attachments[0].kind, RejectedAttachment)
    assert entry.reach.nodes == ()
    assert entry.statvar_count == 0
    assert any(
        issue.severity == "rejection" for issue in index.issues if issue.skill_id == "skill-a"
    )


def test_truncated_breadth_carries_through_to_node_to_skills_match():
    # `match_dcids` ranks with `breadth_truncated`, so it must carry the skill's truncation flag.
    many_statvars = [f"sv-{i}" for i in range(SKILLS_BREADTH_COUNT_MAX_RESULTS)]
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("Person", "populationType"): many_statvars},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
    )

    index = build_index(graph=graph)

    matches = index.node_to_skills["Person"]
    assert all(match.breadth_truncated is True for match in matches)


def test_front_matter_chunk_size_leaves_headroom_for_worst_case_json_escaping():
    # 128 max-size bodies exceed the mixer cap when NULs expand to six JSON bytes.
    # The configured chunk needs headroom.
    escape_heavy_body = "\x00" * SKILL_BODY_MAX_BYTES
    aggregate_response = json.dumps(
        {
            f"skill-{i}": {"text": (escape_heavy_body,)}
            for i in range(build_module._FRONT_MATTER_CHUNK_SIZE)
        }
    )

    assert len(aggregate_response.encode("utf-8")) < build_module._MIXER_RESPONSE_CAP_BYTES


def test_skills_after_the_corpus_budget_is_spent_are_left_out(monkeypatch):
    # With 5 bytes, skill-a's text spends the budget, and every later skill is left out.
    monkeypatch.setattr(build_module, "_FRONT_MATTER_CHUNK_SIZE", 1)
    monkeypatch.setattr(build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", 5)
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b", "skill-c"),
        arcs={
            "skill/skill-a": skill_node("skill-a"),
            "skill/skill-b": skill_node("skill-b") | {"skillAppliesTo": ("Person",)},
            "skill/skill-c": skill_node("skill-c"),
        },
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-a"}
    assert [(issue.code, issue.severity, issue.skill_id) for issue in index.issues] == [
        ("skills_corpus_budget_exhausted", "warning", "skill/skill-b"),
        ("skills_corpus_budget_exhausted", "warning", "skill/skill-c"),
    ]
    assert "corpus-wide budget" in index.issues[0].message
    assert index.may_omit_published_skills is True
    assert index.node_to_skills == {}
    # The budget limits retained bytes. The build reads all three one-skill chunks.
    assert len(graph.arc_calls) == 3


def test_roster_truncation_issue_carries_skill_id_none():
    graph = FakeSkillsGraph(
        subjects=_roster("who-immunization"), budget_exhausted={("AgentSkill", "typeOf")}
    )

    index = build_index(graph=graph)

    roster_issues = [issue for issue in index.issues if issue.code == "skills_roster_truncated"]
    assert len(roster_issues) == 1
    assert roster_issues[0].skill_id is None
    assert roster_issues[0].severity == "warning"


def test_reference_dcid_is_stripped_to_its_bundle_relative_path():
    graph = FakeSkillsGraph(
        subjects=_roster("who-immunization"),
        arcs={
            "skill/who-immunization": skill_node("who-immunization")
            | {
                "agentSkillReferences": ("skill/who-immunization/purpose-codes",),
            },
            "skill/who-immunization/purpose-codes": {
                "name": ("references/purpose-codes.md",),
                "text": ("a,b,c",),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    resources = index.skills["who-immunization"].files[1:]
    assert len(resources) == 1
    assert resources[0].path == "references/purpose-codes.md"
    assert resources[0].content == "a,b,c"


def test_resources_are_listed_in_deterministic_path_order():
    graph = FakeSkillsGraph(
        subjects=_roster("who-immunization"),
        arcs={
            "skill/who-immunization": skill_node("who-immunization")
            | {
                "agentSkillReferences": (
                    "skill/who-immunization/z-last",
                    "skill/who-immunization/a-first",
                )
            },
            "skill/who-immunization/z-last": {
                "name": ("references/z-last.md",),
                "text": ("z",),
                "typeOf": (REFERENCE_TYPE,),
            },
            "skill/who-immunization/a-first": {
                "name": ("references/a-first.md",),
                "text": ("a",),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    paths = [resource.path for resource in index.skills["who-immunization"].files[1:]]
    assert paths == ["references/a-first.md", "references/z-last.md"]


def test_reference_with_no_content_is_dropped_and_named():
    graph = FakeSkillsGraph(
        subjects=_roster("who-immunization"),
        arcs={
            "skill/who-immunization": skill_node("who-immunization")
            | {"agentSkillReferences": ("skill/who-immunization/notes",)},
            "skill/who-immunization/notes": {"name": ("references/notes.md",)},
        },
    )

    index = build_index(graph=graph)

    assert index.skills["who-immunization"].files[1:] == ()
    assert any(
        issue.code == "skill_resource_rejected"
        and issue.severity == "rejection"
        and "skill/who-immunization/notes (references/notes.md)" in issue.message
        for issue in index.issues
    )


@pytest.mark.parametrize(
    "reference_dcid",
    [
        "skill/some-other-skill/notes",
        "skill/who-immunization-2/notes",
        "dc/skill/who-immunization/notes",
        "who-immunization/notes",
    ],
)
def test_reference_dcid_not_under_its_skill_is_rejected_named_and_never_read(reference_dcid):
    graph = FakeSkillsGraph(
        subjects=_roster("who-immunization"),
        arcs={
            "skill/who-immunization": skill_node("who-immunization")
            | {"agentSkillReferences": (reference_dcid,)},
            reference_dcid: {
                "name": ("references/notes.md",),
                "text": ("Notes.",),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    assert index.skills["who-immunization"].files[1:] == ()
    assert any(
        issue.code == "skill_resource_rejected"
        and issue.skill_id == "who-immunization"
        and reference_dcid in issue.message
        for issue in index.issues
    )
    assert reference_dcid not in {dcid for chunk, _ in graph.arc_calls for dcid in chunk}


def _reference_graph(*, name: tuple[str, ...], text: str = "Notes.") -> FakeSkillsGraph:
    return FakeSkillsGraph(
        subjects=_roster("who-immunization"),
        arcs={
            "skill/who-immunization": skill_node("who-immunization")
            | {
                "agentSkillReferences": ("skill/who-immunization/notes",),
            },
            "skill/who-immunization/notes": {
                "name": name,
                "text": (text,),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )


def test_reference_path_comes_from_name_and_not_from_the_dcid():
    graph = _reference_graph(name=("references/deeper/other-file.md",))

    index = build_index(graph=graph)

    (resource,) = index.skills["who-immunization"].files[1:]
    assert resource.path == "references/deeper/other-file.md"


@pytest.mark.parametrize(
    "name",
    [
        "",
        "/references/notes.md",
        "references\\notes.md",
        "references/../notes.md",
        "../notes.md",
        "references/notes.csv",
        "references/notes",
        "references/notes.md.md",
        "references/Notes.md",
        "references/no tes.md",
        "references//notes.md",
        "SKILL.md",
        "skill.md",
        "x.md",
        "other/y.md",
        "references.md",
        "references/" + "a" * 500 + ".md",
    ],
)
def test_reference_with_an_invalid_name_path_is_dropped_and_the_skill_still_serves(name):
    graph = _reference_graph(name=(name,))

    index = build_index(graph=graph)

    entry = index.skills["who-immunization"]
    assert entry.content_text == skill_markdown("who-immunization")
    assert entry.files[1:] == ()
    (issue,) = [i for i in index.issues if i.code == "skill_resource_rejected"]
    assert issue.skill_id == "who-immunization"
    assert "skill/who-immunization/notes" in issue.message
    assert "reference path" in issue.message or "values" in issue.message


def test_reference_name_at_the_length_ceiling_is_admitted():
    name = "references/" + "a" * 498 + ".md"
    assert len(name) == 512
    graph = _reference_graph(name=(name,))

    index = build_index(graph=graph)

    assert [resource.path for resource in index.skills["who-immunization"].files[1:]] == [name]


@pytest.mark.parametrize("names", [(), ("references/a.md", "references/b.md")])
def test_reference_without_exactly_one_name_is_dropped_and_named(names):
    graph = _reference_graph(name=names)

    index = build_index(graph=graph)

    assert index.skills["who-immunization"].files[1:] == ()
    (issue,) = [i for i in index.issues if i.code == "skill_resource_rejected"]
    assert f"carries {len(names)} 'name' values" in issue.message


def test_two_references_carrying_the_same_name_are_both_dropped():
    graph = FakeSkillsGraph(
        subjects=_roster("who-immunization"),
        arcs={
            "skill/who-immunization": skill_node("who-immunization")
            | {
                "agentSkillReferences": (
                    "skill/who-immunization/one",
                    "skill/who-immunization/two",
                    "skill/who-immunization/three",
                ),
            },
            "skill/who-immunization/one": {
                "name": ("references/notes.md",),
                "text": ("One.",),
                "typeOf": (REFERENCE_TYPE,),
            },
            "skill/who-immunization/two": {
                "name": ("references/notes.md",),
                "text": ("Two.",),
                "typeOf": (REFERENCE_TYPE,),
            },
            "skill/who-immunization/three": {
                "name": ("references/other.md",),
                "text": ("Three.",),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    assert [r.path for r in index.skills["who-immunization"].files[1:]] == ["references/other.md"]
    rejected = {
        i.message.split(":")[0] for i in index.issues if i.code == "skill_resource_rejected"
    }
    assert rejected == {"skill/who-immunization/one", "skill/who-immunization/two"}


def test_a_reference_declared_twice_counts_once():
    graph = _reference_graph(name=("references/notes.md",))
    graph.arcs["skill/who-immunization"]["agentSkillReferences"] = (
        "skill/who-immunization/notes",
        "skill/who-immunization/notes",
    )

    index = build_index(graph=graph)

    assert [r.path for r in index.skills["who-immunization"].files[1:]] == ["references/notes.md"]
    assert not [i for i in index.issues if i.code == "skill_resource_rejected"]


def test_a_reference_node_not_listed_by_the_skill_is_never_read():
    """The skill's forward arc is the only link; a node that merely claims to belong
    (by dcid, or by a reverse arc) is not followed."""
    graph = FakeSkillsGraph(
        subjects={
            **_roster("who-immunization"),
            ("skill/who-immunization", "isPartOf"): ["skill/who-immunization/planted"],
        },
        arcs={
            "skill/who-immunization": skill_node("who-immunization"),
            "skill/who-immunization/planted": {
                "name": ("references/planted.md",),
                "text": ("Planted.",),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    assert index.skills["who-immunization"].files[1:] == ()
    assert not [i for i in index.issues if i.code == "skill_resource_rejected"]
    assert "skill/who-immunization/planted" not in {
        dcid for chunk, _ in graph.arc_calls for dcid in chunk
    }


def test_reference_bundle_reads_the_whole_corpus_in_two_batched_calls():
    graph = FakeSkillsGraph(
        subjects=_roster("who-immunization", "crs-purpose-codes"),
        arcs={
            "skill/who-immunization": skill_node("who-immunization")
            | {"agentSkillReferences": ("skill/who-immunization/schedule",)},
            "skill/who-immunization/schedule": {
                "name": ("references/schedule.md",),
                "text": ("Schedule.",),
                "typeOf": (REFERENCE_TYPE,),
            },
            "skill/crs-purpose-codes": skill_node("crs-purpose-codes")
            | {"agentSkillReferences": ("skill/crs-purpose-codes/codes",)},
            "skill/crs-purpose-codes/codes": {
                "name": ("references/codes.md",),
                "text": ("code,label",),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    assert len(graph.arc_calls) == 2
    assert index.skills["who-immunization"].files[1].content == "Schedule."
    assert index.skills["crs-purpose-codes"].files[1].content == "code,label"


def test_reference_past_the_byte_ceiling_is_dropped_and_named_while_its_skill_still_serves():
    oversized_resource = "x" * (SKILL_REFERENCE_MAX_BYTES + 1)
    graph = FakeSkillsGraph(
        subjects=_roster("who-immunization"),
        arcs={
            "skill/who-immunization": skill_node("who-immunization")
            | {
                "agentSkillReferences": ("skill/who-immunization/big",),
            },
            "skill/who-immunization/big": {
                "name": ("references/big.md",),
                "text": (oversized_resource,),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    entry = index.skills["who-immunization"]
    assert entry.content_text == skill_markdown("who-immunization")
    assert entry.files[1:] == ()
    assert any(
        issue.code == "skill_resource_rejected"
        and issue.severity == "rejection"
        and issue.skill_id == "who-immunization"
        and "big.md" in issue.message
        and "ceiling" in issue.message
        for issue in index.issues
    )


def test_manifest_past_the_ceiling_drops_only_the_excess():
    manifest = tuple(f"skill/skill-a/r{i}" for i in range(SKILL_REFERENCES_MAX_ENTRIES + 1))
    arcs = {"skill/skill-a": skill_node("skill-a") | {"agentSkillReferences": manifest}}
    for dcid in manifest:
        arcs[dcid] = {
            "name": (f"references/{dcid.rsplit('/', 1)[1]}.md",),
            "text": ("content",),
            "typeOf": (REFERENCE_TYPE,),
        }
    graph = FakeSkillsGraph(subjects=_roster("skill-a"), arcs=arcs)

    index = build_index(graph=graph)

    resources = index.skills["skill-a"].files[1:]
    dropped_dcid = manifest[-1]
    assert len(resources) == SKILL_REFERENCES_MAX_ENTRIES
    assert all(
        resource.path != f"references/{dropped_dcid.rsplit('/', 1)[1]}.md" for resource in resources
    )
    assert any(
        issue.code == "skill_resource_rejected"
        and issue.skill_id == "skill-a"
        and dropped_dcid in issue.message
        and "reference ceiling" in issue.message
        for issue in index.issues
    )
    # The cap rejects this excess dcid before a graph request.
    requested = {dcid for chunk, _properties in graph.arc_calls for dcid in chunk}
    assert dropped_dcid not in requested


def test_a_failing_reference_chunk_drops_that_chunks_references_and_keeps_serving_their_skills(
    monkeypatch,
):
    monkeypatch.setattr(build_module, "_REFERENCE_CHUNK_SIZE", 1)
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b"),
        arcs={
            "skill/skill-a": skill_node("skill-a", body="A body.")
            | {
                "agentSkillReferences": ("skill/skill-a/notes",),
            },
            "skill/skill-a/notes": {
                "name": ("references/notes.md",),
                "text": ("Notes.",),
                "typeOf": (REFERENCE_TYPE,),
            },
            "skill/skill-b": skill_node("skill-b", body="B body.")
            | {
                "agentSkillReferences": ("skill/skill-b/notes",),
            },
            "skill/skill-b/notes": {
                "name": ("references/notes.md",),
                "text": ("Notes.",),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
        failing_arc_dcids=frozenset({"skill/skill-a/notes"}),
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-a", "skill-b"}
    assert index.skills["skill-a"].content_text == skill_markdown("skill-a", body="A body.")
    assert index.skills["skill-a"].files[1:] == ()
    assert index.skills["skill-b"].files[1].content == "Notes."
    assert any(
        issue.code == "skill_resource_rejected"
        and issue.skill_id == "skill-a"
        and "skill/skill-a/notes" in issue.message
        for issue in index.issues
    )
    assert not any(issue.skill_id == "skill-b" for issue in index.issues)


def test_every_reference_chunk_failing_reraises_instead_of_stripping_every_reference():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={
            "skill/skill-a": skill_node("skill-a")
            | {"agentSkillReferences": ("skill/skill-a/notes",)},
            "skill/skill-a/notes": {"name": ("references/notes.md",), "text": ("Notes.",)},
        },
        failing_arc_dcids=frozenset({"skill/skill-a/notes"}),
    )

    with pytest.raises(RuntimeError):
        build_index(graph=graph)


def test_reference_missing_the_declared_type_is_rejected_and_named():
    graph = FakeSkillsGraph(
        subjects=_roster("who-immunization"),
        arcs={
            "skill/who-immunization": skill_node("who-immunization")
            | {"agentSkillReferences": ("skill/who-immunization/notes",)},
            "skill/who-immunization/notes": {"name": ("references/notes.md",), "text": ("Notes.",)},
        },
    )

    index = build_index(graph=graph)

    assert index.skills["who-immunization"].files[1:] == ()
    assert any(
        issue.code == "skill_resource_rejected"
        and issue.severity == "rejection"
        and issue.skill_id == "who-immunization"
        and "notes.md" in issue.message
        and REFERENCE_TYPE in issue.message
        for issue in index.issues
    )


def test_reference_with_the_wrong_declared_type_is_rejected_and_named():
    graph = FakeSkillsGraph(
        subjects=_roster("who-immunization"),
        arcs={
            "skill/who-immunization": skill_node("who-immunization")
            | {"agentSkillReferences": ("skill/who-immunization/notes",)},
            "skill/who-immunization/notes": {
                "name": ("references/notes.md",),
                "text": ("Notes.",),
                "typeOf": ("SomeOtherType",),
            },
        },
    )

    index = build_index(graph=graph)

    assert index.skills["who-immunization"].files[1:] == ()
    assert any(
        issue.code == "skill_resource_rejected" and REFERENCE_TYPE in issue.message
        for issue in index.issues
    )


def test_reference_declaring_the_resource_type_still_resolves():
    graph = FakeSkillsGraph(
        subjects=_roster("who-immunization"),
        arcs={
            "skill/who-immunization": skill_node("who-immunization")
            | {
                "agentSkillReferences": ("skill/who-immunization/notes",),
            },
            "skill/who-immunization/notes": {
                "name": ("references/notes.md",),
                "text": ("Notes.",),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    resources = index.skills["who-immunization"].files[1:]
    assert len(resources) == 1
    assert resources[0].content == "Notes."
    assert not any(
        issue.code == "skill_resource_rejected" and issue.skill_id == "who-immunization"
        for issue in index.issues
    )


def test_reference_chunk_size_leaves_headroom_for_worst_case_json_escaping():
    # 128 max-size resources exceed the mixer cap when NULs expand to six JSON bytes.
    # The configured chunk needs headroom.
    escape_heavy_resource = "\x00" * SKILL_REFERENCE_MAX_BYTES
    aggregate_response = json.dumps(
        {
            f"skill-a/r{i}.md": {"text": (escape_heavy_resource,)}
            for i in range(build_module._REFERENCE_CHUNK_SIZE)
        }
    )

    assert len(aggregate_response.encode("utf-8")) < build_module._MIXER_RESPONSE_CAP_BYTES


def _text_bytes(skill_id: str, *, body: str = "Body.") -> int:
    return len(skill_markdown(skill_id, body=body).encode("utf-8"))


def _two_skills_with_one_reference_each() -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        "skill/skill-a": skill_node("skill-a") | {"agentSkillReferences": ("skill/skill-a/notes",)},
        "skill/skill-a/notes": {
            "name": ("references/notes.md",),
            "text": ("x" * 8,),
            "typeOf": (REFERENCE_TYPE,),
        },
        "skill/skill-b": skill_node("skill-b") | {"agentSkillReferences": ("skill/skill-b/notes",)},
        "skill/skill-b/notes": {
            "name": ("references/notes.md",),
            "text": ("y" * 8,),
            "typeOf": (REFERENCE_TYPE,),
        },
    }


def test_skill_left_out_by_the_corpus_budget_never_has_its_references_read(monkeypatch):
    # A 5-byte budget admits skill-a's text, which spends it, so skill-b is left out
    # before any of its references enters a read.
    monkeypatch.setattr(build_module, "_FRONT_MATTER_CHUNK_SIZE", 1)
    monkeypatch.setattr(build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", 5)
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b"), arcs=_two_skills_with_one_reference_each()
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-a"}
    assert index.skills["skill-a"].files[1].content == "x" * 8
    assert [(issue.code, issue.skill_id) for issue in index.issues] == [
        ("skills_corpus_budget_exhausted", "skill/skill-b")
    ]
    assert index.may_omit_published_skills is True
    requested_dcids = {dcid for chunk, _properties in graph.arc_calls for dcid in chunk}
    assert "skill/skill-b/notes" not in requested_dcids
    assert "skill/skill-a/notes" in requested_dcids
    # Calls cover two skill-node chunks and one reference chunk.
    assert len(graph.arc_calls) == 3


def test_reference_bytes_alone_can_exhaust_the_corpus_budget(monkeypatch):
    # The budget equals skill-a's text, so both texts are admitted. skill-a's
    # 8-byte reference then takes skill-b's baseline past the budget, and skill-b
    # is left out before its reference is read.
    monkeypatch.setattr(build_module, "_REFERENCE_CHUNK_SIZE", 1)
    monkeypatch.setattr(build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", _text_bytes("skill-a"))
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b"), arcs=_two_skills_with_one_reference_each()
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-a"}
    assert index.skills["skill-a"].files[1].content == "x" * 8
    assert "skill/skill-b/notes" not in {
        dcid for chunk, _properties in graph.arc_calls for dcid in chunk
    }
    assert [(issue.code, issue.skill_id) for issue in index.issues] == [
        ("skills_corpus_budget_exhausted", "skill/skill-b")
    ]
    assert index.may_omit_published_skills is True


def test_issues_recorded_for_a_skill_before_it_is_left_out_are_dropped(monkeypatch):
    monkeypatch.setattr(build_module, "_REFERENCE_CHUNK_SIZE", 1)
    monkeypatch.setattr(build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", _text_bytes("skill-a"))
    arcs = _two_skills_with_one_reference_each()
    arcs["skill/skill-b"] = arcs["skill/skill-b"] | {
        "agentSkillReferences": ("skill/skill-b/notes", "skill/other/notes")
    }
    graph = FakeSkillsGraph(subjects=_roster("skill-a", "skill-b"), arcs=arcs)

    index = build_index(graph=graph)

    assert [(issue.code, issue.skill_id) for issue in index.issues] == [
        ("skills_corpus_budget_exhausted", "skill/skill-b")
    ]
    assert index.may_omit_published_skills is True


def test_withheld_resource_content_is_absent_from_the_retained_structure_itself(
    monkeypatch,
):
    # Inspect the state `_assemble_skills` receives: skill-a's reference content
    # is held and skill-b's was never read.
    # One-reference chunks isolate skill-b's budget check.
    monkeypatch.setattr(build_module, "_REFERENCE_CHUNK_SIZE", 1)
    monkeypatch.setattr(build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", _text_bytes("skill-a"))
    captured = _retained_reference_state(monkeypatch)
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b"), arcs=_two_skills_with_one_reference_each()
    )

    build_index(graph=graph)

    state = captured["state"]
    assert set(state.references_by_skill) == {"skill-a"}
    assert state.references_by_skill["skill-a"].content_by_path == {"references/notes.md": "x" * 8}


def test_reference_dcids_past_the_budget_are_never_placed_in_a_read(monkeypatch):
    # One-resource chunks isolate each budget check. The assertions inspect actual read requests.
    monkeypatch.setattr(build_module, "_REFERENCE_CHUNK_SIZE", 1)
    monkeypatch.setattr(build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", _text_bytes("skill-a"))
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b", "skill-c"),
        arcs={
            "skill/skill-a": skill_node("skill-a")
            | {
                "agentSkillReferences": ("skill/skill-a/notes",),
            },
            "skill/skill-a/notes": {
                "name": ("references/notes.md",),
                "text": ("x" * 8,),
                "typeOf": (REFERENCE_TYPE,),
            },
            "skill/skill-b": skill_node("skill-b")
            | {
                "agentSkillReferences": ("skill/skill-b/notes",),
            },
            "skill/skill-b/notes": {
                "name": ("references/notes.md",),
                "text": ("y" * 8,),
                "typeOf": (REFERENCE_TYPE,),
            },
            "skill/skill-c": skill_node("skill-c")
            | {
                "agentSkillReferences": ("skill/skill-c/notes",),
            },
            "skill/skill-c/notes": {
                "name": ("references/notes.md",),
                "text": ("z" * 8,),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    build_index(graph=graph)

    requested_dcids = {dcid for chunk, _properties in graph.arc_calls for dcid in chunk}
    assert "skill/skill-a/notes" in requested_dcids
    assert "skill/skill-b/notes" not in requested_dcids
    assert "skill/skill-c/notes" not in requested_dcids


# `may_omit_published_skills` is set by every issue that leaves a skill out or
# partial for a budget or an unconfirmed read. Skills left out for their content
# and rejected resources leave it false.


def test_default_build_leaves_the_flag_false():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={"skill/skill-a": skill_node("skill-a")},
    )

    index = build_index(graph=graph)

    assert index.may_omit_published_skills is False


def test_roster_truncation_sets_the_flag():
    graph = FakeSkillsGraph(
        subjects=_roster("who-immunization"), budget_exhausted={("AgentSkill", "typeOf")}
    )

    index = build_index(graph=graph)

    assert index.may_omit_published_skills is True


def test_a_failing_front_matter_chunk_sets_the_flag(monkeypatch):
    monkeypatch.setattr(build_module, "_FRONT_MATTER_CHUNK_SIZE", 1)
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b"),
        arcs={
            "skill/skill-a": skill_node("skill-a"),
            "skill/skill-b": skill_node("skill-b"),
        },
        failing_arc_dcids=frozenset({"skill/skill-a"}),
    )

    index = build_index(graph=graph)

    assert index.may_omit_published_skills is True


def test_corpus_budget_exhausted_alone_sets_the_flag(monkeypatch):
    monkeypatch.setattr(build_module, "_FRONT_MATTER_CHUNK_SIZE", 1)
    monkeypatch.setattr(build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", 5)
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b"),
        arcs={
            "skill/skill-a": skill_node("skill-a"),
            "skill/skill-b": skill_node("skill-b"),
        },
    )

    index = build_index(graph=graph)

    assert [issue.code for issue in index.issues] == ["skills_corpus_budget_exhausted"]
    assert index.may_omit_published_skills is True


def test_rejected_resource_alone_leaves_the_flag_false():
    graph = FakeSkillsGraph(
        subjects=_roster("who-immunization"),
        arcs={
            "skill/who-immunization": skill_node("who-immunization")
            | {
                "agentSkillReferences": ("skill/who-immunization/notes",),
            },
            "skill/who-immunization/notes": {"name": ("references/notes.md",)},
        },
    )

    index = build_index(graph=graph)

    assert any(issue.code == "skill_resource_rejected" for issue in index.issues)
    assert index.may_omit_published_skills is False


def test_each_unusable_text_leaves_its_skill_out_under_its_own_code(monkeypatch):
    # Under a 5-byte budget skill-ok's text is admitted first and spends it.
    monkeypatch.setattr(build_module, "_FRONT_MATTER_CHUNK_SIZE", 1)
    monkeypatch.setattr(build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", 5)
    graph = FakeSkillsGraph(
        subjects=_roster(
            "skill-undecodable",
            "skill-absent",
            "skill-toolarge",
            "skill-invalid",
            "skill-ok",
            "skill-budget",
        ),
        arcs={
            "skill/skill-undecodable": {"text": ('a raw "quote"',)},
            "skill/skill-absent": {},
            "skill/skill-toolarge": {
                "text": (stored(_markdown_of_bytes("skill-toolarge", SKILL_BODY_MAX_BYTES + 1)),)
            },
            "skill/skill-invalid": skill_node("skill-other"),
            "skill/skill-ok": skill_node("skill-ok"),
            "skill/skill-budget": skill_node("skill-budget"),
        },
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-ok"}
    assert [(issue.code, issue.skill_id) for issue in index.issues] == [
        ("skill_content_undecodable", "skill/skill-undecodable"),
        ("skill_content_absent", "skill/skill-absent"),
        ("skill_content_too_large", "skill/skill-toolarge"),
        ("skill_frontmatter_invalid", "skill/skill-invalid"),
        ("skills_corpus_budget_exhausted", "skill/skill-budget"),
    ]
    assert all(issue.severity == "warning" for issue in index.issues)
    # Only the budget leaves out a valid skill, so only it marks the index incomplete.
    assert index.may_omit_published_skills is True


def test_texts_left_out_never_spend_the_corpus_budget(monkeypatch):
    # Oversized and invalid texts contribute zero retained bytes, so the healthy
    # skill still finds the 5-byte budget unspent.
    monkeypatch.setattr(build_module, "_FRONT_MATTER_CHUNK_SIZE", 1)
    monkeypatch.setattr(build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", 5)
    graph = FakeSkillsGraph(
        subjects=_roster("skill-big", "skill-invalid", "skill-healthy"),
        arcs={
            "skill/skill-big": {
                "text": (stored(_markdown_of_bytes("skill-big", SKILL_BODY_MAX_BYTES + 1)),)
            },
            "skill/skill-invalid": skill_node("skill-other", body="x" * 100),
            "skill/skill-healthy": skill_node("skill-healthy"),
        },
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-healthy"}
    assert [issue.code for issue in index.issues] == [
        "skill_content_too_large",
        "skill_frontmatter_invalid",
    ]


def _retained_reference_state(monkeypatch) -> dict[str, object]:
    """Captures the build state `_assemble_skills` receives."""
    captured: dict[str, object] = {}
    original_assemble = build_module._assemble_skills

    def spy_assemble(state, *args, **kwargs):
        captured["state"] = state
        return original_assemble(state, *args, **kwargs)

    monkeypatch.setattr(build_module, "_assemble_skills", spy_assemble)
    return captured


def _held_reference_bytes(state) -> int:
    return sum(
        len(content.encode("utf-8"))
        for references in state.references_by_skill.values()
        for content in references.content_by_path.values()
    )


def _assert_held_equals_counted_at_every_read(monkeypatch, graph: FakeSkillsGraph) -> list[int]:
    """Checks, before every node read and once the stage ends, that retained
    bytes equal the texts and reference content the build holds. Returns the
    held reference bytes seen at each check."""
    captured: dict[str, object] = {}
    held_at_checks: list[int] = []
    original_read_front_matter = build_module._read_front_matter
    original_node_arcs_many = graph.node_arcs_many

    def spy_read_front_matter(state, *args, **kwargs):
        captured["state"] = state
        return original_read_front_matter(state, *args, **kwargs)

    def check() -> None:
        state = captured["state"]
        held = _held_reference_bytes(state)
        text_bytes = sum(len(t.text.encode("utf-8")) for t in state.text_by_skill.values())
        assert state.retained_content_bytes == text_bytes + held
        held_at_checks.append(held)

    def checked_node_arcs_many(dcids, **kwargs):
        if "state" in captured:
            check()
        return original_node_arcs_many(dcids, **kwargs)

    monkeypatch.setattr(build_module, "_read_front_matter", spy_read_front_matter)
    monkeypatch.setattr(graph, "node_arcs_many", checked_node_arcs_many)
    original_assemble = build_module._assemble_skills

    def spy_assemble(state, *args, **kwargs):
        check()
        return original_assemble(state, *args, **kwargs)

    monkeypatch.setattr(build_module, "_assemble_skills", spy_assemble)
    return held_at_checks


def _reference(path: str, content: str, *, type_dcid: str = REFERENCE_TYPE):
    return {"name": (path,), "text": (content,), "typeOf": (type_dcid,)}


def test_references_sharing_a_path_are_neither_held_nor_counted(monkeypatch):
    # Twenty skills each declare two 1,000-byte references at one path. The
    # budget leaves 10 bytes past the texts, so any counted pair would leave
    # later skills out. Each pair is dropped as its second member is read.
    monkeypatch.setattr(build_module, "_REFERENCE_CHUNK_SIZE", 2)
    skill_ids = [f"skill-{n:02d}" for n in range(20)]
    monkeypatch.setattr(
        build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", sum(map(_text_bytes, skill_ids)) + 10
    )
    arcs: dict[str, dict[str, tuple[str, ...]]] = {}
    for skill_id in skill_ids:
        pair = (f"skill/{skill_id}/one", f"skill/{skill_id}/two")
        arcs[f"skill/{skill_id}"] = skill_node(skill_id) | {"agentSkillReferences": pair}
        for dcid in pair:
            arcs[dcid] = _reference("references/x.md", "z" * 1000)
    graph = FakeSkillsGraph(subjects=_roster(*skill_ids), arcs=arcs)
    held_at_checks = _assert_held_equals_counted_at_every_read(monkeypatch, graph)

    index = build_index(graph=graph)

    assert set(index.skills) == set(skill_ids)
    assert all(entry.files[1:] == () for entry in index.skills.values())
    assert held_at_checks and set(held_at_checks) == {0}


@pytest.mark.parametrize(
    ("skill_a_references", "served_paths"),
    [
        pytest.param(
            {
                "d1": _reference("references/x.md", "d" * 1000),
                "d2": _reference("references/x.md", "e" * 1000),
                "u": _reference("references/u.md", "u"),
            },
            ["references/u.md"],
            id="shared-path-pair-read-before-the-check",
        ),
        pytest.param(
            {
                "u": _reference("references/u.md", "u"),
                "d1": _reference("references/x.md", "d" * 1000),
                "d2": _reference("references/x.md", "e" * 1000),
            },
            ["references/u.md"],
            id="shared-path-pair-split-by-the-chunk-boundary",
        ),
        pytest.param(
            {
                "r": _reference("references/r.md", "r" * 1000, type_dcid="Thing"),
                "u1": _reference("references/u1.md", "1"),
                "u2": _reference("references/u2.md", "2"),
            },
            ["references/u1.md", "references/u2.md"],
            id="rejected-reference-read-before-the-check",
        ),
        pytest.param(
            {
                "d1": _reference("references/x.md", "d" * 1000),
                "d2": _reference("references/x.md", "e" * 1000),
                "d3": _reference("references/x.md", "f" * 1000),
            },
            [],
            id="third-reference-at-a-dropped-path",
        ),
    ],
)
def test_a_straddling_list_never_starves_the_next_skill(
    monkeypatch, skill_a_references, served_paths
):
    # Two-reference chunks read skill-a's first two references and leave the
    # third pending at skill-b's budget check. A reference whose path partner is
    # still pending is counted there, so a failing check first reads every
    # pending reference and checks again. skill-b then fits a budget with 100
    # bytes to spare.
    monkeypatch.setattr(build_module, "_REFERENCE_CHUNK_SIZE", 2)
    monkeypatch.setattr(
        build_module,
        "SKILLS_CORPUS_CONTENT_BUDGET_BYTES",
        _text_bytes("skill-a") + _text_bytes("skill-b") + 100,
    )
    dcids = tuple(f"skill/skill-a/{name}" for name in skill_a_references)
    arcs = {
        "skill/skill-a": skill_node("skill-a") | {"agentSkillReferences": dcids},
        **{f"skill/skill-a/{name}": node for name, node in skill_a_references.items()},
        "skill/skill-b": skill_node("skill-b"),
    }
    graph = FakeSkillsGraph(subjects=_roster("skill-a", "skill-b"), arcs=arcs)
    held_at_checks = _assert_held_equals_counted_at_every_read(monkeypatch, graph)

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-a", "skill-b"}
    assert [file.path for file in index.skills["skill-a"].files[1:]] == served_paths
    # At assembly the build holds exactly the reference bytes skill-a serves.
    assert held_at_checks[-1] == sum(file.size for file in index.skills["skill-a"].files[1:])
    assert {issue.code for issue in index.issues} == {"skill_resource_rejected"}


def test_a_skill_whose_references_straddle_a_chunk_counts_what_was_read(monkeypatch):
    # Two-reference chunks read skill-a's first two 8-byte references and leave
    # the third pending when skill-b's budget check runs. Those 16 read bytes
    # count, so skill-b's baseline passes a budget with 10 bytes to spare.
    monkeypatch.setattr(build_module, "_REFERENCE_CHUNK_SIZE", 2)
    monkeypatch.setattr(
        build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", _text_bytes("skill-a") + 10
    )
    references = tuple(f"skill/skill-a/{n}" for n in ("one", "two", "three"))
    arcs = {
        "skill/skill-a": skill_node("skill-a") | {"agentSkillReferences": references},
        **{
            dcid: {
                "name": (f"references/{dcid.rsplit('/', 1)[1]}.md",),
                "text": ("x" * 8,),
                "typeOf": (REFERENCE_TYPE,),
            }
            for dcid in references
        },
        "skill/skill-b": skill_node("skill-b"),
    }

    index = build_index(graph=FakeSkillsGraph(subjects=_roster("skill-a", "skill-b"), arcs=arcs))

    assert set(index.skills) == {"skill-a"}
    assert len(index.skills["skill-a"].files) == 4
    assert [(issue.code, issue.skill_id) for issue in index.issues] == [
        ("skills_corpus_budget_exhausted", "skill/skill-b")
    ]


def test_references_dropped_for_a_shared_path_never_spend_the_corpus_budget(monkeypatch):
    # The budget equals skill-dup's text. Its two references share one path, so
    # both are dropped. Had their 16 bytes counted, skill-healthy's baseline would
    # pass the budget and leave it out.
    monkeypatch.setattr(build_module, "_REFERENCE_CHUNK_SIZE", 1)
    monkeypatch.setattr(
        build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", _text_bytes("skill-dup")
    )
    graph = FakeSkillsGraph(
        subjects=_roster("skill-dup", "skill-healthy"),
        arcs={
            "skill/skill-dup": skill_node("skill-dup")
            | {"agentSkillReferences": ("skill/skill-dup/one", "skill/skill-dup/two")},
            "skill/skill-dup/one": {
                "name": ("references/notes.md",),
                "text": ("x" * 8,),
                "typeOf": (REFERENCE_TYPE,),
            },
            "skill/skill-dup/two": {
                "name": ("references/notes.md",),
                "text": ("y" * 8,),
                "typeOf": (REFERENCE_TYPE,),
            },
            "skill/skill-healthy": skill_node("skill-healthy"),
        },
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-dup", "skill-healthy"}
    assert index.skills["skill-dup"].files[1:] == ()
    assert [(issue.code, issue.skill_id) for issue in index.issues] == [
        ("skill_resource_rejected", "skill-dup"),
        ("skill_resource_rejected", "skill-dup"),
    ]


def test_dropped_resources_never_spend_the_corpus_budget_they_are_dropped_from(
    monkeypatch,
):
    # Oversized and wrong-type resources contribute zero retained bytes. One-item chunks place both
    # ahead of the healthy resource.
    # The budget admits every text. Reference bytes that counted would take
    # skill-healthy's baseline past it.
    monkeypatch.setattr(build_module, "_REFERENCE_CHUNK_SIZE", 1)
    monkeypatch.setattr(
        build_module,
        "SKILLS_CORPUS_CONTENT_BUDGET_BYTES",
        _text_bytes("skill-oversized", body="a") + _text_bytes("skill-wrongtype", body="b"),
    )
    oversized_resource = "x" * (SKILL_REFERENCE_MAX_BYTES + 1)
    graph = FakeSkillsGraph(
        subjects=_roster("skill-oversized", "skill-wrongtype", "skill-healthy"),
        arcs={
            "skill/skill-oversized": skill_node("skill-oversized", body="a")
            | {
                "agentSkillReferences": ("skill/skill-oversized/big",),
            },
            "skill/skill-oversized/big": {
                "name": ("references/big.md",),
                "text": (oversized_resource,),
                "typeOf": (REFERENCE_TYPE,),
            },
            "skill/skill-wrongtype": skill_node("skill-wrongtype", body="b")
            | {
                "agentSkillReferences": ("skill/skill-wrongtype/notes",),
            },
            "skill/skill-wrongtype/notes": {
                "name": ("references/notes.md",),
                "text": ("y" * 8,),
                "typeOf": ("SomeOtherType",),
            },
            "skill/skill-healthy": skill_node("skill-healthy", body="c")
            | {
                "agentSkillReferences": ("skill/skill-healthy/notes",),
            },
            "skill/skill-healthy/notes": {
                "name": ("references/notes.md",),
                "text": ("ok",),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    assert index.skills["skill-oversized"].content_text == skill_markdown(
        "skill-oversized", body="a"
    )
    assert index.skills["skill-oversized"].files[1:] == ()
    assert index.skills["skill-wrongtype"].content_text == skill_markdown(
        "skill-wrongtype", body="b"
    )
    assert index.skills["skill-wrongtype"].files[1:] == ()
    assert index.skills["skill-healthy"].content_text == skill_markdown("skill-healthy", body="c")
    assert index.skills["skill-healthy"].files[1].content == "ok"

    assert not any(issue.code == "skills_corpus_budget_exhausted" for issue in index.issues)
    assert any(
        issue.code == "skill_resource_rejected"
        and issue.skill_id == "skill-oversized"
        and "ceiling" in issue.message
        for issue in index.issues
    )
    assert any(
        issue.code == "skill_resource_rejected"
        and issue.skill_id == "skill-wrongtype"
        and REFERENCE_TYPE in issue.message
        for issue in index.issues
    )
    assert not any(issue.skill_id == "skill-healthy" for issue in index.issues)


def test_manifest_lists_skill_md_then_references_with_sizes_and_digests():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={
            "skill/skill-a": skill_node("skill-a")
            | {"agentSkillReferences": ("skill/skill-a/notes",)},
            "skill/skill-a/notes": {
                "name": ("references/notes.md",),
                "text": (stored("Notes é."),),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.files == (
        SkillFile(
            path="SKILL.md",
            size=54,
            digest="sha256:4cee9a11f8786c0cf709da9353aad5b3c515e88fcfd99843857e2e7196d1eaa6",
            content=skill_markdown("skill-a"),
        ),
        SkillFile(
            path="references/notes.md",
            size=9,
            digest="sha256:d6c45618a1504a1fc0b8c9c4d6d04223f3d8e8d5ba9d4f7929afb790884064c0",
            content="Notes é.",
        ),
    )
    assert entry.front_matter.content_version == (
        "sha256:53cec9d010fa6895001190413d885eb6d6d54abda71816635ac46aa94bd7618c"
    )


def test_content_version_of_a_skill_without_references_covers_skill_md_alone():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"), arcs={"skill/skill-a": skill_node("skill-a")}
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert [file.path for file in entry.files] == ["SKILL.md"]
    assert entry.front_matter.content_version == (
        "sha256:22acd97fc984a0b9504aa6728a76c5c5cb85478f22d637d92b66f8568f1c5151"
    )


def test_content_version_moves_with_any_file_in_the_bundle():
    def version(reference_text: str) -> str:
        graph = FakeSkillsGraph(
            subjects=_roster("skill-a"),
            arcs={
                "skill/skill-a": skill_node("skill-a")
                | {"agentSkillReferences": ("skill/skill-a/notes",)},
                "skill/skill-a/notes": {
                    "name": ("references/notes.md",),
                    "text": (reference_text,),
                    "typeOf": (REFERENCE_TYPE,),
                },
            },
        )
        return build_index(graph=graph).skills["skill-a"].front_matter.content_version

    assert version("Notes.") != version("Notes!")


def test_manifest_follows_resource_path_order():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={
            "skill/skill-a": skill_node("skill-a")
            | {"agentSkillReferences": ("skill/skill-a/z", "skill/skill-a/a")},
            "skill/skill-a/z": {
                "name": ("references/z.md",),
                "text": ("z",),
                "typeOf": (REFERENCE_TYPE,),
            },
            "skill/skill-a/a": {
                "name": ("references/a.md",),
                "text": ("a",),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    assert [file.path for file in index.skills["skill-a"].files] == [
        "SKILL.md",
        "references/a.md",
        "references/z.md",
    ]


def test_attached_is_true_only_for_a_skill_whose_reach_names_a_node():
    graph = FakeSkillsGraph(
        subjects={
            **_roster(
                "skill-reach", "skill-provenance", "skill-none", "skill-rejected", "skill-excluded"
            ),
            ("Person", "populationType"): ["sv-1"],
            ("Nominal", "measurementQualifier"): ["sv-2"],
        },
        budget_exhausted={("Nominal", "isPartOf")},
        arcs={
            "skill/skill-reach": skill_node("skill-reach") | {"skillAppliesTo": ("Person",)},
            "skill/skill-provenance": skill_node("skill-provenance")
            | {"skillAppliesToProvenance": ("provenance/OECDCreditorReportingSystem",)},
            "skill/skill-none": skill_node("skill-none"),
            "skill/skill-rejected": skill_node("skill-rejected") | {"skillAppliesTo": ("nowhere",)},
            "skill/skill-excluded": skill_node("skill-excluded") | {"skillAppliesTo": ("Nominal",)},
        },
    )

    index = build_index(graph=graph)

    assert {skill_id: entry.attached for skill_id, entry in index.skills.items()} == {
        "skill-reach": True,
        "skill-provenance": True,
        "skill-none": False,
        "skill-rejected": False,
        "skill-excluded": False,
    }
    assert {
        dcid: [(match.skill_id, match.via_provenance) for match in matches]
        for dcid, matches in index.node_to_skills.items()
    } == {
        "Person": [("skill-reach", False)],
        "provenance/OECDCreditorReportingSystem": [("skill-provenance", True)],
    }


_CRS = "provenance/OECDCreditorReportingSystem"


@pytest.mark.parametrize(
    ("arc", "is_provenance_attachment"),
    [
        pytest.param("skillAppliesToProvenance", True, id="provenance-arc"),
        # A provenance dcid on the ontology arc goes through classification like any
        # other target. The dev graph's `Provenance <-rangeIncludes` names the
        # publisher's own arc, which is no evidence, so the target is rejected.
        pytest.param("skillAppliesTo", False, id="ontology-arc"),
    ],
)
def test_provenance_attachment_is_known_by_its_arc_and_matches_its_own_dcid(
    arc, is_provenance_attachment
):
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a"),
            ("Person", "populationType"): ["sv-1"],
            ("Provenance", "rangeIncludes"): ["skillAppliesToProvenance", "isPartOf"],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {arc: (_CRS, _CRS)},
            _CRS: {"typeOf": ("Provenance",)},
        },
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    reads_naming_target = [call for call in graph.calls if _CRS in call[0]] + [
        call for call in graph.arc_calls if _CRS in call[0]
    ]
    if is_provenance_attachment:
        assert {attachment.kind for attachment in entry.attachments} == {ProvenanceAttachment()}
        assert entry.reach.nodes == (ReachedNode(dcid=_CRS, hop=0, via_property=None),)
        assert entry.attached is True
        assert entry.statvar_count == 0
        assert entry.statvar_count_truncated is False
        assert [(match.skill_id, match.via_provenance) for match in index.node_to_skills[_CRS]] == [
            ("skill-a", True)
        ]
        assert index.issues == ()
        assert reads_naming_target == []
    else:
        assert all(isinstance(a.kind, RejectedAttachment) for a in entry.attachments)
        assert entry.attached is False
        assert _CRS not in index.node_to_skills
        assert reads_naming_target != []


def test_provenance_and_ontology_attachments_combine_under_one_node_bound():
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a", "skill-b"), ("Person", "populationType"): ["sv-1"]},
        arcs={
            "skill/skill-a": skill_node("skill-a")
            | {"skillAppliesTo": ("Person",), "skillAppliesToProvenance": (_CRS,)},
            "skill/skill-b": skill_node("skill-b")
            | {"skillAppliesToProvenance": (_CRS, "provenance/Other")},
        },
    )

    index = build_index(graph=graph, max_nodes=1)

    for skill_id in ("skill-a", "skill-b"):
        assert index.skills[skill_id].reach.excluded is True
        assert index.skills[skill_id].attached is False
    assert index.node_to_skills == {}
    assert {issue.skill_id for issue in index.issues if issue.code == "skill_reach_too_large"} == {
        "skill-a",
        "skill-b",
    }


def test_failed_reach_read_keeps_the_skill_served_excluded_and_marks_the_index_incomplete():
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a"), ("Person", "populationType"): ["sv-1"]},
        budget_exhausted={("Person", "isPartOf")},
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.reach.excluded is True
    assert entry.attached is False
    assert [(issue.code, issue.skill_id) for issue in index.issues] == [
        ("skill_build_unconfirmed", "skill-a")
    ]
    assert index.may_omit_published_skills is True
    assert index.node_to_skills == {}


# A read that raises during classification, reach, or counting leaves the skill
# served, excluded, and counted at 0, and marks the index incomplete.


def _served_and_unconfirmed(index, skill_id: str) -> None:
    entry = index.skills[skill_id]
    assert entry.reach.excluded is True
    assert entry.reach.nodes == ()
    assert entry.statvar_count == 0
    assert entry.attached is False
    (issue,) = [i for i in index.issues if i.skill_id == skill_id]
    assert issue.code == "skill_build_unconfirmed"
    assert "simulated graph infra failure" in issue.message
    assert index.may_omit_published_skills is True
    assert all(
        match.skill_id != skill_id for matches in index.node_to_skills.values() for match in matches
    )


def test_a_raised_classification_probe_leaves_every_skill_in_its_batch_served_and_excluded():
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a", "skill-b", "skill-c", "skill-d"),
            ("Person", "populationType"): ["sv-1"],
            ("Count", "statType"): ["sv-2"],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)},
            "skill/skill-b": skill_node("skill-b") | {"skillAppliesTo": ("Count",)},
            "skill/skill-c": skill_node("skill-c"),
            # The same dcid skill-a's failed batch held, on the provenance arc,
            # which needs no read.
            "skill/skill-d": skill_node("skill-d") | {"skillAppliesToProvenance": ("Person",)},
        },
        failing_subject_reads={("Person", "measuredProperty")},
    )

    index = build_index(graph=graph)

    _served_and_unconfirmed(index, "skill-a")
    _served_and_unconfirmed(index, "skill-b")
    assert index.skills["skill-d"].attached is True
    assert "Person" in next(i.message for i in index.issues if i.skill_id == "skill-a")
    assert index.skills["skill-c"].reach.excluded is False


def test_a_raised_type_lookup_leaves_its_skill_served_and_excluded():
    graph = FakeSkillsGraph(
        subjects={**_roster("skill-a", "skill-b"), ("Person", "populationType"): ["sv-1"]},
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("mystery-target",)},
            "skill/skill-b": skill_node("skill-b") | {"skillAppliesTo": ("Person",)},
        },
        failing_arc_dcids=frozenset({"mystery-target"}),
    )

    index = build_index(graph=graph)

    _served_and_unconfirmed(index, "skill-a")
    assert index.skills["skill-b"].attached is True
    assert index.skills["skill-b"].statvar_count == 1


def test_a_raised_reach_read_leaves_its_skill_served_and_excluded():
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a", "skill-b"),
            ("Person", "populationType"): ["sv-1"],
            ("Count", "statType"): ["sv-2"],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)},
            "skill/skill-b": skill_node("skill-b") | {"skillAppliesTo": ("Count",)},
        },
        failing_subject_reads={("Person", "specializationOf")},
    )

    index = build_index(graph=graph)

    _served_and_unconfirmed(index, "skill-a")
    assert index.skills["skill-b"].attached is True
    assert index.skills["skill-b"].statvar_count == 1


def test_a_raised_statvar_count_read_leaves_its_skill_served_and_excluded():
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a", "skill-b"),
            ("gender", "constraintProperties"): ["SomeClass"],
            ("Count", "statType"): ["sv-2"],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("gender",)},
            "skill/skill-b": skill_node("skill-b") | {"skillAppliesTo": ("Count",)},
        },
        failing_arc_dcids=frozenset({"gender"}),
    )

    index = build_index(graph=graph)

    _served_and_unconfirmed(index, "skill-a")
    assert index.skills["skill-b"].statvar_count == 1


def _every_hierarchy_read_of(*targets: str) -> set[tuple[str, str]]:
    """The reads of a graph outage confined to the reach of `targets`."""
    return {(target, prop) for target in targets for prop in HIERARCHY_PROPERTIES}


@pytest.mark.parametrize(
    "skill_b_arcs",
    [
        pytest.param({}, id="beside-an-unattached-skill"),
        pytest.param(
            {"skillAppliesToProvenance": ("provenance/X",)},
            id="beside-a-provenance-only-skill",
        ),
        pytest.param({"skillAppliesTo": ("Count",)}, id="every-reach-bearing-skill"),
    ],
)
def test_every_attempted_reach_read_raising_reraises(skill_b_arcs):
    # An unattached or provenance-only skill reads nothing for its reach, so it
    # cannot carry the build through a graph outage.
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a", "skill-b"),
            ("Person", "populationType"): ["sv-1"],
            ("Count", "statType"): ["sv-2"],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)},
            "skill/skill-b": skill_node("skill-b") | skill_b_arcs,
        },
        failing_subject_reads=_every_hierarchy_read_of("Person", "Count"),
    )

    with pytest.raises(RuntimeError, match="simulated graph infra failure"):
        build_index(graph=graph)


def test_an_outage_beside_a_skill_attached_only_to_an_excluded_root_reraises():
    # `Thing` is an excluded root, so skill-t's reach and count make no read and
    # say nothing about whether the graph is answering.
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a", "skill-t"),
            ("Person", "populationType"): ["sv-1"],
            ("Thing", "populationType"): ["sv-2"],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)},
            "skill/skill-t": skill_node("skill-t") | {"skillAppliesTo": ("Thing",)},
        },
        failing_subject_reads=_every_hierarchy_read_of("Person"),
    )

    with pytest.raises(RuntimeError, match="simulated graph infra failure"):
        build_index(graph=graph)


def test_a_raised_count_read_beside_answered_reach_reads_keeps_the_build():
    # skill-a's hierarchy reads succeed and only its StatVar count read raises,
    # so the graph is answering: skill-a is served unconfirmed and the
    # provenance-only skill is served as usual.
    graph = FakeSkillsGraph(
        subjects={
            **_roster("skill-a", "oda-crs"),
            ("Person", "populationType"): ["sv-1"],
            ("Person", "specializationOf"): ["Adult"],
        },
        arcs={
            "skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)},
            "skill/oda-crs": skill_node("oda-crs")
            | {"skillAppliesToProvenance": ("provenance/X",)},
        },
        failing_subject_reads={("Adult", "populationType")},
    )

    index = build_index(graph=graph)

    _served_and_unconfirmed(index, "skill-a")
    assert index.skills["oda-crs"].attached is True


def test_every_classification_read_raising_reraises():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
        failing_subject_reads={
            ("Person", probe) for probe in build_module._CLASSIFICATION_PROBE_PROPERTIES
        },
    )

    with pytest.raises(RuntimeError):
        build_index(graph=graph)
