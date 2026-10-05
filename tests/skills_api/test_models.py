"""`skills_api.models`: issue-to-warning conversion, the corpus and skill split, media
types, and the wire entry built from a catalogue entry."""

from __future__ import annotations

import pytest

from skills_api.models import (
    SkillBody,
    SkillSummary,
    media_type,
    skill_body,
    skill_summary,
    skill_warning_from_issue,
    split_skill_warnings,
)
from skills_index import SkillIssue

from .conftest import skill_entry


def test_rejection_severity_converts_to_degraded():
    issue = SkillIssue(
        skill_id="oda-concepts",
        code="skill_reach_too_deep",
        message="reach exceeds the 4-level depth bound",
    )

    warning = skill_warning_from_issue(issue)

    assert warning.severity == "degraded"
    assert warning.code == "skill_reach_too_deep"
    assert warning.message == "reach exceeds the 4-level depth bound"
    assert warning.skill_id == "oda-concepts"


def test_warning_severity_converts_to_warn():
    issue = SkillIssue(
        skill_id="oda-concepts", code="skill_breadth_high", message="reach covers 5001 StatVars"
    )

    assert skill_warning_from_issue(issue).severity == "warn"


def test_corpus_level_issue_keeps_a_null_skill_id():
    issue = SkillIssue(skill_id=None, code="skills_roster_truncated", message="roster truncated")

    assert skill_warning_from_issue(issue).skill_id is None


def test_split_routes_a_left_out_skills_dcid_to_the_corpus():
    issues = (
        SkillIssue(skill_id=None, code="skills_roster_truncated", message="truncated"),
        SkillIssue(skill_id="skill/left-out", code="skill_frontmatter_invalid", message="bad"),
        SkillIssue(skill_id="oda-concepts", code="skill_breadth_high", message="broad"),
    )

    corpus, per_skill = split_skill_warnings(issues, {"oda-concepts": skill_entry()})

    assert [(w.code, w.skill_id) for w in corpus] == [
        ("skills_roster_truncated", None),
        ("skill_frontmatter_invalid", "skill/left-out"),
    ]
    assert [w.code for w in per_skill["oda-concepts"]] == ["skill_breadth_high"]


def test_markdown_media_type():
    assert media_type("SKILL.md") == "text/markdown"
    assert media_type("references/core-oda.md") == "text/markdown"


def test_a_path_the_index_never_admits_has_no_media_type():
    with pytest.raises(KeyError):
        media_type("references/data.csv")


def test_skill_summary_mirrors_the_catalogue_entry_and_round_trips():
    entry = skill_entry(references={"references/core-oda.md": "# Core ODA\n"}, attached=True)

    summary = skill_summary("oda-concepts", entry, [])

    assert summary.frontmatter == entry.frontmatter
    assert summary.attached is True
    assert [(f.path, f.size, f.digest) for f in summary.resources] == [
        (f.path, f.size, f.digest) for f in entry.files
    ]
    assert SkillSummary.model_validate_json(summary.model_dump_json()) == summary


def test_skill_body_is_the_summary_plus_its_text():
    entry = skill_entry()
    summary = skill_summary("oda-concepts", entry, [])

    body = skill_body("oda-concepts", entry, [])

    assert isinstance(body, SkillSummary)
    assert body.model_dump(exclude={"content_text"}) == summary.model_dump()
    assert body.content_text == entry.content_text
    assert SkillBody.model_json_schema()["required"].count("content_text") == 1
