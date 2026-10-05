"""Tests the `SkillIssueCode` to severity mapping and derived severity field."""

from __future__ import annotations

from typing import get_args

from skills_index import SKILL_ISSUE_SEVERITY, SkillIssue, SkillIssueCode


def test_skill_issue_severity_is_total_over_skill_issue_code():
    """A new issue code without a severity mapping fails this check."""
    assert frozenset(SKILL_ISSUE_SEVERITY) == frozenset(get_args(SkillIssueCode))


def test_skill_issue_severity_is_derived_from_code():
    """`SkillIssue.severity` follows `code` through `SKILL_ISSUE_SEVERITY`."""
    issue = SkillIssue(
        skill_id=None,
        code="skill_reach_too_deep",
        message="reach exceeds the 6-level depth bound",
    )

    assert issue.severity == "rejection"
