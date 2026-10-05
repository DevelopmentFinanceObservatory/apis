"""Check that the independently declared search and skills `SubjectsCoverage` enums match by name and value.
Skills mirrors the enum because it cannot import `search.*`; coverage comparisons depend on value equality.
"""

from __future__ import annotations

from search.engine.clients.graph import SubjectsCoverage as EngineSubjectsCoverage
from skills_index.graph import SubjectsCoverage as PackageSubjectsCoverage


def test_subjects_coverage_members_match_by_name_and_value():
    engine_members = {member.name: member.value for member in EngineSubjectsCoverage}
    package_members = {member.name: member.value for member in PackageSubjectsCoverage}
    assert engine_members == package_members
