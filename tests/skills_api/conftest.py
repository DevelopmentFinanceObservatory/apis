"""Graph-shaped catalogue entries and a test client for the Skills doors."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi.testclient import TestClient

from dfo_platform.app import create_app
from skills_api.app import skills_surface
from skills_index.frontmatter import read_front_matter
from skills_index.manifest import build_manifest, content_version, skill_file
from skills_index.types import (
    ReachedNode,
    SkillEntry,
    SkillFrontMatter,
    SkillIssue,
    SkillReach,
    SkillsIndex,
)

ODA_CONCEPTS_DESCRIPTION = "Core ODA concepts and how the platform measures them."


def skill_markdown(skill_id: str, *, description: str, body: str = "# Concepts\n\nBody.\n") -> str:
    """The `SKILL.md` text the publisher writes: front matter, then the body."""
    return f"---\nname: {skill_id}\ndescription: {description}\nlicense: CC-BY-4.0\n---\n\n{body}"


def skill_entry(
    skill_id: str = "oda-concepts",
    *,
    description: str = ODA_CONCEPTS_DESCRIPTION,
    body: str = "# Concepts\n\nBody.\n",
    references: Mapping[str, str] | None = None,
    url: str | None = None,
    attached: bool = False,
    excluded: bool = False,
    statvar_count: int = 0,
    statvar_count_truncated: bool = False,
) -> SkillEntry:
    """A catalogue entry built the way `skills_index.build` builds one: front matter
    parsed from the text, references in path order, and a real manifest.
    `attached` gives the reach one node; `excluded` empties it."""
    content_text = skill_markdown(skill_id, description=description, body=body)
    parsed = read_front_matter(content_text, skill_id=skill_id)
    files = build_manifest(
        content_text,
        tuple(skill_file(path, content) for path, content in sorted((references or {}).items())),
    )
    nodes = (
        (ReachedNode(dcid="ODA", hop=0, via_property=None),) if attached and not excluded else ()
    )
    return SkillEntry(
        front_matter=SkillFrontMatter(
            id=skill_id,
            name=parsed.name,
            description=parsed.description,
            content_version=content_version(files),
        ),
        frontmatter=parsed.mapping,
        url=url,
        attachments=(),
        reach=SkillReach(skill_id=skill_id, nodes=nodes, excluded=excluded),
        statvar_count=statvar_count,
        statvar_count_truncated=statvar_count_truncated,
        files=files,
    )


def index_of(*entries: SkillEntry, issues: tuple[SkillIssue, ...] = ()) -> SkillsIndex:
    """An index generation holding `entries`, keyed by id in the order given."""
    return SkillsIndex(
        skills={entry.front_matter.id: entry for entry in entries},
        node_to_skills={},
        issues=issues,
        may_omit_published_skills=False,
    )


class NoGraphSharedIndex:
    """A stand-in for `skills_index.SharedSkillsIndex` exposing exactly the one member a
    door may read: `.index`. Any other attribute would be a graph read no door makes, so
    `__getattr__`, which fires only once normal lookup fails, raises.
    """

    def __init__(self, index: SkillsIndex | None) -> None:
        self.index = index

    def __getattr__(self, name: str) -> object:
        raise AssertionError(
            f"a door read {name!r} off the shared index holder; it should read only '.index'"
        )


def surface_client(holder: NoGraphSharedIndex) -> TestClient:
    """A client for the Skills surface serving from `holder`."""
    return TestClient(create_app([skills_surface(skills_index=holder)]))


def client_for(index: SkillsIndex | None) -> TestClient:
    """A client for the Skills surface serving `index`."""
    return surface_client(NoGraphSharedIndex(index))
