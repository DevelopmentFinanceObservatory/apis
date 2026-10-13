"""Wire models shared by the Skills API doors, using Data Commons field names."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath

from pydantic import Field

from dfo_platform import WarningSeverity
from dfo_platform.wire import ResponseModel
from skills_api.codes import WIRE_SEVERITY
from skills_index import (
    FrontMatterValue,
    SkillEntry,
    SkillFile,
    SkillIssue,
)


class SkillWarning(ResponseModel):
    """One corpus-level or skill-level issue, on the wire. `skill_id` is the
    catalogue id for an issue about a served skill, the roster dcid for a skill
    left out of the catalogue, or `None` for an issue about the whole corpus.
    Only an issue about a served skill rides on that skill's entry. The other two
    are corpus warnings on the list door."""

    code: str = Field(description="stable machine code from the platform's code registry")
    severity: WarningSeverity = Field(description="the severity level")
    message: str = Field(description="human-readable explanation of the warning")
    skill_id: str | None = Field(
        default=None,
        description="the served skill's id, the dcid of a skill left out of the "
        "catalogue, or null for a corpus-level warning",
    )


def skill_warning_from_issue(issue: SkillIssue) -> SkillWarning:
    """Convert an index issue to its wire warning and platform severity."""
    return SkillWarning(
        code=issue.code,
        severity=WIRE_SEVERITY[issue.severity],
        message=issue.message,
        skill_id=issue.skill_id,
    )


def split_skill_warnings(
    issues: Iterable[SkillIssue], skills: Mapping[str, object]
) -> tuple[list[SkillWarning], dict[str, list[SkillWarning]]]:
    """Split issues into corpus and per-skill warnings.

    An issue whose `skill_id` names no served skill, such as a left-out skill's
    dcid, joins the corpus warnings.
    """
    corpus_warnings: list[SkillWarning] = []
    skill_warnings: dict[str, list[SkillWarning]] = {}
    for issue in issues:
        warning = skill_warning_from_issue(issue)
        if issue.skill_id is None or issue.skill_id not in skills:
            corpus_warnings.append(warning)
        else:
            skill_warnings.setdefault(issue.skill_id, []).append(warning)
    return corpus_warnings, skill_warnings


class SkillFileListing(ResponseModel):
    """One file in a skill's manifest. The client fetches it at the resources door
    by the skill id and this `path`."""

    path: str = Field(description="the file's path, relative to its skill's own bundle")
    size: int = Field(description="the byte size of the file's UTF-8 content")
    digest: str = Field(description="`sha256:` plus the lower-case hex SHA-256 of that content")
    media_type: str = Field(description="the file's media type, from its extension")


# The index admits only `.md` bundle files, so a path outside this map is a bug.
_MEDIA_TYPES: dict[str, str] = {".md": "text/markdown"}


def media_type(path: str) -> str:
    """Return the media type of a bundle file from its extension."""
    return _MEDIA_TYPES[PurePosixPath(path).suffix]


def skill_file_listing(file: SkillFile) -> SkillFileListing:
    """Convert a manifest entry to its wire listing."""
    return SkillFileListing(
        path=file.path, size=file.size, digest=file.digest, media_type=media_type(file.path)
    )


class SkillSummary(ResponseModel):
    """One skill's list entry: front matter, attachment and breadth figures, own
    warnings, and manifest. `SkillBody` adds the skill's text.

    `url` is the skill's source directory for a human reader. The platform never
    fetches it.
    """

    id: str = Field(description="the skill's id, its dcid without the `skill/` prefix")
    name: str = Field(description="the front matter's `name`, equal to `id`")
    description: str = Field(description="the front matter's `description`")
    frontmatter: dict[str, FrontMatterValue] = Field(
        description="the whole parsed front matter, every scalar a string"
    )
    content_version: str = Field(
        description="`sha256:` digest over the manifest's paths and digests, in order"
    )
    url: str | None = Field(
        description="the skill's source directory for a human reader, never fetched here"
    )
    attached: bool = Field(
        description="true when the skill's reach names at least one node on this build, "
        "so search can name it"
    )
    excluded: bool = Field(
        description="true when search cannot match this skill on this build: its reach "
        "broke a bound or a read the build depended on did not complete"
    )
    statvar_count: int = Field(
        description="the distinct StatVars this skill's downward reach carries. A "
        "provenance attachment adds nothing to the count, so a skill attached only to "
        "provenances reports 0 while covering every series of those datasets"
    )
    statvar_count_truncated: bool = Field(
        description="true when statvar_count stopped at this deployment's count bound "
        "rather than exhausting the reach"
    )
    warnings: list[SkillWarning] = Field(description="this skill's own warnings")
    resources: list[SkillFileListing] = Field(
        description="the skill's manifest: `SKILL.md` first, then its references in path order"
    )


def skill_summary(skill_id: str, entry: SkillEntry, warnings: list[SkillWarning]) -> SkillSummary:
    """Build the list entry for one catalogue skill."""
    return _skill_model(SkillSummary, skill_id, entry, warnings)


def skill_body(skill_id: str, entry: SkillEntry, warnings: list[SkillWarning]) -> SkillBody:
    """Build the skill door's response for one catalogue skill."""
    return _skill_model(SkillBody, skill_id, entry, warnings, content_text=entry.content_text)


def _skill_model[M: SkillSummary](
    model: type[M],
    skill_id: str,
    entry: SkillEntry,
    warnings: list[SkillWarning],
    **text: str,
) -> M:
    return model(
        id=skill_id,
        name=entry.front_matter.name,
        description=entry.front_matter.description,
        frontmatter=entry.frontmatter,
        content_version=entry.front_matter.content_version,
        url=entry.url,
        attached=entry.attached,
        excluded=entry.reach.excluded,
        statvar_count=entry.statvar_count,
        statvar_count_truncated=entry.statvar_count_truncated,
        warnings=warnings,
        resources=[skill_file_listing(file) for file in entry.files],
        **text,
    )


class SkillBody(SkillSummary):
    """The skill door's response: the list entry plus the skill's text."""

    content_text: str = Field(description="`SKILL.md` verbatim, front matter included")


class SkillResourceContent(ResponseModel):
    """The resources door's response: one manifest file and its text.

    Carries `skill_id` and `path` so a client that fires several requests
    concurrently can tell which response answers which request.
    """

    skill_id: str = Field(description="the skill this file belongs to")
    path: str = Field(description="the file's path, relative to its skill's own bundle")
    size: int = Field(description="the byte size of the file's UTF-8 content")
    digest: str = Field(description="`sha256:` plus the lower-case hex SHA-256 of that content")
    media_type: str = Field(description="the file's media type, from its extension")
    content_text: str = Field(description="the file's text")


class SkillListResponse(ResponseModel):
    """`GET /api/skills`'s response body: every served skill's entry, ordered by
    id, plus the corpus warnings. A per-skill warning rides on that skill's own
    entry."""

    skills: list[SkillSummary] = Field(description="every served skill, ordered by id")
    warnings: list[SkillWarning] = Field(
        description="corpus warnings, including one per skill left out of the catalogue"
    )
