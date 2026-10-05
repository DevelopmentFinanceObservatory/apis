"""Dataclasses for skills, attachments, graph reach, and build results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

type FrontMatterValue = str | list[FrontMatterValue] | dict[str, FrontMatterValue]
"""A value in front matter parsed by YAML's base loader, which reads every scalar
as a string."""


@dataclass(frozen=True, slots=True)
class SkillFrontMatter:
    """A skill's declared identity. `name` and `description` come from the front
    matter of its `SKILL.md`, and `content_version` is the `sha256:` digest of its
    manifest."""

    id: str
    name: str
    description: str
    content_version: str


@dataclass(frozen=True, slots=True)
class TupleMemberAttachment:
    """The target is a value at one identifying-tuple position."""

    tuple_property: str


@dataclass(frozen=True, slots=True)
class ConstraintPropertyAttachment:
    """The target is declared in a class's `constraintProperties` list."""


@dataclass(frozen=True, slots=True)
class ConstraintValueAttachment:
    """The target is a value of the named constraint property."""

    property: str


@dataclass(frozen=True, slots=True)
class ProvenanceAttachment:
    """The target is a Provenance node, declared on `skillAppliesToProvenance`.

    The arc alone sets this kind. The target reaches only itself, and a spec
    matches it when the spec's selected provenance facet names the same dcid."""


@dataclass(frozen=True, slots=True)
class RejectedAttachment:
    """Observed incoming arcs identify an unsupported attachment kind. `reason` names why."""

    reason: str


@dataclass(frozen=True, slots=True)
class UnconfirmedAttachment:
    """Classification stopped at `budget_exhausted` before confirming the target's
    kind. The target is omitted from matching, and its skill's reach may be
    understated. Its kind remains unconfirmed."""


AttachmentKind = (
    TupleMemberAttachment
    | ConstraintPropertyAttachment
    | ConstraintValueAttachment
    | ProvenanceAttachment
    | RejectedAttachment
    | UnconfirmedAttachment
)


@dataclass(frozen=True, slots=True)
class Attachment:
    """A declared attachment dcid and its classified kind."""

    dcid: str
    kind: AttachmentKind


@dataclass(frozen=True, slots=True)
class ReachedNode:
    """A reached node, with distance from its attachment and the arc used on its last hop."""

    dcid: str
    hop: int
    via_property: str | None


@dataclass(frozen=True, slots=True)
class SkillReach:
    """Combined reach across attachments. A bound breach sets `excluded` and empties
    `nodes`, preventing a partial reach from understating skill coverage.

    `capped` is true when a hierarchy page stopped at its per-lead result bound,
    so `nodes` may miss descendants the graph holds. An excluded reach is never
    capped."""

    skill_id: str
    nodes: tuple[ReachedNode, ...]
    excluded: bool
    capped: bool = False


SkillIssueCode = Literal[
    "skills_roster_truncated",
    "skills_roster_entry_unaddressable",
    "skill_attachment_rejected",
    "skill_attachment_kind_unconfirmed",
    "skill_build_unconfirmed",
    "skill_reach_read_truncated",
    "skill_reach_too_deep",
    "skill_reach_too_large",
    "skill_breadth_high",
    "skill_resource_rejected",
    "skill_content_absent",
    "skill_content_too_large",
    "skill_content_undecodable",
    "skill_frontmatter_invalid",
    "skills_corpus_budget_exhausted",
]
"""Stable codes for issues reported by the skills index."""

SkillIssueSeverity = Literal["rejection", "warning"]
"""Stable severity values for `SkillIssue`."""

SKILL_ISSUE_SEVERITY: Mapping[SkillIssueCode, SkillIssueSeverity] = MappingProxyType(
    {
        "skills_roster_truncated": "warning",
        "skills_roster_entry_unaddressable": "rejection",
        "skill_attachment_rejected": "rejection",
        "skill_attachment_kind_unconfirmed": "warning",
        "skill_build_unconfirmed": "rejection",
        "skill_reach_read_truncated": "warning",
        "skill_reach_too_deep": "rejection",
        "skill_reach_too_large": "rejection",
        "skill_breadth_high": "warning",
        "skill_resource_rejected": "rejection",
        "skill_content_absent": "warning",
        "skill_content_too_large": "warning",
        "skill_content_undecodable": "warning",
        "skill_frontmatter_invalid": "warning",
        "skills_corpus_budget_exhausted": "warning",
    }
)
"""One immutable severity per issue code, shared by the API wire catalogue."""

INDEX_INCOMPLETE_CODES: frozenset[SkillIssueCode] = frozenset(
    {
        # The roster read stopped at its bound.
        "skills_roster_truncated",
        # A read raised or stopped at its page budget, so a skill is unknown or
        # served excluded from matching.
        "skill_build_unconfirmed",
        # A classification read stopped at its page budget, so an attachment
        # contributes nothing and its skill's reach is understated.
        "skill_attachment_kind_unconfirmed",
        # The corpus budget was spent before a valid skill's turn.
        "skills_corpus_budget_exhausted",
    }
)
"""The issue codes that mark a skill left out of the catalogue or served partial
for a budget or an unconfirmed read. The completeness rule,
`index_may_omit_published_skills`, is these codes plus a capped hierarchy read
(`SkillReach.capped`). A capped read shares `skill_reach_read_truncated` with a
StatVar count floor, which omits nothing search can match, so the reach itself
carries the event."""


def index_may_omit_published_skills(
    issues: Iterable[SkillIssue], skills: Mapping[str, SkillEntry]
) -> bool:
    """Whether a built index may lack a published skill or serve one whose reach
    is understated: any issue code in `INDEX_INCOMPLETE_CODES`, or any served
    skill whose reach is capped."""
    return any(issue.code in INDEX_INCOMPLETE_CODES for issue in issues) or any(
        entry.reach.capped for entry in skills.values()
    )


@dataclass(frozen=True, slots=True)
class SkillIssue:
    """A corpus- or skill-level issue. Severity is derived from `code`. Rejections
    make a reach or resource unusable. Warnings report conditions and, for a skill
    left out of the catalogue, why. `message` carries dcids and numbers.

    `skill_id` is the id of a skill in the catalogue for an issue about that skill,
    the roster dcid of a skill left out of the catalogue, or `None` for an issue
    about the whole corpus."""

    skill_id: str | None
    code: SkillIssueCode
    message: str

    @property
    def severity(self) -> SkillIssueSeverity:
        return SKILL_ISSUE_SEVERITY[self.code]


@dataclass(frozen=True, slots=True)
class SkillFile:
    """One bundle file: its bundle-relative path, the byte size of its decoded
    UTF-8 `content`, and `sha256:` plus the lower-case hex SHA-256 of those bytes."""

    path: str
    size: int
    digest: str
    content: str


@dataclass(frozen=True, slots=True)
class SkillEntry:
    """A skill in the catalogue.

    `front_matter` holds the skill's identity and content version. `frontmatter`
    is the whole mapping parsed from the head of `content_text`, every key the
    author wrote. `url` is the skill node's `url` arc, the human-authored source
    directory, or `None` when absent. Attachments may be empty, and `reach`
    combines their descendants. StatVar count may stop at
    `SKILLS_BREADTH_COUNT_MAX_RESULTS`. `files` is the manifest: `SKILL.md` first,
    then each accepted reference in path order."""

    front_matter: SkillFrontMatter
    frontmatter: dict[str, FrontMatterValue]
    url: str | None
    attachments: tuple[Attachment, ...]
    reach: SkillReach
    statvar_count: int
    statvar_count_truncated: bool
    files: tuple[SkillFile, ...]

    @property
    def content_text(self) -> str:
        """The decoded `SKILL.md`, front matter included."""
        return self.files[0].content

    @property
    def attached(self) -> bool:
        """Whether search can name this skill on this build: its reach names at
        least one node and it is not excluded."""
        return bool(self.reach.nodes) and not self.reach.excluded


@dataclass(frozen=True, slots=True)
class SkillMatch:
    """A node-to-skill match with hop distance and StatVar breadth. `breadth_truncated`
    marks breadth as a lower bound. At equal hops, matching sorts exact counts
    first. `via_provenance` marks a node reached through a `ProvenanceAttachment`.
    A whole dataset is wider than any ontology node inside it, so matching ranks it
    after every ontology match."""

    skill_id: str
    hop: int
    breadth: int
    breadth_truncated: bool
    via_provenance: bool = False


@dataclass(frozen=True, slots=True)
class SkillsIndex:
    """The built skill records, dcid-to-match index, issues, and completeness flag.

    `may_omit_published_skills` is true when `index_may_omit_published_skills`
    holds for the build. Search uses the flag for its
    `skills_index_incomplete` warning. Callers provide the flag explicitly to
    state whether completeness is known."""

    skills: dict[str, SkillEntry]
    node_to_skills: dict[str, tuple[SkillMatch, ...]]
    issues: tuple[SkillIssue, ...]
    may_omit_published_skills: bool
