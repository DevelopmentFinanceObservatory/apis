"""Skills API wire codes include two distinct 404s, 15 warning codes, and shared platform codes.
Clients use `skill_not_found` to refresh the catalogue and `skill_resource_not_found` to
refresh one skill's listing. Catalogue wording matches `registries/codes.json` definitions.
"""

from __future__ import annotations

from dfo_platform import WarningSeverity
from dfo_platform.codes import PLATFORM_CODE_CATALOGUE
from skills_index import SKILL_ISSUE_SEVERITY, SkillIssueSeverity

SKILLS_CODE_CATALOGUE: dict[str, tuple[str, str]] = {
    "skill_not_found": (
        "Skill Not Found",
        "The requested skill is not published on this deployment.",
    ),
    "skill_resource_not_found": (
        "Skill Resource Not Found",
        "The named skill publishes no resource at the requested bundle-relative path.",
    ),
}

# Entries reach clients on HTTP 200.
# `WIRE_SEVERITY` maps index severities to wire values. Typed keys expose new values to
# type-checking.
WIRE_SEVERITY: dict[SkillIssueSeverity, WarningSeverity] = {
    "rejection": "degraded",
    "warning": "warn",
}

SKILLS_WARNING_CATALOGUE: dict[str, tuple[str, str, str]] = {
    "skills_roster_truncated": (
        "Skills Roster Truncated",
        "The read that enumerates published skills stopped at its result bound, so the "
        "catalogue may omit skills this deployment publishes.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skills_roster_truncated"]],
    ),
    "skills_roster_entry_unaddressable": (
        "Skills Roster Entry Unaddressable",
        "A published skill's dcid is unaddressable: it carries a character the Skills API's "
        "addressing grammar cannot express, it runs past the length that grammar admits, or "
        "it collides with a reserved platform meta name, so the index names it and leaves it "
        "out of the catalogue rather than publishing an identifier that answers nothing.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skills_roster_entry_unaddressable"]],
    ),
    "skill_content_absent": (
        "Skill Content Absent",
        "The skill's node carries no body, so the skill is left out of the catalogue.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skill_content_absent"]],
    ),
    "skill_content_too_large": (
        "Skill Content Too Large",
        "The skill's decoded body exceeds the size ceiling this deployment publishes, so the "
        "skill is left out of the catalogue.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skill_content_too_large"]],
    ),
    "skill_content_undecodable": (
        "Skill Content Undecodable",
        "The skill's stored body is not a valid escaped Markdown value, so the skill is left "
        "out of the catalogue.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skill_content_undecodable"]],
    ),
    "skill_frontmatter_invalid": (
        "Skill Front Matter Invalid",
        "The skill's body has front matter that is missing, cannot be parsed, lacks a string "
        "name or description, uses a YAML anchor or alias, or carries a name that differs "
        "from the skill's id, so the skill is left out of the catalogue.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skill_frontmatter_invalid"]],
    ),
    "skills_corpus_budget_exhausted": (
        "Skills Corpus Budget Exhausted",
        "The index build reached the corpus-wide budget for retained skill content before "
        "this skill's turn, so the skill is left out of the catalogue. Nothing is known to be "
        "wrong with the skill, and a later build may include it once the corpus fits within "
        "the budget.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skills_corpus_budget_exhausted"]],
    ),
    "skill_attachment_rejected": (
        "Skill Attachment Rejected",
        "One of the skill's declared attachment points is not a kind the platform accepts, so "
        "it contributes nothing to what the skill matches.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skill_attachment_rejected"]],
    ),
    "skill_attachment_kind_unconfirmed": (
        "Skill Attachment Kind Unconfirmed",
        "The read that classifies one of the skill's declared attachment points stopped at "
        "its bound before settling what kind it is, so the attachment contributes nothing to "
        "what the skill matches on this build and the skill's reach understates what it "
        "covers.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skill_attachment_kind_unconfirmed"]],
    ),
    "skill_build_unconfirmed": (
        "Skill Build Unconfirmed",
        "A read this skill's build depends on did not complete, whether it failed outright or "
        "ran out of its own budget. When the read is of the skill's own node, the skill is left "
        "out of the catalogue on this build and the warning, keyed by its dcid, joins the corpus "
        "warnings. Otherwise the skill stays served and is excluded from matching on this build. "
        "Nothing is known to be wrong with the skill, and a later build may include it.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skill_build_unconfirmed"]],
    ),
    "skill_reach_read_truncated": (
        "Skill Reach Read Truncated",
        "A read inside the skill's reach stopped at its result bound, so the reach "
        "understates what the skill covers.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skill_reach_read_truncated"]],
    ),
    "skill_reach_too_deep": (
        "Skill Reach Too Deep",
        "The skill's reach extends past the depth bound, so the skill is excluded from "
        "matching until its attachment points are narrowed.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skill_reach_too_deep"]],
    ),
    "skill_reach_too_large": (
        "Skill Reach Too Large",
        "The skill's reach holds more nodes than the bound allows, so the skill is excluded "
        "from matching until its attachment points are narrowed.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skill_reach_too_large"]],
    ),
    "skill_breadth_high": (
        "Skill Breadth High",
        "The skill's reach covers more served statistical variables than the breadth bound, "
        "so the skill matches very broadly and may be attached higher in the hierarchy than "
        "intended.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skill_breadth_high"]],
    ),
    "skill_resource_rejected": (
        "Skill Resource Rejected",
        "A resource the skill declares is not published: its identifier does not sit under "
        "the skill's own, its path breaches the addressing rules, it carries no content, it "
        "breaches the size ceiling this deployment publishes, or it does not declare the "
        "resource type the platform requires.",
        WIRE_SEVERITY[SKILL_ISSUE_SEVERITY["skill_resource_rejected"]],
    ),
}

# Shared codes cover pre-build unavailability (503) and overlong path parameters (400).
# The 503 avoids caching an empty catalogue.
# Problem responses use `no-store`.
SHARED_PLATFORM_CODES: frozenset[str] = frozenset(
    {
        "service_unavailable",
        "request_invalid",
    }
)
if not frozenset(PLATFORM_CODE_CATALOGUE) >= SHARED_PLATFORM_CODES:
    raise AssertionError(
        f"SHARED_PLATFORM_CODES names code(s) "
        f"{sorted(SHARED_PLATFORM_CODES - frozenset(PLATFORM_CODE_CATALOGUE))} absent from "
        "PLATFORM_CODE_CATALOGUE"
    )
