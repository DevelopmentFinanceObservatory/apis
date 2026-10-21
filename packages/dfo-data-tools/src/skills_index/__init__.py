"""Public API for building, matching, and sharing the skills index.

`build_index` classifies skill attachments, expands their graph reach, and
reads accepted references. `match_dcids` queries the resulting index. Shared
accessors manage its process-wide holder.
"""

from __future__ import annotations

from skills_index.build import build_index
from skills_index.config import (
    CONSTRAINT_PROPERTIES_ARC,
    CONTENT_ARC,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_NODES,
    DEFAULT_MAX_RESULTS_PER_LEAD,
    EXCLUDED_ROOTS,
    HIERARCHY_PROPERTIES,
    REFERENCE_TYPE,
    SCHEMA_ARC_PROPERTIES,
    SKILL_FILE_PATH,
    SKILL_NODE_ARCS,
    SKILL_REFERENCES_ARC,
    SKILL_TYPE,
    SKILLS_BREADTH_COUNT_MAX_RESULTS,
    SKILLS_BREADTH_WARN_STATVARS,
    TUPLE_PROPERTIES,
    classify_attachment,
)
from skills_index.expand import expand_skill
from skills_index.graph import SkillsGraph, SubjectsPage
from skills_index.match import match_dcids
from skills_index.shared import SharedSkillsIndex, acquire_shared_index, release_shared_index
from skills_index.types import (
    SKILL_ISSUE_SEVERITY,
    Attachment,
    AttachmentKind,
    ConstraintPropertyAttachment,
    ConstraintValueAttachment,
    FrontMatterValue,
    ProvenanceAttachment,
    ReachedNode,
    RejectedAttachment,
    SkillEntry,
    SkillFile,
    SkillFrontMatter,
    SkillIssue,
    SkillIssueCode,
    SkillIssueSeverity,
    SkillMatch,
    SkillReach,
    SkillsIndex,
    TupleMemberAttachment,
    UnconfirmedAttachment,
)

__all__ = [
    "CONSTRAINT_PROPERTIES_ARC",
    "CONTENT_ARC",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_NODES",
    "DEFAULT_MAX_RESULTS_PER_LEAD",
    "EXCLUDED_ROOTS",
    "HIERARCHY_PROPERTIES",
    "REFERENCE_TYPE",
    "SCHEMA_ARC_PROPERTIES",
    "SKILLS_BREADTH_COUNT_MAX_RESULTS",
    "SKILLS_BREADTH_WARN_STATVARS",
    "SKILL_FILE_PATH",
    "SKILL_ISSUE_SEVERITY",
    "SKILL_NODE_ARCS",
    "SKILL_REFERENCES_ARC",
    "SKILL_TYPE",
    "TUPLE_PROPERTIES",
    "Attachment",
    "AttachmentKind",
    "ConstraintPropertyAttachment",
    "ConstraintValueAttachment",
    "FrontMatterValue",
    "ProvenanceAttachment",
    "ReachedNode",
    "RejectedAttachment",
    "SharedSkillsIndex",
    "SkillEntry",
    "SkillFile",
    "SkillFrontMatter",
    "SkillIssue",
    "SkillIssueCode",
    "SkillIssueSeverity",
    "SkillMatch",
    "SkillReach",
    "SkillsGraph",
    "SkillsIndex",
    "SubjectsPage",
    "TupleMemberAttachment",
    "UnconfirmedAttachment",
    "acquire_shared_index",
    "build_index",
    "classify_attachment",
    "expand_skill",
    "match_dcids",
    "release_shared_index",
]
