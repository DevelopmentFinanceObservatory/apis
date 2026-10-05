"""Skills-index graph vocabulary, bounds, attachment rules, and id and path grammar."""

from __future__ import annotations

import re

from skills_index.types import (
    AttachmentKind,
    ConstraintPropertyAttachment,
    ConstraintValueAttachment,
    RejectedAttachment,
    TupleMemberAttachment,
)

# Hierarchy arcs used by CRS purpose codes, classes, and engine fixtures.
HIERARCHY_PROPERTIES: frozenset[str] = frozenset({"specializationOf", "isPartOf", "subClassOf"})

# The five identifying-tuple positions a StatisticalVariable is built
# from.
TUPLE_PROPERTIES: frozenset[str] = frozenset(
    {
        "populationType",
        "measuredProperty",
        "statType",
        "measurementQualifier",
        "measurementDenominator",
    }
)

# The incoming arc that marks a node as a constraint property in its own
# right (some class lists it in that class's own constraintProperties).
# Not itself a hierarchy or tuple property, so it needs its own name here.
CONSTRAINT_PROPERTIES_ARC = "constraintProperties"

# Exclude the universal root by name. Its descendants span every platform spec.
# Person stays eligible: 297 populationType subjects (10 September 2026) are below
# the breadth warning, while its 203 domainIncludes subjects are schema arcs.
EXCLUDED_ROOTS: frozenset[str] = frozenset({"Thing"})

# Incoming schema arcs describe permitted property or class use, separate from graph usage.
SCHEMA_ARC_PROPERTIES: frozenset[str] = frozenset({"domainIncludes", "rangeIncludes", "typeOf"}) | (
    HIERARCHY_PROPERTIES
)

# The walk probes one level past this bound and excludes deeper reaches.
# CRS purpose hierarchies are three levels deep (10 September 2026 probe).
DEFAULT_MAX_DEPTH = 4

# Exclude a reach that exceeds this node count. CRS purpose hierarchies span
# a few hundred nodes. Tuple-position reverse arcs reach 497 to 1000 subjects
# (10 September 2026 probe).
DEFAULT_MAX_NODES = 500

# The publisher-owned skill arc to ontology nodes: tuple members, constraint
# properties, and constraint values. Its targets are classified by their incoming
# arcs and reach downward through HIERARCHY_PROPERTIES.
SKILL_APPLIES_TO_ARC = "skillAppliesTo"

# The publisher-owned skill arc to Provenance nodes, one per dataset the skill
# explains as a whole. Its targets are `ProvenanceAttachment`s by arc alone,
# with no classification read and no hierarchy reach, and match by dcid equality.
SKILL_APPLIES_TO_PROVENANCE_ARC = "skillAppliesToProvenance"

# Reject a skill declaring more targets than this across both attachment arcs
# before graph reads. Each target counts as one hop-0 node, so the
# DEFAULT_MAX_NODES bound also caps this input list.
SKILLS_APPLIES_TO_MAX_TARGETS = DEFAULT_MAX_NODES

# Cap each hierarchy lead at 1000 results, matching the largest observed
# unbounded arc (`<-statType` on `measuredValue`, 10 September 2026).
DEFAULT_MAX_RESULTS_PER_LEAD = 1000

# Reaches above this distinct-StatVar count warn while keeping the skill eligible.
# The corpus gate handles regressions. A `typeOf` read found 600,000 subjects
# in 446 seconds (10 September 2026), and the mixer's page size is fixed at 1000.
SKILLS_BREADTH_WARN_STATVARS = 5000

# The breadth count stops at the first StatVar beyond the warning threshold.
SKILLS_BREADTH_COUNT_MAX_RESULTS = SKILLS_BREADTH_WARN_STATVARS + 1

# Cap the single AgentSkill roster read at 10,000, above the expected single-digit
# launch count and the roughly 20-skill point for a query-based catalogue. The
# mixer page size stays 1000 regardless of this result cap.
SKILLS_ROSTER_MAX_RESULTS = 10_000

# A skill whose text is past this limit is left out of the catalogue. The limit
# measures the decoded Markdown, and the publishing pipeline applies the same
# 256 KiB limit.
SKILL_BODY_MAX_BYTES = 256 * 1024

# Total retained skill content per build, separate from the per-skill limit.
# It counts decoded Markdown bytes.
# Two index generations and a datasets CSV may coexist in the 1 GiB container.
# This 128 MiB budget reserves an eighth for skill content.
SKILLS_CORPUS_CONTENT_BUDGET_BYTES = 128 * 1024 * 1024

# Rebuild `SharedSkillsIndex` from the graph every 15 minutes.
SKILLS_INDEX_REFRESH_SECONDS = 900.0

# Retry failed builds every 2 seconds until the first generation binds.
# Doors return 503 while `index` is `None`. Later failures keep serving the
# last bound generation and use the normal refresh interval.
SKILLS_INDEX_FIRST_BUILD_RETRY_INITIAL_SECONDS = 2.0

# Pre-bind retry delay doubles to 60 seconds. The service runs at most two
# instances, so jitter offers little benefit. The ceiling remains below the
# 15-minute refresh interval.
SKILLS_INDEX_FIRST_BUILD_RETRY_CEILING_SECONDS = 60.0


def classify_attachment(properties: frozenset[str]) -> AttachmentKind:
    """Classify observed incoming arcs, excluding schema arcs and the skill
    publisher's own arcs (`SKILL_PUBLISHER_ARCS`) first.

    Accept a target marked by `constraintProperties`. Otherwise accept one
    tuple position or one other property. Reject empty or ambiguous sets.
    Other properties are trusted as observed by the caller. Verification
    would require another graph read.
    """
    relevant = properties - SCHEMA_ARC_PROPERTIES - SKILL_PUBLISHER_ARCS
    if CONSTRAINT_PROPERTIES_ARC in relevant:
        return ConstraintPropertyAttachment()
    if len(relevant) == 1:
        (only,) = relevant
        if only in TUPLE_PROPERTIES:
            return TupleMemberAttachment(tuple_property=only)
        return ConstraintValueAttachment(property=only)
    if not relevant:
        return RejectedAttachment(reason="no accepted incoming arc")
    return RejectedAttachment(reason=f"ambiguous incoming arcs: {', '.join(sorted(relevant))}")


# Literal Skills API routes take precedence over `{skill_id}` routes, so their
# names such as `healthz` take precedence over templated skill routes.
RESERVED_SKILL_IDS: frozenset[str] = frozenset({"healthz"})

# An AgentSkill node's own type, read by build.build_index's roster walk
# (<-typeOf).
SKILL_TYPE = "AgentSkill"

# The publisher writes a skill's dcid as this prefix plus the skill id.
SKILL_DCID_PREFIX = "skill/"

# The Agent Skills name rule: lower-case alphanumeric runs joined by single
# hyphens. A skill id and every segment of a bundle-relative path match it.
SKILL_ID_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")

# The Agent Skills name rule's length bound for a skill id.
SKILL_ID_GRAMMAR_MAX_LENGTH = 64


def skill_id_from_dcid(dcid: str) -> str | None:
    """Return the skill id in a roster `dcid`, or `None` when the dcid is not
    `SKILL_DCID_PREFIX` plus a valid id.

    A valid id matches `SKILL_ID_PATTERN`, fits `SKILL_ID_GRAMMAR_MAX_LENGTH`, and
    is not one of `RESERVED_SKILL_IDS`.
    """
    skill_id = dcid.removeprefix(SKILL_DCID_PREFIX)
    if (
        skill_id == dcid
        or len(skill_id) > SKILL_ID_GRAMMAR_MAX_LENGTH
        or skill_id in RESERVED_SKILL_IDS
        or not SKILL_ID_PATTERN.fullmatch(skill_id)
    ):
        return None
    return skill_id


# A reference node's own type, mirroring SKILL_TYPE for the bundle files a skill
# declares.
REFERENCE_TYPE = "AgentSkillReference"

# Publisher-owned forward arc from a skill to its reference nodes. Read it with
# front matter because reverse arcs are writable by anyone.
SKILL_REFERENCES_ARC = "agentSkillReferences"

# The content arc: a skill's own inline body and a reference's inline
# content are both carried by this one arc.
CONTENT_ARC = "text"

# The arc on a reference node that holds its bundle-relative path.
REFERENCE_PATH_ARC = "name"

# The bundle-relative path of a skill's own text, first in its manifest.
SKILL_FILE_PATH = "SKILL.md"

# A node arc `decode.DecodedTextGraph` adds when a node's `text` value does not
# decode. It holds the decoder's explanation, and `text` is empty beside it.
# The name is internal to the index build, and no graph read requests it.
UNDECODABLE_CONTENT_ARC = "textUndecodable"

# A node arc `decode.DecodedTextGraph` adds when a node's stored `text` value is
# longer than the escaped form of its decoded byte limit, which it then refuses
# before decoding. It holds the explanation, and `text` is empty beside it.
# The name is internal to the index build, and no graph read requests it.
OVERSIZED_CONTENT_ARC = "textOversized"

# JSON escaping spells one decoded byte in at most six characters (`\u00XX`),
# so a stored `text` value is at most six times its decoded UTF-8 size.
JSON_ESCAPE_EXPANSION_FACTOR = 6

# A skill node's arcs, read in one batched node_arcs_many call in
# build.build_index's roster walk. The skill's front matter comes from the head of
# its CONTENT_ARC text, so this tuple omits the node's own `name` and `description` arcs.
SKILL_NODE_ARCS: tuple[str, ...] = (
    "url",
    CONTENT_ARC,
    SKILL_APPLIES_TO_ARC,
    SKILL_APPLIES_TO_PROVENANCE_ARC,
    SKILL_REFERENCES_ARC,
)

# The arcs the skill publisher writes from a skill node. A skill pointing at a
# node says nothing about how StatVars use that node, so none of them is evidence
# for an attachment kind. `Provenance <-rangeIncludes` returns
# `skillAppliesToProvenance` on the dev graph, which would otherwise classify a
# provenance named on `skillAppliesTo` as a constraint value.
# `classify_attachment` reads this name at call time.
SKILL_PUBLISHER_ARCS: frozenset[str] = frozenset(
    {SKILL_APPLIES_TO_ARC, SKILL_APPLIES_TO_PROVENANCE_ARC, SKILL_REFERENCES_ARC}
)

# Truncate and report reference lists above 50 entries before reading content.
# Bound the list before `node_arcs_many` reads every entry.
SKILL_REFERENCES_MAX_ENTRIES = 50

# References above 256 KiB of decoded content are dropped from the listing and
# reported. The publishing pipeline enforces the same Skills API contract limit.
SKILL_REFERENCE_MAX_BYTES = 256 * 1024

# The longest bundle-relative path. The index applies it to a reference's `name`,
# and the Skills API resource door imports it and returns HTTP 400 above it, so
# every path the index lists is one the door accepts.
RESOURCE_PATH_MAX_LENGTH = 512

# The one extension a bundle file's last path segment carries.
BUNDLE_PATH_EXTENSION = ".md"

# The directory every reference path sits under, with at least one segment below it.
REFERENCES_DIRECTORY = "references"


def reference_path_rejection(path: str) -> str | None:
    """Return why `path` is not a valid reference path, or `None` when it is.

    A valid path fits `RESOURCE_PATH_MAX_LENGTH`, holds no backslash, does not start
    with a slash, ends in `BUNDLE_PATH_EXTENSION`, and sits under
    `REFERENCES_DIRECTORY` with at least one segment below it. Each of its
    `/`-separated segments matches `SKILL_ID_PATTERN` once that extension is
    removed, which also rules out empty, `.` and `..` segments.
    """
    if not path:
        return "is empty"
    if len(path) > RESOURCE_PATH_MAX_LENGTH:
        return f"is {len(path)} characters, past the {RESOURCE_PATH_MAX_LENGTH}-character ceiling"
    if "\\" in path:
        return "carries a backslash"
    if path.startswith("/"):
        return "starts with a slash"
    if not path.endswith(BUNDLE_PATH_EXTENSION):
        return f"does not end in {BUNDLE_PATH_EXTENSION!r}"
    segments = path.removesuffix(BUNDLE_PATH_EXTENSION).split("/")
    if ".." in segments:
        return "carries a '..' segment"
    if segments[0] != REFERENCES_DIRECTORY or len(segments) < 2:
        return f"does not sit under {REFERENCES_DIRECTORY + '/'!r}"
    if not all(SKILL_ID_PATTERN.fullmatch(segment) for segment in segments):
        return f"has a segment outside {SKILL_ID_PATTERN.pattern!r}"
    return None
