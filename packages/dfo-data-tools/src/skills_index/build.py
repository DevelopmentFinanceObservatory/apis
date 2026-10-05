"""Builds a skills index from graph `AgentSkill` nodes and their attachments.

The roster is capped at `SKILLS_ROSTER_MAX_RESULTS`. A dcid that is not `skill/`
plus a valid skill id is left out. Skill nodes are read in response-sized
chunks. A failed chunk leaves its skills unknown with `skill_build_unconfirmed`,
while failure of every chunk aborts the build. A skill enters the catalogue
when its `text` is present, decodes, fits `SKILL_BODY_MAX_BYTES`, and opens
with front matter whose `name` is the skill id and whose `description` is a
string. A skill failing any of these is left out with the code of the rule it
broke. Every graph read goes through `DecodedTextGraph`, so bodies,
references, and their byte limits measure decoded Markdown. Only chunk sizing
and the pre-decode ceiling count stored characters: a stored body or reference
longer than `JSON_ESCAPE_EXPANSION_FACTOR` times its decoded limit is refused
before decoding, with the same issue an oversized decoded one gets. A skill
attaches through two arcs: `skillAppliesTo` to ontology nodes and
`skillAppliesToProvenance` to Provenance nodes. Missing either is valid. A skill
declaring more than `SKILLS_APPLIES_TO_MAX_TARGETS` targets across both is
excluded before attachment reads.

The corpus budget applies per skill in roster order and bounds retained
bytes. A skill reached after the budget is spent is left out with
`skills_corpus_budget_exhausted`, and its reference dcids are withheld from
every reference read. A skill's `agentSkillReferences` list is capped at
`SKILL_REFERENCES_MAX_ENTRIES`. Reference bytes count against the same budget.
A reference dcid outside the skill's `skill/<id>/` prefix, an invalid or
duplicated `name` path, missing or undecodable content, an oversized
reference, and a failed chunk produce `skill_resource_rejected`. Failure of
every reference chunk aborts.

Each skill's manifest lists `SKILL.md` and its references with the size and
digest of their decoded bytes, and its content version digests the manifest.

A `skillAppliesToProvenance` target is a `ProvenanceAttachment` by its arc. It
needs no read, reaches only itself at hop 0, and adds no StatVars to the count.
For `skillAppliesTo` targets, six attachment properties are probed in batches.
Unresolved targets use a type-then-range lookup. Incomplete pages yield
`UnconfirmedAttachment`. Unsafe graph property names are rejected. Attachments
sharing a counting property expand together. StatVar counts are deduplicated and
capped at `SKILLS_BREADTH_COUNT_MAX_RESULTS`. `max_nodes` bounds the total across
all counting-property groups and provenance attachments. Constraint-property
attachments count through their admissible values. A classification read that
raises fails every target in its batch, and a reach or StatVar count read that
raises fails its skill. Each such skill stays served with
`skill_build_unconfirmed`, an excluded empty reach, and a count of 0. The build
aborts when every classification read raises, or when reach and count reads
were made and every one raised, so a graph outage leaves the previous
generation serving. A skill whose reach needs no read, such as a
provenance-only skill, neither masks nor causes an abort.

The index is incomplete, and search warns `skills_index_incomplete`, when
`index_may_omit_published_skills` holds: an issue code in
`INDEX_INCOMPLETE_CODES` marks a skill left out or partial because a read or
the corpus stopped at a budget or a read did not confirm, and a capped
hierarchy read marks a served reach understated.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from materialise.graph import reject_unsafe_property_token
from skills_index.config import (
    CONSTRAINT_PROPERTIES_ARC,
    CONTENT_ARC,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_NODES,
    DEFAULT_MAX_RESULTS_PER_LEAD,
    JSON_ESCAPE_EXPANSION_FACTOR,
    OVERSIZED_CONTENT_ARC,
    REFERENCE_PATH_ARC,
    REFERENCE_TYPE,
    RESERVED_SKILL_IDS,
    SKILL_APPLIES_TO_ARC,
    SKILL_APPLIES_TO_PROVENANCE_ARC,
    SKILL_BODY_MAX_BYTES,
    SKILL_DCID_PREFIX,
    SKILL_ID_GRAMMAR_MAX_LENGTH,
    SKILL_NODE_ARCS,
    SKILL_REFERENCE_MAX_BYTES,
    SKILL_REFERENCES_ARC,
    SKILL_REFERENCES_MAX_ENTRIES,
    SKILL_TYPE,
    SKILLS_APPLIES_TO_MAX_TARGETS,
    SKILLS_BREADTH_COUNT_MAX_RESULTS,
    SKILLS_BREADTH_WARN_STATVARS,
    SKILLS_CORPUS_CONTENT_BUDGET_BYTES,
    SKILLS_ROSTER_MAX_RESULTS,
    TUPLE_PROPERTIES,
    UNDECODABLE_CONTENT_ARC,
    classify_attachment,
    reference_path_rejection,
    skill_id_from_dcid,
)
from skills_index.decode import DecodedTextGraph
from skills_index.expand import expand_skill
from skills_index.frontmatter import FrontMatterInvalid, ParsedFrontMatter, read_front_matter
from skills_index.graph import SkillsGraph, SubjectsCoverage, SubjectsPage
from skills_index.manifest import build_manifest, content_version, skill_file
from skills_index.types import (
    Attachment,
    AttachmentKind,
    ConstraintPropertyAttachment,
    ConstraintValueAttachment,
    ProvenanceAttachment,
    ReachedNode,
    RejectedAttachment,
    SkillEntry,
    SkillFile,
    SkillFrontMatter,
    SkillIssue,
    SkillIssueCode,
    SkillMatch,
    SkillReach,
    SkillsIndex,
    TupleMemberAttachment,
    UnconfirmedAttachment,
    index_may_omit_published_skills,
)

logger = logging.getLogger(__name__)

_CLASSIFICATION_PROBE_PROPERTIES: tuple[str, ...] = tuple(
    sorted(TUPLE_PROPERTIES | {CONSTRAINT_PROPERTIES_ARC})
)
# One confirmed subject establishes an incoming arc, regardless of count.
_CLASSIFICATION_PROBE_MAX_RESULTS = 1

# Mixer reads fail above roughly 32 MiB. Chunk sizing stays below this cap
# with headroom for keys, other arcs, and JSON framing.
_MIXER_RESPONSE_CAP_BYTES = 32 * 1024 * 1024

# Half the response cap and the 6x JSON escape factor leave headroom for
# node keys, other arcs, and response framing.
_FRONT_MATTER_CHUNK_SIZE = (_MIXER_RESPONSE_CAP_BYTES // 2) // (
    SKILL_BODY_MAX_BYTES * JSON_ESCAPE_EXPANSION_FACTOR
)

# Reference chunks use the same half-cap and 6x JSON escape headroom as
# front-matter chunks.
_REFERENCE_CHUNK_SIZE = (_MIXER_RESPONSE_CAP_BYTES // 2) // (
    SKILL_REFERENCE_MAX_BYTES * JSON_ESCAPE_EXPANSION_FACTOR
)

_BUDGET_WHY = (
    "the build spent its corpus-wide budget for retained skill content before this "
    "skill's turn; nothing is known to be wrong with the skill, and a later build may "
    "include it once the corpus fits within the budget"
)


@dataclass(frozen=True, slots=True)
class _SkillText:
    """A skill's decoded `SKILL.md` text and the front matter parsed from it."""

    text: str
    front_matter: ParsedFrontMatter


@dataclass
class _IndexBuildState:
    """Mutable state for one `build_index` call."""

    issues: list[SkillIssue] = field(default_factory=list)
    # Ids of the skills still headed for the catalogue, in roster order. Every
    # per-skill map below is keyed by id.
    skill_ids: tuple[str, ...] = ()
    arcs_by_skill: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)
    text_by_skill: dict[str, _SkillText] = field(default_factory=dict)
    budget_baseline_by_skill: dict[str, int] = field(default_factory=dict)
    retained_content_bytes: int = 0
    # `skillAppliesTo` targets, classified by `_classify_targets`.
    targets_by_skill: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # `skillAppliesToProvenance` targets, which need no classification.
    provenance_targets_by_skill: dict[str, tuple[str, ...]] = field(default_factory=dict)
    reference_dcids_by_skill: dict[str, tuple[str, ...]] = field(default_factory=dict)
    all_targets: set[str] = field(default_factory=set)
    oversized_applies_to: set[str] = field(default_factory=set)
    skill_by_reference_dcid: dict[str, str] = field(default_factory=dict)
    references_by_skill: dict[str, _SkillReferences] = field(default_factory=dict)


def _leave_out(state: _IndexBuildState, skill_id: str, *, code: SkillIssueCode, why: str) -> None:
    """Records that `skill_id` stays out of the catalogue, under its roster dcid.

    Issues already recorded under its id are dropped, since no entry will carry
    them. The caller removes the id from `state.skill_ids`.
    """
    dcid = f"{SKILL_DCID_PREFIX}{skill_id}"
    state.issues = [issue for issue in state.issues if issue.skill_id != skill_id]
    state.issues.append(
        SkillIssue(
            skill_id=dcid,
            code=code,
            message=f"{dcid} is left out of the catalogue: {why}",
        )
    )


def _read_roster(state: _IndexBuildState, *, graph: SkillsGraph) -> None:
    """Reads the capped `AgentSkill` roster and keeps dcids that carry a valid id.

    Records roster truncation and each rejected dcid in `state.issues`.
    """
    roster_page = graph.node_subjects_many(
        (SKILL_TYPE,), property="typeOf", max_results=SKILLS_ROSTER_MAX_RESULTS
    )[SKILL_TYPE]
    if roster_page.coverage != SubjectsCoverage.complete:
        state.issues.append(
            SkillIssue(
                skill_id=None,
                code="skills_roster_truncated",
                message=f"<-typeOf roster read truncated at {SKILLS_ROSTER_MAX_RESULTS} results",
            )
        )
    skill_ids = []
    for dcid in roster_page.dcids:
        if (skill_id := skill_id_from_dcid(dcid)) is not None:
            skill_ids.append(skill_id)
        else:
            state.issues.append(
                SkillIssue(
                    skill_id=dcid,
                    code="skills_roster_entry_unaddressable",
                    message=(
                        f"{dcid!r} is unaddressable: it does not start with "
                        f"{SKILL_DCID_PREFIX!r}, or the rest is not a skill id "
                        f"(lower-case alphanumeric runs joined by single hyphens, at most "
                        f"{SKILL_ID_GRAMMAR_MAX_LENGTH} characters) or is a reserved "
                        f"platform meta name ({', '.join(sorted(RESERVED_SKILL_IDS))}), so "
                        "the index names it and leaves it out of the catalogue rather than "
                        "publishing an identifier that answers nothing"
                    ),
                )
            )
    state.skill_ids = tuple(skill_ids)


def _read_skill_arcs(
    skill_ids: Sequence[str], *, graph: SkillsGraph
) -> dict[str, dict[str, tuple[str, ...]] | None]:
    """Reads the `SKILL_NODE_ARCS` of skills named by id, keyed by id."""
    arcs = graph.node_arcs_many(
        [f"{SKILL_DCID_PREFIX}{skill_id}" for skill_id in skill_ids],
        properties=SKILL_NODE_ARCS,
    )
    return {
        skill_id: arcs[f"{SKILL_DCID_PREFIX}{skill_id}"]
        for skill_id in skill_ids
        if f"{SKILL_DCID_PREFIX}{skill_id}" in arcs
    }


def _read_front_matter(state: _IndexBuildState, *, graph: SkillsGraph) -> None:
    """Reads skill nodes in response-sized chunks and admits each skill's text.

    A skill whose text `_judge_skill_text` refuses is left out with that code.
    A skill reached once retained bytes pass the corpus budget is left out with
    `skills_corpus_budget_exhausted`. A failed chunk adds
    `skill_build_unconfirmed` for each of its skills, which stay unknown to this
    generation. Failure of every chunk propagates.
    """
    chunk_count = 0
    failed_chunk_count = 0
    last_chunk_failure: Exception | None = None
    for chunk_start in range(0, len(state.skill_ids), _FRONT_MATTER_CHUNK_SIZE):
        chunk = state.skill_ids[chunk_start : chunk_start + _FRONT_MATTER_CHUNK_SIZE]
        chunk_count += 1
        try:
            chunk_arcs = _read_skill_arcs(chunk, graph=graph)
        except Exception as exc:
            # Retain the error so all-failed builds can re-raise it.
            failed_chunk_count += 1
            last_chunk_failure = exc
            state.issues.extend(
                SkillIssue(
                    skill_id=f"{SKILL_DCID_PREFIX}{skill_id}",
                    code="skill_build_unconfirmed",
                    message=(
                        f"{SKILL_DCID_PREFIX}{skill_id} is unknown to this build: the read of "
                        f"its node failed with its chunk: {exc}"
                    ),
                )
                for skill_id in chunk
            )
            continue
        for skill_id in chunk:
            node_arcs = chunk_arcs.get(skill_id) or {}
            judged = _judge_skill_text(skill_id, node_arcs)
            if isinstance(judged, _SkillTextRefused):
                _leave_out(state, skill_id, code=judged.code, why=judged.why)
                continue
            if state.retained_content_bytes > SKILLS_CORPUS_CONTENT_BUDGET_BYTES:
                _leave_out(state, skill_id, code="skills_corpus_budget_exhausted", why=_BUDGET_WHY)
                continue
            # Reference checks reuse this pre-body baseline.
            state.budget_baseline_by_skill[skill_id] = state.retained_content_bytes
            state.retained_content_bytes += len(judged.text.encode("utf-8"))
            state.arcs_by_skill[skill_id] = node_arcs
            state.text_by_skill[skill_id] = judged
    if chunk_count and failed_chunk_count == chunk_count:
        assert last_chunk_failure is not None
        raise last_chunk_failure
    state.skill_ids = tuple(
        skill_id for skill_id in state.skill_ids if skill_id in state.text_by_skill
    )


@dataclass(frozen=True, slots=True)
class _SkillTextRefused:
    """Why `_judge_skill_text` refuses a skill's text, and the code it is left out under."""

    code: SkillIssueCode
    why: str


def _judge_skill_text(
    skill_id: str, node_arcs: dict[str, tuple[str, ...]]
) -> _SkillText | _SkillTextRefused:
    """Returns the skill's text and front matter, or the first rule it breaks.

    Checks a `text` value refused as undecodable or oversized before decoding, a
    missing value, the decoded byte limit, then the front matter.
    """
    if undecodable := node_arcs.get(UNDECODABLE_CONTENT_ARC):
        return _SkillTextRefused(
            code="skill_content_undecodable",
            why=f"its {CONTENT_ARC!r} is not decodable Markdown ({undecodable[0]})",
        )
    if oversized := node_arcs.get(OVERSIZED_CONTENT_ARC):
        return _SkillTextRefused(
            code="skill_content_too_large", why=f"its {CONTENT_ARC!r} {oversized[0]}"
        )
    text = _first_or_none(node_arcs.get(CONTENT_ARC, ()))
    if text is None:
        return _SkillTextRefused(
            code="skill_content_absent", why=f"the node carries no {CONTENT_ARC!r}"
        )
    if (text_bytes := len(text.encode("utf-8"))) > SKILL_BODY_MAX_BYTES:
        return _SkillTextRefused(
            code="skill_content_too_large",
            why=(
                f"its {CONTENT_ARC!r} is {text_bytes} bytes, past the "
                f"{SKILL_BODY_MAX_BYTES}-byte ceiling"
            ),
        )
    try:
        front_matter = read_front_matter(text, skill_id=skill_id)
    except FrontMatterInvalid as exc:
        return _SkillTextRefused(code="skill_frontmatter_invalid", why=f"its {CONTENT_ARC!r} {exc}")
    return _SkillText(text=text, front_matter=front_matter)


def _admit_skill_references(state: _IndexBuildState) -> None:
    """Queues each admitted skill's reference dcids for `_read_references`."""
    for skill_id in state.skill_ids:
        state.reference_dcids_by_skill[skill_id] = _admit_reference_dcids(
            state, skill_id, state.arcs_by_skill[skill_id].get(SKILL_REFERENCES_ARC, ())
        )


def _extract_targets(state: _IndexBuildState) -> None:
    """Records each catalogued skill's targets on both attachment arcs.

    Targets past `SKILLS_APPLIES_TO_MAX_TARGETS` across both arcs are refused
    before any attachment read, and their skill is excluded. Only
    `skillAppliesTo` targets are queued for classification.
    """
    for skill_id in state.skill_ids:
        arcs = state.arcs_by_skill[skill_id]
        targets = arcs.get(SKILL_APPLIES_TO_ARC, ())
        provenance_targets = arcs.get(SKILL_APPLIES_TO_PROVENANCE_ARC, ())
        if (declared := len(targets) + len(provenance_targets)) > SKILLS_APPLIES_TO_MAX_TARGETS:
            state.issues.append(
                SkillIssue(
                    skill_id=skill_id,
                    code="skill_reach_too_large",
                    message=(
                        f"declares {declared} {SKILL_APPLIES_TO_ARC} and "
                        f"{SKILL_APPLIES_TO_PROVENANCE_ARC} targets, past the "
                        f"{SKILLS_APPLIES_TO_MAX_TARGETS}-target bound; narrow its attachment "
                        "points"
                    ),
                )
            )
            state.oversized_applies_to.add(skill_id)
            targets, provenance_targets = (), ()
        state.targets_by_skill[skill_id] = targets
        state.provenance_targets_by_skill[skill_id] = provenance_targets
        state.all_targets.update(targets)


def _admit_reference_dcids(
    state: _IndexBuildState, skill_id: str, declared: Sequence[str]
) -> tuple[str, ...]:
    """Returns the declared reference dcids to read for `skill_id`.

    A dcid outside `skill/<id>/` is rejected without a read, since the dcid is
    what places a node under its skill. A repeated dcid counts once. The first
    `SKILL_REFERENCES_MAX_ENTRIES` are kept and the rest rejected. Each kept dcid
    is recorded as owned by the skill.
    """
    prefix = f"{SKILL_DCID_PREFIX}{skill_id}/"
    placed = tuple(dict.fromkeys(dcid for dcid in declared if dcid.startswith(prefix)))
    kept = placed[:SKILL_REFERENCES_MAX_ENTRIES]
    state.issues.extend(
        SkillIssue(
            skill_id=skill_id,
            code="skill_resource_rejected",
            message=f"{dcid}: does not start with {prefix!r}, so it is not placed under this skill",
        )
        for dcid in dict.fromkeys(declared)
        if not dcid.startswith(prefix)
    )
    state.issues.extend(
        SkillIssue(
            skill_id=skill_id,
            code="skill_resource_rejected",
            message=(
                f"{dcid}: dropped, past the skill's "
                f"{SKILL_REFERENCES_MAX_ENTRIES}-entry reference ceiling"
            ),
        )
        for dcid in placed[SKILL_REFERENCES_MAX_ENTRIES:]
    )
    state.skill_by_reference_dcid.update(dict.fromkeys(kept, skill_id))
    return kept


def _read_references(state: _IndexBuildState, *, graph: SkillsGraph) -> None:
    """Reads reference content in chunks. Each budget check uses the skill's
    front-matter baseline plus references retained for earlier skills, before
    queuing any of its reference dcids. An over-budget skill is left out of the
    catalogue with `skills_corpus_budget_exhausted`, and none of its references
    is read. Later skills also exceed the shared budget. A check that would fail
    while references of admitted skills are still pending first reads them all
    and checks again, since a pending reference can drop a counted one that
    shares its path. After that read, every admitted skill's list is judged in
    full, so the check sees exactly the bytes that will be served.

    Each read reference passes through its skill's `_SkillReferences`, which
    holds only the content of a path exactly one retained reference carries
    and returns the change in held bytes. That change is charged at once, so
    counted reference bytes equal held reference bytes at every budget check.
    Peak retained bytes are bounded by the corpus budget, one admitted reference
    list (`SKILL_REFERENCES_MAX_ENTRIES * SKILL_REFERENCE_MAX_BYTES`), and the
    arcs of the one chunk being judged, fewer than `_REFERENCE_CHUNK_SIZE`
    references.

    Partial chunk failures reject those references. Failure of every chunk
    propagates.
    """
    reference_chunk_count = 0
    failed_reference_chunk_count = 0
    last_reference_chunk_failure: Exception | None = None
    pending_reference_dcids: list[str] = []
    reference_stage_committed_bytes = 0

    def _read_pending_reference_chunk() -> None:
        nonlocal reference_chunk_count, failed_reference_chunk_count
        nonlocal last_reference_chunk_failure, reference_stage_committed_bytes
        chunk = pending_reference_dcids[:_REFERENCE_CHUNK_SIZE]
        del pending_reference_dcids[: len(chunk)]
        reference_chunk_count += 1
        try:
            chunk_arcs = graph.node_arcs_many(
                chunk, properties=(REFERENCE_PATH_ARC, CONTENT_ARC, "typeOf")
            )
        except Exception as exc:
            failed_reference_chunk_count += 1
            last_reference_chunk_failure = exc
            state.issues.extend(
                SkillIssue(
                    skill_id=state.skill_by_reference_dcid[reference_dcid],
                    code="skill_resource_rejected",
                    message=f"{reference_dcid}: content read failed for its chunk: {exc}",
                )
                for reference_dcid in chunk
            )
            return
        for reference_dcid in chunk:
            owner = state.references_by_skill[state.skill_by_reference_dcid[reference_dcid]]
            held_change = owner.record(
                reference_dcid, _judge_reference(reference_dcid, chunk_arcs.get(reference_dcid))
            )
            reference_stage_committed_bytes += held_change
            state.retained_content_bytes += held_change

    left_out: set[str] = set()
    for skill_id in state.skill_ids:
        baseline = state.budget_baseline_by_skill[skill_id] + reference_stage_committed_bytes
        if baseline > SKILLS_CORPUS_CONTENT_BUDGET_BYTES and pending_reference_dcids:
            while pending_reference_dcids:
                _read_pending_reference_chunk()
            baseline = state.budget_baseline_by_skill[skill_id] + reference_stage_committed_bytes
        if baseline > SKILLS_CORPUS_CONTENT_BUDGET_BYTES:
            left_out.add(skill_id)
            state.retained_content_bytes -= len(
                state.text_by_skill.pop(skill_id).text.encode("utf-8")
            )
            _leave_out(state, skill_id, code="skills_corpus_budget_exhausted", why=_BUDGET_WHY)
            continue
        state.references_by_skill[skill_id] = _SkillReferences()
        pending_reference_dcids.extend(state.reference_dcids_by_skill[skill_id])
        while len(pending_reference_dcids) >= _REFERENCE_CHUNK_SIZE:
            _read_pending_reference_chunk()
    while pending_reference_dcids:
        _read_pending_reference_chunk()
    state.skill_ids = tuple(skill_id for skill_id in state.skill_ids if skill_id not in left_out)

    if reference_chunk_count and failed_reference_chunk_count == reference_chunk_count:
        assert last_reference_chunk_failure is not None
        raise last_reference_chunk_failure


def _assemble_skills(
    state: _IndexBuildState,
    classified: _ClassifiedTargets,
    *,
    graph: SkillsGraph,
    max_depth: int,
    max_nodes: int,
    max_results_per_lead: int,
) -> tuple[dict[str, SkillEntry], dict[str, list[SkillMatch]]]:
    """Builds each `SkillEntry` and the inverse node-to-skills map.

    A skill with a target whose classification read raised, or whose reach or
    StatVar count read raised, stays served with `skill_build_unconfirmed`, an
    excluded empty reach, and a count of 0. When reach and count reads were made
    and every one raised, the last error propagates.
    """
    skills: dict[str, SkillEntry] = {}
    node_to_skills: dict[str, list[SkillMatch]] = {}
    reach_graph = _ReadCountingGraph(graph)
    for skill_id in state.skill_ids:
        attachments = tuple(
            Attachment(dcid=target, kind=classified.kind_by_target[target])
            for target in state.targets_by_skill[skill_id]
        ) + tuple(
            Attachment(dcid=target, kind=ProvenanceAttachment())
            for target in state.provenance_targets_by_skill[skill_id]
        )
        # A provenance attachment needs no classification read, so another skill's
        # failed read of the same dcid says nothing about it.
        failed_targets = [
            attachment.dcid
            for attachment in attachments
            if not isinstance(attachment.kind, ProvenanceAttachment)
            and attachment.dcid in classified.failure_by_target
        ]
        for attachment in attachments:
            if isinstance(attachment.kind, RejectedAttachment):
                state.issues.append(
                    SkillIssue(
                        skill_id=skill_id,
                        code="skill_attachment_rejected",
                        message=f"{attachment.dcid}: {attachment.kind.reason}",
                    )
                )
            elif (
                isinstance(attachment.kind, UnconfirmedAttachment)
                and attachment.dcid not in classified.failure_by_target
            ):
                state.issues.append(
                    SkillIssue(
                        skill_id=skill_id,
                        code="skill_attachment_kind_unconfirmed",
                        message=(
                            f"{attachment.dcid}'s classification read stopped at its page "
                            "budget before settling its attachment kind, so it contributes "
                            "nothing to what this skill matches on this build and the skill's "
                            "reach understates what it covers"
                        ),
                    )
                )

        excluded_reach = SkillReach(skill_id=skill_id, nodes=(), excluded=True)
        if failed_targets:
            state.issues.append(
                _unconfirmed_issue(
                    skill_id,
                    f"the classification read of {', '.join(failed_targets)} failed",
                    classified.failure_by_target[failed_targets[0]],
                )
            )
            reach, statvar_count, statvar_truncated = excluded_reach, 0, False
        elif skill_id in state.oversized_applies_to:
            reach, statvar_count, statvar_truncated = excluded_reach, 0, False
        elif not any(_reaches(attachment) for attachment in attachments):
            reach = SkillReach(skill_id=skill_id, nodes=(), excluded=False)
            statvar_count, statvar_truncated = 0, False
        else:
            try:
                reach, reach_issues, statvar_count, statvar_truncated = _reach_and_breadth(
                    skill_id,
                    attachments,
                    graph=reach_graph,
                    max_depth=max_depth,
                    max_nodes=max_nodes,
                    max_results_per_lead=max_results_per_lead,
                )
            except Exception as exc:
                state.issues.append(
                    _unconfirmed_issue(skill_id, "a reach or StatVar count read failed", exc)
                )
                reach, statvar_count, statvar_truncated = excluded_reach, 0, False
            else:
                state.issues.extend(reach_issues)

        references = state.references_by_skill[skill_id]
        state.issues.extend(references.issues(skill_id))

        skill_text = state.text_by_skill[skill_id]
        files = build_manifest(skill_text.text, references.files())
        skills[skill_id] = SkillEntry(
            front_matter=SkillFrontMatter(
                id=skill_id,
                name=skill_text.front_matter.name,
                description=skill_text.front_matter.description,
                content_version=content_version(files),
            ),
            frontmatter=skill_text.front_matter.mapping,
            url=_first_or_none(state.arcs_by_skill[skill_id].get("url", ())),
            attachments=attachments,
            reach=reach,
            statvar_count=statvar_count,
            statvar_count_truncated=statvar_truncated,
            files=files,
        )

        provenance_dcids = {
            attachment.dcid
            for attachment in attachments
            if isinstance(attachment.kind, ProvenanceAttachment)
        }
        for node in reach.nodes:
            node_to_skills.setdefault(node.dcid, []).append(
                SkillMatch(
                    skill_id=skill_id,
                    hop=node.hop,
                    breadth=statvar_count,
                    breadth_truncated=statvar_truncated,
                    via_provenance=node.dcid in provenance_dcids,
                )
            )
    if reach_graph.last_failure is not None and reach_graph.succeeded == 0:
        raise reach_graph.last_failure
    return skills, node_to_skills


class _ReadCountingGraph:
    """A `SkillsGraph` that counts the reads that return and keeps the last
    error raised, so the build can tell a graph outage from a failure confined
    to some reads."""

    def __init__(self, graph: SkillsGraph) -> None:
        self._graph = graph
        self.succeeded = 0
        self.last_failure: Exception | None = None

    def _counted[T](self, read: Callable[[], T]) -> T:
        try:
            result = read()
        except Exception as exc:
            self.last_failure = exc
            raise
        self.succeeded += 1
        return result

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        return self._counted(
            lambda: self._graph.node_arcs_many(dcids, properties=properties, deadline=deadline)
        )

    def node_subjects_many(
        self, dcids: Sequence[str], *, property: str, max_results: int
    ) -> dict[str, SubjectsPage]:
        return self._counted(
            lambda: self._graph.node_subjects_many(
                dcids, property=property, max_results=max_results
            )
        )


def _unconfirmed_issue(skill_id: str, what_failed: str, exc: Exception) -> SkillIssue:
    """Returns the `skill_build_unconfirmed` issue for a skill whose read raised."""
    return SkillIssue(
        skill_id=skill_id,
        code="skill_build_unconfirmed",
        message=(f"{what_failed}, so the skill is excluded from matching on this build: {exc}"),
    )


def _reach_and_breadth(
    skill_id: str,
    attachments: Sequence[Attachment],
    *,
    graph: SkillsGraph,
    max_depth: int,
    max_nodes: int,
    max_results_per_lead: int,
) -> tuple[SkillReach, list[SkillIssue], int, bool]:
    """Returns a skill's reach, its issues, and its StatVar count and truncation."""
    reach, issues, groups, constraint_properties = _expand_by_counting_property(
        skill_id,
        attachments,
        graph=graph,
        max_depth=max_depth,
        max_nodes=max_nodes,
        max_results_per_lead=max_results_per_lead,
    )
    statvar_count, statvar_truncated = _count_statvars(
        groups,
        constraint_properties,
        graph=graph,
        max_results_per_lead=max_results_per_lead,
    )
    if statvar_count > SKILLS_BREADTH_WARN_STATVARS:
        issues.append(
            SkillIssue(
                skill_id=skill_id,
                code="skill_breadth_high",
                message=(
                    f"reach carries at least {statvar_count} StatVars, past the "
                    f"{SKILLS_BREADTH_WARN_STATVARS}-StatVar warn bound"
                ),
            )
        )
    elif statvar_truncated:
        issues.append(
            SkillIssue(
                skill_id=skill_id,
                code="skill_reach_read_truncated",
                message=(
                    f"reach's StatVar count stopped at its page budget with "
                    f"{statvar_count} confirmed so far, a floor rather than the true "
                    "total"
                ),
            )
        )
    return reach, issues, statvar_count, statvar_truncated


def build_index(
    *,
    graph: SkillsGraph,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_results_per_lead: int = DEFAULT_MAX_RESULTS_PER_LEAD,
) -> SkillsIndex:
    """Reads the `AgentSkill` roster and builds its `SkillsIndex`.

    Expansion uses the supplied depth, node, and per-lead result limits.
    """
    reference_graph = DecodedTextGraph(graph, max_decoded_bytes=SKILL_REFERENCE_MAX_BYTES)
    graph = DecodedTextGraph(graph, max_decoded_bytes=SKILL_BODY_MAX_BYTES)
    state = _IndexBuildState()
    _read_roster(state, graph=graph)
    _read_front_matter(state, graph=graph)
    _admit_skill_references(state)
    _read_references(state, graph=reference_graph)
    _extract_targets(state)
    classified = _classify_targets(
        sorted(state.all_targets), graph=graph, max_results_per_lead=max_results_per_lead
    )
    skills, node_to_skills = _assemble_skills(
        state,
        classified,
        graph=graph,
        max_depth=max_depth,
        max_nodes=max_nodes,
        max_results_per_lead=max_results_per_lead,
    )

    logger.info(
        "skills index build retained %d bytes of skill body content against a "
        "%d-byte corpus-wide budget (%.1f%%)",
        state.retained_content_bytes,
        SKILLS_CORPUS_CONTENT_BUDGET_BYTES,
        100 * state.retained_content_bytes / SKILLS_CORPUS_CONTENT_BUDGET_BYTES,
    )
    return SkillsIndex(
        skills=skills,
        node_to_skills={dcid: tuple(matches) for dcid, matches in node_to_skills.items()},
        issues=tuple(state.issues),
        may_omit_published_skills=index_may_omit_published_skills(state.issues, skills),
    )


@dataclass(frozen=True, slots=True)
class _ReferenceRetained:
    """A reference node `_judge_reference` keeps. `path` is the bundle-relative
    path read from the node's `name`, and `content` its decoded text."""

    path: str
    content: str


@dataclass(frozen=True, slots=True)
class _ReferenceRejected:
    """A reference node `_judge_reference` drops. `reason` is the exact message
    its `skill_resource_rejected` issue carries."""

    reason: str


type _ReferenceVerdict = _ReferenceRetained | _ReferenceRejected
"""The one verdict `_judge_reference` computes for a reference dcid, recorded by
its skill's `_SkillReferences` as the chunk is read. It depends on the node's
own arcs, since a dcid placed under a skill's prefix has exactly one owner."""


def _judge_reference(
    reference_dcid: str, node_arcs: dict[str, tuple[str, ...]] | None
) -> _ReferenceVerdict:
    """Returns the first rejection or the retained path for one reference node.

    Checks the `name` path, a content arc refused as undecodable or oversized
    before decoding, a missing content arc, the reference type, then the byte
    limit on decoded content. Uniqueness of the path within the skill is checked
    by `_SkillReferences.record`.
    """
    names = node_arcs.get(REFERENCE_PATH_ARC, ()) if node_arcs else ()
    if len(names) != 1:
        return _ReferenceRejected(
            reason=f"{reference_dcid}: carries {len(names)} {REFERENCE_PATH_ARC!r} values, "
            "not the one path it needs"
        )
    path = names[0]
    if (path_problem := reference_path_rejection(path)) is not None:
        return _ReferenceRejected(
            reason=f"{reference_dcid}: its {REFERENCE_PATH_ARC!r} {path[:80]!r} is not a valid "
            f"reference path: it {path_problem}"
        )

    assert node_arcs is not None  # a path was read from it
    label = f"{reference_dcid} ({path})"
    if undecodable := node_arcs.get(UNDECODABLE_CONTENT_ARC):
        return _ReferenceRejected(
            reason=f"{label}: its {CONTENT_ARC!r} content is not decodable Markdown "
            f"({undecodable[0]})"
        )

    if oversized := node_arcs.get(OVERSIZED_CONTENT_ARC):
        return _ReferenceRejected(
            reason=f"{label} {oversized[0]}, and is dropped rather than served"
        )

    content = _first_or_none(node_arcs.get(CONTENT_ARC, ()))
    if content is None:
        return _ReferenceRejected(reason=f"{label}: carries no {CONTENT_ARC!r} content")

    if REFERENCE_TYPE not in node_arcs.get("typeOf", ()):
        return _ReferenceRejected(
            reason=(f"{label}: does not declare the {REFERENCE_TYPE!r} type the platform requires")
        )

    content_bytes = len(content.encode("utf-8"))
    if content_bytes > SKILL_REFERENCE_MAX_BYTES:
        return _ReferenceRejected(
            reason=(
                f"{label} is {content_bytes} bytes, past the "
                f"{SKILL_REFERENCE_MAX_BYTES}-byte ceiling, and is dropped rather "
                "than served"
            )
        )

    return _ReferenceRetained(path=path, content=content)


@dataclass(slots=True)
class _SkillReferences:
    """One skill's references as their chunks are read.

    `rejections` holds each rejected reference's issue message in declaration
    order. `dcids_by_path` lists every retained reference by path. A path carried
    by more than one names no single file, so every reference carrying it is
    dropped. `content_by_path` holds the content of each path exactly one
    retained reference carries, the only reference bytes the build keeps.
    """

    rejections: list[str] = field(default_factory=list)
    dcids_by_path: dict[str, list[str]] = field(default_factory=dict)
    content_by_path: dict[str, str] = field(default_factory=dict)

    def record(self, reference_dcid: str, verdict: _ReferenceVerdict) -> int:
        """Records one reference's verdict and returns the change in held bytes.

        A first reference at a path adds its content. A second drops the first's,
        and any later one at that path adds nothing.
        """
        match verdict:
            case _ReferenceRejected(reason=reason):
                self.rejections.append(reason)
                return 0
            case _ReferenceRetained(path=path, content=content):
                path_dcids = self.dcids_by_path.setdefault(path, [])
                path_dcids.append(reference_dcid)
                if len(path_dcids) == 1:
                    self.content_by_path[path] = content
                    return len(content.encode("utf-8"))
                if len(path_dcids) == 2:
                    return -len(self.content_by_path.pop(path).encode("utf-8"))
                return 0

    def files(self) -> tuple[SkillFile, ...]:
        """The served reference files, in path order."""
        return tuple(
            skill_file(path, content) for path, content in sorted(self.content_by_path.items())
        )

    def issues(self, skill_id: str) -> list[SkillIssue]:
        """A `skill_resource_rejected` issue per rejected reference, then per
        reference dropped for a shared path."""
        messages = list(self.rejections)
        for path, path_dcids in self.dcids_by_path.items():
            if len(path_dcids) > 1:
                messages.extend(
                    f"{reference_dcid}: its path {path!r} is also carried by "
                    f"{len(path_dcids) - 1} other reference(s) of this skill"
                    for reference_dcid in path_dcids
                )
        return [
            SkillIssue(skill_id=skill_id, code="skill_resource_rejected", message=message)
            for message in messages
        ]


def _first_or_none(values: Sequence[str]) -> str | None:
    return values[0] if values else None


@dataclass(frozen=True, slots=True)
class _ClassifiedTargets:
    """Each target's attachment kind, and the error of the batched read that
    failed for each target whose classification read raised. A failed target's
    kind is `UnconfirmedAttachment`."""

    kind_by_target: dict[str, AttachmentKind]
    failure_by_target: dict[str, Exception]


def _classify_targets(
    targets: Sequence[str], *, graph: SkillsGraph, max_results_per_lead: int
) -> _ClassifiedTargets:
    """Classifies targets from batched property probes, then type/range lookups.

    Incomplete pages yield `UnconfirmedAttachment`. Complete coverage confirms
    absence on an empty page. Complete evidence with an unsafe
    graph-sourced property yields `RejectedAttachment`. Tuple properties come
    from a fixed safe vocabulary. A batched read that raises marks every target
    in its batch failed; failure of every batched read propagates.
    """
    if not targets:
        return _ClassifiedTargets(kind_by_target={}, failure_by_target={})

    observed: dict[str, set[str]] = {target: set() for target in targets}
    probe_exhausted: set[str] = set()
    failure_by_target: dict[str, Exception] = {}
    read_count = 0
    failures: list[Exception] = []

    def _fail(batch: Sequence[str], exc: Exception) -> None:
        failures.append(exc)
        for target in batch:
            failure_by_target.setdefault(target, exc)

    for probe_property in _CLASSIFICATION_PROBE_PROPERTIES:
        read_count += 1
        try:
            pages = graph.node_subjects_many(
                targets, property=probe_property, max_results=_CLASSIFICATION_PROBE_MAX_RESULTS
            )
        except Exception as exc:
            _fail(targets, exc)
            continue
        for target in targets:
            page = pages[target]
            if page.dcids:
                observed[target].add(probe_property)
            elif page.coverage == SubjectsCoverage.budget_exhausted:
                probe_exhausted.add(target)

    unresolved = [
        target for target in targets if not observed[target] and target not in failure_by_target
    ]
    if unresolved:
        read_count += 1
        try:
            type_arcs = graph.node_arcs_many(unresolved, properties=("typeOf",))
        except Exception as exc:
            _fail(unresolved, exc)
            type_arcs = {}
        types_by_target = {
            target: arcs.get("typeOf", ())
            for target in unresolved
            if (arcs := type_arcs.get(target)) is not None
        }
        distinct_types = {type_dcid for types in types_by_target.values() for type_dcid in types}
        if distinct_types:
            read_count += 1
            try:
                range_pages = graph.node_subjects_many(
                    sorted(distinct_types),
                    property="rangeIncludes",
                    max_results=max_results_per_lead,
                )
            except Exception as exc:
                _fail([target for target, types in types_by_target.items() if types], exc)
                range_pages = {}
            for target, types in types_by_target.items():
                if target in failure_by_target:
                    continue
                for type_dcid in types:
                    page = range_pages[type_dcid]
                    observed[target].update(page.dcids)
                    if page.coverage == SubjectsCoverage.budget_exhausted:
                        probe_exhausted.add(target)

    if len(failures) == read_count:
        raise failures[-1]

    kind_by_target: dict[str, AttachmentKind] = {}
    for target, props in observed.items():
        kind = classify_attachment(frozenset(props))
        if target in probe_exhausted or target in failure_by_target:
            # Incomplete pages can make a target appear ambiguous or
            # unsupported. Resolve coverage before trusting its kind or
            # checking a graph-sourced property token.
            kind = UnconfirmedAttachment()
        elif isinstance(kind, ConstraintPropertyAttachment):
            kind = _reject_if_unsafe_property_token(target, kind)
        elif isinstance(kind, ConstraintValueAttachment):
            kind = _reject_if_unsafe_property_token(kind.property, kind)
        kind_by_target[target] = kind
    return _ClassifiedTargets(kind_by_target=kind_by_target, failure_by_target=failure_by_target)


# Graph-sourced property names must be bare identifiers. The shared validator
# permits operators used by literal expressions, so this stricter allowlist
# applies only where graph names enter inline arc expressions.
_GRAPH_PROPERTY_NAME_RE = re.compile(r"[A-Za-z0-9_.\-]+")


def _is_safe_graph_property_name(token: str) -> bool:
    return bool(_GRAPH_PROPERTY_NAME_RE.fullmatch(token))


def _reject_if_unsafe_property_token(token: str, kind: AttachmentKind) -> AttachmentKind:
    """Returns `kind` for a safe graph property, otherwise a rejection.

    Graph names must be bare identifiers and pass the inline-expression
    validator. Unsafe names would broaden a read or raise `ValueError`.
    """
    if not _is_safe_graph_property_name(token):
        return RejectedAttachment(
            reason=f"{token!r} is not a valid graph property name and cannot be counted through"
        )
    try:
        reject_unsafe_property_token(token)
    except ValueError as exc:
        return RejectedAttachment(reason=f"{token!r} cannot be safely queried: {exc}")
    return kind


def _reaches(attachment: Attachment) -> bool:
    """Whether the attachment contributes nodes to its skill's reach."""
    return isinstance(attachment.kind, ProvenanceAttachment) or (
        _counting_property(attachment) is not None
    )


def _counting_property(attachment: Attachment) -> str | None:
    """Returns the StatVar counting property. Rejected, unconfirmed, and
    provenance attachments map to `None`. Constraint-property attachments count
    through their admissible values.
    """
    match attachment.kind:
        case TupleMemberAttachment(tuple_property=tuple_property):
            return tuple_property
        case ConstraintValueAttachment(property=property_name):
            return property_name
        case ConstraintPropertyAttachment():
            return attachment.dcid
        case ProvenanceAttachment():
            return None
        case RejectedAttachment():
            return None
        case UnconfirmedAttachment():
            return None


def _expand_by_counting_property(
    skill_id: str,
    attachments: Sequence[Attachment],
    *,
    graph: SkillsGraph,
    max_depth: int,
    max_nodes: int,
    max_results_per_lead: int,
) -> tuple[SkillReach, list[SkillIssue], dict[str, tuple[str, ...]], frozenset[str]]:
    """Expands attachments grouped by counting property and combines their reach.

    Provenance attachments enter the reach first, at hop 0, without expansion.
    `max_nodes` bounds the deduplicated total across groups. Groups run in
    sorted order. Exceeding the bound excludes the skill before another
    group starts. Each `expand_skill` call may itself read past the combined
    bound, which is checked between calls. The returned groups feed StatVar
    counting. Constraint properties count through admissible values.
    """
    dcids_by_property: dict[str, list[str]] = {}
    constraint_properties: set[str] = set()
    for attachment in attachments:
        counting_property = _counting_property(attachment)
        if counting_property is None:
            continue
        dcids_by_property.setdefault(counting_property, []).append(attachment.dcid)
        if isinstance(attachment.kind, ConstraintPropertyAttachment):
            constraint_properties.add(counting_property)

    issues: list[SkillIssue] = []
    provenance_dcids = dict.fromkeys(
        attachment.dcid
        for attachment in attachments
        if isinstance(attachment.kind, ProvenanceAttachment)
    )
    nodes = [ReachedNode(dcid=dcid, hop=0, via_property=None) for dcid in provenance_dcids]
    groups: dict[str, tuple[str, ...]] = {}
    capped = False
    combined_seen = set(provenance_dcids)
    excluded = len(combined_seen) > max_nodes
    if excluded:
        issues.append(_combined_reach_too_large(skill_id, len(combined_seen), max_nodes))
    for counting_property in sorted(dcids_by_property):
        if excluded:
            break
        reach, group_issues = expand_skill(
            skill_id,
            dcids_by_property[counting_property],
            graph=graph,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_results_per_lead=max_results_per_lead,
        )
        issues.extend(group_issues)
        if reach.excluded:
            excluded = True
            continue
        capped = capped or reach.capped
        nodes.extend(node for node in reach.nodes if node.dcid not in provenance_dcids)
        groups[counting_property] = tuple(node.dcid for node in reach.nodes)
        combined_seen.update(node.dcid for node in reach.nodes)
        if len(combined_seen) > max_nodes:
            excluded = True
            issues.append(_combined_reach_too_large(skill_id, len(combined_seen), max_nodes))

    if excluded:
        return SkillReach(skill_id=skill_id, nodes=(), excluded=True), issues, {}, frozenset()
    return (
        SkillReach(skill_id=skill_id, nodes=tuple(nodes), excluded=False, capped=capped),
        issues,
        groups,
        frozenset(constraint_properties),
    )


def _combined_reach_too_large(skill_id: str, total: int, max_nodes: int) -> SkillIssue:
    return SkillIssue(
        skill_id=skill_id,
        code="skill_reach_too_large",
        message=(
            f"reach across every attachment totals {total} distinct nodes, past the "
            f"{max_nodes}-node bound"
        ),
    )


def _admissible_values(
    property_dcid: str, *, graph: SkillsGraph, max_results_per_lead: int
) -> tuple[tuple[str, ...], bool]:
    """Returns values admitted by `property_dcid`'s `rangeIncludes` types.

    `rangeIncludes` arcs are complete. The `typeOf` subject read may be
    truncated. The boolean reports whether returned values are only a floor.
    """
    property_arcs = graph.node_arcs_many((property_dcid,), properties=("rangeIncludes",))
    admissible_types = sorted((property_arcs.get(property_dcid) or {}).get("rangeIncludes", ()))
    if not admissible_types:
        return (), False
    type_pages = graph.node_subjects_many(
        admissible_types, property="typeOf", max_results=max_results_per_lead
    )
    values: set[str] = set()
    truncated = False
    for page in type_pages.values():
        values.update(page.dcids)
        if page.coverage != SubjectsCoverage.complete:
            truncated = True
    return tuple(sorted(values)), truncated


def _count_statvars(
    groups: dict[str, tuple[str, ...]],
    constraint_properties: frozenset[str],
    *,
    graph: SkillsGraph,
    max_results_per_lead: int,
) -> tuple[int, bool]:
    """Counts the deduplicated StatVars reached through each counting property.

    Counts stop once they exceed `SKILLS_BREADTH_WARN_STATVARS`. Incomplete
    pages mark the result truncated. Constraint properties count through
    `_admissible_values`. StatVars point to property values, so counting
    traverses those values.
    """
    statvars: set[str] = set()
    truncated = False
    for counting_property, dcids in groups.items():
        if counting_property in constraint_properties:
            dcids, values_truncated = _admissible_values(
                counting_property, graph=graph, max_results_per_lead=max_results_per_lead
            )
            truncated = truncated or values_truncated
        if not dcids:
            continue
        pages = graph.node_subjects_many(
            dcids, property=counting_property, max_results=SKILLS_BREADTH_COUNT_MAX_RESULTS
        )
        for page in pages.values():
            statvars.update(page.dcids)
            if page.coverage != SubjectsCoverage.complete:
                truncated = True
            if len(statvars) > SKILLS_BREADTH_WARN_STATVARS:
                return len(statvars), True
    return len(statvars), truncated
