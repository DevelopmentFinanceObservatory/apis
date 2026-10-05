"""Pure assembly and grouping for grounded Search regions.

Every input region yields a Part in query order. Definite regions share a spec
only after a successful same-shape collapse. Folded parts receive
``BINDING_FOLDED``. Non-definite and refused regions keep their own payloads.
``combine_regions`` returns the post-collapse regions for task reconciliation.
This module makes no graph or LLM calls.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from search.engine.assemble import (
    build_candidates_part,
    build_definite_part,
    build_no_data_part,
    build_unsupported_part,
    skills_index_incomplete_warning,
    skills_index_unavailable_warning,
)
from search.engine.departures import surviving_facts
from search.engine.envelope import make_diagnostics, make_query_echo, make_response, now_ms
from search.engine.same_shape import FacetOptionsReader, collapse_same_shape, five_tuple_key
from search.engine.unit.conversion import conversion_unavailable_warning
from search.models import (
    AmbiguousEntityCandidate,
    AmbiguousEntityGroup,
    ParsedInput,
    Resolution,
    Timing,
    Warning,
)
from skills_index import SkillsIndex

if TYPE_CHECKING:
    from search.engine.region import RegionResult
    from search.models import GraphRef, NoDataReason, Part, QueryEcho

# ---------------------------------------------------------------------------
# Warning code and pinned severity/message
# ---------------------------------------------------------------------------

BINDING_FOLDED = "binding_folded"
VARIABLES_CLAMPED = "variables_clamped"

# The name of the one step every leg's own timing_by_step seeds from the shared,
# once-per-request extraction stage (see fanout.py's base_timing). It is reported
# once, on the envelope's own timing_ms.by_step.
_SHARED_TIMING_STEP = "extract"


def _binding_folded_msg(variable_text: str | None) -> str:
    if variable_text:
        return (
            f"'{variable_text}' resolved to the same shape as an earlier part; "
            "folded into that part's spec."
        )
    return "This part resolved to the same shape as an earlier part; folded into that part's spec."


# ---------------------------------------------------------------------------
# Pure combiner helpers
# ---------------------------------------------------------------------------


def _dedupe_by(items: list, *, key) -> list:
    """Return items with duplicates removed, preserving order.

    key is a callable that returns a hashable comparison key.
    """
    seen: set = set()
    result = []
    for item in items:
        k = key(item)
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


def _group_definite_regions(
    definite: list[RegionResult],
    effective: list[RegionResult],
    *,
    residual_indices: set[int],
) -> tuple[dict[int, RegionResult], set[int]]:
    """Map each definite part to its representative and mark folded indices.

    Group by ``five_tuple_key``. A residual group keeps each original result, warning, and task.
    Merged members use the earliest region's representative, and
    later members are marked folded.
    """
    groups: dict[tuple, list[RegionResult]] = {}
    for r in definite:
        groups.setdefault(five_tuple_key(r.spec), []).append(r)

    representative_by_index: dict[int, RegionResult] = {}
    folded_indices: set[int] = set()
    for group in groups.values():
        if any(r.earliest_index in residual_indices for r in group):
            for r in group:
                representative_by_index[r.earliest_index] = r
            continue

        first_index = group[0].earliest_index
        representative = next(e for e in effective if e.earliest_index == first_index)
        for r in group:
            representative_by_index[r.earliest_index] = representative
            if r.earliest_index != first_index:
                folded_indices.add(r.earliest_index)
    return representative_by_index, folded_indices


def _shared_timing_by_step(regions: list[RegionResult]) -> dict[str, int]:
    """Read the shared pre-fanout timing from the first region that carries it.

    Deadline placeholders may carry no timing. Return an empty mapping when
    no region records the shared step.
    """
    for region in regions:
        if _SHARED_TIMING_STEP in region.timing_by_step:
            return {_SHARED_TIMING_STEP: region.timing_by_step[_SHARED_TIMING_STEP]}
    return {}


def _no_data_entity_terms(region: RegionResult) -> tuple[list[str] | None, list[str] | None]:
    """Return entity terms and candidates for no-data reasons that carry them.

    Ambiguous entities supply terms and ordered candidate names.
    Role-not-admitted entities supply terms only. Unresolved entities use
    ``unresolved_entity_names``, and so do the names whose direction a
    ``role_not_resolved`` refusal could not tell, each in the request's own words for it
    (``RegionResult.entity_words``). Other reasons or empty values return ``None``.
    """
    if region.no_data_reason == "entity_ambiguous":
        entity_terms = [e.text for e in region.ambiguous_entities] or None
        candidates = [c.name for e in region.ambiguous_entities for c in e.candidates] or None
        return entity_terms, candidates
    if region.no_data_reason == "entity_role_not_admitted":
        entity_terms = [e.text for e in region.role_not_admitted_entities] or None
        return entity_terms, None
    if region.no_data_reason == "entity_not_resolved":
        return list(region.unresolved_entity_names) or None, None
    if region.no_data_reason == "role_not_resolved":
        words = region.entity_words or {}
        return [words.get(name, name) for name in region.unresolved_entity_names] or None, None
    return None, None


def _ambiguous_entity_groups(region: RegionResult) -> list[AmbiguousEntityGroup]:
    """Convert region ambiguity data to the wire shape.

    Engine candidates use ``name``. Wire candidates use ``label``. Other
    fields copy directly.
    """
    return [
        AmbiguousEntityGroup(
            text=entity.text,
            candidates=[
                AmbiguousEntityCandidate(dcid=c.dcid, label=c.name, kind=c.kind)
                for c in entity.candidates
            ],
        )
        for entity in region.ambiguous_entities
    ]


def _data_query(region: RegionResult) -> ParsedInput | None:
    """Return the part's parsed form for an `entity_ambiguous` outcome.

    It is the region's parsed form with every entity the region resolved to one node
    pinned to its dcid, and each ambiguous wording left unpinned for the client to pin
    the user's pick. A leg that cannot name a direction withholds its restatement and
    still carries this form. `None` for every other reason, and when the region has no
    parsed form.
    """
    if region.no_data_reason != "entity_ambiguous" or region.parsed_form is None:
        return None
    part = region.parsed_form
    pinned = [
        entity.model_copy(
            update={"dcid": region.resolved_entity_dcids.get(entity.text, entity.dcid)}
        )
        for entity in part.entities or ()
    ]
    return ParsedInput(parts=[part.model_copy(update={"entities": pinned or None})])


def _build_part(
    region: RegionResult,
    *,
    representative: RegionResult,
    folded: bool,
    include_sentence: bool,
    max_candidates: int,
    include_observations: bool,
    include_skills: bool,
    skills_index: SkillsIndex | None,
    envelope_warnings: list[Warning],
) -> Part:
    """Build a part from the original region and its answer representative.

    Variable text, restatement, and operations belong to the original region, even when a
    merged representative supplies its spec. Shared timing appears on the
    envelope, so part timing excludes it. Folded parts carry only
    ``binding_folded``. The representative's warnings appear on its first part.

    Builders derive the wire restatement reason from the region's restatement
    and reason together, keeping both fields consistent after a door override. A raised
    departure fact reaches the part only while the part's or the envelope's warnings hold
    the warning it was raised beside. On a candidates part only warnings stamped with no
    reading's `spec_id` count, since the part's departures speak for every reading it shows.
    """
    own_timing_by_step = {
        step: ms for step, ms in region.timing_by_step.items() if step != _SHARED_TIMING_STEP
    }
    raised = {**region.departure_facts, **representative.departure_facts}
    timing_ms = Timing(total=sum(own_timing_by_step.values()), by_step=own_timing_by_step or None)
    variable_text = region.variable_text or None

    if representative.status == "definite":
        warnings: list[Warning] = []
        if folded:
            warnings.append(
                Warning(
                    code=BINDING_FOLDED,
                    severity="info",
                    message=_binding_folded_msg(variable_text),
                )
            )
        else:
            warnings = list(representative.warnings)
            if (unavailable := conversion_unavailable_warning(representative.spec)) is not None:
                warnings.append(unavailable)
        return build_definite_part(
            representative.spec,
            variable_text=variable_text,
            warnings=warnings,
            timing_ms=timing_ms,
            include_sentence=include_sentence,
            restatement=region.restatement,
            restatement_reason=region.restatement_reason,
            operations=region.operations,
            include_skills=include_skills,
            skills_index=skills_index,
            departure_facts=surviving_facts(raised, [*warnings, *envelope_warnings]),
            entity_words=region.entity_words,
        )
    if region.status == "candidates":
        return build_candidates_part(
            list(region.specs),
            variable_text=variable_text,
            warnings=list(region.warnings),
            timing_ms=timing_ms,
            max_candidates=max_candidates,
            include_sentence=include_sentence,
            include_observations=include_observations,
            # The near-neighbor gate promotes its runner-up into bind order.
            # Other ambiguous sets sort broadest-first. Score and gate metadata preserve
            # the loose-last relevance partition.
            bind_ranked=region.near_neighbor_fired,
            budget_truncated=region.candidate_budget_truncated,
            candidate_scores=region.candidate_scores,
            near_neighbor_spec_ids=region.candidate_near_neighbor_spec_ids,
            # dataset_resubmit preserves its posted candidate order and labels through this override.
            ordering=region.candidate_ordering_override,
            restatement=region.restatement,
            restatement_reason=region.restatement_reason,
            operations=region.operations,
            include_skills=include_skills,
            skills_index=skills_index,
            # A fact holds for every reading only beside a warning no reading is stamped
            # on; one raised for some readings stays on their stamped warnings.
            departure_facts=surviving_facts(
                raised,
                [w for w in (*region.warnings, *envelope_warnings) if w.spec_id is None],
            ),
            request_words=region.entity_words is not None,
        )
    if region.status == "unsupported":
        assert region.unsupported is not None, (
            "an unsupported region must carry its unsupported payload; a falsy "
            "value here is an assembly bug, not a case to default around"
        )
        return build_unsupported_part(
            region.unsupported.code,
            region.unsupported.pointer,
            region.unsupported.detail,
            region.unsupported.definition,
            variable_text=variable_text,
            warnings=list(region.warnings),
            timing_ms=timing_ms,
            restatement=region.restatement,
            restatement_reason=region.restatement_reason,
            operations=region.operations,
            include_skills=include_skills,
            skills_index=skills_index,
            cause=region.unsupported.cause,
        )
    assert region.no_data_reason, (
        "a no_data region must carry a real reason code; a falsy no_data_reason "
        "here is an assembly bug, not a case to default around"
    )
    entity_terms, ambiguous_candidates = _no_data_entity_terms(region)
    return build_no_data_part(
        cast("NoDataReason", region.no_data_reason),
        variable_text=variable_text,
        warnings=list(region.warnings),
        timing_ms=timing_ms,
        include_sentence=include_sentence,
        nearest_real=list(region.nearest_real) if region.nearest_real else None,
        flagged_match=region.flagged_match[0] if region.flagged_match else None,
        gate=region.gate,
        alternate_match=region.alternate_match,
        elected_stat_vars=region.elected_stat_vars,
        entity_terms=entity_terms,
        role_axis=region.no_data_role_axis,
        bound_role_dcids=region.no_data_bound_role_dcids,
        ambiguous_candidates=ambiguous_candidates,
        ambiguous_entities=_ambiguous_entity_groups(region),
        ambiguous_roles={
            entity.text: entity.role for entity in region.ambiguous_entities if entity.role
        },
        data_query=_data_query(region),
        qualifier_terms=list(region.unbound_qualifiers) or None,
        unit_alternatives=list(region.unit_alternatives) or None,
        unit_alternatives_truncated=region.unit_alternatives_truncated,
        restatement=region.restatement,
        restatement_reason=region.restatement_reason,
        operations=region.operations,
        include_skills=include_skills,
        skills_index=skills_index,
        cause=region.cause,
        window_gap_fact=region.window_gap_fact,
        departure_facts=surviving_facts(raised, [*region.warnings, *envelope_warnings]),
        request_words=region.entity_words is not None,
    )


# ---------------------------------------------------------------------------
# Single-region response assembly
# ---------------------------------------------------------------------------


def assemble_region(
    region: RegionResult,
    *,
    extra_warnings: list[Warning],
    start_ms: int,
    engine_build: str,
    include_sentence: bool,
    max_candidates: int,
    include_observations: bool = False,
    include_skills: bool = False,
    skills_index: SkillsIndex | None = None,
    llm_usage: dict | None = None,
    echo: QueryEcho | None = None,
    query: str = "",
) -> Resolution:
    """Assemble one region into a one-part response.

    ``llm_usage`` is summed by the caller across extract and bind calls. A
    supplied ``echo`` is used verbatim. Otherwise a raw-text echo is built
    from ``query``. For other entry paths, pass their prebuilt echo.

    Skills are attached to the part and response. Missing indexes emit
    ``skills_index_unavailable``. A snapshot that may omit published skills
    emits ``skills_index_incomplete`` once on the envelope.
    """
    all_extra_warnings = list(extra_warnings)
    all_extra_warnings.extend(
        skills_warning
        for skills_warning in (
            skills_index_unavailable_warning(
                include_skills=include_skills, skills_index=skills_index
            ),
            skills_index_incomplete_warning(
                include_skills=include_skills, skills_index=skills_index
            ),
        )
        if skills_warning is not None
    )
    envelope_warnings = _dedupe_by(
        all_extra_warnings, key=lambda w: (w.code, w.severity, w.message)
    )
    echo = echo if echo is not None else make_query_echo(query, extract_skipped=False)
    diag = make_diagnostics(
        engine_build,
        envelope_warnings,
        _shared_timing_by_step([region]),
        now_ms() - start_ms,
        llm_usage=llm_usage,
    )
    part = _build_part(
        region,
        representative=region,
        folded=False,
        include_sentence=include_sentence,
        max_candidates=max_candidates,
        include_observations=include_observations,
        include_skills=include_skills,
        skills_index=skills_index,
        envelope_warnings=envelope_warnings,
    )
    return make_response(
        echo, diag, [part], include_skills=include_skills, skills_index=skills_index
    )


# ---------------------------------------------------------------------------
# Multi-region grouping
# ---------------------------------------------------------------------------


def combine_regions(
    regions: list[RegionResult],
    *,
    extra_warnings: list[Warning],
    start_ms: int,
    engine_build: str,
    include_sentence: bool,
    max_candidates: int,
    include_observations: bool = False,
    include_skills: bool = False,
    skills_index: SkillsIndex | None = None,
    llm_usage: dict | None = None,
    echo: QueryEcho | None = None,
    query: str = "",
    set_ref_for: Callable[[list[str]], GraphRef | None] | None = None,
    facet_options_for: FacetOptionsReader | None = None,
) -> tuple[Resolution, list[RegionResult]]:
    """Combine regions into one response, with one part per input in query order.

    Same-shape definite regions merge only when collapse succeeds. Merged
    non-first parts share the representative spec and get ``binding_folded``.
    Refused groups and other regions keep their own payloads.

    Returns the response and post-collapse regions. Untouched entries retain
    object identity. Merged entries are new with ``slot_options_task`` and
    ``place_options_task`` cleared. Identity lets the caller cancel and relaunch enrichment.
    Its only graph reads are the injected ``set_ref_for`` and ``facet_options_for``
    readers a same-shape merge calls. It performs no task operations.

    ``llm_usage`` is summed by the caller. A supplied ``echo`` is used
    verbatim. Otherwise a raw-text echo is built from ``query``. Skills-index
    warnings are added once on the envelope.
    """
    definite = [r for r in regions if r.status == "definite"]
    others = [r for r in regions if r.status != "definite"]

    collapsed, residual_texts = collapse_same_shape(
        definite, set_ref_for=set_ref_for, facet_options_for=facet_options_for
    )
    residual_indices = {earliest_index for _variable_text, earliest_index in residual_texts}
    refused = [r for r in definite if r.earliest_index in residual_indices]
    effective = sorted(collapsed + refused + others, key=lambda r: r.earliest_index)

    representative_by_index, folded_indices = _group_definite_regions(
        definite,
        effective,
        residual_indices=residual_indices,
    )

    all_extra_warnings = list(extra_warnings)
    all_extra_warnings.extend(
        skills_warning
        for skills_warning in (
            skills_index_unavailable_warning(
                include_skills=include_skills, skills_index=skills_index
            ),
            skills_index_incomplete_warning(
                include_skills=include_skills, skills_index=skills_index
            ),
        )
        if skills_warning is not None
    )
    envelope_warnings = _dedupe_by(
        all_extra_warnings, key=lambda w: (w.code, w.severity, w.message)
    )
    diag = make_diagnostics(
        engine_build,
        envelope_warnings,
        _shared_timing_by_step(regions),
        now_ms() - start_ms,
        llm_usage=llm_usage,
    )
    echo = echo if echo is not None else make_query_echo(query, extract_skipped=False)

    parts = [
        _build_part(
            r,
            representative=representative_by_index.get(r.earliest_index, r),
            folded=r.earliest_index in folded_indices,
            include_sentence=include_sentence,
            max_candidates=max_candidates,
            include_observations=include_observations,
            include_skills=include_skills,
            skills_index=skills_index,
            envelope_warnings=envelope_warnings,
        )
        for r in sorted(regions, key=lambda r: r.earliest_index)
    ]
    response = make_response(
        echo, diag, parts, include_skills=include_skills, skills_index=skills_index
    )
    return response, effective
