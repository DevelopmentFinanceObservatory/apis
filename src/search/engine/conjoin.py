"""Pure assembly and grouping for grounded Search regions.

Every input region yields a Part in query order. Definite regions share a spec
only after a successful same-shape collapse. Folded parts receive
``BINDING_FOLDED``. Non-definite and refused regions keep their own payloads.
``combine_regions`` returns the post-collapse regions for task reconciliation.
This module makes no graph or LLM calls.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from pydantic import ValidationError

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
from search.engine.region.source_choice import scope_records
from search.engine.role_text import (
    parsed_directions_contradicted,
    role_axis_for_text,
    role_text_for_axis,
)
from search.engine.same_shape import FacetOptionsReader, collapse_same_shape, fold_key
from search.engine.unit.bind import unit_text_naming
from search.engine.unit.conversion import conversion_unavailable_warning
from search.models import (
    AmbiguousEntityCandidate,
    AmbiguousEntityGroup,
    AmbiguousSourceCandidate,
    AmbiguousSourceGroup,
    NoDataOption,
    NoDataReason,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    Resolution,
    ResolvedQualifier,
    RoleAxis,
    SourceMention,
    Timing,
    Warning,
)
from search.render.labels import _join_labels
from search.render.no_data import ScopeSources
from skills_index import SkillsIndex

if TYPE_CHECKING:
    from search.engine.region import RegionResult
    from search.engine.region.result import AmbiguousEntityCandidate as EngineCandidate
    from search.engine.region.result import AmbiguousSource
    from search.models import GraphRef, Part, QueryEcho

# ---------------------------------------------------------------------------
# Warning code and pinned severity/message
# ---------------------------------------------------------------------------

BINDING_FOLDED = "binding_folded"
VARIABLES_CLAMPED = "variables_clamped"

# The name of the one step every leg's own timing_by_step seeds from the shared,
# once-per-request extraction stage (see fanout.py's base_timing). It is reported
# once, on the envelope's own timing_ms.by_step.
_SHARED_TIMING_STEP = "extract"

# The steps that time stages inside `shape` (`phase_intake._phase_derive_shapes`). A part's
# by_step reports them, and its total leaves them out, since `shape` already spans them.
NESTED_SHAPE_STEPS = frozenset(
    {
        "shape_arcs",
        "shape_probe",
        "shape_probe_coverage",
        "shape_labels",
        "shape_groups",
        "shape_admission",
    }
)


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

    Group by ``fold_key``, as ``collapse_same_shape`` does, so a region compared with the
    others (``RegionResult.separate``) is its own group and carries no fold. A residual
    group keeps each original result, warning, and task. Merged members use the earliest
    region's representative, and later members are marked folded.
    """
    groups: dict[tuple, list[RegionResult]] = {}
    for r in definite:
        groups.setdefault(fold_key(r), []).append(r)

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
        terms = dict.fromkeys(words.get(name, name) for name in region.unresolved_entity_names)
        return list(terms) or None, None
    return None, None


def _untold_place_terms(region: RegionResult) -> list[list[str]] | None:
    """The terms of a ``role_not_resolved`` refusal grouped by the place each names
    (``RegionResult.untold_groups``), in the request's own words as
    `_no_data_entity_terms` gives them, so the sentence counts places. ``None`` for every
    other reason."""
    if region.no_data_reason != "role_not_resolved":
        return None
    words = region.entity_words or {}
    return [
        list(dict.fromkeys(words.get(name, name) for name in group))
        for group in region.untold_groups
    ] or None


def _ambiguous_entity_groups(region: RegionResult) -> list[AmbiguousEntityGroup]:
    """Convert region ambiguity data to the wire shape, for an ``entity_ambiguous`` region
    only. A region refused for another reason can still name ambiguous entities
    (`phase_intake._phase_check_entities`), and the wire field is empty for it.

    Engine candidates use ``name``. Wire candidates use ``label``. Other
    fields copy directly.
    """
    if region.no_data_reason != "entity_ambiguous":
        return []
    return [
        AmbiguousEntityGroup(
            text=entity.text,
            candidates=[
                AmbiguousEntityCandidate(ref=c.dcid, label=c.name, kind=c.kind)
                for c in entity.candidates
            ],
        )
        for entity in region.ambiguous_entities
    ]


def _ambiguous_source_groups(region: RegionResult) -> list[AmbiguousSourceGroup]:
    """Each ambiguous source wording of a `source_ambiguous` region, or the series code of a
    `source_code_ambiguous` one, with its candidates, in the wire shape."""
    return [
        AmbiguousSourceGroup(
            text=group.text,
            candidates=[
                AmbiguousSourceCandidate(ref=c.ref, label=c.label, kind=c.kind)
                for c in group.candidates
            ],
        )
        for group in region.ambiguous_sources
    ]


def _unbound_terms(
    region: RegionResult,
) -> tuple[list[ResolvedQualifier] | None, list[ResolvedQualifier] | None, ScopeSources | None]:
    """The category terms, the unbound sources and the source scope a no-data `region`
    names.

    A region read under a source scope (`RegionResult.source_scope`) names the sources the
    scope names, else those it leaves out (`source_choice.scope_records`). A
    `qualifier_not_bound` outcome lists them, then the region's other unbound sources, each
    set of references once, beside its category terms. A region with no scope names its unbound sources
    among its terms."""
    terms = list(region.unbound_qualifiers)
    scope = region.source_scope
    if scope is None:
        return [*terms, *region.unbound_sources] or None, None, None
    named = scope_records(scope)
    naming = (
        ScopeSources(
            polarity="include" if scope.named else "exclude",
            labels=tuple(record.text for record in named),
        )
        if named
        else None
    )
    if region.no_data_reason != "qualifier_not_bound":
        return terms or None, None, naming
    sources: dict[tuple[str, ...], ResolvedQualifier] = {}
    for record in (*named, *region.unbound_sources):
        sources.setdefault(tuple(ref.ref for ref in record.named), record)
    return terms or None, list(sources.values()) or None, naming


def _option(mentions: Sequence[str], label: str, part: ParsedPart) -> NoDataOption | None:
    """One option posting `part`, or `None` when `part` breaks the `ParsedPart` bounds or
    states a direction its own wording contradicts, which the parsed door refuses."""
    try:
        posted = ParsedPart.model_validate(part.model_dump())
    except ValidationError:
        return None
    if parsed_directions_contradicted(posted):
        return None
    return NoDataOption(
        mentions=list(mentions), label=label, data_query=ParsedInput(parts=[posted])
    )


def _entity_options(region: RegionResult, part: ParsedPart) -> list[NoDataOption | None]:
    """One option per ambiguous wording and candidate: the part with every entity that
    resolved to one node pinned, and every row with the wording pinned to the candidate.
    A row asked for its members takes a group's own node by the group's registered name,
    with no dcid, since a pin cannot ask for a group's members; a group node with no
    registered name that resolves to it alone has no option. Other ambiguous wordings stay
    unpinned."""
    base = [
        entity.model_copy(
            update={"dcid": region.resolved_entity_dcids.get(entity.text, entity.dcid)}
        )
        for entity in part.entities or ()
    ]

    def chosen(entity: ParsedEntity, candidate: EngineCandidate) -> ParsedEntity | None:
        if entity.members and candidate.group_node:
            if candidate.group_name is None:
                return None
            return entity.model_copy(update={"text": candidate.group_name, "dcid": None})
        return entity.model_copy(update={"dcid": candidate.dcid})

    options: list[NoDataOption | None] = []
    for group in region.ambiguous_entities:
        for candidate in group.candidates:
            entities = [
                chosen(entity, candidate) if entity.text == group.text else entity
                for entity in base
            ]
            if None in entities:
                continue
            options.append(
                _option(
                    [group.text],
                    candidate.name,
                    part.model_copy(update={"entities": entities}),
                )
            )
    return options


def _source_options(region: RegionResult, part: ParsedPart) -> list[NoDataOption | None]:
    """One option per ambiguous source wording and candidate, setting the candidate's ref on
    every ref-less mention with that wording; a wording mentioned twice gives its options
    once, from its first group. A label names an excluded source as excluded."""
    groups: dict[str, AmbiguousSource] = {}
    for group in region.ambiguous_sources:
        groups.setdefault(group.text, group)
    options: list[NoDataOption | None] = []
    for text, group in groups.items():
        mentions = [m for m in part.sources if m.ref is None and m.text == text]
        excluded = bool(mentions) and mentions[0].polarity == "exclude"
        for candidate in group.candidates:
            sources = [
                m.model_copy(update={"ref": candidate.ref}) if m in mentions else m
                for m in part.sources
            ]
            label = f"Excluding {candidate.label}" if excluded else candidate.label
            options.append(_option([text], label, part.model_copy(update={"sources": sources})))
    return options


def _source_code_options(region: RegionResult, part: ParsedPart) -> list[NoDataOption | None]:
    """One option per provenance holding the part's code, naming it as the part's only
    included source: included scopes union, so a source added beside the others would leave
    both holders allowed. Exclusions stay."""
    excluded = [mention for mention in part.sources if mention.polarity == "exclude"]
    return [
        _option(
            [group.text],
            candidate.label,
            part.model_copy(update={"sources": [*excluded, SourceMention(ref=candidate.ref)]}),
        )
        for group in region.ambiguous_sources
        for candidate in group.candidates
    ]


def _role_options(region: RegionResult, part: ParsedPart) -> list[NoDataOption | None]:
    """One option per reading of the directions the question left untold
    (`RegionResult.role_assignments`): every row of a mapped text states its axis, and a row
    the reading repeats is dropped. Its mentions are the untold names
    (`unresolved_entity_names`) the readings map, which every reading maps alike, in the
    request's words and in query order. The label reads the option's own directions."""
    words = region.entity_words or {}
    mapped = region.role_assignments[0] if region.role_assignments else {}
    mentions = list(
        dict.fromkeys(
            words.get(name, name) for name in region.unresolved_entity_names if name in mapped
        )
    )
    options: list[NoDataOption | None] = []
    for reading in region.role_assignments:
        rows = [
            entity.model_copy(update={"role_text": role_text_for_axis(reading[entity.text])})
            if entity.text in reading
            else entity
            for entity in part.entities or ()
        ]
        entities = [row for i, row in enumerate(rows) if row not in rows[:i]]
        label = _direction_label(entities)
        options.append(_option(mentions, label, part.model_copy(update={"entities": entities})))
    return options


def _direction_label(entities: Sequence[ParsedEntity]) -> str:
    """The directions `entities` state, in words: origins "to" destinations, or "From" or
    "To" with the names when one axis holds them."""
    by_axis: dict[RoleAxis, list[str]] = {"origin": [], "destination": []}
    for entity in entities:
        axis = role_axis_for_text(entity.role_text)
        if axis is not None and entity.text not in by_axis[axis]:
            by_axis[axis].append(entity.text)
    origins, destinations = (_join_labels(by_axis[axis]) for axis in ("origin", "destination"))
    if origins and destinations:
        return f"{origins} to {destinations}"
    return f"From {origins}" if origins else f"To {destinations}"


def _unit_options(region: RegionResult, part: ParsedPart) -> list[NoDataOption | None]:
    """One option per currency a `unit_not_resolved` refusal offers, its unit text naming
    that currency in place of the ambiguous words (`unit.bind.unit_text_naming`). A role
    currency's label names the role's currency."""
    if part.unit_text is None:
        return []
    options: list[NoDataOption | None] = []
    for alternative in region.unit_alternatives:
        unit_text = unit_text_naming(part.unit_text, alternative.currency)
        if unit_text is None:
            continue
        label = (
            alternative.label
            if isinstance(alternative.currency, str)
            else f"{alternative.label}'s currency"
        )
        options.append(
            _option([part.unit_text], label, part.model_copy(update={"unit_text": unit_text}))
        )
    return options


_OPTION_BUILDERS: dict[
    NoDataReason, Callable[[RegionResult, ParsedPart], list[NoDataOption | None]]
] = {
    "entity_ambiguous": _entity_options,
    "source_ambiguous": _source_options,
    "source_code_ambiguous": _source_code_options,
    "role_not_resolved": _role_options,
    "unit_not_resolved": _unit_options,
}


def _options(region: RegionResult) -> list[NoDataOption]:
    """The ready requests of a refusal that asks for a choice, each a one-part parsed input
    built from the region's parsed form. Empty for every other reason, when the region has
    no parsed form (its inputs exceed the `ParsedPart` bounds), and for an option whose part
    breaks those bounds or contradicts its own wording."""
    build = _OPTION_BUILDERS.get(region.no_data_reason) if region.no_data_reason else None
    if build is None or region.parsed_form is None:
        return []
    return [option for option in build(region, region.parsed_form) if option is not None]


def _build_part(
    region: RegionResult,
    *,
    representative: RegionResult,
    folded: bool,
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

    A raised departure fact reaches the part only while the part's or the envelope's warnings,
    or the request-wide qualifier warnings its own region leaves unapplied
    (`RegionResult.unapplied_qualifier_warnings`), hold the warning it was raised beside. On
    a candidates part only warnings stamped with no reading's `spec_id` count, since the
    part's departures speak for every reading it shows.
    """
    own_timing_by_step = {
        step: ms for step, ms in region.timing_by_step.items() if step != _SHARED_TIMING_STEP
    }
    raised = {**region.departure_facts, **representative.departure_facts}
    timing_ms = Timing(
        total=sum(ms for step, ms in own_timing_by_step.items() if step not in NESTED_SHAPE_STEPS),
        by_step=own_timing_by_step or None,
    )
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
            separate=region.separate,
            timing_ms=timing_ms,
            restatement=region.restatement,
            operations=region.operations,
            include_skills=include_skills,
            skills_index=skills_index,
            departure_facts=(
                *surviving_facts(
                    raised, [*warnings, *region.unapplied_qualifier_warnings, *envelope_warnings]
                ),
                *(() if folded else representative.inferred_facts),
            ),
            entity_words=region.entity_words,
        )
    if region.status == "candidates":
        return build_candidates_part(
            list(region.specs),
            variable_text=variable_text,
            warnings=list(region.warnings),
            separate=region.separate,
            timing_ms=timing_ms,
            max_candidates=max_candidates,
            include_observations=include_observations,
            budget_truncated=region.candidate_budget_truncated,
            # dataset_resubmit states its posted ordering through this override.
            ordering=region.candidate_ordering_override or "intent_rank",
            restatement=region.restatement,
            operations=region.operations,
            include_skills=include_skills,
            skills_index=skills_index,
            # A fact holds for every reading only beside a warning no reading is stamped
            # on; one raised for some readings stays on their stamped warnings.
            departure_facts=surviving_facts(
                raised,
                [
                    w
                    for w in (
                        *region.warnings,
                        *region.unapplied_qualifier_warnings,
                        *envelope_warnings,
                    )
                    if w.spec_id is None
                ],
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
            separate=region.separate,
            timing_ms=timing_ms,
            restatement=region.restatement,
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
    if region.no_data_reason in ("source_ambiguous", "source_code_ambiguous"):
        ambiguous_candidates = [
            c.label for group in region.ambiguous_sources for c in group.candidates
        ] or None
    qualifier_terms, unbound_sources, scope = _unbound_terms(region)
    return build_no_data_part(
        region.no_data_reason,
        variable_text=variable_text,
        warnings=list(region.warnings),
        separate=region.separate,
        timing_ms=timing_ms,
        nearest_real=list(region.nearest_real) if region.nearest_real else None,
        flagged_match=region.flagged_match[0] if region.flagged_match else None,
        gate=region.gate,
        alternate_match=region.alternate_match,
        elected_stat_vars=region.elected_stat_vars,
        entity_terms=entity_terms,
        entity_groups=_untold_place_terms(region),
        role_axis=region.no_data_role_axis,
        bound_role_dcids=region.no_data_bound_role_dcids,
        ambiguous_candidates=ambiguous_candidates,
        ambiguous_entities=_ambiguous_entity_groups(region),
        ambiguous_roles={
            entity.text: entity.role for entity in region.ambiguous_entities if entity.role
        },
        options=_options(region),
        source_terms=list(region.source_terms) or None,
        ambiguous_sources=_ambiguous_source_groups(region),
        qualifier_terms=qualifier_terms,
        unbound_sources=unbound_sources,
        scope=scope,
        unit_alternatives=list(region.unit_alternatives) or None,
        denominator=region.denominator,
        restatement=region.restatement,
        operations=region.operations,
        include_skills=include_skills,
        skills_index=skills_index,
        cause=region.cause,
        window_gap_fact=region.window_gap_fact,
        departure_facts=surviving_facts(
            raised,
            [*region.warnings, *region.unapplied_qualifier_warnings, *envelope_warnings],
        ),
        request_words=region.entity_words is not None,
        source_code_refusal=region.source_code_refusal,
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
