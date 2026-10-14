"""Assemble response parts and render confirmation sentences without LLM or graph calls."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from dfo_platform import Cause
from dfo_platform.roles import FlowFamily
from search.engine.closest_match import spec_closest_match_fact, substituted_readings_fact
from search.engine.departures import (
    NO_ENTITY_WORDS,
    merged_entity_facts,
    scope_limited_fact,
    spec_departure_facts,
    stated_facts,
    without_request_words,
)
from search.engine.qualifier_match import _normalize
from search.engine.region.codes import (
    ELECTED_SHAPE_NO_DATA,
    RETRIEVAL_SCORE_WEAK,
)
from search.engine.skills import match_spec_skills
from search.heading.departure_fact import DepartureFact, ordered_facts
from search.models import (
    _UNRESOLVED_NO_DATA_REASONS,
    UNIT_REF,
    AmbiguousEntityGroup,
    AmbiguousSourceGroup,
    CandidateSet,
    CandidatesPart,
    DefinitePart,
    Departure,
    DepartureCode,
    EntityRoleKey,
    NoData,
    NoDataDenominator,
    NoDataGate,
    NoDataPart,
    NoDataReason,
    Operation,
    Ordering,
    ParsedInput,
    ParsedPart,
    ResolvedQualifier,
    RoleAxis,
    Timing,
    UnitAlternative,
    Unsupported,
    UnsupportedPart,
    Warning,
)
from search.render.candidates import candidate_differences, candidates_sentence_plan
from search.render.departures import departures_from_facts
from search.render.heading import (
    HeadingPlan,
    declared_axis_names,
    heading_for,
    sibling_how_properties,
)
from search.render.no_data import (
    ScopeSources,
    alternate_preserves_bound_role,
    no_data_message,
    no_data_sentence_plan,
)
from search.render.plan import render_sentence_plan
from search.render.tokens import plan_words, sentence_from_plan
from skills_index import SkillsIndex

if TYPE_CHECKING:
    from search.engine.region.result import RegionResult
    from search.engine.source_code import SourceCodeRefusal
    from search.models import (  # Circular at runtime. Safe under TYPE_CHECKING.
        GraphRef,
        NearestRealSpec,
        Spec,
    )

# Part-scoped warning for omitted observations in a candidates part.
OBSERVATIONS_OMITTED = "observations_omitted"

# Index warnings belong to the envelope. Unmatchable-skill warnings belong to a part.
SKILLS_INDEX_UNAVAILABLE = "skills_index_unavailable"
SKILLS_INDEX_INCOMPLETE = "skills_index_incomplete"
SKILLS_NOT_MATCHABLE = "skills_not_matchable"


def _observations_omitted_msg() -> str:
    return (
        "Observations were not fetched for this candidates part; resubmit a chosen "
        "candidate with include_observations to retrieve them."
    )


def _skills_index_unavailable_msg() -> str:
    return (
        "Skills were requested, but the shared skills index has not completed a "
        "build yet; this response's absence of skill matches says nothing about "
        "which skills exist or what they apply to."
    )


def skills_index_unavailable_warning(
    *, include_skills: bool, skills_index: SkillsIndex | None
) -> Warning | None:
    """Warn when skills were requested before the shared index's first build.

    The request snapshot is captured once in ``core.py``. Return ``None`` when the request
    omits skills or a snapshot is ready.
    """
    if include_skills and skills_index is None:
        return Warning(
            code=SKILLS_INDEX_UNAVAILABLE,
            severity="warn",
            message=_skills_index_unavailable_msg(),
        )
    return None


def _skills_index_incomplete_msg() -> str:
    return (
        "Skills were requested and matches were returned, but the index generation "
        "those matches were read from may omit skills the platform publishes, because "
        "a read stopped at a budget or bound, the build's content budget ran out "
        "before a skill's turn, or a build step failed, so the "
        "matches are not a complete account of which skills apply."
    )


def skills_index_incomplete_warning(
    *, include_skills: bool, skills_index: SkillsIndex | None
) -> Warning | None:
    """Warn when requested skill matches come from an index that may omit published skills.

    Return ``None`` when skills are disabled, the snapshot is unavailable, or its
    ``may_omit_published_skills`` flag is false. Pair this warning with the unavailable-index
    warning at every call site.
    """
    if include_skills and skills_index is not None and skills_index.may_omit_published_skills:
        return Warning(
            code=SKILLS_INDEX_INCOMPLETE,
            severity="warn",
            message=_skills_index_incomplete_msg(),
        )
    return None


def _skills_not_matchable_msg() -> str:
    return (
        "Skills were requested, but this part resolved to no spec, so it binds "
        "no graph nodes to match a skill against."
    )


def with_sentence(
    spec: Spec,
    *,
    keep_how_properties: frozenset[str] = frozenset(),
) -> Spec:
    """Render a confirmation sentence and store its text and token plan on ``resolved``.

    The plan holds the heading and byline alone: how the answer departs from its request
    is the part's ``departures``. Sentence rendering uses the cached axis-label index and
    makes no LLM or graph call. Update the nested ``resolved`` model because
    ``model_copy(update=...)`` skips validation and the sentence fields live there.
    ``keep_how_properties`` preserves sibling differences that sentence folding could
    erase. The sentence states no slot the identity's five-tuple pins
    (`identity_pinned_properties`).
    """
    plan = render_sentence_plan(spec, keep_how_properties=keep_how_properties)
    rendered = sentence_from_plan(plan)
    return spec.model_copy(
        update={
            "resolved": spec.resolved.model_copy(
                update={"rendered_sentence": rendered, "sentence_plan": plan}
            )
        }
    )


def with_skills(spec: Spec, *, index: SkillsIndex) -> Spec:
    """Attach skill matches for the spec and each equivalent reading under ``resolved.skills``.

    Match each twin on its own bindings. Rebuild the parent readings list because
    twins are one level deep. Update ``resolved`` because ``model_copy(update=...)`` skips
    validation and the skills field belongs to that nested model.
    """
    twins = spec.resolved.equivalent_readings
    update: dict[str, object] = {"skills": match_spec_skills(spec, index=index)}
    if twins:
        update["equivalent_readings"] = [
            twin.model_copy(
                update={
                    "resolved": twin.resolved.model_copy(
                        update={"skills": match_spec_skills(twin, index=index)}
                    )
                }
            )
            for twin in twins
        ]
    return spec.model_copy(update={"resolved": spec.resolved.model_copy(update=update)})


def _surviving_qualifier_warnings(
    qualifier_warnings: Sequence[Warning],
    phrases: Sequence[str],
    regions: Sequence[RegionResult],
) -> list[Warning]:
    """Keep dropped-qualifier warnings that remain true across all regions.

    Each warning stays paired with its extracted phrase. One region admitting a phrase
    makes the request-wide warning false. Compare with the same strip-and-casefold
    normalisation used by ``admitted_qualifier_phrases``.
    """
    admitted = {phrase for region in regions for phrase in region.admitted_qualifier_phrases}
    return [
        warning
        for warning, phrase in zip(qualifier_warnings, phrases, strict=True)
        if _normalize(phrase) not in admitted
    ]


def part_departure_facts(
    spec: Spec | None,
    *,
    raised: Sequence[DepartureFact],
    operations: Sequence[Operation],
    entity_words: Mapping[str, str] | None = NO_ENTITY_WORDS,
) -> tuple[DepartureFact, ...]:
    """A part's departure facts in `DEPARTURE_ORDER`, without repeats. They are `raised`,
    the facts whose warnings the part or its envelope holds, and those read off the `spec`
    it answers with and the `operations` it carries (`departures.spec_departure_facts`,
    which asks a group in `entity_words`). A part answering with no Spec reads only its
    operations. The `entity_unavailable` facts of one role merge into one
    (`departures.merged_entity_facts`), the Spec's first, so its entities lead in the
    answer's entity order."""
    return ordered_facts(
        merged_entity_facts(
            [
                *spec_departure_facts(spec, operations=operations, entity_words=entity_words),
                *raised,
            ]
        )
    )


def part_departures(
    facts: Sequence[DepartureFact],
    *,
    family: FlowFamily | None,
    shown: Sequence[Spec],
    shows_data: bool,
    keep_how_properties: frozenset[str] = frozenset(),
    request_words: bool = True,
) -> list[Departure]:
    """The part's `departures`, one per fact it states (`departures.stated_facts`) in
    `DEPARTURE_ORDER`, each with its text (`render.departures.departures_from_facts`).
    `family` is the flow family of the part's answer, which names the roles a breakdown is
    over; `shown` are the Specs whose heading and byline the part shows, under
    `keep_how_properties` on a candidates part, and `shows_data` whether it shows data,
    false on a no-data part. A part whose door holds no words of the request (a resubmit,
    `request_words` false) asks for nothing (`departures.without_request_words`)."""
    stated = [plan_words(_heading(spec, keep_how_properties=keep_how_properties)) for spec in shown]
    facts = stated_facts(facts, stated=stated)
    if not request_words:
        facts = tuple(without_request_words(fact) for fact in facts)
    return departures_from_facts(facts, family=family, shows_data=shows_data)


def _heading(spec: Spec, *, keep_how_properties: frozenset[str] = frozenset()) -> HeadingPlan:
    """The heading and byline `spec` states (`heading_for`), as its sentence plan does."""
    return heading_for(spec, keep_how_properties=keep_how_properties)


def _with_closest_match(
    facts: Sequence[DepartureFact],
    spec: Spec,
    *,
    measure_noun: str,
    variable_text: str | None,
    warnings: Sequence[Warning],
) -> tuple[DepartureFact, ...]:
    """`facts` with the lexical `closest_match` fact of the reading `spec` shows under the
    heading's `measure_noun` (`spec_closest_match_fact`), under a weak retrieval score. A
    part states one `closest_match`, so `facts` holding one already, such as an answer on
    the elected shape's measure sibling, take none."""
    if variable_text is None or any(fact.code == "closest_match" for fact in facts):
        return ordered_facts(facts)
    closest = spec_closest_match_fact(
        spec,
        variable_text=variable_text,
        retrieval_score_weak=any(w.code == RETRIEVAL_SCORE_WEAK for w in warnings),
        measure_noun=measure_noun,
    )
    return ordered_facts([*facts, closest] if closest is not None else facts)


def _with_readings_substituted(
    facts: Sequence[DepartureFact],
    *,
    shown: Sequence[Spec],
    warnings: Sequence[Warning],
    variable_text: str | None,
) -> tuple[DepartureFact, ...]:
    """`facts` of a candidates part with the plural `closest_match` fact
    (`substituted_readings_fact`) when `warnings` hold `elected_shape_no_data`, no fact of
    `facts` states a closest match yet, and none of the readings `shown` names the measure
    asked: the readings were elected in place of a measure that holds no data. A resubmit
    (`variable_text` `None`) names no measure in words for the plural text to ask for, so
    it states none."""
    if (
        variable_text is None
        or any(fact.code == "closest_match" for fact in facts)
        or not any(warning.code == ELECTED_SHAPE_NO_DATA for warning in warnings)
    ):
        return ordered_facts(facts)
    readings = substituted_readings_fact(
        variable_text=variable_text,
        readings=[(spec, _heading(spec).heading.noun.inline_text) for spec in shown],
    )
    return ordered_facts([*facts, readings] if readings is not None else facts)


def _definite_departure_facts(
    spec: Spec,
    *,
    heading: HeadingPlan,
    variable_text: str | None,
    warnings: Sequence[Warning],
    raised: Sequence[DepartureFact],
    operations: Sequence[Operation],
    entity_words: Mapping[str, str] | None,
) -> tuple[DepartureFact, ...]:
    """A definite part's departure facts (`part_departure_facts`), with the byline's price
    basis as `shown` on a price-basis fact that lacks it, and the `closest_match` fact of its
    reading (`_with_closest_match`)."""
    facts = [
        _with_price_basis_shown(fact, unit_text=heading.byline.unit_text or "")
        for fact in part_departure_facts(
            spec, raised=raised, operations=operations, entity_words=entity_words
        )
    ]
    return _with_closest_match(
        facts,
        spec,
        measure_noun=heading.heading.noun.inline_text,
        variable_text=variable_text,
        warnings=warnings,
    )


def _with_price_basis_shown(fact: DepartureFact, *, unit_text: str) -> DepartureFact:
    """`fact` stating the byline's `unit_text` ("current US dollars") as `shown` when it
    is a price basis the answer did not apply, raised before the served unit was known.
    Any other fact, or an empty `unit_text`, stands. A price basis left without `shown` is
    not stated (`departures.stated_facts`)."""
    if fact.binding_ref != UNIT_REF or fact.shown is not None or not unit_text:
        return fact
    return dataclasses.replace(fact, shown=unit_text)


def _shared_family(specs: Sequence[Spec]) -> FlowFamily | None:
    """The flow family every one of `specs` answers in, or `None` when they differ or
    there are none."""
    families = {_heading(spec).heading.family for spec in specs}
    return families.pop() if len(families) == 1 else None


def build_definite_part(
    spec: Spec,
    *,
    variable_text: str | None,
    warnings: list[Warning],
    separate: bool = False,
    timing_ms: Timing | None = None,
    restatement: ParsedPart | None = None,
    operations: Sequence[Operation] = (),
    include_skills: bool = False,
    skills_index: SkillsIndex | None = None,
    departure_facts: Sequence[DepartureFact] = (),
    entity_words: Mapping[str, str] | None = NO_ENTITY_WORDS,
) -> DefinitePart:
    """Build a definite part around its answering spec.

    Every definite spec must contain a StatVar. The part's election warnings
    (`slot_defaulted`, `family_total_absent`) stay on the part. Skill matches attach when
    requested and indexed. ``departure_facts`` are the raised facts whose warnings the part or
    its envelope holds. ``definite_departures`` adds those the spec, the operations
    and the reading's match carry. ``entity_words`` are the request's own words for each
    entity name (`RegionResult.entity_words`), in which a group read as members is asked.
    """
    # A definite spec must carry at least one confirmed StatVar.
    assert spec.resolved.members, "definite Spec has empty stat_vars"
    spec = with_sentence(spec)
    if include_skills and skills_index is not None:
        spec = with_skills(spec, index=skills_index)
    return DefinitePart(
        status="definite",
        variable_text=variable_text,
        separate=separate,
        warnings=warnings,
        timing_ms=timing_ms,
        spec=spec,
        restatement=restatement,
        operations=list(operations),
        departures=definite_departures(
            spec,
            variable_text=variable_text,
            warnings=warnings,
            raised=departure_facts,
            operations=operations,
            entity_words=entity_words,
        ),
    )


def definite_departures(
    spec: Spec,
    *,
    variable_text: str | None,
    warnings: Sequence[Warning],
    raised: Sequence[DepartureFact],
    operations: Sequence[Operation],
    entity_words: Mapping[str, str] | None = NO_ENTITY_WORDS,
) -> list[Departure]:
    """The `departures` of a definite part answering with `spec`: the facts
    `_definite_departure_facts` reads off it, its `raised` facts and its reading's match,
    stated (`part_departures`) under its heading and byline. A streamed part's list once its
    observations are read is built here too (`enrichment.late_departures`)."""
    heading = _heading(spec)
    return part_departures(
        _definite_departure_facts(
            spec,
            heading=heading,
            variable_text=variable_text,
            warnings=warnings,
            raised=raised,
            operations=operations,
            entity_words=entity_words,
        ),
        family=heading.heading.family,
        shown=[spec],
        shows_data=True,
        request_words=entity_words is not None,
    )


# Departures that hold whatever the graph holds, the operations Search does not compute.
_SEARCH_DEPARTURE_CODES: frozenset[DepartureCode] = frozenset({"operation_not_applied"})


def build_no_data_part(
    reason: NoDataReason,
    *,
    variable_text: str | None,
    warnings: list[Warning],
    separate: bool = False,
    timing_ms: Timing | None = None,
    nearest_real: list[NearestRealSpec] | None = None,
    flagged_match: Spec | None = None,
    gate: NoDataGate | None = None,
    alternate_match: Spec | None = None,
    elected_stat_vars: list[GraphRef] | None = None,
    entity_terms: list[str] | None = None,
    role_axis: EntityRoleKey | None = None,
    bound_role_dcids: Mapping[EntityRoleKey, Sequence[str]] | None = None,
    ambiguous_candidates: list[str] | None = None,
    ambiguous_entities: list[AmbiguousEntityGroup] | None = None,
    ambiguous_roles: Mapping[str, RoleAxis] | None = None,
    data_query: ParsedInput | None = None,
    source_terms: list[str] | None = None,
    ambiguous_sources: list[AmbiguousSourceGroup] | None = None,
    qualifier_terms: list[ResolvedQualifier] | None = None,
    unbound_sources: list[ResolvedQualifier] | None = None,
    scope: ScopeSources | None = None,
    unit_alternatives: list[UnitAlternative] | None = None,
    denominator: NoDataDenominator | None = None,
    restatement: ParsedPart | None = None,
    operations: Sequence[Operation] = (),
    include_skills: bool = False,
    skills_index: SkillsIndex | None = None,
    cause: Cause | None = None,
    window_gap_fact: DepartureFact | None = None,
    departure_facts: Sequence[DepartureFact] = (),
    request_words: bool = True,
    source_code_refusal: SourceCodeRefusal | None = None,
) -> NoDataPart:
    """Build a no-data part with optional grounded near-matches.

    ``flagged_match`` is the elected variable and place without data. ``nearest_real`` holds
    confirmed specs with data after constraint relaxation. ``alternate_match`` is a
    different, data-bearing reading when ``gate`` is ``only_alternate_has_data``. The gate
    code lives on ``NoData.gate``. ``elected_stat_vars`` lets sentence composition name an
    exact series match before a nearest substitute.

    ``entity_terms`` preserves the place wording seen by the engine. ``role_axis`` names an
    open origin or destination axis in ``no_observations`` copy. ``bound_role_dcids`` records
    query-bound role values for alternate validation, including when both roles are bound.
    ``ambiguous_candidates`` feeds the wording. ``ambiguous_entities`` carries structured
    graph candidates, and ``ambiguous_roles`` names the flow axis of each ambiguous wording
    the sentence marks as a slot. ``data_query`` is the part's parsed form for
    ``entity_ambiguous``, ``source_ambiguous`` and ``source_code_ambiguous``.
    ``source_terms`` are the source wordings a ``source_not_resolved`` or
    ``source_ambiguous`` outcome rests on, or the series code a ``source_code_*`` outcome
    rests on, and ``ambiguous_sources`` pairs each ambiguous one with its candidates. For
    ``source_ambiguous`` and ``source_code_ambiguous`` ``ambiguous_candidates`` holds the
    candidates' labels. ``source_code_refusal`` is the refusal a ``source_code_*`` outcome
    rests on, whose sentence it words.
    ``qualifier_terms`` are the category terms of a ``qualifier_not_bound`` outcome, and
    ``unbound_sources`` the sources it could not bind. ``NoData.unbound_qualifiers`` holds
    both, terms first. ``scope`` names the sources an outcome the request reached was read
    under (`no_data.scope_sentence`), and an unresolved outcome ignores it.
    ``unit_alternatives`` carries a ``unit_not_resolved`` refusal's capped currencies.
    ``None`` omits it. ``denominator`` carries a ``denominator_not_available`` refusal's
    denominator asked and found, which its sentence names.
    ``NoData.message`` is the bare no-data sentence with every term the part names.

    Sentence plans reuse each near-match's tokens under a gate-specific lead-in. A skills
    request adds ``skills_not_matchable`` for the answer and matches available near-matches
    when an index is available. ``cause`` is populated for ``rate_coverage_gap``.
    ``window_gap_fact`` is the `year_unavailable` fact of
    years asked that hold no point of the elected series (`departures.window_gap_fact`).
    The part carries it unless its sentence shows an alternate.
    ``departure_facts`` are the raised facts whose warnings the part or its envelope holds,
    stated without the request's words when ``request_words`` is false (a resubmit).
    An alternate the sentence shows (`alternate_preserves_bound_role`) adds its reading's
    `closest_match` fact (`_with_closest_match`). The part shows no data, so no departure
    text claims it shows anything.
    A check limited to the values the engine chose for the flagged match adds a
    `scope_limited` departure (`departures.scope_limited_fact`).
    A part whose ``reason`` is unresolved (`NoData.category`) ran no check, so it carries
    only `operation_not_applied`, the one departure about Search itself.
    """
    shown_alternate = (
        alternate_match
        if gate == "only_alternate_has_data"
        and alternate_match is not None
        and alternate_preserves_bound_role(alternate_match, bound_role_dcids)
        else None
    )
    # The years asked hold no point of the elected series. An alternate the sentence shows
    # holds them, so the fact would misstate what the part shows.
    year_facts = [window_gap_fact] if window_gap_fact and shown_alternate is None else []
    scope_fact = scope_limited_fact(flagged_match) if flagged_match is not None else None
    facts = part_departure_facts(
        None,
        raised=[*departure_facts, *year_facts, *([scope_fact] if scope_fact else [])],
        operations=operations,
    )
    if reason in _UNRESOLVED_NO_DATA_REASONS:
        # The request was never read into the graph, so no check ran to leave anything
        # out. Only what Search itself does not do departs, and no source limited it.
        facts = tuple(fact for fact in facts if fact.code in _SEARCH_DEPARTURE_CODES)
        scope = None
    if shown_alternate is not None:
        facts = _with_closest_match(
            facts,
            shown_alternate,
            measure_noun=_heading(shown_alternate).heading.noun.inline_text,
            variable_text=variable_text,
            warnings=warnings,
        )
    departures = part_departures(
        facts,
        family=_heading(flagged_match).heading.family if flagged_match else None,
        shown=[flagged_match] if flagged_match else [],
        shows_data=False,
        request_words=request_words,
    )
    # ``model_copy`` preserves each near-match's concrete subclass. These casts reflect it.
    if nearest_real:
        # Preserve bound/unbound differences among sibling near-matches.
        keep_how_properties = sibling_how_properties(nearest_real)
        nearest_real = cast(
            "list[NearestRealSpec]",
            [with_sentence(s, keep_how_properties=keep_how_properties) for s in nearest_real],
        )
    flagged_match = with_sentence(flagged_match) if flagged_match else None
    alternate_match = with_sentence(alternate_match) if alternate_match else None
    # Reuse near-match sentence tokens after the no-data phrase and gate-specific lead-in.
    plan = no_data_sentence_plan(
        reason,
        gate=gate,
        flagged_match=flagged_match,
        alternate_match=alternate_match,
        nearest_real=nearest_real,
        elected_stat_vars=elected_stat_vars,
        variable_term=variable_text,
        entity_terms=entity_terms,
        role_axis=role_axis,
        bound_role_dcids=bound_role_dcids,
        ambiguous_candidates=ambiguous_candidates,
        ambiguous_roles=ambiguous_roles,
        qualifier_terms=qualifier_terms,
        source_terms=source_terms,
        scope=scope,
        denominator=denominator,
        departures=departures,
        source_code_refusal=source_code_refusal,
    )
    rendered = sentence_from_plan(plan)
    if include_skills and skills_index is not None:
        # Skill matching is independent of sentence rendering.
        if nearest_real:
            nearest_real = cast(
                "list[NearestRealSpec]",
                [with_skills(s, index=skills_index) for s in nearest_real],
            )
        flagged_match = with_skills(flagged_match, index=skills_index) if flagged_match else None
        alternate_match = (
            with_skills(alternate_match, index=skills_index) if alternate_match else None
        )
    part_warnings = list(warnings)
    if include_skills:
        part_warnings.append(
            Warning(code=SKILLS_NOT_MATCHABLE, severity="warn", message=_skills_not_matchable_msg())
        )

    return NoDataPart(
        status="no_data",
        variable_text=variable_text,
        separate=separate,
        warnings=part_warnings,
        timing_ms=timing_ms,
        no_data=NoData(  # type: ignore[arg-type]
            reason=reason,
            message=no_data_message(
                reason,
                variable_term=variable_text,
                entity_terms=entity_terms,
                role_axis=role_axis,
                ambiguous_candidates=ambiguous_candidates,
                elected_stat_vars=elected_stat_vars,
                qualifier_terms=qualifier_terms,
                source_terms=source_terms,
                scope=scope,
                denominator=denominator,
                source_code_refusal=source_code_refusal,
            ),
            nearest_real=nearest_real,
            flagged_match=flagged_match,
            gate=gate,
            alternate_match=alternate_match,
            elected_count=len(elected_stat_vars) if elected_stat_vars is not None else None,
            entity_terms=entity_terms or [],
            ambiguous_entities=ambiguous_entities or [],
            source_terms=source_terms or [],
            ambiguous_sources=ambiguous_sources or [],
            cause=cause,
            unbound_qualifiers=[*(qualifier_terms or ()), *(unbound_sources or ())],
            unit_alternatives=unit_alternatives,
            denominator=denominator,
            data_query=data_query,
        ),
        rendered_sentence=rendered,
        sentence_plan=plan,
        restatement=restatement,
        operations=list(operations),
        departures=departures,
    )


def build_unsupported_part(
    code: str,
    pointer: str,
    detail: str,
    definition: dict[str, object],
    *,
    variable_text: str | None,
    warnings: list[Warning],
    separate: bool = False,
    timing_ms: Timing | None = None,
    restatement: ParsedPart | None = None,
    operations: Sequence[Operation] = (),
    include_skills: bool = False,
    skills_index: SkillsIndex | None = None,
    cause: Cause | None = None,
) -> UnsupportedPart:
    """Wrap an unsupported payload in a part.

    A requested skills match adds ``skills_not_matchable`` because this part has no spec.
    The supplied skills index is unused. ``cause`` is populated for ``recipe_not_declared``.
    """
    part_warnings = list(warnings)
    if include_skills:
        part_warnings.append(
            Warning(code=SKILLS_NOT_MATCHABLE, severity="warn", message=_skills_not_matchable_msg())
        )

    return UnsupportedPart(
        status="unsupported",
        variable_text=variable_text,
        separate=separate,
        warnings=part_warnings,
        timing_ms=timing_ms,
        unsupported=Unsupported(
            code=code,
            pointer=pointer,
            detail=detail,
            definition=definition,
            cause=cause,
        ),
        restatement=restatement,
        operations=list(operations),
    )


def build_candidates_part(
    specs: list[Spec],
    *,
    variable_text: str | None,
    warnings: list[Warning],
    separate: bool = False,
    timing_ms: Timing | None = None,
    max_candidates: int | None = None,
    include_observations: bool = False,
    budget_truncated: bool = False,
    ordering: Ordering = "intent_rank",
    restatement: ParsedPart | None = None,
    operations: Sequence[Operation] = (),
    include_skills: bool = False,
    skills_index: SkillsIndex | None = None,
    departure_facts: Sequence[DepartureFact] = (),
    request_words: bool = True,
) -> CandidatesPart:
    """Build a candidates part from competing specs, in the order given: the elected
    reading first and its rivals in menu order (`region.status_rule`), or the order a
    resubmit posted. ``ordering`` is the wire label, `intent_rank` unless a resubmit posted
    another.

    The cap defaults to the input length. ``budget_truncated`` records upstream omissions
    and combines with the local cap for ``CandidateSet.truncated``. The summary omits a
    possible-total count when upstream omissions make it unknown.

    Sentence rendering preserves differing how-axis values, states shared departures once, and marks
    the remaining distinctions. ``include_observations`` adds a warning because no candidate
    is selected for observation retrieval. Requested skills attach to each clamped candidate
    when the index snapshot is available. ``departure_facts`` are the raised facts that
    hold for every reading the part shows; no single spec answers, so none is read off one.
    Their text names roles in the flow family every clamped reading shares. A part warned
    `elected_shape_no_data` states one `closest_match` departure
    (`_with_readings_substituted`).
    ``request_words`` is false on a door that holds no words of the request (a resubmit).
    """
    cap = max_candidates if max_candidates is not None else len(specs)
    clamped = list(specs)[:cap]
    # Report either upstream budget omissions or this function's own cap.
    truncated = budget_truncated or len(specs) > cap

    keep_how_properties = sibling_how_properties(clamped)
    clamped = [with_sentence(spec, keep_how_properties=keep_how_properties) for spec in clamped]
    # Marks the tokens that set each reading apart (`SentenceToken.distinguishes`).
    candidate_differences(clamped, axis_declarations=declared_axis_names())

    if include_skills and skills_index is not None:
        # Skill matching is independent of sentence rendering.
        clamped = [with_skills(spec, index=skills_index) for spec in clamped]

    departures = part_departures(
        _with_readings_substituted(
            part_departure_facts(None, raised=departure_facts, operations=operations),
            shown=clamped,
            warnings=warnings,
            variable_text=variable_text,
        ),
        family=_shared_family(clamped),
        shown=clamped,
        shows_data=True,
        keep_how_properties=keep_how_properties,
        request_words=request_words,
    )
    plan = candidates_sentence_plan(
        clamped,
        departures=departures,
        truncated=truncated,
        # Show the pre-clamp total only when no upstream budget hid candidates.
        total=len(specs) if not budget_truncated else None,
    )
    rendered = sentence_from_plan(plan)

    part_warnings = list(warnings)
    if include_observations:
        part_warnings.append(
            Warning(code=OBSERVATIONS_OMITTED, severity="warn", message=_observations_omitted_msg())
        )
    return CandidatesPart(
        status="candidates",
        variable_text=variable_text,
        separate=separate,
        warnings=part_warnings,
        timing_ms=timing_ms,
        candidates=CandidateSet(
            ordering=ordering,
            max_candidates=cap,
            specs=clamped,
            truncated=truncated,
        ),
        rendered_sentence=rendered,
        sentence_plan=plan,
        restatement=restatement,
        operations=list(operations),
        departures=departures,
    )
